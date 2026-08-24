"""The SAE **latent-intervention condition**: a declared study condition whose
mechanism is encode → edit the latent → decode the induced delta, never
decoder-direction addition (SAE-VECTOR-INTERVENTION proposal r2 §8 P2-9).

Why a separate manifest key and not a flag on ``conditions``
-----------------------------------------------------------

A latent condition and a vector condition are different mechanisms with
different dose units, different state dependence, and different provenance. If
one lived inside ``conditions[]`` behind a type flag, an engine that did not
recognise the flag would see a condition with **no slots** — which is precisely
how a baseline is declared — and would run it, name it, and report it as a
steered arm that in fact steered nothing. That is the "declared studyType inerts
conditions" failure with the labels still attached, and it is unrecoverable
after the fact because the records look normal.

So latent conditions live in their own top-level list, ``saeLatentConditions``,
parallel to ``variantConditions``; and a ``conditions[]`` entry that carries
``interventionType: "saeLatent"`` is a **verify() violation** that names the
right key, closing the misdeclaration path rather than documenting it.

Server-only, and marked as such in the pinned bytes
---------------------------------------------------

The mechanism is SERVER-ONLY by the same rule as the J-lens instruments: the SAE
artifacts are PyTorch/HF-native and activations do not transfer across
substrates (CLAUDE.md, J-Space hard requirement). ``jlensReadout`` carries that
rule by documentation alone, which is adequate there — an unrecognised READOUT
block costs a study its readout, i.e. missing data. Here the cost of silence is
different in kind: an unrecognised INTERVENTION block costs a study its
intervention while every condition keeps its name, so the arm reports as
"clamp-β=10" and contains baseline text. Each entry therefore carries an
explicit ``"serverOnly": true`` marker, required and validated: it puts the
constraint in the hashed bytes, it lets any reader (or a future Swift-side
gate) refuse the block without understanding it, and it makes an unmarked
latent condition impossible to author rather than merely discouraged.

Everything here validates OFFLINE: no SAE weights, no HuggingFace, no network —
and, since the G7 split, no torch either. The mode vocabulary and the two data
classes come from :mod:`steerlab_server.steering.sae_latent_schema`, the
torch-free half of the steering module, so importing this file for validation
costs nothing beyond the standard library. :func:`materialize` is the one
function that loads tensors, and it does so through the injectable seam in
:mod:`steerlab_server.experiment.gemma_scope`, imported inside the function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# The SCHEMA half of the steering module, deliberately: this file validates a
# manifest and must not drag the injector stack (and therefore torch) into
# `Manifest.verify`. `sae_latent_schema` imports nothing heavy; the names are
# the same objects `steering.sae_latent` re-exports, so `materialize` below
# builds exactly what the intervention consumes.
from ..steering.sae_latent_schema import (CLAMP, MODES, SAELatentEdit,
                                          SAELatentFeature)

#: Top-level manifest key holding the latent conditions.
CONDITION_KEY = "saeLatentConditions"
#: The value ``interventionType`` must carry inside that list.
INTERVENTION_TYPE = "saeLatent"
#: Values of ``interventionType`` that are legal on an ORDINARY condition.
VECTOR_INTERVENTION_TYPES = ("vector",)

#: Closed key set — an unknown key is refused rather than ignored, because a
#: typo'd ``beta`` silently becomes "no dose declared" and a typo'd ``mode``
#: silently becomes a different experiment.
_ALLOWED_KEYS = frozenset({
    "name", "interventionType", "serverOnly", "release", "saeID", "feature",
    "mode", "beta", "layer", "constructLabel", "description",
})
_REQUIRED_KEYS = ("name", "interventionType", "serverOnly", "release", "saeID",
                  "feature", "mode", "beta")


@dataclass
class SAELatentConditionSpec:
    """One declared latent condition, validated but not yet materialized.

    Holds identity and dose only. The four tensors arrive at RUN time through
    :func:`materialize`, exactly as vector conditions re-derive their vectors
    from pinned recipes rather than carrying bytes in the manifest.
    """

    name: str
    release: str
    sae_id: str
    feature: int
    #: ``"add"`` or ``"clamp"`` — semantics in
    #: :mod:`steerlab_server.steering.sae_latent`.
    mode: str
    #: The edit magnitude, in LATENT units (see :data:`BETA_UNITS`).
    beta: float
    #: The SAE's layer. Optional in the manifest; when present it must agree
    #: with the SAE's own layer, which is checked at materialization.
    layer: int | None = None
    construct_label: str = ""
    description: str = ""
    raw: dict = field(default_factory=dict)


#: What ``beta`` is denominated in. Stamped into provenance so no reader can
#: mistake a latent dose for an ``alpha`` in residual-norm units — the two are
#: not convertible, and a latent β is not comparable across features either.
BETA_UNITS = "latent"


def _finite_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def entry_violations(entry: object, index: int) -> list[str]:
    """Validation for ONE ``saeLatentConditions`` entry. Offline and total."""
    where = f"{CONDITION_KEY}[{index}]"
    if not isinstance(entry, dict):
        return [f"{where} must be an object"]
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        where = f"{CONDITION_KEY} '{name.strip()}'"
    violations: list[str] = []

    unknown = sorted(set(entry) - _ALLOWED_KEYS)
    if unknown:
        violations.append(
            f"{where} has unknown key(s) {', '.join(unknown)} — the latent "
            f"condition shape is {', '.join(sorted(_ALLOWED_KEYS))}")
    missing = [key for key in _REQUIRED_KEYS if key not in entry]
    if missing:
        violations.append(
            f"{where} is missing required key(s) {', '.join(missing)} — a "
            "half-declared intervention is not a condition")

    if not isinstance(name, str) or not name.strip():
        violations.append(f"{where} needs a non-empty name")

    declared = entry.get("interventionType")
    if "interventionType" in entry and declared != INTERVENTION_TYPE:
        violations.append(
            f"{where} declares interventionType {declared!r} — entries of "
            f"{CONDITION_KEY} are {INTERVENTION_TYPE!r} by definition")

    if "serverOnly" in entry and entry.get("serverOnly") is not True:
        violations.append(
            f"{where} must declare \"serverOnly\": true — SAE latent "
            "interventions are PyTorch/HF-native and cannot run on the "
            "Swift/MLX engine, and an unmarked latent condition would run "
            "there as an unsteered arm still carrying its condition name")

    for key, label in (("release", "release"), ("saeID", "saeID")):
        value = entry.get(key)
        if key in entry and (not isinstance(value, str) or not value.strip()):
            violations.append(f"{where}: {label} must be a non-empty string")

    feature = entry.get("feature")
    if "feature" in entry and (isinstance(feature, bool)
                               or not isinstance(feature, int) or feature < 0):
        violations.append(
            f"{where}: feature must be a non-negative integer id in the SAE's "
            "own dictionary")

    mode = entry.get("mode")
    if "mode" in entry and mode not in MODES:
        violations.append(
            f"{where}: unknown mode {mode!r} — expected one of "
            f"{', '.join(MODES)} ('add' edits the pre-activation and "
            "re-evaluates the JumpReLU gate; 'clamp' sets the latent outright, "
            "which can activate a dormant feature)")

    beta = _finite_number(entry.get("beta")) if "beta" in entry else None
    if "beta" in entry and beta is None:
        violations.append(
            f"{where}: beta must be a finite number, in LATENT units — the "
            "feature's own activation scale, never an alpha in residual-norm "
            "units")
    elif beta is not None and mode == CLAMP and beta < 0:
        violations.append(
            f"{where}: clamp beta must be >= 0 (got {beta}) — a JumpReLU "
            "latent is non-negative by construction, so a negative clamp "
            "target is outside the dictionary's range")

    layer = entry.get("layer")
    if "layer" in entry and layer is not None and (
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0):
        violations.append(
            f"{where}: layer must be a non-negative integer (or absent — an "
            "SAE's dictionary lives at exactly one layer, which the SAE itself "
            "declares)")

    label = entry.get("constructLabel")
    if label is not None and not isinstance(label, str):
        violations.append(f"{where}: constructLabel must be a string")
    elif isinstance(label, str) and ":" in label:
        violations.append(
            f"{where}: constructLabel {label!r} may not contain ':' — it is "
            "composed into a colon-delimited identity string")
    return violations


def condition_violations(raw: dict) -> list[str]:
    """``verify()`` for the whole latent-condition surface of a manifest.

    Covers three things a study can get wrong, all of which are silent
    otherwise:

    1. a malformed / half-declared / unmarked entry in ``saeLatentConditions``;
    2. a latent condition MISDECLARED inside ``conditions[]`` (where it would
       execute as a slotless baseline under a steered arm's name);
    3. a NAME collision with an ordinary or variant condition — two conditions
       sharing a name make every per-condition table ambiguous.

    ABSENT key = no latent conditions = no violations; legacy manifests are
    untouched and their content hashes unchanged.
    """
    violations: list[str] = []
    conditions = raw.get("conditions")
    if isinstance(conditions, list):
        for index, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                continue
            declared = condition.get("interventionType")
            if declared is None:
                continue
            named = condition.get("name") or f"conditions[{index}]"
            if declared == INTERVENTION_TYPE:
                violations.append(
                    f"condition '{named}' declares interventionType "
                    f"{INTERVENTION_TYPE!r} inside 'conditions' — latent "
                    f"interventions are a DIFFERENT mechanism from vector "
                    f"addition and are declared in '{CONDITION_KEY}'. Left "
                    f"here it would run as a slotless (baseline) arm under a "
                    f"steered condition's name")
            elif declared not in VECTOR_INTERVENTION_TYPES:
                violations.append(
                    f"condition '{named}' declares unknown interventionType "
                    f"{declared!r} — ordinary conditions are "
                    f"{', '.join(VECTOR_INTERVENTION_TYPES)} (or omit the key)")

    entries = raw.get(CONDITION_KEY)
    if entries is None:
        return violations
    if not isinstance(entries, list):
        return violations + [
            f"{CONDITION_KEY} must be a list of latent-condition objects"]
    for index, entry in enumerate(entries):
        violations += entry_violations(entry, index)

    # Name uniqueness across EVERY condition surface: readers key per-condition
    # tables by name, so a collision silently merges two mechanisms into one row.
    def _names(block, key="name") -> list[str]:
        items = raw.get(block)
        if not isinstance(items, list):
            return []
        return [str(item.get(key)).strip() for item in items
                if isinstance(item, dict) and item.get(key)]

    seen: dict[str, str] = {}
    for source in ("conditions", "variantConditions", CONDITION_KEY):
        for name in _names(source):
            if name in seen:
                violations.append(
                    f"condition name '{name}' is declared in both "
                    f"'{seen[name]}' and '{source}' — condition names key "
                    f"every per-condition table and must be unique")
            else:
                seen[name] = source
    return violations


def parse(raw: dict) -> list[SAELatentConditionSpec]:
    """The manifest's validated latent conditions.

    Raises ``ValueError`` listing every violation rather than returning a
    partially-trusted list: a run must not start on a manifest whose declared
    mechanism the engine had to guess at.
    """
    violations = condition_violations(raw)
    if violations:
        raise ValueError("; ".join(violations))
    specs: list[SAELatentConditionSpec] = []
    for entry in (raw.get(CONDITION_KEY) or []):
        specs.append(SAELatentConditionSpec(
            name=str(entry["name"]).strip(),
            release=str(entry["release"]).strip(),
            sae_id=str(entry["saeID"]).strip(),
            feature=int(entry["feature"]),
            mode=str(entry["mode"]),
            beta=float(entry["beta"]),
            layer=(int(entry["layer"]) if entry.get("layer") is not None else None),
            construct_label=str(entry.get("constructLabel") or ""),
            description=str(entry.get("description") or ""),
            raw=dict(entry)))
    return specs


def materialize(spec: SAELatentConditionSpec, *, loader=None
                ) -> tuple[SAELatentEdit, dict]:
    """Load the feature's four tensors and return ``(edit, provenance)``.

    The ONE function in this module that reaches the network, and it does so
    through the injectable seam
    (:data:`~steerlab_server.experiment.gemma_scope.SAELatentFeatureLoader`),
    so every test above runs offline.

    The returned provenance block is the run-config / per-record stamp: the
    exact repository COMMIT the tensors were read from (a floating ref is not
    provenance), the SAE config hash, the encoder and decoder row hashes, the
    activation family and whether ``b_dec`` was folded into the bias, and the
    declared mode/β with ``betaUnits: "latent"`` + ``latentUnits: true`` so β
    can never be read as an α.

    Deliberately NOT done here: any calibration pass over a corpus to convert β
    into a percentile of the feature's observed activation. Dose calibration for
    latent interventions is future work (proposal r2 §8 P2-9 follow-up); loading
    an intervention must not silently run a model.
    """
    from . import gemma_scope

    load = loader or gemma_scope.load_sae_latent_feature
    loaded = load(spec.release, spec.sae_id, spec.feature)
    if not getattr(loaded, "repo_id", "") or not getattr(loaded, "repo_revision", ""):
        raise ValueError(
            f"latent condition '{spec.name}': the SAE loader returned no "
            "repository id/revision — the condition pins the exact published "
            "bytes it read, so unpinned sources are refused")

    parsed = gemma_scope.parse_sae_id(spec.sae_id)
    config_layer = gemma_scope._config_layer(loaded.config or {})
    derived_layer = config_layer if config_layer is not None else parsed["layer"]
    if spec.layer is not None and derived_layer is not None \
            and int(spec.layer) != int(derived_layer):
        raise ValueError(
            f"latent condition '{spec.name}': declared layer {spec.layer} "
            f"disagrees with the SAE's own layer {derived_layer} "
            f"({spec.sae_id}) — an SAE's dictionary lives at exactly one "
            "layer, so this is a mis-specified condition, not a choice")
    layer = spec.layer if spec.layer is not None else derived_layer
    if layer is None:
        raise ValueError(
            f"latent condition '{spec.name}': cannot determine the layer for "
            f"SAE {spec.sae_id!r} — its config names no hook layer and the id "
            "does not follow the 'layer_<n>_…' grammar; declare the layer")

    feature = SAELatentFeature(
        encoder_row=tuple(float(x) for x in loaded.encoder_row),
        decoder_row=tuple(float(x) for x in loaded.decoder_row),
        encoder_bias=float(loaded.encoder_bias),
        threshold=float(loaded.threshold))
    edit = SAELatentEdit(
        layer=int(layer), feature=feature, mode=spec.mode, beta=spec.beta,
        feature_id=spec.feature, label=spec.construct_label)

    provenance = {
        "interventionType": INTERVENTION_TYPE,
        "condition": spec.name,
        "release": spec.release,
        "saeID": spec.sae_id,
        "feature": spec.feature,
        "layer": int(layer),
        "mode": spec.mode,
        "beta": float(spec.beta),
        # β is in the feature's own activation scale. Both keys, deliberately:
        # the string names the unit, the boolean is the thing a reader (or a
        # results viewer) can branch on without parsing prose.
        "betaUnits": BETA_UNITS,
        "latentUnits": True,
        "repository": loaded.repo_id,
        "repositoryRevision": loaded.repo_revision,
        "saeConfigHash": gemma_scope.sae_config_hash(loaded.config or {}),
        "encoderRowHash": gemma_scope.decoder_row_hash(list(loaded.encoder_row)),
        "decoderRowHash": gemma_scope.decoder_row_hash(list(loaded.decoder_row)),
        "encoderBias": float(loaded.encoder_bias),
        "threshold": float(loaded.threshold),
        "activation": getattr(loaded, "activation", "jumprelu"),
        "bDecFoldedIntoBias": bool(getattr(loaded, "b_dec_folded", False)),
        # Named for what it is: no corpus was read, so no dose calibration
        # exists. Absent would read as "not recorded"; false reads as "not done".
        "doseCalibrated": False,
        "substrate": _substrate(),
    }
    if spec.construct_label:
        provenance["constructLabel"] = spec.construct_label
    if getattr(loaded, "sparsity", None) is not None:
        provenance["featureSparsity"] = float(loaded.sparsity)
    if parsed.get("width"):
        provenance["width"] = parsed["width"]
    if parsed.get("l0Target"):
        provenance["l0Target"] = parsed["l0Target"]
    return edit, provenance


def _substrate() -> str:
    from ..steering import vector_store
    return vector_store.SUBSTRATE


def intervention_state(spec: SAELatentConditionSpec, provenance: dict) -> dict:
    """Per-record provenance for a latent condition — the twin of
    ``tasks._intervention_state``.

    Carries the SAME cross-engine key skeleton every other condition emits
    (``slots`` / ``bandWidth`` / ``alphaInNormUnits`` / ``controlType``) so one
    reader parses every record shape, with the vector-only fields explicitly
    NULLED rather than omitted: a latent condition has no slots, no band, and no
    α, and a reader that finds ``alphaInNormUnits: null`` next to
    ``latentUnits: true`` cannot misread the dose. The mechanism block lives
    under ``saeLatent``.
    """
    return {
        "slots": [],
        "bandWidth": None,
        "alphaInNormUnits": None,
        "controlType": None,
        "interventionType": INTERVENTION_TYPE,
        "saeLatent": dict(provenance),
    }

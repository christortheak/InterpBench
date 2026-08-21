"""Model variants — base model + adapters + injections + neutral basis + prompt
settings (parallel to Swift ``ModelVariantArtifact`` + ``ExperimentTasks``
``injections(for:)`` / ``loadAdapter``).

A variant is a frozen, reusable steering configuration: which saved vectors to
inject (at which layer/alpha, with optional band + neutral-PC projection +
norm-unit alpha), which LoRA adapter to load, and the prompt mode / system
prompt / temperature / thinking flag to generate under. Studies reference
variants as ``variantConditions``; this module loads one, builds its injection
cells, and (best-effort) applies its PEFT adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from . import neutral, paths
from .generate import CellInjection
from ..steering import vector_math as vm
from ..steering import vector_store


@dataclass
class ModelVariant:
    name: str
    base_model_id: str
    base_revision: str | None = None
    adapters: list[dict] = field(default_factory=list)       # {adapterDirectory, adapterHash}
    injections: list[dict] = field(default_factory=list)     # {concept, vectorArtifactID, layer, alpha}
    band_width: int = 1
    # Programmatic-construction default only (norm units are the default for
    # NEW work, matching Swift's ModelVariantArtifact init). Decoding a dict
    # goes through from_dict, which reads an ABSENT key as raw — see there.
    alpha_in_norm_units: bool = True
    neutral_pc_basis_path: str | None = None
    prompt_mode: str = "chatAssistant"
    qwen_thinking_enabled: bool = False
    temperature: float = 0.0
    system_prompt: str | None = None
    created_at: str | None = None
    # Sweep-promotion birth certificate (cross-engine contract): present only
    # on agents minted by `promote` from a sweep-selected cell — experiment,
    # sweep run, resolved criterion, dev-split hash, winning cell, metrics,
    # promotedBy "criterion"|"manualOverride", substrate, appVersion. Absent =
    # hand-created variant. Preserved verbatim across round-trips.
    promotion: dict | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVariant":
        return cls(
            name=d.get("name", "variant"), base_model_id=d["baseModelID"],
            base_revision=d.get("baseRevision"), adapters=d.get("adapters", []),
            injections=d.get("injections", []), band_width=int(d.get("bandWidth", 1)),
            # Cross-engine decode rule (mirrors Swift ModelVariantArtifact's
            # decoder): an ABSENT alphaInNormUnits reads as RAW alpha — the
            # conservative literal α·v meaning for legacy artifacts that
            # predate norm capture. Same bytes must inject the same on both
            # engines. Both encoders always WRITE the key explicitly
            # (to_dict below), which is what makes this default safe.
            alpha_in_norm_units=bool(d.get("alphaInNormUnits", False)),
            neutral_pc_basis_path=d.get("neutralPCBasisPath"),
            prompt_mode=d.get("promptMode", "chatAssistant"),
            qwen_thinking_enabled=bool(d.get("qwenThinkingEnabled", False)),
            temperature=float(d.get("temperature", 0.0)),
            system_prompt=d.get("systemPrompt"),
            # Preserve the artifact's provenance across a server round-trip so an
            # uploaded variant keeps its original creation timestamp.
            created_at=d.get("createdAt"),
            promotion=d.get("promotion") if isinstance(d.get("promotion"), dict) else None)

    @classmethod
    def from_file(cls, path: str) -> "ModelVariant":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict:
        return {
            "schemaVersion": 1, "name": self.name, "baseModelID": self.base_model_id,
            "baseRevision": self.base_revision, "adapters": self.adapters,
            "injections": self.injections, "bandWidth": self.band_width,
            "alphaInNormUnits": self.alpha_in_norm_units,
            "neutralPCBasisPath": self.neutral_pc_basis_path,
            "promptMode": self.prompt_mode, "qwenThinkingEnabled": self.qwen_thinking_enabled,
            "temperature": self.temperature, "systemPrompt": self.system_prompt,
            **({"createdAt": self.created_at} if self.created_at else {}),
            **({"promotion": self.promotion} if self.promotion else {}),
        }


def variant_injections(variant: ModelVariant, *,
                       root: str | None = None) -> list[CellInjection]:
    """Build the variant's injection cells (parallel to Swift
    ``ExperimentTasks.injections(for:)``): per injection, load the vector, apply
    the layer band, project out the neutral basis if set, and convert norm-unit
    alphas folding out the vector norm.

    ``root`` scopes artifact resolution to an EXPLICIT workspace. Absent (every
    historical caller) it resolves against the process default, exactly as
    before; a caller that was handed a root must pass it, or it silently reads
    a different workspace's vectors (external review round 2)."""
    # `resolve_artifact`, not bare `resolve`: a reference recorded on another
    # machine (an app-promoted agent's absolute Mac path) rebases onto this
    # workspace when the artifact is actually here — see paths.resolve_artifact.
    basis = (neutral.load_basis(
                paths.resolve_artifact(variant.neutral_pc_basis_path, root))
             if variant.neutral_pc_basis_path else None)
    half = max(1, variant.band_width) // 2
    cells: list[CellInjection] = []
    for inj in variant.injections:
        vector_id = paths.resolve_artifact(inj["vectorArtifactID"], root)
        directory, name = os.path.dirname(vector_id), os.path.basename(vector_id)
        vectors, sidecar = vector_store.load(directory, name)
        # A stored vector is only meaningful in the activation basis of the
        # engine that extracted it: refuse foreign-substrate artifacts (e.g.
        # swift-mlx vectors on a shared tree); unstamped legacy passes.
        vector_store.require_native_substrate(sidecar, vector_id)
        # An ABLATING injection covers the whole network and takes no band:
        # a removal at one layer is usually rewritten by the layers above it.
        # Before this was read (2026-07-27) an ablation composed in the app
        # arrived here as an ordinary steering injection with alpha = λ —
        # the wire spells "add" by omission, so a dropped mode was
        # indistinguishable from steering, and the cluster steered while the
        # UI said ablate.
        is_ablation = str(inj.get("mode") or "add") == "ablate"
        declared = inj.get("layers")
        center = min(max(0, int(inj["layer"])), vectors.layer_count - 1)
        if is_ablation:
            layer_range = ([l for l in declared if 0 <= l < vectors.layer_count]
                           if declared else range(vectors.layer_count))
        else:
            layer_range = range(max(0, center - half),
                                min(vectors.layer_count - 1, center + half) + 1)
        # Ablation-direction centering — declared per injection, never silent.
        # "neutralMean" projects the artifact's stored neutral residual mean
        # out of the direction (v − (v·m̂)m̂): extracted vectors routinely
        # share a large component with that mean, and ablating it at λ=1
        # collapses generation into single-token repetition (2026-08-06
        # collapse study; identical semantics in Swift
        # ``ChatService.currentInjections``). Absent/"none" keeps the raw
        # direction and runs the mean-alignment preflight instead.
        centering = str(inj.get("centering") or "none")
        neutral_mean = None
        if is_ablation:
            if centering not in ("none", "neutralMean"):
                raise ValueError(
                    f"variant '{variant.name}' injection '{inj.get('concept')}' "
                    f"declares unknown centering '{centering}' — this engine "
                    f"implements 'none' and 'neutralMean'")
            neutral_mean = vector_store.load_neutral_mean(directory, name)
            if centering == "neutralMean" and neutral_mean is None:
                raise ValueError(
                    f"variant '{variant.name}' declares neutral-mean centering "
                    f"for '{inj.get('concept')}' but the vector artifact "
                    f"carries no stored neutral mean — re-extract the concept "
                    f"with a neutral corpus (artifacts stamp neutralMeanSource "
                    f"since 2026-08-06)")
            if centering == "none":
                _preflight_mean_alignment(
                    variant=variant, inj=inj, vectors=vectors,
                    layer_range=layer_range, neutral_mean=neutral_mean)
        elif centering != "none":
            raise ValueError(
                f"variant '{variant.name}' declares centering on a STEERING "
                f"injection ('{inj.get('concept')}') — centering is an "
                f"ablation-direction transform; remove it or set mode ablate")
        for layer in layer_range:
            vector = vectors.per_layer[layer]
            if is_ablation and centering == "neutralMean":
                vector = vm.mean_centered(vector, neutral_mean[layer])
            if basis and layer in basis:
                vector = vm.projecting_out(vector, basis[layer])
            vnorm = vm.l2_norm(vector)
            if vnorm <= 0:
                continue
            if is_ablation:
                # λ is never scaled by the residual norm: that denominator
                # exists to make α comparable, and ablation removes exactly
                # what is present, so it already scales itself. It also means
                # an ablating variant never hits the missing-norms refusal
                # below for a conversion it does not perform.
                alpha = float(inj["alpha"])
            elif variant.alpha_in_norm_units:
                norms = sidecar.residualNormPerLayer
                if not norms:
                    raise ValueError(
                        f"variant '{variant.name}' uses residual-norm alpha but vector "
                        f"{inj.get('concept')} has no residual norms — backfill norms "
                        f"(POST /api/vectors/backfill-norms) or switch the variant to "
                        f"raw alpha")
                alpha = vm.norm_unit_scale(float(inj["alpha"]),
                                           norms[min(layer, len(norms) - 1)], vnorm)
            else:
                alpha = float(inj["alpha"])
            cells.append(CellInjection(
                layer=layer, vector=vector, alpha=alpha,
                mode="ablate" if is_ablation else "add",
                concept=str(inj.get("concept") or "")))
    return cells


def _preflight_mean_alignment(*, variant: ModelVariant, inj: dict, vectors,
                              layer_range, neutral_mean) -> None:
    """Loud, non-fatal preflight for an UNCENTERED ablating injection —
    the variant twin of ``tasks._warn_on_mean_aligned_ablation`` (same
    threshold, same remedy)."""
    import warnings as _warnings
    concept = inj.get("concept") or "?"
    if neutral_mean is None:
        _warnings.warn(
            f"variant '{variant.name}': ablating '{concept}' with no stored "
            f"neutral mean — mean-alignment preflight impossible (artifact "
            f"predates the neutral-mean stamp or was extracted without a "
            f"neutral corpus). λ=1 ablation of a mean-aligned direction "
            f"collapses generation; re-extract to enable the check and "
            f"neutral-mean centering", UserWarning, stacklevel=4)
        return
    worst_layer, worst = -1, 0.0
    for layer in layer_range:
        alignment = vm.mean_alignment(vectors.per_layer[layer], neutral_mean[layer])
        if alignment > worst:
            worst_layer, worst = layer, alignment
    if worst > vm.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD:
        _warnings.warn(
            f"variant '{variant.name}': ablation direction for '{concept}' is "
            f"strongly aligned with the neutral residual mean (|cos| "
            f"{worst:.2f} at layer {worst_layer}; warn threshold "
            f"{vm.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD}). Full ablation of "
            f"mean-aligned directions collapses generation into single-token "
            f"repetition — declare \"centering\": \"neutralMean\" on the "
            f"injection, or expect incoherent output", UserWarning,
            stacklevel=4)


def save_variant(variant: ModelVariant, root: str | None = None) -> dict:
    """Persist a variant artifact to a fresh run directory + content hash."""
    from .run_config import write_run_config
    run_dir = paths.make_unique_run_directory(f"variant-{variant.name}", root)
    write_run_config(run_dir, "variant-save", model_id=variant.base_model_id,
                     revision=variant.base_revision)
    path = os.path.join(run_dir, f"{variant.name}.json")
    blob = json.dumps(variant.to_dict(), indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(blob)
    return {"runDirectory": run_dir, "path": path,
            "hash": hashlib.sha256(blob.encode()).hexdigest()}


def list_variants(root: str | None = None) -> list[dict]:
    runs = paths.runs_directory(root)
    out: list[dict] = []
    if not os.path.isdir(runs):
        return out
    for entry in sorted(os.listdir(runs), reverse=True):
        run_dir = os.path.join(runs, entry)
        if not os.path.isdir(run_dir):
            continue
        for fname in os.listdir(run_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(run_dir, fname), encoding="utf-8") as handle:
                    d = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if "baseModelID" in d and "injections" in d and "promptMode" in d:
                out.append({"name": d.get("name", fname[:-5]),
                            "baseModelID": d["baseModelID"],
                            "path": os.path.join(run_dir, fname),
                            "injections": len(d.get("injections", [])),
                            "adapters": len(d.get("adapters", []))})
    return out


def missing_artifacts(variant: ModelVariant,
                      root: str | None = None) -> list[dict]:
    """Referenced vector/adapter/neutral-basis artifacts that do not resolve
    on this host — through the SAME resolution the generation path uses
    (:func:`paths.resolve_artifact`, rebase fallback included), so this
    predicts exactly what generation would fail to open.

    One entry per miss: ``{"kind": "vector"|"adapter"|"neutralPCBasis",
    "reference": str, "reason": str}``. Shared by the upload route (reports
    the list), inline variant specs (400 on any miss), and the run/panel
    artifact preflight (refuses before the model loads — a dangling
    reference used to surface as a FileNotFoundError after a 51 GiB load,
    2026-08-04)."""
    missing: list[dict] = []
    for inj in variant.injections:
        ref = inj.get("vectorArtifactID")
        if not ref:
            missing.append({"kind": "vector", "reference": "",
                            "reason": "missing reference"})
            continue
        resolved = paths.resolve_artifact(ref, root)
        if not (os.path.exists(resolved)
                or (os.path.exists(resolved + ".safetensors")
                    and os.path.exists(resolved + ".json"))):
            missing.append({"kind": "vector", "reference": ref,
                            "reason": "not found"})
    for adapter in variant.adapters:
        ref = adapter.get("adapterDirectory") or adapter.get("artifactPath") or ""
        resolved = paths.resolve_artifact(ref, root) if ref else ref
        if not ref or not os.path.isdir(resolved):
            missing.append({"kind": "adapter", "reference": ref,
                            "reason": "not found"})
    if variant.neutral_pc_basis_path:
        resolved = paths.resolve_artifact(variant.neutral_pc_basis_path, root)
        if not os.path.exists(resolved):
            missing.append({"kind": "neutralBasis",
                            "reference": variant.neutral_pc_basis_path,
                            "reason": "not found"})
    return missing


def _adapter_directory(variant: ModelVariant, root: str | None = None) -> str:
    """The variant's first adapter directory, resolved and existence-checked.

    ``root`` scopes resolution to an EXPLICIT workspace. Absent (every
    historical caller) it resolves against the process default, exactly as
    before; a caller handed a root must pass it or it loads a different
    workspace's adapter (external review round 3)."""
    adapter_dir = paths.resolve_artifact(
        variant.adapters[0].get("adapterDirectory") or "", root)
    if not adapter_dir or not os.path.isdir(adapter_dir):
        raise ValueError(f"variant '{variant.name}' adapter directory not found: {adapter_dir}")
    return adapter_dir


# This engine's native adapter format (see lora_train.ADAPTER_FORMAT — the
# pinned cross-engine contract; Swift stamps "mlx-lora"). "peft" is this
# engine's own pre-contract legacy stamp and loads fine here.
_NATIVE_ADAPTER_FORMATS = {"hf-peft-lora", "peft"}


def adapter_sidecar_path(adapter_dir: str) -> str:
    """WHERE the adapter's provenance sidecar lives: ``<run>/<name>.json``
    beside ``<run>/<name>/``. Split out from :func:`adapter_sidecar` so the
    freeze/verify pin surface can hash the sidecar BYTES (the file is a
    pinned input under the LoRA readiness contract §9) without re-deriving
    the layout convention in a second place."""
    normalized = os.path.normpath(adapter_dir)
    return os.path.join(os.path.dirname(normalized),
                        f"{os.path.basename(normalized)}.json")


def adapter_sidecar(adapter_dir: str) -> dict | None:
    """The adapter's provenance sidecar, written by training NEXT TO the
    adapter directory (``<run>/<name>.json`` beside ``<run>/<name>/``).
    Returns None when absent or unreadable (legacy adapters have none)."""
    path = adapter_sidecar_path(adapter_dir)
    try:
        with open(path, encoding="utf-8") as handle:
            d = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return d if isinstance(d, dict) else None


class AdapterIdentityError(ValueError):
    """The adapter on disk is not the one the agent declares.

    A refusal, never a downgrade: a mismatch means the bytes that will shape
    generation are not the bytes the agent was pinned with, so nothing
    measured through it is attributable.
    """


_ADAPTER_WEIGHTS_BY_FORMAT = {
    "hf-peft-lora": "adapter_model.safetensors",
    "peft": "adapter_model.safetensors",
    "mlx-lora": "adapters.safetensors",
}
_ADAPTER_WEIGHT_FILENAMES = ("adapters.safetensors", "adapter_model.safetensors")

#: Stable machine identifier for the composite hash, plus the specification a
#: reader needs to recompute it. A bare prose description is not an algorithm:
#: the NUL separator between filename and bytes, the sort order, and the file
#: filter all change the digest, and a consumer a year from now must be able to
#: tell whether a stored number was produced by this exact rule.
ADAPTER_CONTENT_HASH_ALGORITHM = "steerlab-adapter-content-v1"
ADAPTER_CONTENT_HASH_SPEC = (
    "sha256 over the adapter's forward-pass files in sorted filename order: "
    "adapter_config.json plus every *.safetensors/*.bin, each contributed as "
    "utf-8(filename) || 0x00 || file bytes. Non-files and other names are "
    "skipped. Implementation: model_variant.adapter_content_hash")


def _sha256_file(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def adapter_weights_file(adapter_dir: str) -> str:
    """The ONE weights file this adapter's declared format uses.

    Resolution order: the sidecar's ``adapterFormat`` decides when it names a
    format we know; otherwise the single present candidate wins. A directory
    holding BOTH candidates with no format stamp is refused rather than
    guessed — a stale or converted directory is exactly where hashing the
    wrong file would look like success.
    """
    present = [f for f in _ADAPTER_WEIGHT_FILENAMES
               if os.path.isfile(os.path.join(adapter_dir, f))]
    if not present:
        raise AdapterIdentityError(
            f"adapter directory {adapter_dir!r} contains none of "
            f"{list(_ADAPTER_WEIGHT_FILENAMES)} — there is nothing to verify")
    sidecar = adapter_sidecar(adapter_dir) or {}
    declared = _ADAPTER_WEIGHTS_BY_FORMAT.get(str(sidecar.get("adapterFormat")))
    if declared:
        if declared not in present:
            raise AdapterIdentityError(
                f"adapter {adapter_dir!r} declares format "
                f"{sidecar.get('adapterFormat')!r}, whose weights file "
                f"{declared!r} is not present (found {present})")
        return declared
    if len(present) > 1:
        raise AdapterIdentityError(
            f"adapter {adapter_dir!r} contains {present} and declares no "
            f"adapterFormat — refusing to guess which one the loader will "
            f"read. Hashing the wrong file would verify a set of weights the "
            f"forward pass never uses")
    return present[0]


def verified_adapter_identity(variant, root: str | None = None) -> list[dict]:
    """What the agent's adapters ARE, measured — not what they claim to be.

    ONE implementation, shared by the standalone probe and by study-run
    condition resolution. It previously existed only in the probe, and the
    study path grew a second, weaker copy that read "pinned" as "a hash string
    was declared". Two consequences, both live (external review round 8):

    * The copy read the entries with ``getattr``, but ``ModelVariant`` keeps
      adapters as DICTS — so every fully pinned production agent resolved to
      ``adapterDirectory: "?"``, unpinned, and was stamped exploratory. The
      focused tests passed because they used stand-in objects.
    * Presence of a declaration is not verification. A declared hash is a
      claim ABOUT an adapter, not a measurement of the one that will load, so
      an adapter could drift after pinning and still be called qualified.

    Returns one entry per adapter: what was declared, what is on disk, the
    file that was hashed, and whether they match. Raises on a MISMATCH (the
    bytes are wrong — refuse) and on an unhashable declared file. A LEGACY
    absent pin is not an error: it is recorded as unpinned, which downgrades
    the claim without blocking exploratory work.
    """
    from . import paths

    out: list[dict] = []
    for declared in (getattr(variant, "adapters", None) or []):
        # Adapters are dicts on the artifact, and stay dicts through
        # ModelVariant.from_dict. Tolerate an object form too, so this cannot
        # silently degrade if a caller passes a richer type.
        def field(*names):
            for name in names:
                if isinstance(declared, dict):
                    if declared.get(name) is not None:
                        return declared[name]
                elif getattr(declared, name, None) is not None:
                    return getattr(declared, name)
            return None

        reference = field("adapterDirectory", "adapter_directory") or ""
        resolved = paths.resolve_artifact(reference, root)
        if not resolved or not os.path.isdir(resolved):
            raise AdapterIdentityError(
                f"variant adapter directory {reference!r} is not present in "
                f"this workspace — the readout would describe a trajectory "
                f"the agent's adapter never shaped")
        entry = {"adapterDirectory": reference,
                 "adapterName": field("name")}
        # The ESTABLISHED contract is two separate hashes: `adapterHash` is
        # the WEIGHTS FILE alone and `configHash` is `adapter_config.json`
        # alone — what both writers emit (lora_train's adapterBytesHash /
        # adapterConfigHash, and Swift's FineTuningPanel).
        for key, filename in (
                ("adapterHash", adapter_weights_file(resolved)),
                ("configHash", "adapter_config.json")):
            claimed = field(key, "adapter_hash" if key == "adapterHash"
                            else "config_hash")
            if not claimed:
                # Absent is recorded EXPLICITLY, never silently skipped:
                # "we checked what was there" and "this is unverifiable"
                # must not read identically.
                entry[f"{key}Pinned"] = False
                continue
            candidate = os.path.join(resolved, filename)
            actual = _sha256_file(candidate) if os.path.isfile(candidate) else None
            entry[f"{key}Pinned"] = True
            entry[f"{key}File"] = filename
            entry[f"{key}Declared"] = claimed
            entry[f"{key}Live"] = actual
            if actual is None:
                raise AdapterIdentityError(
                    f"adapter {reference!r} declares {key} but {filename!r} "
                    f"is not present to hash — the agent cannot be verified")
            if actual != claimed:
                raise AdapterIdentityError(
                    f"adapter {reference!r} {key} is {actual[:12]}… but the "
                    f"variant declares {claimed[:12]}… — the adapter on disk "
                    f"is not the one this agent was pinned with")
            entry[f"{key}Verified"] = True
        entry["adapterContentHash"] = adapter_content_hash(resolved)
        entry["adapterContentHashAlgorithm"] = ADAPTER_CONTENT_HASH_ALGORITHM
        entry["adapterContentHashSpec"] = ADAPTER_CONTENT_HASH_SPEC
        if not entry.get("configHashPinned"):
            entry["configurationUnpinned"] = (
                "this agent declares no configHash, so adapter_config.json is "
                "unverified — changing it alters execution while the agent's "
                "declared identity is unchanged. adapterContentHash records "
                "the LIVE state but cannot prove it matches what was pinned")
        out.append(entry)
    return out


def _peft_unloadable_config(adapter_dir: str) -> bool:
    """True when ``adapter_config.json`` exists but has no ``peft_type`` —
    the MLX-LoRA layout (keys like ``fine_tune_type``/``lora_parameters``).
    PEFT hard-fails on such a config ("missing required keys: {'peft_type'}"),
    so this is a fact about the bytes, not a guess about provenance. A
    missing/unreadable config is left for PEFT to report itself."""
    try:
        with open(os.path.join(adapter_dir, "adapter_config.json"),
                  encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(cfg, dict) and "peft_type" not in cfg


def require_native_adapter(adapter_dir: str) -> None:
    """Refuse an adapter this engine provably cannot load.

    Cross-loading actually breaks — PEFT cannot read the MLX ``adapters.
    safetensors`` layout — so an EXPLICIT foreign ``adapterFormat``/
    ``substrate`` stamp is a clear refusal, not a weird downstream failure.
    An absent sidecar or absent fields is legacy/unknown and is not treated
    as a guessable value — but the config CONTENT is still checked: a
    pre-stamp MLX adapter (no sidecar, config without ``peft_type``) refuses
    with the same retrain message instead of PEFT's opaque TypeError.
    """
    from ..steering.vector_store import SUBSTRATE
    sidecar = adapter_sidecar(adapter_dir) or {}
    fmt = sidecar.get("adapterFormat")
    substrate = sidecar.get("substrate")
    foreign_format = fmt is not None and str(fmt) not in _NATIVE_ADAPTER_FORMATS
    foreign_substrate = substrate is not None and str(substrate) != SUBSTRATE
    if foreign_format or foreign_substrate:
        raise ValueError(
            f"adapter '{adapter_dir}' was trained as '{fmt or 'unknown'}' on "
            f"'{substrate or 'unknown'}'; this engine loads hf-peft-lora "
            f"adapters — retrain on this substrate")
    if _peft_unloadable_config(adapter_dir):
        raise ValueError(
            f"adapter '{adapter_dir}' has an adapter_config.json without "
            f"'peft_type' — that is the MLX-LoRA layout, which PEFT cannot "
            f"load; this engine loads hf-peft-lora adapters — retrain on "
            f"this substrate")


def _require_adapter_api(model) -> None:
    try:
        import peft  # noqa: F401 - the transformers adapter API requires peft installed
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("variant adapters need peft: pip install -e .[lora]") from exc
    if not hasattr(model.model, "load_adapter"):
        raise RuntimeError("loaded model does not support PEFT adapters (transformers too old)")


def apply_adapter(model, variant: ModelVariant, *,
                  root: str | None = None):
    """Load the variant's first PEFT adapter onto the model via the
    transformers-native adapter API (``load_adapter``/``set_adapter``).

    This keeps ``model.model`` a normal causal LM with the adapter *active* — so
    ``model.model.generate`` actually uses it and the forward hooks keep firing —
    and crucially leaves ``delete_adapter`` available so the adapter can be
    removed between variant conditions (the discarded ``PeftModel.from_pretrained``
    wrapper could neither be trusted to apply nor cleanly removed). Returns the
    adapter name to remove later, or None for pure-steering variants.
    """
    if not variant.adapters:
        return None
    adapter_dir = _adapter_directory(variant, root)
    # Substrate/format gate BEFORE touching the adapter API: an explicit
    # foreign stamp (e.g. an mlx-lora adapter on a shared tree) refuses with
    # the retrain message instead of a peft load failure.
    require_native_adapter(adapter_dir)
    _require_adapter_api(model)
    name = variant.name
    model.model.load_adapter(adapter_dir, adapter_name=name)
    model.model.set_adapter(name)         # make it the active adapter
    if hasattr(model.model, "enable_adapters"):
        # PEFT's disable flag is per-*layer*, not per-adapter-name: if a chat
        # request previously parked a cached adapter disabled on this shared
        # registry slot, set_adapter alone would leave the tuner layers
        # disabled and this adapter would silently not apply. Re-enable
        # explicitly (a no-op on a never-disabled model).
        model.model.enable_adapters()
    return name


def remove_adapter(model, handle) -> None:
    if handle is None:
        return
    try:
        # Disable then delete so generation reverts to the base weights cleanly.
        if hasattr(model.model, "disable_adapters"):
            model.model.disable_adapters()
        model.model.delete_adapter(handle)
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


_ADAPTER_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def adapter_content_hash(adapter_dir: str) -> str:
    """SHA-256 over the adapter files that affect the forward pass
    (``adapter_config.json`` + the adapter weight files). Deliberately content-
    based — mtime is not sufficient to detect a retrained adapter written to
    the same path."""
    digest = hashlib.sha256()
    for fname in sorted(os.listdir(adapter_dir)):
        if fname != "adapter_config.json" and not fname.endswith(_ADAPTER_WEIGHT_SUFFIXES):
            continue
        path = os.path.join(adapter_dir, fname)
        if not os.path.isfile(path):
            continue
        digest.update(fname.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


class ChatAdapterCache:
    """Per-model-slot LoRA adapter cache for the interactive CHAT path only.

    Measured experiment paths (``tasks.py`` / ``multi_agent.py``) keep the
    strict load → generate → delete lifecycle via :func:`apply_adapter` /
    :func:`remove_adapter`; chat repeats the same variant every turn, so
    re-reading the adapter from disk per request is pure waste. The cache keeps
    at most one adapter loaded per model slot, keyed by
    ``(realpath(adapter dir), content hash)``, and parks it *loaded but
    disabled* between requests.

    Why disabled-not-deleted is safe: transformers'
    ``PeftAdapterMixin.disable_adapters()`` sets PEFT's per-layer
    ``disable_adapters`` flag, and a disabled, never-merged LoRA layer forwards
    through ``base_layer`` alone — so a subsequent plain generation on the same
    model is bit-identical to a never-adapted model. We never call
    ``merge_adapter``, so there is nothing to unmerge.

    Orthogonal to steering: this class only touches the transformers adapter
    API on ``model.model`` — never ``hooks.py`` / the injector sessions.
    """

    ADAPTER_NAME = "steerlab-chat-adapter"  # distinct from experiment adapters
                                            # (named after the variant) so the
                                            # two lifecycles can never collide

    def __init__(self):
        self._key: tuple[str, str] | None = None

    def activate(self, model, variant: ModelVariant) -> bool:
        """Make the variant's adapter active (load-or-reuse); ensure adapters
        are inactive for adapter-less variants. Returns True when an adapter
        is active for this generation."""
        if not variant.adapters:
            self.deactivate(model)
            return False
        adapter_dir = _adapter_directory(variant)
        require_native_adapter(adapter_dir)  # same gate as the measured path
        _require_adapter_api(model)
        key = (os.path.realpath(adapter_dir), adapter_content_hash(adapter_dir))
        lm = model.model
        if self._key != key:
            if self._key is not None:
                # Different adapter (or same path, new bytes): drop the stale one.
                self._key = None
                lm.delete_adapter(self.ADAPTER_NAME)
            lm.load_adapter(adapter_dir, adapter_name=self.ADAPTER_NAME)
            self._key = key
        lm.set_adapter(self.ADAPTER_NAME)
        if hasattr(lm, "enable_adapters"):
            lm.enable_adapters()  # re-arm layers parked disabled by deactivate()
        return True

    def deactivate(self, model) -> None:
        """Park the cached adapter dormant after a generation (or before a
        plain / adapter-less one). No-op when nothing is cached."""
        if self._key is None:
            return
        try:
            if hasattr(model.model, "disable_adapters"):
                model.model.disable_adapters()
        except Exception:  # pragma: no cover - best-effort parking
            pass

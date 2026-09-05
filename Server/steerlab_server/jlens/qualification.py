"""Stage 4: qualify an imported lens against ONE exact runtime.

The consumer side of this has existed since 2026-07-31 and found nothing,
because there was no producer: ``schemas.Qualification`` and
``JLensRecord.qualification_for`` are what the freeze gate calls
(``experiment_store._check_jlens_readout``), and with no verb able to write a
qualification, **no J-lens study could freeze non-force by any supported
path**. This module is the producer. It makes the existing gate satisfiable;
it does not make it softer.

What a qualification asserts (plan §3.3)
----------------------------------------
Not that the lens is correct in general, and emphatically not that the unknown
fit-time model revision has been recovered. It asserts that *these exact
imported bytes were tested against this exact runtime and accepted as an
external instrument*. "This exact runtime" is model + revision + **dtype +
quantization state**, because geometry cannot see numerics: a float16 or
bitsandbytes-8bit 27B has the same layer count, hidden size, vocabulary, and
head shape as the bf16 one, so every shape check passes identically while the
residual the Jacobian is applied to has different numerics from the one it was
fit against.

**Absent is not a match.** A runtime whose dtype cannot be resolved is refused
rather than assumed (:func:`resolve_runtime`). An unquantized runtime is a
*resolved* answer and is stamped ``None`` — the same value the manifest
carries when it declares no quantization — so the gate's exact-match rule
lines the two up rather than comparing ``"none"`` against nothing.

Per-check verdicts, not one boolean
-----------------------------------
Seven checks (:data:`CHECKS`), each individually reported with the numbers it
measured. A record that says only ``passed: false`` cannot tell a researcher
whether the lens is wrong, the runtime is wrong, or the reference package is
missing — and the interesting failures are exactly the ones a single boolean
erases. All seven are blocking (:data:`BLOCKING`); what is deliberately NOT
blocking is a *fit-dtype divergence*, which ``runtimeNumerics`` stamps and
passes, because refusing on a config field would decide by declaration what
the remaining checks exist to decide by measurement.

Failure is evidence, and records are append-only
------------------------------------------------
A failed qualification is written too: "we tested this runtime and it did not
pass" is a finding a later reader needs, and deleting it would leave the
absence looking like an untested runtime. Re-running appends a new entry
(:func:`append_qualification` refuses to replace an existing one), so the
history of what was tested when survives, and the freeze gate keeps its own
rule — it looks for a *passing* entry for the exact runtime and ignores the
rest.

Tier is read from the curated table or the import declaration, never from
the qualify caller
-----------------------------------------------------------------------------
``jlens qualify`` RUNS on a testing-tier model on purpose — rehearsing the
mechanics on 4B before a 27B node is booked is the whole point of the cheap
pass — and the record it writes is stamped ``tier: "testing"`` and refused by
freeze. The tier comes from :func:`importer.tier_of` — the curated row when
the study has decided about the model, else the tier the researcher declared
at ``jlens import --tier`` and the record carries — computed here, so no
qualify argument and no later edit of an existing record can upgrade it.

Server-only (lens artifacts are PyTorch/HF-native and activations do not
transfer across substrates); any model with a published lens whose final norm
is a foldable RMSNorm (:mod:`norm_convention`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

from .schemas import JLensError, Qualification, _iso

#: Every check, in execution order. The list is the artifact's contract: a
#: record naming fewer checks than this was written by an older engine, and a
#: reader can tell rather than infer.
CHECKS = (
    "geometry",            # layer count, width, vocabulary, tokenizer identity
    "runtimeNumerics",     # dtype + quantization, resolved and pinned
    "jacobianFinite",      # finite values and square shape, every source layer
    "referenceAgreement",  # adapter vs the pinned reference on fixed residuals
    "stableReadouts",      # committed prompts: finite, repeatable, recorded
    "causalSmoke",         # dose response of one derived token direction
    "capabilityGuard",     # the battery at the intended layer/alpha range
)

#: Checks whose failure means the lens is not qualified for this runtime.
#: ``capabilityGuard`` is here because a lens whose intended dose range breaks
#: the model is not usable at that range, which is what the qualification is
#: about; its TOLERANCE is a declared number (:data:`CAPABILITY_TOLERANCE`),
#: not a hidden one.
BLOCKING = ("geometry", "runtimeNumerics", "jacobianFinite",
            "referenceAgreement", "stableReadouts", "causalSmoke",
            "capabilityGuard")

#: Max absolute logit difference tolerated between this engine's canonical
#: readout and the pinned reference implementation on identical residuals,
#: compared with the reference's output head promoted to float32 (the default
#: mode since the ruling below). So it bounds the MATH: a missing RMSNorm gain
#: rescales by ~an order of magnitude on Gemma (plan §11.1) and cannot hide
#: under it, while two correct float32 implementations sit near 1e-6 apart.
#: Pinned by test.
#:
#: RULING (2026-09-05): the comparison runs in float32 by default. The
#: reference computes at the model's runtime dtype, so every bf16 reference
#: logit lies on the bf16 grid, whose spacing exceeds this tolerance once
#: |logit| >= 8 (one ulp is 0.0625 there). Measured on the 4B testing lens:
#: 0.0837 at the runtime dtype against 1.9e-6 under promotion, both vocabulary
#: paths, so the deviation was the reference's rounding and nothing else; the
#: 27B run's 0.07059 on a fixture logit of ~96 is the same phenomenon. An
#: absolute tolerance below the reference's own resolution could never pass,
#: and scaling it with magnitude would have made the number a function of the
#: fixtures rather than of the math. Promotion touches only the output head
#: the comparison reads (:data:`REFERENCE_OUTPUT_HEAD`), never the decoder
#: stack, so it costs a second copy of the head and nothing more. The
#: instruments stay: ``measured["perComparison"]`` records the raw (layer,
#: fixture row, token) logit PAIR behind every comparison, and
#: :data:`REFERENCE_FP32_ENV` set to ``0`` runs the runtime-dtype comparison
#: as a diagnostic. The record stamps which mode produced it either way.
REFERENCE_TOLERANCE = 5e-2

#: The reference path's dtype mode for the agreement check. Unset (or
#: ``1``/``true``/``yes``/``on``): the reference's output head is promoted to
#: float32 for the comparison — this engine's path is float32 already — and
#: restored afterwards. That is the DEFAULT. ``0``/``false``/``no``/``off``:
#: the reference keeps its runtime dtype, a diagnostic for one question — how
#: far the runtime-dtype reference sits from the float32 math — whose failures
#: above the tolerance at |logit| >= 8 are the expected reading. Any other
#: value is refused, not guessed. The record stamps the mode AND where it came
#: from in every case, so a diagnostic run can never be read as the default by
#: omission. See :func:`_promote_reference_to_fp32`.
REFERENCE_FP32_ENV = "STEERLAB_JLENS_REFERENCE_FP32"

#: The reference attributes the comparison reads through ``unembed``: the
#: final norm and the output head. ``_embed_tokens`` is the tied twin of
#: ``_lm_head`` on Gemma and is listed so an untied head is promoted at both
#: ends; ``layers`` is deliberately absent — the comparison transports through
#: OUR float32 Jacobian, never through the reference's decoder blocks, and
#: promoting the whole stack would double a 27B model's weight memory for
#: nothing. (The walk over every module that preceded this list did exactly
#: that, which is why promotion could only ever have been opt-in.)
REFERENCE_OUTPUT_HEAD = ("_final_norm", "_lm_head", "_embed_tokens")

#: How many per-comparison rows the record keeps. The check runs
#: ``len(layers) × REFERENCE_FIXTURE_ROWS`` comparisons, so this bites only on
#: an all-layer qualification; when it does, the rows retained are the LARGEST
#: deviations (the ones a tolerance question is about) and the record says so.
REFERENCE_MAX_RECORDED_COMPARISONS = 512

#: Deterministic residual fixtures for the reference-agreement check: seeded
#: draws, scaled to a plausible residual magnitude. Seeded rather than
#: committed as bytes because the check compares TWO implementations on the
#: SAME input — the input's provenance does not enter the comparison, only its
#: reproducibility does.
REFERENCE_FIXTURE_SEED = 20260813
REFERENCE_FIXTURE_ROWS = 4
REFERENCE_FIXTURE_SCALE = 30.0

#: The dose ladder the causal smoke test walks, in residual-norm units (the
#: denominator is measured on the fixture prompts themselves — see
#: :func:`_causal_smoke`). Overridable per invocation with ``--alpha-range``.
DEFAULT_ALPHA_LADDER = (0.04, 0.08, 0.12)

#: How far the capability battery may fall below baseline at the top of the
#: ladder before the guard fails. The same number the historical sweep
#: selection rule uses (``sweep_selection.DEFAULT_CAPABILITY_TOLERANCE``), so
#: "the dose range is usable" means one thing across the engine; a test pins
#: the two equal.
CAPABILITY_TOLERANCE = 0.15

#: Committed prompts (checks 5 and 6). Ships with the ENGINE rather than the
#: workspace: qualification is an instrument self-test, and a check whose
#: inputs vary per workspace would not be comparable between two of them.
FIXTURE_PROMPTS = os.path.join(os.path.dirname(__file__), "fixtures",
                               "qualification-prompts.jsonl")


class QualificationRefused(JLensError):
    """The qualification could not be ATTEMPTED — a distinct outcome from
    failing it. A refused attempt writes no record: there is nothing to
    record, because nothing was tested."""


@dataclass
class CheckResult:
    """One named verdict plus the numbers behind it.

    ``measured`` is the point of the artifact. A researcher reading a stored
    qualification a month later needs the tokenizer hash that was compared,
    the max logit deviation that was seen, and the accuracies the battery
    scored — not the conclusion that they were fine.
    """

    name: str
    passed: bool
    detail: str = ""
    measured: dict = field(default_factory=dict)
    #: Set when the check could not run at all (missing optional dependency,
    #: absent fixture). Distinguished from a measured failure because the
    #: remedy is completely different.
    skipped: bool = False

    def to_dict(self) -> dict:
        out = {"passed": self.passed, "detail": self.detail,
               "measured": self.measured}
        if self.skipped:
            out["skipped"] = True
        return out


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------

def resolve_runtime(model) -> tuple[str, str | None]:
    """``(dtype, quantization)`` for a loaded model, or a refusal.

    The dtype comes from the model's own parameters when the loader did not
    stamp one, never from a request or a default: the whole reason this is a
    qualification input is that a *declared* dtype can be a false claim while
    the load ran something else.

    Quantization is read from the config's ``quantization_config``. An ABSENT
    one is a resolved answer — the runtime is not quantized — and is returned
    as ``None`` so it matches a manifest that declares no quantization. A
    present but unnameable one is refused: "there is a quantizer here and I
    cannot say which" must never be recorded as "not quantized".
    """
    from ..steering.model_loader import normalize_dtype

    dtype = normalize_dtype(getattr(model, "dtype", None))
    if dtype is None:
        inner = getattr(model, "model", None)
        try:
            param = next(inner.parameters())
        except (AttributeError, StopIteration, TypeError):
            param = None
        if param is not None:
            dtype = normalize_dtype(str(param.dtype).replace("torch.", ""))
    if dtype is None:
        raise QualificationRefused(
            "could not resolve the runtime dtype — a J-lens qualification is "
            "keyed by numeric configuration, and an unresolvable one is "
            "refused rather than assumed (an absent dtype must never match a "
            "qualified one). Load the model through this engine's loader, "
            "which stamps the dtype it actually ran")

    config = getattr(getattr(model, "model", None), "config", None)
    raw = getattr(config, "quantization_config", None)
    if raw is None:
        return dtype, None
    method = None
    if isinstance(raw, dict):
        method = raw.get("quant_method") or raw.get("quantization_method")
    else:
        method = (getattr(raw, "quant_method", None)
                  or getattr(raw, "quantization_method", None))
    method = str(getattr(method, "value", method) or "").strip()
    if not method:
        raise QualificationRefused(
            "the runtime carries a quantization configuration this engine "
            "cannot name — refusing rather than recording it as unquantized, "
            "which is the one reading that would let a differently-quantized "
            "runtime inherit this qualification")
    return dtype, method


def tier_for(model_id: str, record=None) -> str:
    """The evidence tier of a model: the curated row, else the lens record's
    import declaration, else ``"unknown"``.

    Computed here on every write so a qualify caller cannot supply it and an
    existing record cannot be edited into a higher one: the tier is this
    project's scope decision (CLAUDE.md 2026-07-27, extended 2026-09-05 to a
    declaration at import for uncurated models), and it is what stops a
    rehearsal from being cited as evidence.
    """
    from .importer import tier_of

    return tier_of(model_id, record)[0]


def qualification_id(*, lens_id: str, lens_sha: str | None, model_id: str,
                     revision: str, dtype: str, quantization: str | None,
                     qualified_at: str) -> str:
    """Content-addressed id over everything the entry is about, plus its
    timestamp — so re-qualifying the same runtime appends a distinguishable
    entry rather than colliding with the one it is meant to sit beside."""
    payload = json.dumps({
        "lensID": lens_id, "lensSHA256": lens_sha, "modelID": model_id,
        "revision": revision, "dtype": dtype,
        "quantization": quantization, "qualifiedAt": qualified_at,
    }, sort_keys=True, separators=(",", ":"))
    return "q-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def append_qualification(record, qualification: Qualification, *,
                         root: str | None = None) -> str:
    """Append one entry to a lens record and persist it.

    Refuses to replace an existing entry with the same id. Qualifications are
    evidence about what was tested when; a re-run that overwrote its
    predecessor would erase a failure and leave a pass where the history said
    something else.
    """
    from . import lens_store

    existing = {q.qualificationID for q in record.qualifications}
    if qualification.qualificationID in existing:
        raise QualificationRefused(
            f"lens '{record.lensID}' already carries qualification "
            f"{qualification.qualificationID} — qualifications are appended, "
            f"never replaced, so a re-run cannot overwrite the verdict it is "
            f"meant to sit beside")
    record.qualifications.append(qualification)
    return lens_store.save(record, root)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def verify_converted(record, root: str | None = None) -> None:
    """Re-hash the converted per-layer artifact against the record.

    The upstream tensor is hash-pinned by the manifest; the CONVERTED cache —
    the bytes generation actually reads — was not checked anywhere outside
    G0's acquisition check. It is a derived, rebuildable file living in a
    mutable library subtree, so it is exactly the thing that can be rebuilt
    from different upstream bytes while keeping its path and its record entry.

    Called once per readout build rather than per ``load_layer``: at 27B the
    artifact is several GB and hashing it per layer would add minutes to every
    arming for no additional assurance.
    """
    from ..experiment import paths
    from .schemas import sha256_file

    if record.converted is None or not record.converted.sha256:
        return
    path = paths.resolve(record.converted.path, root)
    if not os.path.exists(path):
        raise QualificationRefused(
            f"converted lens artifact missing at '{path}' — it is a derived "
            f"cache, so re-import from the hash-pinned upstream")
    live = sha256_file(path)
    if live != record.converted.sha256:
        raise QualificationRefused(
            f"converted lens artifact for '{record.lensID}' does not match the "
            f"hash its import recorded (have {live[:12]}…, recorded "
            f"{record.converted.sha256[:12]}…) — the bytes generation reads "
            f"are not the bytes that were imported; re-import from the "
            f"hash-pinned upstream")


def load_fixture_prompts(path: str | None = None) -> list[dict]:
    """The committed prompt fixtures, header row dropped."""
    target = path or FIXTURE_PROMPTS
    rows: list[dict] = []
    try:
        with open(target, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QualificationRefused(
                        f"{target}:{line_no} is not valid JSON: {exc}") from exc
                if "fixtureSet" in row:
                    continue                     # header
                if not row.get("prompt"):
                    raise QualificationRefused(
                        f"{target}:{line_no} has no 'prompt'")
                rows.append(row)
    except OSError as exc:
        raise QualificationRefused(
            f"no qualification prompt fixtures at {target}: {exc}") from exc
    if not rows:
        raise QualificationRefused(
            f"{target} contains no prompt rows — the stable-readout and "
            f"causal-smoke checks have nothing to run on")
    return rows


def _reference_residuals(d_model: int, count: int = REFERENCE_FIXTURE_ROWS):
    """Deterministic ``[count, d_model]`` residual fixtures."""
    import torch

    generator = torch.Generator().manual_seed(REFERENCE_FIXTURE_SEED)
    rows = torch.randn(count, d_model, generator=generator,
                       dtype=torch.float32)
    # Scaled to a plausible residual magnitude: agreement at 1e-3 norms would
    # be agreement about nothing, since both paths are near-linear there.
    return rows * (REFERENCE_FIXTURE_SCALE / rows.norm(dim=-1, keepdim=True))


_REFERENCE_FP32_ON = frozenset({"1", "true", "yes", "on"})
_REFERENCE_FP32_OFF = frozenset({"0", "false", "no", "off"})


def _reference_fp32_mode() -> tuple[bool, str]:
    """Which dtype mode the reference path runs in, and where that came from.

    Returns ``(promote, source)``: ``promote`` is whether the reference's
    output head is promoted to float32 for the comparison, ``source`` is
    ``"default"`` when the environment said nothing and ``"env"`` when it
    chose. A value this function cannot read is refused rather than mapped to
    either mode, because a check whose mode nobody can name is not a check
    anyone can cite.
    """
    raw = os.environ.get(REFERENCE_FP32_ENV)
    if raw is None or not raw.strip():
        return True, "default"
    text = raw.strip().lower()
    if text in _REFERENCE_FP32_ON:
        return True, "env"
    if text in _REFERENCE_FP32_OFF:
        return False, "env"
    raise JLensError(
        f"{REFERENCE_FP32_ENV}={raw!r} is not a mode this check can read: "
        f"1/true/yes/on promotes the reference's output head to float32 (the "
        f"default when unset), 0/false/no/off keeps its runtime dtype (a "
        f"diagnostic). Nothing was compared.")


def _promote_reference_to_fp32(ref_model, names):
    """Promote the named float tensors of the reference model to float32.

    ``names`` are attributes of ``ref_model`` (:data:`REFERENCE_OUTPUT_HEAD`
    for the agreement check); one that is absent is skipped, and one that is
    already float32 is skipped and NOT listed as promoted. Returns
    ``(promoted_names, restore)``. The caller MUST call ``restore`` in a
    ``finally``: the reference wraps the study model's live modules, so this
    is a temporary promotion of shared objects, not a copy — and a module tied
    to one already promoted (Gemma's embedding and head share their weight) is
    found already float32 and left to the first one's undo.

    ALL OR NOTHING (external review, 2026-08-26). The promotion mutates those
    shared modules as it walks them, and ``restore`` used to be handed over
    only once the whole walk had succeeded — so a promotion that died part way
    (OOM is the plausible way in: a 27B output head is several GiB) left the
    modules it had already reached in float32 with nobody holding an undo
    for them. The rest of the
    qualification would then run against a model in a dtype nobody chose. Any
    exception now unwinds what this call did, in reverse, before it propagates:
    either the reference is fully promoted or it is exactly as it was found.

    Restoring is exact rather than approximately exact. bfloat16 and float16
    values are all representable in float32, so ``bf16 → fp32 → bf16`` returns
    the identical bits; tensor attributes are restored by putting the ORIGINAL
    object back, which does not even round-trip.

    This costs a second copy of the output head while the check runs — at 27B
    a few GiB, which the default mode pays because the alternative was a
    tolerance the reference's own rounding could never meet (see
    :data:`REFERENCE_TOLERANCE`). It is bounded by ``names``: the decoder
    stack is never touched, so the cost does not scale with the model's depth.

    Note what it does NOT touch: this engine's watchlist path reads
    ``readout.watched_rows``, materialized in float32 at build time, so
    promoting the reference cannot move that operand. The full-vocabulary path
    projects through the model's LIVE output head, which IS the promoted
    module — so under promotion that path is compared float32 against
    float32, which is the math question this check asks; how that path behaves
    at the runtime dtype is ``runtimeNumerics``' subject, not this check's.
    """
    import torch

    attrs = vars(ref_model)
    promoted: list[str] = []
    undo: list = []
    try:
        for name in names:
            if name not in attrs:
                continue
            value = attrs[name]
            if isinstance(value, torch.nn.Module):
                dtypes = {t.dtype for t in
                          list(value.parameters()) + list(value.buffers())
                          if t.is_floating_point()}
                if not dtypes or dtypes == {torch.float32}:
                    continue
                if len(dtypes) > 1:
                    # Mixed float widths: restoring would have to guess which
                    # tensor was which, so leave it and let the stamp show that
                    # this module was not promoted.
                    continue
                original = dtypes.pop()
                # Registered BEFORE the mutation, not after: `Module.to`
                # converts tensor by tensor in place, so a failure inside one
                # module leaves that module half promoted too, and the undo
                # has to cover it.
                undo.append(lambda m=value, d=original: m.to(d))
                value.to(torch.float32)
                promoted.append(name)
            elif isinstance(value, torch.Tensor) and value.is_floating_point() \
                    and value.dtype != torch.float32:
                undo.append(lambda n=name, v=value: setattr(ref_model, n, v))
                setattr(ref_model, name, value.to(torch.float32))
                promoted.append(name)
    except BaseException:
        _unwind_promotion(undo, quiet=True)
        raise

    def restore():
        _unwind_promotion(undo, quiet=False)

    return promoted, restore


def _unwind_promotion(undo: list, *, quiet: bool) -> None:
    """Run a promotion's undo steps in reverse.

    ``quiet`` is the rollback path, where a failing step must not displace the
    exception that caused the rollback — that exception is the news, and a
    swallowed secondary failure still leaves the model closer to how it was
    found than not trying would. The ordinary ``restore`` path does not
    swallow: there, a failure to restore IS the news.
    """
    for step in reversed(undo):
        if quiet:
            try:
                step()
            except Exception:
                pass
        else:
            step()


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def _check_geometry(record, model, *, model_id: str,
                    revision: str | None) -> CheckResult:
    """Shape and coordinate-system compatibility.

    The tokenizer hash is load-bearing and easy to under-rate: the readout is
    indexed by token ID, so the vocabulary IS its coordinate system. A
    tokenizer change re-points every watched token at a different piece while
    every number still looks plausible.
    """
    from . import lens_store
    from .derive import tokenizer_identity_hash

    report = lens_store.compatibility(
        record, model_id=model_id, revision=revision,
        dtype=getattr(model, "dtype", None),
        num_layers=int(getattr(model, "num_layers", 0) or 0),
        hidden_size=int(getattr(model, "hidden_size", 0) or 0))
    problems = list(report["problems"])

    vocab = None
    head = getattr(getattr(model, "model", None), "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is not None:
        vocab = int(weight.shape[0])
    declared = None
    config = getattr(getattr(model, "model", None), "config", None)
    text_config = getattr(config, "text_config", config)
    if text_config is not None:
        declared = getattr(text_config, "vocab_size", None)
    if vocab is not None and declared is not None and int(declared) != vocab:
        problems.append(
            f"output head has {vocab} rows but the config declares "
            f"vocab_size {int(declared)} — the readout's coordinate system is "
            f"ambiguous")

    try:
        tokenizer_hash = tokenizer_identity_hash(model_id, revision)
    except JLensError as exc:
        tokenizer_hash = None
        problems.append(f"tokenizer identity is unreadable: {exc}")
    if tokenizer_hash is None and not problems:
        problems.append(
            "no tokenizer files in the pinned snapshot — the readout is "
            "indexed by token ID, so an unpinnable vocabulary means the "
            "trace's coordinate system cannot be verified later")

    unfitted = [l for l in record.sourceLayers
                if l >= int(getattr(model, "num_layers", 0) or 0)]
    if unfitted:
        problems.append(
            f"lens fits source layers {unfitted} the runtime does not have")

    return CheckResult(
        name="geometry", passed=not problems,
        detail="; ".join(problems) or "geometry and tokenizer identity agree",
        measured={
            "lensDModel": record.dModel,
            "runtimeHiddenSize": int(getattr(model, "hidden_size", 0) or 0),
            "lensTargetLayer": record.targetLayer,
            "runtimeLayerCount": int(getattr(model, "num_layers", 0) or 0),
            "sourceLayers": [record.sourceLayers[0], record.sourceLayers[-1]],
            "vocabSize": vocab,
            "tokenizerHash": tokenizer_hash,
            "fitModelID": record.fit.modelID,
        })


def _check_runtime_numerics(record, *, dtype: str,
                            quantization: str | None) -> CheckResult:
    """Pin the numeric configuration and stamp any divergence from the fit.

    A runtime dtype that differs from the lens's fit dtype does NOT refuse.
    The whole point of qualification is empirical acceptance, and refusing on
    a config field would decide by declaration what this verb exists to decide
    by measurement. The divergence is stamped so it can never be discovered
    later by inference.
    """
    fit_dtype = record.fit.dtype
    diverges = bool(fit_dtype) and fit_dtype != dtype
    detail = f"runtime {dtype}"
    detail += f"/{quantization}" if quantization else ", unquantized"
    if diverges:
        detail += (f"; DIVERGES from the lens fit dtype {fit_dtype} — the "
                   f"remaining checks are what accept or reject that "
                   f"divergence, and this record stamps it either way")
    elif fit_dtype:
        detail += f"; matches the lens fit dtype {fit_dtype}"
    return CheckResult(
        name="runtimeNumerics", passed=True, detail=detail,
        measured={"dtype": dtype, "quantization": quantization,
                  "fitDtype": fit_dtype,
                  "divergesFromFitDtype": diverges})


def _check_jacobian_finite(record, *, root: str | None) -> CheckResult:
    """Every available source-layer Jacobian: square, right width, finite."""
    import torch

    from . import lens_store

    problems: list[str] = []
    checked = 0
    worst = 0.0
    for layer in record.sourceLayers:
        try:
            j = lens_store.load_layer(record, layer, root=root)
        except JLensError as exc:
            problems.append(f"layer {layer}: {exc}")
            continue
        checked += 1
        if tuple(j.shape) != (record.dModel, record.dModel):
            problems.append(
                f"layer {layer} has shape {tuple(j.shape)}, expected "
                f"({record.dModel}, {record.dModel})")
            continue
        promoted = j.to(torch.float32)
        if not torch.isfinite(promoted).all():
            problems.append(f"layer {layer} contains non-finite entries")
            continue
        worst = max(worst, float(promoted.abs().max()))
    return CheckResult(
        name="jacobianFinite", passed=not problems,
        detail="; ".join(problems)
               or f"{checked} source-layer Jacobian(s) square and finite",
        measured={"layersChecked": checked,
                  "maxAbsEntry": worst,
                  "expectedShape": [record.dModel, record.dModel]})


def _check_reference_agreement(record, model, readout, layers: list[int], *,
                               root: str | None) -> CheckResult:
    """This engine's canonical readout vs the PINNED reference, same residuals.

    Two independent implementations of ``softcap(U · RMSNorm(J_l h))`` must
    agree on identical inputs, or one of them is wrong. The reference is the
    ``jlens`` package behind the optional extra; when it is absent the check
    FAILS rather than skips, because "we could not check" and "we checked and
    it agreed" are not the same evidence and a qualification is the artifact
    that decides which one a study is standing on.

    Also asserts, explicitly, that the reference's ``unembed`` still applies
    the model's final norm (plan §12.1). The ``g ⊙ u_t`` fold this engine
    performs silently depends on that, so a future reference version that
    moved it would leave both paths self-consistent and both wrong.

    **Both of this engine's vocabulary paths are compared, not only the
    watchlist** (external review, 2026-09-05). ``LensReadout`` implements the
    canonical formula twice — once folding ``g`` into a watched slice of ``U``,
    once scaling ``z`` and projecting through the model's live output head —
    and this check exercised only the first. The second had been dropping ``g``
    altogether, which reorders tokens rather than rescaling them, so every
    top-k and full-vocabulary rank taken from it read a distribution the model
    does not compute; a green referenceAgreement said nothing about it. The
    full path is compared restricted to the watchlist, against the same
    reference numbers, and is armed only when top-k is configured — when it is
    not, ``fullVocabArmed`` is False and the detail line says so, because "not
    checked" and "checked and agreed" must not look alike.

    **The record carries every comparison, not only the worst one.** For each
    (source layer × fixture row) the record keeps the token the deviation is
    largest at, OUR logit, the REFERENCE logit, and the absolute difference.
    The max alone cannot answer the question a failing run actually raises —
    whether a deviation is large in absolute terms because the operands are
    large — and a 27B investigation stalled three times for exactly that
    reason. ``maxAbsLogitDeviation`` is unchanged and still the summary.

    There are no prompt or position axes here by construction: the fixtures are
    seeded residual ROWS (:func:`_reference_residuals`), so a comparison is
    identified by ``(layer, fixtureRow, tokenID)``.

    The reference's output head is promoted to float32 for the comparison BY
    DEFAULT (ruling 2026-09-05, see :data:`REFERENCE_TOLERANCE`): the two
    paths used to cast differently on the way into the output head, and the
    deviation that produced was the reference's bf16 rounding, which the
    tolerance sits below once logits reach |8|. :data:`REFERENCE_FP32_ENV` set
    to ``0`` keeps the reference at its runtime dtype as a diagnostic; a value
    nobody can read FAILS the check. The record stamps the mode, its source
    and the head's dtype in every case. The tolerance itself is untouched by
    the mode.
    """
    from . import backend as backend_mod

    try:
        fp32_forced, fp32_source = _reference_fp32_mode()
    except JLensError as exc:
        return CheckResult(
            name="referenceAgreement", passed=False, detail=str(exc),
            measured={"referencePackage": backend_mod.REFERENCE_PACKAGE,
                      "referenceCommit": backend_mod.REFERENCE_COMMIT,
                      "referenceFP32Forced": None,
                      "referenceFP32ModeSource": "invalid",
                      "referenceFP32EnvVar": REFERENCE_FP32_ENV})
    try:
        backend_mod.require_reference()
        import jlens as reference
    except JLensError as exc:
        return CheckResult(
            name="referenceAgreement", passed=False, skipped=True,
            detail=str(exc),
            measured={"referencePackage": backend_mod.REFERENCE_PACKAGE,
                      "referenceCommit": backend_mod.REFERENCE_COMMIT,
                      "referenceFP32Forced": fp32_forced,
                      "referenceFP32ModeSource": fp32_source,
                      "referenceFP32EnvVar": REFERENCE_FP32_ENV})

    inner = model.model
    ref_model = reference.from_hf(inner, getattr(model, "tokenizer", None),
                                  # Never mutate the study tokenizer's BOS
                                  # behaviour to run a check on it.
                                  force_bos=False)

    promoted: list[str] = []
    restore = None
    # The promotion is INSIDE the guard that undoes it, not before it (external
    # review, 2026-08-26). `_promote_reference_to_fp32` unwinds its own partial
    # work, so this is belt and braces — but a reader should not have to go and
    # check that to see that the mutation and its restoration are one region.
    try:
        if fp32_forced:
            promoted, restore = _promote_reference_to_fp32(
                ref_model, REFERENCE_OUTPUT_HEAD)
        result = _compare_against_reference(
            record, readout, layers, ref_model, reference, backend_mod,
            root=root, fp32_forced=fp32_forced, fp32_source=fp32_source,
            promoted=promoted)
    finally:
        if restore is not None:
            restore()
    return result


def _compare_against_reference(record, readout, layers: list[int], ref_model,
                               reference, backend_mod, *, root: str | None,
                               fp32_forced: bool, fp32_source: str,
                               promoted: list[str]) -> CheckResult:
    """The comparison itself, with the reference model already in whatever
    numeric mode the caller established. Split out so the fp32 promotion can be
    undone in a ``finally`` no matter how the comparison ends."""
    import torch

    from . import lens_store

    residuals = _reference_residuals(record.dModel)
    device = readout.device

    # The norm-fold assertion, before any comparison: if unembed stopped
    # normalizing, the two paths would still agree with each other and both
    # would be reading something the model never computes.
    probe = residuals[:1].to(device=device, dtype=torch.float32)
    unembedded = ref_model.unembed(probe).to(torch.float32)
    unnormed = ref_model._lm_head(
        probe.to(ref_model._lm_head.weight.dtype)).to(torch.float32)
    norm_applied = bool(
        (unembedded - unnormed).abs().max() > REFERENCE_TOLERANCE)
    # The dtype the comparison actually ran against, read off the head AFTER
    # any promotion: this is what makes "promoted nothing" visible.
    head_dtype = str(ref_model._lm_head.weight.dtype)

    watch = list(readout.watchlist)
    # BOTH vocabulary paths, not just the watchlist (external review,
    # 2026-09-05). The readout has two implementations of the same formula —
    # the watchlist slice, which folds the norm gain into its own copy of the
    # token rows, and the full-vocabulary projection through the model's live
    # output head, which cannot fold and must scale the residual instead — and
    # this check used to exercise only the first. The second was dropping the
    # gain entirely, and because a passing referenceAgreement is what a study
    # stands on, the full path was inheriting a validity nobody had measured.
    # It is compared restricted to the watchlist, so both paths are held to the
    # SAME reference numbers on the same tokens. Armed only when top-k is
    # configured, and when it is not the record says so rather than leaving
    # "not armed" and "checked" indistinguishable.
    full_vocab_armed = getattr(readout, "lm_head", None) is not None
    worst = 0.0
    compared = 0
    per_path_compared: dict[str, int] = {"watchlist": 0, "fullVocab": 0}
    per_comparison: list[dict] = []
    for layer in layers:
        j = lens_store.load_layer(record, layer, root=root).to(torch.float32)
        ref_lens = reference.JacobianLens(
            jacobians={layer: j}, n_prompts=record.nPrompts,
            d_model=record.dModel)
        for index, row in enumerate(residuals):
            h = row.to(device=device, dtype=torch.float32)
            ours = readout.watched_scores(h, layer)
            if ours is None:
                continue
            transported = ref_lens.transport(h, layer)
            theirs = ref_model.unembed(transported).to(torch.float32)[watch]
            theirs_cpu = theirs.to(torch.float32).reshape(-1).cpu()
            mine = [("watchlist", ours)]
            if full_vocab_armed:
                mine.append(
                    ("fullVocab", readout.logits(h, layer)[watch]))
            for path, scores in mine:
                ours_cpu = scores.to(torch.float32).reshape(-1).cpu()
                deviation = (ours_cpu - theirs_cpu).abs()
                at = int(deviation.argmax())
                worst = max(worst, float(deviation[at]))
                compared += 1
                per_path_compared[path] += 1
                # The OPERANDS, not just their difference: whether 0.07 is a
                # large deviation depends entirely on whether these are ~10 or
                # ~96. And WHICH path produced them, or a failure cannot say
                # which of the two implementations moved.
                per_comparison.append({
                    "layer": int(layer),
                    "fixtureRow": int(index),
                    "path": path,
                    "tokenID": int(watch[at]) if at < len(watch) else None,
                    "watchIndex": at,
                    "ours": float(ours_cpu[at]),
                    "reference": float(theirs_cpu[at]),
                    "absDeviation": float(deviation[at]),
                })
    recorded = per_comparison
    truncated = len(per_comparison) > REFERENCE_MAX_RECORDED_COMPARISONS
    if truncated:
        # Keep the largest deviations — the rows a tolerance question is
        # about — and put them back in run order so the record still reads as
        # a walk over (layer, fixture row).
        recorded = sorted(
            sorted(per_comparison, key=lambda c: -c["absDeviation"])
            [:REFERENCE_MAX_RECORDED_COMPARISONS],
            key=lambda c: (c["layer"], c["fixtureRow"], c["path"]))

    problems: list[str] = []
    if not norm_applied:
        problems.append(
            "the reference's unembed() no longer applies the model's final "
            "norm — this engine folds the final norm's gain g into the token "
            "rows on the assumption that it does, so both paths would now be "
            "self-consistently wrong")
    if fp32_forced and ref_model._lm_head.weight.dtype != torch.float32:
        # The mode says float32 and the head is not: the promotion skipped it
        # (mixed float widths, or a reference whose head is not where this
        # engine expects). A comparison at the runtime dtype under a float32
        # stamp is the one record this check must never write as a pass.
        problems.append(
            f"the reference's output head is {head_dtype}, not float32, after "
            f"promotion (promoted: {', '.join(promoted) or 'nothing'}) — the "
            "comparison ran at the runtime dtype and cannot be read as the "
            "float32 agreement the mode claims")
    if not compared:
        problems.append(
            "no residual/layer pair was comparable — the watchlist is empty, "
            "so there was nothing to read out")
    elif worst > REFERENCE_TOLERANCE:
        problems.append(
            f"max logit deviation {worst:.4g} exceeds the pinned tolerance "
            f"{REFERENCE_TOLERANCE:g}")
    worst_comparison = max(per_comparison,
                           key=lambda c: c["absDeviation"], default=None)
    paths_read = ("both vocabulary paths" if full_vocab_armed
                  else "the watchlist path only (the full-vocabulary path is "
                       "not armed for this configuration)")
    detail = ("; ".join(problems)
              or (f"agrees with {backend_mod.REFERENCE_PACKAGE} within "
                  f"{worst:.3g} over {compared} fixture(s), {paths_read}"))
    # In the detail as well as the measurements: this is what a reader sees
    # in the CLI summary, and the mode must be legible there in BOTH
    # directions — a diagnostic line that did not say so would read as the
    # default-mode agreement.
    if fp32_forced:
        origin = ("default mode" if fp32_source == "default"
                  else f"{REFERENCE_FP32_ENV} set")
        detail += (f"; reference output head compared in float32 ({origin}; "
                   f"promoted: "
                   f"{', '.join(promoted) or 'nothing, already float32'})")
    else:
        detail += (f"; REFERENCE PATH AT ITS RUNTIME DTYPE ({head_dtype}; "
                   f"{REFERENCE_FP32_ENV} off) — diagnostic mode, not the "
                   "default-mode agreement: a deviation within one ulp of the "
                   "reference logit at that dtype is the reference's rounding, "
                   "not a disagreement")
    return CheckResult(
        name="referenceAgreement", passed=not problems,
        detail=detail,
        measured={"maxAbsLogitDeviation": worst,
                  "tolerance": REFERENCE_TOLERANCE,
                  "comparisons": compared,
                  "finalNormApplied": norm_applied,
                  # Which implementations were actually held to the reference.
                  # A watchlist-only qualification is a legitimate state (the
                  # full path is armed by top-k), but it must be readable as
                  # one: the full path's numbers are then unverified, not
                  # verified-and-agreeing.
                  "fullVocabArmed": full_vocab_armed,
                  "watchlistComparisons": per_path_compared["watchlist"],
                  "fullVocabComparisons": per_path_compared["fullVocab"],
                  # Per-comparison operands (2026-08-24). The summary above is
                  # unchanged; these are what make a deviation's MAGNITUDE
                  # readable instead of only its size.
                  "perComparison": recorded,
                  "perComparisonRecorded": len(recorded),
                  "perComparisonTruncated": truncated,
                  "perComparisonBasis": ("seeded residual fixtures; a "
                                         "comparison is (layer, fixtureRow, "
                                         "path, tokenID) — no prompt or "
                                         "position axis exists in this check. "
                                         "'path' is which of the readout's two "
                                         "vocabulary implementations produced "
                                         "'ours'"),
                  "worstComparison": worst_comparison,
                  # Always stamped, in BOTH modes, so the absence of a flag can
                  # never be read as the default mode by omission — and WHERE
                  # the mode came from, so a default-mode record and an
                  # explicitly requested one stay distinguishable.
                  "referenceFP32Forced": fp32_forced,
                  "referenceFP32ModeSource": fp32_source,
                  "referenceFP32EnvVar": REFERENCE_FP32_ENV,
                  "referenceFP32Promoted": promoted,
                  "referenceHeadDtype": head_dtype,
                  "referencePackage": backend_mod.REFERENCE_PACKAGE,
                  "referenceVersion": backend_mod.reference_version(),
                  "referenceCommit": backend_mod.REFERENCE_COMMIT,
                  # Which gain fold the readout observed on this runtime's
                  # final norm; the reference applies the module itself, so
                  # agreement here is also agreement about the fold.
                  "normGainConvention": getattr(readout, "gain_convention",
                                                None)})


def _block_outputs(model, prompt: str, layers: list[int], *,
                   interventions=None):
    """``({layer: residual at the final position}, final logits, prompt len)``.

    One forward pass with ``output_hidden_states``. HF appends each entry
    BEFORE running the block, so ``hidden_states[l + 1]`` is block ``l``'s
    output — the same place the online recorder observes, and (because the
    forward hook's RETURN value is what propagates) post-intervention when an
    injector is armed. The last entry is deliberately never used: it is
    post-final-norm, and every layer this is called with is a fitted source
    layer, so ``l + 1`` can at most reach ``n_layers - 1``.
    """
    import torch

    tokenizer = model.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.model.parameters()).device
    ids = encoded["input_ids"].to(device)

    def _forward():
        return model.model(input_ids=ids, output_hidden_states=True)

    with torch.no_grad():
        if interventions:
            model.hooked.reset_offsets()
            with model.hooked.session(list(interventions)):
                out = _forward()
        else:
            out = _forward()
    states = out.hidden_states
    residuals = {l: states[l + 1][0, -1].detach().to(torch.float32)
                 for l in layers if l + 1 < len(states)}
    return residuals, out.logits[0, -1].detach().to(torch.float32), int(ids.shape[1])


def _check_stable_readouts(model, readout, layers: list[int], prompts: list[dict],
                           *, tokenizer) -> CheckResult:
    """Committed prompts read twice: finite, and identical both times.

    Repeating the pass is the cheap way to catch a readout that carries state
    it should not — a cached transport, an accumulating buffer, a device
    mismatch that rounds differently on the second visit. Declared
    expectations (``expectTopKPieces``) are checked when a fixture row has
    them; rows without them are recorded as observations, which is the honest
    state until a 27B measurement fills them in.
    """
    import torch

    problems: list[str] = []
    observations: list[dict] = []
    for row in prompts:
        residuals, logits, _ = _block_outputs(model, row["prompt"], layers)
        if not torch.isfinite(logits).all():
            problems.append(f"{row['id']}: model logits are not finite")
        repeat, _, _ = _block_outputs(model, row["prompt"], layers)
        for layer, h in residuals.items():
            first = readout.watched_scores(h, layer)
            if first is None:
                continue
            if not torch.isfinite(first).all():
                problems.append(
                    f"{row['id']} layer {layer}: watchlist readout is not "
                    f"finite")
                continue
            again = readout.watched_scores(repeat[layer], layer)
            drift = float((first - again).abs().max())
            if drift > 0.0:
                problems.append(
                    f"{row['id']} layer {layer}: readout changed by {drift:.3g} "
                    f"between two identical passes — the readout is carrying "
                    f"state it should not")
            entry = {"id": row["id"], "layer": layer,
                     "maxWatchedScore": float(first.max()),
                     "repeatDrift": drift}
            expected = row.get("expectTopKPieces") or []
            if expected and readout.lm_head is not None and tokenizer is not None:
                ids, _values = readout.topk(h, layer, max(len(expected), 10))
                pieces = [tokenizer.decode([int(t)]) for t in ids.cpu()]
                missing = [p for p in expected if p not in pieces]
                entry["topKPieces"] = pieces
                entry["missingExpected"] = missing
                if missing:
                    problems.append(
                        f"{row['id']} layer {layer}: expected piece(s) "
                        f"{missing} absent from the top-k")
            observations.append(entry)
    return CheckResult(
        name="stableReadouts", passed=not problems,
        detail="; ".join(problems)
               or f"{len(observations)} readout(s) finite and repeatable",
        measured={"observations": observations,
                  "promptCount": len(prompts),
                  "declaredExpectations": sum(
                      1 for r in prompts if r.get("expectTopKPieces"))})


def _causal_smoke(record, model, readout, *, layer: int, token_id: int,
                  prompt: str, ladder: tuple[float, ...],
                  root: str | None) -> CheckResult:
    """Inject one derived token direction over a dose ladder; read it back.

    The direction is ``v_l = J_l^T (g ⊙ u_t)`` — the same construction
    :mod:`derive` persists, computed in memory here because the check needs
    the numbers, not an artifact. Dose is in residual-norm units with the
    denominator measured on THIS prompt at THIS layer, so the ladder means
    "fraction of the residual magnitude actually present" rather than a
    fraction of an unrelated calibration.

    What passes: the token's canonical score rises monotonically with dose and
    every logit stays finite. What this deliberately does NOT establish is
    that the direction is useful — raising a token's own score is the cheapest
    thing its direction can do, and the anti-lexical control that separates
    vocabulary from disposition lives in the G0 gate (plan §3.4), not here.
    """
    import torch

    from ..steering.injector import Injection, VectorInjector
    from ..steering.vector_math import norm_unit_scale
    from . import lens_store
    from .derive import read_token_row_and_gain

    try:
        u_t, gain = read_token_row_and_gain(
            getattr(model, "model_id", None) or record.fit.modelID, token_id,
            getattr(model, "revision", None))
    except JLensError as exc:
        return CheckResult(name="causalSmoke", passed=False, skipped=True,
                           detail=f"could not derive a token direction: {exc}",
                           measured={"tokenID": token_id, "layer": layer})
    j = lens_store.load_layer(record, layer, root=root).to(torch.float32)
    direction = (j.T @ (gain * u_t)).to(torch.float32)
    vector_norm = float(direction.norm())
    if vector_norm <= 0:
        return CheckResult(
            name="causalSmoke", passed=False,
            detail=f"the derived direction for token {token_id} at layer "
                   f"{layer} is all zeros",
            measured={"tokenID": token_id, "layer": layer})

    baseline_residuals, baseline_logits, _ = _block_outputs(model, prompt, [layer])
    if layer not in baseline_residuals:
        return CheckResult(
            name="causalSmoke", passed=False,
            detail=f"no block output at layer {layer} for the carrier prompt",
            measured={"tokenID": token_id, "layer": layer})
    residual_norm = float(baseline_residuals[layer].norm())
    watch_index = readout.watchlist.index(token_id) \
        if token_id in readout.watchlist else None

    def _score(residual) -> float:
        if watch_index is None:
            return float("nan")
        scores = readout.watched_scores(residual, layer)
        return float(scores[watch_index])

    rows = [{"alpha": 0.0, "score": _score(baseline_residuals[layer]),
             "finiteLogits": bool(torch.isfinite(baseline_logits).all()),
             "residualNorm": residual_norm}]
    vector = [float(x) for x in direction]
    for alpha in ladder:
        scale = norm_unit_scale(alpha, residual_norm, vector_norm)
        injector = VectorInjector(
            {layer: Injection(vector=vector, alpha=scale)},
            prompt_token_count=None)
        residuals, logits, _ = _block_outputs(model, prompt, [layer],
                                              interventions=[injector])
        rows.append({
            "alpha": float(alpha),
            "score": _score(residuals[layer]),
            "finiteLogits": bool(torch.isfinite(logits).all()),
            "residualNorm": float(residuals[layer].norm()),
            "rawScale": scale,
        })

    problems: list[str] = []
    if watch_index is None:
        problems.append(
            f"token {token_id} is not in the readout watchlist, so its "
            f"canonical score was never computed — the dose response has no "
            f"quantity to be about")
    non_finite = [r["alpha"] for r in rows if not r["finiteLogits"]]
    if non_finite:
        problems.append(f"non-finite logits at dose(s) {non_finite}")
    scores = [r["score"] for r in rows]
    if watch_index is not None:
        monotone = all(b > a for a, b in zip(scores, scores[1:]))
        if not monotone:
            problems.append(
                f"the token's canonical score is not monotone in dose "
                f"({[round(s, 3) for s in scores]}) — either the direction "
                f"does not do what its construction says, or the injection "
                f"never reached the position the readout observes")
    else:
        monotone = False

    return CheckResult(
        name="causalSmoke", passed=not problems,
        detail="; ".join(problems)
               or (f"token {token_id} at layer {layer}: monotone over "
                   f"{len(ladder)} dose(s), all logits finite"),
        measured={"tokenID": token_id, "layer": layer,
                  "ladder": list(ladder), "monotone": monotone,
                  "vectorNorm": vector_norm,
                  "doseUnits": "residual-norm units measured on the carrier "
                               "prompt at this layer",
                  "rows": rows})


def _check_capability(model, *, model_id: str, layer: int, alpha: float,
                      vector: list[float], residual_norm: float,
                      vector_norm: float, battery_path: str | None,
                      root: str | None) -> CheckResult:
    """The existing capability battery at the top of the intended dose range.

    Reuses the battery machinery rather than inventing a second capability
    check (plan §3.3, and the standing rule that one measurement has one
    implementation). Scored through the answer-token instrument where the
    battery declares it, so the number cannot move with response length or
    format compliance.
    """
    from ..experiment import battery as battery_mod
    from ..experiment.tasks import _battery_backends
    from ..steering.vector_math import norm_unit_scale
    from ..experiment.generate import CellInjection

    path = battery_path or battery_mod.DEFAULT_BATTERY_FILE
    try:
        spec = battery_mod.load_spec(path, root)
    except (OSError, ValueError) as exc:
        return CheckResult(
            name="capabilityGuard", passed=False, skipped=True,
            detail=f"no usable capability battery at '{path}': {exc}",
            measured={"battery": path})
    arming = battery_mod.resolve_arming(spec, prompt_mode=None,
                                        system_prompt=None)
    advisory = battery_mod.contamination_advisory(spec, arming)

    def _accuracy(injections) -> float:
        generate_fn, choice_fn = _battery_backends(model, model_id, injections)
        correct = 0
        for item in spec.items:
            fields = battery_mod.score_item(spec, item, arming,
                                            generate_fn=generate_fn,
                                            choice_fn=choice_fn)
            correct += 1 if fields["correct"] else 0
        return correct / len(spec.items) if spec.items else 0.0

    scale = norm_unit_scale(alpha, residual_norm, vector_norm)
    base = _accuracy([])
    steered = _accuracy([CellInjection(layer=layer, vector=vector,
                                       alpha=scale)])
    drop = base - steered
    passed = drop <= CAPABILITY_TOLERANCE
    detail = (f"battery {base:.2f} → {steered:.2f} at α {alpha:g} "
              f"(drop {drop:+.2f}, tolerance {CAPABILITY_TOLERANCE:g})")
    if advisory:
        detail += f"; {advisory}"
    return CheckResult(
        name="capabilityGuard", passed=passed, detail=detail,
        measured={"battery": path, "batteryHash": spec.digest,
                  "batteryFormat": spec.format_version,
                  "items": len(spec.items),
                  "baselineAccuracy": base, "steeredAccuracy": steered,
                  "drop": drop, "tolerance": CAPABILITY_TOLERANCE,
                  "alpha": float(alpha), "layer": layer,
                  "contaminationAdvisory": advisory})


# ---------------------------------------------------------------------------
# The verb
# ---------------------------------------------------------------------------

def qualify(lens_id: str, model_id: str, *, revision: str | None = None,
            layers: list[int] | None = None,
            alpha_range: tuple[float, ...] | None = None,
            token_id: int | None = None,
            watchlist: list[int] | None = None,
            prompts_path: str | None = None,
            battery_path: str | None = None,
            root: str | None = None, model=None, device: str | None = None,
            dtype: str | None = None,
            log=None) -> dict:
    """Run every check and append the resulting :class:`Qualification`.

    ``model`` is an already-loaded :class:`SteeredModel`; when absent one is
    loaded here. Everything else is a declared choice recorded in the
    artifact: which layers were exercised, which dose ladder, which token, and
    which battery.

    Returns the record summary. Raises :class:`QualificationRefused` only when
    the attempt could not be made at all — a lens that fails its checks
    produces a stored, failed qualification, because "we tested this and it
    did not pass" is evidence and its absence is not.
    """
    from ..steering import model_loader
    from . import lens_store
    from .readout import LensReadout, ReadoutConfig

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    record = lens_store.resolve(lens_id, root)
    if record.fit.modelID and record.fit.modelID != model_id:
        raise QualificationRefused(
            f"lens '{lens_id}' was fitted on '{record.fit.modelID}' but "
            f"qualification was asked for '{model_id}' — a lens is qualified "
            f"against the model it reads, and this pairing is nonsense rather "
            f"than merely unqualified")

    if model is None:
        emit(f"loading {model_id} ({revision or 'cached revision'}) …")
        model = model_loader.load(model_id, revision, dtype=dtype,
                                  device=device)
    revision = revision or getattr(model, "revision", None)
    if not revision:
        raise QualificationRefused(
            "no model revision could be resolved — a qualification is keyed "
            "by revision, and an unkeyed one would be inherited by whatever "
            "the cache happens to hold later")

    runtime_dtype, quantization = resolve_runtime(model)
    tier = tier_for(model_id, record)
    emit(f"runtime resolved: {runtime_dtype}"
         + (f"/{quantization}" if quantization else ", unquantized")
         + f"; tier {tier}")

    armed = sorted(set(layers)) if layers else _default_layers(record)
    unknown = [l for l in armed if l not in record.sourceLayers]
    if unknown:
        raise QualificationRefused(
            f"layers {unknown} are not fitted source layers of '{lens_id}' "
            f"(have {record.sourceLayers[0]}..{record.sourceLayers[-1]}) — "
            f"qualifying a layer the lens never fitted would certify nothing")

    prompts = load_fixture_prompts(prompts_path)
    carrier = next((r for r in prompts if r.get("causalSmoke")), prompts[-1])
    tokenizer = getattr(model, "tokenizer", None)
    token = token_id if token_id is not None else _default_token(tokenizer)
    watch = sorted(set(list(watchlist or []) + [token]))

    readout = LensReadout.build(
        record=record,
        config=ReadoutConfig(layers=armed, watchlist=watch, topK=10,
                             topKLayers=armed, logitLensCompanion=True),
        model=model, root=root)

    started = time.monotonic()
    results: list[CheckResult] = [
        _check_geometry(record, model, model_id=model_id, revision=revision),
        _check_runtime_numerics(record, dtype=runtime_dtype,
                                quantization=quantization),
        _check_jacobian_finite(record, root=root),
        _check_reference_agreement(record, model, readout, armed, root=root),
        _check_stable_readouts(model, readout, armed, prompts,
                               tokenizer=tokenizer),
    ]
    for result in results:
        emit(f"  {result.name}: {'ok' if result.passed else 'FAILED'} — "
             f"{result.detail}")

    ladder = tuple(alpha_range) if alpha_range else DEFAULT_ALPHA_LADDER
    smoke = _causal_smoke(record, model, readout, layer=armed[len(armed) // 2],
                          token_id=token, prompt=carrier["prompt"],
                          ladder=ladder, root=root)
    results.append(smoke)
    emit(f"  {smoke.name}: {'ok' if smoke.passed else 'FAILED'} — "
         f"{smoke.detail}")

    guard = _capability_from_smoke(model, model_id=model_id, smoke=smoke,
                                   record=record, ladder=ladder,
                                   battery_path=battery_path, root=root)
    results.append(guard)
    emit(f"  {guard.name}: {'ok' if guard.passed else 'FAILED'} — "
         f"{guard.detail}")

    checks = {r.name: r.to_dict() for r in results}
    blocking_failures = sorted(r.name for r in results
                               if r.name in BLOCKING and not r.passed)
    passed = not blocking_failures
    qualified_at = _iso()
    qualification = Qualification(
        qualificationID=qualification_id(
            lens_id=lens_id, lens_sha=record.source.tensorSHA256,
            model_id=model_id, revision=revision, dtype=runtime_dtype,
            quantization=quantization, qualified_at=qualified_at),
        modelID=model_id, revision=revision, dtype=runtime_dtype,
        quantization=quantization, tier=tier, passed=passed,
        # Bound to the bytes and the layers this acceptance actually saw, so a
        # re-import or a differently-armed study cannot inherit it.
        lensSHA256=record.source.tensorSHA256,
        convertedSHA256=(record.converted.sha256 if record.converted else None),
        layers=list(armed),
        checks={
            "schemaVersion": 1,
            "checkNames": list(CHECKS),
            "blockingFailures": blocking_failures,
            "armedLayers": armed,
            "watchlist": watch,
            "alphaLadder": list(ladder),
            "promptFixtures": os.path.basename(prompts_path or FIXTURE_PROMPTS),
            "lensSHA256": record.source.tensorSHA256,
            "substrate": record.substrate,
            "engineVersion": _engine_version(),
            "elapsedSeconds": round(time.monotonic() - started, 3),
            "results": checks,
        },
        qualifiedAt=qualified_at)
    # Persisted BEFORE the summary is printed: a crash between the checks and
    # the log would otherwise throw away a GPU slot's worth of measurement.
    path = append_qualification(record, qualification, root=root)

    verdict = "PASSED" if passed else f"FAILED ({', '.join(blocking_failures)})"
    emit(f"qualification {qualification.qualificationID} {verdict} "
         f"[tier {tier}] → {path}")
    if passed and tier != "evidence":
        emit(f"NOTE: '{model_id}' is a {tier}-tier model for J-lens work. "
             f"This record rehearses the mechanics and is refused by freeze — "
             f"it can never satisfy a study's readout gate.")
    return {
        "lensID": lens_id,
        "qualificationID": qualification.qualificationID,
        "passed": passed,
        "tier": tier,
        "modelID": model_id,
        "revision": revision,
        "dtype": runtime_dtype,
        "quantization": quantization,
        "blockingFailures": blocking_failures,
        "armedLayers": armed,
        "record": path,
        "checks": checks,
    }


def _capability_from_smoke(model, *, model_id: str, smoke: CheckResult,
                           record, ladder, battery_path, root) -> CheckResult:
    """The capability guard, armed with the direction the smoke test derived.

    Split out so the guard is scored on exactly the direction, layer, and top
    dose that were just shown to move the readout — a guard run on some other
    arming would answer a question nobody asked.
    """
    import torch

    from . import lens_store
    from .derive import read_token_row_and_gain

    layer = smoke.measured.get("layer")
    token = smoke.measured.get("tokenID")
    rows = smoke.measured.get("rows") or []
    if layer is None or token is None or not rows:
        return CheckResult(
            name="capabilityGuard", passed=False, skipped=True,
            detail="the causal smoke test produced no arming to guard "
                   "(it was refused before a direction existed)",
            measured={})
    residual_norm = float(rows[0].get("residualNorm") or 0.0)
    try:
        u_t, gain = read_token_row_and_gain(
            model_id, int(token), getattr(model, "revision", None))
        j = lens_store.load_layer(record, int(layer), root=root).to(torch.float32)
    except JLensError as exc:
        return CheckResult(name="capabilityGuard", passed=False, skipped=True,
                           detail=f"could not rebuild the direction: {exc}",
                           measured={"layer": layer, "tokenID": token})
    direction = (j.T @ (gain * u_t)).to(torch.float32)
    vector_norm = float(direction.norm())
    if vector_norm <= 0 or residual_norm <= 0:
        return CheckResult(
            name="capabilityGuard", passed=False, skipped=True,
            detail="no usable dose denominator (zero residual or vector norm)",
            measured={"layer": layer, "tokenID": token,
                      "residualNorm": residual_norm,
                      "vectorNorm": vector_norm})
    return _check_capability(
        model, model_id=model_id, layer=int(layer), alpha=float(max(ladder)),
        vector=[float(x) for x in direction], residual_norm=residual_norm,
        vector_norm=vector_norm, battery_path=battery_path, root=root)


#: Where the lens is expected to READ, as a fraction of model depth.
#:
#: From the paper's three functional regions — sensory (~0–38%), **workspace
#: (~38–92%)**, motor (~92–100%). This is the readout band; it is NOT the
#: "middle third" heuristic in CLAUDE.md, which is about where CAA steering
#: vectors tend to work. Conflating the two is how this default previously
#: came to arm a pre-workspace layer (L20 of 62 = 32% depth).
#:
#: A PRIOR, not a measurement: the regions were identified on other models,
#: and G0's readout arm exists to confirm them at 27B. Corroborated locally
#: on gemma-3-4b-it (2026-08-15): layer 4 (12%) read as noise, layer 12 (35%)
#: already carried abstract content, layer 20 (59%) concrete semantics, and
#: layer 28 (82%) had turned syntactic.
WORKSPACE_BAND = (0.38, 0.92)


def _default_layers(record, count: int = 3) -> list[int]:
    """``count`` fitted source layers spaced inside the workspace band.

    A default rather than a rule: qualification is about the layers a study
    intends to arm, and ``--layers`` is how a study says so. This is where to
    look when nobody has said.

    Depth comes from the record's own target layer (``n_layers - 1`` by the
    lens's construction), so no runtime is needed to compute it. Interior
    points only — the band edges are where the regions blur, and a default
    should not sit on a boundary.
    """
    fitted = list(record.sourceLayers)
    if len(fitted) <= count:
        return fitted
    depth = record.targetLayer + 1
    low, high = WORKSPACE_BAND
    picked: set[int] = set()
    for i in range(count):
        fraction = low + (high - low) * (i + 1) / (count + 1)
        # Truncating, matching resolve_sweep_layers' rule, so a depth fraction
        # names the same block everywhere in the engine.
        target = int(depth * fraction)
        # Snap to the nearest FITTED layer: the band is about the model, the
        # available layers are about the lens, and only their intersection is
        # armable.
        picked.add(min(fitted, key=lambda l: (abs(l - target), l)))
    return sorted(picked)


def _default_token(tokenizer) -> int:
    """A vocabulary token for the causal smoke test when none was named.

    Deliberately a common, meaning-poor piece: the smoke test asks whether
    injection moves the readout at all, and choosing a loaded concept word by
    default would invite reading a mechanical check as a semantic result.
    """
    if tokenizer is None:
        raise QualificationRefused(
            "no tokenizer available to resolve a default smoke-test token — "
            "pass --token-id")
    ids = tokenizer.encode(" the", add_special_tokens=False)
    if not ids:
        raise QualificationRefused(
            "could not resolve a default smoke-test token — pass --token-id")
    return int(ids[-1])


def _engine_version() -> str:
    from ..build_identity import engine_version

    return engine_version()


def summarize(record) -> list[dict]:
    """Compact rows for ``jlens list``/``inspect`` and the API."""
    return [{"qualificationID": q.qualificationID, "modelID": q.modelID,
             "revision": q.revision, "dtype": q.dtype,
             "quantization": q.quantization, "tier": q.tier,
             "passed": q.passed, "qualifiedAt": q.qualifiedAt,
             "blockingFailures": (q.checks or {}).get("blockingFailures", [])}
            for q in record.qualifications]

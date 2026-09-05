"""The G0 1b feasibility gate: seven checks, two arms, verdicted separately.

Plan §3.4. The gate exists because "the lens works" is two different claims
that can come apart, and the original AND-together wording would have let a
lens whose directions steer cleanly be failed by a readout that looked murky
at 27B — taking down the arm that worked.

* **The STEERING arm** — derive a direction, inject it, use it in a study —
  is licensed by the causal dose response *plus its anti-lexical control*, the
  resource figures, and alignment.
* **The READOUT arm** — top-k, ranks, traces, any claim about what the model is
  poised to verbalize — is licensed by reference agreement plus the committed
  fixtures. Read its scope carefully: it establishes IMPLEMENTATION FIDELITY
  (this engine computes what the reference computes), not that the readout
  anticipates anything. The fixtures carry no declared expectations, so a
  preregistered scientific criterion does not yet exist; see
  :data:`READOUT_LICENCE`.

Passing one and failing the other is a real outcome and is reportable as one.
What neither arm licenses alone is the conjunction: "the direction steers AND
the readout says why."

The anti-lexical control is not optional
----------------------------------------
A direction for ``" courage"`` is literally ``J^T`` applied to that token's
unembedding row, so the cheapest thing it can do is make the model *say*
courage-words more often without moving the study's outcome at all. A
monotonic dose response is entirely consistent with that. The steering arm's
endpoint is therefore ``logprobShift`` on a declared outcome endpoint — never
marker density
— and the discriminating observation is **endpoint moves while the token's own
frequency does not**. Both moving together fails the arm no matter how clean
the dose curve looks.

Mechanical vs scientific, enforced by schema
--------------------------------------------
The 4B rehearsal gates MECHANICS ONLY. A mechanical failure at 4B is
scale-independent and blocks the 27B run; a scientific result at 4B is a prior
and never blocks, because the paper's own finding is that early-layer readouts
are noise and some workspace-band cells resist interpretation even at Claude
scale. Letting a small model veto the only measurement that counts would be
exactly backwards. This lives in the report's shape — separate ``mechanical``
and ``scientific`` sections with independent verdicts, and arm verdicts that
are ``null`` at testing tier — rather than in prose someone has to remember.

Server-only (hard requirement); any model with a published lens.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from .qualification import CheckResult, load_fixture_prompts, tier_for
from .schemas import JLensError
from . import norm_convention

RUN_TYPE = "jlens-g0"
REPORT_FILENAME = "jlens-g0-report.json"
SCHEMA_VERSION = 1

#: Scale-independent. A failure here is a defect that would waste cluster
#: time, so a 4B failure blocks the 27B run (plan §3.4).
MECHANICAL_CHECKS = (
    "acquisition",         # (1) exact published bytes, hashed
    "loadPath",            # (2, path half) converted artifact + runtime load
    "resolution",          # (3) layers, target semantics, norm gain, softcap
    "alignment",           # (6) prediction alignment + recorder non-interference
    "tracePersistence",    # (6) the durable trace round-trips
    "slicePositioning",    # (6, offline twin) slice positions mean what they say
    "preflightMechanics",  # (7, mechanical half) estimates exist; over-budget refuses
)

#: Scale-bound. A 4B result here is recorded as a PRIOR and never blocks.
SCIENTIFIC_CHECKS = (
    "referenceReadouts",   # (4) reference agreement + unspoken intermediate
    "causalDoseResponse",  # (5) monotone dose response on an outcome endpoint
    "antiLexicalControl",  # (5) endpoint moves, token frequency does not
    "memoryBudget",        # (2, resource half)
    "costMeasurement",     # (7) latency, memory, trace volume, per-projection cost
)

ARMS = ("steering", "readout")

#: What licenses READING, independent of how the reading is obtained.
#:
#: Restructured 2026-08-15. The readout arm used to be
#: ``(alignment, referenceReadouts, costMeasurement)`` — one verdict over three
#: checks that are about DIFFERENT OBJECTS. Only ``referenceReadouts`` is about
#: reading; ``alignment`` and trace volume are properties of one particular
#: machine, the online recorder. So passing the arm licensed claims made from
#: the online trace and said nothing about claims made any other way, while
#: looking like a blanket licence for "the readout".
#:
#: That mattered once the OFFLINE slice stopped being hypothetical: it is the
#: reference package's own instrument, it is what a position-resolved study
#: needs, and it is the ONLY way to read prefill positions (the recorder
#: discards every prompt position but the last, by design). It uses none of
#: the alignment machinery.
#:
#: Both paths compute the identical ``softcap(U · RMSNorm(J_l h))``, so the
#: science is shared and lives here; the machinery each path needs lives in
#: :data:`INSTRUMENT_CHECKS`.
#: **What this verdict does and does not say.** ``referenceReadouts`` passes
#: when this engine's readout agrees numerically with the pinned reference and
#: the unspoken-intermediate fixtures produce finite, recorded observations. It
#: does NOT establish that the readout anticipates anything: the fixtures carry
#: no declared expectations (inventing them would fabricate an acceptance
#: criterion), so the observations are RECORDED and never evaluated.
#:
#: So the arm licenses IMPLEMENTATION FIDELITY, not scientific informativeness
#: (external review 2026-08-16). The honest upgrade path is the
#: ``readableBand`` measurement: once a 27B sweep shows where the lens resolves,
#: a preregistered criterion over it can become a check and this constant can
#: grow. Until then the verdict says what it means, in the report.
READOUT_SCIENCE = ("referenceReadouts",)

#: Stamped beside the readout arm's verdict so "licensed" is never read as a
#: broader claim than it is.
READOUT_LICENCE = (
    "implementation fidelity: this engine's readout agrees with the pinned "
    "reference implementation, and the committed fixtures read finite and "
    "repeatable. NOT a claim that the readout is scientifically informative — "
    "no preregistered criterion for that is declared, and the "
    "unspoken-intermediate observations are recorded rather than evaluated. "
    "See measurements.readableBand for where the lens actually resolves.")

#: Per-instrument plumbing. Neither instrument inherits the other's licence:
#: a slice-based claim is not licensed by an alignment check that never looked
#: at slices, and an alignment failure does not invalidate a claim that never
#: used the aligner.
#:
#: ``costMeasurement`` appears under both because it is itself split — the
#: per-projection cost is shared (both paths run full-vocabulary projections),
#: while the trace-volume figure is online-only. The check reports each
#: separately.
INSTRUMENT_CHECKS = {
    "onlineTrace": ("alignment", "tracePersistence", "costMeasurement"),
    "offlineSlice": ("slicePositioning", "costMeasurement"),
}

#: Which checks license which arm. The steering arm is unchanged. Alignment
#: remains a steering precondition: a steering run whose sampled tokens
#: changed under a read-only recorder would invalidate the readout paired with
#: it.
ARM_CHECKS = {
    "steering": ("alignment", "causalDoseResponse", "antiLexicalControl",
                 "memoryBudget", "costMeasurement"),
    "readout": READOUT_SCIENCE,
}

#: Dose ladder for the steering arm, in residual-norm units.
DEFAULT_ALPHA_LADDER = (0.04, 0.08, 0.12)

#: Every Nth fitted layer in the readable-band sweep. Four keeps a 62-layer
#: model to ~16 layers, which is enough to see where the band starts and ends
#: without paying for a full-vocabulary projection at every layer and position.
#: ``0`` disables the sweep.
DEFAULT_BAND_STRIDE = 4

#: Max new tokens for the frequency half of the anti-lexical control. Long
#: enough that a vocabulary effect has room to show; short enough that the
#: ladder is a handful of generations.
FREQUENCY_MAX_TOKENS = 192

#: A |Δ logP| below this is not treated as endpoint movement. Declared rather
#: than inferred: the control's whole logic is "endpoint moved, frequency did
#: not", and both halves need a stated threshold or the verdict is a vibe.
ENDPOINT_MOVEMENT_NATS = 0.10

#: Relative change in the watched token's own frequency, above which the
#: movement is attributed to vocabulary rather than disposition.
FREQUENCY_MOVEMENT_FRACTION = 0.25

#: Fraction of final-norm gain entries that may be non-positive before the
#: resolution check calls the gain degenerate.
#:
#: This number replaces a sign gate that could never pass. Gemma-3's final
#: RMSNorm gammas legitimately go negative at every published size — measured
#: on the runtime weights: **4B** min −0.0546875 with 2/2560 non-positive dims,
#: **12B** min −0.117 with 8/3840, **27B** min −0.25 — so
#: ``gain_min <= 0`` indicted Google's weights, not the lens or this engine
#: (investigation 2026-08-24, old §13.2 DISSOLVED). A handful of negative dims
#: is the model; a gain that is mostly or entirely non-positive is a wrong
#: module or a dead read, which is what this check is actually for. The
#: measured populations sit at 0.08% (4B) and 0.21% (12B), two orders below
#: this limit, so it separates the real cases without pretending to a
#: precision nobody has evidence for.
GAIN_NONPOSITIVE_FRACTION = 0.01


class G0Error(JLensError):
    """The gate could not be run. Distinct from failing it."""


@dataclass
class Endpoint:
    """A declared OUTCOME endpoint for the steering arm.

    Choice rows scored through the answer-token logprob instrument — the
    deterministic, temperature-free instrument a study uses for categorical
    outcomes. Marker density is deliberately not an option here.

    **The file is the study's own ``choicePromptsFile``, parsed by the study's
    own loader** (:func:`sweep_selection.load_choice_rows`), not by a private
    reimplementation. That matters twice over. Practically, the gate can be
    pointed at the choice rows a study already has, instead of a
    near-identical parallel file someone has to keep in sync. Structurally, one
    measurement has one implementation: a second parser would drift — it would
    have missed ``text`` as an alias for ``prompt``, and the ``target``
    defaulting to ``options[0]`` that the study runner applies — so the gate
    would have measured a different endpoint than the study it licenses.

    The file's SHA-256 travels with the rows, because the endpoint is a
    measurement-side input and the report has to be able to say which bytes
    were scored.
    """

    rows: tuple = ()
    path: str = ""
    hash: str = ""

    @classmethod
    def load(cls, path: str, root: str | None = None) -> "Endpoint":
        from ..experiment import paths
        from ..experiment.sweep_selection import load_choice_rows

        resolved = paths.resolve(path, root)
        try:
            rows, digest = load_choice_rows(resolved, path)
        except (OSError, ValueError) as exc:
            # Remapped to the gate's typed error, message intact: the study
            # loader's refusals already name the row and the reason.
            raise G0Error(str(exc)) from exc
        return cls(rows=rows, path=path, hash=digest)


# ---------------------------------------------------------------------------
# Mechanical checks
# ---------------------------------------------------------------------------

def _check_acquisition(record, *, root: str | None) -> CheckResult:
    """The exact published bytes, re-hashed rather than trusted.

    The import already hashed them; this re-reads the CONVERTED artifact and
    compares it with the hash the record stamped, which is the leg that would
    silently rot if the derived cache were rebuilt from different upstream
    bytes.
    """
    from ..experiment import paths
    from .schemas import sha256_file

    problems: list[str] = []
    measured = {
        "repo": record.source.repo,
        "folder": record.source.folder,
        "tensorFile": record.source.tensorFile,
        "commit": record.source.commit,
        "tensorSHA256": record.source.tensorSHA256,
        "configSHA256": record.source.configSHA256,
    }
    if not record.source.tensorSHA256:
        problems.append(
            "the record carries no upstream tensor hash — it was imported "
            "before hashing existed, or from a snapshot that had no tensor")
    if record.converted is None:
        problems.append("the lens has no converted per-layer artifact")
    else:
        path = paths.resolve(record.converted.path, root)
        measured["convertedPath"] = record.converted.path
        measured["convertedDtype"] = record.converted.dtype
        measured["convertedLayerCount"] = record.converted.layerCount
        if not os.path.exists(path):
            problems.append(f"converted artifact missing at '{path}'")
        elif record.converted.sha256:
            live = sha256_file(path)
            measured["convertedSHA256"] = live
            if live != record.converted.sha256:
                problems.append(
                    f"converted artifact hash drifted (have {live[:12]}…, "
                    f"recorded {record.converted.sha256[:12]}…) — the derived "
                    f"cache no longer matches the import that produced it")
    return CheckResult(name="acquisition", passed=not problems,
                       detail="; ".join(problems)
                              or "published bytes and derived cache both hash "
                                 "as recorded",
                       measured=measured)


def _check_load_path(record, model, *, root: str | None) -> CheckResult:
    """One layer at a time out of the converted artifact, against a live model.

    The access pattern IS the feature: at 27B the whole lens is ~6.6 GiB once
    promoted and a single layer is ~58 MB, so a check that loaded everything
    would pass while proving the opposite of what it claims.
    """
    from . import lens_store

    problems: list[str] = []
    loaded = 0
    widest = 0
    element_size = 0
    for layer in record.sourceLayers:
        try:
            j = lens_store.load_layer(record, layer, root=root)
        except JLensError as exc:
            problems.append(f"layer {layer}: {exc}")
            break
        loaded += 1
        element_size = int(j.element_size())
        widest = max(widest, int(j.numel() * element_size))
        del j
    hidden = int(getattr(model, "hidden_size", 0) or 0)
    if hidden and hidden != record.dModel:
        problems.append(
            f"runtime hidden size {hidden} != lens d_model {record.dModel}")
    return CheckResult(
        name="loadPath", passed=not problems,
        detail="; ".join(problems)
               or f"{loaded} layer(s) loaded individually "
                  f"({widest / (1 << 20):.1f} MiB each)",
        measured={"layersLoaded": loaded, "perLayerBytes": widest,
                  "storedElementBytes": element_size,
                  # What the reference loader would hold instead: every layer,
                  # promoted to float32. The number this access pattern exists
                  # to avoid, stated so the report can show the gap.
                  "wholeLensBytesIfPromoted":
                      (widest // element_size * 4 * loaded)
                      if element_size else None,
                  "runtimeHiddenSize": hidden})


def _check_resolution(record, model, readout) -> CheckResult:
    """Layer mapping, target-layer semantics, RMSNorm gain, and logit softcap
    resolved from the runtime — never assumed from the family name.

    The gain fold is the one that bites: measured on gemma-3-4b-it the BODY of
    the ``g = 1 + w`` distribution sits around 7.6–10.5 (median 7.19), so
    omitting the fold would rescale every direction by about an order of
    magnitude, unevenly per element. That body is not the range: the full 4B
    span is **−1.055 … 53.5**, and an earlier docstring quoted the body
    ("~6.6–9.5") as if it were the range, which is what made a sign gate look
    reasonable.

    **What the gain is, and why there is no sign check.** ``gain`` is not a
    lens-artifact field — the published lens carries ``J``/``d_model``/
    ``n_prompts``/``source_layers`` and nothing else. It is read live from the
    RUNTIME model's final RMSNorm (``1 + norm.weight`` or ``norm.weight``,
    whichever fold the module is observed to apply), and Gemma-3's gammas
    go negative at every size (4B min −0.0546875, 2/2560 non-positive dims;
    12B −0.117, 8/3840; 27B −0.25). A ``gain_min <= 0`` failure therefore could
    not pass on ANY supported model, and what it indicted was Google's weights
    (investigation 2026-08-24; old §13.2, DISSOLVED).

    What that gate was reaching for survives as two convention checks, which
    catch the defects a sign test was standing in for:

    * **Width** — ``len(gain) == record.dModel``. A gain read off the wrong
      norm module (a per-layer norm, a vision-tower norm) is the realistic
      bug, and it shows up here as a width, not a sign.
    * **Non-degeneracy** — all-zero, all-non-positive, or more than
      :data:`GAIN_NONPOSITIVE_FRACTION` of the dimensions non-positive fails.
      An unloaded/zeroed buffer and a ``w`` that is really ``g`` (so ``1 + w``
      lands near zero or below) both land here; the model's own handful of
      negative dims does not.

    ``gainMin``/``gainMax`` stay in ``measured`` either way — the diagnostic
    value is real even though neither is a verdict.
    """
    problems: list[str] = []
    layers = int(getattr(model, "num_layers", 0) or 0)
    if record.targetLayer != layers - 1:
        problems.append(
            f"lens target layer {record.targetLayer} != n_layers-1 "
            f"({layers - 1}) — transport at the target is the identity only "
            f"when those line up")
    contiguous = record.sourceLayers == list(
        range(record.sourceLayers[0], record.sourceLayers[-1] + 1))
    if not contiguous:
        problems.append("source layers are not contiguous")
    gain = getattr(readout, "gain", None)
    gain_min = gain_max = None
    gain_width = None
    non_positive = None
    non_positive_fraction = None
    if gain is None:
        problems.append("no final-norm gain resolved from the runtime")
    else:
        gain_min, gain_max = float(gain.min()), float(gain.max())
        gain_width = int(gain.numel())
        if gain_width != record.dModel:
            problems.append(
                f"the resolved gain has {gain_width} entries but the lens "
                f"d_model is {record.dModel} — this is read as "
                f"the gain off the runtime's FINAL norm, and a width "
                f"mismatch means a different norm module was resolved")
        non_positive = int((gain <= 0).sum())
        non_positive_fraction = (non_positive / gain_width) if gain_width else None
        if gain_width and gain_min == 0.0 and gain_max == 0.0:
            problems.append(
                "the resolved gain is all zeros — an unloaded or zeroed norm "
                "buffer, not a runtime's weights")
        elif gain_width and non_positive == gain_width:
            problems.append(
                f"every one of the {gain_width} resolved gain entries is "
                f"non-positive (max {gain_max:g}) — the convention here is "
                f"g = {getattr(readout, 'gain_convention', None) or '?'} "
                f"(observed on the module), and a whole-vector sign flip "
                f"means the fold does not match what the norm computes")
        elif (non_positive_fraction is not None
                and non_positive_fraction > GAIN_NONPOSITIVE_FRACTION):
            problems.append(
                f"{non_positive}/{gain_width} "
                f"({non_positive_fraction:.2%}) of the resolved gain entries "
                f"are non-positive, above the "
                f"{GAIN_NONPOSITIVE_FRACTION:.0%} degeneracy limit — Gemma's "
                f"own gammas put a handful of dimensions below zero "
                f"(4B: 2/2560), never a population this size")
    return CheckResult(
        name="resolution", passed=not problems,
        detail="; ".join(problems)
               or (f"layers {record.sourceLayers[0]}..{record.sourceLayers[-1]}"
                   f" → target {record.targetLayer}; gain "
                   f"{gain_min:.2f}..{gain_max:.2f} over {gain_width} dims "
                   f"({non_positive} non-positive); softcap "
                   f"{readout.softcap}"),
        measured={"sourceLayers": [record.sourceLayers[0],
                                   record.sourceLayers[-1]],
                  "contiguous": contiguous,
                  "targetLayer": record.targetLayer,
                  "runtimeLayerCount": layers,
                  "gainMin": gain_min, "gainMax": gain_max,
                  # Diagnostics, not verdicts: negative gammas are the model's
                  # own (see the docstring), so these are recorded for the
                  # reader rather than compared against a sign.
                  "gainWidth": gain_width,
                  "gainNonPositive": non_positive,
                  "gainNonPositiveFraction": non_positive_fraction,
                  "gainNonPositiveFractionLimit": GAIN_NONPOSITIVE_FRACTION,
                  "finalLogitSoftcap": readout.softcap,
                  # Observed on the runtime's own norm module at build time.
                  "gainConvention": getattr(readout, "gain_convention", None),
                  "gainConventionDescription": (
                      norm_convention.describe(readout.gain_convention)
                      if getattr(readout, "gain_convention", None) else None)})


def _check_alignment(model, readout, config, prompt: str) -> CheckResult:
    """Armed and disarmed runs emit IDENTICAL sampled token ids.

    The acceptance test for the whole online-readout stage. A recorder that
    perturbed generation would invalidate every trace it produced *and* every
    steering result measured alongside one, so this is a precondition for both
    arms rather than a readout nicety.

    Also checks the alignment arithmetic itself: every observation must name a
    token that was actually sampled, at the index the convention says.
    """
    from ..experiment.generate import generate

    from .recorder import JLensReadoutRecorder

    disarmed_ids: list = []
    generate(model, prompt, model_id=getattr(model, "model_id", None),
             max_tokens=24, temperature=0.0, token_ids_out=disarmed_ids)

    recorder = JLensReadoutRecorder(readout, config)
    armed_ids: list = []
    generate(model, prompt, model_id=getattr(model, "model_id", None),
             max_tokens=24, temperature=0.0, token_ids_out=armed_ids,
             observers=[recorder])
    recorder.join_token_ids(list(armed_ids))

    problems: list[str] = []
    if not disarmed_ids or not armed_ids:
        problems.append("generation returned no token ids")
    elif list(disarmed_ids) != list(armed_ids):
        first = next((i for i, (a, b) in enumerate(zip(disarmed_ids, armed_ids))
                      if a != b), min(len(disarmed_ids), len(armed_ids)))
        problems.append(
            f"arming the read-only recorder CHANGED the sampled tokens "
            f"(diverges at index {first}) — every trace and every steering "
            f"result measured alongside one would be invalid")
    if not recorder.complete:
        problems.append(
            f"the recorder did not produce a complete trace: "
            f"{recorder.failureReason}")
    return CheckResult(
        name="alignment", passed=not problems,
        detail="; ".join(problems)
               or (f"{len(armed_ids)} sampled token(s) identical armed and "
                   f"disarmed; {len(recorder.observations)} observation(s) "
                   f"aligned"),
        measured={"sampledTokens": len(armed_ids),
                  "identicalTokenIDs": list(disarmed_ids) == list(armed_ids),
                  "observations": len(recorder.observations),
                  "traceComplete": recorder.complete,
                  "failureReason": recorder.failureReason,
                  "recorder": recorder.summary(
                      expected_tokens=len(armed_ids))})


def _check_trace_persistence(model, readout, config, run_directory: str,
                             prompt: str) -> CheckResult:
    """A trace written, closed, and read back as a whole readout.

    Reads the summary from the FILE rather than from the writer's own counters:
    a summary that reports on itself cannot detect truncation, which is the
    failure this exists to catch.
    """
    from ..experiment.generate import generate

    from . import trace as trace_mod
    from .recorder import JLensReadoutRecorder

    probe = os.path.join(run_directory, "trace-probe")
    os.makedirs(probe, exist_ok=True)
    recorder = JLensReadoutRecorder(readout, config)
    ids: list = []
    generate(model, prompt, model_id=getattr(model, "model_id", None),
             max_tokens=16, temperature=0.0, token_ids_out=ids,
             observers=[recorder])
    recorder.join_token_ids(list(ids))

    writer = trace_mod.TraceWriter(probe)
    identity = trace_mod.TraceIdentity(
        run=os.path.basename(probe), condition="g0-probe",
        promptID="g0", promptIndex=0, sampleIndex=0,
        modelID=getattr(model, "model_id", None),
        modelRevision=getattr(model, "revision", None),
        dtype=getattr(model, "dtype", None))
    row = trace_mod.trace_record(recorder, identity, prompt_ids=[],
                                 generated_ids=list(ids),
                                 watchlist=config.watchlist)
    writer.write(row, recorder)
    writer.close()
    summary = trace_mod.read_summary(probe)

    problems: list[str] = []
    if summary is None:
        problems.append("no trace file was written")
    else:
        if not summary["complete"]:
            problems.append(
                f"the round-tripped trace reads incomplete "
                f"({summary['incompleteRecords']} record(s))")
        if summary["traceRows"] != 1:
            problems.append(
                f"wrote 1 row, read back {summary['traceRows']}")
    return CheckResult(
        name="tracePersistence", passed=not problems,
        detail="; ".join(problems) or "trace round-trips as a whole readout",
        measured={"summary": summary, "probeDirectory": probe})


def _check_slice_positioning(model, layers, prompts) -> CheckResult:
    """Offline-slice plumbing: does "position p" mean position p?

    The online path earns its ``alignment`` check because it has to work out
    which emitted token each observation preceded. The offline path has no such
    problem — you index positions directly — but it is NOT free of position
    bookkeeping, and the way it goes wrong is quieter.

    The concrete hazard, found while wiring the reference package:
    ``HFLensModel`` sets ``tokenizer.add_bos_token = True`` by default
    (``force_bos``). Left on against a rendering that did not add BOS, every
    position shifts by one and every readout still looks entirely plausible.

    So this measures alignment the way a reader would check it: teacher-force
    the fixture and ask whether the model's own next-token argmax at position
    ``p`` is the token at ``p+1``, and compare that against the same statistic
    under a deliberate ±1 shift. Correct indexing must beat both shifts. It is
    a discriminative test rather than a threshold on natural-language
    predictability, which nobody could justify.
    """
    import torch

    from .qualification import _block_outputs

    tokenizer = model.tokenizer
    rows: list[dict] = []
    problems: list[str] = []
    for row in prompts:
        encoded = tokenizer(row["prompt"], return_tensors="pt")
        ids = [int(t) for t in encoded["input_ids"][0]]
        device = next(model.model.parameters()).device
        with torch.no_grad():
            out = model.model(
                input_ids=encoded["input_ids"].to(device))
        argmax = [int(x) for x in out.logits[0].argmax(dim=-1)]

        def agreement(offset: int) -> float:
            pairs = [(argmax[i], ids[i + offset])
                     for i in range(len(ids))
                     if 0 <= i + offset < len(ids)]
            if not pairs:
                return 0.0
            return sum(a == b for a, b in pairs) / len(pairs)

        aligned, early, late = agreement(1), agreement(0), agreement(2)
        rows.append({"id": row["id"], "tokens": len(ids),
                     "bosPresent": bool(ids) and ids[0] == getattr(
                         tokenizer, "bos_token_id", None),
                     "nextTokenAgreement": aligned,
                     "shiftMinusOne": early, "shiftPlusOne": late})
        if not (aligned > early and aligned > late):
            problems.append(
                f"{row['id']}: position indexing does not beat a ±1 shift "
                f"(aligned {aligned:.2f} vs {early:.2f}/{late:.2f}) — the "
                f"slice's positions are probably off by one")
    return CheckResult(
        name="slicePositioning", passed=not problems,
        detail="; ".join(problems)
               or (f"{len(rows)} fixture(s): next-token agreement beats both "
                   f"±1 shifts at every one"),
        measured={"rows": rows,
                  "note": "HFLensModel's force_bos defaults True and would "
                          "shift every position by one; this engine passes "
                          "force_bos=False"})


def _measure_replay_fidelity(model, readout, config, layers,
                             prompt: str) -> dict:
    """How closely does teacher-forced REPLAY reproduce the ONLINE readout?

    Not a check — a stamped RESOLUTION LIMIT, because there is no tolerance
    here anyone could justify as a threshold. What the numbers license is a
    claim-level constraint that applies to both instruments: if top-1
    reproduces and exact top-k ORDER does not, then a finding resting on
    "rank 7 vs rank 9" is below the runtime's own noise floor.

    Measured on gemma-3-4b-it 2026-08-15: top-1 identical at 24/24 steps at
    both armed layers, top-10 set overlap 97-99%, and the same divergence
    flipped the MODEL's own greedy argmax at 1 of 24 steps — so the floor
    belongs to the runtime (decode computes a position with ``seq_len=1``
    against a KV cache; replay computes it inside one long matmul), not to the
    lens.
    """
    import torch

    from ..experiment.generate import generate

    from .qualification import _block_outputs
    from .recorder import JLensReadoutRecorder

    recorder = JLensReadoutRecorder(readout, config)
    sampled: list = []
    generate(model, prompt, model_id=getattr(model, "model_id", None),
             max_tokens=24, temperature=0.0, token_ids_out=sampled,
             observers=[recorder])
    generated = list(sampled)
    recorder.join_token_ids(generated)
    prompt_ids = list(getattr(recorder, "prompt_ids", []) or [])
    if not (generated and prompt_ids and recorder.prompt_length):
        return {"measured": False,
                "reason": "no online capture to compare against"}

    full = prompt_ids + generated
    device = next(model.model.parameters()).device
    with torch.no_grad():
        out = model.model(
            input_ids=torch.tensor([full], device=device),
            output_hidden_states=True)
    states, P = out.hidden_states, recorder.prompt_length

    per_layer: dict[int, dict] = {}
    argmax_agree = argmax_total = 0
    for index, token in enumerate(generated):
        position = index + P - 1
        if position < len(full):
            argmax_total += 1
            argmax_agree += int(out.logits[0, position].argmax()) == token
    for obs in recorder.observations:
        position = obs.predictedIndex + P - 1
        if position >= len(full):
            continue
        h = states[obs.layer + 1][0, position].detach().to(torch.float32)
        stats = per_layer.setdefault(
            obs.layer, {"n": 0, "worstLogitDelta": 0.0, "top1Same": 0,
                        "topKOverlap": 0.0})
        stats["n"] += 1
        if obs.watched:
            online = torch.tensor(obs.watched, dtype=torch.float32)
            replay = readout.watched_scores(h, obs.layer).to(
                torch.float32).cpu()
            stats["worstLogitDelta"] = max(
                stats["worstLogitDelta"],
                float((online - replay).abs().max()))
        if obs.topKIDs:
            ids, _ = readout.topk(h, obs.layer, len(obs.topKIDs))
            replay_ids = [int(x) for x in ids.cpu()]
            stats["top1Same"] += int(replay_ids[0] == obs.topKIDs[0])
            stats["topKOverlap"] += (
                len(set(replay_ids) & set(obs.topKIDs)) / len(replay_ids))
    for stats in per_layer.values():
        if stats["n"]:
            stats["topKOverlap"] /= stats["n"]
    return {
        "measured": True,
        "generatedTokens": len(generated),
        "modelArgmaxAgreement": (argmax_agree / argmax_total
                                 if argmax_total else None),
        "perLayer": {str(k): v for k, v in sorted(per_layer.items())},
        "convention": "teacher-forced replay of the realized sequence vs the "
                      "online recorder, unsteered",
        "interpretation": "the resolution limit for BOTH instruments: what "
                          "reproduces may be claimed, what does not may not. "
                          "A steered replay is a different question — it must "
                          "re-apply the injection position-exactly.",
    }


def _measure_readable_band(model, record, readout_for, prompts, *,
                           stride: int, root) -> dict:
    """Per-layer readout quality across the stack — the band, from measurement.

    The armed layers are currently chosen from a PRIOR: the paper's three
    functional regions (sensory ~0-38%, workspace ~38-92%, motor ~92-100%),
    corroborated on a 4B slice. That prior is what
    ``qualification.WORKSPACE_BAND`` encodes, and a study has to pin its
    ``layers`` before it can freeze — so the prior is currently load-bearing
    for a decision that is hard to revisit (pinned layers feed ``configHash``;
    changing them after freeze means duplicating the experiment).

    This replaces it with numbers on the model that counts. Two statistics per
    layer, both mirroring the paper's own panels and both cheap because the
    readout is already computed:

    The statistic is the **rank of the true next token**, not top-1 accuracy.
    That is not a stylistic choice: measured on gemma-3-4b-it, top-1 read 0.00
    at EVERY layer including the motor end, because untrained and reserved
    vocabulary entries (``'ꗜ'``, ``'<unused338>'``…) carry large unembedding
    norms and outrank the correct token everywhere — ``' Paris'`` sat at rank 5
    for "the capital city of France is" while the model's own argmax was
    ``' Paris'``. A top-1 band chart would have been flat zero and said
    nothing. See :meth:`LensReadout.token_rank`.

    Reported, never verdicted: which band to arm is a study-design decision,
    and a threshold invented here would be a guess wearing a number.
    """
    import torch

    from .qualification import _block_outputs

    tokenizer = model.tokenizer
    layers = [l for l in record.sourceLayers if l % stride == 0]
    depth = record.targetLayer + 1
    device = next(model.model.parameters()).device
    # Armed once for every layer on the ladder: building a readout per prompt
    # would reload the same Jacobians for every fixture.
    readout = readout_for(layers)
    per_layer: dict[int, dict] = {
        l: {"n": 0, "top1Correct": 0, "ranks": []} for l in layers}
    for row in prompts:
        encoded = tokenizer(row["prompt"], return_tensors="pt")
        ids = [int(t) for t in encoded["input_ids"][0]]
        if len(ids) < 3:
            continue
        with torch.no_grad():
            out = model.model(input_ids=encoded["input_ids"].to(device),
                              output_hidden_states=True)
        for layer in layers:
            states = out.hidden_states[layer + 1][0]
            stats = per_layer[layer]
            # Every position whose successor is known — i.e. all but the last,
            # whose next token the prompt does not contain.
            for position in range(len(ids) - 1):
                h = states[position].detach().to(torch.float32)
                target = ids[position + 1]
                rank = readout.token_rank(h, layer, [target])[0]
                stats["n"] += 1
                stats["ranks"].append(rank)
                stats["top1Correct"] += int(rank == 1)
    out_rows = []
    for layer in layers:
        stats = per_layer[layer]
        if not stats["n"]:
            continue
        ranks = sorted(stats["ranks"])
        out_rows.append({
            "layer": layer,
            "depthFraction": round(layer / depth, 3),
            "positionsScored": stats["n"],
            # The headline: lower is more readable. Median rather than mean —
            # one unpredictable position can sit at rank 100k and would drag a
            # mean across the whole vocabulary.
            "medianNextTokenRank": ranks[len(ranks) // 2],
            "meanReciprocalRank": sum(1.0 / r for r in ranks) / len(ranks),
            "nextTokenTop1": stats["top1Correct"] / stats["n"],
        })
    return {
        "measured": bool(out_rows),
        "stride": stride,
        "depth": depth,
        "priorBand": list(_workspace_band()),
        "perLayer": out_rows,
        "statistic": "full-vocabulary rank of the true next token; lower is "
                     "more readable. top-1 is reported too but is NOT the "
                     "headline — untrained vocabulary entries with large "
                     "unembedding norms outrank correct tokens at every "
                     "layer, so top-1 reads ~0 even where the readout is right",
        "interpretation": "per-layer readability. Reported, "
                          "never verdicted: which band to arm is a study "
                          "decision. Compare against priorBand — the paper's "
                          "workspace region, which this measurement is meant "
                          "to replace for THIS model.",
    }


def _workspace_band():
    from .qualification import WORKSPACE_BAND

    return WORKSPACE_BAND


def _check_preflight_mechanics(config) -> CheckResult:
    """Estimates are produced, and an over-budget configuration REFUSES.

    Both halves matter. A preflight that produced numbers and then let
    everything through would be a decoration, and the compute ceiling is the
    one that actually binds: lowering ``k`` shrinks the stored rows but not the
    full-vocabulary matmul that produced them.
    """
    from .readout import Budget, preflight

    estimate = preflight(config, generations=100, max_new_tokens=500)
    tiny = Budget(maxFullVocabProjections=1, maxObservations=1,
                  maxTraceBytes=1)
    refused = preflight(config, generations=100, max_new_tokens=500,
                        budget=tiny)
    problems: list[str] = []
    for key in ("steps", "observations", "fullVocabProjections",
                "projectedTraceBytes"):
        if key not in estimate:
            problems.append(f"preflight produced no '{key}'")
    if refused["withinBudget"]:
        problems.append(
            "an impossible budget did not refuse — the ceiling is decoration")
    return CheckResult(
        name="preflightMechanics", passed=not problems,
        detail="; ".join(problems)
               or (f"100×500 projects {estimate['observations']} observation(s)"
                   f", {estimate['fullVocabProjections']} full-vocab "
                   f"projection(s); an impossible budget refuses"),
        measured={"estimate": estimate,
                  "refusalProblems": refused["problems"]})


# ---------------------------------------------------------------------------
# Scientific checks
# ---------------------------------------------------------------------------

def _check_reference_readouts(record, model, readout, layers, prompts, *,
                              root: str | None) -> CheckResult:
    """Reference agreement plus the unspoken-intermediate fixture.

    Reuses the qualification module's reference comparison rather than
    restating it — one implementation of "does this engine agree with the
    reference", so the two verdicts can never drift apart. The
    unspoken-intermediate half is reported as an OBSERVATION, with the fixture
    row's own declared expectation checked when it has one: a 4B null here is
    uninformative rather than disqualifying (plan §3.4), and inventing an
    expectation to fail against would be worse than reporting none.
    """
    from .qualification import _check_reference_agreement, _block_outputs

    agreement = _check_reference_agreement(record, model, readout, layers,
                                           root=root)
    observations = []
    for row in prompts:
        if not row.get("carriesUnspokenIntermediate"):
            continue
        residuals, _logits, _ = _block_outputs(model, row["prompt"], layers)
        for layer, h in residuals.items():
            if readout.lm_head is None:
                continue
            ids, values = readout.topk(h, layer, 10)
            companion_ids, _ = readout.topk(h, layer, 10, use_jacobian=False)
            observations.append({
                "id": row["id"], "layer": layer,
                "topKIDs": [int(t) for t in ids.cpu()],
                "topKLogits": [float(v) for v in values.cpu()],
                "companionTopKIDs": [int(t) for t in companion_ids.cpu()],
            })
    return CheckResult(
        name="referenceReadouts", passed=agreement.passed,
        skipped=agreement.skipped,
        detail=agreement.detail
               + f"; {len(observations)} unspoken-intermediate observation(s) "
                 f"recorded",
        measured={"agreement": agreement.measured,
                  "unspokenIntermediate": observations})


def _endpoint_shift(model, endpoint: Endpoint, injections) -> dict:
    """Mean Δ logP of each row's target option — the ``logprobShift`` objective.

    The same instrument a study's categorical outcome endpoints use, for the
    same reason: it is deterministic, temperature-free, and cannot move with
    response length or format compliance.
    """
    from ..experiment import logprob

    out = {}
    for row in endpoint.rows:
        result = logprob.score_options(
            model, row.prompt, list(row.options),
            model_id=getattr(model, "model_id", None), injections=injections)
        # Keyed by the row's own id — the study loader guarantees they are
        # unique, because a duplicate would silently overwrite its twin and
        # corrupt the shift mean.
        out[row.id] = next(
            s.logprob for s in result.options if s.option == row.target)
    return out


def _token_frequency(model, endpoint: Endpoint, injections, piece: str) -> dict:
    """How often the watched token's own piece is emitted, per 1000 tokens."""
    from ..experiment.generate import generate

    emitted = occurrences = 0
    for row in endpoint.rows:
        ids: list = []
        text = generate(model, row.prompt,
                        model_id=getattr(model, "model_id", None),
                        max_tokens=FREQUENCY_MAX_TOKENS, temperature=0.0,
                        injections=injections, token_ids_out=ids)
        emitted += len(ids)
        occurrences += text.lower().count(piece.strip().lower()) if piece.strip() \
            else 0
    return {"tokensEmitted": emitted, "occurrences": occurrences,
            "perThousand": (1000.0 * occurrences / emitted) if emitted else 0.0}


def _steering_arm_science(record, model, readout, *, layer: int, token_id: int,
                          piece: str, endpoint: Endpoint | None,
                          ladder: tuple[float, ...], carrier: str,
                          root: str | None) -> tuple[CheckResult, CheckResult]:
    """``(causalDoseResponse, antiLexicalControl)`` — measured together.

    They share every generation: the endpoint shift and the token-frequency
    count come from the same doses of the same direction, because the whole
    control is a comparison BETWEEN them and measuring them apart would invite
    comparing two different arms.
    """
    import torch

    from ..experiment.generate import CellInjection
    from ..steering.vector_math import norm_unit_scale
    from . import lens_store
    from .derive import read_token_row_and_gain
    from .qualification import _block_outputs

    if endpoint is None:
        reason = ("no outcome endpoint was declared (--endpoint), so the "
                  "steering arm's endpoint could not be measured. Marker "
                  "density is NOT a substitute: it is the surface-prose "
                  "confound this control exists to detect")
        skipped = CheckResult(name="causalDoseResponse", passed=False,
                              skipped=True, detail=reason, measured={})
        return skipped, CheckResult(name="antiLexicalControl", passed=False,
                                    skipped=True, detail=reason, measured={})

    u_t, gain = read_token_row_and_gain(
        getattr(model, "model_id", None) or record.fit.modelID, token_id,
        getattr(model, "revision", None))
    j = lens_store.load_layer(record, layer, root=root).to(torch.float32)
    direction = (j.T @ (gain * u_t)).to(torch.float32)
    vector = [float(x) for x in direction]
    vector_norm = float(direction.norm())
    residuals, _logits, _ = _block_outputs(model, carrier, [layer])
    residual_norm = float(residuals[layer].norm())

    baseline_shift = _endpoint_shift(model, endpoint, [])
    baseline_freq = _token_frequency(model, endpoint, [], piece)
    rows = [{"alpha": 0.0, "meanShift": 0.0,
             "frequencyPerThousand": baseline_freq["perThousand"]}]
    for alpha in ladder:
        scale = norm_unit_scale(alpha, residual_norm, vector_norm)
        injections = [CellInjection(layer=layer, vector=vector, alpha=scale)]
        shift = _endpoint_shift(model, endpoint, injections)
        freq = _token_frequency(model, endpoint, injections, piece)
        rows.append({
            "alpha": float(alpha),
            "meanShift": sum(shift[k] - v for k, v in baseline_shift.items())
                         / len(baseline_shift),
            "frequencyPerThousand": freq["perThousand"],
            "rawScale": scale,
        })

    shifts = [r["meanShift"] for r in rows]
    monotone = (all(b >= a for a, b in zip(shifts, shifts[1:]))
                or all(b <= a for a, b in zip(shifts, shifts[1:])))
    top = rows[-1]
    endpoint_moved = abs(top["meanShift"]) >= ENDPOINT_MOVEMENT_NATS
    base_freq = rows[0]["frequencyPerThousand"]
    frequency_change = (abs(top["frequencyPerThousand"] - base_freq)
                        / base_freq) if base_freq else (
        float("inf") if top["frequencyPerThousand"] else 0.0)
    frequency_moved = frequency_change > FREQUENCY_MOVEMENT_FRACTION

    dose = CheckResult(
        name="causalDoseResponse",
        passed=bool(monotone and endpoint_moved),
        detail=(f"logprobShift {shifts[0]:+.3f} → {shifts[-1]:+.3f} nats over "
                f"{len(ladder)} dose(s), "
                + ("monotone" if monotone else "NOT monotone")
                + ("" if endpoint_moved else
                   f"; the endpoint did not move beyond "
                   f"{ENDPOINT_MOVEMENT_NATS:g} nats")),
        measured={"endpoint": endpoint.path, "endpointHash": endpoint.hash,
                  "endpointRows": len(endpoint.rows), "rows": rows,
                  "monotone": monotone, "endpointMoved": endpoint_moved,
                  "objective": "logprobShift",
                  "layer": layer, "tokenID": token_id, "piece": piece,
                  "movementThresholdNats": ENDPOINT_MOVEMENT_NATS})

    control = CheckResult(
        name="antiLexicalControl",
        passed=bool(endpoint_moved and not frequency_moved),
        detail=(f"token frequency {base_freq:.2f} → "
                f"{top['frequencyPerThousand']:.2f} per 1000 "
                f"({frequency_change:.0%} change) while the endpoint moved "
                f"{top['meanShift']:+.3f} nats"
                + ("" if not frequency_moved else
                   " — BOTH moved, so what was measured is vocabulary, not "
                   "disposition, and the steering arm is not licensed no "
                   "matter how clean the dose curve looks")),
        measured={"baselinePerThousand": base_freq,
                  "topDosePerThousand": top["frequencyPerThousand"],
                  "relativeChange": frequency_change,
                  "frequencyMoved": frequency_moved,
                  "endpointMoved": endpoint_moved,
                  "frequencyThresholdFraction": FREQUENCY_MOVEMENT_FRACTION,
                  "rows": rows})
    return dose, control


def _check_memory_budget(model) -> CheckResult:
    """Peak device memory while the lens and the runtime are both resident.

    Reported, not judged against a hardcoded ceiling: what fits is a property
    of the node class this ran on, and a threshold invented here would be a
    guess wearing a number. A device that cannot report its peak says so.
    """
    import torch

    device = str(getattr(model, "device", ""))
    measured: dict = {"device": device}
    peak = None
    if device.startswith("cuda"):
        index = torch.cuda.current_device()
        peak = int(torch.cuda.max_memory_allocated(index))
        measured.update({
            "peakAllocatedBytes": peak,
            "peakReservedBytes": int(torch.cuda.max_memory_reserved(index)),
            "totalBytes": int(
                torch.cuda.get_device_properties(index).total_memory),
            "deviceName": torch.cuda.get_device_name(index)})
    elif device.startswith("mps"):
        peak = int(torch.mps.current_allocated_memory())
        measured["allocatedBytes"] = peak
    else:
        measured["note"] = ("no device-level memory counter on this backend — "
                            "the figure a 27B node needs is a CUDA one")
    return CheckResult(
        name="memoryBudget", passed=peak is not None,
        detail=(f"peak {peak / (1 << 30):.1f} GiB on {device}"
                if peak is not None else measured["note"]),
        measured=measured)


def _check_cost(model, readout, config, layers, prompt: str) -> CheckResult:
    """Readout latency, per-projection cost, and realized trace volume.

    The point of the measurement half (plan §3.4 check 7): the §8.4 ``Budget``
    defaults are currently ESTIMATES, and this is what lets them be reset from
    measurement. Every figure is per-observation so it scales to a study of any
    size without re-measuring.
    """
    import torch

    from .qualification import _block_outputs
    from .readout import preflight

    residuals, _logits, prompt_len = _block_outputs(model, prompt, layers)
    layer = layers[len(layers) // 2]
    h = residuals[layer]

    def _time(fn, repeats: int = 5) -> float:
        fn()                                   # warm the kernels
        if str(getattr(model, "device", "")).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            fn()
        if str(getattr(model, "device", "")).startswith("cuda"):
            torch.cuda.synchronize()
        return (time.perf_counter() - started) / repeats

    watch_seconds = (_time(lambda: readout.watched_scores(h, layer))
                     if config.watchlist else None)
    topk_seconds = (_time(lambda: readout.topk(h, layer, config.topK))
                    if readout.lm_head is not None else None)
    companion_seconds = (
        _time(lambda: readout.topk(h, layer, config.topK, use_jacobian=False))
        if readout.lm_head is not None and config.logitLensCompanion else None)

    estimate = preflight(config, generations=100, max_new_tokens=500)
    projected_seconds = None
    if topk_seconds is not None:
        projected_seconds = topk_seconds * estimate["fullVocabProjections"]
    return CheckResult(
        name="costMeasurement", passed=True,
        detail=(f"watchlist {1e3 * watch_seconds:.3f} ms/obs; "
                if watch_seconds is not None else "")
               + (f"top-k {1e3 * topk_seconds:.3f} ms/projection"
                  if topk_seconds is not None else "no top-k armed"),
        measured={
            "promptTokens": prompt_len,
            "watchlistSecondsPerObservation": watch_seconds,
            "topKSecondsPerProjection": topk_seconds,
            "companionSecondsPerProjection": companion_seconds,
            "projectedStudy": estimate,
            "projectedTopKSecondsFor100x500": projected_seconds,
            "note": ("feed these back into readout.Budget's defaults — they "
                     "are estimates until a 27B measurement replaces them"),
        })


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def _verdict(results: dict, names, *, tier: str, scientific: bool) -> dict:
    """A section verdict plus the names behind it.

    At testing tier the SCIENTIFIC section is verdicted ``informational``: a
    4B result describes a different model and cannot stand in for 27B's, and
    the split is enforced here rather than left to whoever writes it up.
    """
    considered = [n for n in names if n in results]
    failures = sorted(n for n in considered if not results[n]["passed"])
    skipped = sorted(n for n in considered if results[n].get("skipped"))
    if scientific and tier != "evidence":
        return {"verdict": "informational", "tier": tier,
                "checks": considered, "failed": failures, "skipped": skipped,
                "reason": (f"'{tier}'-tier model: scientific results are "
                           f"recorded as a PRIOR and never block. Only a "
                           f"27B (evidence-tier) run can verdict these")}
    return {"verdict": "pass" if not failures else "fail",
            "tier": tier, "checks": considered,
            "failed": failures, "skipped": skipped}


def _arm_verdicts(results: dict, *, tier: str) -> dict:
    """Steering and readout, verdicted independently.

    ``null`` at testing tier is the point: both arms rest on scientific
    checks, so a 4B run has nothing to say about either, and a stamped
    ``null`` says that where a stamped ``pass`` would be a lie a reader could
    not detect.
    """
    out: dict = {}
    for arm in ARMS:
        names = [n for n in ARM_CHECKS[arm] if n in results]
        failures = sorted(n for n in names if not results[n]["passed"])
        if tier != "evidence":
            out[arm] = {
                "verdict": None, "licensed": False, "checks": names,
                "failed": failures,
                "reason": (f"unverdicted: '{tier}' tier. This arm rests on "
                           f"scale-bound checks, and a smaller model's answer "
                           f"to them is a prior, not a licence")}
            continue
        out[arm] = {"verdict": "pass" if not failures else "fail",
                    "checks": names, "failed": failures}
        if arm == "readout":
            # NOT `licensed`. The prose said "implementation fidelity" while
            # the machine-readable field said `licensed: true`, and downstream
            # code reads the boolean (external review round 2). The scoped
            # field cannot be mistaken for the broader claim, and the broader
            # claim stays explicitly unknown until a preregistered criterion
            # over readableBand exists.
            out[arm]["implementationFidelityPassed"] = not failures
            out[arm]["scientificInformativeness"] = "unknown"
            out[arm]["licences"] = READOUT_LICENCE
        else:
            out[arm]["licensed"] = not failures
    # The conjunction is "the direction steers AND the readout says why" — and
    # the second half is exactly the claim that is unknown. It therefore
    # cannot be licensed today, whatever the checks say, and reports that
    # rather than computing an AND over a boolean that overstates one side.
    out["conjunction"] = {
        "licensed": False,
        "blockedBy": "readout.scientificInformativeness",
        "note": ("neither arm licenses the conjunction on its own, and it "
                 "cannot be licensed at all while readout informativeness is "
                 "unknown: 'the readout says why' is precisely the "
                 "undetermined half"),
    }
    # Per-instrument plumbing, reported BESIDE the readout arm rather than
    # folded into it. The arm says whether reading means anything here; each
    # instrument says whether the machinery that obtains it is sound. A claim
    # needs both, and needs the plumbing of the instrument it actually used —
    # never the other one's.
    instruments: dict = {}
    for name, names in INSTRUMENT_CHECKS.items():
        present = [n for n in names if n in results]
        failures = sorted(n for n in present if not results[n]["passed"])
        # Fidelity, not a licence: an instrument is USABLE when the readout
        # computes what the reference computes and its own plumbing is sound.
        # Whether the result is informative is a separate, open question.
        readable = out["readout"].get("implementationFidelityPassed")
        instruments[name] = {
            "plumbing": "pass" if not failures else "fail",
            "checks": present,
            "failed": failures,
            # Usable = the shared science licenses reading AND this
            # instrument's own machinery is sound.
            "usable": bool(readable and not failures),
            "usableMeans": ("readout implementation fidelity AND this "
                            "instrument's plumbing; NOT scientific "
                            "informativeness, which is unknown"),
        }
    out["instruments"] = instruments
    out["note"] = (
        "the readout arm licenses READING; each instrument additionally "
        "declares its own plumbing. An offline slice is not licensed by an "
        "alignment check that never looked at slices, and an alignment "
        "failure does not invalidate a claim that never used the aligner.")
    return out


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def run(model_id: str, *, lens_id: str | None = None,
        revision: str | None = None, layers: list[int] | None = None,
        watchlist: list[int] | None = None, token_id: int | None = None,
        piece: str | None = None, endpoint_path: str | None = None,
        alpha_range: tuple[float, ...] | None = None, top_k: int = 10,
        band_stride: int = DEFAULT_BAND_STRIDE,
        prompts_path: str | None = None, root: str | None = None,
        model=None, device: str | None = None, dtype: str | None = None,
        log=None) -> dict:
    """Execute the gate and write ``jlens-g0-report.json``.

    Returns the report. Never raises on a FAILED check — a failure is the
    gate's output — and raises only when the gate could not be attempted.
    """
    from ..experiment import paths
    from ..experiment.run_config import write_run_config
    from ..steering import model_loader
    from . import lens_store
    from .readout import LensReadout, ReadoutConfig

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    lens = lens_id or lens_store.for_model(model_id, root)
    record = lens_store.resolve(lens, root)
    tier = tier_for(model_id, record)
    prompts = load_fixture_prompts(prompts_path)
    carrier = next((r for r in prompts if r.get("causalSmoke")), prompts[-1])
    endpoint = Endpoint.load(endpoint_path, root) if endpoint_path else None

    if model is None:
        emit(f"loading {model_id} …")
        model = model_loader.load(model_id, revision, dtype=dtype,
                                  device=device)
    revision = revision or getattr(model, "revision", None)

    armed = sorted(set(layers)) if layers else _default_layers(record)
    tokenizer = getattr(model, "tokenizer", None)
    token = token_id if token_id is not None else _default_token(tokenizer)
    resolved_piece = piece if piece is not None else (
        tokenizer.decode([token]) if tokenizer is not None else "")
    watch = sorted(set(list(watchlist or []) + [token]))

    config = ReadoutConfig(layers=armed, watchlist=watch, topK=top_k,
                           topKLayers=armed, logitLensCompanion=True)
    readout = LensReadout.build(record=record, config=config, model=model,
                                root=root)

    run_directory = paths.make_unique_run_directory(
        f"{RUN_TYPE}-{model_id.replace('/', '--')}", root)
    write_run_config(run_directory, RUN_TYPE, model_id=model_id,
                     revision=revision, dtype=getattr(model, "dtype", None),
                     notes={"lensID": lens, "tier": tier,
                            "armedLayers": armed,
                            "endpoint": endpoint.path if endpoint else None,
                            "endpointHash": endpoint.hash if endpoint else None})

    started = time.monotonic()
    results: dict = {}

    def record_result(result: CheckResult) -> None:
        results[result.name] = result.to_dict()
        emit(f"  {result.name}: "
             + ("SKIPPED" if result.skipped
                else ("ok" if result.passed else "FAILED"))
             + f" — {result.detail}")

    emit("mechanical checks (a failure here blocks the 27B run):")
    record_result(_check_acquisition(record, root=root))
    record_result(_check_load_path(record, model, root=root))
    record_result(_check_resolution(record, model, readout))
    record_result(_check_alignment(model, readout, config,
                                   carrier["prompt"]))
    record_result(_check_trace_persistence(model, readout, config,
                                           run_directory, carrier["prompt"]))
    record_result(_check_slice_positioning(model, armed, prompts))
    record_result(_check_preflight_mechanics(config))

    emit(f"scientific checks (tier '{tier}'"
         + ("" if tier == "evidence" else "; recorded as a prior, never a gate")
         + "):")
    record_result(_check_reference_readouts(record, model, readout, armed,
                                            prompts, root=root))
    ladder = tuple(alpha_range) if alpha_range else DEFAULT_ALPHA_LADDER
    dose, control = _steering_arm_science(
        record, model, readout, layer=armed[len(armed) // 2], token_id=token,
        piece=resolved_piece, endpoint=endpoint, ladder=ladder,
        carrier=carrier["prompt"], root=root)
    record_result(dose)
    record_result(control)
    record_result(_check_memory_budget(model))
    record_result(_check_cost(model, readout, config, armed,
                              carrier["prompt"]))

    # Measurements, not checks. Neither has a threshold anyone could justify,
    # and both exist to REPLACE a number the study is currently inheriting: the
    # resolution limit replaces "we assume replay is faithful", and the band
    # replaces the paper's workspace prior for THIS model.
    emit("measurements (stamped, never verdicted):")
    measurements: dict = {}
    try:
        measurements["replayFidelity"] = _measure_replay_fidelity(
            model, readout, config, armed, carrier["prompt"])
        fidelity = measurements["replayFidelity"]
        if fidelity.get("measured"):
            emit(f"  replay fidelity: model argmax agreement "
                 f"{fidelity['modelArgmaxAgreement']:.2f}; per-layer "
                 + ", ".join(
                     f"L{k} top-1 {v['top1Same']}/{v['n']}"
                     for k, v in fidelity["perLayer"].items()))
    except Exception as exc:  # noqa: BLE001 - a measurement never sinks a gate
        measurements["replayFidelity"] = {"measured": False,
                                          "error": f"{type(exc).__name__}: {exc}"}
        emit(f"  replay fidelity: not measured ({exc})")
    if band_stride > 0:
        try:
            measurements["readableBand"] = _measure_readable_band(
                model, record,
                lambda ls: LensReadout.build(
                    record=record,
                    config=ReadoutConfig(layers=ls, watchlist=watch,
                                         topK=1, topKLayers=ls,
                                         logitLensCompanion=False),
                    model=model, root=root),
                prompts, stride=band_stride, root=root)
            rows = measurements["readableBand"].get("perLayer") or []
            if rows:
                best = min(rows, key=lambda r: r["medianNextTokenRank"])
                emit(f"  readable band: {len(rows)} layer(s) scored; best "
                     f"median next-token rank {best['medianNextTokenRank']} "
                     f"at L{best['layer']} ({best['depthFraction']:.2f} depth)")
        except Exception as exc:  # noqa: BLE001
            measurements["readableBand"] = {"measured": False,
                                            "error": f"{type(exc).__name__}: {exc}"}
            emit(f"  readable band: not measured ({exc})")
    else:
        measurements["readableBand"] = {
            "measured": False,
            "reason": "band sweep disabled (--band-stride 0)"}

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "runType": RUN_TYPE,
        "runID": os.path.basename(run_directory),
        "gate": "G0 1b (plan §3.4)",
        "model": {"modelID": model_id, "revision": revision,
                  "dtype": getattr(model, "dtype", None),
                  "device": str(getattr(model, "device", "")),
                  "tier": tier},
        "lens": {"lensID": lens, "sourceSHA256": record.source.tensorSHA256,
                 "sourceLayers": [record.sourceLayers[0],
                                  record.sourceLayers[-1]],
                 "targetLayer": record.targetLayer,
                 "referencePackage": record.referencePackage,
                 "referenceCommit": record.referenceCommit},
        "configuration": {"armedLayers": armed, "watchlist": watch,
                          "topK": top_k, "logitLensCompanion": True,
                          "alphaLadder": list(ladder),
                          "tokenID": token, "piece": resolved_piece,
                          "endpoint": endpoint.path if endpoint else None,
                          "endpointHash": endpoint.hash if endpoint else None,
                          "promptFixtures": os.path.basename(
                              prompts_path or "qualification-prompts.jsonl")},
        # The split is the schema, not the prose: a reader cannot accidentally
        # read a 4B scientific result as a gate.
        "mechanical": _verdict(results, MECHANICAL_CHECKS, tier=tier,
                               scientific=False),
        "scientific": _verdict(results, SCIENTIFIC_CHECKS, tier=tier,
                               scientific=True),
        "arms": _arm_verdicts(results, tier=tier),
        "checks": results,
        "measurements": measurements,
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "runDirectory": run_directory,
    }
    path = os.path.join(run_directory, REPORT_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report["reportPath"] = path

    emit(f"mechanical: {report['mechanical']['verdict']}; "
         f"scientific: {report['scientific']['verdict']}; "
         f"steering arm: {report['arms']['steering']['verdict']}; "
         f"readout arm: {report['arms']['readout']['verdict']}")
    emit(path)
    return report


def _default_layers(record) -> list[int]:
    from .qualification import _default_layers as pick

    return pick(record)


def _default_token(tokenizer) -> int:
    from .qualification import _default_token as pick

    return pick(tokenizer)

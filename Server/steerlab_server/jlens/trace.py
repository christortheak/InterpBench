"""The durable J-lens trace: ``jlens-readout.jsonl``.

A separate stream from ``generations.jsonl``, for the same reason the capability
battery is separate: these are per-step observations, and folding them into the
outcome records would swamp the file everything else reads. ``generations.jsonl``
gets only a reference, a hash, and small declared summaries.

Reuses ``resume.GenerationWriter``, so append discipline, fsync, the skip set,
between-record checkpointing, and shard filtering all behave exactly as they do
for generations — one discipline, not a second one that drifts.

**A failed or interrupted trace must never pass as a complete readout.** Each
record carries its own completion state, and the run-level summary carries the
expected token count next to the realized one, so a consumer can tell a whole
trace from a truncated one without re-deriving anything (plan §9).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict

from .recorder import ALIGNMENT_CONVENTION, OBSERVATION_CONVENTION
from .schemas import CANONICAL_READOUT, DIRECTION_CONVENTION, JLensError

TRACE_FILENAME = "jlens-readout.jsonl"


@dataclass
class TraceIdentity:
    """Everything a record must name to be interpretable on its own.

    Deliberately verbose per record rather than a header: shards are merged by
    concatenation, and a header would either be duplicated or lost. A row that
    cannot say which model, lens, and steering condition produced it is not
    evidence of anything.
    """

    run: str
    condition: str
    promptID: str | None = None
    promptIndex: int | None = None
    sampleIndex: int | None = None
    modelID: str | None = None
    modelRevision: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    tokenizerHash: str | None = None
    lensID: str | None = None
    lensSHA256: str | None = None
    qualificationID: str | None = None
    #: The readout configuration these observations were produced under. A row
    #: without it cannot be told apart from one recorded with a different
    #: watchlist, top-k width, or budget — so a report could pool two
    #: measurements into one mean (external review round 2).
    configHash: str | None = None
    #: Trustworthiness of the READOUT — a property of the model tier and the
    #: lens qualification, and of nothing else. It is deliberately NOT renamed
    #: (existing traces and the app decode this key), but read its scope
    #: exactly: it answers "can this lens be believed on this runtime", never
    #: "is this row reportable". The latter is `conditionClaim`.
    evidenceTier: str = "testing"
    #: Reportability of THIS ROW: the weaker of the lens's tier and the
    #: identity of the condition being read (external review round 7).
    #:
    #: A qualified lens over an agent whose `adapter_config.json` is unpinned
    #: is still an exploratory measurement — rank, target modules, scaling and
    #: WHICH layers the adapter touches can all change while the agent's
    #: declared identity stays byte-identical. Before this field the study
    #: path stamped the lens tier on every condition's rows alike, so a legacy
    #: agent's trace was labelled evidence; only the standalone probe was
    #: honest. Consumers describing reportability must read THIS.
    conditionClaim: str | None = None
    #: Why `conditionClaim` says what it says: the pin status and hashes of
    #: the agent this condition ran. Present for variant conditions; absent
    #: for baseline/concept conditions, which have no adapter to pin.
    conditionIdentity: dict | None = None
    substrate: str = "python-hf-transformers"
    # The arming that produced these activations. Without it a steered row and
    # a baseline row are indistinguishable after the fact.
    steering: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def config_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def readout_config_payload(config, *, record=None, qualification_id=None,
                           budget=None, tokenizer_hash: str | None = None) -> dict:
    """The pinned ``jlensReadout`` block a frozen study verifies against."""
    payload = {
        "layers": sorted(set(config.layers)),
        "watchlist": list(config.watchlist),
        "topK": config.topK,
        "topKLayers": config.armed_topk_layers(),
        "logitLensCompanion": bool(config.logitLensCompanion),
        "observationConvention": OBSERVATION_CONVENTION,
        "alignmentConvention": ALIGNMENT_CONVENTION,
        "readoutConvention": CANONICAL_READOUT,
        "directionConvention": DIRECTION_CONVENTION,
    }
    if record is not None:
        payload["lensID"] = record.lensID
        payload["lensSHA256"] = record.source.tensorSHA256
    if tokenizer_hash is not None:
        # The readout is indexed by token ID, so the vocabulary IS its
        # coordinate system: a tokenizer change re-points every watched token
        # at a different piece while every number still looks plausible.
        payload["tokenizerHash"] = tokenizer_hash
    if qualification_id is not None:
        payload["qualificationID"] = qualification_id
    if budget is not None:
        payload["budget"] = asdict(budget)
    return payload


def mention_mask(token_ids: list[int], prompt_ids: list[int],
                 generated_ids: list[int], upto_index: int) -> dict[int, bool]:
    """Which watched tokens have already APPEARED by this prediction step.

    The paper's mention-priming effect: a target token present verbatim in the
    rendered stimulus sits near ceiling at baseline, so its raw loading says
    more about the prompt than about the model's state. Extended past the
    prompt to the generated prefix, because a token the model has already
    emitted is primed for the same reason.

    Reported per step, never used to silently drop rows — a mentioned token's
    score stays in the trace and is simply distinguishable from an unmentioned
    one (plan §8.2).
    """
    seen = set(prompt_ids) | set(generated_ids[:max(0, upto_index)])
    return {int(t): (int(t) in seen) for t in token_ids}


def trace_record(recorder, identity: TraceIdentity, *,
                 prompt_ids: list[int], generated_ids: list[int],
                 watchlist: list[int]) -> dict:
    """ONE row per generation, with the per-step observations nested.

    The granularity is forced, and it is also the right one.
    ``resume.record_key`` identifies a record by
    ``(condition, promptIndex, promptID, sampleIndex, kind)``, so every
    observation from a single generation shares a key — emitting them as
    separate rows would let the writer's idempotency drop all but the first.
    Nesting makes the key match the generation exactly, so skip, resume, and
    the shard filter all work with no special case, and resume granularity
    becomes atomic: half a generation's trace is meaningless, so it is never
    what a resumed run inherits.

    It is also much smaller — the identity block appears once instead of once
    per layer per step.
    """
    row = identity.to_dict()
    # Descriptive only. resume.record_kind derives a record's kind from the
    # presence of "error"/"instrument" and ignores any "kind" field, so naming
    # this key "kind" would look like it participates in the resume key when it
    # does not — and a skip() asking for a kind the reloader never reconstructs
    # would silently re-append every trace row on resume.
    row["recordType"] = "jlensReadout"
    row["observations"] = [o.to_dict() for o in recorder.observations]
    row["observationCount"] = len(recorder.observations)
    row["generatedTokenCount"] = len(generated_ids)
    row["traceComplete"] = recorder.complete
    if recorder.failureReason:
        row["traceFailureReason"] = recorder.failureReason
    if watchlist:
        row["watchlistTokenIDs"] = list(watchlist)
        # Per STEP, because priming accrues as the model writes: a token is
        # mentioned once it has appeared in the prompt or in the prefix
        # generated so far.
        row["mentionMask"] = {
            str(o.predictedIndex): mention_mask(
                watchlist, prompt_ids, generated_ids, o.predictedIndex)
            for o in recorder.observations
        }
    return row


class TraceWriter:
    """Append-only ``jlens-readout.jsonl`` with the house discipline."""

    def __init__(self, run_directory: str, *, checkpoint=None, resume: bool = False,
                 log=None, allowed_keys=None):
        from ..experiment import resume as resume_mod

        self._writer = resume_mod.GenerationWriter(
            run_directory, verb="run", checkpoint=checkpoint, resume=resume,
            log=log, filename=TRACE_FILENAME, allowed_keys=allowed_keys)
        self.path = self._writer.path
        self.rows_written = 0
        self.observations_written = 0
        self.digest = hashlib.sha256()
        self.incomplete_records = 0

    def skip(self, condition: str, prompt_index, prompt_id, sample_index) -> bool:
        """Whether this generation's trace already exists (or is another
        shard's) — the same pre-generation check the run loop makes.

        Uses the DEFAULT kind on purpose. This file has its own key space (one
        row per generation, in its own JSONL), so the key only has to be
        internally consistent; asking for a custom kind would not match what
        ``record_kind`` reconstructs when the file is reloaded, and a resumed
        run would duplicate every row.
        """
        return self._writer.skip(condition, prompt_index, prompt_id,
                                 sample_index)

    def write(self, row: dict, recorder=None) -> None:
        self._writer.emit(row)
        self.digest.update(json.dumps(row).encode("utf-8"))
        self.rows_written += 1
        self.observations_written += int(row.get("observationCount") or 0)
        if recorder is not None and not recorder.complete:
            self.incomplete_records += 1

    def close(self) -> None:
        self._writer.close()

    def summary(self, *, expected_records: int | None = None) -> dict:
        """The block ``generations.jsonl``/``report.json`` reference.

        Carries the counts rather than the observations, so the compact files
        stay compact while still being able to prove the trace whole.
        """
        return {
            "trace": TRACE_FILENAME,
            "traceRows": self.rows_written,
            "traceObservations": self.observations_written,
            "traceSHA256": self.digest.hexdigest(),
            "incompleteRecords": self.incomplete_records,
            "expectedRecords": expected_records,
            # A single incomplete record poisons the run for reportable use:
            # gate on this rather than on the presence of the file.
            #
            # ZERO rows is not complete. This said `incomplete_records == 0`,
            # which is vacuously true for a trace that recorded nothing — so an
            # armed readout that never ran (a choice-only study) stamped
            # `complete: true` over an empty file, while `read_summary`, which
            # gates reportable use, said false for the same trace. Two
            # functions answering one question differently (external review
            # round 3).
            "complete": self.incomplete_records == 0 and self.rows_written > 0,
        }


def require_complete(summary: dict, *, expected_rows: int | None = None,
                     expected_hash: str | None = None) -> None:
    """Gate for reportable consumers (plan §9).

    Presence of a trace file is not evidence it is whole. Refuse loudly rather
    than let a truncated readout be analyzed as a complete one.
    """
    if not summary.get("complete"):
        raise JLensError(
            f"J-lens trace is incomplete ({summary.get('incompleteRecords')} "
            f"record(s) failed or were truncated) — not usable as a readout")
    if expected_rows is not None and summary.get("traceRows") != expected_rows:
        raise JLensError(
            f"J-lens trace has {summary.get('traceRows')} rows, expected "
            f"{expected_rows} — the trace does not cover the run")
    if expected_hash is not None and summary.get("traceSHA256") != expected_hash:
        raise JLensError(
            "J-lens trace hash does not match the pinned value — the trace was "
            "modified or belongs to a different run")


class TraceSession:
    """Run-local coordinator: one per run, one recorder per generation.

    The run loop holds this and asks it two things — give me a recorder for
    this item, and record what it saw. Everything else (the writer, the
    device-resident readout, the identity block) stays here so the loop is not
    threaded through with lens state.

    Buffers are per-recorder and never shared between generations: a recorder
    that outlived its generation would attribute one item's activations to
    another.
    """

    def __init__(self, *, record, config, readout, writer: TraceWriter,
                 run_id: str, evidence_tier: str = "testing",
                 qualification_id: str | None = None, root=None,
                 condition_identities: dict | None = None,
                 tokenizer_hash: str | None = None, budget=None):
        self.record = record
        self.config = config
        self.readout = readout
        self.writer = writer
        self.run_id = run_id
        self.evidence_tier = evidence_tier
        self.qualification_id = qualification_id
        self.root = root
        # Verified ONCE per condition at run start and carried into every row.
        # Verification is file I/O over adapter weights, and an identity is
        # immutable for the life of a run — recomputing it per generated row
        # re-read hundreds of MB thousands of times to learn the same answer
        # (external review round 9). `asdict` deep-copies on the way into each
        # record, so one shared block per condition cannot be mutated by a
        # consumer.
        self.condition_identities = dict(condition_identities or {})
        self.tokenizer_hash = tokenizer_hash
        self.budget = budget
        # The budget is part of the pinned identity: it is a declared choice
        # that bounds what the run may record, so an edit to it after freeze
        # must be a verify violation like any other measurement-side change
        # (external review 2026-08-16).
        self.configHash = config_hash(readout_config_payload(
            config, record=record, qualification_id=qualification_id,
            tokenizer_hash=tokenizer_hash, budget=budget))

    @classmethod
    def open(cls, *, run_directory: str, lens_id: str, config, model,
             root=None, checkpoint=None, resume: bool = False, log=None,
             evidence_tier: str = "testing", qualification_id: str | None = None,
             budget=None, condition_identities: dict | None = None):
        """Resolve and validate at RUN START, not at first use.

        A misconfigured readout must refuse before the model slot is spent, not
        after the first item has already generated.
        """
        from . import lens_store
        from .readout import LensReadout

        record = lens_store.resolve(lens_id, root)
        readout = LensReadout.build(record=record, config=config, model=model,
                                    root=root)
        writer = TraceWriter(run_directory, checkpoint=checkpoint,
                             resume=resume, log=log)
        tokenizer_hash = None
        if record.fit.modelID:
            from .derive import tokenizer_identity_hash

            try:
                tokenizer_hash = tokenizer_identity_hash(record.fit.modelID)
            except Exception:  # noqa: BLE001 - absence is recorded, not fatal
                tokenizer_hash = None
        return cls(record=record, config=config, readout=readout, writer=writer,
                   run_id=os.path.basename(run_directory.rstrip(os.sep)),
                   evidence_tier=evidence_tier,
                   qualification_id=qualification_id, root=root,
                   tokenizer_hash=tokenizer_hash, budget=budget,
                   condition_identities=condition_identities)

    def recorder_for(self, prompt: dict):
        """A fresh recorder, or None when this item's trace already exists."""
        from .recorder import JLensReadoutRecorder

        return JLensReadoutRecorder(self.readout, self.config)

    def identity_for(self, eff):
        """This condition's verified agent identity — computed at most once.

        Order matters, and it is an order of TRUSTWORTHINESS, not convenience:

        1. The identity attached to this EffectiveCondition, verified against
           the bytes immediately before its adapter loaded. Authoritative: it
           describes the adapter that actually shaped these rows, and it is
           carried on the object that ran rather than looked up by display
           name — condition names are not guaranteed unique, and a name-keyed
           lookup would stamp one agent's rows with another agent's identity
           (external review round 10).
        2. The run-start preflight map, keyed by name. A fast-failure result
           computed possibly hours earlier; used only when nothing was
           attached.
        3. Verify now, and memoize — so the fallback cannot silently become
           the hot path. Verification is file I/O over adapter weights
           (hundreds of MB for a 27B LoRA), and re-reading them per generated
           row learns the same answer thousands of times (round 9).
        """
        attached = getattr(eff, "verified_identity", None)
        if attached is not None:
            return attached
        name = getattr(eff, "name", None)
        if name in self.condition_identities:
            return self.condition_identities[name]
        identity = condition_identity(eff, self.root)
        if name is not None:
            self.condition_identities[name] = identity
        return identity

    def record_generation(self, recorder, eff, prompt, prompt_index,
                          sample_index, *, model, manifest,
                          generated_ids: list[int]) -> dict:
        """Join, persist the row, and return the compact reference block."""
        recorder.join_token_ids(generated_ids)
        cond_identity = self.identity_for(eff)
        identity = TraceIdentity(
            run=self.run_id, condition=eff.name,
            promptID=prompt.get("id"), promptIndex=prompt_index,
            sampleIndex=sample_index,
            modelID=getattr(manifest, "model_id", None),
            modelRevision=getattr(model, "revision", None),
            dtype=getattr(model, "dtype", None),
            lensID=self.record.lensID,
            lensSHA256=self.record.source.tensorSHA256,
            qualificationID=self.qualification_id,
            configHash=self.configHash,
            tokenizerHash=self.tokenizer_hash,
            evidenceTier=self.evidence_tier,
            conditionIdentity=cond_identity,
            conditionClaim=condition_claim(
                self.evidence_tier, self.qualification_id, cond_identity),
            steering=_steering_provenance(eff))
        # The recorder's own prompt ids, supplied by the generation driver.
        # This was `[]` until 2026-08-15, which quietly disabled the PROMPT
        # half of the mention mask: a watched token sitting verbatim in the
        # stimulus was averaged in at ceiling and `excludedObservations` read
        # zero, so the guard looked like it had run and found nothing.
        row = trace_record(recorder, identity,
                           prompt_ids=getattr(recorder, "prompt_ids", []) or [],
                           generated_ids=generated_ids,
                           watchlist=self.config.watchlist)
        self.writer.write(row, recorder)
        return {
            "trace": TRACE_FILENAME,
            "configHash": self.configHash,
            "observations": len(recorder.observations),
            "complete": recorder.complete,
            **({"failureReason": recorder.failureReason}
               if recorder.failureReason else {}),
        }

    def close(self, *, expected_records: int | None = None) -> dict:
        self.writer.close()
        return self.writer.summary(expected_records=expected_records)


def condition_identity(eff, root: str | None = None) -> dict | None:
    """The VERIFIED identity of the agent this condition ran, or None.

    Baseline and concept conditions have no adapter, so they have nothing to
    pin and get no identity block — their claim rests on the lens alone.

    Delegates to the one shared verifier (`model_variant`), which reads the
    adapters as the dicts they actually are and hashes the files on disk.
    This function previously had its own copy that used `getattr` and trusted
    a declaration: against a real `ModelVariant` every fully pinned agent
    resolved to "?" / unpinned and was stamped exploratory, and a drifted
    adapter would still have been called qualified (external review round 8).

    Raises `AdapterIdentityError` on a mismatch, so a run refuses BEFORE
    generating rather than labelling the output afterwards.
    """
    from ..experiment.model_variant import verified_adapter_identity

    variant = getattr(eff, "variant", None)
    if variant is None:
        return None
    return {"variantName": getattr(variant, "name", None),
            "adapters": verified_adapter_identity(variant, root)}


def condition_claim(evidence_tier: str, qualification_id, identity) -> str:
    """The weaker of "is the lens believable" and "is the condition pinned".

    Both must hold for a row to be reportable, and they are independent facts:
    a qualified lens says the readout can be believed, and says nothing at all
    about the agent being read (external review round 7).
    """
    if evidence_tier != "evidence" or not qualification_id:
        return "exploratory"
    for adapter in (identity or {}).get("adapters") or []:
        # Both halves declared AND both verified against the bytes. A
        # declaration alone is a claim about an adapter, not a measurement of
        # the one that loaded.
        for key in ("adapterHash", "configHash"):
            if not adapter.get(f"{key}Pinned") or not adapter.get(f"{key}Verified"):
                return "exploratory"
    return "qualified"


def _steering_provenance(eff) -> list[dict]:
    """The arming that produced these activations.

    Without it a steered row and a baseline row are indistinguishable after the
    fact, which would make the whole trace uninterpretable for exactly the
    comparison it exists to support.
    """
    out = []
    for cell in (getattr(eff, "injections", None) or []):
        out.append({
            "layer": getattr(cell, "layer", None),
            "alpha": getattr(cell, "alpha", None),
            "mode": getattr(cell, "mode", "add"),
            "concept": getattr(cell, "concept", "") or None,
        })
    return out


def require_reportable(run_directory: str, *, name: str,
                       live_hash: str | None = None, live_manifest=None,
                       allow_unverified_epoch: bool = False,
                       expected_rows: int | None = None,
                       expected_hash: str | None = None) -> dict:
    """The full gate before a trace may be analyzed as evidence.

    Two independent questions, and passing one does not answer the other:

    * **Is the trace whole?** A file exists is not the same as a readout
      completed (:func:`require_complete`).
    * **Does it belong to THIS study?** The manifest-epoch guard, the same one
      evaluate/analyze/promote apply. A trace produced before the manifest
      changed describes settings that are no longer the study's, and the
      readout is exactly the kind of measurement-side declaration an edit can
      invalidate — a changed watchlist or band makes the stored numbers answers
      to a different question.

    The epoch check is per-engine by design (canonicalization differs), so a
    trace is judged on the engine that produced it — hence the substrate stamp
    on every row.
    """
    from ..experiment import run_epoch

    summary = read_summary(run_directory)
    if summary is None:
        raise JLensError(
            f"no J-lens trace in '{run_directory}' — nothing to report")
    require_complete(summary, expected_rows=expected_rows,
                     expected_hash=expected_hash)
    if live_hash is not None:
        refusal, unverified, _ = run_epoch.epoch_refusal(
            "jlens readout", name, live_hash, run_directory,
            allow_unverified=allow_unverified_epoch,
            live_manifest=live_manifest)
        if refusal:
            raise JLensError(refusal)
        summary = {**summary, "epochUnverified": unverified}
    return summary


def read_summary(run_directory: str) -> dict | None:
    """Recompute the trace summary from a run directory on disk.

    Derived from the file rather than trusted from a stamp: a summary that
    reports on itself cannot detect truncation after the fact, which is the
    failure mode this guards.
    """
    path = os.path.join(run_directory, TRACE_FILENAME)
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    rows = observations = incomplete = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            digest.update(line.encode("utf-8"))
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                incomplete += 1
                continue
            observations += int(row.get("observationCount") or 0)
            if not row.get("traceComplete"):
                incomplete += 1
    return {
        "trace": TRACE_FILENAME,
        "traceRows": rows,
        "traceObservations": observations,
        "traceSHA256": digest.hexdigest(),
        "incompleteRecords": incomplete,
        "expectedRecords": None,
        "complete": incomplete == 0 and rows > 0,
    }

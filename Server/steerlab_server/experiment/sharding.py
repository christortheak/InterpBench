"""Multi-GPU sharded study runs: shard planning, shard.json, and the merge.

The enabling fact (test-pinned): every generation record of a study run is
independent — ``derive_seed(experimentHash, condition, promptID, sampleIndex)``
has no cross-record state, and greedy records never touch the RNG at all — so
ANY partition of a run's record set produces byte-identical outputs. Sharding
collects that payoff: the full record set of a run is the executor's existing
deterministic iteration order (condition × prompt × sampleIndex, instrument
readouts included), K shards are K contiguous balanced ranges of that order,
and the merge concatenates shard partials back into one ordinary immutable
run directory whose ``generations.jsonl`` is byte-identical to a single-job
run.

Hard rules enforced here:

- ``parallelJobs`` is execution logistics: it NEVER enters the manifest or
  the content hash. Shard provenance lives in ``shard.json`` (per partial)
  and in the merged ``report.json``'s ``sharded`` block — never in
  ``config.json`` (closed schema-2 contract) and never in the manifest.
- The merge REFUSES on incompleteness: every expected (condition, promptID,
  sampleIndex) cell must be present exactly once across shards. Missing or
  duplicated cells refuse loudly and leave the shard partials intact for
  resume — the merge re-runs once the incomplete shard completes.
- Runs stay immutable: the merge writes a NEW run directory assembled from
  the partials; it never mutates a shard partial.

This module is deliberately light on imports at module scope (stdlib +
``resume``) so tests and the job reconciler can import it without paying the
torch import tax; everything model/manifest-shaped is imported lazily inside
the functions that need it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass

from . import resume as resume_mod
from . import run_status
from . import voice_lint

SHARD_FILENAME = "shard.json"
SHARD_SCHEMA = 1

#: Loud upper bound on the fan-out. A typo'd parallelJobs of 300 should be a
#: refusal at submit time, not 300 sbatch calls.
MAX_PARALLEL_JOBS = 64

#: Files that are PER-SHARD by construction and therefore never copied from a
#: shard partial into the merged run directory (the merge writes or rebuilds
#: its own). Everything else in a shard partial is a deterministic run
#: artifact (vectors, substrate.json, advisories.txt) that must be
#: byte-identical across shards — verified, then copied once.
_PER_SHARD_FILES = frozenset({
    "generations.jsonl", "battery.jsonl", "jlens-readout.jsonl",
    "metrics.csv", "summaries.csv",
    # A panel shard's voice-lint roll-up covers only the transcripts that
    # shard ran, so two shards' copies differ by construction. Rebuilt below
    # over the merged records, exactly like metrics.csv.
    voice_lint.VOICE_LINT_FILENAME,
    "report.json", SHARD_FILENAME, resume_mod.RESUME_STATE_FILENAME,
    "experiment.json", "experiment-hash.txt", "task.txt", "config.json",
    # A shard's own failure/progress record, timestamped and shard-local by
    # nature. Nothing writes it on the success path a merge consumes, so
    # this is not a bug being fixed — it is the class of file that must
    # never be cross-shard compared, listed before it can become one
    # (external review round 4, finding 3).
    run_status.STATUS_FILENAME, run_status.FAILURE_NOTE_FILENAME,
})


class ShardError(ValueError):
    """A shard specification that cannot be honored (parse/range errors)."""


class ShardMergeError(RuntimeError):
    """A merge refusal: the message is the actionable, plain-language text."""


@dataclass(frozen=True)
class ShardSpec:
    """One shard's identity: 0-based ``index`` of ``count`` total shards."""
    index: int
    count: int

    def __post_init__(self):
        if self.count < 1:
            raise ShardError(f"shard count must be >= 1, got {self.count}")
        if self.count > MAX_PARALLEL_JOBS:
            raise ShardError(
                f"shard count {self.count} exceeds the fan-out cap "
                f"({MAX_PARALLEL_JOBS})")
        if not 0 <= self.index < self.count:
            raise ShardError(
                f"shard index must be in [0, {self.count}), got {self.index}")

    @property
    def label(self) -> str:
        return f"{self.index}/{self.count}"


def parse_shard(text: str) -> ShardSpec:
    """Parse the CLI's ``--shard k/K`` (0-based k)."""
    head, sep, tail = (text or "").strip().partition("/")
    if not sep or not head.lstrip("-").isdigit() or not tail.isdigit():
        raise ShardError(
            f"--shard must be k/K with 0-based k (e.g. 0/3), got {text!r}")
    return ShardSpec(index=int(head), count=int(tail))


def shard_bounds(total: int, index: int, count: int) -> tuple[int, int]:
    """Balanced contiguous [start, end) over ``total`` records: the first
    ``total % count`` shards carry one extra record. The union over all
    indices partitions [0, total) exactly."""
    base, remainder = divmod(total, count)
    start = index * base + min(index, remainder)
    return start, start + base + (1 if index < remainder else 0)


def expected_record_keys(*, condition_names: list[str], prompts: list[dict],
                         wants_choice: bool, wants_sampled: bool,
                         sample_count: int,
                         instrument_scope: dict | None = None) -> list[tuple]:
    """The FULL expected record-key list of a run, in the executor's exact
    emission order (``tasks._execute_condition``): per condition, per prompt,
    one instrument readout when the choice instrument applies to the item,
    then one sampled record per sample index. Keys use the resume module's
    canonical record-key vocabulary, so shard membership, resume skipping,
    and merge completeness all speak the same identity.

    ``instrument_scope`` is the manifest's ``outcomeInstrumentScope``, and it
    MUST be threaded through whenever the manifest declares one: the executor
    declines to measure out-of-scope items (``response_format.scope_includes``
    — declining is the whole point of declaring a scope), so an expected-key
    list built without it counts instrument records nothing will ever emit.
    Observed live 2026-08-04 (`test-compare-2-2`): 24 option-bearing items,
    scope covering 12 — every shard stamp expected 12 phantom instrument
    readouts per condition and the merge refused a run whose shards had all
    succeeded, with a message blaming an incomplete shard. Resubmits can
    never converge on such a plan; the fix is that the planner and the
    executor apply the SAME scope rule."""
    from . import response_format
    keys: list[tuple] = []
    for condition in condition_names:
        for prompt_index, prompt in enumerate(prompts):
            in_scope = response_format.scope_includes(
                instrument_scope,
                {"id": prompt.get("id"), "hasOptions": True,
                 "format": prompt.get("responseFormat")})
            if wants_choice and prompt.get("options") and in_scope:
                keys.append(resume_mod.make_key(
                    condition, prompt_index, prompt["id"], None,
                    resume_mod.KIND_INSTRUMENT))
            if wants_sampled:
                for sample_index in range(sample_count):
                    keys.append(resume_mod.make_key(
                        condition, prompt_index, prompt["id"], sample_index,
                        resume_mod.KIND_SAMPLED))
    return keys


@dataclass(frozen=True)
class ShardPlan:
    """One shard's resolved slice of a run: its allowed record keys, and the
    conditions this shard OWNS for condition-level work (the capability
    battery and variant-failure error records) — ownership is the shard
    containing the condition's FIRST expected record, so each condition's
    battery runs exactly once across the fleet."""
    spec: ShardSpec
    total_records: int
    record_range: tuple[int, int]
    keys: tuple
    owned_conditions: tuple

    @property
    def allowed_keys(self) -> set:
        return set(self.keys)

    def owns_condition(self, name: str) -> bool:
        return name in self.owned_conditions

    def condition_participates(self, name: str) -> bool:
        """Whether this shard has ANY work for the condition (records to
        generate, or condition-level ownership)."""
        return name in self.owned_conditions or any(
            key[0] == name for key in self.keys)


def plan_shard(spec: ShardSpec, *, all_keys: list[tuple],
               condition_names: list[str]) -> ShardPlan:
    start, end = shard_bounds(len(all_keys), spec.index, spec.count)
    keys = tuple(all_keys[start:end])
    owned: list[str] = []
    for name in condition_names:
        first = next((i for i, key in enumerate(all_keys) if key[0] == name),
                     None)
        if first is None:
            # A condition with no expected records (e.g. choice-only
            # instruments and no option-bearing items): deterministically
            # owned by the shard whose range contains the position it WOULD
            # occupy, clamped to the last shard.
            position = min(
                _condition_start_position(all_keys, condition_names, name),
                max(0, len(all_keys) - 1))
            owner = _owner_of_index(position, len(all_keys), spec.count)
        else:
            owner = _owner_of_index(first, len(all_keys), spec.count)
        if owner == spec.index:
            owned.append(name)
    return ShardPlan(spec=spec, total_records=len(all_keys),
                     record_range=(start, end), keys=keys,
                     owned_conditions=tuple(owned))


def plan_panel_shard(spec: ShardSpec, *, condition_names: list[str],
                    replicates: int, turn_ids: list[str]) -> ShardPlan:
    """Shard a PANEL study over transcripts while speaking the ordinary
    record-key vocabulary.

    Two constraints have to hold at once. Turns within a transcript are
    ordered — turn k is conditioned on 1..k-1 — so a transcript can never be
    split across workers. But the merge proves completeness over per-RECORD
    keys, so the stamp must list every record, not every transcript. So the
    split is computed on the transcript list and then expanded to that
    slice's record keys.

    ``sampleIndex`` carries the replicate: a panel's replicate IS its sample
    axis (``samplesPerItem`` drives it), so the existing 5-tuple identity
    works unchanged and ``resume.record_key`` needs no panel special case.
    """
    transcripts = [(condition, replicate)
                   for condition in condition_names
                   for replicate in range(max(1, replicates))]
    start, end = shard_bounds(len(transcripts), spec.index, spec.count)
    owned_transcripts = transcripts[start:end]

    def keys_for(condition: str, replicate: int) -> list[tuple]:
        return [resume_mod.make_key(condition, index, turn_id, replicate,
                                    resume_mod.KIND_SAMPLED)
                for index, turn_id in enumerate(turn_ids)]

    all_keys = [key for condition, replicate in transcripts
                for key in keys_for(condition, replicate)]
    owned_keys = [key for condition, replicate in owned_transcripts
                  for key in keys_for(condition, replicate)]

    per_transcript = len(turn_ids)
    record_range = (start * per_transcript, end * per_transcript)
    # Same ownership rule as the record planner: the shard holding a
    # condition's FIRST record owns its condition-level work.
    owned_conditions = []
    for name in condition_names:
        first = next((i for i, (c, _) in enumerate(transcripts) if c == name), None)
        if first is not None and start <= first < end:
            owned_conditions.append(name)
    return ShardPlan(spec=spec, total_records=len(all_keys),
                     record_range=record_range, keys=tuple(owned_keys),
                     owned_conditions=tuple(owned_conditions))


def _condition_start_position(all_keys: list[tuple],
                              condition_names: list[str], name: str) -> int:
    """Where a zero-record condition would sit in the emission order: after
    every record of every earlier condition."""
    earlier = set(condition_names[:condition_names.index(name)])
    return sum(1 for key in all_keys if key[0] in earlier)


def _owner_of_index(index: int, total: int, count: int) -> int:
    if total <= 0:
        return 0
    index = min(max(0, index), total - 1)
    for shard in range(count):
        start, end = shard_bounds(total, shard, count)
        if start <= index < end:
            return shard
    return count - 1


# --- shard.json ----------------------------------------------------------------

def shard_path(run_directory: str) -> str:
    return os.path.join(run_directory, SHARD_FILENAME)


def write_shard_stamp(run_directory: str, plan: ShardPlan,
                      experiment_hash: str) -> None:
    """Deterministic shard provenance for the partial run directory (no
    timestamps — the merge byte-compares nothing here, but determinism keeps
    re-executions honest)."""
    payload = {
        "schemaVersion": SHARD_SCHEMA,
        "shardIndex": plan.spec.index,
        "shardCount": plan.spec.count,
        "recordRange": [plan.record_range[0], plan.record_range[1]],
        "expectedRecords": len(plan.keys),
        "totalRecords": plan.total_records,
        "expectedKeys": [list(key) for key in plan.keys],
        "ownedConditions": list(plan.owned_conditions),
        "experimentHash": experiment_hash,
    }
    with open(shard_path(run_directory), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_shard_stamp(run_directory: str) -> dict | None:
    try:
        with open(shard_path(run_directory), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# --- the merge -----------------------------------------------------------------

def _parse_records(blob: bytes, *, where: str) -> list[dict]:
    records = []
    for line_number, line in enumerate(blob.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ShardMergeError(
                f"shard merge refused: {where} line {line_number} is not "
                f"valid JSON ({type(exc).__name__}) — the shard partial is "
                "torn; resume or re-run that shard") from exc
    return records


def _key_from_list(raw: list) -> tuple:
    return tuple(raw)


def merge_shard_runs(name: str, shard_dirs: list[str], *, root: str,
                     shard_job_ids: list[str] | None = None,
                     job_id: str | None = None,
                     log=None) -> str:
    """Assemble one ordinary immutable run directory from K shard partials.

    Completeness is proved BEFORE anything is written: every expected cell
    present exactly once across shards (an errored variant condition — one
    error record instead of generations, same as a single-job run — excuses
    its own cells), no cross-shard duplicates, every shard complete. The
    merged ``generations.jsonl`` is the byte concatenation in shard order
    (= global record order), ``report.json`` is rebuilt over the merged
    records through the SAME report-building code the single-job path uses,
    and ``config.json`` (schema 2, closed contract) is written once by the
    merge. Shard provenance goes into ``report.json`` as the ``sharded``
    block only.

    Shard partials are KEPT after a successful merge (deliberate): the runs
    area is immutable by contract and automated deletion of run directories
    is exactly the class of destructive act this instrument avoids — the
    merged report's ``sharded`` block names the partials so a researcher can
    archive or remove them by hand.
    """
    _log = log or (lambda *_: None)
    stamps: list[dict] = []
    for directory in shard_dirs:
        stamp = read_shard_stamp(directory)
        if stamp is None:
            raise ShardMergeError(
                f"shard merge refused: {directory} has no readable "
                f"{SHARD_FILENAME} — not a shard partial")
        stamp["_dir"] = directory
        stamps.append(stamp)
    stamps.sort(key=lambda s: int(s.get("shardIndex", -1)))
    count = len(stamps)
    indices = [int(s.get("shardIndex", -1)) for s in stamps]
    counts = {int(s.get("shardCount", 0)) for s in stamps}
    if counts != {count} or indices != list(range(count)):
        raise ShardMergeError(
            "shard merge refused: shard set is not exactly indices "
            f"0..{count - 1} of a {count}-shard run (have indices {indices}, "
            f"declared counts {sorted(counts)}) — a shard is missing, "
            "duplicated, or from a different fan-out")
    totals = {int(s.get("totalRecords", -1)) for s in stamps}
    hashes = {str(s.get("experimentHash", "")) for s in stamps}
    if len(totals) != 1 or len(hashes) != 1:
        raise ShardMergeError(
            "shard merge refused: shard partials disagree on the run's "
            "record count or experiment hash — they are not shards of the "
            "same run")
    experiment_hash = hashes.pop()

    incomplete = [s for s in stamps if not resume_mod.is_complete(s["_dir"])]
    if incomplete:
        first = incomplete[0]
        raise ShardMergeError(
            f"shard merge refused: shard {first['shardIndex']} "
            f"({first['_dir']}) is not complete — a shard may have failed "
            "or been cancelled; resume the incomplete shard and the merge "
            "will re-run")

    # Parse every partial, prove completeness, and collect the byte streams.
    shard_blobs: list[bytes] = []
    shard_records: list[list[dict]] = []
    key_owner: dict[tuple, int] = {}
    error_conditions: set[str] = set()
    generated_conditions: set[str] = set()
    for stamp in stamps:
        path = os.path.join(stamp["_dir"], "generations.jsonl")
        try:
            with open(path, "rb") as handle:
                blob = handle.read()
        except OSError as exc:
            raise ShardMergeError(
                f"shard merge refused: shard {stamp['shardIndex']} has no "
                f"readable generations.jsonl ({exc})") from exc
        records = _parse_records(
            blob, where=f"shard {stamp['shardIndex']} generations.jsonl")
        shard_blobs.append(blob)
        shard_records.append(records)
        for record in records:
            key = resume_mod.record_key(record)
            if key[-1] == resume_mod.KIND_ERROR:
                error_conditions.add(str(record.get("condition")))
                continue
            generated_conditions.add(str(record.get("condition")))
            if key in key_owner:
                raise ShardMergeError(
                    "shard merge refused: record cell "
                    f"{_describe_key(key)} appears in shard "
                    f"{key_owner[key]} AND shard {stamp['shardIndex']} — "
                    "duplicated cells; the partials are inconsistent")
            key_owner[key] = int(stamp["shardIndex"])

    inconsistent = error_conditions & generated_conditions
    if inconsistent:
        raise ShardMergeError(
            "shard merge refused: condition(s) "
            f"{sorted(inconsistent)} carry BOTH an error record and "
            "generated records across shards — the shards did not agree on "
            "whether the condition ran; re-run the fan-out")

    missing: list[tuple] = []
    unexpected: list[tuple] = []
    for stamp in stamps:
        expected = [_key_from_list(raw)
                    for raw in (stamp.get("expectedKeys") or [])]
        expected_set = set(expected)
        actual = {key for key, owner in key_owner.items()
                  if owner == int(stamp["shardIndex"])}
        missing.extend(
            key for key in expected
            if key not in actual and key[0] not in error_conditions)
        unexpected.extend(sorted(actual - expected_set))
    if missing:
        # Every shard already passed the is_complete gate above, so "resume
        # the incomplete shard" would be impossible advice here — a key that
        # is expected but absent from COMPLETE partials means the fan-out
        # plan and the executor disagreed about what the run emits (the
        # 2026-08-04 case: a scope-blind plan expecting instrument readouts
        # the executor rightly declined). Deterministic fan-out means an
        # unchanged resubmit reproduces this exactly.
        raise ShardMergeError(
            f"shard merge refused: {len(missing)} expected records are "
            f"missing (first: {_describe_key(missing[0])}) although every "
            "shard reports complete — the shard plan expected records the "
            "executor never emits (plan/executor disagreement, not a failed "
            "shard). Resubmitting the same bundle will fail identically; "
            "this needs a code fix or a changed study, not a retry")
    if unexpected:
        raise ShardMergeError(
            f"shard merge refused: {len(unexpected)} records were produced "
            "outside their shard's declared range (first: "
            f"{_describe_key(unexpected[0])}) — the partials are "
            "inconsistent")

    # Heavy imports only after the cheap refusals.
    from . import paths, reasoning_style, tasks
    from .manifest import Manifest

    manifest = Manifest.load(name, root)
    if manifest.content_hash() != experiment_hash:
        raise ShardMergeError(
            "shard merge refused: the live manifest hashes "
            f"{manifest.content_hash()[:12]}… but the shards ran under "
            f"{experiment_hash[:12]}… — the experiment drifted since the "
            "fan-out (epoch guard); the shard partials are untouched")

    merged_records = [record for records in shard_records
                      for record in records]

    # Battery: byte concatenation in shard order + a recomputed summary
    # equivalent to what a single job's in-run battery stamps.
    battery_blobs: list[bytes] = []
    battery_records: list[dict] = []
    battery_condition_owner: dict[str, int] = {}
    for stamp in stamps:
        path = os.path.join(stamp["_dir"], "battery.jsonl")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as handle:
            blob = handle.read()
        records = _parse_records(
            blob, where=f"shard {stamp['shardIndex']} battery.jsonl")
        for record in records:
            condition = str(record.get("condition"))
            prior = battery_condition_owner.get(condition)
            if prior is not None and prior != int(stamp["shardIndex"]):
                raise ShardMergeError(
                    "shard merge refused: capability-battery condition "
                    f"'{condition}' was scored by shard {prior} AND shard "
                    f"{stamp['shardIndex']} — battery ownership must be "
                    "exactly one shard per condition")
            battery_condition_owner[condition] = int(stamp["shardIndex"])
        battery_blobs.append(blob)
        battery_records.extend(records)

    merged_dir = paths.make_unique_run_directory(f"exp-{name}-run", root)
    _log(f"shard merge: assembling {len(merged_records)} records from "
         f"{count} shard(s) → {merged_dir}")

    # Canonical per-run stamps, written ONCE by the merge (config.json is the
    # closed schema-2 contract — no new keys; shard provenance never lands
    # here). `job_id` is the PARENT job's record id, passed explicitly: the
    # merge runs on the controller, whose own SLURM_JOB_ID env would
    # otherwise be stamped — one stale allocation id repeated across every
    # run the same controller incarnation merged (observed 2026-08-06). The
    # per-shard Slurm ids live in report.json's `sharded` block.
    from .manifest import inert_machinery_note
    inert_note = inert_machinery_note(manifest.raw)
    tasks._write_config_snapshot(manifest, merged_dir, "run", job_id=job_id,
                                 notes=({"inertConceptMachinery": inert_note}
                                        if inert_note is not None else None))

    # Deterministic shared artifacts (vectors, substrate.json, …): identical
    # across shards by the determinism claim — verified byte-for-byte, then
    # copied once from shard 0. A mismatch is a real cross-shard
    # nondeterminism signal and refuses the merge.
    _copy_invariant_artifacts(stamps, merged_dir)
    # Per-transcript SUBDIRECTORIES (a panel's <condition>/replicate-N trees).
    # `_copy_invariant_artifacts` walks only files, so before this every
    # sharded panel merge produced a run directory with a complete
    # generations.jsonl and NO transcript layer at all — the whole
    # human-readable half of a panel silently absent, with nothing in any log
    # saying so (2026-08-20 ledger). Shards own disjoint transcripts, so
    # these are collected from EVERY partial, not just shard 0.
    copied = _copy_shard_subdirectories(stamps, merged_dir)
    if copied:
        _log(f"shard merge: carried {copied} transcript file(s) from the "
             "shard partials into the merged run")

    with open(os.path.join(merged_dir, "generations.jsonl"), "wb") as handle:
        for blob in shard_blobs:
            handle.write(blob)
    if battery_blobs:
        with open(os.path.join(merged_dir, "battery.jsonl"), "wb") as handle:
            for blob in battery_blobs:
                handle.write(blob)

    style = reasoning_style.load_pinned(manifest, root)
    numeric_parser = None
    if manifest.numeric_parser:
        from . import parser_registry
        numeric_parser = parser_registry.resolve(manifest.numeric_parser, root)
    battery_summary = (
        _battery_summary(battery_records) if battery_records else None)
    tasks._write_metrics_csv(merged_records, merged_dir, style=style)
    tasks._write_summaries_csv(merged_records, merged_dir)
    # Panel voice lint over the WHOLE matrix. Each shard rolled up only its own
    # transcripts; the per (speaker × condition) rate is only meaningful once
    # every transcript is in hand. Empty (no file) for non-panel runs.
    lint_rows = voice_lint.csv_rows(merged_records)
    if lint_rows:
        voice_lint.write_csv(
            os.path.join(merged_dir, voice_lint.VOICE_LINT_FILENAME), lint_rows)
    sharded_block = {
        "shardCount": count,
        "shardRuns": [os.path.basename(s["_dir"].rstrip(os.sep))
                      for s in stamps],
    }
    if shard_job_ids:
        sharded_block["shardJobIDs"] = list(shard_job_ids)
    tasks._write_report(name, manifest, merged_records, merged_dir,
                        battery=battery_summary, style=style,
                        numeric_parser=numeric_parser,
                        sharded=sharded_block)
    _log(f"shard merge complete → {merged_dir} (shard partials kept: "
         + ", ".join(sharded_block["shardRuns"]) + ")")
    return merged_dir


def _describe_key(key: tuple) -> str:
    condition, _prompt_index, prompt_id, sample_index, kind = key
    parts = [f"condition={condition}", f"promptID={prompt_id}"]
    if sample_index is not None:
        parts.append(f"sampleIndex={sample_index}")
    if kind != resume_mod.KIND_SAMPLED:
        parts.append(f"kind={kind}")
    return "(" + ", ".join(parts) + ")"


#: Sidecar fields that record WHEN or WHERE a vector was derived rather than
#: WHAT it is. Shards extract independently, seconds or minutes apart, so
#: these legitimately differ between them — comparing them made a wall-clock
#: reading look like cross-shard nondeterminism (external review round 3,
#: finding 1). Deliberately an explicit, minimal ALLOWLIST of things known
#: not to bear on the science: everything else still has to match exactly.
_VOLATILE_SIDECAR_FIELDS = frozenset({"extractionDate"})


#: Keys that identify a JSON object as a VECTOR SIDECAR rather than some
#: other artifact that happens to be JSON. All are required fields of
#: `vector_store.SteeringVectorSidecar` (the Swift Codable keys match), so
#: this recognizes a sidecar without depending on the file's name.
_SIDECAR_SIGNATURE = frozenset({
    "modelID", "concept", "stimulusSetHash", "normsPerLayer", "extractionDate",
})


def _is_vector_sidecar(data: object) -> bool:
    """Whether a parsed artifact is a vector sidecar.

    The volatile-field exception is scoped to sidecars deliberately (external
    review round 4, finding 3): sidecars are the artifact shards legitimately
    derive at different wall-clock times. Some FUTURE JSON artifact with an
    `extractionDate` would otherwise inherit an exception nobody reasoned
    about, and a merge that tolerates a difference it should have refused is
    the failure mode this whole check exists to prevent.
    """
    return (isinstance(data, dict)
            and _SIDECAR_SIGNATURE.issubset(data.keys()))


def _comparable_json(payload: bytes) -> tuple[bool, object]:
    """``(is_sidecar, canonical_form)`` for a shared artifact.

    Returns ``(False, ...)`` for anything that is not a vector sidecar — a
    non-JSON artifact (above all the ``.safetensors`` weights, which are the
    thing this check exists to protect), a JSON array, or a JSON object of
    some other kind. Those keep strict byte equality.
    """
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False, None
    if not _is_vector_sidecar(data):
        return False, None
    return True, {k: v for k, v in data.items()
                  if k not in _VOLATILE_SIDECAR_FIELDS}


def _shared_artifacts_agree(payload: bytes, other: bytes) -> bool:
    """Whether two shards' copies of one shared artifact are the same
    ARTIFACT — byte-identical, or vector sidecars that differ only in fields
    recording when the derivation ran."""
    if payload == other:
        return True
    left_is_sidecar, left = _comparable_json(payload)
    right_is_sidecar, right = _comparable_json(other)
    return left_is_sidecar and right_is_sidecar and left == right


def _differing_json_fields(payload: bytes, other: bytes) -> str | None:
    """Comma-joined ``field (value vs value)`` detail for two JSON objects,
    or None when either side is not a JSON object (weights and other binary
    artifacts have no fields to name).

    Exists so the substrate.json refusal can NAME what differs — a merge
    refused because shard 1 ran on a different GPU should say ``gpu (NVIDIA
    A100… vs NVIDIA H100…)``, not leave the researcher diffing files to
    learn whether the nondeterminism is in the science or in the scheduler's
    node placement."""
    try:
        left = json.loads(payload.decode("utf-8"))
        right = json.loads(other.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None

    def _show(value) -> str:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        return text if len(text) <= 40 else text[:36] + "…"

    diffs = [f"{key} ({_show(left.get(key))} vs {_show(right.get(key))})"
             for key in sorted(set(left) | set(right))
             if left.get(key) != right.get(key)]
    return ", ".join(diffs) if diffs else None


def _copy_invariant_artifacts(stamps: list[dict], merged_dir: str) -> None:
    reference = stamps[0]["_dir"]
    for entry in sorted(os.listdir(reference)):
        source = os.path.join(reference, entry)
        if not os.path.isfile(source) or entry in _PER_SHARD_FILES:
            continue
        if entry.endswith(".evidence-bundle.tar.gz"):
            continue
        with open(source, "rb") as handle:
            payload = handle.read()
        for stamp in stamps[1:]:
            other = os.path.join(stamp["_dir"], entry)
            if not os.path.isfile(other):
                raise ShardMergeError(
                    f"shard merge refused: shard {stamp['shardIndex']} lacks "
                    f"the shared artifact '{entry}' that shard 0 produced — "
                    "the shards did not derive identical run inputs")
            with open(other, "rb") as other_handle:
                other_payload = other_handle.read()
            if not _shared_artifacts_agree(payload, other_payload):
                fields = _differing_json_fields(payload, other_payload)
                raise ShardMergeError(
                    f"shard merge refused: shared artifact '{entry}' "
                    f"differs between shard 0 and shard "
                    f"{stamp['shardIndex']}"
                    + (f" in {fields}" if fields else "")
                    + " — cross-shard nondeterminism; do not trust these "
                    "partials")
        shutil.copyfile(source, os.path.join(merged_dir, entry))


def _copy_shard_subdirectories(stamps: list[dict], merged_dir: str) -> int:
    """Carry every shard partial's SUBDIRECTORY tree into the merged run.

    A panel shards over transcripts, and a transcript's artifacts
    (``turns.jsonl``, ``report.json``, ``transcript.md``) live in
    ``<condition>/replicate-N/``. Shards own disjoint transcripts, so the
    union across partials is exactly the merged run's transcript layer — no
    shard is authoritative for it the way shard 0 is for the invariant files
    above, which is why this is a separate pass.

    Same honesty rule as :func:`_copy_invariant_artifacts`: two shards
    claiming the SAME relative path must agree byte for byte, or the merge
    refuses rather than picking a winner. Returns the number of files copied.
    """
    copied = 0
    # Digests, not payloads: a panel's transcripts are the bulkiest thing in a
    # run directory and the merge has no reason to hold them all at once.
    written: dict[str, tuple[int, str]] = {}
    for stamp in stamps:
        source_root = stamp["_dir"]
        for entry in sorted(os.listdir(source_root)):
            source = os.path.join(source_root, entry)
            if not os.path.isdir(source):
                continue
            for dirpath, _dirs, filenames in os.walk(source):
                for filename in sorted(filenames):
                    path = os.path.join(dirpath, filename)
                    relative = os.path.relpath(path, source_root)
                    digest = _sha256_file(path)
                    prior = written.get(relative)
                    if prior is not None:
                        if prior[1] == digest:
                            continue
                        raise ShardMergeError(
                            f"shard merge refused: '{relative}' was produced "
                            f"by shard {prior[0]} AND shard "
                            f"{stamp['shardIndex']} with different contents "
                            "— the partials disagree about a transcript that "
                            "exactly one shard should own")
                    destination = os.path.join(merged_dir, relative)
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    shutil.copyfile(path, destination)
                    written[relative] = (int(stamp["shardIndex"]), digest)
                    copied += 1
    return copied


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _battery_summary(records: list[dict]) -> dict:
    """Recompute the per-condition capability-battery summary from merged
    battery records — the same three cross-engine keys the in-run battery
    stamps ({"accuracy", "itemCount", "batteryHash"})."""
    by_condition: dict[str, list[dict]] = {}
    for record in records:
        by_condition.setdefault(str(record.get("condition")), []).append(record)
    return {
        condition: {
            "accuracy": (sum(1 for r in items if r.get("correct"))
                         / len(items)) if items else 0.0,
            "itemCount": len(items),
            "batteryHash": next((r.get("batteryHash") for r in items
                                 if r.get("batteryHash")), None),
        }
        for condition, items in sorted(by_condition.items())
    }


# --- pipeline continuation seeding ---------------------------------------------

def seed_pipeline_directory(name: str, root: str, merged_run_dir: str,
                            *, job_id: str | None = None,
                            log=None) -> tuple[str, list[str]]:
    """Create a pipeline run directory whose ledger records the ``run`` stage
    as completed at the MERGED run, so the existing pipeline resume machinery
    (`tasks.pipeline` with a resume pointer) executes only the remaining
    stages (evaluate/analyze) — the merged run flows into the normal
    continuation exactly as a single-job pipeline's own run stage would.

    Returns ``(pipeline_dir, remaining_stages)``. When no stages remain the
    ledger is stamped ``disposition: "completed"`` and the caller needs no
    continuation job."""
    _log = log or (lambda *_: None)
    from . import paths, pipeline_spec, tasks
    from .manifest import Manifest
    manifest = Manifest.load(name, root)
    raw_block = manifest.raw.get("pipeline")
    if raw_block is None:
        raise ShardMergeError(
            f"experiment '{name}' declares no pipeline block — cannot seed "
            "a pipeline continuation")
    spec = pipeline_spec.resolve_pipeline(raw_block)
    if not spec.stages or spec.stages[0] != "run":
        raise ShardMergeError(
            "pipeline continuation requires a chain whose FIRST stage is "
            f"'run' (declared: {', '.join(spec.stages)})")
    pipeline_dir = paths.make_unique_run_directory(
        f"exp-{name}-pipeline", root)
    # Explicit job id for the same reason as the shard merge: this directory
    # is created on the controller, not inside the job that will run the
    # remaining stages, so the env fallback would stamp the controller's
    # own allocation id.
    tasks._write_config_snapshot(manifest, pipeline_dir, "pipeline",
                                 job_id=job_id)
    remaining = list(spec.stages[1:])
    ledger = {
        "schema": tasks.PIPELINE_LEDGER_SCHEMA,
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "manifestStatus": manifest.status,
        "stages": list(spec.stages),
        "stageResults": {
            "run": {"status": "completed", "runDirectory": merged_run_dir},
        },
        "disposition": ("completed" if not remaining else None),
    }
    tasks._write_pipeline_ledger(pipeline_dir, ledger)
    _log(f"pipeline continuation seeded → {pipeline_dir} "
         f"(run stage adopted from {merged_run_dir}; remaining: "
         + (", ".join(remaining) if remaining else "none") + ")")
    return pipeline_dir, remaining

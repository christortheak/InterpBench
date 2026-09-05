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
- The merged run appears ATOMICALLY: it is assembled under
  ``runs/.merge-staging/<name>/`` and renamed onto its reserved name only
  once every file (``report.json`` last) is in place, so no reader ever sees
  a run-shaped directory with records and no report. A merged run found in
  that state anyway (an interrupted pre-staging merge) is COMPLETED in place
  by the reconciler — adding only the missing files, never rewriting one
  (``complete_merged_run``).

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
import time
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


#: Where the merge ASSEMBLES a run before it becomes visible: a hidden
#: sibling of the run directories, ``runs/.merge-staging/<final name>/``. Every
#: reader of ``runs/`` — the app's catalog, ``_latest_run``, the Mac's
#: ``cluster import``, a researcher's ``ls`` — sees a merged run only once
#: every one of its files is in place, because the last step of a merge is one
#: atomic ``rename(2)`` of the finished staging directory onto its reserved
#: name. Before 2026-09-05 the merge wrote straight into the final directory,
#: and a controller ``scancel`` (SIGTERM, no handler) 30 s into a merge left a
#: parent with a complete ``generations.jsonl`` and no ``report.json`` that
#: every reader took for a run.
#:
#: The staging directory carries the FINAL basename (``config.json`` stamps
#: ``runId`` from it), which is why the final name is reserved first.
MERGE_STAGING_DIRNAME = ".merge-staging"

#: A staging directory older than this is the residue of a merge whose process
#: died mid-assembly. Generous on purpose: a live panel merge copies
#: transcripts for minutes at most, and the sweep only ever runs from a later
#: merge or reconcile pass — never against a merge in flight in this process,
#: which the reconciler's merge lock serialises.
STALE_STAGING_AGE_SECONDS = 3600.0


@dataclass
class _ProvenShards:
    """A shard set that has passed every completeness proof — the only input
    the write phase (and the in-place completion of a half-written merge) is
    allowed to consume. Everything here is derived from the partials and the
    live manifest; nothing is written to produce it."""
    stamps: list[dict]
    count: int
    experiment_hash: str
    #: Each shard's ``generations.jsonl`` bytes, in shard order — the merged
    #: file is exactly their concatenation.
    shard_blobs: list[bytes]
    merged_records: list[dict]
    battery_blobs: list[bytes]
    battery_records: list[dict]
    manifest: object

    @property
    def shard_runs(self) -> list[str]:
        return [os.path.basename(s["_dir"].rstrip(os.sep)) for s in self.stamps]

    def sharded_block(self, shard_job_ids: list[str] | None) -> dict:
        block = {"shardCount": self.count, "shardRuns": self.shard_runs}
        if shard_job_ids:
            block["shardJobIDs"] = list(shard_job_ids)
        return block


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

    The assembly is ATOMIC from a reader's point of view: the final run name
    is reserved (an empty directory, the same ``mkdir(2)`` exclusivity every
    run relies on), every file is written into
    ``runs/.merge-staging/<that name>/``, and the finished staging directory
    is renamed onto the reservation in one step. A process that dies mid-way
    leaves a hidden staging directory and an empty reservation — never a
    run-shaped directory with records and no report — and a later merge or
    reconcile pass sweeps both (:func:`sweep_stale_merge_staging`). A merge
    that REFUSES after the reservation (a shared-artifact mismatch) discards
    its own staging and the empty reservation itself.

    Shard partials are KEPT after a successful merge (deliberate): the runs
    area is immutable by contract and automated deletion of run directories
    is exactly the class of destructive act this instrument avoids — the
    merged report's ``sharded`` block names the partials so a researcher can
    archive or remove them by hand.
    """
    _log = log or (lambda *_: None)
    proven = _prove_shards(name, shard_dirs, root=root)
    sweep_stale_merge_staging(root, log=_log)

    from . import paths
    # Reserve the NAME first — config.json stamps ``runId`` from the
    # directory's basename, so the staging twin must carry the final name.
    merged_dir = paths.make_unique_run_directory(f"exp-{name}-run", root)
    staging = _staging_path_for(merged_dir)
    try:
        os.makedirs(os.path.dirname(staging), exist_ok=True)
        os.mkdir(staging)
    except OSError:
        _rmdir_if_empty(merged_dir)
        raise
    _log(f"shard merge: assembling {len(proven.merged_records)} records from "
         f"{proven.count} shard(s) → {merged_dir}")
    try:
        _assemble(name, proven, staging, root=root, shard_job_ids=shard_job_ids,
                  job_id=job_id, log=_log)
        # One atomic step: rename(2) replaces the EMPTY reservation with the
        # finished tree. A non-empty target would refuse (ENOTEMPTY), which
        # is the immutability rule enforced by the kernel.
        os.rename(staging, merged_dir)
    except BaseException:
        # SystemExit/KeyboardInterrupt included: whatever ends this merge
        # must not leave a run-shaped directory behind. A SIGKILL cannot be
        # caught; that case is the sweep's.
        shutil.rmtree(staging, ignore_errors=True)
        _rmdir_if_empty(merged_dir)
        raise
    _log(f"shard merge complete → {merged_dir} (shard partials kept: "
         + ", ".join(proven.shard_runs) + ")")
    return merged_dir


def complete_merged_run(name: str, merged_dir: str, shard_dirs: list[str], *,
                        root: str, shard_job_ids: list[str] | None = None,
                        log=None) -> list[str]:
    """Finish a merged run directory that an interrupted merge left without
    its derived files — above all ``report.json``, the completion marker.

    This is the repair for a directory the PRE-staging merge half-wrote
    (records in, report never written), and for any later loss of a merged
    run's report. It re-proves the shard set exactly as a merge does, then
    checks that the byte streams already in the directory are EXACTLY what a
    merge writes (``generations.jsonl`` == the shard concatenation, and
    ``battery.jsonl`` likewise when present): completing a torn or foreign
    record stream would certify records the shards never produced, so that
    refuses. The derived files are rebuilt through the same writers the merge
    uses, into a hidden staging twin, and ONLY the ones the directory lacks
    are added — by ``link(2)``, which fails rather than replaces — with
    ``report.json`` last, so a reader that sees the marker sees everything
    else. Any already-present derived file must match the rebuilt bytes, or
    the completion refuses before adding anything. Nothing in the run
    directory is ever rewritten.

    Returns the basenames added (empty when the directory was already
    complete). Raises :class:`ShardMergeError` when it cannot be completed in
    place; the shard partials are untouched either way.
    """
    _log = log or (lambda *_: None)
    if resume_mod.is_complete(merged_dir):
        return []
    proven = _prove_shards(name, shard_dirs, root=root)
    _verify_stream(os.path.join(merged_dir, "generations.jsonl"),
                   b"".join(proven.shard_blobs), what="generations.jsonl",
                   merged_dir=merged_dir, required=True)
    battery_path = os.path.join(merged_dir, "battery.jsonl")
    if proven.battery_blobs:
        _verify_stream(battery_path, b"".join(proven.battery_blobs),
                       what="battery.jsonl", merged_dir=merged_dir,
                       required=False)
    elif os.path.exists(battery_path):
        raise ShardMergeError(
            f"merge completion refused: {merged_dir} holds a battery.jsonl "
            "but no shard partial carries battery records — the directory "
            "is not the merge of these shards")

    sweep_stale_merge_staging(root, log=_log)
    staging = _staging_path_for(merged_dir)
    os.makedirs(os.path.dirname(staging), exist_ok=True)
    os.mkdir(staging)
    added: list[str] = []
    try:
        if proven.battery_blobs and not os.path.exists(battery_path):
            with open(os.path.join(staging, "battery.jsonl"), "wb") as handle:
                for blob in proven.battery_blobs:
                    handle.write(blob)
        _write_derived_artifacts(name, proven, staging, root=root,
                                 sharded_block=proven.sharded_block(shard_job_ids))
        staged = sorted(
            os.listdir(staging),
            # The completion marker goes in LAST.
            key=lambda entry: (entry == resume_mod.COMPLETION_FILENAME, entry))
        # Refuse BEFORE adding anything: every derived file already present
        # must be byte-identical to what the shards rebuild.
        for entry in staged:
            target = os.path.join(merged_dir, entry)
            if os.path.exists(target) and not _same_bytes(
                    target, os.path.join(staging, entry)):
                raise ShardMergeError(
                    f"merge completion refused: {merged_dir} already holds a "
                    f"'{entry}' that differs from the one the shard partials "
                    "rebuild — the directory is torn or foreign and cannot be "
                    "completed in place; the shard partials are intact")
        for entry in staged:
            target = os.path.join(merged_dir, entry)
            if os.path.exists(target):
                continue
            if _add_never_replace(os.path.join(staging, entry), target):
                added.append(entry)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    _log(f"merge completion: added {', '.join(added) or 'nothing'} to "
         f"{merged_dir} (rebuilt from {proven.count} shard partial(s))")
    return added


def sweep_stale_merge_staging(root: str, *, log=None,
                              min_age_seconds: float | None = None) -> list[str]:
    """Remove the residue of merges whose process died mid-assembly: entries
    under ``runs/.merge-staging/`` older than ``min_age_seconds`` (default
    :data:`STALE_STAGING_AGE_SECONDS`), plus the EMPTY reservation each one
    left under ``runs/`` — ``rmdir(2)`` refuses anything but an empty
    directory, so a populated twin is never touched. A staging directory is
    scratch this module created and never a run; the reservation is a name
    with nothing behind it. Returns the staging basenames swept."""
    _log = log or (lambda *_: None)
    age_floor = (STALE_STAGING_AGE_SECONDS if min_age_seconds is None
                 else min_age_seconds)
    from . import paths
    runs_root = paths.runs_directory(root)
    staging_root = os.path.join(runs_root, MERGE_STAGING_DIRNAME)
    try:
        entries = sorted(os.listdir(staging_root))
    except OSError:
        return []
    now = time.time()
    swept: list[str] = []
    for entry in entries:
        path = os.path.join(staging_root, entry)
        try:
            age = now - os.stat(path).st_mtime
        except OSError:
            continue
        if age < age_floor:
            continue
        twin = os.path.join(runs_root, entry)
        if not os.path.isdir(twin):
            twin_note = "no reservation under runs/"
        elif _rmdir_if_empty(twin):
            twin_note = "removed its empty reservation under runs/"
        else:
            twin_note = f"runs/{entry} is populated and was left alone"
        shutil.rmtree(path, ignore_errors=True)
        swept.append(entry)
        _log(f"swept stale merge staging '{entry}' ({int(age)} s old — an "
             f"interrupted merge's scratch); {twin_note}")
    return swept


def find_merged_runs_for_job(name: str, root: str,
                             job_id: str) -> list[tuple[str, bool]]:
    """The merged runs a given parent record has already produced, complete or
    not: run-shaped directories (``<stamp>-exp-<name>-run``, with or without a
    collision suffix) under ``runs/`` whose ``config.json`` stamps ``jobId``
    equal to ``job_id`` — the merge writes the PARENT's record id there, and
    nothing else does. Newest first, as ``(path, is_complete)``.

    The reconciler consults this before merging so a controller that died
    after assembling a run — before recording it on the parent (the window
    the evidence packaging widens), or, under the pre-staging merge, before
    writing the report — adopts or finishes its own earlier directory rather
    than merging again beside it and leaving the first as an orphan."""
    from . import paths
    runs_root = paths.runs_directory(root)
    try:
        entries = os.listdir(runs_root)
    except OSError:
        return []
    suffix = f"-exp-{name}-run"
    found: list[tuple[str, bool]] = []
    for entry in entries:
        if entry.startswith("."):
            continue
        stem = entry
        head, dash, tail = entry.rpartition("-")
        if dash and tail.isdigit() and head.endswith(suffix):
            stem = head  # ``…-run-2``: a same-millisecond collision suffix
        if not stem.endswith(suffix):
            continue
        path = os.path.join(runs_root, entry)
        if not os.path.isdir(path):
            continue
        try:
            with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict) or config.get("jobId") != job_id:
            continue
        found.append((path, resume_mod.is_complete(path)))
    found.sort(key=lambda item: os.path.basename(item[0]), reverse=True)
    return found


def _prove_shards(name: str, shard_dirs: list[str], *, root: str) -> _ProvenShards:
    """Every refusal a merge can make, and none of its writes."""
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

    return _ProvenShards(
        stamps=stamps, count=count, experiment_hash=experiment_hash,
        shard_blobs=shard_blobs, merged_records=merged_records,
        battery_blobs=battery_blobs, battery_records=battery_records,
        manifest=manifest)


def _assemble(name: str, proven: _ProvenShards, into: str, *, root: str,
              shard_job_ids: list[str] | None, job_id: str | None,
              log) -> None:
    """Write a complete merged run into ``into`` (a staging directory that
    already carries the final basename). Order matters only in that the
    derived files come last; nothing outside ``into`` is touched."""
    from . import tasks

    # Canonical per-run stamps, written ONCE by the merge (config.json is the
    # closed schema-2 contract — no new keys; shard provenance never lands
    # here). `job_id` is the PARENT job's record id, passed explicitly: the
    # merge runs on the controller, whose own SLURM_JOB_ID env would
    # otherwise be stamped — one stale allocation id repeated across every
    # run the same controller incarnation merged (observed 2026-08-06). The
    # per-shard Slurm ids live in report.json's `sharded` block.
    from .manifest import inert_machinery_note
    inert_note = inert_machinery_note(proven.manifest.raw)
    tasks._write_config_snapshot(proven.manifest, into, "run", job_id=job_id,
                                 notes=({"inertConceptMachinery": inert_note}
                                        if inert_note is not None else None))

    # Deterministic shared artifacts (vectors, substrate.json, …): identical
    # across shards by the determinism claim — verified byte-for-byte, then
    # copied once from shard 0. A mismatch is a real cross-shard
    # nondeterminism signal and refuses the merge.
    _copy_invariant_artifacts(proven.stamps, into)
    # Per-transcript SUBDIRECTORIES (a panel's <condition>/replicate-N trees).
    # `_copy_invariant_artifacts` walks only files, so before this every
    # sharded panel merge produced a run directory with a complete
    # generations.jsonl and NO transcript layer at all — the whole
    # human-readable half of a panel silently absent, with nothing in any log
    # saying so (2026-08-20 ledger). Shards own disjoint transcripts, so
    # these are collected from EVERY partial, not just shard 0.
    copied = _copy_shard_subdirectories(proven.stamps, into)
    if copied:
        log(f"shard merge: carried {copied} transcript file(s) from the "
            "shard partials into the merged run")

    with open(os.path.join(into, "generations.jsonl"), "wb") as handle:
        for blob in proven.shard_blobs:
            handle.write(blob)
    if proven.battery_blobs:
        with open(os.path.join(into, "battery.jsonl"), "wb") as handle:
            for blob in proven.battery_blobs:
                handle.write(blob)
    _write_derived_artifacts(name, proven, into, root=root,
                             sharded_block=proven.sharded_block(shard_job_ids))


def _write_derived_artifacts(name: str, proven: _ProvenShards, into: str, *,
                             root: str, sharded_block: dict) -> None:
    """Everything a merged run derives from its records — metrics, summaries,
    the panel voice lint, and last of all ``report.json`` — through the SAME
    writers the single-job path uses. Shared by the merge and by the in-place
    completion of a half-written merge, so the two can never disagree about
    what a complete merged run contains."""
    from . import reasoning_style, tasks

    style = reasoning_style.load_pinned(proven.manifest, root)
    numeric_parser = None
    if proven.manifest.numeric_parser:
        from . import parser_registry
        numeric_parser = parser_registry.resolve(proven.manifest.numeric_parser,
                                                 root)
    battery_summary = (
        _battery_summary(proven.battery_records) if proven.battery_records
        else None)
    tasks._write_metrics_csv(proven.merged_records, into, style=style)
    tasks._write_summaries_csv(proven.merged_records, into)
    # Panel voice lint over the WHOLE matrix. Each shard rolled up only its own
    # transcripts; the per (speaker × condition) rate is only meaningful once
    # every transcript is in hand. Empty (no file) for non-panel runs.
    lint_rows = voice_lint.csv_rows(proven.merged_records)
    if lint_rows:
        voice_lint.write_csv(
            os.path.join(into, voice_lint.VOICE_LINT_FILENAME), lint_rows)
    tasks._write_report(name, proven.manifest, proven.merged_records, into,
                        battery=battery_summary, style=style,
                        numeric_parser=numeric_parser,
                        sharded=sharded_block)


def _staging_path_for(merged_dir: str) -> str:
    """``runs/.merge-staging/<basename of merged_dir>``."""
    normalized = os.path.normpath(merged_dir)
    return os.path.join(os.path.dirname(normalized), MERGE_STAGING_DIRNAME,
                        os.path.basename(normalized))


def _rmdir_if_empty(path: str) -> bool:
    """``rmdir(2)``: removes an EMPTY directory and refuses anything else —
    the one deletion this module performs against ``runs/`` itself, and one
    that by construction can never take a record with it."""
    try:
        os.rmdir(path)
    except OSError:
        return False
    return True


def _verify_stream(path: str, expected: bytes, *, what: str, merged_dir: str,
                   required: bool) -> None:
    try:
        with open(path, "rb") as handle:
            actual = handle.read()
    except FileNotFoundError:
        if required:
            raise ShardMergeError(
                f"merge completion refused: {merged_dir} has no {what} — "
                "there is nothing to complete; assemble a fresh merge instead")
        return
    if actual != expected:
        raise ShardMergeError(
            f"merge completion refused: {merged_dir}/{what} is not the byte "
            f"concatenation of the shard partials ({len(actual)} B on disk vs "
            f"{len(expected)} B rebuilt) — the stream is torn or foreign, and "
            "completing it would certify records the shards never produced; "
            "the shard partials are intact")


def _same_bytes(left: str, right: str) -> bool:
    with open(left, "rb") as a, open(right, "rb") as b:
        return a.read() == b.read()


def _add_never_replace(source: str, target: str) -> bool:
    """Move ``source`` to ``target`` without ever replacing an existing file:
    ``link(2)`` fails with EEXIST where ``rename(2)`` would silently clobber.
    Falls back to rename (after an existence check) only on a filesystem
    that refuses hard links. Returns whether the file was added."""
    try:
        os.link(source, target)
    except FileExistsError:
        return False
    except OSError:
        if os.path.exists(target):
            return False
        os.rename(source, target)
        return True
    os.unlink(source)
    return True


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

"""Run-bundle and evidence-bundle packaging.

Run bundles are the handoff unit for remote compute: a manifest plus the files
it pins. Evidence bundles are the return unit: one immutable run directory plus
a hash manifest so local clients can import and verify results.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Iterable

from . import experiment_store, paths
from . import resume as resume_mod
from . import run_status
from .manifest import Manifest


BUNDLE_SCHEMA = 1

#: How much of one archive member is held in memory at a time while it is
#: hashed and written. Big enough that a multi-GB safetensors member is not
#: read a page at a time, small enough to be irrelevant next to a model load.
MEMBER_CHUNK_BYTES = 4 * 1024 * 1024


def max_member_bytes() -> int:
    """Upper bound on ONE member's UNCOMPRESSED size, in bytes.

    The upload cap (``routes._max_upload_bytes``, ``STEERLAB_MAX_UPLOAD_BYTES``,
    4 GiB) counts COMPRESSED bytes, which says nothing about what an archive
    expands to: a gzip member of highly repetitive bytes expands by three
    orders of magnitude, so a bundle well inside that cap can carry a member
    far larger than the machine's RAM. This is the uncompressed companion —
    generous (a sharded safetensors member is GBs, not tens of GBs) but
    bounded, so a hostile or simply broken bundle is a REPORTED refusal
    instead of an OOM kill of the whole server process.

    Read per call and tunable the same way its compressed sibling is:
    ``STEERLAB_MAX_BUNDLE_MEMBER_BYTES``; default 16 GiB.
    """
    return int(os.environ.get("STEERLAB_MAX_BUNDLE_MEMBER_BYTES",
                              str(16 * 1024**3)))


def max_total_bytes() -> int:
    """Upper bound on what the WHOLE archive expands to, in bytes.

    :func:`max_member_bytes` bounds ONE member, which is exactly the wrong
    shape for the cheapest expansion attack there is: ten thousand members of
    15 GiB each are individually inside a 16-GiB per-member cap and together
    fill any disk on earth. A cap that is only per-member is a cap on the
    largest brick, not on the wall.

    Checked twice, because a tar header is a claim and the sum of claims is
    still only a claim: against the DECLARED sizes in preflight (so a hostile
    archive is refused before a byte is written) and against the ACTUAL bytes
    while members stream (so understated headers are caught on the way past).

    Generous by design — a sharded multi-GB study bundle must import without
    anyone touching a knob. Read per call and tunable beside its siblings:
    ``STEERLAB_MAX_BUNDLE_TOTAL_BYTES``; default 64 GiB.
    """
    return int(os.environ.get("STEERLAB_MAX_BUNDLE_TOTAL_BYTES",
                              str(64 * 1024**3)))


def max_member_count() -> int:
    """Upper bound on HOW MANY members one archive may carry.

    The companion to :func:`max_total_bytes`, and not redundant with it: a
    million empty members costs almost no uncompressed bytes and still spends
    a million inode creations, a million path walks, and however long the
    metadata pass takes. Members that this importer SKIPS (``._`` resource
    forks, ``.DS_Store``) count too — they are skipped after the tar index has
    already been read, so they are work whether or not they land.

    Read per call and tunable beside its siblings:
    ``STEERLAB_MAX_BUNDLE_MEMBERS``; default 10000.
    """
    return int(os.environ.get("STEERLAB_MAX_BUNDLE_MEMBERS", "10000"))


def max_metadata_bytes() -> int:
    """The same bound for the bundle's own JSON documents
    (``steerlab-bundle.json``, ``steerlab-evidence.json``,
    ``steerlab-pipeline.json``), which ARE parsed whole because they must be.
    A manifest of hash entries is measured in MB even for a large study.
    Override with ``STEERLAB_MAX_BUNDLE_METADATA_BYTES``; default 64 MiB.
    """
    return int(os.environ.get("STEERLAB_MAX_BUNDLE_METADATA_BYTES",
                              str(64 * 1024**2)))


class BundleError(Exception):
    pass


def _refuse_oversized_member(member, limit: int, *, what: str = "member") -> None:
    """Refuse on the DECLARED size, before a single byte is read."""
    if member.size > limit:
        raise BundleError(
            f"bundle {what} {member.name!r} declares {member.size} uncompressed "
            f"bytes, over the {limit}-byte limit "
            "(STEERLAB_MAX_BUNDLE_MEMBER_BYTES / "
            "STEERLAB_MAX_BUNDLE_METADATA_BYTES) — refusing to expand it")


def _refuse_oversized_archive(members) -> None:
    """The AGGREGATE bounds, on the DECLARED archive, before anything lands.

    Per-member caps answer "is this one brick too big"; these answer "is this
    wall too big", which is the question a bundle of ten thousand
    just-under-the-cap members exists to dodge.
    """
    count_limit = max_member_count()
    if len(members) > count_limit:
        raise BundleError(
            f"bundle carries {len(members)} members, over the "
            f"{count_limit}-member limit (STEERLAB_MAX_BUNDLE_MEMBERS) — "
            "refusing to expand it")
    total_limit = max_total_bytes()
    declared = sum(member.size for member in members if member.isfile())
    if declared > total_limit:
        raise BundleError(
            f"bundle declares {declared} uncompressed bytes across "
            f"{len(members)} member(s), over the {total_limit}-byte limit "
            "(STEERLAB_MAX_BUNDLE_TOTAL_BYTES) — refusing to expand it")


def _read_metadata_member(tar, member) -> bytes:
    """Whole-member read for a bundle's own JSON, with the size refusal in
    front of it (these must be parsed as one document, so they cannot stream)."""
    _refuse_oversized_member(member, max_metadata_bytes(), what="metadata member")
    handle = tar.extractfile(member)
    if handle is None:
        return b""
    with handle:
        return handle.read()


def _stream_member(tar, member, temp_path: str, limit: int, *,
                   total_before: int = 0,
                   total_limit: int | None = None) -> tuple[str, int]:
    """Copy one member to ``temp_path`` in chunks; return ``(sha256, bytes)``.

    The member is never materialized: ``source.read()`` with no argument
    pulled the WHOLE thing into memory, which the 4-GiB compressed upload cap
    does not bound (see :func:`max_member_bytes`). Hashing as the bytes go by
    preserves the pre-landing integrity check exactly — the digest is of what
    the tar actually carried, computed before ``dest`` is touched, so the
    tamper firewall neither weakens nor starts racing a sibling shard.

    The declared size was already refused above; this re-checks the ACTUAL
    byte count, because a tar header is just a claim. ``total_before`` /
    ``total_limit`` carry the same re-check for the WHOLE archive
    (:func:`max_total_bytes`): the declared sum was refused in preflight, and
    this is the running total that catches headers which understated it.
    """
    digest = hashlib.sha256()
    written = 0
    source = tar.extractfile(member)
    if source is None:  # pragma: no cover - callers filter these out first
        return digest.hexdigest(), 0
    with source, open(temp_path, "wb") as handle:
        while True:
            chunk = source.read(MEMBER_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise BundleError(
                    f"bundle member {member.name!r} expands past the "
                    f"{limit}-byte limit (STEERLAB_MAX_BUNDLE_MEMBER_BYTES) — "
                    "its tar header understated its size; refusing")
            if total_limit is not None \
                    and total_before + written > total_limit:
                raise BundleError(
                    f"bundle expands past the {total_limit}-byte total limit "
                    f"(STEERLAB_MAX_BUNDLE_TOTAL_BYTES) while streaming "
                    f"{member.name!r} — its tar headers understated the "
                    "archive; refusing")
            digest.update(chunk)
            handle.write(chunk)
    return digest.hexdigest(), written


@dataclass
class BundleEntry:
    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


def package_experiment(name: str, *, output_path: str | None = None,
                       root: str | None = None) -> dict:
    manifest = Manifest.load(name, root)
    base = paths.project_root() if root is None else root
    files = _experiment_files(manifest, base, root)
    if output_path is None:
        out_dir = paths.make_unique_run_directory(f"bundle-{name}", root)
        output_path = os.path.join(out_dir, f"{name}.run-bundle.tar.gz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    meta = {
        "schemaVersion": BUNDLE_SCHEMA,
        "kind": "runBundle",
        "createdAt": time.time(),
        "experiment": name,
        "experimentContentHash": manifest.content_hash(),
        "validationScopeHash": manifest.validation_scope_hash(),
        "rootRelative": True,
        "verificationViolations": manifest.verify(root),
        "entries": [],
    }
    with tarfile.open(output_path, "w:gz") as tar:
        entries = _add_files(tar, files)
        meta["entries"] = [entry.to_dict() for entry in entries]
        _add_json(tar, "steerlab-bundle.json", meta)
    meta["bundlePath"] = output_path
    meta["bundleSha256"] = sha256_file(output_path)
    return meta


def _pipeline_evidence_directories(run_dir: str):
    """Stage run directories a pipeline ledger references — the ACTUAL
    evidence (vectors, validation report, sweep results, generations,
    judgments, analysis) lives in SIBLING run dirs; the pipeline dir holds
    only the ledger and pointers, so packaging it alone would ship
    references to absolute cluster paths and nothing they name (engineer
    review 2026-07-18, second round). Promote's variant artifacts are
    included via their run directories too.

    Returns ``(directories, missing, disposition)`` or None when the dir
    holds no pipeline ledger. CONTAINMENT (third round): only directories
    under the same runs root as the pipeline dir are followed — a tampered
    ledger must not make packaging traverse arbitrary readable paths; an
    out-of-root or absent reference lands in ``missing`` (the caller
    decides whether that refuses or is stamped incomplete)."""
    ledger_path = os.path.join(run_dir, "pipeline.json")
    if not os.path.isfile(ledger_path):
        return None
    try:
        with open(ledger_path, encoding="utf-8") as handle:
            ledger = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"pipeline.json is unreadable ({type(exc).__name__})"], None
    if not isinstance(ledger, dict):
        return [], ["pipeline.json is not an object"], None
    runs_root = os.path.realpath(os.path.dirname(run_dir.rstrip(os.sep)))
    directories: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    def _add(candidate, what: str) -> None:
        if not isinstance(candidate, str) or not candidate:
            return
        resolved = os.path.realpath(candidate)
        if not resolved.startswith(runs_root + os.sep):
            missing.append(f"{what}: '{candidate}' is outside the runs root")
            return
        if not os.path.isdir(resolved):
            missing.append(f"{what}: '{candidate}' not found")
            return
        if resolved in seen:
            return
        seen.add(resolved)
        directories.append(resolved)

    for stage, entry in (ledger.get("stageResults") or {}).items():
        entry = entry if isinstance(entry, dict) else {}
        _add(entry.get("runDirectory"), f"stage '{stage}'")
        for concept, pin in (entry.get("concepts") or {}).items():
            # Full pin dicts ({path, hash, sweepRun, winningCell}) since
            # the fourth review round; bare path strings on older ledgers.
            artifact_path = (pin.get("path") if isinstance(pin, dict)
                             else pin)
            if isinstance(artifact_path, str) and artifact_path:
                _add(os.path.dirname(artifact_path),
                     f"promoted agent '{concept}'")
    return directories, missing, ledger.get("disposition")


def _portable_pipeline_ledger(run_dir: str) -> dict | None:
    """A PORTABLE projection of the pipeline ledger for the evidence bundle
    (stage 5, engineer review 2026-07-18 third round): the on-disk ledger
    references absolute cluster paths, which are meaningless after import
    to the Mac. This projection references everything by imported RUN ID
    (the ``runs/<id>/`` prefix the bundle itself uses) and keeps the
    original paths as provenance. The on-disk ledger is never rewritten —
    the run directory is immutable."""
    ledger_path = os.path.join(run_dir, "pipeline.json")
    if not os.path.isfile(ledger_path):
        return None
    try:
        with open(ledger_path, encoding="utf-8") as handle:
            ledger = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(ledger, dict):
        return None
    stage_runs: dict = {}
    original_paths: dict = {}
    promoted: dict = {}
    for stage, entry in (ledger.get("stageResults") or {}).items():
        entry = entry if isinstance(entry, dict) else {}
        stage_dir = entry.get("runDirectory")
        if isinstance(stage_dir, str) and stage_dir:
            stage_runs[stage] = os.path.basename(stage_dir.rstrip(os.sep))
            original_paths[stage] = stage_dir
        for concept, pin in (entry.get("concepts") or {}).items():
            path = pin.get("path") if isinstance(pin, dict) else pin
            if not isinstance(path, str) or not path:
                continue
            record = {
                "runID": os.path.basename(os.path.dirname(path)),
                "artifact": os.path.basename(path)}
            if isinstance(pin, dict):
                for key in ("hash", "sweepRun", "winningCell"):
                    if pin.get(key) is not None:
                        record[key] = pin[key]
            promoted[concept] = record
            original_paths[f"agent:{concept}"] = path
    stage_results = ledger.get("stageResults") or {}
    stage_status = {
        stage: ((stage_results.get(stage) or {}).get("status") or "pending")
        for stage in (ledger.get("stages") or [])}
    portable: dict = {
        "schema": 1,
        "kind": "pipelinePortable",
        "experiment": ledger.get("experiment"),
        "experimentHash": ledger.get("experimentHash"),
        "ledgerSchema": ledger.get("schema"),
        "disposition": ledger.get("disposition"),
        "updatedAt": ledger.get("updatedAt"),
        "manifestStatus": ledger.get("manifestStatus"),
        "stages": ledger.get("stages") or [],
        "stageStatus": stage_status,
        "stageRuns": stage_runs,
        "originalPaths": original_paths,
    }
    if promoted:
        portable["promotedAgents"] = promoted
    if isinstance(ledger.get("parked"), dict):
        # The orphan stamp travels with the evidence (2026-08-06): an
        # imported dead chain should still say WHY it stopped where it did.
        portable["parked"] = ledger["parked"]
    if isinstance(ledger.get("epochDriftAtContinuation"), list) \
            and ledger["epochDriftAtContinuation"]:
        portable["epochDriftAtContinuation"] = \
            ledger["epochDriftAtContinuation"]
    if isinstance(ledger.get("abort"), dict):
        # Portable means PORTABLE (sixth round): the abort's evidence
        # reference becomes an imported run ID like everything else; the
        # cluster path survives only under originalPaths.
        abort = dict(ledger["abort"])
        evidence_dir = abort.pop("evidenceRunDirectory", None)
        if isinstance(evidence_dir, str) and evidence_dir:
            abort["evidenceRunID"] = os.path.basename(
                evidence_dir.rstrip(os.sep))
            original_paths["abort"] = evidence_dir
        portable["abort"] = abort
    return portable


def _jlens_trace_problem(run_directory: str) -> str | None:
    """``None`` when there is no trace or the trace is whole.

    Absent is fine — most runs declare no readout. Present-but-incomplete is
    not, because the trace is measurement evidence and a partial one cannot be
    told from a whole one by looking at the file.
    """
    try:
        from ..jlens.trace import read_summary
    except Exception:  # noqa: BLE001 - the bundler must never fail on this
        return None
    try:
        summary = read_summary(run_directory)
    except Exception:  # noqa: BLE001
        return None
    if summary is None or summary.get("complete"):
        return None
    return (f"J-lens trace in '{os.path.basename(run_directory)}' is "
            f"incomplete ({summary.get('incompleteRecords')} of "
            f"{summary.get('traceRows')} record(s)) — not usable as a readout")


#: What a pipeline run directory holds at BIRTH (`_write_config_snapshot`
#: + the ledger) plus the abort record a stopped chain may add. Anything
#: outside this set is evidence, and evidence always packages.
_PIPELINE_SEED_FILES = frozenset({
    "config.json", "experiment.json", "experiment-hash.txt", "task.txt",
    "pipeline.json", "pipeline-abort.json",
})


def failure_record_skip(run_directory: str) -> dict | None:
    """A structured SKIP for a pipeline directory that is a failure record
    with nothing to bundle — or ``None`` when normal packaging (including
    its loud refusals) should proceed.

    The 2026-08-11 factorial-memo-study import: a pipeline continuation REFUSED
    at start, leaving a chain directory holding only the seed snapshot and
    the ledger ({config, experiment, experiment-hash, pipeline, task} — no
    run-status.json, no stage outputs), and the app's bulk import died
    wholesale trying to make an evidence bundle out of it. Such a
    directory is not a packaging error — it is a failure RECORD whose
    evidence never existed. The evidence route answers with this skip so
    an importer can note it and keep going.

    Deliberately narrow; everything else stays on the loud path:

    - no ledger, or an unreadable/foreign-shaped one → ``None`` (a corrupt
      file on an otherwise-real run must stay an error, never a skip);
    - any file beyond the seed/ledger set → ``None`` — run-status.json
      included: a run that STARTED has partial-retention, not a skip;
    - any ledger-named stage directory still on disk → ``None`` (that
      evidence must come home);
    - a terminal disposition → ``None``: a COMPLETED chain missing its
      evidence is the existing refusal, and an ABORTED chain's abort
      record is a scientific determination worth bundling.
    """
    run_dir = os.path.realpath(run_directory)
    ledger_path = os.path.join(run_dir, "pipeline.json")
    try:
        with open(ledger_path, encoding="utf-8") as handle:
            ledger = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(ledger, dict) or ledger.get("disposition") is not None:
        return None
    for dirpath, _dirs, filenames in os.walk(run_dir):
        for name in filenames:
            if name.endswith(".evidence-bundle.tar.gz"):
                continue  # a prior bundle is not evidence of this run
            if dirpath != run_dir or name not in _PIPELINE_SEED_FILES:
                return None
    info = _pipeline_evidence_directories(run_dir)
    directories, missing, _disposition = info if info is not None \
        else ([], [], None)
    if directories:
        return None
    reason = ("pipeline failure record with no stage outputs — nothing to "
              "bundle beyond the ledger snapshot")
    parked = ledger.get("parked")
    if isinstance(parked, dict) and parked.get("reason"):
        reason += f"; parked: {parked['reason']}"
    if missing:
        reason += "; unreachable stage evidence: " + "; ".join(missing)
    return {
        "skipped": True,
        "kind": "evidenceSkip",
        "runID": os.path.basename(run_dir.rstrip(os.sep)),
        "runDirectory": run_dir,
        "reason": reason,
    }


def package_evidence(run_directory: str, *, output_path: str | None = None,
                     root: str | None = None, failure: dict | None = None,
                     extra_files: list[tuple[str, str]] | None = None,
                     extra_run_directories: list[str] | None = None) -> dict:
    """Package a run directory as a hash-pinned evidence bundle.

    ``failure`` (retention 2026-07-24) marks this as a PARTIAL bundle: the
    stage did not complete, the bundle is stamped ``evidenceComplete:
    false`` with the failure recorded, and the completeness refusal below is
    skipped — refusing to package a failure is exactly the behaviour that
    stranded data on the cluster. The filename carries ``.partial`` while
    keeping the ``.evidence-bundle.tar.gz`` suffix, so every existing
    scanner and importer still finds it and the tier is still obvious in a
    directory listing.

    ``extra_files`` are ``(source_path, archive_name)`` members from outside
    the run directory — scheduler logs and job records, which live beside
    the submission rather than in the run.

    ``extra_run_directories`` are additional run directories to walk as if
    the ledger had named them. A pipeline STAGE that died mid-flight is
    typically absent from `pipeline.json` — it never completed, so nothing
    recorded it — yet its partial output is exactly what a researcher
    needs. Naming it here keeps the chain root authoritative (it still
    contributes the ledger and every completed stage) without losing the
    stage that failed.
    """
    run_dir = os.path.realpath(run_directory)
    if not os.path.isdir(run_dir):
        raise BundleError(f"run directory not found: {run_directory}")
    run_id = os.path.basename(run_dir.rstrip(os.sep))
    if output_path is None:
        infix = ".partial" if failure else ""
        output_path = os.path.join(
            run_dir, f"{run_id}{infix}.evidence-bundle.tar.gz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pipeline_info = _pipeline_evidence_directories(run_dir)
    stage_directories: list[str] = []
    missing_evidence: list[str] = []
    if pipeline_info is not None:
        stage_directories, missing_evidence, disposition = pipeline_info
        if missing_evidence and disposition == "completed" and not failure:
            # A COMPLETED chain's evidence bundle must be complete — a
            # nominally successful bundle silently missing its validation,
            # sweep, generations, or judgments is worse than a refusal.
            raise BundleError(
                "pipeline evidence incomplete — refusing to package a "
                "completed chain without: " + "; ".join(missing_evidence))
    # CONTAINMENT, same rule `_pipeline_evidence_directories` already
    # enforces on ledger-named stages (external review round 3, finding 5):
    # an evidence packager must not be talked into walking arbitrary
    # readable paths. Today's only caller passes an internally-derived
    # directory, so this is not reachable from the UI — but a packager that
    # is safe only because of who happens to call it is one call site away
    # from not being safe. Fails CLOSED: an out-of-root directory is
    # skipped, never packaged.
    runs_root = os.path.realpath(os.path.dirname(run_dir.rstrip(os.sep)))
    already = {os.path.realpath(d) for d in [run_dir, *stage_directories]}
    for extra in extra_run_directories or []:
        resolved = os.path.realpath(extra)
        if not resolved.startswith(runs_root + os.sep):
            missing_evidence.append(
                f"failed stage: '{extra}' is outside the runs root")
            continue
        if os.path.isdir(resolved) and resolved not in already:
            stage_directories.append(resolved)
            already.add(resolved)
    # A J-lens trace travels automatically (the walk below takes the whole run
    # directory), but a bundle that stamps evidenceComplete over a TRUNCATED
    # readout would assert something false about its own contents. An
    # incomplete trace is recorded as missing evidence, exactly like a missing
    # stage, so the completeness marker stays honest.
    for source_dir in [run_dir, *stage_directories]:
        problem = _jlens_trace_problem(source_dir)
        if problem:
            missing_evidence.append(problem)

    files = []
    for source_dir in [run_dir, *stage_directories]:
        source_id = os.path.basename(source_dir.rstrip(os.sep))
        for dirpath, _dirs, filenames in os.walk(source_dir):
            for name in filenames:
                path = os.path.join(dirpath, name)
                if os.path.realpath(path) == os.path.realpath(output_path):
                    continue
                if name.endswith(".evidence-bundle.tar.gz"):
                    continue  # never nest a stage's own prior bundle
                files.append((path, os.path.join(
                    "runs", source_id, os.path.relpath(path, source_dir))))
    for source_path, archive_name in (extra_files or []):
        if os.path.isfile(source_path):
            files.append((source_path, archive_name))
    meta = {
        "schemaVersion": BUNDLE_SCHEMA,
        "kind": "evidenceBundle",
        "createdAt": time.time(),
        "runID": run_id,
        "runDirectory": run_dir,
        "entries": [],
    }
    if failure:
        # The tier marker. A partial bundle may be inspected, retried, and
        # cited as a FAILURE record; it must never satisfy a
        # completed-stage or evidence-grade gate, and every reader decides
        # that from these two keys rather than from the filename.
        meta["evidenceComplete"] = False
        meta["failure"] = failure
    # DECLARE every sibling directory that was packed, pipeline or not.
    #
    # This key used to be written only for pipeline bundles, so a bundle
    # carrying `extra_run_directories` outside a chain packed those bytes,
    # hashed them into `entries`, and never named them — and the Swift
    # importer moves only the primary run plus DECLARED siblings. The data
    # travelled inside an archive that called itself complete and was then
    # discarded on arrival. Today's only caller is a pipeline stage partial,
    # where the key happened to be written anyway; the docstring's promise
    # ("as if the ledger had named them") is now kept for every caller.
    if stage_directories:
        meta["pipelineStageDirectories"] = [
            os.path.basename(d.rstrip(os.sep)) for d in stage_directories]
    portable_ledger = None
    if pipeline_info is not None:
        # Honest incompleteness (aborted/interrupted chains only — a
        # completed chain with missing evidence refused above): what could
        # not be packaged is NAMED, never silently absent. A failure bundle
        # stays incomplete regardless: a chain that died can have every
        # stage directory it reached and still not be a finished chain.
        meta["evidenceComplete"] = not missing_evidence and not failure
        if missing_evidence:
            meta["missingEvidence"] = missing_evidence
        portable_ledger = _portable_pipeline_ledger(run_dir)
        if portable_ledger is not None:
            meta["pipelinePortable"] = "steerlab-pipeline.json"
            # Hash-pinned like every other member (sixth round — an
            # unverifiable member fails both importers): the digest is over
            # exactly the bytes _add_json writes.
            meta["pipelinePortableSha256"] = hashlib.sha256(
                json.dumps(portable_ledger, indent=2,
                           sort_keys=True).encode("utf-8")).hexdigest()
    with tarfile.open(output_path, "w:gz") as tar:
        entries = _add_files(tar, files)
        meta["entries"] = [entry.to_dict() for entry in entries]
        if portable_ledger is not None:
            _add_json(tar, "steerlab-pipeline.json", portable_ledger)
        _add_json(tar, "steerlab-evidence.json", meta)
    meta["bundlePath"] = output_path
    meta["bundleSha256"] = sha256_file(output_path)
    return meta


def inspect_bundle(bundle_path: str) -> dict:
    with tarfile.open(bundle_path, "r:gz") as tar:
        for candidate in ("steerlab-bundle.json", "steerlab-evidence.json"):
            try:
                member = tar.getmember(candidate)
            except KeyError:
                continue
            payload = _read_metadata_member(tar, member)
            if not payload:
                break
            data = json.loads(payload.decode("utf-8"))
            data["bundlePath"] = bundle_path
            data["bundleSha256"] = sha256_file(bundle_path)
            return data
    raise BundleError("not a SteerLab bundle")


def _is_manifest_path(dest: str, target: str) -> bool:
    """Both manifest layouts: experiments/<name>/experiment.json AND the
    legacy flat file experiments/<name>.json — a bundle must not dodge the
    frozen guard by shipping the flat form."""
    if os.path.basename(dest) == "experiment.json":
        return True
    experiments = os.path.join(target, "experiments")
    return (os.path.dirname(dest) == experiments and dest.endswith(".json"))


def _is_frozen_manifest(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("status") == "frozen"
    except (OSError, ValueError):
        return False


def _clears_every_arm(path: str, payload: bytes) -> bool:
    """True when ``payload`` would take the DRAFT manifest at ``path`` from
    holding a measured surface to holding none (open-issues §8).

    Unreadable on either side answers False: an unparseable member is the
    per-entry hash check's business, and an unparseable file on disk has no
    arms this can claim to protect."""
    try:
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        incoming = json.loads(payload.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    if not isinstance(existing, dict) or not isinstance(incoming, dict):
        return False
    if existing.get("status") != "draft":
        return False
    return experiment_store._clears_every_arm(existing, incoming)


def _check_bundle_closure(meta: dict, members) -> None:
    """Pre-extraction COMPLETENESS proof (eighth round): per-member hash
    checks verify what is present, but only a closure check catches what
    was REMOVED — a bundle whose declared entry (or declared portable
    ledger) is simply absent would otherwise import "successfully" while
    silently missing evidence. Every declared path must appear exactly
    once; a stamped portable pin requires its member."""
    counts: dict[str, int] = {}
    for member in members:
        if member.isfile():
            counts[member.name] = counts.get(member.name, 0) + 1
    declared = [str(entry.get("path") or "")
                for entry in meta.get("entries", [])]
    if len(set(declared)) != len(declared):
        raise BundleError(
            "bundle metadata declares duplicate entry paths — refusing an "
            "ambiguous bundle")
    missing = [path for path in declared if counts.get(path, 0) == 0]
    if missing:
        raise BundleError(
            "bundle is missing declared member(s): "
            + ", ".join(sorted(missing)[:5])
            + (" …" if len(missing) > 5 else "")
            + " — refusing an incomplete bundle")
    duplicated = [path for path in declared if counts.get(path, 0) > 1]
    if duplicated:
        raise BundleError(
            "bundle carries duplicate member(s): "
            + ", ".join(sorted(duplicated)[:5])
            + " — refusing an ambiguous bundle")
    # PRESENCE of the field, not its truthiness (external review,
    # 2026-09-05): the stamp is the claim, and a bundle that declares an
    # EMPTY pin and ships no ledger would otherwise import clean — pin
    # stamped, evidence silently gone. A malformed-but-present pin is the
    # per-member check's business; that the member exists at all is this
    # check's.
    if meta.get("pipelinePortableSha256") is not None \
            and counts.get("steerlab-pipeline.json", 0) != 1:
        raise BundleError(
            "bundle metadata pins a portable pipeline ledger the bundle "
            "does not carry (exactly once) — refusing an incomplete bundle")


def _refuse_outer_hash_mismatch(bundle_path: str, expected: str | None) -> None:
    """The OUT-OF-BAND pin, checked before the archive is opened at all.

    Portability gap G3 (``docs/PORTABILITY-CONTRACTS.md``): the archive digest
    travels beside the archive (a job record's ``bundleSha256``), never inside
    it, and the recipient checks it. The Mac has always done that BEFORE
    extracting (``EvidenceBundleImporter.importEvidenceBundle(_:expectedSHA256:)``,
    ``Sources/ExperimentKit/ClusterClient.swift``); this engine offered no
    equivalent, so a Python client had to hash the file itself and the
    verification story was one story per engine.

    Substitution, not corruption, is what this catches: a wholesale swapped
    archive is internally consistent — every member matches its own metadata —
    so the per-member checks inside :func:`import_bundle` cannot see it. Both
    hashes are NAMED, because a client that cannot see which side moved cannot
    tell a stale pin from a tampered file.
    """
    # An ABSENT pin (``None``, ``""``, blank) is "no pin", which is the
    # documented option every call site already relies on — ``import_bundle``
    # only calls this when the caller supplied something. Anything else must
    # be a digest: a non-string used to raise ``AttributeError`` from
    # ``.strip()`` (a crash, not a refusal) and a garbage string was reported
    # as a hash MISMATCH, which tells the reader to go re-fetch an archive
    # that was never the problem (external review, 2026-09-05).
    if expected is None or (isinstance(expected, str) and not expected.strip()):
        return
    expected = _require_digest(expected, what="the out-of-band bundle pin")
    actual = sha256_file(bundle_path)
    if actual != expected:
        raise BundleError(
            f"bundle hash mismatch: expected {expected}, got {actual} — "
            "nothing was extracted. Re-fetch the bundle and retry with the "
            "expectedSha256 the job record stamped, or drop the pin only if "
            "you fetched the archive over a channel you already trust")


#: A SHA-256 digest as bundle metadata may spell one. Hex carries no case, so
#: BOTH cases are accepted and every comparison is made in lowercase — a
#: producer that stamped uppercase is VERIFIED, not refused over a difference
#: that means nothing. Length and alphabet are exact, because the whole point
#: of checking shape is that ``""`` and ``"not a digest"`` become refusals
#: rather than a comparison that silently cannot fail (external review,
#: 2026-09-05).
_SHA256_TEXT = re.compile(r"\A[0-9a-fA-F]{64}\Z")


def _require_digest(value, *, what: str) -> str:
    """``value`` as a lowercase hex digest, or a refusal.

    Anything that is not 64 hex characters is refused: absent, ``""``, blank,
    the wrong length, the wrong alphabet, or not a string at all.

    The distinction this exists for (external review, 2026-09-05). The
    importer used to ask only whether a pin was ``None`` and then compare it
    with ``if expected and digest != expected`` — so ``"sha256": ""`` passed
    the presence test AND switched the comparison off, and the member landed
    in the canonical workspace with nothing having verified a byte of it. A
    pin that cannot be compared is not a weaker pin; it is NO pin, and an
    unpinned member is precisely what this module refuses everywhere else.
    """
    if not isinstance(value, str) or not _SHA256_TEXT.match(value):
        raise BundleError(
            f"{what} declares {value!r}, which is not a SHA-256 digest "
            "(64 hex characters) — refusing an unverifiable bundle")
    return value.lower()


def _require_safe_segment(value, *, field: str) -> str:
    """One path segment, and one that cannot walk out of the directory it
    names: no separators, no NUL, never absolute, never ``.`` or ``..``.

    Bundle METADATA is attacker-controlled in exactly the way member names
    are, and the importer DERIVES destinations from it — so a metadata
    identifier gets the member rule (external review, 2026-09-05). This is the
    cheap, legible half, and it names the field that is wrong; canonical
    containment (:func:`_contained_destination`) is still checked afterwards,
    because a segment that looks innocent can still be a symlink.
    """
    if (not isinstance(value, str) or not value.strip()
            or value in (os.curdir, os.pardir) or os.path.isabs(value)
            or "/" in value or "\\" in value or "\0" in value):
        raise BundleError(
            f"bundle metadata field {field!r} is not a single safe path "
            f"segment: {value!r} — refusing a bundle whose derived "
            "destination could leave the import root")
    return value


def _contained_destination(target: str, *parts: str, what: str) -> str:
    """A DERIVED destination, canonicalized and held to the member rule.

    Literally the two lines :func:`_plan_import` runs on every archive member
    — ``os.path.realpath`` then ``os.path.commonpath`` — because a destination
    the importer COMPUTES from metadata deserves no more trust than one the
    archive states, and a symlink is a tunnel whichever way the path was
    reached (external review, 2026-09-05).
    """
    dest = os.path.realpath(os.path.join(target, *parts))
    if dest != target and os.path.commonpath([target, dest]) != target:
        raise BundleError(
            f"bundle metadata places {what} outside the target root "
            f"({dest}) — refusing")
    return dest


def _preflight_run_directory(meta: dict, target: str, *,
                             portable: bool) -> str | None:
    """The run directory this bundle's metadata derives, VALIDATED — or
    ``None`` when the bundle names no run and needs none.

    The bug this closes (external review, 2026-09-05). ``runID`` came straight
    out of the bundle's own metadata and was joined onto ``<target>/runs``
    with no check at all, so a validly pinned evidence bundle declaring
    ``runID: "../../outside"`` wrote its portable ledger into a sibling of the
    import root — and a ``runs/<id>`` that was a symlink carried it out the
    same way. Neither hash helps: the outer pin legitimately pins an archive
    whose METADATA is unsafe, and the ledger's own pin attests that the bytes
    are the right bytes, never where they may go.

    Decided HERE, with the rest of preflight, so the refusal costs the
    workspace nothing — not at commit time, once the archive's ordinary
    members have already landed.
    """
    raw = meta.get("runID")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if portable:
            # A verified ledger with nowhere to go used to be dropped in
            # silence, which is a "successful" import missing the very
            # evidence it was carrying.
            raise BundleError(
                "bundle carries a portable pipeline ledger but declares no "
                "runID to place it under — refusing an unplaceable bundle")
        return None
    run_id = _require_safe_segment(raw, field="runID")
    return _contained_destination(target, "runs", run_id,
                                  what=f"runs/{run_id}")


#: The staging directory's name prefix. Created INSIDE the target root, so the
#: commit below is a same-volume rename rather than a copy, and removed in a
#: ``finally`` whatever happens.
_STAGING_PREFIX = ".steerlab-import-"


@dataclass
class _PlannedMember:
    """One member that PASSED preflight and is therefore going to land."""

    member: object                       # tarfile.TarInfo
    #: Absolute path in the target root this member commits to.
    dest: str
    #: The pinned digest from the bundle's entry list, normalized to lowercase
    #: hex. Always a usable digest: preflight refuses a member whose pin is
    #: absent, empty, or malformed (:func:`_require_digest`), so the staging
    #: pass compares UNCONDITIONALLY and has no "no pin here" branch left to
    #: fall through (external review, 2026-09-05).
    expected: str
    #: Whether OVERWRITING this destination was permitted at preflight, carried
    #: into the commit pass so the commit can ENFORCE the preflight's answer
    #: rather than assume it still holds (third-round review, 2026-08-24).
    #: Preflight refuses an existing destination when the caller did not pass
    #: ``allow_overwrite`` — but a file created in the window between that
    #: check and the commit was then backed up and replaced anyway, which is
    #: the refusal being worked around by the passage of time. A member that
    #: was NOT granted permission commits with a primitive that cannot
    #: overwrite, so the rule survives the race as well as the check.
    may_overwrite: bool = False
    #: Where the verified bytes wait between staging and commit.
    staged: str = ""


@dataclass
class _Landed:
    """One member that HAS landed this call, and how to put it back."""

    dest: str
    #: A hardlink (or, on a filesystem without them, a copy) of whatever stood
    #: at ``dest`` before this call overwrote it — ``None`` when nothing did,
    #: in which case rollback simply removes ``dest``.
    backup: str | None


def _plan_import(tar, meta: dict, members, *, target: str,
                 allow_overwrite: bool) -> tuple[list[_PlannedMember],
                                                 bytes | None, str | None]:
    """PREFLIGHT the complete archive: every refusal that can be decided
    without the member's bytes, decided before a single byte lands.

    The bug this closes (external review, 2026-08-24). Members used to be
    verified-and-landed one at a time, so an archive whose second member
    escaped the target root left the FIRST one installed in the canonical
    workspace — and the never-overwrite rule then refused the clean retry,
    leaving the workspace in a state only manual surgery could clear. A
    refusal must leave the workspace exactly as it found it, which means the
    whole archive is judged before any of it is committed.

    Returns the members that will land, in archive order, the verified
    portable pipeline ledger (or ``None``), and the validated run directory
    the metadata derives (or ``None``). Reads nothing but tar metadata and
    the bundle's own small JSON documents.

    The DERIVED destination is judged here too (external review, 2026-09-05).
    Preflight used to cover only what the archive names; the ledger's
    destination is computed from ``runID``, landed after the members, and had
    never been checked at all — so a refusal that belongs in preflight was
    instead a write outside the import root. See
    :func:`_preflight_run_directory`.
    """
    plan: list[_PlannedMember] = []
    portable_payload: bytes | None = None
    member_limit = max_member_bytes()
    for member in members:
        if member.isdir() or member.name in {"steerlab-bundle.json",
                                             "steerlab-evidence.json"}:
            continue
        if member.name == "steerlab-pipeline.json":
            # The portable pipeline ledger: pin REQUIRED (seventh round — an
            # unpinned member is unverifiable), verified, and RETAINED inside
            # the imported pipeline run dir after extraction so local readers
            # resolve stage references without cluster paths.
            declared = meta.get("pipelinePortableSha256")
            if declared is None or (isinstance(declared, str)
                                    and not declared.strip()):
                raise BundleError(
                    "portable pipeline ledger carries no hash pin — "
                    "refusing an unverifiable bundle")
            # Shape BEFORE comparison (external review, 2026-09-05): a pin
            # that is not a digest is a broken bundle, not a tampered one, and
            # `!=` reporting it as tampering sends the reader hunting the
            # wrong thing.
            expected = _require_digest(declared,
                                       what="the portable pipeline ledger's "
                                            "hash pin")
            payload = _read_metadata_member(tar, member)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise BundleError(
                    "portable pipeline ledger failed its hash pin — "
                    "refusing a tampered bundle")
            portable_payload = payload
            continue
        basename = os.path.basename(member.name)
        if basename.startswith("._") or basename == ".DS_Store":
            # macOS tar resource-fork metadata — never workspace data, and
            # never listed in the hash entries. Skip, don't refuse: bundles
            # packed by older Swift builds carry these.
            continue
        # Every imported file must be hash-verifiable: a member the bundle's
        # entry list does not name would otherwise land in the workspace
        # unchecked (parallel to the Swift evidence importer's
        # ensureAllVerified).
        declared = _entry_hash(meta, member.name)
        if declared is _UNLISTED:
            raise BundleError(
                f"bundle member not listed in the bundle's hash entries "
                f"(unverifiable): {member.name}")
        # LISTED is not the same as PINNED. The entry exists; whether it
        # carries a digest that can be compared is a separate question, and
        # the one ``"sha256": ""`` used to slip through (external review,
        # 2026-09-05).
        expected = _require_digest(
            declared, what=f"the bundle's entry for {member.name}")
        dest = os.path.realpath(os.path.join(target, member.name))
        if os.path.commonpath([target, dest]) != target:
            raise BundleError(
                f"bundle member escapes target root: {member.name}")
        # Refuse on the DECLARED uncompressed size before opening the member
        # at all: the 4-GiB upload cap counts compressed bytes, so a bundle
        # inside it can still declare a member nothing on this machine can
        # hold (see ``max_member_bytes``).
        _refuse_oversized_member(member, member_limit)
        if tar.extractfile(member) is None:
            continue
        # The never-overwrite rule is a PREFLIGHT refusal, not a commit-time
        # one: deciding it against the target here is what lets the commit
        # pass be a sequence of renames with nothing left to change its mind.
        # The ANSWER travels with the member (`may_overwrite`), because the
        # commit must be able to enforce it — the check below is a point in
        # time, and a destination that appears after it must meet the same
        # rule, not slip past it.
        if os.path.exists(dest) and not allow_overwrite:
            raise BundleError(
                f"refusing to overwrite existing file: {member.name}")
        plan.append(_PlannedMember(member=member, dest=dest,
                                   expected=expected,
                                   may_overwrite=bool(allow_overwrite)))
    run_dir = _preflight_run_directory(meta, target,
                                       portable=portable_payload is not None)
    return plan, portable_payload, run_dir


def _stage_plan(tar, plan: list[_PlannedMember], *, staging: str,
                target: str) -> None:
    """Stream every planned member into ``staging`` and verify it there.

    Nothing under the target's real layout is touched by this pass: a failure
    anywhere in it leaves the workspace untouched and the staging tree is
    thrown away whole. The digest check, the frozen-manifest firewall and the
    draft-manifest firewall all run HERE — before the commit pass — so the
    commit has no refusal left in it.
    """
    member_limit = max_member_bytes()
    total_limit = max_total_bytes()
    total_written = 0
    for planned in plan:
        member = planned.member
        # Mirror the member's own relative path under the staging root, plus
        # the historical `.tmp` suffix: same volume as `dest`, so the commit
        # is a rename, and unique per call because `staging` is.
        relative = os.path.relpath(planned.dest, target)
        planned.staged = os.path.join(staging, relative) + ".tmp"
        os.makedirs(os.path.dirname(planned.staged), exist_ok=True)
        # The member streams into the staged file in chunks and is hashed on
        # the way past (never held whole in memory).
        digest, written = _stream_member(
            tar, member, planned.staged, member_limit,
            total_before=total_written, total_limit=total_limit)
        total_written += written
        # The integrity check runs on the STAGED bytes, BEFORE `dest` is
        # touched (shard race, 2026-08-04): the old order wrote dest and then
        # re-hashed it from disk — but concurrent shard jobs of one submission
        # extract the same bundle into the same workspace, so another shard's
        # in-progress (identical) rewrite could be read torn, refusing a
        # perfectly good bundle after 3 seconds ("hash mismatch after
        # extracting …"). Hashing what the tar actually carried is the tamper
        # firewall this check exists for, and it cannot race.
        # UNCONDITIONAL. It read ``if planned.expected and digest != …``,
        # which made an empty pin disable its own check — a guard that a
        # tampered bundle could satisfy by supplying nothing (external review,
        # 2026-09-05). Preflight now guarantees a usable lowercase digest, so
        # there is nothing left to guard against and no way to opt out.
        if digest != planned.expected:
            raise BundleError(
                f"hash mismatch after extracting {member.name}")
        if os.path.exists(planned.dest) and _is_manifest_path(planned.dest,
                                                              target):
            # Manifests are small JSON documents, and both firewalls below
            # compare whole documents — so this is the one place the staged
            # bytes are read back.
            with open(planned.staged, "rb") as staged_handle:
                payload = staged_handle.read()
            # Circularity firewall: never silently replace a FROZEN manifest
            # with different content, even when allow_overwrite is set (which
            # bundle-execute does for run artifacts). Re-importing the
            # identical frozen manifest is fine; changed content is a
            # freeze/verify violation surfaced as an error, not a silent stomp.
            if _is_frozen_manifest(planned.dest):
                with open(planned.dest, "rb") as existing:
                    if existing.read() != payload:
                        raise BundleError(
                            f"refusing to overwrite frozen manifest "
                            f"{member.name} with different content (freeze "
                            "firewall)")
            # The same firewall one tier down, for DRAFTS (open-issues §8): a
            # bundle carrying a skeleton manifest must not silently take a
            # draft that has since gained concepts and conditions to
            # both-empty. `_is_frozen_manifest` is False for a draft, so the
            # check above never saw this; the arms simply vanished and only
            # the run directory's snapshot remembered them.
            if _clears_every_arm(planned.dest, payload):
                raise BundleError(
                    f"refusing to overwrite draft manifest {member.name} with a "
                    "document that has no concepts and no conditions — the "
                    "workspace copy holds arms this bundle does not (import "
                    "into a clean target root, or duplicate the workspace "
                    "study before re-importing)")


def _backup_existing(dest: str, rollback_root: str, index: int) -> str | None:
    """Preserve whatever stands at ``dest`` so the commit can be undone.

    A hardlink, not a move: moving the old file aside would open a window in
    which ``dest`` does not exist at all, and the whole reason the landing is
    a rename is that a concurrent shard READING this artifact must never
    observe an in-between state. A filesystem without hardlinks falls back to
    a copy, which is slower and just as correct.
    """
    if not os.path.exists(dest):
        return None
    backup = os.path.join(rollback_root, f"{index:06d}-{os.path.basename(dest)}")
    try:
        os.link(dest, backup)
    except OSError:
        shutil.copy2(dest, backup)
    return backup


def _commit_one(staged: str, dest: str, *, may_overwrite: bool) -> None:
    """Land ONE staged member. The single mutating step of the commit pass,
    factored out so a test can make the commit fail where nothing else can.

    Two primitives, chosen by what PREFLIGHT decided about this member:

    * ``may_overwrite`` — ``os.replace``, exactly as before. The rename is
      atomic for a concurrent shard READING the artifact (see the shard-race
      note in :func:`_stage_plan`), which is why an overwrite-permitted landing
      must stay a rename and must not become unlink-then-create.
    * otherwise — :func:`_commit_no_replace`, which CANNOT overwrite. Preflight
      established that nothing stood at ``dest``; if something does now, it
      arrived after the check, and ``os.replace`` would have silently destroyed
      it in the name of a rule that had just refused exactly that
      (third-round review, 2026-08-24).
    """
    if may_overwrite:
        os.replace(staged, dest)
        return
    _commit_no_replace(staged, dest)


def _commit_no_replace(staged: str, dest: str) -> None:
    """Land a staged member onto ``dest``, incapable of overwriting it.

    The TWIN of ``client.runner._commit_no_replace`` — same primitive, same
    fallback, same reasoning — deliberately mirrored rather than shared: that
    module is the thin HTTP client and must stay importable from an install
    that has only httpx, while this one reaches the whole archive and workspace
    machinery. (``sha256_file`` is duplicated across the same seam for the same
    reason, and says so.) A change to either belongs in both.

    ``os.link`` + ``os.remove``, not ``os.replace``: link raises
    ``FileExistsError`` rather than clobbering, and it commits bytes that are
    already on the disk instead of copying them a second time. The O_EXCL
    reservation is the fallback for a filesystem without hardlinks, where it is
    slower and just as unable to overwrite anything — with the same stated
    residual as its twin: on that path the destination NAME is visible, empty
    then partial, for as long as the copy takes. Nothing can overwrite it, and
    a failed copy removes it again.

    Raises ``FileExistsError`` when ``dest`` exists; leaves ``staged`` in place
    for the caller (here, ``_rollback`` plus the staging teardown) when it does.

    BOTH-OR-NEITHER (review round 10, finding 8): ``dest`` is this call's
    reservation, and a commit that cannot FINISH must not leave it standing.
    Dropping the staging name is the last step, and it can fail on its own
    (a read-only staging directory, an interrupt) — after ``dest`` has landed
    and BEFORE ``_commit_one`` returns, so the member never reaches ``landed``
    and ``_rollback`` never knew to undo it. The removal is therefore wrapped,
    and a failure takes ``dest`` back out before it propagates.
    """
    try:
        os.link(staged, dest)
    except FileExistsError:
        raise
    except OSError:
        handle_fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        landed = False
        try:
            try:
                landing = os.fdopen(handle_fd, "wb")
            except BaseException:
                os.close(handle_fd)
                raise
            with landing, open(staged, "rb") as source:
                shutil.copyfileobj(source, landing)
            landed = True
        finally:
            if not landed:
                # A copy that died half way must not leave a short file wearing
                # the destination's name — the reservation comes back out.
                try:
                    os.remove(dest)
                except OSError:
                    pass
    try:
        os.remove(staged)
    except BaseException:
        try:
            os.remove(dest)
        except OSError:
            pass
        raise


def _rollback(landed: list[_Landed], created_dirs: list[str]) -> None:
    """Undo everything this call landed, newest first.

    Best effort by necessity — the refusal the caller is about to see is the
    news, and a rollback that raised would replace it with something less
    useful. Directories are removed only when empty, so a directory that
    already held workspace data is never touched.
    """
    for entry in reversed(landed):
        try:
            if entry.backup is not None:
                os.replace(entry.backup, entry.dest)
            else:
                os.remove(entry.dest)
        except OSError:
            pass
    for directory in sorted(created_dirs, key=len, reverse=True):
        try:
            os.rmdir(directory)
        except OSError:
            pass


def _makedirs_tracked(directory: str, created: list[str]) -> None:
    """``os.makedirs(exist_ok=True)``, remembering what it actually made, so a
    rollback removes the empty scaffolding it created and nothing else."""
    missing: list[str] = []
    cursor = directory
    while cursor and not os.path.isdir(cursor):
        missing.append(cursor)
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        cursor = parent
    os.makedirs(directory, exist_ok=True)
    created.extend(missing)


def import_bundle(bundle_path: str, *, target_root: str | None = None,
                  allow_overwrite: bool = False,
                  expected_sha256: str | None = None) -> dict:
    """Extract a bundle into ``target_root``, as ONE transaction.

    Three passes, and the split is the whole point:

    1. **preflight** (:func:`_plan_import`) — closure, per-member containment,
       hash-entry coverage (shape included, so an empty pin is a refusal
       rather than a check that switches itself off), the containment of every
       destination DERIVED from metadata, the size caps, and the
       never-overwrite rule, all decided against the real target before
       anything is written;
    2. **stage** (:func:`_stage_plan`) — every member streams into a temporary
       tree on the destination filesystem and is verified there, including the
       frozen- and draft-manifest firewalls;
    3. **commit** — the staged tree moves into place, member by member, with
       every landing recorded; any failure rolls the whole call back. The
       commit ENFORCES what preflight decided rather than trusting it to have
       stayed true: a member that was not granted overwrite permission lands
       with a primitive that cannot overwrite (:func:`_commit_no_replace`), so
       a destination created in the window between the two passes is the same
       refusal it would have been a second earlier — not a silent replacement.

    Before this was transactional, a refusal partway through installed
    everything before it and then the never-overwrite rule blocked the clean
    retry — a workspace that could only be repaired by hand. Now a refused
    import is a no-op and retrying it is ordinary.

    ``expected_sha256`` is the OPTIONAL out-of-band outer pin (G3): when
    supplied it is verified before the archive is opened, and a mismatch
    refuses with both hashes named and nothing written. Omitted (the default)
    is exactly the historical behaviour — the caller has then taken
    responsibility for the outer digest itself, which is what every existing
    call site does.
    """
    target = os.path.realpath(target_root or paths.project_root())
    # BEFORE `inspect_bundle` — before the archive is opened, let alone
    # extracted — so a substituted archive never gets to speak for itself.
    if expected_sha256:
        _refuse_outer_hash_mismatch(bundle_path, expected_sha256)
    meta = inspect_bundle(bundle_path)
    extracted: list[str] = []
    staging: str | None = None
    landed: list[_Landed] = []
    created_dirs: list[str] = []
    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            members = tar.getmembers()
            # The AGGREGATE caps come first: they are the cheapest refusal in
            # the file and the one a hostile archive is built to outrun.
            _refuse_oversized_archive(members)
            _check_bundle_closure(meta, members)
            plan, portable_payload, run_dir = _plan_import(
                tar, meta, members, target=target,
                allow_overwrite=allow_overwrite)
            if plan:
                os.makedirs(target, exist_ok=True)
                # Same volume as every destination, because it is INSIDE the
                # target root — which is what makes the commit a rename.
                staging = tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=target)
                _stage_plan(tar, plan, staging=staging, target=target)
        # ---- commit -------------------------------------------------------
        rollback_root = None
        if staging is not None:
            rollback_root = os.path.join(staging, "rollback")
            os.makedirs(rollback_root, exist_ok=True)
        for index, planned in enumerate(plan):
            _makedirs_tracked(os.path.dirname(planned.dest), created_dirs)
            # Only an overwrite-permitted member has anything to preserve: for
            # the others preflight established the destination was absent, and
            # if something IS there the commit refuses without touching it, so
            # copying it aside first would be work done on a file this call
            # must leave exactly as it found it.
            backup = (_backup_existing(planned.dest, rollback_root, index)
                      if planned.may_overwrite else None)
            try:
                _commit_one(planned.staged, planned.dest,
                            may_overwrite=planned.may_overwrite)
            except FileExistsError:
                # The window between preflight and commit, closed (third-round
                # review, 2026-08-24). Everything landed by THIS call rolls
                # back; the file that appeared is not this import's to keep or
                # to destroy, and is left untouched.
                raise BundleError(
                    f"refusing to overwrite existing file: "
                    f"{planned.member.name} — it appeared AFTER preflight "
                    "checked the destination, between that check and this "
                    "commit. Nothing of this bundle was kept: import into a "
                    "clean target root, or move that file aside and "
                    "retry") from None
            landed.append(_Landed(dest=planned.dest, backup=backup))
            extracted.append(planned.member.name)
        # Retain the (verified) portable ledger inside the imported pipeline
        # run dir — parallel to the Swift importer — so local readers resolve
        # stage references by run ID, never by cluster path. Part of the same
        # transaction: it lands in the canonical workspace like anything else.
        if portable_payload is not None:
            # `run_dir` is preflight's, not this line's: the metadata that
            # names it was validated as a single safe segment and its
            # destination canonically contained before a member landed
            # (external review, 2026-09-05). Preflight also guarantees it is
            # not None whenever there is a payload to place.
            run_id = str(meta.get("runID") or "")
            # …and re-canonicalized HERE, at commit time, for the same reason
            # `_commit_one` re-enforces the overwrite rule: preflight ran
            # before this call's own members created `runs/`, and a symlink
            # swapped into that window would carry the write out of the root
            # preflight cleared. Cheap, and the check must be true at the
            # moment it is relied on, not merely at the moment it was made.
            run_dir = _contained_destination(target, "runs", run_id,
                                             what=f"runs/{run_id}")
            if os.path.isdir(run_dir):
                local = os.path.join(run_dir, "pipeline-portable.json")
                # Retention is still "only if it is not already there" — a
                # re-import must leave a ledger that stands alone, exactly as
                # before. What changed is HOW the bytes land (external review,
                # 2026-08-26). It was ``if not exists: open(local, "wb")``,
                # which is the one write in this commit that bypassed the
                # transaction twice over: a file appearing between the check
                # and the open was TRUNCATED rather than refused, and the
                # bytes reached the workspace before anything registered them,
                # so a write that died half way left a partial ledger no
                # rollback knew about. Staging it and landing it through
                # :func:`_commit_one` puts it under the same machinery as
                # every other member — a partial write happens in the staging
                # tree, and the landing either takes the name or refuses it.
                if not os.path.exists(local):
                    if staging is None:
                        # No member landed, so there is no staging tree yet —
                        # inside the target root, like the other one, so the
                        # landing stays a same-filesystem operation.
                        staging = tempfile.mkdtemp(prefix=_STAGING_PREFIX,
                                                   dir=target)
                    staged_ledger = os.path.join(staging,
                                                 "pipeline-portable.json")
                    with open(staged_ledger, "wb") as handle:
                        handle.write(portable_payload)
                    # Registered BEFORE the landing, so no failure between the
                    # moment `local` acquires bytes and the moment this call
                    # returns can leave them behind unrolled-back.
                    ledger_entry = _Landed(dest=local, backup=None)
                    landed.append(ledger_entry)
                    try:
                        _commit_one(staged_ledger, local, may_overwrite=False)
                    except FileExistsError:
                        # Identical to the member loop's window: the file
                        # arrived after the check above and is not this
                        # import's to keep or to destroy — so the registration
                        # comes back out before the rollback runs, or the
                        # rollback would delete somebody else's file.
                        landed.pop()
                        raise BundleError(
                            f"refusing to overwrite existing file: "
                            f"runs/{run_id}/pipeline-portable.json — it "
                            "appeared AFTER preflight checked the "
                            "destination, between that check and this commit. "
                            "Nothing of this bundle was kept: import into a "
                            "clean target root, or move that file aside and "
                            "retry") from None
                    extracted.append(f"runs/{run_id}/pipeline-portable.json")
    except BaseException:
        _rollback(landed, created_dirs)
        raise
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    result = {"bundle": meta, "targetRoot": target, "extracted": extracted}
    if meta.get("kind") == "evidenceBundle":
        # The adoption reconciliation the Swift auto-import always ran and
        # this raw path historically skipped (the 2026-08-06 replication-run
        # recovery): a run whose server-side verb auto-pinned the model
        # revision must offer that pin to the same-named local draft, or the
        # next analyze/evaluate refuses on an epoch diff the researcher
        # never authored. Loud in the RESULT, never a refusal — importing
        # evidence into a workspace with no matching experiment stays legal.
        from . import experiment_store
        # The SAME derived directory, and therefore the same preflight-
        # validated one: adoption reads the run's own artifacts and writes the
        # local draft, so a `runID` that escaped the root would have aimed
        # both at whatever it pleased (external review, 2026-09-05).
        if run_dir is not None and os.path.isdir(run_dir):
            result["revisionAdoption"] = \
                experiment_store.adopt_evidence_revision(run_dir, target)
    return result


def resolve_against_target(path: str | None, target_root: str) -> str | None:
    """A caller-supplied RUN-DIRECTORY path, anchored to the target root.

    **The bug this closes** (ledger 2026-08-21, reproduced live on jobs
    47606365/47606373). ``study submit … --source runs/<dir>`` renders the
    path VERBATIM into the bundle command, and the rendered sbatch ``cd``s
    into ``<submitdir>/slurm`` before ``srun`` — so a relative path resolved
    against the slurm directory, not against the workspace the very same
    command line names with ``--target``. The read then found nothing, and
    the epoch guard blamed the run's provenance for the operator's path.

    The fix belongs in the CHILD rather than only in the renderer, because a
    hand-written ``bundle execute`` has exactly the same cwd problem and no
    renderer to fix it. Absolute paths are untouched; ``None``/empty pass
    through so "not supplied" keeps meaning "not supplied".

    The rule is the one already true of every other workspace-relative
    reference in this engine (``paths.resolve``, ``_load_prompts``): a
    relative artifact path names a location under the artifact root.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(target_root, path))


def _partial_directory_for(verb: str, name: str, target_root: str,
                           started: float) -> str | None:
    """Best-effort recovery of a run directory a failing stage did not name.

    ``run`` and ``evaluate`` report their directory (a callback and an
    exception attribute respectively). ``extract``, ``validate``, and
    ``sweep`` do not, so a failure there had nothing to package — the exact
    hole this closes.

    Matching is deliberately strict, because packaging the WRONG run would
    be worse than packaging none: the directory must carry this
    experiment-and-verb slug AND have been created after this job started,
    and if more than one candidate qualifies we refuse to guess rather than
    pick one. A concurrent job in the same root is the case that would make
    a guess wrong, and it is exactly the case that produces two candidates.
    """
    runs_root = os.path.join(target_root, "runs")
    if not os.path.isdir(runs_root):
        return None
    prefix = f"exp-{name}-{verb}"
    candidates = []
    for entry in os.listdir(runs_root):
        # `<stamp>-exp-<name>-<verb>` plus shard/suffix variants.
        if not _matches_slug_component(entry, prefix):
            continue
        path = os.path.join(runs_root, entry)
        if not os.path.isdir(path):
            continue
        try:
            if os.stat(path).st_ctime + 1.0 < started:
                continue  # predates this job (1s slack for clock coarseness)
        except OSError:
            continue
        candidates.append(path)
    if len(candidates) != 1:
        return None
    return candidates[0]


#: The ONLY components that may follow a run-directory slug. Anything else
#: means the slug is a prefix of a DIFFERENT name, not this one's suffix.
#:
#: - a collision counter (``paths.make_unique_run_directory`` appends ``-2``,
#:   ``-3``, … when a stamp repeats)
#: - a shard token. ``tasks`` writes ``shard<i>of<n>`` (one component); the
#:   split ``shard-<i>`` spelling is accepted too, since a bare ``shard``
#:   plus a digits component reads the same way and existing coverage
#:   asserts it. Both are safe: what made the old rule dangerous was
#:   admitting WORDS (``run``, ``evaluate``, ``thing``) that continue another
#:   experiment's name, not admitting a second shard spelling.
_RUN_DIRECTORY_SUFFIX_COMPONENT = re.compile(r"\A(?:\d+|shard\d*(?:of\d+)?)\Z")


def _matches_slug_component(entry: str, prefix: str) -> bool:
    """Whether a run-directory name IS this slug, allowing only the known
    suffixes.

    Plain substring matching marked FOREIGN directories: experiment ``a`` +
    verb ``run`` (slug ``exp-a-run``) substring-matched experiment
    ``a-runner``'s directories, and when that wrong directory was the single
    candidate it was partial-marked — a WRITE into another experiment's run
    directory — and packaged as this experiment's evidence.

    A hyphen boundary alone does NOT close that hole (2026-07-27): the hyphen
    after the slug can just as easily continue another experiment's NAME.
    Experiment ``a`` matched ``…-exp-a-run-run`` (experiment ``a-run``, verb
    ``run``), ``…-exp-a-run-evaluate`` (experiment ``a-run``, verb
    ``evaluate``), and ``…-exp-a-run-thing-run`` (experiment
    ``a-run-thing``) — the same cross-experiment write, through a narrower
    door. So the tail after the slug must be EMPTY or consist only of known
    suffix components; an unrecognised component means a different name.

    This is the rule Swift's ``SweepRunCatalog.directoryNameMatches`` already
    applied (slug at the end, or slug + ``-`` + an all-digits tail); the
    engines now agree.
    """
    index = entry.find(prefix)
    while index != -1:
        end = index + len(prefix)
        if index == 0 or entry[index - 1] == "-":
            tail = entry[end:]
            if not tail:
                return True
            if tail[0] == "-" and all(
                    _RUN_DIRECTORY_SUFFIX_COMPONENT.match(component)
                    for component in tail[1:].split("-")):
                return True
        index = entry.find(prefix, index + 1)
    return False


#: Verb names that can appear as the trailing token of a run-directory slug.
_RUN_DIRECTORY_STAGES = (
    "extract", "validate", "sweep", "run", "evaluate", "analyze", "pipeline")


def _stage_of_run_directory(directory: str, fallback: str) -> str:
    """The stage a run directory belongs to, from its own name.

    A pipeline's failing STAGE directory should say which stage it was, not
    "pipeline" — the chain root is the one that is a pipeline. Falls back to
    the caller's verb whenever the name carries no known stage token (shard
    suffixes, collision suffixes, anything unrecognised).

    Stage tokens match at hyphen-COMPONENT boundaries, scanning from the
    END: directory names are ``<stamp>-exp-<name>-<verb>[-<suffix>…]``, so
    the last stage-word component is the verb, while an experiment NAME that
    happens to contain a stage word (``my-run-thing``) sits earlier and must
    not claim the stage (the old substring scan reported ``run`` for
    ``…-exp-my-run-thing-evaluate``).
    """
    base = os.path.basename(directory.rstrip(os.sep))
    for component in reversed(base.split("-")):
        if component in _RUN_DIRECTORY_STAGES:
            return component
    return fallback


def experiment_name_of_run_directory(directory: str, fallback_verb: str) -> str:
    """The EXPERIMENT a run directory belongs to, from its own name.

    Both failure-retention call sites derived this by hand, identically, and
    identically wrong (2026-07-27):

        name = os.path.basename(directory).split("-exp-")[-1]
        name = name.rsplit(f"-{verb}", 1)[0] if verb in name else name

    That trims the verb only when the CALLER's verb happens to be the
    directory's stage. It is not, whenever a job's kind differs from the stage
    that failed — a ``pipeline`` job whose ``run`` stage dies hands over
    ``…-exp-a-run`` with verb ``pipeline``, so the name came out ``a-run``.
    Then ``_mark_partial_run`` stamps that into the failure record's
    ``experiment`` field, and the job result offers the researcher a targeted
    retry naming an experiment that does not exist. ``split("-exp-")[-1]`` also
    truncates any name that itself contains the delimiter.

    Derived structurally instead, from ``<stamp>-exp-<name>-<stage>[-<suffix>…]``:
    the FIRST ``-exp-`` is the real boundary (a timestamp cannot contain it),
    known trailing suffix components come off, and a trailing STAGE token comes
    off. Same source of truth as `_stage_of_run_directory`, which already
    derives the stage from the directory rather than trusting the caller.

    Falls back to the historical verb-trim when the trailing component is not a
    recognised stage, so an unfamiliar directory shape is never read worse than
    before.
    """
    base = os.path.basename(directory.rstrip(os.sep))
    marker = "-exp-"
    index = base.find(marker)
    if index == -1:
        return base
    remainder = base[index + len(marker):]
    components = remainder.split("-")
    # Suffixes first: the stage token sits UNDER them.
    while len(components) > 1 and _RUN_DIRECTORY_SUFFIX_COMPONENT.match(
            components[-1]):
        components.pop()
    if len(components) > 1 and components[-1] in _RUN_DIRECTORY_STAGES:
        return "-".join(components[:-1])
    suffix = f"-{fallback_verb}"
    return (remainder.rsplit(suffix, 1)[0] if suffix in remainder
            else remainder)


def _mark_partial_run(directory: str, *, verb: str, name: str,
                      exc: BaseException) -> None:
    """Stamp a failure record onto a run directory that has none.

    Stages with something specific to say write their own status (evaluate
    names which judges completed and which did not). This is the floor for
    every other verb: without it a failed `run`'s partial generations would
    come home looking like an ordinary, citable run directory — data with
    no account of itself, which is worse than data plus a refusal.

    Never overwrites an existing status: the stage's own account is always
    the better one.
    """
    # A stage's own status wins; a TORN one does not — replacing an
    # unreadable file with a real failure record is strictly better than
    # leaving a directory that can only be read as "something died here".
    if run_status.has_readable_status(directory):
        return
    stage = _stage_of_run_directory(directory, verb)
    status = run_status.RunStatus(
        directory, stage=stage, experiment=name,
        item_label="record" if stage in ("run", "sweep") else "artifact")
    try:
        status.item_count = _generation_count(directory) or 0
    except Exception:  # noqa: BLE001 - a count is a nicety, not the record
        pass
    status.fail(exc)


def _diagnostic_files(record_path: str | None) -> list[tuple[str, str]]:
    """Scheduler logs and job bookkeeping to carry home with a failure.

    These live beside the SUBMISSION, not in the run directory, so nothing
    would otherwise package them — and on a failed job they are usually the
    most informative thing on the cluster (the traceback, the OOM, the
    module-load error). Returns ``(source, archive_name)`` pairs under
    ``diagnostics/``.

    Best-effort and read-only: a job whose scheduler wrote its logs
    elsewhere simply contributes fewer members, which is a smaller bundle,
    never a failed one."""
    if not record_path:
        return []
    directory = os.path.dirname(os.path.abspath(record_path))
    if not os.path.isdir(directory):
        return []
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    job_id = os.environ.get("SLURM_JOB_ID") or ""
    patterns = [f"slurm-{job_id}.*"] if job_id else []
    # The unqualified globs also catch sibling shards' logs, which is what
    # you want when diagnosing a fan-out: the shard that failed is rarely
    # the one that explains why.
    patterns += ["slurm-*.out", "slurm-*.err", "*.resume"]
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            real = os.path.realpath(path)
            if real in seen or not os.path.isfile(real):
                continue
            seen.add(real)
            found.append((path, os.path.join("diagnostics",
                                             os.path.basename(path))))
    return found


def execute_run_bundle(bundle_path: str, *, verb: str, target_root: str | None = None,
                       dtype: str = "auto", device: str | None = None,
                       prompts_path: str | None = None, source_path: str | None = None,
                       package_evidence_on_complete: bool = True,
                       record_path: str | None = None,
                       checkpoint: "resume_mod.CheckpointFlag | None" = None,
                       shard: str | None = None,
                       resume_from: str | None = None,
                       resume_directory: str | None = None,
                       sample_per_condition=None,
                       sample_seed=None) -> dict:
    """Import a run bundle, execute one experiment verb, and package evidence.

    This is intended as the Slurm child-process entry point. It deliberately
    uses the same task functions as the normal CLI so batch jobs and interactive
    jobs produce identical run artifacts.

    Reliability contract (WS2): when ``record_path`` names this job's child
    record, a resume POINTER lives beside it (``<job>.resume``). On start the
    pointer is consulted — a checkpointed, incomplete run of this same job is
    CONTINUED record-by-record; a completed run is returned idempotently (the
    Slurm requeue re-executes the identical sbatch script, so both cases are
    normal, not errors); anything else starts fresh, as before. ``checkpoint``
    (the headless CLI's SIGUSR1/SIGTERM flag) makes the study loop park the run
    and raise ``CheckpointRequested``, which is re-raised here after the child
    record is stamped ``checkpointed`` — the caller exits 85.
    """
    # Tee this child's stdout into a bounded buffer so the durable record can
    # carry it home. The record's `logs` key existed from day one and was
    # never populated: on a local run the parent's job log captures the
    # streamed lines, but on Slurm the record file IS what comes home — a
    # 144-turn run's per-turn progress and memory-probe readings otherwise
    # live only in a scheduler .out file that evidence packaging may or may
    # not find. Line-buffered writes pass straight through to the real
    # stdout, so parent streaming is unchanged.
    captured_lines: list[str] = []
    sys.stdout = _TeeStdout(sys.stdout, captured_lines)
    try:
        return _execute_run_bundle_inner(
            bundle_path, verb=verb, target_root=target_root, dtype=dtype,
            device=device, prompts_path=prompts_path, source_path=source_path,
            package_evidence_on_complete=package_evidence_on_complete,
            record_path=record_path, checkpoint=checkpoint, shard=shard,
            resume_from=resume_from, resume_directory=resume_directory,
            sample_per_condition=sample_per_condition,
            sample_seed=sample_seed,
            captured_logs=captured_lines)
    finally:
        sys.stdout = sys.stdout.wrapped  # type: ignore[union-attr]


class _TeeStdout:
    """Pass-through stdout that keeps the most recent lines (bounded)."""

    #: Same bound as the parent job manager's in-memory deque.
    LIMIT = 2000

    def __init__(self, wrapped, sink: list[str]):
        self.wrapped = wrapped
        self._sink = sink
        self._partial = ""

    def write(self, text: str) -> int:
        result = self.wrapped.write(text)
        self._partial += text
        *complete, self._partial = self._partial.split("\n")
        self._sink.extend(complete)
        if len(self._sink) > self.LIMIT:
            del self._sink[:len(self._sink) - self.LIMIT]
        return result if isinstance(result, int) else len(text)

    def flush(self) -> None:
        self.wrapped.flush()

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


def _execute_run_bundle_inner(bundle_path: str, *, verb: str,
                              target_root: str | None, dtype: str,
                              device: str | None, prompts_path: str | None,
                              source_path: str | None,
                              package_evidence_on_complete: bool,
                              record_path: str | None,
                              checkpoint: "resume_mod.CheckpointFlag | None",
                              shard: str | None, resume_from: str | None,
                              captured_logs: list[str],
                              resume_directory: str | None = None,
                              sample_per_condition=None,
                              sample_seed=None) -> dict:
    # The seeded-subsample ask is checked FIRST, before the bundle is even
    # opened (2026-08-29): it needs nothing from the bundle, and a malformed
    # ask must leave the workspace untouched — on this path `import_bundle`
    # writes into the target root well before the task is called.
    if (sample_per_condition is not None or sample_seed is not None) \
            and verb != "evaluate":
        # A flag a verb cannot honour is REFUSED, never dropped — the rule
        # `--shard` and `--resume` follow below. A `run` that silently ignored
        # this would generate the full matrix while its command line said
        # otherwise.
        raise BundleError(
            "--sample-per-condition/--sample-seed apply to the 'evaluate' "
            f"verb only (got {verb!r}) — they choose which of a completed "
            "run's records are coded, and no other verb reads a prior run's "
            "records that way")
    from . import evaluate_subsample
    evaluate_subsample.resolve_request(sample_per_condition, sample_seed,
                                       program="steerlab-server")
    meta = inspect_bundle(bundle_path)
    if meta.get("kind") != "runBundle":
        raise BundleError("bundle execute requires a runBundle")
    # Shard filter (multi-GPU fan-out): only the run verb has the
    # per-record-independent record set the shard partition is defined over.
    shard_spec = None
    if shard is not None:
        from . import sharding as sharding_mod
        if verb != "run":
            raise BundleError(
                f"--shard applies to the 'run' verb only (got {verb!r}) — "
                "other verbs have no independent per-record record set to "
                "partition")
        shard_spec = sharding_mod.parse_shard(shard)
    # A flag a verb cannot honour is REFUSED, never dropped (open-issues §16,
    # the same rule `--shard` follows above): a correct-looking resume that
    # silently started a fresh run would spend the whole allocation again and
    # look like it worked.
    if resume_directory is not None and verb not in ("run", "sweep", "pipeline"):
        raise BundleError(
            f"--resume applies to the 'run', 'sweep' and 'pipeline' verbs only "
            f"(got {verb!r}) — the other verbs write a fresh run directory "
            "each time and have no checkpoint to continue")
    target = os.path.realpath(target_root or paths.project_root())
    # Anchor every caller-supplied RUN-DIRECTORY path to the target root
    # BEFORE anything reads it. The rendered sbatch cd's into its own slurm
    # directory before srun, so the child's cwd is not the workspace and a
    # relative path resolved somewhere nobody meant (ledger 2026-08-21).
    # `--prompts` needs nothing here: `_load_prompts` already joins a relative
    # prompts file to the root it is handed, which is this same `target`.
    source_path = resolve_against_target(source_path, target)
    resume_directory = resolve_against_target(resume_directory, target)
    imported = import_bundle(bundle_path, target_root=target, allow_overwrite=True)
    name = meta["experiment"]

    from . import tasks
    from .manifest import Manifest

    started = time.time()
    run_directory: str | None = None
    status = "succeeded"
    error: str | None = None
    result: dict = {
        "experiment": name,
        "verb": verb,
        "targetRoot": target,
        "imported": imported,
        "startedAt": started,
    }
    if shard_spec is not None:
        result["shard"] = {"index": shard_spec.index,
                          "count": shard_spec.count}

    # Scope the retention context to THIS unit of work. Without a reset a
    # second execute in the same process would inherit the first's
    # directory and package the wrong run — worse than packaging none.
    run_directory_token = run_status.current_run_directory.set(None)
    pointer_path = (resume_mod.pointer_path_for_record(record_path)
                    if record_path else None)
    resume_dir: str | None = None
    complete_dir: str | None = None
    if resume_directory is not None:
        # An EXPLICIT `--resume <run-dir>` (the operator named a parked
        # directory) outranks the pointer, which only remembers what THIS job
        # record started. Resuming a run parked by an earlier job — the reason
        # an operator writes a raw sbatch at all — has no pointer to consult,
        # so the pointer's disposition must not be allowed to override or
        # contradict what was asked for. The resume ADMISSION gate still
        # applies inside the task (complete directories refuse; a drifted
        # manifest refuses); this only chooses which directory it judges.
        resume_dir = resume_directory
        result["resumedFrom"] = resume_directory
    elif verb in ("run", "sweep") and pointer_path is not None:
        # Sweeps checkpoint/resume too (2026-08-03: a walltime-killed sweep
        # lost ~20 minutes of finished cells) — same pointer contract, with
        # recommendations.json as the sweep's completion marker.
        disposition, pointed = resume_mod.resolve_pointer(pointer_path, verb=verb)
        if disposition == "resume":
            resume_dir = pointed
            result["resumedFrom"] = pointed
        elif disposition == "complete":
            complete_dir = pointed
            result["alreadyComplete"] = True
    resume_pipeline_dir: str | None = None
    if verb == "pipeline" and resume_directory is not None:
        resume_pipeline_dir = resume_directory
    elif verb == "pipeline" and pointer_path is not None:
        # Pipeline resume classification is the LEDGER's job (pipeline.json:
        # completed stages skip; a completed/aborted disposition returns
        # idempotently — an aborted chain is a recorded scientific stop that
        # a requeue must NOT re-run). The pointer only remembers which
        # directory this job started.
        pointer = resume_mod.read_pointer(pointer_path) or {}
        pointed = pointer.get("runDirectory")
        if (pointer.get("verb") == verb and pointed
                and os.path.isdir(pointed)):
            resume_pipeline_dir = pointed
            result["resumedFrom"] = pointed

    # The directory a stage created, captured the moment it exists rather
    # than on return: a run that fails midway never reaches the assignment
    # of ``run_directory``, and its partial evidence would otherwise be
    # unreachable from the failure path (retention 2026-07-24).
    created_directory: list[str] = []

    def _note_run_directory(created: str) -> None:
        # Persist the run-dir pointer the moment the directory exists, so a
        # requeued re-execution of this same script finds it. Best-effort: a
        # pointer write failure must not kill the run, only its resumability.
        created_directory.append(created)
        if pointer_path is None:
            return  # retention-only caller: nothing to make resumable
        try:
            resume_mod.write_pointer(pointer_path, created, verb=verb,
                                     experiment=name)
        except OSError as exc:
            result["pointerError"] = f"{type(exc).__name__}: {exc}"

    try:
        if verb == "verify":
            result["violations"] = Manifest.load(name, target).verify(target)
        elif verb == "extract":
            run_directory = tasks.extract(name, target, dtype, device)
        elif verb == "validate":
            run_directory = tasks.validate(name, target, dtype, device)
        elif verb == "sweep":
            if complete_dir is not None:
                run_directory = complete_dir
            else:
                run_directory = tasks.sweep(
                    name, target, dtype, device,
                    checkpoint=checkpoint, run_directory=resume_dir,
                    on_run_directory=_note_run_directory)
        elif verb == "run":
            if complete_dir is not None:
                run_directory = complete_dir
            else:
                run_directory = tasks.run(
                    name, prompts_path, target, dtype, device,
                    checkpoint=checkpoint, run_directory=resume_dir,
                    on_run_directory=_note_run_directory,
                    shard=shard_spec)
        elif verb == "evaluate":
            # `resume_from` completes a FAILED evaluation by judging only
            # the cells it never decided (2026-07-24). Plumbed here so the
            # retry is reachable from the app and the cluster, not just the
            # CLI — the affordance that told a researcher a partial was
            # "retryable" previously had nothing behind it.
            # `sample_per_condition`/`sample_seed` draw a seeded, stratified
            # subsample of the source run's records instead of coding all of
            # them (2026-08-29). Threaded here for the same reason
            # `resume_from` is: the cluster is where a judged evaluate of a
            # large corpus actually runs, so the design has to be expressible
            # from the submission and not only from a Mac terminal.
            run_directory = tasks.evaluate(
                name, target, source_path, resume_from=resume_from,
                sample_per_condition=sample_per_condition,
                sample_seed=sample_seed)
        elif verb == "analyze":
            # Statistics-only pass over a prior run (no model load). The
            # epoch and measurement-drift guards live inside tasks.analyze
            # itself — same enforcement as the direct API/CLI paths. The
            # source run threads exactly as evaluate's (--source /
            # sourcePath); absent, the newest completed run is analyzed.
            run_directory = tasks.analyze(name, target, source_path)
        elif verb == "pipeline":
            run_directory = tasks.pipeline(
                name, target, dtype, device, checkpoint=checkpoint,
                pipeline_run_directory=resume_pipeline_dir,
                on_pipeline_directory=_note_run_directory)
        else:
            raise BundleError(f"unsupported bundled experiment verb {verb!r}")
        if run_directory:
            result["runDirectory"] = run_directory
            if verb == "pipeline":
                # Judge fan-out handshake (2026-07-23): a pipeline whose
                # evaluate stage emitted packets for local judge workers
                # STOPPED at that stage — the child record carries the
                # fan-out request so the controller submits one worker per
                # distinct judge model, and NO evidence is packaged (the
                # pipeline is not finished; a bundle here would look
                # imported while judgments are still pending).
                request = tasks.read_judge_fanout_request(run_directory)
                if request is not None and _pipeline_awaits_judgment(
                        run_directory):
                    result["awaitingJudgeFanout"] = request
            if package_evidence_on_complete \
                    and "awaitingJudgeFanout" not in result:
                result["evidenceBundle"] = package_evidence(run_directory)
    except resume_mod.CheckpointRequested as exc:
        # Not a failure: the run is durably parked and resumable. The child
        # record says so, and the exception continues to the exit-85 path.
        status = "checkpointed"
        run_directory = exc.run_directory
        result["runDirectory"] = exc.run_directory
        result["resumeState"] = {"completedRecords": exc.completed_records,
                                 "reason": exc.reason}
        # Marked as CHECKPOINTED, never failed (2026-07-24): it is still
        # incomplete, so no evidence gate may take it, but calling a
        # working requeue a failure would train the researcher to ignore
        # the marker that actually matters.
        try:
            if not run_status.has_readable_status(exc.run_directory):
                parked = run_status.RunStatus(
                    exc.run_directory, stage=verb, experiment=name,
                    item_label="record")
                parked.item_count = exc.completed_records or 0
                parked.checkpointed(reason=exc.reason)
        except Exception:  # noqa: BLE001 - never break a working requeue
            pass
        raise
    except Exception as exc:  # noqa: BLE001 - recorded for child-job reconciliation
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        result["error"] = error
        # Retention (2026-07-24): the failure stands, but whatever the stage
        # actually produced comes HOME. Previously this path recorded the
        # error and re-raised, so a failed job left its generations,
        # judgments, and logs reachable only over SSH — "the data still
        # exists somewhere under /scratch" was the retention policy.
        # REPORTED beats OBSERVED (external review round 2, finding 4).
        #
        # `created_directory` is what the verb itself declared through its
        # callback; for a pipeline that is the chain ROOT. The exception
        # attribute and the context variable observe whichever directory
        # was created most recently — the failing STAGE. A stage directory
        # is more specific but far less complete: only the root carries
        # `pipeline.json`, and only from the root does `package_evidence`
        # follow the ledger to every stage. Packaging the stage alone ships
        # the evaluate partial and silently drops the generations, which
        # are the expensive output and the whole point of retention.
        observed = run_status.partial_run_directory(exc)
        # Verbs that never report their directory (extract, validate,
        # sweep) fall back to a strict slug+time match, or None when that
        # would be a guess.
        reported = created_directory[-1] if created_directory else None
        stage_partial: str | None = None
        if verb == "pipeline" and reported:
            partial = reported
            if observed and os.path.realpath(observed) != os.path.realpath(reported):
                stage_partial = observed
        else:
            partial = (observed or reported
                       or _partial_directory_for(verb, name, target, started))
        # Mark BEFORE packaging, so the bundle carries failure records
        # rather than unexplained directories. The failing STAGE gets its
        # own marker (that is where the specifics are), and the chain root
        # gets one too — otherwise an imported pipeline root reads as
        # unannotated, i.e. citable. Best-effort: never let a marker's
        # failure mask the run's.
        for directory in filter(None, (stage_partial, partial)):
            try:
                _mark_partial_run(directory, verb=verb, name=name, exc=exc)
            except Exception:  # noqa: BLE001
                pass
        if package_evidence_on_complete and partial:
            try:
                result["evidenceBundle"] = package_evidence(
                    partial,
                    failure={"error": error,
                             "errorType": type(exc).__name__,
                             "verb": verb, "experiment": name,
                             "traceback": traceback.format_exc()},
                    extra_files=_diagnostic_files(record_path),
                    # A pipeline stage that died is usually absent from the
                    # ledger — nothing recorded a stage that never finished
                    # — so name it explicitly or its partial output is lost
                    # even though the chain root was packaged.
                    extra_run_directories=[stage_partial] if stage_partial else None)
                result["partialEvidence"] = True
                # What a client needs to offer a targeted RETRY, at the
                # RESULT root where the app reads it. Without these the
                # retry affordance could never appear on a bundled/Slurm
                # job — which is the case where redoing a judged evaluate
                # is most expensive.
                result["experiment"] = name
                result["verb"] = verb
                result["partialRunID"] = os.path.basename(
                    (stage_partial or partial).rstrip(os.sep))
            except Exception as pack_exc:  # noqa: BLE001
                # A packaging failure must never replace the real error —
                # the researcher would debug the packager instead of the
                # run. It is recorded and the original exception continues.
                result["evidencePackagingError"] = \
                    f"{type(pack_exc).__name__}: {pack_exc}"
        raise
    finally:
        run_status.current_run_directory.reset(run_directory_token)
        result["finishedAt"] = time.time()
        if record_path:
            write_child_record(record_path, kind=f"bundle-execute:{verb}",
                               status=status, result=result, error=error,
                               # The teed child stdout (bounded): per-turn
                               # progress and memory-probe lines survive in
                               # the record that comes home from Slurm.
                               logs=list(captured_logs),
                               elapsed_seconds=time.time() - started,
                               record_count=_generation_count(run_directory))
    return result


def _pipeline_awaits_judgment(pipeline_directory: str) -> bool:
    """Whether the pipeline ledger's evaluate stage is still awaiting
    judgments — a completed/adopted evaluate must NOT re-trigger the
    fan-out on a resumed continuation."""
    try:
        with open(os.path.join(pipeline_directory, "pipeline.json"),
                  encoding="utf-8") as handle:
            ledger = json.load(handle)
    except (OSError, ValueError):
        return False
    stage = (ledger.get("stageResults") or {}).get("evaluate") or {}
    return stage.get("status") == "awaitingJudgment"


def _generation_count(run_directory: str | None) -> int | None:
    """Cheap record count for the child record: lines in generations.jsonl."""
    if not run_directory:
        return None
    path = os.path.join(run_directory, "generations.jsonl")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def write_child_record(record_path: str, *, kind: str, status: str,
                       result: dict | None = None, error: str | None = None,
                       logs: list[str] | None = None,
                       elapsed_seconds: float | None = None,
                       record_count: int | None = None) -> None:
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "id": os.environ.get("STEERLAB_JOB_ID") or os.path.splitext(os.path.basename(record_path))[0],
        "kind": kind,
        "status": status,
        "executor": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "executorJobID": os.environ.get("SLURM_JOB_ID"),
        "finishedAt": time.time(),
        "result": result,
        "error": error,
        "logs": logs or [],
    }
    # Throughput/provenance fields (WS2 contract): stamped when cheaply known,
    # never fabricated — the reconciler folds them into the job result.
    if elapsed_seconds is not None:
        payload["elapsedSeconds"] = float(elapsed_seconds)
    if record_count is not None:
        payload["recordCount"] = int(record_count)
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


# Engine-default input paths, read at execute time when the manifest declares
# no replacement. MUST match tasks.sweep / battery.DEFAULT_BATTERY_FILE.
_DEFAULT_DEV_PROMPTS_FILE = "prompts/dev/dev-prompts.jsonl"


def _experiment_files(manifest: Manifest, base: str, root: str | None) -> list[tuple[str, str]]:
    """Every file a run bundle must carry: the manifest plus the WHOLE pin
    surface, derived mechanically from the same enumeration the freeze
    cleanliness gate and pinned/ snapshot use
    (``experiment_store.pinned_input_entries``) — a pin kind added there is
    packed here automatically, never hand-listed. A REQUIRED pinned input
    missing on disk fails packaging loudly (a bundle that silently lacks a
    pinned input only fails child-side, hours later, on the cluster)."""
    out: list[tuple[str, str]] = []
    exp_path = experiment_store._path(manifest.name, root)  # same canonical path as authoring
    out.append((exp_path, os.path.relpath(exp_path, base)))

    for entry in experiment_store.pinned_input_entries(manifest.raw, root):
        rel = os.path.relpath(entry.path, base)
        if not os.path.exists(entry.path):
            if entry.required:
                raise BundleError(
                    f"cannot package '{manifest.name}': pinned input missing "
                    f"— {entry.label} at {rel}")
            continue
        if rel.startswith(".."):
            raise BundleError(
                f"cannot package '{manifest.name}': pinned input outside the "
                f"workspace cannot be bundled — {entry.label} at {entry.path}")
        if os.path.isdir(entry.path):
            out.extend(_walk(entry.path, rel))
        else:
            out.append((entry.path, rel))

    # Implicit engine defaults (not manifest-declared, read at execute time):
    # packed when present so the bundle stays executable for every verb on
    # the remote engine. Optional by definition — a missing default is the
    # remote engine's ordinary refusal, not a packaging failure.
    defaults: list[str] = []
    sweep = manifest.raw.get("sweep") if isinstance(manifest.raw.get("sweep"), dict) else {}
    if not manifest.raw.get("capabilityBatteryFile"):
        from . import battery as battery_mod
        defaults.append(battery_mod.DEFAULT_BATTERY_FILE)
    if not sweep.get("devPromptsFile"):
        defaults.append(_DEFAULT_DEV_PROMPTS_FILE)
    if not sweep.get("batteryFile"):
        from . import battery as battery_mod
        defaults.append(battery_mod.DEFAULT_BATTERY_FILE)
    for rel in defaults:
        path = os.path.join(base, rel)
        if os.path.exists(path):
            out.append((path, rel))
    return _dedupe_existing(out)


def _walk(directory: str, rel_base: str) -> Iterable[tuple[str, str]]:
    if not os.path.isdir(directory):
        return []
    out = []
    for dirpath, _dirs, filenames in os.walk(directory):
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.join(rel_base, os.path.relpath(path, directory))
            out.append((path, rel))
    return out


def _dedupe_existing(files: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for path, rel in files:
        real = os.path.realpath(path)
        if real in seen or not os.path.exists(real) or os.path.isdir(real):
            continue
        seen.add(real)
        out.append((real, rel))
    return out


def _add_files(tar: tarfile.TarFile, files: Iterable[tuple[str, str]]) -> list[BundleEntry]:
    entries: list[BundleEntry] = []
    for path, rel in files:
        rel = rel.replace(os.sep, "/")
        tar.add(path, arcname=rel, recursive=False)
        entries.append(BundleEntry(path=rel, sha256=sha256_file(path),
                                   bytes=os.path.getsize(path)))
    return entries


def _add_json(tar: tarfile.TarFile, arcname: str, payload: dict) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = time.time()
    import io
    tar.addfile(info, io.BytesIO(data))


#: Returned when the entry list does not NAME the member at all — a different
#: failure from an entry that names it and pins nothing, and one that deserves
#: its own refusal. ``None`` cannot say which, because ``"sha256": null`` is
#: itself one of the malformed pins being caught (external review,
#: 2026-09-05).
_UNLISTED = object()


def _entry_hash(meta: dict, path: str):
    """The digest the bundle's entry list DECLARES for ``path``, exactly as
    declared — or :data:`_UNLISTED`. Validation is the caller's
    (:func:`_require_digest`); this only looks it up."""
    for entry in meta.get("entries", []):
        if entry.get("path") == path:
            return entry.get("sha256")
    return _UNLISTED


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

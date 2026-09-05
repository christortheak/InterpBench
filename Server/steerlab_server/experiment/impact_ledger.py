"""The REM-01 impact ledger: which EXISTING artifacts the 2026-09-05 science
fixes reach, with the evidence read for every verdict.

Four defects were fixed on 2026-09-05 (external review). Each fix landed with
a regression suite, and those suites establish that the behaviour is covered
*from now on* — they say nothing whatever about the artifacts already sitting
in a workspace. This module answers that second question, and only that one:
for every artifact a fix could have reached, what does the artifact's own
record say, and what does the researcher therefore have to do about it?

Three rules shape everything here, because the ledger is evidence about
evidence and inherits its discipline:

* **Missing metadata is never evidence of non-exposure.** An artifact that
  cannot say which build produced it is ``unknown``, never ``unaffected``.
  ``unknown`` is a real, actionable state — the entry names the fact it is
  missing and the ``requiredAction`` is ``resolveProvenance``.
* **Nothing is written into a run, an analysis, or a frozen artifact.**
  Everything this module produces lands in a fresh
  ``diagnostics/impact-ledger-<stamp>/`` directory. A correction is a NEW
  artifact carrying a provenance link back to the original, which is exactly
  what ``promoted-movers.reassessed.json`` is.
* **The tool classifies; the researcher disposes.** ``exposure`` is derived
  from bytes and ``disposition`` starts at ``unresolved`` for anything the
  evidence does not settle. Filling dispositions, naming owners, and linking
  replacement artifacts is the researcher's job and is deliberately not
  automated: "this run is superseded by that one" is a scientific claim.

Screening, not proof. Several rules here are SCREENS: an accumulation factor
above one says the pre-fix LoRA objective was reachable, not that the trained
weights moved by any particular amount; a pre-fix build stamp on a
full-vocabulary J-lens readout says the gain was missing, not how far the
token order shifted. Each entry's ``assessment`` says which kind of statement
it is making.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

from . import paths

#: The LEDGER document's version, never an artifact's.
SCHEMA_VERSION = 1

#: Where a ledger lands. A sibling of ``runs/`` rather than a run: the ledger
#: describes runs, is regenerated whenever the question is re-asked, and must
#: never be mistaken for measured evidence.
DIAGNOSTICS_SUBTREE = "diagnostics"
OUTPUT_PREFIX = "impact-ledger-"
LEDGER_JSON = "impact-ledger.json"
LEDGER_MARKDOWN = "impact-ledger.md"

#: Reassessed promotions live one directory down, keyed by the analysis run
#: they reassess, so the required filename stays exactly what it is while a
#: workspace with twenty screens produces twenty unambiguous files.
REASSESSED_SUBTREE = "reassessed"
REASSESSED_PROMOTIONS = "promoted-movers.reassessed.json"

EXPOSED = "exposed"
UNAFFECTED = "unaffected"
UNKNOWN = "unknown"
#: The closed exposure vocabulary, in decreasing severity.
EXPOSURES = (EXPOSED, UNKNOWN, UNAFFECTED)

#: The closed required-action vocabulary. ``rerun`` and ``recompute`` are
#: deliberately different verbs: a readout whose producer did not retain its
#: residuals can only be produced again by RUNNING the model, while a
#: promotion decision is a pure function of an effect table that is still on
#: disk. ``resolveProvenance`` is the action for an ``unknown``: find the
#: build, then re-run this verb.
REQUIRED_ACTIONS = ("none", "reassess", "recompute", "rerun",
                    "resolveProvenance")

#: The closed disposition vocabulary. ``unaffected`` is the ONLY disposition
#: this tool ever writes on its own, and only where the artifact's own bytes
#: establish it; everything else starts ``unresolved`` and is the
#: researcher's to close.
DISPOSITIONS = ("unaffected", "unresolved")

#: The ledger entry's CLOSED key set, sorted. A reader keys by name and a
#: writer may not invent a field: an entry that carried an ad-hoc key would
#: make two ledgers incomparable, which is the one thing a ledger is for.
ENTRY_KEYS = (
    "artifactID", "artifactType", "assessment", "disposition", "evidence",
    "exposure", "finding", "owner", "path", "pins", "producingRevision",
    "replacementArtifact", "requiredAction",
)


class LedgerRefusal(Exception):
    """A typed refusal, carrying the state the envelope should report and the
    runnable repair. Raised rather than returned because every one of these
    means the scan cannot start at all."""

    def __init__(self, *, code: str, reason: str, repair_action: str,
                 state: str = "refused") -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.repair_action = repair_action
        self.state = state


# ---------------------------------------------------------------------------
# The findings table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """One 2026-09-05 fix, as DATA: the ledger's classification rules, the
    ancestry test, and the rendered document all read this table, so a fifth
    finding is a row here and a classifier, not a new document shape."""

    id: str
    #: The commit(s) that fixed it. An artifact is post-fix only when EVERY
    #: one of them is an ancestor of its producing revision — a partial fix
    #: is not a fix.
    fix_commits: tuple[str, ...]
    landed: str
    summary: str
    #: True when exposure cannot be decided from the artifact's own bytes and
    #: needs the producing revision (and therefore ``--code-checkout``).
    revision_dependent: bool


FINDINGS: tuple[Finding, ...] = (
    Finding(
        id="SCI-01",
        fix_commits=("ff4c47a",),
        landed="2026-09-05",
        summary="The J-lens FULL-VOCABULARY readout projected the transported "
                "residual through the output head without the final-norm gain "
                "g = 1 + norm.weight; because g varies by coordinate this "
                "reordered tokens rather than rescaling them.",
        revision_dependent=True),
    Finding(
        id="SCI-02",
        fix_commits=("d280762",),
        landed="2026-09-05",
        summary="Python LoRA gradient accumulation divided each micro-batch's "
                "MEAN loss by the nominal accumulation factor — a mean of "
                "means weighted by the micro-batch cut, and an incomplete "
                "final group scaled down by a factor it never filled.",
        revision_dependent=False),
    Finding(
        id="SCI-03",
        fix_commits=("c692877",),
        landed="2026-09-05",
        summary="The Swift/MLX instruction trainer rendered the COMPLETED "
                "answer with a generation prompt appended, so the supervised "
                "target carried template tokens no answer contains.",
        revision_dependent=True),
    Finding(
        id="SCI-04",
        fix_commits=("00275b1", "3cb8a59", "5309b5f"),
        landed="2026-09-05",
        summary="A flat ladder, or one with a single distinct dose, counted "
                "as a monotone dose response — and an SAE qualification's "
                "declared dose response was never checked against its own "
                "rows.",
        revision_dependent=False),
)

FINDINGS_BY_ID = {finding.id: finding for finding in FINDINGS}


def findings_table() -> list[dict]:
    """The table as the envelope and the document carry it."""
    return [{"finding": f.id, "fixCommits": list(f.fix_commits),
             "landed": f.landed, "revisionDependent": f.revision_dependent,
             "summary": f.summary}
            for f in FINDINGS]


# ---------------------------------------------------------------------------
# Ancestry — "was this artifact produced before the fix?"
# ---------------------------------------------------------------------------

#: Seconds any single git invocation may take. A checkout on a cold network
#: filesystem is slow; a HANGING one must not take the ledger with it.
GIT_TIMEOUT_SECONDS = 20.0


class Ancestry:
    """``git merge-base --is-ancestor <fix> <producing revision>``, memoized.

    The question a build stamp answers is ancestry, not ordering: two commits
    on different branches have timestamps that say nothing about which code
    ran. Without a checkout to ask, every revision-dependent finding is
    ``unknown`` — that is the honest answer, and the entry says how to get a
    better one.

    A commit the checkout does not contain is ``unknown`` WITH THE REASON,
    never "not an ancestor": a workspace produced by a build whose commit was
    never pushed is exactly the case where guessing does the most damage.
    """

    def __init__(self, checkout: str | None) -> None:
        self.checkout = checkout
        self._cache: dict[tuple[str, str], tuple[bool | None, str]] = {}

    @property
    def available(self) -> bool:
        return bool(self.checkout)

    def _is_ancestor(self, fix: str, revision: str) -> tuple[bool | None, str]:
        key = (fix, revision)
        if key in self._cache:
            return self._cache[key]
        try:
            proc = subprocess.run(
                ["git", "-C", self.checkout, "merge-base", "--is-ancestor",
                 fix, revision],
                capture_output=True, text=True,
                timeout=GIT_TIMEOUT_SECONDS, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            answer = (None, f"git could not answer whether {fix} is an "
                            f"ancestor of {revision}: {exc}")
        else:
            if proc.returncode == 0:
                answer = (True, f"{fix} is an ancestor of {revision} in "
                                f"{self.checkout}")
            elif proc.returncode == 1:
                answer = (False, f"{fix} is NOT an ancestor of {revision} in "
                                 f"{self.checkout}")
            else:
                detail = (proc.stderr or "").strip().splitlines()
                answer = (None,
                          f"{self.checkout} cannot resolve {fix} or "
                          f"{revision} as commits"
                          + (f" ({detail[0]})" if detail else ""))
        self._cache[key] = answer
        return answer

    def carries_fix(self, finding: Finding,
                    revision: str | None) -> tuple[bool | None, list[str]]:
        """``(True post-fix, False pre-fix, None unresolvable)`` plus the
        facts read, ready to go straight into an entry's ``evidence``."""
        if revision is None:
            return None, [
                f"{finding.id}: the artifact carries no producing revision, "
                "so the code that produced it cannot be dated"]
        if not self.available:
            return None, [
                f"{finding.id}: producing revision {revision} was not dated — "
                "no --code-checkout was given, so "
                f"{'/'.join(finding.fix_commits)} could not be tested for "
                "ancestry"]
        facts: list[str] = []
        verdict: bool | None = True
        for fix in finding.fix_commits:
            answer, why = self._is_ancestor(fix, revision)
            facts.append(why)
            if answer is None:
                verdict = None
                break
            if answer is False:
                # A partial fix is not a fix: one missing commit settles it.
                verdict = False
                break
        return verdict, facts


def open_checkout(path: str | None) -> Ancestry:
    """Validate ``--code-checkout`` before a single artifact is read: a typo'd
    path that silently degraded every entry to ``unknown`` would produce a
    ledger that looks like a finding."""
    if not path:
        return Ancestry(None)
    resolved = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    if not os.path.isdir(resolved):
        raise LedgerRefusal(
            code="codeCheckoutNotFound", state="blocked",
            reason=f"--code-checkout {path!r} is not a directory",
            repair_action="pass --code-checkout <path to a git checkout of "
                          "this repository>, or omit it and accept 'unknown' "
                          "for the revision-dependent findings")
    try:
        proc = subprocess.run(
            ["git", "-C", resolved, "rev-parse", "--git-dir"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS,
            check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LedgerRefusal(
            code="codeCheckoutUnreadable", state="blocked",
            reason=f"--code-checkout {path!r} could not be read by git: {exc}",
            repair_action="check that git is installed and the path is "
                          "readable") from exc
    if proc.returncode != 0:
        raise LedgerRefusal(
            code="codeCheckoutNotAGitRepository", state="blocked",
            reason=f"--code-checkout {path!r} is not a git repository",
            repair_action="pass a git checkout of this repository, so a "
                          "build stamp can be tested for ancestry against "
                          "the fix commits")
    return Ancestry(resolved)


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def require_workspace(root: str | None = None) -> str:
    """The data workspace this ledger describes, or a typed refusal.

    A workspace is a tree with the study subtrees in it. The SOURCE CHECKOUT
    is refused by name (``AGENTS.md``: study data never lives in the
    checkout) — pointing the ledger at the code would write a diagnostics
    directory into the repository and describe nothing.
    """
    resolved = os.path.realpath(os.path.abspath(root or paths.project_root()))
    if not os.path.isdir(resolved):
        raise LedgerRefusal(
            code="workspaceNotFound", state="notFound",
            reason=f"{resolved} is not a directory",
            repair_action="steerlab-server --root <workspace> ledger impact")
    if paths.looks_like_source_checkout(resolved):
        raise LedgerRefusal(
            code="rootIsSourceCheckout", state="notFound",
            reason=f"{resolved} is the SteerLab source checkout, not a data "
                   "workspace — a ledger describes study artifacts, and the "
                   "checkout holds none",
            repair_action="steerlab-server --root <workspace> ledger impact")
    has_marker = os.path.isfile(os.path.join(resolved, "WORKSPACE.md"))
    has_subtree = any(os.path.isdir(os.path.join(resolved, sub))
                      for sub in ("runs", "experiments", "prompts"))
    if not (has_marker or has_subtree):
        raise LedgerRefusal(
            code="workspaceNotFound", state="notFound",
            reason=f"{resolved} carries no WORKSPACE.md and none of runs/, "
                   "experiments/, prompts/ — it is not a workspace",
            repair_action="steerlab-server --root <workspace> ledger impact")
    return resolved


# ---------------------------------------------------------------------------
# Small readers (tolerant on purpose: an unreadable artifact is a FACT)
# ---------------------------------------------------------------------------

def _read_json(path: str):
    """The parsed document, or ``None``. Every caller treats ``None`` as a
    fact to record rather than an error to raise: a half-written artifact in a
    workspace is exactly what a ledger exists to notice."""
    try:
        with open(path, "rb") as handle:
            return json.loads(handle.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _sha256_file(path: str) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _relative(path: str, root: str) -> str:
    """Workspace-relative, POSIX-spelled. A ledger is read on the machine that
    fixes the artifacts, which is not always the one that scanned them.

    BOTH sides are symlink-resolved before the relpath: a workspace reached
    through a symlink (``/tmp`` → ``/private/tmp``, a ``/scratch`` mount) would
    otherwise produce a path full of ``..`` segments that names nothing.
    """
    return os.path.relpath(os.path.realpath(path),
                           os.path.realpath(root)).replace(os.sep, "/")


def _commit_token(stamp: str | None) -> tuple[str | None, bool]:
    """``(commit, dirty)`` from a build-identity string.

    Both engines spell it ``<name> <version>+<sha8>[-dirty]``
    (:mod:`build_identity`, Swift ``SteerLabVersion``), so the token after the
    last ``+`` is the commit. A ``-dirty`` suffix is kept as a separate fact:
    it means the worktree differed from the commit, which weakens — but does
    not void — the ancestry answer.
    """
    if not isinstance(stamp, str) or "+" not in stamp:
        return None, False
    token = stamp.rsplit("+", 1)[1].strip()
    dirty = token.endswith("-dirty")
    if dirty:
        token = token[:-len("-dirty")]
    return (token or None), dirty


_STAMP_KEYS = ("runType", "appVersion", "modelID", "revision", "experiment",
               "experimentHash", "substrate", "createdAt")


def _run_stamp(run_dir: str) -> dict:
    """The canonical ``config.json`` fields, tolerantly — a legacy or
    foreign-shaped file degrades to nulls per field, never to an exception
    (the same rule ``catalog._read_run_stamp`` follows)."""
    out = {key: None for key in _STAMP_KEYS}
    data = _read_json(os.path.join(run_dir, "config.json"))
    if not isinstance(data, dict):
        return out
    for key in _STAMP_KEYS:
        value = data.get(key)
        if isinstance(value, str):
            out[key] = value
    return out


def _pins(**fields) -> dict:
    """The pins an artifact actually carries. Absent pins are OMITTED rather
    than nulled: ``pins`` is "what this artifact says it was measured
    against", and a null there would read as a recorded absence."""
    return {key: value for key, value in sorted(fields.items())
            if value not in (None, "", [], {})}


def _entry(*, artifact_id: str, artifact_type: str, path: str,
           producing_revision: str | None, pins: dict, finding: str,
           exposure: str, evidence: list, assessment: str,
           required_action: str) -> dict:
    """One ledger entry, key-checked at construction.

    ``disposition`` is derived, not passed: ``unaffected`` is the only one
    this tool writes, and only where the exposure verdict is itself
    ``unaffected``. Everything a human must still look at starts
    ``unresolved``.
    """
    if exposure not in EXPOSURES:
        raise ValueError(f"unknown exposure {exposure!r}")
    if required_action not in REQUIRED_ACTIONS:
        raise ValueError(f"unknown requiredAction {required_action!r}")
    if finding not in FINDINGS_BY_ID:
        raise ValueError(f"unknown finding {finding!r}")
    entry = {
        "artifactID": artifact_id,
        "artifactType": artifact_type,
        "assessment": assessment,
        "disposition": UNAFFECTED if exposure == UNAFFECTED else "unresolved",
        "evidence": list(evidence),
        "exposure": exposure,
        "finding": finding,
        "owner": None,
        "path": path,
        "pins": dict(pins),
        "producingRevision": producing_revision,
        "replacementArtifact": None,
        "requiredAction": required_action,
    }
    assert tuple(sorted(entry)) == ENTRY_KEYS, "entry key set drifted"
    return entry


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

#: How far below a scanned subtree a candidate may sit. ``runs/fine-tunes/
#: <stamp>-fine-tune-<slug>/`` is two, an adapter weight directory inside one
#: is three; the cap keeps a workspace with a deep incidental tree (a cloned
#: model cache dropped under ``runs/``) from turning the scan into a full
#: filesystem walk.
MAX_SCAN_DEPTH = 4

#: Subtrees a candidate artifact can live in. ``adapters/`` is one of the
#: workspace's rebasable roots (``paths._REBASABLE_ROOTS``) and some
#: workspaces keep adapters there rather than in a run.
SCANNED_SUBTREES = ("runs", "adapters")

_SKIPPED_DIRECTORY_NAMES = {".git", "__pycache__", ".ipynb_checkpoints"}


def candidate_directories(root: str) -> list[str]:
    """Every directory under the scanned subtrees, depth-capped and sorted.

    Sorted absolute paths, so two scans of the same workspace produce the same
    ledger in the same order — a diff between two ledgers has to be about the
    artifacts.
    """
    found: list[str] = []
    for subtree in SCANNED_SUBTREES:
        base = os.path.join(root, subtree)
        if not os.path.isdir(base):
            continue
        found.append(base)
        for current, directories, _files in os.walk(base):
            directories[:] = sorted(
                d for d in directories if d not in _SKIPPED_DIRECTORY_NAMES
                and not d.startswith("."))
            depth = current[len(base):].count(os.sep)
            if depth >= MAX_SCAN_DEPTH:
                directories[:] = []
                continue
            found.extend(os.path.join(current, d) for d in directories)
    return sorted(set(found))


def scan(root: str, ancestry: Ancestry | None = None) -> list[dict]:
    """Every ledger entry this workspace earns, sorted by (finding, path)."""
    ancestry = ancestry or Ancestry(None)
    entries: list[dict] = []
    for directory in candidate_directories(root):
        names = _listdir(directory)
        entries += _jlens_entries(root, directory, names, ancestry)
        entries += _adapter_entries(root, directory, names, ancestry)
        entries += _promotion_entries(root, directory, names)
        entries += _sae_entries(root, directory, names)
    return sorted(entries, key=lambda e: (e["finding"], e["path"],
                                          e["artifactID"]))


def _listdir(directory: str) -> set:
    try:
        return set(os.listdir(directory))
    except OSError:
        return set()


# ---------------------------------------------------------------------------
# SCI-01 — the J-lens full-vocabulary readout
# ---------------------------------------------------------------------------

#: Run types whose product came off the full-vocabulary path when top-k was
#: armed. Named from the fix commit's own impact paragraph: trace top-k
#: columns, probe top-k and pinned ranks, the J-space topKDelta/topKEmergent
#: tables, and the G0 replay/rank tables.
_JLENS_RUN_TYPES = {
    "jlens-probe": "jlensProbeRun",
    "jlens-g0": "jlensG0Run",
    "optvec-jspace": "optvecJSpaceRun",
}

#: The trace a study run writes when its manifest declared a jlensReadout
#: block (``jlens.trace.TRACE_FILENAME``). Read as text, not JSON: the file
#: is one row per scored step and can be very large, and the only question is
#: whether the full-vocabulary column was ever written.
_JLENS_TRACE = "jlens-readout.jsonl"
_JLENS_TOPK_KEY = '"topKIDs"'

_JLENS_TOPK_MARKERS = {
    "jlensProbeRun": ("probe-topk.csv",),
    "jlensG0Run": ("jlens-g0-report.json",),
    "jlensTraceRun": ("jlens-topk.csv",),
}


def _jlens_entries(root: str, directory: str, names: set,
                   ancestry: Ancestry) -> list:
    stamp = _run_stamp(directory)
    artifact_type = _JLENS_RUN_TYPES.get(stamp["runType"] or "")
    if artifact_type is None and _JLENS_TRACE in names:
        artifact_type = "jlensTraceRun"
    if artifact_type is None:
        return []

    evidence = [f"config.json runType is {stamp['runType']!r}"
                if stamp["runType"] else
                "the directory carries no config.json runType"]
    armed, armed_facts = _full_vocabulary_armed(directory, names, artifact_type)
    evidence += armed_facts

    revision = stamp["appVersion"]
    commit, dirty = _commit_token(revision)
    if commit:
        evidence.append(
            f"config.json appVersion {revision!r} names build {commit}"
            + (" (worktree was DIRTY at build time)" if dirty else ""))
    elif revision:
        evidence.append(
            f"config.json appVersion {revision!r} carries no +<commit> "
            "build identity")

    finding = FINDINGS_BY_ID["SCI-01"]
    if armed is False:
        exposure, action = UNAFFECTED, "none"
        assessment = (
            "The fix changed only the full-vocabulary projection "
            "(LensReadout.logits/topk). This artifact carries no "
            "full-vocabulary readout, and the watchlist path already folded "
            "the final-norm gain into its token rows, so nothing here came "
            "off the corrected path.")
    else:
        carries, ancestry_facts = ancestry.carries_fix(finding, commit)
        evidence += ancestry_facts
        if carries is True:
            exposure, action = UNAFFECTED, "none"
            assessment = (
                "Produced by a build that already contained the fix, so the "
                "final-norm gain was applied on the full-vocabulary path.")
        elif carries is False:
            exposure, action = EXPOSED, "rerun"
            assessment = (
                "Produced before the fix with the full-vocabulary path armed: "
                "the readout omitted g = 1 + norm.weight, which varies by "
                "coordinate and therefore REORDERED tokens rather than "
                "rescaling them. Top-k identities and ranks from this "
                "artifact are not trustworthy. RE-RUN, not recompute: none of "
                "these producers retain the transported residuals the readout "
                "would need. Energies, null ratios, and the linearity "
                "residual go through `transported`, not `logits`, and are "
                "unaffected.")
        else:
            exposure, action = UNKNOWN, "resolveProvenance"
            assessment = (
                "The full-vocabulary path is armed (or cannot be ruled out) "
                "and the producing build is not datable from what is on disk, "
                "so this artifact cannot be placed on either side of the fix. "
                "Resolve the build — re-run with --code-checkout pointed at a "
                "checkout that contains the stamped commit — before treating "
                "any top-k table from it as evidence.")

    return [_entry(
        artifact_id=os.path.basename(directory),
        artifact_type=artifact_type,
        path=_relative(directory, root),
        producing_revision=revision,
        pins=_pins(modelID=stamp["modelID"], revision=stamp["revision"],
                   experiment=stamp["experiment"],
                   experimentHash=stamp["experimentHash"],
                   substrate=stamp["substrate"]),
        finding=finding.id, exposure=exposure, evidence=evidence,
        assessment=assessment, required_action=action)]


def _full_vocabulary_armed(directory: str, names: set,
                           artifact_type: str) -> tuple[bool | None, list]:
    """``(armed, facts)`` — did this artifact take the full-vocabulary path?

    ``False`` is a POSITIVE finding here, not an absence: the top-k tables are
    written only when top-k is armed, so their absence is the artifact saying
    it never touched the corrected code.
    """
    facts: list[str] = []
    if artifact_type == "jlensTraceRun":
        hit, inspected = _trace_carries_topk(os.path.join(directory,
                                                          _JLENS_TRACE))
        if hit is None:
            facts.append(f"{_JLENS_TRACE} could not be read")
            return None, facts
        facts.append(
            f"{_JLENS_TRACE}: a row carries topKIDs" if hit else
            f"{_JLENS_TRACE}: no row of {inspected} carries topKIDs")
        if hit:
            return True, facts
        # A report beside an un-armed trace was built from watchlist rows.
        for marker in _JLENS_TOPK_MARKERS[artifact_type]:
            if marker in names:
                facts.append(f"{marker} is present beside the trace")
                return True, facts
        return False, facts

    if artifact_type == "optvecJSpaceRun":
        report = _read_json(os.path.join(directory, "jspace.json"))
        if not isinstance(report, dict):
            facts.append("jspace.json is missing or unreadable")
            return None, facts
        tables = _jspace_topk_rows(report)
        facts.append(
            f"jspace.json carries {tables} topKDelta/topKEmergent row(s)")
        return (tables > 0), facts

    for marker in _JLENS_TOPK_MARKERS[artifact_type]:
        if marker in names:
            facts.append(f"{marker} is present")
            return True, facts
    facts.append("none of "
                 + ", ".join(_JLENS_TOPK_MARKERS[artifact_type])
                 + " is present")
    return False, facts


def _trace_carries_topk(path: str) -> tuple[bool | None, int]:
    """Whether any trace row carries a full-vocabulary top-k column.

    Substring-matched against the raw line rather than parsed: the key is a
    JSON key at the row's top level, the file can run to hundreds of
    megabytes, and the answer is boolean. Streams and stops at the first hit.
    """
    inspected = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                inspected += 1
                if _JLENS_TOPK_KEY in line:
                    return True, inspected
    except OSError:
        return None, inspected
    return False, inspected


def _jspace_topk_rows(report: dict) -> int:
    """How many rows the J-space report's top-k tables actually hold — the
    tables are written empty when the mean delta is unavailable, and an empty
    table is not a readout."""
    total = 0
    vectors = report.get("vectors")
    candidates = vectors if isinstance(vectors, list) else []
    if not candidates and isinstance(report.get("layers"), list):
        candidates = [report]
    for vector in candidates:
        if not isinstance(vector, dict):
            continue
        for layer in vector.get("layers") or []:
            if not isinstance(layer, dict):
                continue
            for key in ("topKDelta", "topKEmergent"):
                rows = layer.get(key)
                if isinstance(rows, list):
                    total += len(rows)
    return total


# ---------------------------------------------------------------------------
# SCI-02 / SCI-03 — adapter sidecars
# ---------------------------------------------------------------------------

PYTHON_SUBSTRATE = "python-hf-transformers"
SWIFT_SUBSTRATE = "swift-mlx"

#: The Swift engine's adapter sidecar filename (``FineTuneStore``); the Python
#: engine writes ``<adapter name>.json`` beside the weights, so its sidecars
#: are found by their stamps rather than by name.
SWIFT_SIDECAR = "fine-tune.json"

#: The Swift training modes (``FineTuneTrainingMode``). Only the
#: instruction/chat render appended a generation prompt.
SWIFT_INSTRUCTION_MODE = "instruction_chat"

#: The Python trainer's legacy inline path, which has no optimizer-step group
#: at all — it steps once per iteration, so no accumulation weighting exists
#: to have been wrong.
PYTHON_LEGACY_MODE = "legacy_inline"

#: The post-fix objective stamp (``lora_train.TRAINING_OBJECTIVE``). Its
#: PRESENCE is the artifact's own statement that it was trained under the
#: corrected objective.
OBJECTIVE_STAMP = "tokenMeanPerOptimizerStep"


def _adapter_sidecars(directory: str, names: set) -> list:
    """``(filename, document)`` for every adapter sidecar in this directory.

    Recognised by the CROSS-ENGINE stamps (``substrate`` + ``adapterFormat``,
    the pinned contract both engines write), not by filename: the Python
    engine names its sidecar after the adapter, so there is no name to match.
    Only the directories that could hold one are read — a lora-train run, a
    directory carrying the Swift sidecar, or one holding PEFT weights — so
    the scan does not parse every JSON file in a workspace.
    """
    stamp = _run_stamp(directory)
    interesting = (stamp["runType"] == "lora-train"
                   or SWIFT_SIDECAR in names
                   or os.path.basename(directory) in ("adapters", "fine-tunes")
                   or any(os.path.isfile(os.path.join(directory, name,
                                                      "adapter_config.json"))
                          for name in sorted(names)))
    if not interesting:
        return []
    found = []
    for name in sorted(n for n in names if n.endswith(".json")):
        document = _read_json(os.path.join(directory, name))
        if isinstance(document, dict) and "adapterFormat" in document:
            found.append((name, document))
    return found


def _adapter_entries(root: str, directory: str, names: set,
                     ancestry: Ancestry) -> list:
    entries = []
    for name, sidecar in _adapter_sidecars(directory, names):
        substrate = sidecar.get("substrate")
        path = os.path.join(directory, name)
        if substrate == PYTHON_SUBSTRATE:
            entries.append(_python_adapter_entry(root, path, sidecar,
                                                 directory))
        elif substrate == SWIFT_SUBSTRATE:
            entries.append(_swift_adapter_entry(root, path, sidecar,
                                                directory, ancestry))
    return entries


def _accumulation(sidecar: dict) -> tuple[int | None, str]:
    """``(accumulation, how it was read)``.

    The schedule block is the direct statement. Where it is absent the factor
    is derived from ``effectiveBatchSize / batchSize``, which the same block
    computes — and where neither is there, the answer is None, which is an
    ``unknown``, not a 1.
    """
    schedule = sidecar.get("schedule")
    if isinstance(schedule, dict):
        value = schedule.get("gradientAccumulation")
        if isinstance(value, int) and value >= 1:
            return value, "schedule.gradientAccumulation"
        effective = schedule.get("effectiveBatchSize")
        batch = schedule.get("batchSize")
        if (isinstance(effective, int) and isinstance(batch, int)
                and batch > 0 and effective % batch == 0):
            return (effective // batch,
                    "schedule.effectiveBatchSize / schedule.batchSize")
    value = sidecar.get("gradientAccumulation")
    if isinstance(value, int) and value >= 1:
        return value, "gradientAccumulation"
    return None, "nothing in the sidecar states it"


def _objective_stamp(sidecar: dict, directory: str) -> tuple[str | None, str]:
    """The training objective this adapter was trained under, and where it was
    read. The sidecar's schedule block is authoritative; ``training-history``
    carries the same string and answers for a sidecar written by a build that
    stamped only the history."""
    schedule = sidecar.get("schedule")
    if isinstance(schedule, dict) and isinstance(schedule.get("objective"),
                                                 str):
        return schedule["objective"], "schedule.objective"
    history_name = sidecar.get("historyFile") or "training-history.json"
    history = _read_json(os.path.join(directory, str(history_name)))
    if isinstance(history, dict) and isinstance(history.get("objective"), str):
        return history["objective"], f"{history_name}.objective"
    return None, "no objective stamp in the sidecar or the history"


def _python_adapter_entry(root: str, path: str, sidecar: dict,
                          directory: str) -> dict:
    finding = FINDINGS_BY_ID["SCI-02"]
    mode = sidecar.get("trainingMode")
    build = sidecar.get("buildIdentity")
    commit = build.get("commit") if isinstance(build, dict) else None
    stamp = _run_stamp(directory)
    revision = stamp["appVersion"]
    evidence = [f"sidecar substrate is {PYTHON_SUBSTRATE!r}, adapterFormat "
                f"{sidecar.get('adapterFormat')!r}",
                f"sidecar trainingMode is {mode!r}"]
    if commit:
        evidence.append(f"sidecar buildIdentity names build {commit}"
                        + (" (DIRTY worktree)"
                           if isinstance(build, dict) and build.get("dirty")
                           else ""))
    objective, objective_source = _objective_stamp(sidecar, directory)
    accumulation, accumulation_source = _accumulation(sidecar)
    evidence.append(f"objective: {objective!r} (from {objective_source})")
    evidence.append(
        f"gradient accumulation: {accumulation!r} "
        f"(from {accumulation_source})")

    if mode == PYTHON_LEGACY_MODE:
        exposure, action = UNAFFECTED, "none"
        assessment = (
            "Trained on the legacy inline path, which steps the optimizer "
            "once per iteration and never groups micro-batches — the "
            "accumulation weighting the fix corrected does not exist there.")
    elif objective == OBJECTIVE_STAMP:
        exposure, action = UNAFFECTED, "none"
        assessment = (
            f"The adapter carries the post-fix objective stamp "
            f"{OBJECTIVE_STAMP!r}, which only the corrected trainer writes: "
            "one optimizer step minimized the token-average loss over its "
            "whole group, independent of the micro-batch cut.")
    elif accumulation == 1:
        exposure, action = UNAFFECTED, "none"
        assessment = (
            "No objective stamp, so this adapter predates the fix — but its "
            "accumulation factor is 1, so every optimizer-step group was a "
            "single micro-batch. With one micro-batch the mean-of-means and "
            "the token-average objectives are the SAME number, and the "
            "gradient is identical either way. Nothing to redo.")
    elif accumulation is not None and accumulation > 1:
        exposure, action = EXPOSED, "rerun"
        assessment = (
            f"No objective stamp and gradientAccumulation = {accumulation}: "
            "this adapter was trained on the pre-fix path, where each "
            "micro-batch's MEAN loss was divided by the nominal factor. That "
            "weights micro-batches by how their supervised-target counts "
            "happened to fall, and scales an incomplete final group down by a "
            "factor it never filled. THIS IS A SCREENING FLAG, NOT A "
            "MEASURED EFFECT: accumulation > 1 says the wrong objective was "
            "reachable, not that these weights moved by any particular "
            "amount — the size of the difference depends on how uneven the "
            "target counts actually were, which the sidecar does not record. "
            "Retraining is the only way to produce weights under the stated "
            "objective; the pre-fix checkpoint cannot be resumed under it "
            "(the objective is in the config fingerprint), and that refusal "
            "is correct.")
    else:
        exposure, action = UNKNOWN, "resolveProvenance"
        assessment = (
            "No objective stamp and no recoverable accumulation factor, so "
            "the adapter cannot say whether more than one micro-batch was "
            "accumulated into an optimizer step. Missing evidence: the "
            "schedule block (gradientAccumulation, or batchSize with "
            "effectiveBatchSize). Recover it from the training job's "
            "configuration before citing this adapter.")

    return _entry(
        artifact_id=f"{os.path.basename(directory)}/{os.path.basename(path)}",
        artifact_type="loraAdapterSidecar",
        path=_relative(path, root),
        producing_revision=revision,
        pins=_pins(modelID=sidecar.get("baseModelID"),
                   revision=(sidecar.get("revisionResolved")
                             or sidecar.get("revision")),
                   adapterBytesHash=sidecar.get("adapterBytesHash"),
                   adapterConfigHash=sidecar.get("adapterConfigHash"),
                   buildCommit=commit),
        finding=finding.id, exposure=exposure, evidence=evidence,
        assessment=assessment, required_action=action)


def _swift_adapter_entry(root: str, path: str, sidecar: dict, directory: str,
                         ancestry: Ancestry) -> dict:
    finding = FINDINGS_BY_ID["SCI-03"]
    mode = sidecar.get("trainingMode")
    stamp = _run_stamp(directory)
    revision = stamp["appVersion"]
    commit, dirty = _commit_token(revision)
    evidence = [f"sidecar substrate is {SWIFT_SUBSTRATE!r}, adapterFormat "
                f"{sidecar.get('adapterFormat')!r}",
                f"sidecar trainingMode is {mode!r}"]
    if commit:
        evidence.append(
            f"config.json appVersion {revision!r} names app build {commit}"
            + (" (worktree was DIRTY at build time)" if dirty else ""))
    elif revision:
        evidence.append(f"config.json appVersion {revision!r} carries no "
                        "+<commit> build identity")
    else:
        evidence.append("no config.json appVersion beside the sidecar — the "
                        "app build that trained this adapter is not recorded")
    # Stated on EVERY entry: the decisive evidence is not in the sidecar at
    # all, and a reader must not mistake a clean classification for one.
    caveat = ("The sidecar cannot show whether the template appended a "
              "generation prompt to the supervised target — the decisive "
              "evidence is the chat template's render on the app build that "
              "trained this adapter, which no artifact here retains.")

    if mode is not None and mode != SWIFT_INSTRUCTION_MODE:
        exposure, action = UNAFFECTED, "none"
        assessment = (
            f"Trained in {mode!r} mode. The fix touched only the "
            "instruction/chat render, which is the one path that called the "
            "template with a generation prompt; a document-mode target is "
            "raw text and never passes through it. " + caveat)
        evidence.append("the recorded mode is not instruction/chat, so the "
                        "corrected render was never reached")
        return _swift_entry(root, path, sidecar, directory, revision, commit,
                            finding, exposure, evidence, assessment, action)

    carries, ancestry_facts = ancestry.carries_fix(finding, commit)
    evidence += ancestry_facts
    if mode == SWIFT_INSTRUCTION_MODE and carries is False:
        exposure, action = EXPOSED, "rerun"
        assessment = (
            "Instruction/chat mode on an app build that predates the fix: the "
            "completed answer was rendered with a generation prompt appended, "
            "so the supervised target carried template tokens the answer does "
            "not contain and the adapter was trained to emit them. " + caveat)
    elif mode == SWIFT_INSTRUCTION_MODE and carries is True:
        exposure, action = UNAFFECTED, "none"
        assessment = (
            "Instruction/chat mode on an app build that already contained the "
            "fix, so the answer was rendered without a generation prompt and "
            "the template's prefix check was armed. " + caveat)
    else:
        exposure, action = UNKNOWN, "resolveProvenance"
        missing = []
        if mode is None:
            missing.append("the sidecar records no trainingMode")
        if commit is None:
            missing.append("no app build identity is recorded beside it")
        elif carries is None:
            missing.append("the recorded app build could not be dated "
                           "against the fix")
        assessment = (
            "Cannot be placed on either side of the fix: "
            + "; ".join(missing) + ". " + caveat + " Resolve the app build "
            "(and the mode) before citing this adapter as evidence.")
    return _swift_entry(root, path, sidecar, directory, revision, commit,
                        finding, exposure, evidence, assessment, action)


def _swift_entry(root, path, sidecar, directory, revision, commit, finding,
                 exposure, evidence, assessment, action) -> dict:
    return _entry(
        artifact_id=f"{os.path.basename(directory)}/{os.path.basename(path)}",
        artifact_type="mlxAdapterSidecar",
        path=_relative(path, root),
        producing_revision=revision,
        pins=_pins(modelID=sidecar.get("baseModelID"),
                   revision=sidecar.get("baseRevision"),
                   adapterHash=sidecar.get("adapterHash"),
                   configHash=sidecar.get("configHash"),
                   trainingDataHash=sidecar.get("trainingDataHash"),
                   validationDataHash=sidecar.get("validationDataHash"),
                   buildCommit=commit),
        finding=finding.id, exposure=exposure, evidence=evidence,
        assessment=assessment, required_action=action)


# ---------------------------------------------------------------------------
# SCI-04 — flat and single-dose ladders
# ---------------------------------------------------------------------------

PROMOTED_MOVERS = "promoted-movers.json"
EFFECT_SIZES = "effect-sizes.csv"
SOURCE_RUN = "source-run.txt"


def _movers_decisions(document: dict) -> list:
    promoted = document.get("promoted")
    rejected = document.get("rejected")
    out = []
    for group in (promoted, rejected):
        if isinstance(group, list):
            out += [d for d in group if isinstance(d, dict)]
    return out


def _flat_ladder_passes(decisions: list) -> list:
    """The concepts whose ladder was counted monotone with an UNDEFINED rho.

    That pairing is the pre-fix signature and nothing else: under the current
    helper a monotone verdict requires two distinct doses AND a nonzero effect
    range, and both of those are exactly the conditions under which Spearman's
    rho is defined. So ``doseMonotone: true`` beside a null/absent
    ``doseSpearmanRho`` is a ladder the old code graded as rising while its
    own correlation had no answer.
    """
    flagged = []
    for decision in decisions:
        if decision.get("doseMonotone") is not True:
            continue
        rho = decision.get("doseSpearmanRho")
        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            flagged.append(str(decision.get("concept") or "?"))
    return sorted(flagged)


def _promotion_entries(root: str, directory: str, names: set) -> list:
    if PROMOTED_MOVERS not in names:
        return []
    path = os.path.join(directory, PROMOTED_MOVERS)
    document = _read_json(path)
    stamp = _run_stamp(directory)
    evidence = []
    if not isinstance(document, dict):
        return [_entry(
            artifact_id=f"{os.path.basename(directory)}/{PROMOTED_MOVERS}",
            artifact_type="promotedMovers", path=_relative(path, root),
            producing_revision=stamp["appVersion"],
            pins=_pins(experiment=stamp["experiment"],
                       experimentHash=stamp["experimentHash"]),
            finding="SCI-04", exposure=UNKNOWN,
            evidence=[f"{PROMOTED_MOVERS} is not readable JSON"],
            assessment="The funnel artifact cannot be read, so whether a flat "
                       "ladder passed cannot be determined from it. Missing "
                       "evidence: a parseable promoted-movers.json.",
            required_action="resolveProvenance")]

    decisions = _movers_decisions(document)
    flagged = _flat_ladder_passes(decisions)
    evidence.append(f"{PROMOTED_MOVERS} carries {len(decisions)} decision(s)")
    evidence.append(
        f"{len(flagged)} of them declare doseMonotone true with no "
        "doseSpearmanRho"
        + (f": {', '.join(flagged)}" if flagged else ""))
    for required in (EFFECT_SIZES, SOURCE_RUN):
        evidence.append(f"{required} is "
                        + ("present" if required in names else "ABSENT")
                        + " beside it")

    if flagged:
        exposure, action = EXPOSED, "recompute"
        assessment = (
            f"{len(flagged)} concept(s) were graded monotone in dose while "
            "their own Spearman rho was undefined — the pre-fix signature of "
            "a flat ladder (identical effects satisfy every consecutive step "
            "trivially) or a ladder with a single distinct dose (the sort's "
            "tie-break, not the dose, ordered the effects). Under the current "
            "rule neither is monotone. This is a RECOMPUTE, not a re-run: the "
            "promotion decision is a pure function of the effect table, which "
            "is still on disk — see the reassessed promotions file this "
            "ledger emits. The funnel artifact itself is immutable and is "
            "NOT edited.")
    else:
        exposure, action = UNAFFECTED, "none"
        assessment = (
            "No decision in this funnel claims a monotone ladder without a "
            "defined rho, so no flat or single-dose ladder was counted as a "
            "dose response here.")

    return [_entry(
        artifact_id=f"{os.path.basename(directory)}/{PROMOTED_MOVERS}",
        artifact_type="promotedMovers", path=_relative(path, root),
        producing_revision=stamp["appVersion"],
        pins=_pins(experiment=document.get("experiment") or stamp["experiment"],
                   experimentHash=(document.get("experimentHash")
                                   or stamp["experimentHash"]),
                   sha256=_sha256_file(path)),
        finding="SCI-04", exposure=exposure, evidence=evidence,
        assessment=assessment, required_action=action)]


def _sae_entries(root: str, directory: str, names: set) -> list:
    from . import sae_qualification as saq

    if saq.FILENAME not in names:
        return []
    path = os.path.join(directory, saq.FILENAME)
    stamp = _run_stamp(directory)
    identity = {
        "artifact_id": f"{os.path.basename(directory)}/{saq.FILENAME}",
        "artifact_type": "saeFeatureQualification",
        "path": _relative(path, root),
        "producing_revision": stamp["appVersion"],
    }
    try:
        record, digest = saq.load(path)
    except saq.QualificationError as exc:
        return [_entry(
            **identity, pins=_pins(sha256=_sha256_file(path)),
            finding="SCI-04", exposure=UNKNOWN,
            evidence=[f"the record does not load: {exc}"],
            assessment="The qualification cannot be parsed, so its declared "
                       "dose response cannot be checked against its own rows. "
                       "Missing evidence: a loadable qualification record.",
            required_action="resolveProvenance")]

    # The promote verb's own check, reused rather than restated: a record may
    # claim LESS than its rows show, never more.
    violations = saq.dose_response_violations(record)
    geometry = saq.dose_response_geometry(record)
    declared = record.dose_response
    evidence = [
        f"declared doseResponse: monotone={declared.get('monotone')!r}, "
        f"signSymmetric={declared.get('signSymmetric')!r}, "
        f"spearmanRho={declared.get('spearmanRho')!r}"]
    for sign in sorted(geometry):
        side = geometry[sign]
        evidence.append(
            f"computed from the record's own constructProbe rows "
            f"({sign}): monotone={side['monotone']}, "
            f"rho={side['spearmanRho']}, trend={side['trend']}, "
            f"rows={side['rows']}")
    evidence += [f"promote's consistency check: {v}" for v in violations]

    if violations:
        exposure, action = EXPOSED, "reassess"
        assessment = (
            "The record's declared dose response claims more than its own "
            "constructProbe rows show. Before the fix nothing checked this, "
            "so the declaration — which is what a citation reader sees — "
            "stood unopposed. The record is IMMUTABLE: the repair is a new "
            "qualification record whose declaration matches its rows (or new "
            "rows), linked back to this one, and a re-examination of every "
            "promotion that cited it. The decision itself stays the "
            "researcher's; this ledger only reports that the numbers and the "
            "claim disagree.")
    else:
        exposure, action = UNAFFECTED, "none"
        assessment = (
            "The declared dose response is borne out by the record's own "
            "constructProbe rows under the current helper, so the check the "
            "fix added would pass on this record as written.")

    return [_entry(
        **identity,
        pins=_pins(sha256=digest, decoderRowHash=record.decoder_row_hash,
                   artifact=record.artifact),
        finding="SCI-04", exposure=exposure, evidence=evidence,
        assessment=assessment, required_action=action)]


# ---------------------------------------------------------------------------
# Promotion rescoring (SCI-04) — a NEW artifact, never an edit
# ---------------------------------------------------------------------------

class ReconstructionImpossible(Exception):
    """The screen's decisions cannot be recomputed from what survives. Carried
    into the ledger entry's evidence rather than guessed around."""


def _parse_float(text: str) -> float:
    """The inverse of ``study_stats._fmt``: an empty cell is NaN (that is what
    the writer emits for one), anything else is a ``%.6g`` float."""
    text = (text or "").strip()
    if not text:
        return float("nan")
    return float(text)


def effect_rows_from_csv(path: str) -> tuple[list, dict]:
    """Rebuild the POOLED effect rows the analysis handed the promotion rule.

    ``analyze`` writes the pooled rows first and the stratified ones after,
    and passes only the pooled list to ``_promotion_decisions`` — so the
    stratified rows (including the within-item ``diagnostic`` ones, which
    carry no adjusted p at all) are skipped here for the same reason and are
    counted so the skip is visible rather than assumed.

    ``BootstrapCI.replicates`` and ``seed`` are NOT in the CSV. They are set
    to 0 and the reconstruction notes say so: no promotion criterion reads
    them (``promotion.decide`` uses the mean, the adjusted p, and the
    interval), and inventing plausible-looking values would be a provenance
    forgery.
    """
    from .study_stats import BootstrapCI, EffectRow

    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            records = list(reader)
    except OSError as exc:
        raise ReconstructionImpossible(f"{EFFECT_SIZES} is unreadable: {exc}")

    required = ("condition", "endpoint", "n", "deltaMean", "ciLower",
                "ciUpper", "wilcoxonW", "wilcoxonP", "adjustedP")
    missing = [name for name in required if name not in header]
    if missing:
        raise ReconstructionImpossible(
            f"{EFFECT_SIZES} is missing the column(s) "
            f"{', '.join(missing)} — the effect rows cannot be rebuilt")

    notes: list[str] = []
    if "stratifyBy" not in header:
        notes.append(
            f"{EFFECT_SIZES} predates the stratified-row columns, so every "
            "row is a pooled row (which is what the analysis that wrote it "
            "produced)")
    rows = []
    skipped = 0
    for record in records:
        stratify = (record.get("stratifyBy") or "pooled").strip() or "pooled"
        if stratify != "pooled":
            skipped += 1
            continue
        try:
            ci = BootstrapCI(
                n=int(record["n"]), mean=_parse_float(record["deltaMean"]),
                ci_lower=_parse_float(record["ciLower"]),
                ci_upper=_parse_float(record["ciUpper"]),
                replicates=0, seed=0)
            rows.append(EffectRow(
                condition=record["condition"], endpoint=record["endpoint"],
                ci=ci, wilcoxon_w=_parse_float(record["wilcoxonW"]),
                wilcoxon_p=_parse_float(record["wilcoxonP"]),
                adjusted_p=_parse_float(record["adjustedP"]),
                correction=(record.get("correction") or ""),
                modality=(record.get("modality") or ""),
                stratify_by=stratify, stratum=(record.get("stratum") or ""),
                unit=(record.get("unit") or ""),
                estimand=(record.get("estimand") or ""),
                inference=(record.get("inference") or "")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionImpossible(
                f"{EFFECT_SIZES} row {record!r} could not be rebuilt: {exc}")
    if not rows:
        raise ReconstructionImpossible(
            f"{EFFECT_SIZES} carries no pooled rows "
            f"({skipped} stratified row(s) skipped)")
    notes.append(
        "BootstrapCI.replicates and .seed are not columns of "
        f"{EFFECT_SIZES}; they are recorded as 0 here and are read by no "
        "promotion criterion")
    return rows, {"rowsRead": len(records), "pooledRows": len(rows),
                  "skippedNonPooledRows": skipped, "notes": notes}


def reassess_promotions(root: str, analysis_directory: str) -> dict:
    """Recompute one screen's promotion decisions under the CURRENT
    dose-monotonicity rule, without touching a byte of the original.

    Reuses the analysis path itself — ``tasks._promotion_decisions`` with the
    manifest snapshot the source run froze — so the reassessment differs from
    the original in exactly one thing: the code. Re-deriving the funnel by
    hand here would make the comparison meaningless.
    """
    names = _listdir(analysis_directory)
    for required in (PROMOTED_MOVERS, EFFECT_SIZES, SOURCE_RUN):
        if required not in names:
            raise ReconstructionImpossible(
                f"{required} is absent from {_relative(analysis_directory, root)}")

    original_path = os.path.join(analysis_directory, PROMOTED_MOVERS)
    original = _read_json(original_path)
    if not isinstance(original, dict):
        raise ReconstructionImpossible(f"{PROMOTED_MOVERS} is not readable JSON")

    try:
        with open(os.path.join(analysis_directory, SOURCE_RUN),
                  encoding="utf-8") as handle:
            source_run_id = handle.read().strip()
    except OSError as exc:
        raise ReconstructionImpossible(f"{SOURCE_RUN} is unreadable: {exc}")
    if not source_run_id:
        raise ReconstructionImpossible(f"{SOURCE_RUN} is empty")

    run_dir = os.path.join(paths.runs_directory(root), source_run_id)
    snapshot = os.path.join(run_dir, "experiment.json")
    if not os.path.isfile(snapshot):
        raise ReconstructionImpossible(
            f"the source run {source_run_id} has no experiment.json snapshot "
            f"at {_relative(run_dir, root)} — the manifest the screen ran "
            "under cannot be recovered, and reassessing under a DIFFERENT "
            "manifest would not be a reassessment")

    raw = _read_json(snapshot)
    if not isinstance(raw, dict):
        raise ReconstructionImpossible(
            f"{source_run_id}/experiment.json is not readable JSON")

    from .manifest import Manifest
    try:
        manifest = Manifest.from_dict(raw)
    except Exception as exc:                      # noqa: BLE001 - reported
        raise ReconstructionImpossible(
            f"the manifest snapshot does not load: {exc}")
    if manifest.promotion_rule is None:
        raise ReconstructionImpossible(
            "the manifest snapshot declares no promotionRule, so there is no "
            "rule to re-apply")

    rows, reconstruction = effect_rows_from_csv(
        os.path.join(analysis_directory, EFFECT_SIZES))

    from . import promotion as promotion_mod
    from . import tasks

    decisions = tasks._promotion_decisions(manifest, rows, run_dir,
                                           promotion_mod)
    reassessed = [decision.as_json() for decision in decisions]

    from ..build_identity import build_commit, engine_version
    rule = manifest.promotion_rule
    return {
        "schemaVersion": SCHEMA_VERSION,
        "changedVerdicts": _changed_verdicts(_movers_decisions(original),
                                             reassessed),
        "finding": "SCI-04",
        "original": {
            "path": _relative(original_path, root),
            "sha256": _sha256_file(original_path),
            "document": original,
        },
        "provenance": {
            "buildCommit": build_commit(),
            "engineVersion": engine_version(),
            "fixCommits": list(FINDINGS_BY_ID["SCI-04"].fix_commits),
            "manifestHash": manifest.content_hash(),
            "observedAt": _utc_now(),
            "reconstruction": reconstruction,
            "sourceAnalysis": _relative(analysis_directory, root),
            "sourceRun": _relative(run_dir, root),
        },
        "reassessed": {
            "promoted": [d.as_json() for d in decisions if d.promoted],
            "promotionRule": {
                "capabilityGate": rule.capability_gate,
                "doseMonotone": rule.dose_monotone,
                "exceedsRandomFloor": rule.exceeds_random_floor,
                "fdrThreshold": rule.fdr_threshold,
            },
            "rejected": [d.as_json() for d in decisions if not d.promoted],
        },
    }


#: The fields whose change makes a verdict a CHANGED verdict. ``promoted`` is
#: the headline; the two dose fields are here because a concept that stayed
#: rejected for a NEW reason is still a different scientific statement.
_VERDICT_FIELDS = ("promoted", "doseMonotone", "doseSpearmanRho")


def _changed_verdicts(original: list, reassessed: list) -> list:
    by_concept = {str(d.get("concept")): d for d in original}
    now = {str(d.get("concept")): d for d in reassessed}
    changed = []
    for concept in sorted(set(by_concept) | set(now)):
        was = by_concept.get(concept)
        is_now = now.get(concept)
        if was is None or is_now is None:
            changed.append({
                "concept": concept,
                "change": ("addedByReassessment" if was is None
                           else "absentFromReassessment"),
                "was": was, "now": is_now})
            continue
        differences = {field: {"was": was.get(field), "now": is_now.get(field)}
                       for field in _VERDICT_FIELDS
                       if was.get(field) != is_now.get(field)}
        if differences:
            changed.append({
                "concept": concept, "change": "verdictChanged",
                "fields": differences,
                "wasReasons": list(was.get("reasons") or []),
                "nowReasons": list(is_now.get("reasons") or [])})
    return changed


# ---------------------------------------------------------------------------
# The ledger document
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_output_directory(root: str) -> str:
    """A fresh ``diagnostics/impact-ledger-<stamp>/``.

    Created with ``os.makedirs`` here rather than through ``paths``: this is
    not a run, and putting a diagnostics-tree helper into the run-layout
    module would invite the next writer to treat one as the other.
    """
    base = os.path.join(root, DIAGNOSTICS_SUBTREE)
    os.makedirs(base, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}"
    target = os.path.join(base, OUTPUT_PREFIX + stamp)
    counter = 1
    while True:
        try:
            os.makedirs(target)
            return target
        except FileExistsError:
            counter += 1
            target = os.path.join(base, f"{OUTPUT_PREFIX}{stamp}-{counter}")
            if counter > 100:
                raise


def counts(entries: list) -> dict:
    """``{finding: {exposure: n, …, "total": n}}`` for every finding, present
    in the workspace or not — a finding with no candidates must be visible as
    a zero, because "we found nothing" and "we did not look" are the two
    answers this document exists to separate."""
    table = {finding.id: {exposure: 0 for exposure in EXPOSURES}
             for finding in FINDINGS}
    for entry in entries:
        table[entry["finding"]][entry["exposure"]] += 1
    for block in table.values():
        block["total"] = sum(block[exposure] for exposure in EXPOSURES)
    return table


def build(root: str, *, ancestry: Ancestry | None = None) -> dict:
    """Scan, reassess, and write the ledger. Returns the summary the envelope
    carries."""
    ancestry = ancestry or Ancestry(None)
    entries = scan(root, ancestry)
    output = make_output_directory(root)

    reassessments = []
    for entry in entries:
        if entry["finding"] != "SCI-04" or entry["artifactType"] != "promotedMovers":
            continue
        analysis = os.path.join(root, entry["path"])
        analysis = os.path.dirname(analysis)
        try:
            document = reassess_promotions(root, analysis)
        except ReconstructionImpossible as exc:
            entry["evidence"].append(
                f"promotion reassessment not possible: {exc}")
            if entry["exposure"] == EXPOSED:
                entry["requiredAction"] = "reassess"
            continue
        target_dir = os.path.join(output, REASSESSED_SUBTREE,
                                  os.path.basename(analysis))
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, REASSESSED_PROMOTIONS)
        _write_json(target, document)
        relative = _relative(target, root)
        entry["evidence"].append(
            f"reassessed under the current rule → {relative} "
            f"({len(document['changedVerdicts'])} changed verdict(s))")
        reassessments.append({
            "changedVerdicts": len(document["changedVerdicts"]),
            "path": relative,
            "sourceAnalysis": document["provenance"]["sourceAnalysis"]})

    ledger = {
        "schemaVersion": SCHEMA_VERSION,
        "counts": counts(entries),
        "entries": entries,
        "findings": findings_table(),
        "generatedAt": _utc_now(),
        "reassessments": reassessments,
        "workspace": root,
    }
    _write_json(os.path.join(output, LEDGER_JSON), ledger)
    _write_markdown(os.path.join(output, LEDGER_MARKDOWN), ledger, root)
    return {
        "counts": ledger["counts"],
        "entryCount": len(entries),
        "findings": findings_table(),
        "ledgerFile": _relative(os.path.join(output, LEDGER_JSON), root),
        "outputDirectory": _relative(output, root),
        "reassessments": reassessments,
        "revisionDating": ("codeCheckout" if ancestry.available
                           else "unavailable"),
    }


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_markdown(path: str, ledger: dict, root: str) -> None:
    """The readable face of the same document: one section per finding, its
    counts, and a table an owner can fill in."""
    lines = [
        "# Impact ledger — the 2026-09-05 science fixes",
        "",
        f"Workspace: `{root}`  ",
        f"Generated: {ledger['generatedAt']}  ",
        f"Entries: {len(ledger['entries'])}",
        "",
        "This ledger says which EXISTING artifacts each fix could have "
        "reached, and what was read to decide. The fixes' own regression "
        "suites establish covered behaviour going forward; they establish "
        "nothing about the artifacts below.",
        "",
        "`unknown` is not a soft `unaffected`: it means the artifact cannot "
        "say which build produced it. `disposition`, `owner`, and "
        "`replacementArtifact` are the researcher's to fill in.",
        "",
    ]
    for finding in FINDINGS:
        block = ledger["counts"][finding.id]
        lines += [
            f"## {finding.id} — fixed by "
            + ", ".join(f"`{sha}`" for sha in finding.fix_commits)
            + f" ({finding.landed})",
            "",
            finding.summary,
            "",
            f"**{block['total']} candidate artifact(s)** — "
            f"{block[EXPOSED]} exposed, {block[UNKNOWN]} unknown, "
            f"{block[UNAFFECTED]} unaffected.",
            "",
        ]
        rows = [e for e in ledger["entries"] if e["finding"] == finding.id]
        if not rows:
            lines += ["No candidate artifacts of this kind in this workspace.",
                      ""]
            continue
        lines += [
            "| exposure | artifact | type | producing revision | required "
            "action | disposition |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for entry in rows:
            lines.append(
                f"| {entry['exposure']} | `{entry['path']}` | "
                f"{entry['artifactType']} | "
                f"{entry['producingRevision'] or '—'} | "
                f"{entry['requiredAction']} | {entry['disposition']} |")
        lines.append("")
        for entry in rows:
            lines += [f"### `{entry['path']}` — {entry['exposure']}", "",
                      entry["assessment"], "", "Evidence read:", ""]
            lines += [f"- {fact}" for fact in entry["evidence"]]
            lines.append("")
    if ledger["reassessments"]:
        lines += ["## Reassessed promotions", "",
                  "Recomputed under the current dose-monotonicity rule from "
                  "each screen's own effect table. The original funnel "
                  "artifacts are untouched.", "",
                  "| reassessed file | source analysis | changed verdicts |",
                  "| --- | --- | --- |"]
        for item in ledger["reassessments"]:
            lines.append(f"| `{item['path']}` | `{item['sourceAnalysis']}` | "
                         f"{item['changedVerdicts']} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


__all__ = [
    "Ancestry", "DISPOSITIONS", "ENTRY_KEYS", "EXPOSED", "EXPOSURES",
    "FINDINGS", "FINDINGS_BY_ID", "Finding", "LEDGER_JSON",
    "LEDGER_MARKDOWN", "LedgerRefusal", "REASSESSED_PROMOTIONS",
    "REASSESSED_SUBTREE", "REQUIRED_ACTIONS", "ReconstructionImpossible",
    "SCHEMA_VERSION", "UNAFFECTED", "UNKNOWN", "build",
    "candidate_directories", "counts", "effect_rows_from_csv",
    "findings_table", "make_output_directory", "open_checkout",
    "reassess_promotions", "require_workspace", "scan",
]

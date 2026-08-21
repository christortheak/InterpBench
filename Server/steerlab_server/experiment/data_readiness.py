"""Study-data readiness — the server side of the ``data check`` layer.

The Swift CLI's ``data check <experiment>`` walks a MANIFEST and reports one
line per data requirement. OptVec is server-only (hard requirement: no MLX
training path), and its dataset bundle is authored BEFORE any config exists —
the researcher pins the files by hash into train/eval configs afterward — so
its readiness check is bundle-directory-driven and lives here. Same output
vocabulary as the Swift layer (``invalid``/``missing``/``partial``/
``present``, blockers first, exit 2 on any blocker), so the researcher reads
both checks the same way.

The OptVec bundle template checks exactly what the authoring spec
(``prompts/generation/COWORK-JOB-optvec-datasets.md``) promises and the
engine's own loaders enforce:

- all nine bundle files exist (eight choice-row files + ``neutral-fluency``);
- every choice file parses under the STRICT cross-engine loader
  (:func:`sweep_selection.load_choice_rows` — the same rules the sweep's
  logprobShift instrument and ``optvec train`` refuse on);
- every option is a single character (the tokenizer-free approximation of the
  training path's single-token refusal — ``optvec train`` refuses any option
  that tokenizes to more than one token, and ``"(A)"`` or word options never
  survive it);
- per-file A/B target balance within 45–55% (a pure position preference must
  score zero shift);
- item ids are unique across the WHOLE bundle (the engine keys baselines by
  id and refuses cross-split duplicates);
- ``neutral-fluency.jsonl`` is one ``{"text": …}`` JSON object per line, as
  the authoring spec pins (the eval loader is more permissive — a plain line
  is accepted there — but a bundle that needs that permissiveness was not
  authored to spec);
- ``bundle.json`` — the bundle's table of contents — exists, parses, carries
  the three research directives (a bundle whose science rationale did not
  travel with it is not a deliverable), pins ALL nine files, and every pinned
  hash agrees with the file's actual bytes (a stale table of contents is
  exactly the drift it exists to prevent);
- ``REPORT.md`` presence is reported NON-blocking: it is the human half of
  acceptance (leakage QC, domain quality) and mechanical checks cannot stand
  in for it.

Per-file SHA-256 is emitted for every VALID file so the researcher can paste
the hashes into train/eval configs verbatim. An invalid or missing file gets
no hash — this layer never hands out a paste-able pin for bytes the engine
would refuse.

Item COUNTS (90/30/30, …) are authoring targets, not engine refusals — they
are reported in the detail text, never as blockers. Cross-split CONTENT
leakage (paraphrase, shared fact patterns) is undetectable mechanically and
stays the authoring job's QC responsibility.

The second template is ``lora``: a workspace ``adapters/`` dataset package
(``adapters/<package>-manifest.json`` + the per-adapter ``training/`` and
``validation/`` files the builders in ``scripts/build_*_lora_*.py`` emit).
Same shape of question, asked of the training data: does the manifest parse,
does every declared file exist with the bytes it was pinned at, do its rows
parse under the STRICT loader the trainer refuses on
(:mod:`lora_data`), does the split hold (no duplicate rows, no train↔validation
overlap), and did the package's own QC pass. It is the pre-flight for
``docs/CLUSTER-LORA-READINESS.md`` §2.1–2.2: a dataset that fails here cannot
produce an evidence-grade adapter, and finding that out before a GPU
allocation is the whole point.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from . import lora_data, paths
from .sweep_selection import load_choice_rows

#: Where the bundle-authoring contract lives (named in refusal details).
AUTHORING_SPEC = "prompts/generation/COWORK-JOB-optvec-datasets.md"

#: Default bundle directory, relative to the workspace root.
DEFAULT_BUNDLE_DIRECTORY = os.path.join("prompts", "optvec")

#: The eight strict choice-row files, in role order.
CHOICE_FILES = (
    "target-train.jsonl",
    "target-val.jsonl",
    "target-test.jsonl",
    "anchor-train.jsonl",
    "anchor-val.jsonl",
    "anchor-test.jsonl",
    "capability-train.jsonl",
    "capability-eval.jsonl",
)

NEUTRAL_FILE = "neutral-fluency.jsonl"

#: All nine bundle DATA files. ``bundle.json`` and ``REPORT.md`` are checked
#: separately: the former is the bundle's table of contents (required, hash
#: agreement enforced), the latter the author's human-QC artifact (presence
#: reported, never a blocker).
BUNDLE_FILES = CHOICE_FILES + (NEUTRAL_FILE,)

BUNDLE_JSON = "bundle.json"
REPORT_FILE = "REPORT.md"

#: The three research directives (four keys — issue and direction are two
#: fields of directive 1) that must travel with the data in ``bundle.json``.
DIRECTIVE_KEYS = ("targetIssue", "shiftDirection", "caseFamilies",
                  "anchorIssues")

#: Per-file A/B target balance window: the share of rows targeting the FIRST
#: option (canonically "A") must land inside it, so a pure position
#: preference scores zero shift.
BALANCE_LOW = 0.45
BALANCE_HIGH = 0.55

#: The cross-file id-uniqueness requirement's display name.
BUNDLE_IDS_REQUIREMENT = "bundle ids"


@dataclass(frozen=True)
class Requirement:
    """One readiness line: the ``data check`` vocabulary (``invalid`` and
    ``missing`` block; ``partial`` and ``present`` do not)."""

    status: str          # "invalid" | "missing" | "partial" | "present"
    name: str            # bundle file name, or BUNDLE_IDS_REQUIREMENT
    detail: str
    sha256: str | None = None   # only for files that PASSED every check
    rows: int | None = None

    @property
    def blocker(self) -> bool:
        return self.status in ("invalid", "missing")

    def to_dict(self) -> dict:
        out = {"status": self.status, "name": self.name, "detail": self.detail}
        if self.sha256 is not None:
            out["sha256"] = self.sha256
        if self.rows is not None:
            out["rows"] = self.rows
        return out


_STATUS_ORDER = {"invalid": 0, "missing": 1, "partial": 2, "present": 3}


@dataclass(frozen=True)
class BundleReport:
    directory: str
    requirements: tuple[Requirement, ...] = field(default_factory=tuple)
    #: Which authoring contract this report was measured against — the OptVec
    #: bundle spec by default, the LoRA readiness plan for ``lora`` packages.
    authoring_spec: str = AUTHORING_SPEC
    #: The package manifest a ``lora`` report was read from (OptVec has none).
    manifest_path: str | None = None

    @property
    def blockers(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.blocker)

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict:
        out = {"bundleDirectory": self.directory,
               "authoringSpec": self.authoring_spec,
               "requirements": [r.to_dict() for r in self.requirements],
               "blockerCount": len(self.blockers),
               "ready": self.ready}
        if self.manifest_path is not None:
            out["manifestPath"] = self.manifest_path
        return out


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _check_choice_file(directory: str,
                       name: str) -> tuple[Requirement, tuple[str, ...]]:
    """One choice file through the strict loader plus the bundle-spec checks.

    Returns the requirement and, when the file PASSED, its item ids (in file
    order) so the caller can check uniqueness across the whole bundle. A
    failed file contributes no ids.
    """
    path = os.path.join(directory, name)
    if not os.path.isfile(path):
        return Requirement("missing", name,
                           f"not found — author it per {AUTHORING_SPEC}"), ()
    try:
        rows, digest = load_choice_rows(path, name)
    except ValueError as exc:
        return Requirement("invalid", name, str(exc)), ()

    multi_char = [(row.id, option) for row in rows for option in row.options
                  if len(option) != 1]
    if multi_char:
        row_id, option = multi_char[0]
        return Requirement(
            "invalid", name,
            f"row '{row_id}' option {option!r} is not a single character "
            f"({len(multi_char)} such option(s) in the file) — training "
            "refuses any option that tokenizes to more than one token; use "
            "bare letters, never words or \"(A)\""), ()

    first_option_targets = sum(1 for row in rows if row.target == row.options[0])
    share = first_option_targets / len(rows)
    if not (BALANCE_LOW <= share <= BALANCE_HIGH):
        return Requirement(
            "invalid", name,
            f"first-option (A) target share is {share:.1%} of {len(rows)} "
            f"row(s) — outside the {BALANCE_LOW:.0%}–{BALANCE_HIGH:.0%} "
            "balance window; a skewed file lets a pure position preference "
            "read as a shift"), ()

    return Requirement(
        "present", name,
        f"{len(rows)} row(s), first-option targets {share:.1%}",
        sha256=digest, rows=len(rows)), tuple(row.id for row in rows)


def _check_neutral_file(directory: str) -> Requirement:
    """``neutral-fluency.jsonl`` strictly to the authoring spec: one JSON
    object per line with a non-empty string ``text`` and nothing else needed
    of it. Stricter than the eval loader on purpose (see module docstring)."""
    path = os.path.join(directory, NEUTRAL_FILE)
    if not os.path.isfile(path):
        return Requirement("missing", NEUTRAL_FILE,
                           f"not found — author it per {AUTHORING_SPEC}")
    with open(path, "rb") as handle:
        data = handle.read()
    texts = 0
    for i, raw in enumerate(data.decode("utf-8", errors="replace").splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            return Requirement(
                "invalid", NEUTRAL_FILE,
                f"line {i + 1} is not valid JSON: {exc} — the bundle spec "
                "pins one {\"text\": …} object per line")
        if not isinstance(obj, dict) or not isinstance(obj.get("text"), str) \
                or not obj["text"].strip():
            return Requirement(
                "invalid", NEUTRAL_FILE,
                f"line {i + 1} is not a {{\"text\": …}} object with a "
                "non-empty string text")
        texts += 1
    if texts == 0:
        return Requirement("invalid", NEUTRAL_FILE,
                           "parsed to zero texts — the fluency guard needs "
                           "real passages")
    return Requirement("present", NEUTRAL_FILE, f"{texts} text(s)",
                       sha256=hashlib.sha256(data).hexdigest(), rows=texts)


def _check_bundle_ids(file_results: list[Requirement],
                      duplicates: list[tuple[str, str, str]]) -> Requirement:
    """The cross-file requirement: ids unique across the WHOLE bundle."""
    parsed = [r for r in file_results
              if r.name in CHOICE_FILES and r.status == "present"]
    if duplicates:
        shown = ", ".join(f"'{d}' ({a} and {b})" for d, a, b in duplicates[:5])
        more = f" (+{len(duplicates) - 5} more)" if len(duplicates) > 5 else ""
        return Requirement(
            "invalid", BUNDLE_IDS_REQUIREMENT,
            f"{len(duplicates)} id(s) duplicated across files: {shown}{more} "
            "— the engine keys baselines by id and refuses cross-split "
            "duplicates")
    if len(parsed) < len(CHOICE_FILES):
        return Requirement(
            "partial", BUNDLE_IDS_REQUIREMENT,
            f"uniqueness verified over {len(parsed)} of {len(CHOICE_FILES)} "
            "choice files — resolve the blockers above to check the full "
            "bundle")
    total = sum(r.rows or 0 for r in parsed)
    return Requirement("present", BUNDLE_IDS_REQUIREMENT,
                       f"{total} id(s) unique across all "
                       f"{len(CHOICE_FILES)} choice files")


def _check_bundle_json(directory: str) -> Requirement:
    """The table of contents: required, complete, and hash-true.

    ``bundle.json`` is a data-side convention (the engine reads per-file
    hashes from its own configs), but the check enforces it because it is the
    delegation contract's carrier: the three research directives travel in
    it, and its hashes are what a human reads to know which bytes the bundle
    means. A pinned hash that disagrees with the file's actual bytes is a
    stale table of contents — the precise drift it exists to prevent.
    """
    path = os.path.join(directory, BUNDLE_JSON)
    if not os.path.isfile(path):
        return Requirement(
            "missing", BUNDLE_JSON,
            f"not found — the bundle's table of contents (directives + file "
            f"pins); author it per {AUTHORING_SPEC}")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except ValueError as exc:
        return Requirement("invalid", BUNDLE_JSON, f"not valid JSON: {exc}")
    if not isinstance(payload, dict):
        return Requirement("invalid", BUNDLE_JSON, "must be a JSON object")

    empty = [key for key in DIRECTIVE_KEYS
             if not (isinstance(payload.get(key), str) and payload[key].strip()
                     or isinstance(payload.get(key), list) and payload[key])]
    if empty:
        return Requirement(
            "invalid", BUNDLE_JSON,
            "research directive(s) absent or empty: " + ", ".join(empty)
            + " — the three directives are the bundle's science and must "
              "travel with the data (spec: REQUIRED INPUTS)")

    files = payload.get("files")
    if not isinstance(files, dict):
        return Requirement("invalid", BUNDLE_JSON,
                           "'files' must be an object pinning the nine "
                           "bundle files")
    by_basename: dict[str, str] = {}
    for entry in files.values():
        if isinstance(entry, dict) and isinstance(entry.get("path"), str) \
                and isinstance(entry.get("sha256"), str):
            by_basename[os.path.basename(entry["path"])] = \
                entry["sha256"].strip().lower()
    omitted = [name for name in BUNDLE_FILES if name not in by_basename]
    if omitted:
        return Requirement(
            "invalid", BUNDLE_JSON,
            "files table omits: " + ", ".join(omitted)
            + " — every bundle file is pinned in the table of contents")
    stale = []
    for name in BUNDLE_FILES:
        file_path = os.path.join(directory, name)
        if not os.path.isfile(file_path):
            continue    # the file's own requirement line already blocks
        if _sha256_file(file_path) != by_basename[name]:
            stale.append(name)
    if stale:
        return Requirement(
            "invalid", BUNDLE_JSON,
            "pinned hash disagrees with the file's bytes for: "
            + ", ".join(stale)
            + " — re-hash after the last edit; a stale table of contents is "
              "the drift bundle.json exists to prevent")
    return Requirement(
        "present", BUNDLE_JSON,
        f"{len(BUNDLE_FILES)} file(s) pinned, hashes agree; directives present")


def _check_report_file(directory: str) -> Requirement:
    """``REPORT.md`` presence — deliberately NON-blocking either way: the
    report is the human half of acceptance (leakage QC, domain quality), and a
    mechanical check can neither validate its content nor stand in for it."""
    path = os.path.join(directory, REPORT_FILE)
    if not os.path.isfile(path):
        return Requirement(
            "partial", REPORT_FILE,
            "not found — the human half of acceptance (cross-split leakage "
            "QC, domain quality) is not yet recorded")
    return Requirement("present", REPORT_FILE,
                       f"{os.path.getsize(path)} byte(s) — content is the "
                       "researcher's read, not this check's")


def check_optvec_bundle(directory: str) -> BundleReport:
    """Run the OptVec dataset-bundle readiness template over ``directory``.

    Purely local file checks — no model, no tokenizer, no network. Returns a
    :class:`BundleReport` with requirements sorted blockers-first
    (``invalid``, ``missing``, ``partial``, ``present``), stable within each
    status.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(
            f"bundle directory not found: {directory} — author the bundle "
            f"per {AUTHORING_SPEC}")

    bundle_ids: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    results: list[Requirement] = []
    for name in CHOICE_FILES:
        result, ids = _check_choice_file(directory, name)
        # load_choice_rows already refused WITHIN-file duplicates; any id
        # already claimed by an earlier file is a cross-file duplicate.
        for row_id in ids:
            if row_id in bundle_ids:
                duplicates.append((row_id, bundle_ids[row_id], name))
            else:
                bundle_ids[row_id] = name
        results.append(result)
    results.append(_check_neutral_file(directory))
    results.append(_check_bundle_json(directory))
    results.append(_check_report_file(directory))
    results.append(_check_bundle_ids(results, duplicates))

    ordered = sorted(results,
                     key=lambda r: _STATUS_ORDER.get(r.status, len(_STATUS_ORDER)))
    return BundleReport(directory=directory, requirements=tuple(ordered))


# --- LoRA dataset packages -------------------------------------------------

#: Where the LoRA dataset contract lives (named in refusal details).
LORA_AUTHORING_SPEC = "docs/CLUSTER-LORA-READINESS.md"

#: Package manifests are ``adapters/<package>-manifest.json`` (every builder
#: in ``scripts/build_*_lora_*.py`` writes that name).
LORA_MANIFEST_SUFFIX = "-manifest.json"

#: Default package directory, relative to the workspace root.
DEFAULT_LORA_DIRECTORY = "adapters"

#: The manifest keys under which builders nest their arms. ``outputs`` is the
#: flat spelling (crit/stance), ``families`` the nested one
#: (consciousness/paired-concept); the walk below is generic, so a third
#: spelling (virtue's ``designs``) is found too — these are only what the
#: usage text names.
LORA_ARM_CONTAINERS = ("outputs", "families")

#: Sequence budget used when the split check loads rows. Loading never
#: tokenizes (no tokenizer here by design), so this only has to satisfy the
#: spec's own bounds check.
LORA_SPLIT_CHECK_MAX_TOKENS = 8192

LORA_MANIFEST_REQUIREMENT = "manifest"
LORA_QC_REQUIREMENT = "qcReport"


def _lora_arms(node: object, trail: tuple[str, ...] = ()) -> list[tuple[str, dict]]:
    """Every training arm in a package manifest, whatever the nesting.

    An arm is any object carrying a ``training`` block — ``outputs{}``,
    ``families{}.arms{}``, ``families{}.outputs{}`` and the virtue package's
    ``designs{}`` all reduce to that. Labelled by its ``adapter`` name when it
    declares one (the name the trainer and variant attachment use), else by
    its path through the manifest.
    """
    if not isinstance(node, dict):
        return []
    if isinstance(node.get("training"), dict):
        adapter = node.get("adapter")
        label = adapter if isinstance(adapter, str) and adapter.strip() \
            else ".".join(trail) or "arm"
        return [(label, node)]
    arms: list[tuple[str, dict]] = []
    for key in sorted(node):
        arms.extend(_lora_arms(node[key], trail + (key,)))
    return arms


def _lora_root(manifest_path: str, declared: str | None,
               root: str | None) -> str:
    """Where the manifest's workspace-relative paths resolve from.

    The workspace root normally (``STEERLAB_ROOT``/cwd), but a package checked
    by absolute path from elsewhere still resolves: walk up from the manifest's
    own directory (``<workspace>/adapters/x-manifest.json`` → the workspace is
    two levels up) and take the first candidate where the first declared file
    actually exists. A fallback only — the configured root always wins when it
    holds the data.
    """
    if root is not None:
        return root
    candidates = [paths.project_root()]
    current = os.path.dirname(os.path.abspath(manifest_path))
    for _ in range(3):
        candidates.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    if declared and not os.path.isabs(declared):
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, declared)):
                return candidate
    return candidates[0]


def _check_lora_file(block: dict, *, label: str, role: str, root: str,
                     training_mode: str) -> tuple[Requirement,
                                                  lora_data.DatasetFile | None]:
    """One declared split file: pinned, present, hash-true, strictly parseable.

    Returns the requirement and — only when the file passed every check — the
    :class:`lora_data.DatasetFile` the split-integrity check then loads. A
    failed file contributes nothing to the split check (its own line blocks).
    """
    name = f"{label} {role}"
    declared = block.get("path")
    if not isinstance(declared, str) or not declared.strip():
        return Requirement("invalid", name,
                           f"manifest {role} block has no string 'path'"), None
    name = declared
    pinned = block.get("sha256")
    if not isinstance(pinned, str) or not pinned.strip():
        return Requirement(
            "invalid", name,
            f"manifest {role} block pins no sha256 — an unpinned file cannot "
            f"be shown to be the bytes that trained ({LORA_AUTHORING_SPEC} "
            "§2.1)"), None
    resolved = declared if os.path.isabs(declared) else os.path.join(root, declared)
    if not os.path.isfile(resolved):
        return Requirement("missing", name,
                           f"{role} file declared by the manifest is not on "
                           f"disk (looked in {root})"), None
    digest = _sha256_file(resolved)
    if digest != pinned.strip().lower():
        return Requirement(
            "invalid", name,
            f"sha256 mismatch — the manifest pinned {pinned.strip()}, the "
            f"file on disk is {digest}; the training data drifted after it "
            "was pinned"), None
    try:
        with open(resolved, encoding="utf-8") as handle:
            rows = lora_data.parse_rows(handle.read(), path=declared,
                                        training_mode=training_mode)
    except (lora_data.LoRADataError, UnicodeDecodeError) as exc:
        return Requirement("invalid", name, str(exc)), None
    if not rows:
        return Requirement("invalid", name, "no rows"), None
    declared_rows = block.get("rows")
    if isinstance(declared_rows, int) and declared_rows != len(rows):
        return Requirement(
            "invalid", name,
            f"manifest declares {declared_rows} row(s), the file parses to "
            f"{len(rows)} — the manifest no longer describes the file"), None
    return (Requirement("present", name,
                        f"{len(rows)} {training_mode} row(s), hash agrees",
                        sha256=digest, rows=len(rows)),
            lora_data.DatasetFile(path=declared, expected_sha256=digest,
                                  role="train" if role == "training"
                                  else "validation"))


def _check_lora_split(label: str, train: lora_data.DatasetFile,
                      validation: lora_data.DatasetFile,
                      *, root: str, training_mode: str) -> Requirement:
    """Split integrity through the loader the trainer itself uses: duplicate
    rows within a split, and any exact row/content overlap across them."""
    name = f"{label} split"
    spec = lora_data.LoRADatasetSpec(
        training_mode=training_mode, train_files=[train],
        validation_files=[validation],
        max_sequence_tokens=LORA_SPLIT_CHECK_MAX_TOKENS)
    try:
        loaded = lora_data.load_split_rows(spec, evidence_grade=True, root=root)
    except lora_data.LoRADataError as exc:
        return Requirement("invalid", name, str(exc))
    counts = loaded.counts
    return Requirement(
        "present", name,
        f"{counts['trainRows']} train / {counts['validationRows']} validation "
        "row(s), no duplicate rows, no cross-split overlap")


def _check_lora_qc(manifest: dict, *, root: str) -> Requirement:
    """The package's own QC report — the leakage/quality evidence the builder
    produced. ``status: "pass"`` or it blocks: a package that failed its own
    QC is not a training input, whatever its hashes say."""
    declared = manifest.get(LORA_QC_REQUIREMENT)
    if not isinstance(declared, str) or not declared.strip():
        return Requirement(
            "missing", LORA_QC_REQUIREMENT,
            "the manifest declares no QC report — leakage QC is the evidence "
            f"that the splits mean anything ({LORA_AUTHORING_SPEC} §4)")
    resolved = declared if os.path.isabs(declared) else os.path.join(root, declared)
    if not os.path.isfile(resolved):
        return Requirement("missing", declared,
                           "QC report declared by the manifest is not on disk")
    try:
        with open(resolved, encoding="utf-8") as handle:
            payload = json.load(handle)
    except ValueError as exc:
        return Requirement("invalid", declared, f"not valid JSON: {exc}")
    if not isinstance(payload, dict):
        return Requirement("invalid", declared, "must be a JSON object")
    status = payload.get("status")
    if status != "pass":
        errors = payload.get("errors")
        detail = f"status is {status!r}, not 'pass'"
        if isinstance(errors, list) and errors:
            detail += " — " + "; ".join(str(e) for e in errors[:3])
        return Requirement("invalid", declared, detail)
    warnings = payload.get("warnings")
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    return Requirement("present", declared,
                       f"status pass, {warning_count} warning(s)",
                       sha256=_sha256_file(resolved))


def _lora_manifest_path(target: str) -> str:
    """Resolve a ``data check lora`` target to one manifest file."""
    if os.path.isfile(target):
        return target
    if not os.path.isdir(target):
        raise FileNotFoundError(
            f"LoRA dataset package not found: {target} — pass a package "
            f"manifest (adapters/<package>{LORA_MANIFEST_SUFFIX}) or the "
            "directory holding exactly one")
    manifests = sorted(name for name in os.listdir(target)
                       if name.endswith(LORA_MANIFEST_SUFFIX))
    if not manifests:
        raise FileNotFoundError(
            f"no *{LORA_MANIFEST_SUFFIX} in {target} — a LoRA dataset package "
            f"is checked through its manifest ({LORA_AUTHORING_SPEC})")
    if len(manifests) > 1:
        raise IsADirectoryError(
            f"{target} holds {len(manifests)} package manifests "
            f"({', '.join(manifests)}) — name the one to check")
    return os.path.join(target, manifests[0])


def check_lora_package(directory: str, *, root: str | None = None) -> BundleReport:
    """Run the LoRA dataset-package readiness template.

    ``directory`` is either a package manifest
    (``adapters/<package>-manifest.json``) or a directory holding exactly one.
    Purely local file checks — no model, no tokenizer, no network — so this
    runs before a GPU allocation is spent, which is the point.

    Checks, in the order a training run would hit them: the manifest parses
    and declares a known ``trainingMode``; every arm's train/validation file is
    pinned, present, hash-true and parses under the strict row schema
    (:func:`lora_data.parse_rows`); each arm's split holds under
    :func:`lora_data.load_split_rows` (no duplicate rows, no train↔validation
    overlap); and the package's own QC report says ``pass``.
    """
    manifest_path = _lora_manifest_path(directory)
    package_directory = os.path.dirname(os.path.abspath(manifest_path))
    display = os.path.basename(manifest_path)

    def report(requirements: list[Requirement]) -> BundleReport:
        ordered = sorted(requirements,
                         key=lambda r: _STATUS_ORDER.get(r.status,
                                                         len(_STATUS_ORDER)))
        return BundleReport(directory=package_directory,
                            requirements=tuple(ordered),
                            authoring_spec=LORA_AUTHORING_SPEC,
                            manifest_path=manifest_path)

    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except ValueError as exc:
        return report([Requirement("invalid", display,
                                   f"not valid JSON: {exc}")])
    if not isinstance(manifest, dict):
        return report([Requirement("invalid", display,
                                   "must be a JSON object")])

    training_mode = manifest.get("trainingMode")
    if training_mode not in lora_data.TRAINING_MODES:
        return report([Requirement(
            "invalid", display,
            f"trainingMode {training_mode!r} is not one of "
            f"{', '.join(lora_data.TRAINING_MODES)} — the row schema and the "
            "loss mask both follow from it")])

    arms = _lora_arms(manifest)
    if not arms:
        return report([Requirement(
            "invalid", display,
            "declares no training arms — expected "
            f"{' or '.join(LORA_ARM_CONTAINERS)} objects whose entries carry "
            "'training' (and 'validation') blocks")])

    first_declared = None
    for _label, arm in arms:
        candidate = arm.get("training", {}).get("path")
        if isinstance(candidate, str) and candidate.strip():
            first_declared = candidate
            break
    resolved_root = _lora_root(manifest_path, first_declared, root)

    results = [Requirement(
        "present", display,
        f"{training_mode} package '{manifest.get('dataset', display)}' with "
        f"{len(arms)} arm(s), resolved against {resolved_root}",
        sha256=_sha256_file(manifest_path))]

    for label, arm in arms:
        train_requirement, train_file = _check_lora_file(
            arm.get("training", {}), label=label, role="training",
            root=resolved_root, training_mode=training_mode)
        results.append(train_requirement)
        validation_block = arm.get("validation")
        if not isinstance(validation_block, dict):
            results.append(Requirement(
                "missing", f"{label} validation",
                "the arm declares no validation split — evidence-grade "
                "training refuses the legacy fractional split "
                f"({LORA_AUTHORING_SPEC} §2.1)"))
            continue
        validation_requirement, validation_file = _check_lora_file(
            validation_block, label=label, role="validation",
            root=resolved_root, training_mode=training_mode)
        results.append(validation_requirement)
        if train_file is not None and validation_file is not None:
            results.append(_check_lora_split(
                label, train_file, validation_file, root=resolved_root,
                training_mode=training_mode))
        else:
            results.append(Requirement(
                "partial", f"{label} split",
                "not checked — resolve the file blocker(s) above first"))

    results.append(_check_lora_qc(manifest, root=resolved_root))
    return report(results)

"""Pin a judge rubric by path in a reviewed draft, computing its hash.

The Python client's twin of the Mac's ``experiment pin-rubric``
(``ExperimentStore.setJudgeRubric`` and the verb's judge parsing in
``ExperimentCLIRunner``): the rubric is named by its workspace path, its
SHA-256 is computed from the file's bytes here — never typed — and, with
``--judges``, the judge panel is declared beside it, together with the
``evaluation`` block the pair implies. Before this, pinning a rubric on this
client meant typing the hash into ``set-protocol``.

The write happens only in a reviewed draft: the caller supplies the
manifest's file digest from ``experiment inspect``, and a draft that changed
since is refused, as every reviewed client edit is. No model, no network.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import authoring_files as files
from ..experiment import command_vocabulary as vocabulary
from ..experiment import experiment_store as store
from ..experiment import manifest_declaration_policy as policy
from ..experiment.manifest_errors import ExperimentStoreError

#: The judge kinds a panel may name. Swift twin: ``ExperimentStore.knownJudgeKinds``.
JUDGE_KINDS = ("claude", "local", "openrouter")
RUBRIC_DIRECTORY = "prompts/rubrics"


class JudgeSpecError(ValueError):
    """A ``--judges`` or ``--judge-pin`` value the grammar cannot read: a
    malformed invocation, answered ``blocked`` (64)."""

    def __init__(self, reason: str, repair: str):
        super().__init__(reason)
        self.reason = reason
        self.repair_action = repair


def _repair(name: str, tail: str = "") -> str:
    return (f"{vocabulary.CLIENT_PROGRAM} experiment pin-rubric {name} "
            f"{RUBRIC_DIRECTORY}/<file>.md{tail} --manifest-sha256 <manifestFileSHA256 "
            f"from: {vocabulary.CLIENT_PROGRAM} experiment inspect {name}>")


def parse_judges(spec: str, name: str = "<name>") -> list[dict]:
    """``<name>:<kind>[:<model>[:<provider>]][,…]``, the Mac grammar exactly
    (``ExperimentCLIRunner.parseJudges``): an empty field is absent, and the
    fourth field is OpenRouter's serving provider and nobody else's."""
    shape = ("each judge is <name>:<kind>[:<model>[:<provider>]], "
             "comma-separated — kinds: " + " | ".join(JUDGE_KINDS))
    judges: list[dict] = []
    for raw in spec.split(","):
        field = raw.strip()
        if not field:
            continue
        parts = [part.strip() for part in field.split(":")]
        if not 2 <= len(parts) <= 4 or not parts[0]:
            raise JudgeSpecError(f"bad judge '{field}' — {shape}",
                                 _repair(name, " --judges <name>:<kind>[,…]"))
        if parts[1] not in JUDGE_KINDS:
            raise JudgeSpecError(
                f"unknown judge kind '{parts[1]}' in '{field}' — "
                + " | ".join(JUDGE_KINDS),
                _repair(name, " --judges <name>:<kind>[,…]"))
        judge = {"name": parts[0], "kind": parts[1]}
        if len(parts) >= 3 and parts[2]:
            judge["model"] = parts[2]
        if len(parts) == 4 and parts[3]:
            if parts[1] != "openrouter":
                raise JudgeSpecError(
                    f"judge '{parts[0]}' is {parts[1]}, which pins no serving "
                    "provider — the fourth field is OpenRouter's",
                    _repair(name, " --judges <name>:<kind>[,…]"))
            judge["provider"] = parts[3]
        judges.append(judge)
    if not judges:
        raise JudgeSpecError(f"--judges is empty — {shape}",
                             _repair(name, " --judges <name>:<kind>[,…]"))
    return judges


def parse_judge_pin(raw: str, name: str = "<name>") -> dict:
    """``<judge-name>=<revision>[:<dtype>]``: a LOCAL judge's weights pin
    (``ExperimentCLIRunner.parseJudgePin``). The revision must be a commit
    hash, and the dtype is stored in its canonical spelling."""
    shape = ("each pin is <judge-name>=<revision>[:<dtype>] — dtypes: "
             + " | ".join(policy.JUDGE_DTYPE_VOCABULARY)
             + " (aliases bf16/fp16/fp32)")
    judge, sep, value = raw.partition("=")
    judge = judge.strip()
    parts = [part.strip() for part in value.split(":")]
    if not sep or not judge or len(parts) > 2 or not parts[0]:
        raise JudgeSpecError(f"bad judge pin '{raw}' — {shape}",
                             _repair(name, " --judge-pin <judge-name>=<revision>[:<dtype>]"))
    revision = parts[0]
    if not policy.is_commit_like(revision):
        raise JudgeSpecError(
            f"judge pin '{judge}' names revision '{revision}', which is not a "
            "commit hash — a branch or tag is re-pointed by definition, so it "
            "cannot identify the weights a run used",
            _repair(name, f" --judge-pin {judge}=<commit-hash>[:<dtype>]"))
    pin = {"name": judge, "revision": revision}
    if len(parts) == 2 and parts[1]:
        canonical = policy.normalize_judge_dtype(parts[1])
        if canonical is None:
            raise JudgeSpecError(
                f"unknown judge dtype '{parts[1]}' — the loader accepts only "
                + ", ".join(policy.JUDGE_DTYPE_VOCABULARY)
                + " (aliases bf16/fp16/fp32). An unrecognized value used to "
                "load float32 silently, so the pin would be a false claim",
                _repair(name, f" --judge-pin {judge}=<revision>:<"
                        + "|".join(policy.JUDGE_DTYPE_VOCABULARY) + ">"))
        pin["dtype"] = canonical
    return pin


def _kind_owned(judge: dict) -> dict:
    """Only the fields the judge's kind owns (Swift ``keepingKindOwnedFields``):
    a local judge pins revision and dtype, an OpenRouter judge its provider,
    a Claude judge neither."""
    kept = dict(judge)
    kind = (kept.get("kind") or "").strip() or "claude"
    drop = {"local": ("provider",), "openrouter": ("revision", "dtype"),
            "claude": ("provider", "revision", "dtype")}.get(kind, ())
    for key in drop:
        kept.pop(key, None)
    return kept


def merge_judge_pins(declared: list[dict], pins: list[dict],
                     previous: list[dict], name: str) -> tuple[list[dict], list[str]]:
    """The panel to write, and what it kept or dropped from the one before
    (``ExperimentCLIRunner.mergingJudgePins``). ``--judges`` replaces the
    roster; a ``--judge-pin`` names a local judge's pins outright; otherwise a
    row whose name survives, still local, with the same model, inherits its
    pins, and a row whose model changed drops them — every inheritance and
    drop said in the notes."""
    def model(judge):
        return (judge.get("model") or "").strip() or None

    roster = [judge["name"] for judge in declared]
    for pin in pins:
        if pin["name"] not in roster:
            raise JudgeSpecError(
                f"--judge-pin '{pin['name']}' names no judge in the panel — "
                "declared: " + (", ".join(roster) if roster else "none"),
                _repair(name, f" --judges {pin['name']}:local:<model> "
                              f"--judge-pin {pin['name']}=<revision>[:<dtype>]"))
    pins_by_name = {pin["name"]: pin for pin in pins}
    previous_by_name: dict = {}
    for judge in previous:
        previous_by_name.setdefault(judge.get("name"), judge)
    notes: list[str] = []
    panel: list[dict] = []
    for judge in declared:
        row = dict(judge)
        pin = pins_by_name.get(judge["name"])
        if pin is not None:
            if judge["kind"] != "local":
                provider = (" and serving provider)" if judge["kind"] == "openrouter"
                            else ")")
                raise JudgeSpecError(
                    f"judge '{judge['name']}' is {judge['kind']}, which pins no "
                    "revision or dtype — those are local-judge pins (a "
                    f"{judge['kind']} judge's identity is its model slug{provider}",
                    _repair(name, f" --judges {judge['name']}:local:<model> "
                                  f"--judge-pin {judge['name']}=<revision>[:<dtype>]"))
            row["revision"] = pin["revision"]
            if pin.get("dtype"):
                row["dtype"] = pin["dtype"]
        old = previous_by_name.get(judge["name"])
        if (old is not None and old.get("kind") == "local"
                and row["kind"] == "local" and model(old) == model(row)):
            if not row.get("revision") and old.get("revision"):
                row["revision"] = old["revision"]
                notes.append(f"judge '{judge['name']}' revision "
                             f"{old['revision'][:12]}…")
            if not row.get("dtype") and old.get("dtype"):
                row["dtype"] = old["dtype"]
                notes.append(f"judge '{judge['name']}' dtype {old['dtype']}")
        elif (old is not None and old.get("kind") == "local"
              and (old.get("revision") or old.get("dtype"))):
            notes.append(
                f"dropped judge '{judge['name']}' revision/dtype pins — "
                + ("it now names a different model" if row["kind"] == "local"
                   else "it is no longer a local judge"))
        panel.append(row)
    return panel, notes


def evaluation_declaration(judges: list[dict], rubric_file: str,
                           existing: dict | None, study_model: str) -> dict | None:
    """The ``evaluation`` block a judge rubric and panel imply (Swift
    ``ExperimentStore.evaluationDeclaration``): a pinned rubric with at least
    one judge is a paired-judge evaluation whose judges and rubric live in
    their own fields; with no panel, a draft's inline rubric text keeps its
    inline declaration; with neither there is no evaluation to declare."""
    existing = existing if isinstance(existing, dict) else {}
    structured = existing.get("structuredPrompt")
    if judges and rubric_file.strip():
        block = {"kind": "pairedJudge", "judgeModel": "", "judgePrompt": ""}
    else:
        inline = str(existing.get("judgePrompt") or "").strip()
        if not inline:
            return None
        block = {"kind": "pairedJudge",
                 "judgeModel": existing.get("judgeModel") or study_model,
                 "judgePrompt": inline}
    if structured is not None:
        block["structuredPrompt"] = structured
    return block


def pin(name: str, rubric: str, *, expected: str, root: Path,
        judges: list[dict] | None = None, judge_pins: list[dict] = ()) -> dict:
    """Pin ``rubric`` (a workspace path) into the reviewed draft ``name``.

    ``judges`` None keeps the panel the draft has (``--judge-pin`` alone pins
    the judges already there); a list replaces it."""
    relative = rubric.strip()
    path = files.workspace_file(root, relative, read_only=True)
    if not path.is_file():
        raise ExperimentStoreError(
            f"judge rubric file not found: {path}",
            gate="missingPrerequisite",
            repair=(f"author {relative} under {RUBRIC_DIRECTORY}/ (a seeded "
                    "workspace ships prompts/rubrics/default-paired-v1.md), "
                    f"then {_repair(name)}"))
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with files.reviewed_draft(name, root, expected) as before:
        document = dict(before["document"])
        previous = [dict(j) for j in document.get("judges") or [] if isinstance(j, dict)]
        panel, notes = previous, []
        if judges is not None or judge_pins:
            panel, notes = merge_judge_pins(
                judges if judges is not None else previous, list(judge_pins),
                previous, name)
            panel = [_kind_owned(j) for j in panel if str(j.get("name") or "").strip()]
        # The bytes must not have changed between the read above and the write.
        if path.read_bytes() != data:
            files.refuse("The rubric changed while it was being pinned; pin it again.",
                         gate="staleManifest")
        document["judgeRubricFile"] = relative
        document["judgeRubricHash"] = digest
        if panel:
            document["judges"] = panel
        else:
            document.pop("judges", None)
        block = evaluation_declaration(panel, relative, document.get("evaluation"),
                                       str(document.get("modelID") or ""))
        if block is None:
            document.pop("evaluation", None)
        else:
            document["evaluation"] = block
        changed = document != before["document"]
        if changed:
            store.save_raw(document, str(root), expected_file_sha256=expected)
        return {"study": files.snapshot(name, root), "changed": changed,
                "experiment": name, "judgeRubricFile": relative,
                "judgeRubricHash": digest, "judges": panel,
                "evaluationKind": block["kind"] if block else None,
                **({"inheritedFromExistingDeclaration": notes} if notes else {})}

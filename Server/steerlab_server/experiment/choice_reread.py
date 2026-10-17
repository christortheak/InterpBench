"""Re-reading a run's stored choices with the current choice parser.

A choice is read once, when a run writes its records: ``parsedChoice`` in
``generations.jsonl`` (``condition_execution._execute_condition`` here,
``ExperimentTasks.judicialParses`` on the Mac). Runs are immutable, so a run
read by an older parser keeps that parser's mistakes, and every analysis,
exclusion, report, and export of it repeats them.

This module re-reads every record that carries ``parsedChoice`` from the reply
the record stored, in memory, with the parser this engine has now
(:func:`judicial.parse_choice`, version :data:`judicial.CHOICE_PARSER_VERSION`).
Nothing is ever written into a run directory. It has two callers:

* ``results check-choices`` (the client): a read-only report of what would
  change, per condition, with a few records of each kind to look at;
* ``analyze --reread-choices`` (both engines): the re-read values replace the
  stored ones for that one analysis, and the analysis writes
  :data:`STAMP_FILE` saying so.

**Where each item's options come from.** Sampled records carry ``promptID``
but not the options the item offered, so the options are read from the task
prompts file the run's own settings name (``taskPromptsFile`` and
``taskPromptsHash`` in the run's ``experiment.json``): the workspace file when
its SHA-256 matches, otherwise a matching copy under
``experiments/<study>/pinned/``. A file that does not match is never used.

**Cut-off replies.** ``truncated`` is the record's own ``finishReason`` (a
length or reasoning-length stop). Records without one come from engines older
than that field, which never stored whether a reply was cut off; they are read
as complete replies and counted separately, so a reader can see how many
readings rest on that assumption.

Swift twin: ``Sources/ExperimentKit/ChoiceReread.swift``. The shared
hand-written fixture ``Tests/Fixtures/cross-engine/choice-reread/`` holds both
engines to the same counts and the same effect rows.
"""

from __future__ import annotations

import hashlib
import json
import os

from . import judicial, lifecycle_gates, truncation_gate

#: The analysis stamp. Written only by ``analyze --reread-choices``.
STAMP_FILE = "choice-reread.json"

#: The advisory an ordinary analysis carries when the current parser would
#: read some stored choices differently (``cli_envelope.ADVISORY_CODES``).
ADVISORY_CODE = "storedChoicesReadDifferently"

#: The three ways a re-read can differ from the stored reading, in the order
#: every count and every sentence lists them.
CHANGED_TO_OTHER_OPTION = "changedToOtherOption"
CHANGED_TO_NO_ANSWER = "changedToNoAnswer"
GAINED_ANSWER = "gainedAnswer"
KINDS: tuple[str, ...] = (CHANGED_TO_OTHER_OPTION, CHANGED_TO_NO_ANSWER,
                          GAINED_ANSWER)

#: Records shown per kind of change: enough to look at, never the whole list.
EXAMPLES_PER_KIND = 5

#: Every count, in this order. ``storedChoices`` is every record that carries
#: ``parsedChoice`` (a stored parse failure counts); ``changed`` is the sum of
#: the three kinds; ``noFinishReason`` counts records from engines that did
#: not record why generation ended, and ``noFinishReasonChanged`` how many of
#: those read differently now.
COUNT_KEYS: tuple[str, ...] = ("storedChoices", "unchanged") + KINDS + (
    "changed", "noFinishReason", "noFinishReasonChanged")

#: The analysis stamp's note: what happened, and what did not.
NOTE = ("Every record that carries parsedChoice was read again from its "
        "stored reply with the choice parser this engine has now. The "
        "re-read choices replace the stored ones in this analysis only: "
        "every choice outcome, exclusion, and table here uses them. The run "
        "directory, including generations.jsonl, was not changed. Records "
        "without finishReason come from engines that did not record whether "
        "a reply was cut off by the length limit; they were read as complete "
        "replies.")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(path: str, root: str) -> str:
    try:
        relative = os.path.relpath(path, root)
    except ValueError:
        return path
    return path if relative.startswith(os.pardir) else relative.replace(os.sep, "/")


def _run_settings(run_dir: str) -> dict:
    path = os.path.join(run_dir, "experiment.json")
    try:
        with open(path, encoding="utf-8") as handle:
            settings = json.load(handle)
    except (OSError, ValueError):
        settings = None
    if not isinstance(settings, dict):
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"Run '{os.path.basename(run_dir)}' has no readable experiment.json, "
            "so the task prompts it was made from, and the answer options each "
            "item offered, cannot be established. Its stored choices cannot be "
            "re-read.",
            repair="Choose a run that holds its own experiment.json (every "
                   "complete run does); a run copied in without it cannot be "
                   "re-read.")
    return settings


def _pinned_candidates(root: str, studies: list[str], relative: str) -> list[str]:
    """Likely copies first (the engine's mirror of the workspace path, and the
    Mac's ``task-prompts-<name>``), then every other JSONL file in the
    studies' ``pinned/`` folders."""
    ordered: list[str] = []
    for study in studies:
        pinned = os.path.join(root, "experiments", study, "pinned")
        if not os.path.isdir(pinned):
            continue
        ordered.append(os.path.join(pinned, *relative.split("/")))
        ordered.append(os.path.join(pinned, "task-prompts-" + os.path.basename(relative)))
        for folder, _directories, files in sorted(os.walk(pinned)):
            for name in sorted(files):
                if name.endswith(".jsonl"):
                    ordered.append(os.path.join(folder, name))
    seen, unique = set(), []
    for path in ordered:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return unique


def task_prompt_options(run_dir: str, root: str, *,
                        study: str | None = None) -> tuple[dict, dict]:
    """``(options by item id, where they were read from)`` for one run.

    Refuses (``missingPrerequisite`` or ``pinDrift``) rather than guessing when
    no file with the run's pinned hash can be found."""
    from . import task_inputs
    settings = _run_settings(run_dir)
    relative = settings.get("taskPromptsFile")
    digest = settings.get("taskPromptsHash")
    run_name = os.path.basename(os.path.normpath(run_dir))
    if not isinstance(relative, str) or not relative or not isinstance(digest, str) or not digest:
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"Run '{run_name}' names no pinned task prompts file (taskPromptsFile "
            "and taskPromptsHash in its experiment.json), so the answer options "
            "each item offered cannot be verified. Its stored choices cannot be "
            "re-read.",
            repair="Only a run made from pinned task prompts can be re-read. "
                   "Pin the study's task prompts before its next run.")
    studies = []
    for name in (settings.get("name"), study):
        if isinstance(name, str) and name and name not in studies and "/" not in name:
            studies.append(name)
    workspace_file = relative if os.path.isabs(relative) else os.path.join(root, relative)
    candidates = [(workspace_file, "workspace")] + [
        (path, "pinned") for path in _pinned_candidates(root, studies, relative)]
    for path, location in candidates:
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        if _sha256(data) != digest:
            continue
        prompts = task_inputs.parse_prompts(data.decode("utf-8"))
        options = {str(prompt["id"]): [str(option) for option in prompt["options"]]
                   for prompt in prompts
                   if isinstance(prompt.get("options"), list) and prompt["options"]}
        return options, {"file": relative, "sha256": digest,
                         "readFrom": _relative(os.path.abspath(path), os.path.abspath(root)),
                         "location": location}
    pinned_hint = (f"experiments/{studies[0]}/pinned/{relative}" if studies
                   else f"experiments/<study>/pinned/{relative}")
    raise lifecycle_gates.refusing(
        lifecycle_gates.PIN_DRIFT,
        f"No copy of the task prompts file run '{run_name}' was made from could "
        f"be found: {relative} with SHA-256 {digest[:12]}…. The file in the "
        "workspace is missing or has changed since the run, and no matching "
        "copy is in the study's pinned folder. A file with other bytes might "
        "offer other options, so the stored choices are not re-read with it.",
        repair=f"Restore {relative} to the bytes the run used (for example from "
               f"version control), or put a copy with those bytes at "
               f"{pinned_hint}; then run this command again.")


def _record_id(line: int, record: dict) -> dict:
    found = {"line": line, "condition": record.get("condition", ""),
             "promptID": str(record.get("promptID", ""))}
    if isinstance(record.get("sampleIndex"), int) and not isinstance(record.get("sampleIndex"), bool):
        found["sampleIndex"] = record["sampleIndex"]
    return found


def _zero_counts() -> dict:
    return {key: 0 for key in COUNT_KEYS}


def kind_of(stored, fresh) -> str | None:
    """Which kind of change ``stored`` → ``fresh`` is, or ``None`` when the
    two readings are the same."""
    if stored == fresh:
        return None
    if stored is None:
        return GAINED_ANSWER
    if fresh is None:
        return CHANGED_TO_NO_ANSWER
    return CHANGED_TO_OTHER_OPTION


def reread(entries, options: dict, *, prompts_hash: str | None = None) -> tuple[dict, dict]:
    """Re-read ``entries`` — ``(line, record)`` pairs in file order — and count.

    Returns ``(fresh, report)``: ``fresh`` maps each re-read line to its new
    reading (``None`` for no answer), and ``report`` holds ``total``,
    ``conditions``, and ``examples``. Refuses (``pinDrift``) when a record
    names an item the verified file offers no options for, or carries its own
    ``taskPromptsHash`` naming another file: its options cannot be known."""
    fresh_by_line: dict[int, object] = {}
    total = _zero_counts()
    conditions: dict[str, dict] = {}
    candidates: dict[str, dict[str, list[dict]]] = {kind: {} for kind in KINDS}
    unmatched: list[str] = []
    for line, record in entries:
        if not isinstance(record, dict) or "parsedChoice" not in record:
            continue
        prompt_id = str(record.get("promptID", ""))
        item_options = options.get(prompt_id)
        own_hash = record.get("taskPromptsHash")
        if item_options is None or (prompts_hash and isinstance(own_hash, str)
                                    and own_hash and own_hash != prompts_hash):
            if prompt_id not in unmatched:
                unmatched.append(prompt_id)
            continue
        reason = record.get(truncation_gate.RECORD_KEY)
        output = record.get("output")
        fresh = judicial.parse_choice(
            output if isinstance(output, str) else "", list(item_options),
            truncated=reason in truncation_gate.CUT_OFF_REASONS)
        stored = record["parsedChoice"]
        fresh_by_line[line] = fresh
        condition = str(record.get("condition", ""))
        block = conditions.setdefault(condition, _zero_counts())
        kind = kind_of(stored, fresh)
        for counts in (total, block):
            counts["storedChoices"] += 1
            if kind is None:
                counts["unchanged"] += 1
            else:
                counts[kind] += 1
                counts["changed"] += 1
            if reason is None:
                counts["noFinishReason"] += 1
                if kind is not None:
                    counts["noFinishReasonChanged"] += 1
        if kind is not None:
            shown = candidates[kind].setdefault(condition, [])
            if len(shown) < EXAMPLES_PER_KIND:
                shown.append({**_record_id(line, record), "stored": stored,
                              "reread": fresh})
    if unmatched:
        listed = ", ".join(f"'{item}'" for item in unmatched[:3])
        more = len(unmatched) - 3
        raise lifecycle_gates.refusing(
            lifecycle_gates.PIN_DRIFT,
            f"{len(unmatched)} item(s) with stored choices in this run ({listed}"
            f"{f', and {more} more' if more > 0 else ''}) are not items with "
            "answer options in the task prompts file the run's settings name, or "
            "their records name a different prompts file. The run was probably "
            "made from a different prompts file, so the options those items "
            "offered cannot be known and nothing was re-read.",
            repair="Re-read only runs made from the study's own pinned task "
                   "prompts. A run made with a different prompts file has to be "
                   "made again to be read with the current parser.")
    examples = {}
    for kind in KINDS:
        by_condition = candidates[kind]
        picked: list[dict] = []
        round_number = 0
        while len(picked) < EXAMPLES_PER_KIND:
            added = False
            for condition in sorted(by_condition):
                shown = by_condition[condition]
                if round_number < len(shown) and len(picked) < EXAMPLES_PER_KIND:
                    picked.append(shown[round_number])
                    added = True
            if not added:
                break
            round_number += 1
        examples[kind] = picked
    return fresh_by_line, {"total": total, "conditions": conditions,
                           "examples": examples}


def _numbered_records(run_dir: str) -> tuple[list[tuple[int, dict]], int]:
    """``((line, record) pairs, unreadable lines)``, counting non-blank lines
    from 1 — the numbering both engines use."""
    entries, unreadable, line = [], 0, 0
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            line += 1
            try:
                record = json.loads(text)
            except ValueError:
                unreadable += 1
                continue
            entries.append((line, record))
    return entries, unreadable


def check(run_dir: str, root: str, *, study: str | None = None) -> dict:
    """The read-only report ``results check-choices`` answers with."""
    options, source = task_prompt_options(run_dir, root, study=study)
    entries, unreadable = _numbered_records(run_dir)
    _fresh, report = reread(entries, options, prompts_hash=source["sha256"])
    return {"parserVersion": judicial.CHOICE_PARSER_VERSION,
            "run": os.path.basename(os.path.normpath(run_dir)),
            "taskPrompts": source, "unreadableLines": unreadable, **report}


def stamp(run_dir: str, source: dict, report: dict) -> dict:
    """The analysis stamp (cross-engine shape; Swift twin
    ``ChoiceReread.Stamp``)."""
    return {"parserVersion": judicial.CHOICE_PARSER_VERSION,
            "sourceRun": os.path.basename(os.path.normpath(run_dir)),
            "runUnchanged": True, "taskPrompts": source,
            "total": report["total"], "conditions": report["conditions"],
            "examples": report["examples"], "note": NOTE}


def for_analysis(run_dir: str, records: list[dict], root: str | None, *,
                 study: str | None = None, log=None) -> tuple[list[dict], dict]:
    """``analyze --reread-choices``: re-read ``records`` (the run's records in
    file order, already in memory) and return them with each ``parsedChoice``
    replaced by its re-read value, plus the stamp. The run directory is not
    touched; only these in-memory dicts change."""
    from . import paths
    base = paths.project_root() if root is None else root
    options, source = task_prompt_options(run_dir, base, study=study)
    entries = list(enumerate(records, start=1))
    fresh, report = reread(entries, options, prompts_hash=source["sha256"])
    for line, value in fresh.items():
        records[line - 1]["parsedChoice"] = value
    result = stamp(run_dir, source, report)
    if log is not None:
        log(f"choice re-read: {sentence(result)}")
    return records, result


def write_stamp(out: str, value: dict) -> None:
    with open(os.path.join(out, STAMP_FILE), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def sentence(value: dict) -> str:
    """One plain sentence from a stamp's own numbers, and a second when some
    readings rest on records that never said whether a reply was cut off.
    Readers in other languages build the same words (the explorer's
    ``choiceRereadSentence``, the app's ``ChoiceReread.sentence``)."""
    total = value.get("total") if isinstance(value.get("total"), dict) else {}
    changed = total.get("changed", 0)
    stored = total.get("storedChoices", 0)
    text = (f"Choices were re-read from the stored replies with choice parser "
            f"version {value.get('parserVersion', 'unknown')}; {changed} of "
            f"{_plural(stored, 'stored choice')} changed, and the run itself is "
            "unchanged.")
    older = total.get("noFinishReason", 0)
    if older:
        text += (f" {older} of those replies come from an older engine that did "
                 "not record whether a reply was cut off by the length limit, so "
                 f"they were read as complete replies ({total.get('noFinishReasonChanged', 0)} "
                 "of them changed).")
    return text


def advisory(run_dir: str, root: str | None, *, study: str | None = None) -> dict | None:
    """The discoverability count for an ordinary analysis: how many stored
    choices in ``run_dir`` the current parser would read differently.

    Never raises and never blocks: a run whose options cannot be verified, or
    that has no stored choices, answers ``None``."""
    from . import paths
    try:
        base = paths.project_root() if root is None else root
        report = check(run_dir, base, study=study)
    except Exception:  # noqa: BLE001 - an advisory never stops an analysis
        return None
    total = report["total"]
    if not total["changed"]:
        return None
    return {"run": report["run"], "changed": total["changed"],
            "storedChoices": total["storedChoices"],
            "parserVersion": report["parserVersion"]}


def advisory_for_analysis(analysis_dir: str, root: str | None, *,
                          study: str | None = None) -> dict | None:
    """:func:`advisory` for the run an analysis directory names in its
    ``source-run.txt`` (a sibling under ``runs/``). Never raises."""
    try:
        with open(os.path.join(analysis_dir, "source-run.txt"), encoding="utf-8") as handle:
            source = os.path.basename(handle.readline().strip().rstrip("/"))
    except OSError:
        return None
    if not source:
        return None
    return advisory(os.path.join(os.path.dirname(os.path.normpath(analysis_dir)), source),
                    root, study=study)


def advisory_detail(found: dict, *, check_command: str, analyze_command: str) -> str:
    """The advisory's sentence, with the asking command line's spellings.
    Swift twin: ``ChoiceReread.advisoryDetail``."""
    return (f"{found['changed']} of {_plural(found['storedChoices'], 'stored choice')} "
            f"in run '{found['run']}' would be read differently by the current "
            f"choice parser (version {found['parserVersion']}). This analysis used "
            "the choices as the run stored them. To see which changed: "
            f"{check_command}. To analyze with the current reading: "
            f"{analyze_command}. The run itself is never changed.")

"""Export a completed run's stored results as files a researcher can take elsewhere.

One implementation for both command lines and the app. It reads a study's run,
analysis, and evaluation directories from a workspace and writes a NEW folder
outside ``runs/``: tables that open in R, Stata, SPSS, or a spreadsheet,
transcripts for coding by hand, a plain-language methods summary, a codebook,
and a manifest that traces every file back to the run it came from.

Three rules hold everywhere in this module:

* **A run directory is never modified.** Everything under ``runs/`` is read,
  never written: custody verification re-reads a run's files, and an added
  file would break it. Output lands in a folder this module creates.
* **No statistic is recomputed.** Estimates, intervals, and p-values are
  copied from what the engine stored. Names and labels are normalized so one
  set of columns covers both engines; numbers are not touched.
* **What was not stored is said to be not available.** An empty cell means the
  run did not record the value, and ``methods.md`` lists what is missing.

Standard library only. Nothing here loads a model, imports an engine module at
import time, or needs a GPU, so it works on an authoring-only install.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone

EXPORT_SCHEMA_VERSION = 1
EXPORT_KIND = "steerlab.resultsExport"

#: Where an export lands when no destination is named: a new folder under this
#: directory of the workspace. Never ``runs/``.
EXPORTS_DIRECTORY = "exports"

#: What a line break inside a text cell is written as. Stata and SPSS read a
#: table one line per row, so a response with paragraphs would otherwise break
#: their import. The mark is visible, and replacing it with a line break
#: restores the recorded text.
LINE_BREAK_MARK = " ¶ "

#: Manifest keys that may hold a study's declared primary outcome. One place
#: to change if the declaring stream settles on a different spelling.
PRIMARY_OUTCOME_KEYS = ("primaryOutcome",)

PYTHON_CLIENT = "steerlab"
MAC_CLIENT = "steerlab-cli"
APP_CLIENT = "app"
CLIENTS = (PYTHON_CLIENT, MAC_CLIENT, APP_CLIENT)

NOT_FOUND_CODE = "notFound"
NO_COMPLETED_RUN_CODE = "noCompletedRun"
RUN_NOT_USABLE_CODE = "runNotUsable"
DESTINATION_REFUSED_CODE = "exportDestinationRefused"
USAGE_CODE = "usage"


class ResultsExportRefusal(ValueError):
    """An export that was declined, with a repair the caller can carry out.

    ``state`` uses the command-line envelope's vocabulary: ``notFound`` when
    the named study or run does not exist, ``blocked`` for a malformed
    request, and ``refused`` when a well-formed request cannot be met yet.
    """

    def __init__(self, reason, *, code, repair_action, state="refused"):
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.repair_action = repair_action
        self.state = state


# --- how a person does each step on the client that asked ----------------------


def _steps(client, study):
    if client == MAC_CLIENT:
        return {
            "list": "steerlab-cli experiment list  (the studies in this workspace)",
            "run": f"steerlab-cli experiment run {study}",
            "analyze": f"steerlab-cli experiment analyze {study}",
            "evaluate": f"steerlab-cli experiment evaluate {study}",
            "export": f"steerlab-cli results export {study}",
            "newest": "Leave --run out to export the newest completed run.",
            "destination": "Name a new or empty folder with --out, or leave --out "
                           f"out to write under {EXPORTS_DIRECTORY}/.",
        }
    if client == APP_CLIENT:
        return {
            "list": "Choose a study on the Studies page.",
            "run": "Run the study",
            "analyze": "Analyze the run",
            "evaluate": "Evaluate the run with the study's judges",
            "export": "choose Export Results again",
            "newest": "Select a completed run in the study's results, or export "
                      "the newest completed run.",
            "destination": "Choose a new or empty folder.",
        }
    return {
        "list": "steerlab experiment list  (the studies in this workspace)",
        "run": f"steerlab run {study} --runner <url>",
        "analyze": f"steerlab run {study} --runner <url> --verb analyze",
        "evaluate": f"steerlab run {study} --runner <url> --verb evaluate",
        "export": f"steerlab results export {study}",
        "newest": "Leave --run out to export the newest completed run.",
        "destination": "Name a new or empty folder with --out, or leave --out "
                       f"out to write under {EXPORTS_DIRECTORY}/.",
    }


# --- reading, with a record of exactly what was read ---------------------------


class _Sources:
    """Reads files and remembers each one's bytes, so the export's manifest
    names what it was built from."""

    def __init__(self, root):
        self.root = root
        self.entries = []

    def _note(self, path, role, digest, size):
        self.entries.append({"path": _relative(path, self.root), "role": role,
                             "sha256": digest, "bytes": size})

    def read(self, path, role):
        """The file's bytes, or None when it is absent or unreadable."""
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            return None
        self._note(path, role, hashlib.sha256(data).hexdigest(), len(data))
        return data

    def json(self, path, role):
        data = self.read(path, role)
        if data is None:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None

    def text(self, path, role):
        data = self.read(path, role)
        return None if data is None else data.decode("utf-8", errors="replace")

    def lines(self, path, role):
        """Each line of a large file, hashed as it streams past."""
        digest = hashlib.sha256()
        size = 0
        with open(path, "rb") as handle:
            for raw in handle:
                digest.update(raw)
                size += len(raw)
                yield raw.decode("utf-8", errors="replace")
        self._note(path, role, digest.hexdigest(), size)


def _relative(path, root):
    try:
        relative = os.path.relpath(path, root)
    except ValueError:
        return path
    if relative.startswith(os.pardir):
        return path
    return relative.replace(os.sep, "/")


def _peek_json(path):
    """A small JSON file read for discovery only, not recorded as a source."""
    try:
        with open(path, "rb") as handle:
            value = json.loads(handle.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _peek_line(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.readline().strip()
    except (OSError, UnicodeDecodeError):
        return ""


# --- finding the run, its analysis, and its evaluation -------------------------


def _runs_root(root):
    return os.path.join(root, "runs")


def _directories(runs_root):
    try:
        names = sorted(os.listdir(runs_root), reverse=True)
    except OSError:
        return []
    return [name for name in names
            if os.path.isdir(os.path.join(runs_root, name))]


def _experiment_of(directory):
    """The study a run directory says it belongs to, or None."""
    config = _peek_json(os.path.join(directory, "config.json")) or {}
    if isinstance(config.get("experiment"), str) and config["experiment"]:
        return config["experiment"]
    snapshot = _peek_json(os.path.join(directory, "experiment.json")) or {}
    name = snapshot.get("name")
    return name if isinstance(name, str) and name else None


def _run_pattern(study):
    return re.compile(r"-exp-" + re.escape(study) + r"-(?:multi-agent-)?run(?:-\d+)?$")


def _analysis_pattern(study):
    return re.compile(r"-exp-" + re.escape(study) + r"-analyze(?:-\d+)?$")


def _evaluation_pattern(study):
    return re.compile(r"-exp-" + re.escape(study) + r"-evaluate(?:-judgment)?(?:-\d+)?$")


def _missing_run_files(directory):
    return [name for name in ("generations.jsonl", "report.json")
            if not os.path.isfile(os.path.join(directory, name))]


def _study_runs(root, study):
    """(completed, unfinished) run directory names of a study, newest first."""
    runs_root = _runs_root(root)
    pattern = _run_pattern(study)
    completed, unfinished = [], []
    for name in _directories(runs_root):
        if not pattern.search(name):
            continue
        directory = os.path.join(runs_root, name)
        owner = _experiment_of(directory)
        if owner is not None and owner != study:
            continue
        (unfinished if _missing_run_files(directory) else completed).append(name)
    return completed, unfinished


def _analysis_source(directory):
    """The run an analysis directory says it analyzed (a directory name)."""
    named = _peek_line(os.path.join(directory, "source-run.txt"))
    if named:
        return os.path.basename(named.rstrip("/"))
    report = _peek_json(os.path.join(directory, "analysis.json")) or {}
    source = report.get("sourceRun")
    return os.path.basename(source.rstrip("/")) if isinstance(source, str) and source else None


def _find_analysis(root, study, run_name):
    runs_root = _runs_root(root)
    pattern = _analysis_pattern(study)
    for name in _directories(runs_root):
        directory = os.path.join(runs_root, name)
        if (pattern.search(name)
                and os.path.isfile(os.path.join(directory, "effect-sizes.csv"))
                and _analysis_source(directory) == run_name):
            return name
    return None


#: The two shapes an evaluation takes, each with its completion report and its
#: row file. A directory with rows and no report is an unfinished evaluation,
#: which an engine never summarizes, so it is not exported either.
_EVALUATION_KINDS = {
    "paired": ("judge-report.json", "judgments.jsonl"),
    "coding": ("coding-report.json", "codings.jsonl"),
}


def _evaluation_source(report):
    for key in ("sourceRun", "sourceRunDirectory"):
        value = report.get(key)
        if isinstance(value, str) and value:
            return os.path.basename(value.rstrip("/"))
    return None


def _find_evaluation(root, study, run_name, kind):
    runs_root = _runs_root(root)
    pattern = _evaluation_pattern(study)
    report_name, rows_name = _EVALUATION_KINDS[kind]
    for name in _directories(runs_root):
        if not pattern.search(name):
            continue
        directory = os.path.join(runs_root, name)
        if not os.path.isfile(os.path.join(directory, rows_name)):
            continue
        report = _peek_json(os.path.join(directory, report_name))
        if report is not None and _evaluation_source(report) == run_name:
            return name
    return None


def _study_exists(root, study):
    return (os.path.isfile(os.path.join(root, "experiments", study, "experiment.json"))
            or os.path.isfile(os.path.join(root, "experiments", study + ".json")))


def _check_study_name(study, steps):
    if (not isinstance(study, str) or not study.strip() or study != study.strip()
            or "/" in study or "\\" in study or study.startswith(".")):
        raise ResultsExportRefusal(
            f"'{study}' is not a study name.", code=USAGE_CODE, state="blocked",
            repair_action=steps["list"])


def _select_run(root, study, run, steps, client):
    """The run directory name to export, or a refusal that says why not."""
    runs_root = _runs_root(root)
    if run is None:
        completed, unfinished = _study_runs(root, study)
        if completed:
            return completed[0]
        if not unfinished and not _study_exists(root, study):
            raise ResultsExportRefusal(
                f"There is no study named '{study}' in this workspace, and no "
                "run of one.", code=NOT_FOUND_CODE, state="notFound",
                repair_action=steps["list"])
        reason = (f"Study '{study}' has no completed run in this workspace "
                  "yet, so there are no results to export.")
        if unfinished:
            count = len(unfinished)
            reason += (f" {count} run{'s' if count != 1 else ''} of it "
                       f"{'are' if count != 1 else 'is'} unfinished: a run "
                       "writes report.json when it completes, and "
                       f"{'these have' if count != 1 else 'this one has'} "
                       "none.")
        raise ResultsExportRefusal(
            reason, code=NO_COMPLETED_RUN_CODE,
            repair_action=f"{steps['run']}, then {steps['export']}. A run that "
                          "finished on other hardware has to be brought into "
                          "this workspace before it can be exported.")

    candidates = [run] if os.path.isabs(run) else [
        os.path.join(root, run), os.path.join(runs_root, run)]
    directory = next((c for c in candidates if os.path.isdir(c)), None)
    real_runs = os.path.realpath(runs_root)
    if (directory is None
            or os.path.dirname(os.path.realpath(directory)) != real_runs):
        raise ResultsExportRefusal(
            f"'{run}' is not a run directory in this workspace's runs/ folder.",
            code=NOT_FOUND_CODE, state="notFound",
            repair_action="Name a directory under runs/, for example "
                          f"runs/<run-directory>. {steps['newest']}")
    name = os.path.basename(os.path.realpath(directory))
    missing = _missing_run_files(os.path.join(runs_root, name))
    if missing:
        raise ResultsExportRefusal(
            f"Run '{name}' cannot be exported: it has no "
            f"{_and(missing)}. A run writes report.json when it completes.",
            code=RUN_NOT_USABLE_CODE,
            repair_action=f"Export a completed run. {steps['newest']}")
    owner = _experiment_of(os.path.join(runs_root, name))
    if owner is not None and owner != study and not (
            _find_analysis(root, study, name)
            or any(_find_evaluation(root, study, name, kind)
                   for kind in _EVALUATION_KINDS)):
        repair = (f"Export it from the study '{owner}' instead."
                  if client == APP_CLIENT else
                  "Export it under its own study: "
                  f"{_steps(client, owner)['export']} --run runs/{name}")
        raise ResultsExportRefusal(
            f"Run '{name}' belongs to study '{owner}', and study '{study}' "
            "has no analysis or evaluation of it.", code=RUN_NOT_USABLE_CODE,
            repair_action=repair)
    return name


def _destination(root, study, out, steps, now):
    """The folder to create, checked BEFORE anything is read or written."""
    real_runs = os.path.realpath(_runs_root(root))
    if out is None:
        base = os.path.join(root, EXPORTS_DIRECTORY,
                            f"{study}-results-{now.strftime('%Y%m%d-%H%M%S')}")
        target, counter = base, 2
        while os.path.exists(target):
            target = f"{base}-{counter}"
            counter += 1
        return target
    target = out if os.path.isabs(out) else os.path.join(root, out)
    target = os.path.abspath(target)
    real = os.path.realpath(target)
    if real == real_runs or real.startswith(real_runs + os.sep):
        raise ResultsExportRefusal(
            "Results cannot be exported into runs/. Run directories are never "
            "modified, because verifying a run re-reads its files.",
            code=DESTINATION_REFUSED_CODE, repair_action=steps["destination"])
    if os.path.exists(target) and not os.path.isdir(target):
        raise ResultsExportRefusal(
            f"'{out}' is a file, and an export is a folder of files.",
            code=DESTINATION_REFUSED_CODE, repair_action=steps["destination"])
    if os.path.isdir(target) and os.listdir(target):
        raise ResultsExportRefusal(
            f"The folder '{out}' already contains files. An export is always "
            "written to a new or empty folder, so nothing is overwritten.",
            code=DESTINATION_REFUSED_CODE, repair_action=steps["destination"])
    return target


# --- cells and tables ----------------------------------------------------------

#: Every character a program may treat as the end of a line. The two Unicode
#: separators are built with chr() so this source holds no invisible character.
_LINE_BREAKS = re.compile("\r\n|[\n\r\x0b\x0c\x85" + chr(0x2028) + chr(0x2029) + "]")
_REPLACEMENT = chr(0xFFFD)
_NON_FINITE = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity",
               "+infinity", "-infinity"}


def _one_line(text):
    return _LINE_BREAKS.sub(LINE_BREAK_MARK, text.replace("\x00", _REPLACEMENT))


def _cell(value):
    """One table cell. Empty means the value was not recorded."""
    if value is None:
        return ""
    if value is True:
        return "1"
    if value is False:
        return "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ""
    if isinstance(value, str):
        return _one_line(value)
    return _one_line(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _stored_number(text):
    """A number copied from an engine's table, unchanged. A value that is not
    a finite number (an undefined statistic) becomes an empty cell, because
    Stata and SPSS would otherwise read the whole column as text."""
    text = (text or "").strip()
    return "" if text.lower() in _NON_FINITE else text


def _column_slug(key):
    return re.sub(r"[^0-9a-z]+", "_", str(key).lower()).strip("_") or "value"


class _Table:
    """One exported table: its columns, what each means, and its rows."""

    def __init__(self, file, title, row_is):
        self.file = file
        self.title = title
        self.row_is = row_is
        self.columns = []       # (name, meaning, getter)
        self.rows = []
        self.notes = []

    def add(self, name, meaning, getter):
        taken = {column[0] for column in self.columns}
        unique, counter = name, 2
        while unique in taken:
            unique = f"{name}_{counter}"
            counter += 1
        self.columns.append((unique, meaning, getter))

    def fill(self, records):
        self.rows = [[_cell(getter(record)) for _, _, getter in self.columns]
                     for record in records]

    @property
    def names(self):
        return [name for name, _, _ in self.columns]

    def csv_text(self):
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.names)
        writer.writerows(self.rows)
        return buffer.getvalue()


def _key(name):
    return lambda record: record.get(name)


def _nested(outer, inner):
    def getter(record):
        value = record.get(outer)
        return value.get(inner) if isinstance(value, dict) else None
    return getter


def _parse_failed(name):
    """1 when the engine tried to read the value and could not (the record
    stores null), 0 when it read one, empty when the reading does not apply."""
    def getter(record):
        if name not in record:
            return None
        return record[name] is None
    return getter


def _endpoint_unparsed(record):
    """The engines' own rule for a turn's declared outcome
    (``turn_endpoint.csv_rows``): unparsed when the stamp says so, or when it
    holds no value."""
    endpoint = record.get("endpoint")
    if not isinstance(endpoint, dict) or not endpoint.get("name"):
        return None
    return bool(endpoint.get("unparsed")) or endpoint.get("value") is None


def _dynamic_keys(records, field):
    keys = []
    for record in records:
        value = record.get(field)
        if isinstance(value, dict):
            for key in value:
                if key not in keys:
                    keys.append(key)
    return keys


# --- generations.jsonl ---------------------------------------------------------

#: Record fields the response table carries. Everything else a record holds
#: stays in the run, and the codebook names it.
_RESPONSE_FIELDS = (
    "experiment", "condition", "promptID", "promptIndex", "sampleIndex",
    "seed", "replicateIndex", "turnTitle", "speakerAgentID", "speakerName",
    "prompt", "output", "finishReason", "wordCount", "distinct2",
    "parsedChoice", "parsedMonths", "target", "arm", "caseID", "anchorMonths",
    "severity", "factors", "markerDensity", "readerScores", "endpoint",
    "voiceLint", "modelID", "modelRevision",
)
_READOUT_FIELDS = (
    "experiment", "condition", "promptID", "promptIndex", "instrument",
    "target", "targetSource", "selected", "margin", "choiceProbability",
    "logOdds", "ordinalPosition", "arm", "caseID", "anchorMonths", "severity",
    "factors",
)
#: Settings each record repeats. Their distinct values go into the methods
#: summary; they are not outcomes, so they are not table columns.
_OBSERVED_FIELDS = ("modelID", "modelRevision", "device", "dtype", "engine",
                    "temperature", "topP", "topK", "doSample", "promptMode",
                    "seedPolicy", "experimentHash", "taskPromptsFile",
                    "taskPromptsHash")


def _load_generations(path, sources):
    responses, readouts, failed_conditions = [], [], []
    response_fields, readout_fields = set(), set()
    observed = {name: [] for name in _OBSERVED_FIELDS}
    unreadable = other = 0
    for line in sources.lines(path, "run"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            unreadable += 1
            continue
        if not isinstance(record, dict):
            unreadable += 1
            continue
        if "error" in record:
            failed_conditions.append({"condition": record.get("condition"),
                                      "error": record.get("error")})
            continue
        if "instrument" in record:
            readout_fields.update(record)
            readouts.append({k: record[k] for k in _READOUT_FIELDS if k in record})
            continue
        if "output" not in record:
            other += 1
            continue
        response_fields.update(record)
        for name in _OBSERVED_FIELDS:
            value = record.get(name)
            if isinstance(value, (str, int, float, bool)) and value not in observed[name]:
                observed[name].append(value)
        responses.append({k: record[k] for k in _RESPONSE_FIELDS if k in record})
    return {"responses": responses, "readouts": readouts,
            "failedConditions": failed_conditions,
            "responseFields": response_fields, "readoutFields": readout_fields,
            "observed": observed, "unreadableLines": unreadable,
            "otherRecords": other}


def _is_turn(record):
    return record.get("speakerName") is not None or record.get("turnTitle") is not None


def _item_columns(table, records, fields):
    """Item metadata and factorial cells, shared by both record tables."""
    for name, column, meaning in (
            ("target", "target", "The answer option the item declares as its target, when it declares one."),
            ("arm", "arm", "The item's arm label, as written in the task file."),
            ("caseID", "case_id", "The item's case identifier, as written in the task file."),
            ("anchorMonths", "anchor_months", "The item's anchor value, as written in the task file."),
            ("severity", "severity", "The item's severity value, as written in the task file.")):
        if name in fields:
            table.add(column, meaning, _key(name))
    for factor in _dynamic_keys(records, "factors"):
        table.add("factor_" + _column_slug(factor),
                  f"The item's level on the design factor '{factor}', as written in the task file.",
                  _nested("factors", factor))


def _responses_table(study, run_name, loaded):
    records, fields = loaded["responses"], loaded["responseFields"]
    table = _Table("responses.csv", "Responses", "one response the model generated")
    table.add("study", "The study the response was generated under.",
              lambda r: r.get("experiment") or study)
    table.add("run", "The run directory the response comes from.", lambda r: run_name)
    table.add("condition", "The condition (arm) the response was generated under. "
              "The arm with no intervention is named baseline.", _key("condition"))
    table.add("item", "The identifier of the task item the model answered. In a "
              "multi-agent study it is the identifier of the turn.", _key("promptID"))
    table.add("item_index", "The item's position in the task file, counting from 0.",
              _key("promptIndex"))
    table.add("sample_index", "Which sample of the same condition and item this is, "
              "counting from 0. A response pairs with the baseline response that has "
              "the same item and sample_index.", _key("sampleIndex"))
    table.add("seed", "The random seed the response was generated with. Seeds can be "
              "very large; read this column as text to keep every digit.", _key("seed"))
    if any(_is_turn(record) for record in records):
        table.add("replicate", "Which play-through of the conversation the turn "
                  "belongs to, counting from 0.", _key("replicateIndex"))
        table.add("turn_title", "The title of the turn in the panel's script.",
                  _key("turnTitle"))
        table.add("speaker_seat", "The identifier of the seat that spoke. Two seats "
                  "can share a display name; the seat identifier is unique.",
                  _key("speakerAgentID"))
        table.add("speaker", "The display name of the speaker.", _key("speakerName"))
    table.add("prompt", "The text the model was given for this response.", _key("prompt"))
    table.add("response", "The text the model generated, as recorded.", _key("output"))
    for name, column, meaning in (
            ("finishReason", "finish_reason", "Why generation ended, as the engine "
             "recorded it. A value that names a length or token limit means the "
             "response was cut off rather than finished."),
            ("wordCount", "word_count", "The number of words in the response, as "
             "counted by the engine."),
            ("distinct2", "distinct_2", "The share of two-word sequences in the "
             "response that are distinct (1 means no repetition), as computed by "
             "the engine.")):
        if name in fields:
            table.add(column, meaning, _key(name))
    if "parsedChoice" in fields:
        table.add("parsed_choice", "The answer option the engine read from the "
                  "response.", _key("parsedChoice"))
        table.add("parsed_choice_failed", "1 when the engine looked for an answer "
                  "option and found none, 0 when it found one, and empty when the "
                  "item has no answer options.", _parse_failed("parsedChoice"))
    if "parsedMonths" in fields:
        table.add("parsed_number", "The number the study's parser read from the "
                  "response (stored by the engine as parsedMonths; the unit is the "
                  "one the study's parser declares).", _key("parsedMonths"))
        table.add("parsed_number_failed", "1 when the parser found no number in the "
                  "response, 0 when it found one, and empty when no parser applies.",
                  _parse_failed("parsedMonths"))
    _item_columns(table, records, fields)
    for concept in _dynamic_keys(records, "markerDensity"):
        table.add("marker_density_" + _column_slug(concept),
                  f"The share of the response's words that are marker words for the "
                  f"concept '{concept}', as computed by the engine.",
                  _nested("markerDensity", concept))
    for concept in _dynamic_keys(records, "readerScores"):
        table.add("reader_score_" + _column_slug(concept),
                  f"The reader's score of the response for the concept '{concept}', "
                  "as computed by the engine.", _nested("readerScores", concept))
    if "endpoint" in fields:
        table.add("endpoint_name", "The name of the outcome this turn was declared "
                  "to produce.", _nested("endpoint", "name"))
        table.add("endpoint_value", "The value the engine read for that outcome from "
                  "the turn's text.", _nested("endpoint", "value"))
        table.add("endpoint_unparsed", "1 when the engine could not read the declared "
                  "outcome from the turn's text, 0 when it could, and empty when "
                  "the turn declares no outcome.", _endpoint_unparsed)
    if "voiceLint" in fields:
        table.add("speaks_for_others", "1 when a line of the turn is written as if "
                  "another participant were speaking, as flagged by the engine.",
                  _nested("voiceLint", "speaksForOthers"))
        table.add("third_person_self", "How many times the speaker refers to itself "
                  "by name, as counted by the engine.",
                  _nested("voiceLint", "thirdPersonSelf"))
    if "modelID" in fields:
        table.add("model", "The model that generated the response.", _key("modelID"))
    if "modelRevision" in fields:
        table.add("model_revision", "The exact revision of that model.",
                  _key("modelRevision"))
    table.fill(records)
    left = sorted(fields - set(_RESPONSE_FIELDS))
    if left:
        table.notes.append(
            "Each response record in the run's generations.jsonl also holds these "
            "fields, which are not table columns: " + ", ".join(left) + ".")
    return table


def _readouts_table(study, run_name, loaded):
    records, fields = loaded["readouts"], loaded["readoutFields"]
    table = _Table("choice-readouts.csv", "Answer-option readouts",
                   "one reading of the answer options for a condition and item")
    table.add("study", "The study the reading was taken under.",
              lambda r: r.get("experiment") or study)
    table.add("run", "The run directory the reading comes from.", lambda r: run_name)
    table.add("condition", "The condition (arm) the reading was taken under.",
              _key("condition"))
    table.add("item", "The identifier of the task item.", _key("promptID"))
    table.add("item_index", "The item's position in the task file, counting from 0.",
              _key("promptIndex"))
    table.add("instrument", "The instrument that took the reading. It reads how "
              "likely the model is to give each answer option, without sampling a "
              "response.", _key("instrument"))
    table.add("target", "The answer option the item declares as its target, when it "
              "declares one.", _key("target"))
    table.add("selected", "The answer option the model rated most likely.",
              _key("selected"))
    table.add("margin", "How far the most likely option led the next one, as "
              "recorded by the engine.", _key("margin"))
    table.add("target_probability", "The probability the model gave the target "
              "option, copied from the record. Empty when the item declares no "
              "target.", lambda r: _for_target(r, "choiceProbability"))
    table.add("target_log_odds", "The log-odds of the target option, copied from the "
              "record. Empty when the item declares no target.",
              lambda r: _for_target(r, "logOdds"))
    if "ordinalPosition" in fields:
        table.add("ordinal_position", "The position on the item's rating scale, "
                  "counting from 1, as computed by the engine.", _key("ordinalPosition"))
    _item_columns(table, records, fields - {"target"})
    table.fill(records)
    table.notes.append(
        "The probability and log-odds of every answer option, not only the target, "
        "stay in the run's generations.jsonl.")
    return table


def _for_target(record, field):
    target, values = record.get("target"), record.get(field)
    if target is None or not isinstance(values, dict):
        return None
    return values.get(target)


# --- transcripts ---------------------------------------------------------------


def _file_slug(text):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("._") or "condition"


def _conversations(study, run_name, responses):
    """Turn records grouped into conversations: one per condition and
    replicate, baseline first, turns in script order."""
    grouped, order = {}, []
    for position, record in enumerate(responses):
        if not _is_turn(record):
            continue
        key = (str(record.get("condition", "")), record.get("replicateIndex") or 0)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((position, record))
    conditions = []
    for condition, _ in order:
        if condition not in conditions:
            conditions.append(condition)
    conditions.sort(key=lambda name: name != "baseline")
    conversations, taken = [], set()
    for condition in conditions:
        for replicate in sorted(r for c, r in order if c == condition):
            turns = sorted(grouped[(condition, replicate)], key=lambda entry: (
                entry[1].get("promptIndex") if isinstance(entry[1].get("promptIndex"), int)
                else entry[0], entry[0]))
            base = f"{_file_slug(condition)}-replicate-{replicate}"
            name, counter = base, 2
            while name in taken:
                name = f"{base}-{counter}"
                counter += 1
            taken.add(name)
            numbered = []
            for position, (_, record) in enumerate(turns):
                index = record.get("promptIndex")
                numbered.append((index + 1 if isinstance(index, int) else position + 1,
                                 record))
            conversations.append({"name": name, "condition": condition,
                                  "replicate": replicate, "turns": numbered})
    return conversations


def _transcript_text(study, run_name, conversation):
    lines = [f"Study: {study}", f"Run: {run_name}",
             f"Condition: {conversation['condition']}",
             f"Replicate: {conversation['replicate']}",
             f"Turns: {len(conversation['turns'])}", "",
             "This transcript was rebuilt from the turns the run recorded. The "
             "text of each turn is exactly as recorded.", ""]
    for number, record in conversation["turns"]:
        title = record.get("turnTitle") or record.get("promptID") or ""
        speaker = record.get("speakerName") or "speaker not recorded"
        seat = record.get("speakerAgentID")
        lines.append("-" * 60)
        lines.append(f"Turn {number}" + (f": {title}" if title else ""))
        lines.append(f"Speaker: {speaker}" + (f" (seat {seat})" if seat else ""))
        lines.append("")
        lines.append(str(record.get("output") or ""))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _turns_table(study, run_name, conversations):
    table = _Table("transcripts/turns.csv", "Conversation turns",
                   "one turn of one conversation")
    rows = [dict(record, _conversation=conversation["name"], _turn=number,
                 _replicate=conversation["replicate"])
            for conversation in conversations
            for number, record in conversation["turns"]]
    table.add("study", "The study the conversation was generated under.",
              lambda r: r.get("experiment") or study)
    table.add("run", "The run directory the conversation comes from.",
              lambda r: run_name)
    table.add("conversation", "The conversation the turn belongs to: one "
              "play-through of the panel under one condition. It is also the name "
              "of the conversation's text file in this folder.", _key("_conversation"))
    table.add("condition", "The condition (arm) the conversation was generated "
              "under.", _key("condition"))
    table.add("replicate", "Which play-through of that condition this is, counting "
              "from 0.", _key("_replicate"))
    table.add("turn", "The turn's place in the conversation, counting from 1.",
              _key("_turn"))
    table.add("turn_id", "The identifier of the turn in the panel's script.",
              _key("promptID"))
    table.add("turn_title", "The title of the turn in the panel's script.",
              _key("turnTitle"))
    table.add("seat", "The identifier of the seat that spoke. Empty when the run "
              "did not record it.", _key("speakerAgentID"))
    table.add("speaker", "The display name of the speaker.", _key("speakerName"))
    table.add("text", "What the speaker said, as recorded.", _key("output"))
    table.add("word_count", "The number of words in the turn, as counted by the "
              "engine.", _key("wordCount"))
    table.fill(rows)
    return table


# --- effect sizes --------------------------------------------------------------

#: Column names each engine writes, mapped to one set. The Mac engine writes
#: ``metric`` and ``meanDiff``; the Python engine writes ``endpoint`` and
#: ``deltaMean`` and adds ``modality``.
_EFFECT_ALIASES = {
    "outcome": ("endpoint", "metric"),
    "condition": ("condition",),
    "estimate": ("deltaMean", "meanDiff"),
    "ci_lower": ("ciLower",),
    "ci_upper": ("ciUpper",),
    "n_pairs": ("n",),
    "test_statistic": ("wilcoxonW",),
    "p_value": ("wilcoxonP",),
    "adjusted_p_value": ("adjustedP",),
    "correction": ("correction",),
    "modality": ("modality",),
    "stratify_by": ("stratifyBy",),
    "stratum": ("stratum",),
    "unit": ("unit",),
    "estimand": ("estimand",),
    "inference": ("inference",),
}
_EFFECT_NUMBERS = ("estimate", "ci_lower", "ci_upper", "n_pairs",
                   "test_statistic", "p_value", "adjusted_p_value")
WILCOXON = "wilcoxon_signed_rank"


def _read_effects(text):
    """The engine's effect table as normalized rows, plus what its header
    could and could not supply."""
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    source = {name: next((alias for alias in aliases if alias in header), None)
              for name, aliases in _EFFECT_ALIASES.items()}
    rows = []
    for raw in reader:
        row = {name: (raw.get(column) or "").strip() if column else ""
               for name, column in source.items()}
        for name in _EFFECT_NUMBERS:
            row[name] = _stored_number(row[name])
        rows.append(row)
    return rows, source


def _effects_tables(study, run_name, analysis_label, rows, source, stamped_unit):
    has_test = source["test_statistic"] is not None or source["p_value"] is not None

    def unit(row):
        return row["unit"] or stamped_unit or "item"

    def unit_source(row):
        return "recorded" if (row["unit"] or stamped_unit) else "engine_default"

    def common(table):
        table.add("study", "The study the estimate belongs to.", lambda r: study)
        table.add("run", "The run directory the estimate was computed from.",
                  lambda r: run_name)
        table.add("analysis", "The directory, beside the run under runs/, that holds "
                  "the engine's analysis this row was copied from.",
                  lambda r: analysis_label)
        table.add("outcome", "The outcome that was compared, under the name the "
                  "engine uses. The methods summary says how each was measured.",
                  _key("outcome"))
        table.add("condition", "The condition that was compared with the baseline.",
                  _key("condition"))

    def numbers(table):
        table.add("estimate", "The mean of the paired differences, condition minus "
                  "baseline, as computed by the engine.", _key("estimate"))
        table.add("ci_lower", "The lower end of the engine's bootstrap confidence "
                  "interval for the estimate.", _key("ci_lower"))
        table.add("ci_upper", "The upper end of that interval.", _key("ci_upper"))
        table.add("n_pairs", "How many paired differences the estimate rests on. "
                  "Each pair is one unit of analysis measured under the condition "
                  "and under the baseline.", _key("n_pairs"))
        table.add("unit_of_analysis", "What one paired difference is: an item (its "
                  "samples averaged), a transcript (one play-through of a "
                  "multi-agent conversation), or a sample.", unit)
        table.add("unit_of_analysis_source", "Where the unit comes from: recorded "
                  "when the analysis stamped it, and engine_default when it did "
                  "not, in which case the engines' documented default, the item, "
                  "is shown.", unit_source)
        table.add("test", "The significance test the engine ran on the paired "
                  f"differences: {WILCOXON} is the Wilcoxon signed-rank test. Empty "
                  "when the analysis stored no test.",
                  lambda r: WILCOXON if has_test else None)
        table.add("test_statistic", "The test statistic (W for the Wilcoxon "
                  "signed-rank test). Empty when the test is undefined, for example "
                  "when every difference is zero.", _key("test_statistic"))
        table.add("p_value", "The test's p-value before any correction for multiple "
                  "comparisons.", _key("p_value"))
        table.add("adjusted_p_value", "The p-value after the correction named in "
                  "the next column. Empty when the engine did not correct this "
                  "row.", _key("adjusted_p_value"))
        table.add("correction", "The correction for multiple comparisons: bh is "
                  "Benjamini-Hochberg (false discovery rate), and holm is Holm. "
                  "Outcomes are corrected separately, across conditions.",
                  _key("correction"))

    pooled = _Table("effects.csv", "Effects",
                    "one outcome compared between one condition and the baseline, "
                    "over all items")
    common(pooled)
    numbers(pooled)
    pooled.add("modality", "The kind of intervention the condition applies, as "
               "recorded by the engine (for example injection, adapter, or "
               "systemPrompt). Empty when the analysis did not record it.",
               _key("modality"))
    pooled.fill([row for row in rows if row["stratify_by"] in ("", "pooled")])

    strata = _Table("effects-by-stratum.csv", "Effects within subgroups",
                    "one outcome compared between one condition and the baseline, "
                    "within one subgroup of the items")
    common(strata)
    strata.add("stratify_by", "What defines the subgroups: promptID for single "
               "items, or the name of a design factor.", _key("stratify_by"))
    strata.add("stratum", "The subgroup this row covers.", _key("stratum"))
    numbers(strata)
    strata.add("estimand", "What the row estimates: itemLevel is an effect across "
               "the items of the subgroup; withinItemSamples describes the samples "
               "of a single item and supports no claim about other items.",
               _key("estimand"))
    strata.add("inference", "What the row's p-value supports: corrected when the "
               "row belongs to a corrected family of tests, and diagnostic when "
               "the engine deliberately left it out, in which case the p-value is "
               "a description and not a test.", _key("inference"))
    strata.fill([row for row in rows if row["stratify_by"] not in ("", "pooled")])
    return pooled, strata


# --- judgments and codings -----------------------------------------------------

_OUTCOME_LABELS = {"variant": "condition", "condition": "condition",
                   "baseline": "baseline", "tie": "tie"}


def _read_rows(path, sources, role):
    rows, unreadable = [], 0
    for line in sources.lines(path, role):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            unreadable += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            unreadable += 1
    return rows, unreadable


def _judge_details(report):
    """Per-judge facts from an evaluation report, by judge name. The Python
    engine's paired report keeps them in ``judges``; coding reports on both
    engines keep them in ``judgeDetails``."""
    details = {}
    for key in ("judgeDetails", "judges"):
        for entry in report.get(key) or []:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                details.setdefault(entry["name"], {}).update(entry)
    return details


def _judge_columns(table, details):
    def from_report(row, field):
        entry = details.get(row.get("judge")) or {}
        return entry.get(field)

    table.add("judge", "The name of the judge in the study's judge panel. It is a "
              "label, not a model.", _key("judge"))
    table.add("judge_kind", "The kind of judge: local (a model run by the engine), "
              "claude, or openrouter.",
              lambda r: r.get("judgeKind") or from_report(r, "kind"))
    table.add("judge_model", "The model that judged.",
              lambda r: r.get("judgeModel") or from_report(r, "requestedModel"))
    table.add("judge_provider", "The provider that served an openrouter judge.",
              lambda r: r.get("judgeProvider") or from_report(r, "provider"))
    table.add("judge_revision", "The exact revision of a local judge model.",
              lambda r: r.get("judgeRevision") or from_report(r, "revision"))


def _judgments_table(study, run_name, evaluation_label, rows, report):
    details = _judge_details(report)
    table = _Table("judgments.csv", "Paired judgments",
                   "one judge's comparison of a condition's response with the "
                   "baseline's response to the same item")

    def verdict(row, field):
        judgment = row.get("judgment")
        return judgment.get(field) if isinstance(judgment, dict) else None

    def outcome(row):
        return _OUTCOME_LABELS.get(row.get("outcome") or row.get("conditionResult"))

    table.add("study", "The study the judgment belongs to.",
              lambda r: r.get("experiment") or study)
    table.add("run", "The run directory whose responses were judged.",
              lambda r: run_name)
    table.add("evaluation", "The directory, beside the run under runs/, that holds "
              "the evaluation this row was copied from.", lambda r: evaluation_label)
    _judge_columns(table, details)
    table.add("condition", "The condition whose response was compared with the "
              "baseline's.", _key("condition"))
    table.add("item", "The identifier of the task item.", _key("promptID"))
    table.add("sample_index", "Which sample of the item was judged, counting from "
              "0. With item and condition it links the row to responses.csv.",
              _key("sampleIndex"))
    table.add("baseline_seed", "The seed of the baseline response in the pair.",
              _key("baselineSeed"))
    table.add("condition_seed", "The seed of the condition's response in the pair.",
              _key("variantSeed"))
    table.add("baseline_shown_as", "Whether the judge saw the baseline response as "
              "A or as B. The order is varied so the judge cannot tell which is "
              "which.", _key("baselineWas"))
    table.add("judge_choice", "What the judge answered: A, B, or tie.",
              lambda r: verdict(r, "winner"))
    table.add("outcome", "Which response the judge preferred: condition, baseline, "
              "or tie. Empty when the judge gave no verdict.", outcome)
    table.add("confidence", "The judge's stated confidence, from 0 to 1.",
              lambda r: r.get("confidence") if r.get("confidence") is not None
              else verdict(r, "confidence"))
    table.add("reason", "The judge's brief reason, as recorded.",
              lambda r: verdict(r, "brief_reason"))
    table.add("noncompliant", "1 when the judge answered without giving a usable "
              "verdict. Such rows have no outcome, and the engine leaves them out "
              "of every count and agreement statistic.",
              lambda r: bool(r.get("noncompliant")))
    table.add("noncompliance_reason", "Why the judge's answer could not be used.",
              _key("noncomplianceReason"))
    table.add("verdict_salvaged", "1 when the judge's answer was cut off and only "
              "the choice could be read.",
              lambda r: bool(r.get("verdictSalvaged")
                             or verdict(r, "reasoningTruncated")))
    for side, field in (("a", "a_scores"), ("b", "b_scores")):
        names = []
        for row in rows:
            scores = verdict(row, field)
            for name in scores if isinstance(scores, dict) else ():
                if name not in names:
                    names.append(name)
        for name in names:
            table.add(f"score_{side}_{_column_slug(name)}",
                      f"The judge's score for response {side.upper()} on the rubric "
                      f"criterion '{name}'.",
                      lambda r, field=field, name=name: (verdict(r, field) or {}).get(name))
    names = []
    for row in rows:
        structured = verdict(row, "structured_fields")
        for name in structured if isinstance(structured, dict) else ():
            if name not in names:
                names.append(name)
    for name in names:
        table.add(f"field_{_column_slug(name)}",
                  f"The judge's answer to the rubric's structured question '{name}'.",
                  lambda r, name=name: (verdict(r, "structured_fields") or {}).get(name))
    table.fill(rows)
    return table


def _codings_table(study, run_name, evaluation_label, rows, report):
    details = _judge_details(report)
    table = _Table("codings.csv", "Response codings",
                   "one judge's coding of one response")
    table.add("study", "The study the coding belongs to.",
              lambda r: r.get("experiment") or study)
    table.add("run", "The run directory whose responses were coded.",
              lambda r: run_name)
    table.add("evaluation", "The directory, beside the run under runs/, that holds "
              "the evaluation this row was copied from.", lambda r: evaluation_label)
    _judge_columns(table, details)
    table.add("condition", "The condition the coded response was generated under. "
              "The judge did not see it.", _key("condition"))
    table.add("item", "The identifier of the task item.", _key("promptID"))
    table.add("sample_index", "Which sample of the item was coded, counting from 0. "
              "With item and condition it links the row to responses.csv.",
              _key("sampleIndex"))
    table.add("seed", "The seed of the coded response.", _key("seed"))
    table.add("word_count", "The number of words in the coded response, as counted "
              "by the engine.", _key("wordCount"))
    declared = [field.get("name") for field in report.get("fields") or []
                if isinstance(field, dict) and isinstance(field.get("name"), str)]
    kinds = {field.get("name"): field for field in report.get("fields") or []
             if isinstance(field, dict)}
    for row in rows:
        for name in row.get("codes") if isinstance(row.get("codes"), dict) else ():
            if name not in declared:
                declared.append(name)
    for name in declared:
        field = kinds.get(name) or {}
        detail = f" ({field['type']})" if isinstance(field.get("type"), str) else ""
        values = field.get("values")
        allowed = (" Allowed values: " + ", ".join(str(v) for v in values) + "."
                   if isinstance(values, list) and values else "")
        table.add(f"code_{_column_slug(name)}",
                  f"The judge's code for the rubric field '{name}'{detail}. True and "
                  f"false are written as 1 and 0.{allowed}",
                  _nested("codes", name))
    table.add("reason", "The judge's brief reason, as recorded.", _key("briefReason"))
    table.add("noncompliant", "1 when the judge answered without giving usable "
              "codes. Such rows have no codes, and the engine leaves them out of "
              "every count and agreement statistic.",
              lambda r: bool(r.get("noncompliant")))
    table.add("noncompliance_reason", "Why the judge's answer could not be used.",
              _key("noncomplianceReason"))
    table.fill(rows)
    if any(isinstance(row.get("undeclaredCodes"), dict) for row in rows):
        table.notes.append(
            "Fields a judge added that the rubric does not declare are not "
            "exported. They stay in the evaluation's codings.jsonl.")
    return table


# --- the methods summary -------------------------------------------------------

_ENGINES = {
    "python-hf-transformers": "the Python engine (Hugging Face Transformers)",
    "swift-mlx": "the Mac engine (MLX)",
}
_GATES = {
    "revision": "the check that the model revision is pinned",
    "validateEvidence": "the check that the steering directions were validated",
    "batteryEvidence": "the check that capability battery evidence exists",
    "judgeValidity": "the check on the judge panel",
    "variantValidity": "the check on the study's agents",
    "gitClean": "the check that the pinned inputs are committed",
    "measurementPins": "the check on the measurement settings",
}
_OUTCOMES = {
    "wordCount": "the number of words in the response",
    "distinct2": "the share of two-word sequences in the response that are distinct, a measure of repetition",
    "choiceLogOdds": "the log-odds the model gave the item's declared target option, read from the answer options without sampling",
    "choiceRate": "whether the sampled response chose the item's declared target option",
    "ordinalPosition": "the position on the item's rating scale, read from the answer options without sampling",
    "meanMonths": "the mean of the numbers the study's parser read from an item's responses",
    "monthsSpread": "the spread of the numbers the study's parser read from an item's responses",
    "parsedValueMean": "the mean of the numbers the study's parser read from an item's responses",
    "parsedValueSpread": "the spread of the numbers the study's parser read from an item's responses",
}


def _describe_outcome(name):
    if name in _OUTCOMES:
        return _OUTCOMES[name]
    if name.startswith("readerScore:"):
        return f"the reader's score of the response for the concept '{name.split(':', 1)[1]}'"
    if name.startswith("rs_"):
        return f"the reasoning-style feature '{name[3:]}', scored from the response text with the study's pinned taxonomy"
    if name.endswith("MarkerDensity"):
        return f"the share of the response's words that are marker words for the concept '{name[:-len('MarkerDensity')]}'"
    return "recorded by the engine under this name; this version has no description for it"


def _and(items):
    items = [str(item) for item in items]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _number(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _code(value):
    return f"`{value}`"


def _strength(value, norm_units):
    if norm_units is True:
        return f"{_number(value)} (in units of the typical activation size at that layer)"
    if norm_units is False:
        return f"{_number(value)} (in raw activation units)"
    return f"{_number(value)} (units not recorded)"


def _injection(slot, norm_units, layer_key="layer"):
    verb = "removes" if slot.get("mode") == "ablate" else "adds"
    layers = slot.get("layers")
    where = (f"layers {_and(layers)}" if isinstance(layers, list) and layers
             else f"layer {slot.get(layer_key)}")
    return (f"{verb} the direction for the concept '{slot.get('concept')}' at "
            f"{where}, with strength {_strength(slot.get('alpha'), norm_units)}")


def _describe_condition(name, snapshot):
    """What one condition applied, from the run's settings snapshot."""
    for condition in snapshot.get("conditions") or []:
        if not isinstance(condition, dict) or condition.get("name") != name:
            continue
        slots = [s for s in condition.get("slots") or [] if isinstance(s, dict)]
        if not slots:
            return "no intervention (the comparison arm)."
        parts = [_injection(slot, condition.get("alphaInNormUnits")) for slot in slots]
        text = _and(parts)
        width = condition.get("bandWidth")
        if isinstance(width, int) and width > 1:
            text += f", applied across {width} layers"
        control = condition.get("controlType")
        if control:
            text += f". This is a control arm of the type {_code(control)}"
        return "steering: " + text + "."
    for variant in snapshot.get("variantConditions") or []:
        if not isinstance(variant, dict) or variant.get("name") != name:
            continue
        artifact = variant.get("artifact") if isinstance(variant.get("artifact"), dict) else {}
        parts = [f"the saved agent {_code(artifact.get('name') or variant.get('name'))}"]
        if variant.get("artifactPath"):
            parts.append(f"file {_code(variant['artifactPath'])}")
        if variant.get("artifactHash"):
            parts.append(f"SHA-256 {_code(variant['artifactHash'])}")
        text = "an agent arm, " + ", ".join(parts) + "."
        carried = []
        injections = [i for i in artifact.get("injections") or [] if isinstance(i, dict)]
        if injections:
            carried.append("steering that " + _and(
                _injection(i, artifact.get("alphaInNormUnits")) for i in injections))
        adapters = [a.get("name") or a.get("adapterDirectory")
                    for a in artifact.get("adapters") or [] if isinstance(a, dict)]
        if adapters:
            carried.append("the fine-tuned adapter" + ("s " if len(adapters) > 1 else " ")
                           + _and(_code(a) for a in adapters))
        if artifact.get("systemPrompt"):
            carried.append("its own system prompt")
        policies = artifact.get("interventionPolicies")
        if isinstance(policies, list) and policies:
            carried.append(f"{len(policies)} intervention polic"
                           + ("ies" if len(policies) != 1 else "y"))
        if isinstance(variant.get("fromPromotion"), dict):
            carried.append("settings chosen by a sweep and promoted before the run")
        text += (" The agent carries " + _and(carried) + "." if carried
                 else " The settings snapshot records no intervention for this agent.")
        return text
    latent = snapshot.get("saeLatentConditions")
    entries = latent if isinstance(latent, list) else []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name") == name:
            settings = ", ".join(f"{key} {_code(json.dumps(value, ensure_ascii=False))}"
                                 for key, value in sorted(entry.items()) if key != "name")
            return ("a latent arm, which edits one feature of a sparse autoencoder. "
                    "Declared settings: " + (settings or "none recorded") + ".")
    if name == "baseline":
        return "no intervention (the comparison arm)."
    return ("not described in the run's settings snapshot, so what it applied is "
            "not available here.")


def _methods(context):
    """methods.md, built only from stored facts."""
    study, run_name = context["study"], context["runName"]
    snapshot, config, report = context["snapshot"], context["config"], context["report"]
    loaded, observed = context["loaded"], context["loaded"]["observed"]
    missing = context["missing"]
    lines = []

    def say(text=""):
        lines.append(text)

    def unavailable(what, why):
        missing.append({"what": what, "why": why})
        return f"not available ({why})"

    say(f"# Methods summary: {study}")
    say()
    say(f"Written by SteerLab {context['version']} on {context['date']} from one "
        "completed run. Every statement below is taken from files that the run, "
        "its analysis, and its evaluation stored. Nothing was recalculated. Where "
        "a fact was not stored, this summary says it is not available. Adapt the "
        "wording for your own paper, and check it against your own records.")
    say()

    # -- what a reader must not miss, said first and again in its own section
    notices = []
    if snapshot.get("freezeForced"):
        notices.append("This study was frozen with force: some freeze checks were "
                       "skipped.")
    elif snapshot.get("status") not in (None, "frozen", "complete"):
        notices.append("This study was not frozen when the run was made.")
    exempt = [entry.get("condition") for entry in snapshot.get("capabilityBatteryNotApplied") or []
              if isinstance(entry, dict)]
    if exempt:
        notices.append("The capability battery was not applied to "
                       + _and(exempt) + ".")
    if notices:
        say("**Read this first.** " + " ".join(notices) + " The section \"Freeze "
            "status and identifying hashes\" below gives the details.")
        say()

    # -- sources
    say("## What this export was built from")
    say()
    say(f"- Run: {_code('runs/' + run_name)}")
    owner = context["runStudy"]
    if owner and owner != study:
        say(f"- The run was generated under the study {_code(owner)}. It is "
            f"exported here with the analysis and evaluation made under {_code(study)}.")
    if context["analysisLabel"]:
        say(f"- Analysis: {_code(context['analysisLabel'])}")
    else:
        say("- Analysis: none found for this run, so effects.csv was not written.")
    if context["pairedName"]:
        say(f"- Paired judging: {_code('runs/' + context['pairedName'])}")
    if context["codingName"]:
        say(f"- Response coding: {_code('runs/' + context['codingName'])}")
    if not context["pairedName"] and not context["codingName"]:
        say("- Evaluation by judges: none found for this run.")
    say("- manifest.json in this folder lists every file that was read, with its "
        "SHA-256 hash.")
    say()

    # -- model and software
    say("## Model and software")
    say()
    models = observed["modelID"] or ([snapshot.get("modelID")] if snapshot.get("modelID") else [])
    revisions = observed["modelRevision"] or (
        [snapshot.get("modelRevision")] if snapshot.get("modelRevision") else [])
    if models:
        say(f"- Model: {_and(_code(m) for m in models)}")
    else:
        say("- Model: " + unavailable("the model", "the run does not name it"))
    if revisions:
        say(f"- Model revision: {_and(_code(r) for r in revisions)}")
    else:
        say("- Model revision: " + unavailable(
            "the model revision", "the run did not record one"))
    substrate = config.get("substrate") or (observed["engine"][0] if observed["engine"] else None)
    if substrate:
        say(f"- Engine: {_ENGINES.get(substrate, 'an engine this version does not know')} "
            f"(recorded as {_code(substrate)})")
    else:
        say("- Engine: " + unavailable("the engine", "the run has no config.json stamp"))
    backend = []
    if observed["device"]:
        backend.append("device " + _and(_code(d) for d in observed["device"]))
    precision = config.get("dtype") or (observed["dtype"][0] if observed["dtype"] else None)
    if precision:
        backend.append(f"numeric precision {_code(precision)}")
    if config.get("platform"):
        backend.append(f"platform {_code(config['platform'])}")
    say("- Back end: " + (", ".join(backend) if backend else unavailable(
        "the back end", "the run recorded no device, precision, or platform")))
    if config.get("appVersion"):
        say(f"- Software that made the run: {_code(config['appVersion'])}")
    else:
        say("- Software that made the run: " + unavailable(
            "the software version", "the run has no config.json stamp"))
    environment = config.get("pythonEnvironment")
    if isinstance(environment, dict):
        packages = environment.get("packages") if isinstance(environment.get("packages"), dict) else {}
        named = ([f"Python {environment['python']}"] if isinstance(environment.get("python"), str) else [])
        named += [f"{name} {packages[name]}" for name in ("torch", "transformers", "numpy", "scipy")
                  if isinstance(packages.get(name), str)]
        if named:
            say("- Libraries the run used: " + ", ".join(named))
    say()

    # -- design
    say("## Design")
    say()
    kind = snapshot.get("studyType") or snapshot.get("studyKind")
    if kind:
        say(f"- Kind of study: {_code(kind)}")
    if snapshot.get("experimentDescription"):
        say(f"- Description, as the researcher wrote it: {_one_line(str(snapshot['experimentDescription']))}")
    if snapshot.get("taskDescription"):
        say(f"- Task, as the researcher wrote it: {_one_line(str(snapshot['taskDescription']))}")
    responses = loaded["responses"]
    items = []
    per_condition, per_cell = {}, {}
    for record in responses:
        item = record.get("promptID")
        if item not in items:
            items.append(item)
        condition = str(record.get("condition", ""))
        per_condition[condition] = per_condition.get(condition, 0) + 1
        per_cell[(condition, item)] = per_cell.get((condition, item), 0) + 1
    prompts_file = snapshot.get("taskPromptsFile") or (
        observed["taskPromptsFile"][0] if observed["taskPromptsFile"] else None)
    prompts_hash = snapshot.get("taskPromptsHash") or (
        observed["taskPromptsHash"][0] if observed["taskPromptsHash"] else None)
    panel = any(_is_turn(record) for record in responses)
    if panel:
        say("- This is a multi-agent study: several seats speak in turn, and each "
            "response is one turn of a conversation.")
        if snapshot.get("multiAgentScenarioPath"):
            say(f"- Panel script: {_code(snapshot['multiAgentScenarioPath'])}"
                + (f", SHA-256 {_code(snapshot['multiAgentScenarioHash'])}"
                   if snapshot.get("multiAgentScenarioHash") else ""))
        say(f"- Turns in the script: {len(items)}. The text each speaker was given "
            "is in the prompt column of responses.csv.")
    elif prompts_file:
        say(f"- Task items: {len(items)}, from {_code(prompts_file)}"
            + (f", SHA-256 {_code(prompts_hash)}" if prompts_hash else "")
            + ". The text of every item is in the prompt column of responses.csv.")
    else:
        say(f"- Task items: {len(items)}. The task file is "
            + unavailable("the task file's name", "the run's settings do not name it")
            + "; the text of every item is in the prompt column of responses.csv.")
    counts = ", ".join(f"{name} {count}" for name, count in per_condition.items())
    say(f"- Responses: {len(responses)} in total" + (f" ({counts})" if counts else "") + ".")
    if panel:
        say(f"- Conversations: {context['conversations']}, one for each condition "
            "and play-through (transcripts/).")
    elif per_cell:
        low, high = min(per_cell.values()), max(per_cell.values())
        say("- Samples for each condition and item: "
            + (str(low) if low == high else f"between {low} and {high}") + ".")
    if loaded["readouts"]:
        say(f"- Readings of the answer options, taken without sampling: {len(loaded['readouts'])} "
            "(choice-readouts.csv).")
    for failure in loaded["failedConditions"]:
        say(f"- The condition {_code(failure.get('condition'))} produced no responses. "
            f"The run recorded this error: {_one_line(str(failure.get('error')))}")
    sampling = []
    temperatures = observed["temperature"] or (
        [snapshot["temperature"]] if "temperature" in snapshot else [])
    if temperatures:
        sampling.append("temperature " + _and(_number(t) for t in temperatures))
    for key, label in (("topP", "top-p"), ("topK", "top-k")):
        if observed[key]:
            sampling.append(f"{label} " + _and(_number(v) for v in observed[key]))
    if "maxTokens" in snapshot:
        sampling.append(f"at most {snapshot['maxTokens']} new tokens for each response")
    samples = config.get("samplesPerItem") or snapshot.get("samplesPerItem")
    if samples:
        sampling.append(f"{samples} sample{'s' if samples != 1 else ''} for each item")
    policy = config.get("seedPolicy") or snapshot.get("seedPolicy") or (
        observed["seedPolicy"][0] if observed["seedPolicy"] else None)
    if policy:
        sampling.append(f"seed policy {_code(policy)}")
    if snapshot.get("seeds"):
        sampling.append("declared seeds " + _and(snapshot["seeds"]))
    if snapshot.get("reasoningEffort"):
        sampling.append(f"reasoning effort {_code(snapshot['reasoningEffort'])}")
    if snapshot.get("reasoningMaxTokens"):
        sampling.append(f"at most {snapshot['reasoningMaxTokens']} reasoning tokens")
    mode = snapshot.get("promptMode") or (observed["promptMode"][0] if observed["promptMode"] else None)
    if mode:
        sampling.append(f"prompt mode {_code(mode)}")
    say("- Sampling settings: " + ("; ".join(sampling) if sampling else unavailable(
        "the sampling settings", "the run has no settings snapshot")) + ".")
    if snapshot.get("systemPrompt"):
        say(f"- System prompt for every condition: \"{_one_line(str(snapshot['systemPrompt']))}\"")
    elif snapshot:
        say("- System prompt: none declared for the study.")
    truncation = report.get("truncation") if isinstance(report.get("truncation"), dict) else None
    if truncation and truncation.get("classified") is not None:
        say(f"- Responses cut off at a token limit: {truncation.get('lengthStopped')} of "
            f"{truncation.get('classified')}, as the engine counted them.")
    say()

    # -- conditions
    say("## Conditions")
    say()
    declared = [c.get("name") for key in ("conditions", "variantConditions")
                for c in snapshot.get(key) or [] if isinstance(c, dict)]
    latent = snapshot.get("saeLatentConditions")
    declared += [c.get("name") for c in (latent if isinstance(latent, list) else [])
                 if isinstance(c, dict)]
    names = list(per_condition)
    names.sort(key=lambda name: name != "baseline")
    for name in names:
        say(f"- **{name}**: {_describe_condition(name, snapshot)}")
    for name in declared:
        if name and name not in per_condition and not any(
                f.get("condition") == name for f in loaded["failedConditions"]):
            say(f"- **{name}**: declared for the study, and the run holds no "
                "responses for it.")
    if not names:
        say("- " + unavailable("the conditions", "the run holds no responses"))
    battery = report.get("capabilityBatteryNotApplied") or snapshot.get("capabilityBatteryNotApplied")
    scored = {name: block["capabilityBattery"] for name, block in (report.get("conditions") or {}).items()
              if isinstance(block, dict) and isinstance(block.get("capabilityBattery"), dict)}
    if scored or battery:
        say()
        say("Capability battery (a set of unrelated questions, scored under each "
            "condition to check that the intervention did not damage other abilities):")
        say()
        for name, block in scored.items():
            say(f"- {name}: accuracy {_number(block.get('accuracy'))} on "
                f"{block.get('itemCount')} items, as the engine scored it.")
        for entry in battery if isinstance(battery, list) else []:
            if isinstance(entry, dict):
                say("- " + _battery_sentence(entry))
    say()

    # -- outcomes
    say("## Outcomes and how each was measured")
    say()
    primary = next((snapshot[key] for key in PRIMARY_OUTCOME_KEYS if snapshot.get(key)), None)
    if isinstance(primary, dict):
        primary = primary.get("name") or primary.get("outcome") or json.dumps(primary, sort_keys=True)
    if primary:
        say(f"- Primary outcome, declared by the researcher before the run: {_code(primary)}.")
    else:
        say("- Primary outcome: the study's settings declare none.")
    instruments = snapshot.get("outcomeInstruments")
    if instruments:
        say("- Instruments declared for the study: " + _and(_code(i) for i in instruments) + ".")
    if snapshot.get("numericParser"):
        say(f"- Numbers were read from responses with the parser {_code(snapshot['numericParser'])}"
            + (f" (parser registry SHA-256 {_code(snapshot['parserRegistryHash'])})"
               if snapshot.get("parserRegistryHash") else "") + ".")
    if snapshot.get("reasoningStyleTaxonomyPath"):
        say(f"- Reasoning-style features use the taxonomy {_code(snapshot['reasoningStyleTaxonomyPath'])}"
            + (f" (SHA-256 {_code(snapshot['reasoningStyleTaxonomyHash'])})"
               if snapshot.get("reasoningStyleTaxonomyHash") else "") + ".")
    outcomes = context["outcomes"]
    if outcomes:
        say("- Outcomes in effects.csv:")
        for name in outcomes:
            say(f"  - {_code(name)}: {_describe_outcome(name)}.")
    else:
        say("- Outcomes compared between conditions: " + NOT_AVAILABLE_NO_ANALYSIS)
    if context["pairedName"]:
        say("- Judged outcome: which response each judge preferred, the condition's "
            "or the baseline's (the outcome column of judgments.csv).")
    if context["codingName"]:
        coded = [field.get("name") for field in context["codingReport"].get("fields") or []
                 if isinstance(field, dict) and field.get("name")]
        say("- Coded outcomes: the rubric's fields, coded by the judges for each "
            "response (codings.csv)" + (": " + _and(_code(name) for name in coded) if coded else "") + ".")
    say("- codebook.md describes every column of every table, including the "
        "outcomes recorded for each response.")
    say()

    # -- judging
    say("## Judges and rubric")
    say()
    if not context["pairedName"] and not context["codingName"]:
        say("No evaluation by judges was found for this run, so this export has no "
            "judgments. If the study was meant to be judged, evaluate the run and "
            "export again.")
        if snapshot.get("judges"):
            say()
            say("The study's settings declare these judges: "
                + _and(_code(j.get("name")) for j in snapshot["judges"] if isinstance(j, dict)) + ".")
    for kind, name, rows_key, label in (
            ("paired", context["pairedName"], "judgmentRows", "judgments.csv"),
            ("coding", context["codingName"], "codingRows", "codings.csv")):
        if not name:
            continue
        evaluation = context[kind + "Report"]
        rows = context[rows_key]
        if kind == "paired":
            say("Paired judging: each judge saw a condition's response and the "
                "baseline's response to the same item, in varied order and without "
                "being told which was which, and chose the one that better met the "
                "rubric, or a tie.")
        else:
            say("Response coding: each judge coded one response at a time against "
                "the rubric's fields, without being told its condition.")
        say()
        details = _judge_details(evaluation)
        judge_names = [j["name"] if isinstance(j, dict) else j for j in evaluation.get("judges") or []]
        for judge in judge_names:
            entry = details.get(judge, {})
            row = next((r for r in rows if r.get("judge") == judge), {})
            facts = [f"kind {_code(entry.get('kind') or row.get('judgeKind'))}"
                     if (entry.get("kind") or row.get("judgeKind")) else None,
                     f"model {_code(entry.get('requestedModel') or row.get('judgeModel'))}"
                     if (entry.get("requestedModel") or row.get("judgeModel")) else None,
                     f"revision {_code(entry.get('revision') or row.get('judgeRevision'))}"
                     if (entry.get("revision") or row.get("judgeRevision")) else None,
                     f"provider {_code(entry.get('provider') or row.get('judgeProvider'))}"
                     if (entry.get("provider") or row.get("judgeProvider")) else None]
            facts = [fact for fact in facts if fact]
            say(f"- Judge {_code(judge)}: " + (", ".join(facts) if facts else "details not recorded") + ".")
        if not judge_names and evaluation.get("judgeModel"):
            say(f"- Judge model: {_code(evaluation['judgeModel'])}.")
        rubric = evaluation.get("judgeRubricFile") or evaluation.get("rubricFile")
        rubric_hash = evaluation.get("judgeRubricHash") or evaluation.get("rubricHash")
        if rubric:
            say(f"- Rubric: {_code(rubric)}" + (f", SHA-256 {_code(rubric_hash)}" if rubric_hash else "") + ".")
        else:
            say("- Rubric: " + unavailable("the rubric file", "the evaluation does not name one") + ".")
        noncompliant = sum(1 for row in rows if row.get("noncompliant"))
        say(f"- Rows in {label}: {len(rows)}, of which {noncompliant} "
            f"{'is' if noncompliant == 1 else 'are'} marked noncompliant (the judge "
            "answered without a usable result).")
        subsample = evaluation.get("sampling")
        if isinstance(subsample, dict):
            say(f"- The judges coded a seeded subsample, not every response: "
                f"{subsample.get('sampledRecords')} of {subsample.get('sourceRecords')} "
                f"responses, {subsample.get('samplePerCondition')} for each condition, "
                f"drawn with seed {_code(subsample.get('sampleSeed'))}.")
        for entry in evaluation.get("agreement") or evaluation.get("judgeAgreement") or []:
            if not isinstance(entry, dict):
                continue
            pair = entry.get("judges") or [entry.get("judgeA"), entry.get("judgeB")]
            say(f"- Agreement between {_and(_code(j) for j in pair)}, as the engine "
                f"computed it over {entry.get('n', entry.get('items'))} shared pairs: "
                f"proportion in agreement {_stored(entry.get('percentAgreement'))}, "
                f"Cohen's kappa {_stored(entry.get('kappa'))}.")
        agreement = evaluation.get("fieldAgreement")
        if isinstance(agreement, list) and agreement:
            say("- Agreement between judges on each rubric field, as the engine computed it:")
            for entry in agreement:
                if isinstance(entry, dict):
                    parts = [f"{label} {_stored(entry[key])}" for key, label in
                             (("percentAgreement", "proportion in agreement"),
                              ("kappa", "Cohen's kappa"),
                              ("meanAbsoluteDifference", "mean absolute difference"))
                             if entry.get(key) is not None]
                    say(f"  - {_code(entry.get('field'))}, {entry.get('judgeA')} and "
                        f"{entry.get('judgeB')}, over {entry.get('n')} shared responses: "
                        + (", ".join(parts) if parts else "no statistic recorded") + ".")
        elif evaluation.get("fieldAgreementAbsentReason"):
            say("- Agreement between judges: not available. "
                + _one_line(str(evaluation["fieldAgreementAbsentReason"])))
        for flag, text in (("epochUnverified", "The evaluation accepted a run that carries no stamp of the study's settings, so it could not confirm the run matches them."),
                           ("measurementDrift", None)):
            if evaluation.get(flag):
                say("- " + (text or "The study's judging settings changed after the run was "
                            f"generated. The evaluation recorded: {_one_line(str(evaluation[flag]))}"))
        say()
    if not context["pairedName"] and not context["codingName"]:
        say()

    # -- exclusions
    say("## Exclusions")
    say()
    stamps = [(label, stamp) for label, stamp in context["exclusions"] if isinstance(stamp, dict)]
    rules = snapshot.get("exclusionRules")
    if stamps:
        for label, stamp in stamps:
            say(f"{label} applied the study's declared exclusion rules and removed "
                f"{stamp.get('excludedRecords')} record{'s' if stamp.get('excludedRecords') != 1 else ''}.")
            say()
            for rule in stamp.get("rules") or []:
                if isinstance(rule, dict):
                    described = _one_line(str(rule.get("description") or "no description recorded"))
                    say(f"- Rule {_code(rule.get('rule'))}: {described}"
                        + ("" if described.endswith((".", "!", "?")) else "."))
            for condition, by_rule in (stamp.get("excludedByRule") or {}).items():
                if isinstance(by_rule, dict):
                    removed = ", ".join(f"{rule} {count}" for rule, count in by_rule.items())
                    kept = (stamp.get("survivingN") or {}).get(condition)
                    considered = (stamp.get("consideredN") or {}).get(condition)
                    say(f"- {condition}: {considered} records considered; removed by rule: "
                        f"{removed or 'none'}; {kept} kept.")
            if stamp.get("note"):
                say(f"- The engine's note: {_one_line(str(stamp['note']))}")
            say()
        say("Excluded records stay in the run and in responses.csv. The engine does "
            "not record which individual records it excluded, only the counts above.")
    elif rules:
        say("The study declares exclusion rules ("
            + _and(_code(r.get("rule")) for r in rules if isinstance(r, dict))
            + "), and how many records they removed is "
            + unavailable("the exclusion counts", "no analysis or evaluation of this run "
                          "recorded them") + ".")
    elif snapshot:
        say("The study declares no exclusion rules, so no records were excluded.")
    else:
        say("Whether the study declares exclusion rules is "
            + unavailable("the exclusion rules", "the run has no settings snapshot")
            + ". No analysis or evaluation of this run recorded an exclusion.")
    say()

    # -- statistics
    say("## Statistics")
    say()
    if context["effectsSource"] is None:
        say("No analysis of this run was found, so this export has no effect "
            "estimates. Analyze the run, then export again.")
    else:
        source = context["effectsSource"]
        say("The engine compared each condition with the baseline on each outcome. "
            "effects.csv copies its results without recalculating them.")
        say()
        say("- Estimate: the mean of the paired differences, condition minus "
            "baseline, where a pair is the same unit of analysis measured under "
            "both.")
        units = context["units"]
        if context["unitRecorded"]:
            say(f"- Unit of analysis: {_and(_code(u) for u in units)}, as the analysis recorded.")
        else:
            say("- Unit of analysis: the item, with an item's samples averaged within "
                "each condition. This is the engines' documented default; the "
                "analysis did not stamp the unit itself.")
        say("- Interval: a bootstrap confidence interval for the estimate. The "
            "number of bootstrap resamples, the random seed, and the confidence "
            "level are " + unavailable(
                "the bootstrap settings", "the analysis files do not record them; "
                "they are fixed by the software version named above") + ".")
        if source["p_value"] or source["test_statistic"]:
            say("- Test: the Wilcoxon signed-rank test on the paired differences.")
        else:
            say("- Test: " + unavailable("the significance test", "the analysis stored none") + ".")
        corrections = context["corrections"]
        if corrections:
            named = {"bh": "Benjamini-Hochberg (false discovery rate)", "holm": "Holm"}
            say("- Correction for multiple comparisons: "
                + _and(named.get(c, c) for c in corrections)
                + ", applied to each outcome separately, across conditions.")
        else:
            say("- Correction for multiple comparisons: none recorded.")
        if context["strataRows"]:
            say(f"- effects-by-stratum.csv holds {context['strataRows']} further rows, "
                "one for each subgroup of the items. Rows marked diagnostic describe "
                "the samples of one item and are not tests.")
        for flag, text in (("epochUnverified", "The analysis accepted a run that carries no stamp of the study's settings, so it could not confirm the run matches them."),):
            if context["analysisFlags"].get(flag):
                say("- " + text)
        if context["analysisFlags"].get("measurementDrift"):
            say("- The study's measurement settings changed after the run was "
                "generated. The analysis recorded: "
                + _one_line(str(context["analysisFlags"]["measurementDrift"])))
    say()

    # -- freeze and integrity
    say("## Freeze status and identifying hashes")
    say()
    status = snapshot.get("status")
    if snapshot.get("freezeForced"):
        skipped = snapshot.get("forcedGatesSkipped") or []
        say("- **This study was frozen with force.** Freezing fixes a study's "
            "settings before any behavior is measured, and it normally requires "
            "every check to pass. Here these checks were skipped: "
            + (_and(f"{_GATES.get(g, 'a check')} ({_code(g)})" for g in skipped)
               if skipped else "not available (the settings do not list them)")
            + ". SteerLab records a forced freeze permanently and treats the study "
            "as not citable. Say so wherever you report these results.")
    elif status in ("frozen", "complete"):
        say("- The study was frozen before this run, with every freeze check passed"
            + (f" (frozen at {snapshot['frozenAt']})" if snapshot.get("frozenAt") else "") + ".")
    elif status:
        say(f"- **The study was not frozen when this run was made** (its status was "
            f"{_code(status)}). Its settings were not fixed before behavior was measured.")
    else:
        say("- Freeze status: " + unavailable(
            "the freeze status", "the run's settings snapshot does not record it"
            if snapshot else "the run has no settings snapshot"))
    for entry in snapshot.get("capabilityBatteryNotApplied") or []:
        if isinstance(entry, dict):
            say("- **Capability battery exemption.** " + _battery_sentence(entry))
    stamped = context["experimentHash"]
    say("- Hash of the study's settings, as stamped on the run: "
        + (_code(stamped) if stamped else unavailable(
            "the run's settings hash", "the run carries no stamp")))
    for key, label in (("freezeHash", "Freeze hash"), ("gitCommit", "Workspace git commit at freeze"),
                       ("appVersion", "Software that froze the study")):
        if snapshot.get(key):
            say(f"- {label}: {_code(snapshot[key])}")
    if config.get("createdAt"):
        say(f"- Run started: {config['createdAt']}")
    say()

    # -- not available
    say("## What is not available")
    say()
    seen, unique = set(), []
    for entry in missing:
        if entry["what"] not in seen:
            seen.add(entry["what"])
            unique.append(entry)
    if unique:
        for entry in unique:
            say(f"- {entry['what']}: {entry['why']}.")
    else:
        say("Nothing this summary looks for was missing from the run.")
    say()
    return "\n".join(lines)


NOT_AVAILABLE_NO_ANALYSIS = "not available (no analysis of this run was found)."


def _stored(value):
    return "not recorded" if value is None else _number(value)


def _battery_sentence(entry):
    """The engines' own sentence for a battery exemption, so the export says
    exactly what freeze said."""
    condition, reason = str(entry.get("condition", "?")), str(entry.get("reason", ""))
    try:
        from ..experiment import freeze_policy
        return freeze_policy.battery_not_applied_sentence(condition, reason)
    except Exception:  # noqa: BLE001 - the summary must not fail on an import
        return (f"The capability battery was not applied to {condition} (recorded "
                f"reason: {reason}). This study has no capability control for that agent.")


# --- the codebook --------------------------------------------------------------


def _codebook(study, tables, transcripts):
    lines = [f"# Codebook: {study}", "",
             "Every column of every table in this folder, in a sentence each.", "",
             "## Reading the tables", "",
             "- The tables are comma-separated text in UTF-8, with one header row and "
             "no comment lines. They open in R, Stata, SPSS, and spreadsheet programs. "
             "In a spreadsheet, import the file as UTF-8 text if accented characters "
             "look wrong.",
             "- An empty cell means the run did not record the value. It never means zero.",
             "- Yes and no are written as 1 and 0.",
             f"- A line break inside a text cell is written as{LINE_BREAK_MARK}(a "
             "paragraph mark with a space on each side), so that every row stays on "
             "one line. Replace the mark with a line break to restore the text as "
             "recorded. The run's own files hold the text unchanged.",
             "- Text is otherwise written as recorded. A spreadsheet may read a cell "
             "that begins with =, +, -, or @ as a formula; import the column as text "
             "if that matters.",
             "- Numbers are copied from what the engine stored. Nothing in these "
             "tables was recalculated.", ""]
    for table in tables:
        lines += [f"## {table.file}", "",
                  f"{table.title}. Each row is {table.row_is}. Rows: {len(table.rows)}.", "",
                  "| Column | Meaning |", "|---|---|"]
        lines += [f"| `{name}` | {meaning} |" for name, meaning, _ in table.columns]
        lines.append("")
        for note in table.notes:
            lines += [note, ""]
    if transcripts:
        lines += ["## transcripts/", "",
                  f"One text file for each of the {transcripts} conversations, named "
                  "after the condition and the replicate. Each file gives the turns in "
                  "order, with the speaker and the text exactly as recorded, line "
                  "breaks included. transcripts/turns.csv holds the same turns as a "
                  "table.", ""]
    lines += ["## Other files", "",
              "- `methods.md`: a plain-language account of how the results were "
              "produced, built from the run's stored facts.",
              "- `manifest.json`: every file this export was built from, with its "
              "SHA-256 hash, and every file it wrote.", ""]
    return "\n".join(lines)


# --- the export ----------------------------------------------------------------


def export_results(root, study, *, run=None, out=None, client=PYTHON_CLIENT, now=None):
    """Export ``study``'s results from the workspace at ``root``.

    ``run`` names a run directory (absolute, relative to the workspace, or a
    bare directory name under ``runs/``); absent, the newest completed run is
    used, with its newest analysis and evaluation. ``out`` names a new or
    empty destination folder; absent, a new folder under ``exports/`` in the
    workspace. A relative ``out`` is taken from the workspace root.

    Returns a JSON-ready description of what was written. Raises
    :class:`ResultsExportRefusal` before anything is written when the request
    cannot be met.
    """
    if client not in CLIENTS:
        client = PYTHON_CLIENT
    root = os.path.abspath(root)
    # The folder's name and the summary's date are for a person, so they use
    # the local calendar; the manifest's timestamp is UTC.
    now = (now or datetime.now(timezone.utc)).astimezone()
    steps = _steps(client, study)
    _check_study_name(study, steps)
    run_name = _select_run(root, study, run, steps, client)
    target = _destination(root, study, out, steps, now)

    runs_root = _runs_root(root)
    run_directory = os.path.join(runs_root, run_name)
    sources = _Sources(root)
    missing = []

    # The run.
    loaded = _load_generations(os.path.join(run_directory, "generations.jsonl"), sources)
    report = sources.json(os.path.join(run_directory, "report.json"), "run") or {}
    config = sources.json(os.path.join(run_directory, "config.json"), "run") or {}
    snapshot = sources.json(os.path.join(run_directory, "experiment.json"), "run")
    if not isinstance(snapshot, dict):
        snapshot = {}
        missing.append({"what": "the study's settings as the run saw them",
                        "why": "the run directory has no experiment.json snapshot"})
    if not isinstance(report, dict):
        report = {}
    if not isinstance(config, dict):
        config = {}
    stamp = sources.text(os.path.join(run_directory, "experiment-hash.txt"), "run")
    experiment_hash = (stamp or "").strip() or config.get("experimentHash") or (
        loaded["observed"]["experimentHash"][0] if loaded["observed"]["experimentHash"] else None)

    tables = [_responses_table(study, run_name, loaded)]
    if loaded["readouts"]:
        tables.append(_readouts_table(study, run_name, loaded))
    if loaded["unreadableLines"]:
        missing.append({"what": f"{loaded['unreadableLines']} line(s) of generations.jsonl",
                        "why": "they could not be read as JSON records"})

    # The analysis: the newest analysis directory of this run, or the run's
    # own effect table (the Mac engine writes one beside the generations).
    analysis_name = _find_analysis(root, study, run_name)
    effects_directory = None
    if analysis_name:
        effects_directory = os.path.join(runs_root, analysis_name)
    elif os.path.isfile(os.path.join(run_directory, "effect-sizes.csv")):
        effects_directory = run_directory
    analysis_label = _relative(effects_directory, root) if effects_directory else None
    effects_source, outcomes, corrections, units = None, [], [], []
    unit_recorded, strata_rows, analysis_flags = False, 0, {}
    exclusions = []
    if effects_directory:
        role = "analysis" if analysis_name else "run"
        text = sources.text(os.path.join(effects_directory, "effect-sizes.csv"), role) or ""
        rows, effects_source = _read_effects(text)
        stamped_unit = None
        if analysis_name:
            unit_stamp = sources.json(os.path.join(effects_directory, "unit-of-analysis.json"), role)
            if isinstance(unit_stamp, dict):
                stamped_unit = unit_stamp.get("unitOfAnalysis")
            analysis_report = sources.json(os.path.join(effects_directory, "analysis.json"), role)
            if isinstance(analysis_report, dict):
                analysis_flags = {k: analysis_report[k] for k in ("epochUnverified", "measurementDrift")
                                  if analysis_report.get(k)}
            for flag, file in (("epochUnverified", "epoch-unverified.json"),
                               ("measurementDrift", "measurement-drift.json")):
                stamp_file = sources.json(os.path.join(effects_directory, file), role)
                if isinstance(stamp_file, dict) and stamp_file.get(flag):
                    analysis_flags[flag] = stamp_file[flag]
            stamp_file = sources.json(os.path.join(effects_directory, "exclusions.json"), role)
            if stamp_file is None and isinstance(analysis_report, dict):
                stamp_file = analysis_report.get("exclusions")
            exclusions.append(("The analysis", stamp_file))
        else:
            stamped_unit = report.get("unitOfAnalysis")
            exclusions.append(("The run's own analysis", report.get("exclusions")))
        if not isinstance(stamped_unit, str) or not stamped_unit:
            stamped_unit = None
        pooled, strata = _effects_tables(
            study, run_name, os.path.basename(effects_directory), rows,
            effects_source, stamped_unit)
        tables.append(pooled)
        if strata.rows:
            tables.append(strata)
        strata_rows = len(strata.rows)
        pooled_rows = [row for row in rows if row["stratify_by"] in ("", "pooled")]
        for row in pooled_rows:
            if row["outcome"] and row["outcome"] not in outcomes:
                outcomes.append(row["outcome"])
            if row["correction"] and row["correction"] not in corrections:
                corrections.append(row["correction"])
            unit = row["unit"] or stamped_unit
            if unit and unit not in units:
                units.append(unit)
        unit_recorded = bool(units)
    else:
        missing.append({"what": "effects.csv",
                        "why": "no analysis of this run was found; analyze the run, "
                               "then export again",
                        "repair": f"{steps['analyze']}, then {steps['export']}"})

    # The evaluations: the newest completed one of each kind for this run.
    context_rows = {"judgmentRows": [], "codingRows": []}
    reports = {"paired": {}, "coding": {}}
    names = {}
    for kind, (report_name, rows_name) in _EVALUATION_KINDS.items():
        name = _find_evaluation(root, study, run_name, kind)
        names[kind] = name
        if not name:
            continue
        directory = os.path.join(runs_root, name)
        evaluation = sources.json(os.path.join(directory, report_name), "evaluation")
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        rows, unreadable = _read_rows(os.path.join(directory, rows_name), sources, "evaluation")
        if unreadable:
            missing.append({"what": f"{unreadable} line(s) of {rows_name}",
                            "why": "they could not be read as JSON records"})
        reports[kind] = evaluation
        if kind == "paired":
            context_rows["judgmentRows"] = rows
            tables.append(_judgments_table(study, run_name, name, rows, evaluation))
        else:
            context_rows["codingRows"] = rows
            tables.append(_codings_table(study, run_name, name, rows, evaluation))
        stamp_file = sources.json(os.path.join(directory, "exclusions.json"), "evaluation")
        if stamp_file is None:
            stamp_file = evaluation.get("exclusions")
        exclusions.append(("The evaluation", stamp_file))
    if not any(names.values()) and (snapshot.get("judges") or snapshot.get("judgeRubricFile")):
        missing.append({"what": "judgments.csv",
                        "why": "the study declares judges, and no completed evaluation "
                               "of this run was found; evaluate the run, then export again",
                        "repair": f"{steps['evaluate']}, then {steps['export']}"})

    # Transcripts, for a multi-agent study.
    conversations = _conversations(study, run_name, loaded["responses"])
    if conversations:
        tables.append(_turns_table(study, run_name, conversations))

    from .. import __version__
    context = {
        "study": study, "runName": run_name, "runStudy": _experiment_of(run_directory),
        "snapshot": snapshot, "config": config, "report": report, "loaded": loaded,
        "missing": missing, "version": __version__, "date": now.strftime("%Y-%m-%d"),
        "conversations": len(conversations),
        "analysisLabel": analysis_label, "pairedName": names["paired"],
        "codingName": names["coding"], "pairedReport": reports["paired"],
        "codingReport": reports["coding"], "outcomes": outcomes,
        "corrections": corrections, "units": units, "unitRecorded": unit_recorded,
        "strataRows": strata_rows, "effectsSource": effects_source,
        "analysisFlags": analysis_flags, "exclusions": exclusions,
        "experimentHash": experiment_hash, **context_rows,
    }
    methods = _methods(context)
    codebook = _codebook(study, tables, len(conversations))

    files = {table.file: table.csv_text() for table in tables}
    for conversation in conversations:
        files[f"transcripts/{conversation['name']}.txt"] = _transcript_text(
            study, run_name, conversation)
    files["methods.md"] = methods
    files["codebook.md"] = codebook

    written = []
    row_counts = {table.file: table for table in tables}
    for name, text in files.items():
        data = text.encode("utf-8")
        entry = {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if name in row_counts:
            entry["rows"] = len(row_counts[name].rows)
            entry["columns"] = row_counts[name].names
        written.append(entry)
    seen, not_available = set(), []
    for entry in missing:
        if entry["what"] not in seen:
            seen.add(entry["what"])
            not_available.append(entry)
    # The files stay neutral about which client made them: a repair that
    # names a command goes only into the answer returned to the caller.
    recorded = [{"what": entry["what"], "why": entry["why"]} for entry in not_available]
    manifest = {
        "schemaVersion": EXPORT_SCHEMA_VERSION,
        "kind": EXPORT_KIND,
        "study": study,
        "createdAt": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exporter": f"steerlab {__version__}",
        "engine": config.get("substrate"),
        "run": "runs/" + run_name,
        "analysis": analysis_label,
        "evaluations": {kind: ("runs/" + name if name else None) for kind, name in names.items()},
        "experimentHash": experiment_hash,
        "freezeForced": bool(snapshot.get("freezeForced")),
        "forcedGatesSkipped": list(snapshot.get("forcedGatesSkipped") or []),
        "capabilityBatteryNotApplied": list(snapshot.get("capabilityBatteryNotApplied") or []),
        "lineBreakMark": LINE_BREAK_MARK,
        "sources": sources.entries,
        "files": written,
        "notAvailable": recorded,
    }
    files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    created = not os.path.isdir(target)
    try:
        os.makedirs(target, exist_ok=True)
        for name, text in files.items():
            path = os.path.join(target, *name.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
    except OSError:
        if created:
            shutil.rmtree(target, ignore_errors=True)
        raise

    return {
        "changed": True,
        "study": study,
        "exportDirectory": target,
        "run": "runs/" + run_name,
        "analysis": analysis_label,
        "evaluations": manifest["evaluations"],
        "engine": manifest["engine"],
        "files": [{k: entry[k] for k in ("file", "rows", "columns") if k in entry}
                  for entry in written] + [{"file": "manifest.json"}],
        "notAvailable": not_available,
        "freezeForced": manifest["freezeForced"],
        "capabilityBatteryNotApplied": manifest["capabilityBatteryNotApplied"],
    }

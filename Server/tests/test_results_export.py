"""``results export``: a study's stored results as files a researcher can use.

Every fixture here is synthetic and small. The Python engine's analysis is the
real one (``tasks.analyze`` over a synthetic run); the Mac engine's files are
written in the shapes its writers produce, and the Swift suite exports a real
Mac analysis through the bridge (``ResultsExportTests``).

What these tests hold:

* the tables' columns and row counts, for a run from each engine;
* both effect-size dialects normalize to one table, with the numbers copied
  and never recalculated;
* a judged study (paired judging and response coding), noncompliant rows kept
  and marked;
* a multi-agent study's transcripts;
* a forced study and a battery exemption are stated plainly;
* exclusions are reported from the engine's own stamp;
* every CSV parses with the standard library's reader under a plain header,
  and the codebook describes every column;
* nothing under ``runs/`` changes, and what a run did not store is said to be
  not available.
"""

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys

import pytest

from steerlab_server import cli_envelope, client_cli
from steerlab_server.client import diagnostic_commands, results_commands, results_export
from steerlab_server.experiment import study_stats, tasks
from steerlab_server.experiment.manifest import Manifest

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOSITORY = os.path.dirname(SERVER_DIR)

STUDY = "tone-study"
RUN = f"20261004T000000000-exp-{STUDY}-run"
ANALYSIS = f"20261004T010000000-exp-{STUDY}-analyze"
EVALUATION = f"20261004T020000000-exp-{STUDY}-evaluate"
ITEMS = ("item-1", "item-2", "item-3", "item-4")
SAMPLES = 2

#: The header the Mac engine writes (``StudyAnalysisRendering.effectSizesCSV``).
#: ``test_the_mac_header_here_is_the_one_the_mac_engine_writes`` keeps it true.
MAC_EFFECTS_HEADER = ("condition,metric,n,meanDiff,ciLower,ciUpper,wilcoxonW,"
                      "wilcoxonP,adjustedP,correction,stratifyBy,stratum,unit,"
                      "estimand,inference")

EFFECT_COLUMNS = [
    "study", "run", "analysis", "outcome", "condition", "estimate", "ci_lower",
    "ci_upper", "n_pairs", "unit_of_analysis", "unit_of_analysis_source",
    "test", "test_statistic", "p_value", "adjusted_p_value", "correction",
    "modality"]


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    monkeypatch.setattr(cli_envelope, "now", lambda: 1_000.0)


@pytest.fixture
def root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    return workspace


# --- fixtures: a run as each engine writes it ----------------------------------


def _manifest(**extra):
    return {"name": STUDY, "modelID": "test/model", "modelRevision": "rev-1",
            "status": "draft",
            "concepts": [], "taskPromptsFile": None, "temperature": 0.7,
            "maxTokens": 64, "samplesPerItem": SAMPLES, "seeds": [0],
            "experimentDescription": "Does a formal register change the answers?",
            "conditions": [
                {"name": "baseline", "slots": [], "alphaInNormUnits": True},
                {"name": "formal", "alphaInNormUnits": True, "bandWidth": 1,
                 "slots": [{"concept": "formality", "layer": 6, "alpha": 4.0}]}],
            **extra}


def _text(condition, item, sample):
    """A response with a comma, a quotation mark, an accent, and a paragraph
    break, so the tables are tested against the awkward cases."""
    words = " ".join(["indeed"] * (3 + sample + (4 if condition == "formal" else 0)))
    return f'Dear reader, "{item}" is naïve.\n\n{words}'


def _python_records(manifest_hash, conditions=("baseline", "formal"), **extra):
    records = []
    for condition in conditions:
        for index, item in enumerate(ITEMS):
            for sample in range(SAMPLES):
                text = _text(condition, item, sample)
                records.append({
                    "experiment": STUDY, "experimentHash": manifest_hash,
                    "modelID": "test/model", "modelRevision": "rev-1",
                    "promptMode": "chatAssistant", "condition": condition,
                    "seed": 1000 + sample, "seedPolicy": "derivedSHA256",
                    "sampleIndex": sample, "promptIndex": index, "promptID": item,
                    "prompt": f"Describe a quiet weekend ({item}).",
                    "interventionState": {"slots": []}, "output": text,
                    "wordCount": len(text.split()), "distinct2": 0.5,
                    "finishReason": "stop", "temperature": 0.7, "doSample": True,
                    "topP": 0.9, "topK": None, "dtype": "torch.float32",
                    "device": "cpu", "engine": "python-hf-transformers", **extra})
    return records


def _mac_records(manifest_hash):
    records = []
    for condition in ("baseline", "formal"):
        for index, item in enumerate(ITEMS):
            text = _text(condition, item, 0)
            records.append({
                "experiment": STUDY, "experimentHash": manifest_hash,
                "modelID": "test/model", "modelRevision": "rev-1",
                "taskPromptsFile": "prompts/tasks/weekend.jsonl",
                "taskPromptsHash": "a" * 64, "promptMode": "chatAssistant",
                "systemPromptComposition": {"agent": None, "study": None},
                "qwenThinkingEnabled": False, "condition": condition, "seed": 0,
                "seedInert": True, "promptIndex": index, "promptID": item,
                "prompt": f"Describe a quiet weekend ({item}).", "output": text,
                "wordCount": len(text.split()), "distinct2": 0.5,
                "finishReason": "stop", "markerDensity": {"formality": 0.25}})
    return records


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _write_lines(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))


def _write_run(root, manifest, records, *, engine="python", name=RUN,
               report=None, complete=True):
    """One run directory, with the files both engines write for a run."""
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    directory = os.path.join(str(root), "runs", name)
    _write_json(os.path.join(directory, "experiment.json"), manifest)
    with open(os.path.join(directory, "experiment-hash.txt"), "w", encoding="utf-8") as handle:
        handle.write(manifest_hash + "\n")
    _write_json(os.path.join(directory, "config.json"), {
        "schemaVersion": 4, "runId": name, "runType": "run",
        "createdAt": "2026-10-04T00:00:00Z",
        "substrate": "python-hf-transformers" if engine == "python" else "swift-mlx",
        "appVersion": ("steerlab-server 0.9.6+0a1b2c3d" if engine == "python"
                       else "swift-app 0.9.6+0a1b2c3d"),
        "platform": "linux-x86_64" if engine == "python" else "macOS-arm64",
        "modelID": manifest["modelID"], "revision": manifest.get("modelRevision"),
        "experiment": manifest["name"], "experimentHash": manifest_hash,
        "temperature": manifest.get("temperature"),
        "samplesPerItem": manifest.get("samplesPerItem"),
        "seedPolicy": "derivedSHA256" if engine == "python" else "manifestSeeds",
        "dtype": "float32" if engine == "python" else None,
        "pythonEnvironment": ({"python": "3.12.0", "implementation": "cpython",
                               "packages": {"torch": "2.0.0", "transformers": "4.0.0"}}
                              if engine == "python" else None),
        "jobId": None, "notes": {}})
    _write_lines(os.path.join(directory, "generations.jsonl"), records)
    if complete:
        _write_json(os.path.join(directory, "report.json"), report or {
            "experiment": manifest["name"], "experimentHash": manifest_hash,
            "conditions": {}, "truncation": {"classified": len(records),
                                             "lengthStopped": 0}})
    return directory


def _write_study(root, manifest):
    _write_json(os.path.join(str(root), "experiments", manifest["name"],
                             "experiment.json"), manifest)


def _python_study(root, manifest=None, records=None):
    """A Python-engine run with the engine's REAL analysis of it."""
    manifest = manifest or _manifest()
    _write_study(root, manifest)
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    _write_run(root, manifest, records or _python_records(manifest_hash))
    analysis = tasks.analyze(STUDY, root=str(root), log=lambda _: None)
    return os.path.basename(analysis)


def _mac_effect_rows():
    """Effect rows as the Mac engine's report holds them."""
    return [
        {"condition": "formal", "metric": "wordCount", "n": 4, "meanDiff": 4.5,
         "ciLower": 4.25, "ciUpper": 4.75, "wilcoxonW": 0.5, "wilcoxonP": 0.125,
         "adjustedP": 0.25, "correction": "bh"},
        {"condition": "formal", "metric": "distinct2", "n": 4, "meanDiff": 0.5,
         "ciLower": 0.25, "ciUpper": 0.75, "wilcoxonW": None, "wilcoxonP": None,
         "adjustedP": None, "correction": "bh"},
        {"condition": "formal", "metric": "wordCount", "n": 2, "meanDiff": 4.5,
         "ciLower": 4.25, "ciUpper": 4.75, "wilcoxonW": 0.5, "wilcoxonP": 0.5,
         "adjustedP": None, "correction": None, "stratifyBy": "promptID",
         "stratum": "item-1", "unit": "sample", "estimand": "withinItemSamples",
         "inference": "diagnostic"},
    ]


def _mac_effects_csv(rows):
    """``StudyAnalysisRendering.effectSizesCSV``, line for line."""
    def cell(value):
        return "" if value is None else str(value)
    lines = [MAC_EFFECTS_HEADER]
    for row in rows:
        lines.append(",".join([
            row["condition"], row["metric"], str(row["n"]), str(row["meanDiff"]),
            str(row["ciLower"]), str(row["ciUpper"]), cell(row["wilcoxonW"]),
            cell(row["wilcoxonP"]), cell(row["adjustedP"]), cell(row["correction"]),
            row.get("stratifyBy") or "pooled", cell(row.get("stratum")),
            cell(row.get("unit")), cell(row.get("estimand")),
            cell(row.get("inference"))]))
    return "\n".join(lines) + "\n"


def _write_mac_analysis(root, manifest, rows, name=ANALYSIS, run=RUN, **extra):
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    directory = os.path.join(str(root), "runs", name)
    _write_json(os.path.join(directory, "experiment.json"), manifest)
    _write_json(os.path.join(directory, "config.json"), {
        "schemaVersion": 4, "runId": name, "runType": "analyze",
        "substrate": "swift-mlx", "experiment": manifest["name"],
        "experimentHash": manifest_hash})
    _write_json(os.path.join(directory, "analysis.json"), {
        "experiment": manifest["name"], "experimentHash": manifest_hash,
        "sourceRun": run, "sourceRunExperimentHash": manifest_hash,
        "effectSizes": rows, **extra})
    with open(os.path.join(directory, "effect-sizes.csv"), "w", encoding="utf-8") as handle:
        handle.write(_mac_effects_csv(rows))
    return name


def _mac_study(root, manifest=None):
    manifest = manifest or _manifest(samplesPerItem=1)
    _write_study(root, manifest)
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    _write_run(root, manifest, _mac_records(manifest_hash), engine="mac")
    return _write_mac_analysis(root, manifest, _mac_effect_rows())


def _python_analysis_by_hand(root, rows, name=ANALYSIS, run=RUN):
    """An analysis directory in the Python engine's layout, with rows written
    by the engine's own row type."""
    directory = os.path.join(str(root), "runs", name)
    os.makedirs(directory, exist_ok=True)
    _write_json(os.path.join(directory, "config.json"), {
        "runType": "analyze", "substrate": "python-hf-transformers",
        "experiment": STUDY})
    with open(os.path.join(directory, "source-run.txt"), "w", encoding="utf-8") as handle:
        handle.write(run + "\n")
    with open(os.path.join(directory, "effect-sizes.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(study_stats.EFFECT_SIZES_HEADER)
        for row in rows:
            writer.writerow(row.as_csv_row())
    return name


def _python_effect_rows():
    """The same numbers as ``_mac_effect_rows``, in the Python engine's row type."""
    nan = float("nan")

    def row(endpoint, n, mean, low, high, w, p, adjusted, correction, **extra):
        return study_stats.EffectRow(
            condition="formal", endpoint=endpoint,
            ci=study_stats.BootstrapCI(n=n, mean=mean, ci_lower=low, ci_upper=high,
                                       replicates=10_000, seed=0),
            wilcoxon_w=w, wilcoxon_p=p, adjusted_p=adjusted, correction=correction,
            modality="injection", **extra)
    return [
        row("wordCount", 4, 4.5, 4.25, 4.75, 0.5, 0.125, 0.25, "bh"),
        row("distinct2", 4, 0.5, 0.25, 0.75, nan, nan, nan, "bh"),
        row("wordCount", 2, 4.5, 4.25, 4.75, 0.5, 0.5, nan, "",
            stratify_by="promptID", stratum="item-1", unit="sample",
            estimand="withinItemSamples", inference="diagnostic"),
    ]


def _export(root, **arguments):
    return results_export.export_results(str(root), STUDY, **arguments)


def _table(result, name):
    with open(os.path.join(result["exportDirectory"], *name.split("/")),
              encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read(result, name):
    with open(os.path.join(result["exportDirectory"], *name.split("/")),
              encoding="utf-8") as handle:
        return handle.read()


def _tree(directory):
    """Every file under a directory with its hash and modification time."""
    found = {}
    for base, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(base, name)
            with open(path, "rb") as handle:
                found[os.path.relpath(path, directory)] = (
                    hashlib.sha256(handle.read()).hexdigest(),
                    os.stat(path).st_mtime_ns)
    return found


# --- the tables ----------------------------------------------------------------


def test_a_python_engine_run_exports_every_file_from_its_real_analysis(root):
    analysis = _python_study(root)
    before = _tree(os.path.join(str(root), "runs"))

    result = _export(root)

    assert result["changed"] is True and result["run"] == "runs/" + RUN
    assert result["analysis"] == "runs/" + analysis
    assert result["engine"] == "python-hf-transformers"
    directory = result["exportDirectory"]
    assert os.path.dirname(directory) == os.path.join(str(root), "exports")
    assert sorted(os.listdir(directory)) == [
        "codebook.md", "effects-by-stratum.csv", "effects.csv", "manifest.json",
        "methods.md", "responses.csv"]

    # One row per response, with the outcomes the run recorded for each.
    responses = _table(result, "responses.csv")
    assert len(responses) == 2 * len(ITEMS) * SAMPLES
    assert list(responses[0]) == [
        "study", "run", "condition", "item", "item_index", "sample_index",
        "seed", "prompt", "response", "finish_reason", "word_count",
        "distinct_2", "model", "model_revision"]
    first = responses[0]
    assert (first["study"], first["run"], first["condition"], first["item"]) == (
        STUDY, RUN, "baseline", "item-1")
    assert (first["item_index"], first["sample_index"], first["seed"]) == ("0", "0", "1000")
    assert first["word_count"] == str(len(_text("baseline", "item-1", 0).split()))

    # The effect table is the engine's, row for row and digit for digit.
    with open(os.path.join(str(root), "runs", analysis, "effect-sizes.csv"),
              encoding="utf-8") as handle:
        stored = list(csv.DictReader(handle))
    pooled = [row for row in stored if row["stratifyBy"] == "pooled"]
    effects = _table(result, "effects.csv")
    assert list(effects[0]) == EFFECT_COLUMNS
    assert len(effects) == len(pooled) > 0
    for exported, engine in zip(effects, pooled):
        assert exported["outcome"] == engine["endpoint"]
        assert exported["condition"] == engine["condition"]
        assert exported["estimate"] == engine["deltaMean"]
        assert exported["ci_lower"] == engine["ciLower"]
        assert exported["ci_upper"] == engine["ciUpper"]
        assert exported["n_pairs"] == engine["n"] == str(len(ITEMS))
        assert exported["test_statistic"] == engine["wilcoxonW"]
        assert exported["p_value"] == engine["wilcoxonP"]
        assert exported["adjusted_p_value"] == engine["adjustedP"]
        assert exported["correction"] == engine["correction"] == "bh"
        assert exported["test"] == "wilcoxon_signed_rank"
        assert exported["modality"] == engine["modality"]
        assert (exported["run"], exported["analysis"]) == (RUN, analysis)
    strata = _table(result, "effects-by-stratum.csv")
    assert len(strata) == len(stored) - len(pooled) > 0
    assert {row["stratify_by"] for row in strata} == {"promptID"}

    # Nothing under runs/ changed: no file added, removed, rewritten, or touched.
    assert _tree(os.path.join(str(root), "runs")) == before


def test_a_mac_engine_run_exports_the_same_columns(root):
    analysis = _mac_study(root)
    result = _export(root)

    assert result["engine"] == "swift-mlx"
    assert result["analysis"] == "runs/" + analysis
    responses = _table(result, "responses.csv")
    assert len(responses) == 2 * len(ITEMS)
    # The Mac engine records marker density, and no sample index at
    # temperature zero: the column stays, and the cell is empty.
    assert responses[0]["marker_density_formality"] == "0.25"
    assert responses[0]["sample_index"] == ""
    effects = _table(result, "effects.csv")
    assert list(effects[0]) == EFFECT_COLUMNS
    assert [(row["outcome"], row["estimate"], row["n_pairs"]) for row in effects] == [
        ("wordCount", "4.5", "4"), ("distinct2", "0.5", "4")]
    # An undefined test is an empty cell, and the Mac engine records no modality.
    assert (effects[1]["test_statistic"], effects[1]["p_value"]) == ("", "")
    assert {row["modality"] for row in effects} == {""}
    [stratum] = _table(result, "effects-by-stratum.csv")
    assert (stratum["stratify_by"], stratum["stratum"], stratum["unit_of_analysis"],
            stratum["estimand"], stratum["inference"]) == (
        "promptID", "item-1", "sample", "withinItemSamples", "diagnostic")


def test_both_effect_size_dialects_normalize_to_the_same_table(tmp_path):
    """The Mac engine writes ``metric`` and ``meanDiff``; the Python engine
    writes ``endpoint``, ``deltaMean``, and ``modality``. One set of numbers in
    each dialect has to export as one table."""
    exported = {}
    for engine in ("python", "mac"):
        workspace = tmp_path / engine
        manifest = _manifest()
        manifest_hash = Manifest.from_dict(manifest).content_hash()
        records = (_python_records(manifest_hash) if engine == "python"
                   else _mac_records(manifest_hash))
        _write_run(workspace, manifest, records, engine=engine)
        if engine == "python":
            _python_analysis_by_hand(workspace, _python_effect_rows())
        else:
            _write_mac_analysis(workspace, manifest, _mac_effect_rows())
        result = _export(workspace)
        exported[engine] = {name: _table(result, name)
                            for name in ("effects.csv", "effects-by-stratum.csv")}

    for name in ("effects.csv", "effects-by-stratum.csv"):
        python_rows, mac_rows = exported["python"][name], exported["mac"][name]
        assert list(python_rows[0]) == list(mac_rows[0])
        assert len(python_rows) == len(mac_rows)
        for python_row, mac_row in zip(python_rows, mac_rows):
            # Only the Python engine records the kind of intervention.
            assert python_row.pop("modality", "") in ("injection", "")
            mac_row.pop("modality", None)
            assert python_row == mac_row
    assert [row["estimate"] for row in exported["mac"]["effects.csv"]] == ["4.5", "0.5"]


def test_the_mac_header_here_is_the_one_the_mac_engine_writes():
    source = os.path.join(REPOSITORY, "Sources", "ExperimentKit",
                          "StudyAnalysisRendering.swift")
    with open(source, encoding="utf-8") as handle:
        text = "".join(re.findall(r'"([^"\n]*)"', handle.read()))
    assert MAC_EFFECTS_HEADER in text
    assert study_stats.EFFECT_SIZES_HEADER[:4] == ["condition", "endpoint", "n", "deltaMean"]


def test_the_mac_engines_own_run_table_is_used_when_no_analysis_exists(root):
    """A Mac run writes effect-sizes.csv beside its generations. With no
    separate analysis, that table is exported, and the unit of analysis comes
    from the run's own report."""
    manifest = _manifest(samplesPerItem=1)
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    directory = _write_run(root, manifest, _mac_records(manifest_hash), engine="mac",
                           report={"experiment": STUDY, "conditions": {},
                                   "unitOfAnalysis": "transcript"})
    with open(os.path.join(directory, "effect-sizes.csv"), "w", encoding="utf-8") as handle:
        handle.write(_mac_effects_csv(_mac_effect_rows()[:1]))

    result = _export(root)
    [row] = _table(result, "effects.csv")
    assert result["analysis"] == "runs/" + RUN
    assert (row["unit_of_analysis"], row["unit_of_analysis_source"]) == (
        "transcript", "recorded")


def test_an_unstamped_unit_of_analysis_is_shown_as_the_engine_default(root):
    _python_study(root)
    result = _export(root)
    assert {(row["unit_of_analysis"], row["unit_of_analysis_source"])
            for row in _table(result, "effects.csv")} == {("item", "engine_default")}
    assert "engines' documented default" in _read(result, "methods.md")


def test_a_run_without_analysis_says_effects_are_not_available(root):
    manifest = _manifest()
    _write_run(root, manifest, _python_records("0" * 64))

    result = _export(root, client=results_export.MAC_CLIENT)

    assert not os.path.exists(os.path.join(result["exportDirectory"], "effects.csv"))
    [entry] = [e for e in result["notAvailable"] if e["what"] == "effects.csv"]
    assert "no analysis of this run was found" in entry["why"]
    assert entry["repair"] == (f"steerlab-cli experiment analyze {STUDY}, then "
                               f"steerlab-cli results export {STUDY}")
    methods = _read(result, "methods.md")
    assert "effects.csv was not written" in methods
    assert "No analysis of this run was found" in methods
    manifest_file = json.loads(_read(result, "manifest.json"))
    assert manifest_file["analysis"] is None
    # The files name no client: a repair that spells a command stays out of them.
    assert {"what": entry["what"], "why": entry["why"]} in manifest_file["notAvailable"]
    assert not any("repair" in recorded for recorded in manifest_file["notAvailable"])
    assert "steerlab-cli" not in methods + _read(result, "codebook.md")


def test_instrument_readouts_get_their_own_table(root):
    manifest = _manifest()
    records = _python_records("0" * 64)
    for condition in ("baseline", "formal"):
        records.append({
            "experiment": STUDY, "condition": condition, "promptIndex": 0,
            "promptID": "item-1", "prompt": "Pick one.", "target": "B",
            "targetSource": "declared", "instrument": "answerTokenLogprob",
            "options": ["A", "B"], "choiceProbability": {"A": 0.25, "B": 0.75},
            "logOdds": {"A": -1.5, "B": 1.5}, "selected": "B", "margin": 3.0,
            "temperature": 0.0})
    _write_run(root, manifest, records)

    result = _export(root)
    readouts = _table(result, "choice-readouts.csv")
    assert len(readouts) == 2
    assert (readouts[0]["target"], readouts[0]["selected"], readouts[0]["margin"],
            readouts[0]["target_probability"], readouts[0]["target_log_odds"]) == (
        "B", "B", "3.0", "0.75", "1.5")
    # A reading is not a response, and its temperature of zero is not the study's.
    assert len(_table(result, "responses.csv")) == 2 * len(ITEMS) * SAMPLES
    assert "temperature 0.7;" in _read(result, "methods.md")


# --- judged studies ------------------------------------------------------------


def _python_judgments():
    rows = []
    for judge in ("strict", "lenient"):
        for index, item in enumerate(ITEMS):
            row = {"promptID": item, "sampleIndex": 0, "condition": "formal",
                   "baselineSeed": 1000, "variantSeed": 1001,
                   "baselineWas": "A" if index % 2 == 0 else "B", "judge": judge}
            if judge == "lenient" and item == "item-4":
                row.update(outcome=None, noncompliant=True, judgment=None,
                           noncomplianceReason="The judge wrote an essay,\nnot a verdict.")
            else:
                outcome = ("variant", "baseline", "tie", "variant")[index]
                winner = "tie" if outcome == "tie" else (
                    "B" if (outcome == "variant") == (row["baselineWas"] == "A") else "A")
                row.update(outcome=outcome, confidence=0.75, judgment={
                    "winner": winner, "confidence": 0.75,
                    "brief_reason": "More formal, and clearer.",
                    "a_scores": {"clarity": 4}, "b_scores": {"clarity": 5},
                    "structured_fields": {"register": "formal"}})
            rows.append(row)
    return rows


def _write_python_paired_evaluation(root, name=EVALUATION, run=RUN):
    directory = os.path.join(str(root), "runs", name)
    rows = _python_judgments()
    _write_lines(os.path.join(directory, "judgments.jsonl"), rows)
    _write_json(os.path.join(directory, "config.json"),
                {"runType": "evaluate", "experiment": STUDY})
    _write_json(os.path.join(directory, "judge-report.json"), {
        "experiment": STUDY, "sourceRun": run,
        "rubricFile": "prompts/rubrics/register.md", "rubricHash": "b" * 64,
        "judges": [
            {"name": "strict", "kind": "local", "requestedModel": "judge/model-a",
             "actualModel": "judge/model-a", "revision": "rev-j", "pairs": 4,
             "conditions": {"formal": {"baselineWins": 1, "variantWins": 2, "ties": 1, "n": 4}}},
            {"name": "lenient", "kind": "openrouter", "requestedModel": "judge/model-b",
             "actualModel": "judge/model-b", "provider": "example-provider", "pairs": 4,
             "noncompliantJudgments": 1,
             "conditions": {"formal": {"baselineWins": 1, "variantWins": 1, "ties": 1, "n": 3}}}],
        "agreement": [{"judges": ["strict", "lenient"], "n": 3,
                       "percentAgreement": 1.0, "kappa": 1.0}],
        "pairs": 4, "noncompliantJudgments": 1, "judgedOn": "server"})
    return rows


def _write_mac_paired_evaluation(root, name=EVALUATION, run=RUN):
    directory = os.path.join(str(root), "runs", name)
    source = os.path.join(str(root), "runs", run)
    rows = []
    for index, item in enumerate(ITEMS):
        common = {"promptID": item, "sampleIndex": 0, "condition": "formal",
                  "baselineSeed": 0, "variantSeed": 0,
                  "baselineWas": "A" if index % 2 == 0 else "B", "judge": "strict"}
        if item == "item-4":
            rows.append({**common, "outcome": None, "noncompliant": True,
                         "noncomplianceReason": "No verdict.", "judgment": None})
            continue
        result = ("condition", "baseline", "tie")[index]
        rows.append({**common, "experiment": STUDY, "experimentHash": "c" * 64,
                     "sourceRunDirectory": source, "judgeKind": "claude",
                     "judgeModel": "judge/model-a", "judgePrompt": "Compare A and B.",
                     "judgeRubricFile": "prompts/rubrics/register.md",
                     "judgeRubricHash": "b" * 64, "structuredPrompt": None,
                     "prompt": "Describe a quiet weekend.",
                     "conditionWas": "B" if common["baselineWas"] == "A" else "A",
                     "judgment": {"winner": "tie" if result == "tie" else "B",
                                  "confidence": 0.5, "brief_reason": "A reason."},
                     "conditionResult": result})
    _write_lines(os.path.join(directory, "judgments.jsonl"), rows)
    _write_json(os.path.join(directory, "config.json"),
                {"runType": "evaluate", "experiment": STUDY})
    _write_json(os.path.join(directory, "judge-report.json"), {
        "experiment": STUDY, "experimentHash": "c" * 64,
        "sourceRunDirectory": source, "judgeModel": "judge/model-a",
        "judges": ["strict"], "judgeRubricFile": "prompts/rubrics/register.md",
        "judgeRubricHash": "b" * 64, "noncompliantJudgments": 1,
        "conditions": {"formal": {"pairs": 3, "conditionWins": 1, "baselineWins": 1,
                                  "ties": 1, "meanConfidence": 0.5,
                                  "structuredSummaries": {}}}})
    return rows


def test_a_judged_python_study_exports_every_judgment_with_noncompliant_rows_marked(root):
    _python_study(root)
    rows = _write_python_paired_evaluation(root)

    result = _export(root)

    judgments = _table(result, "judgments.csv")
    assert len(judgments) == len(rows) == 8
    assert result["evaluations"] == {"paired": "runs/" + EVALUATION, "coding": None}
    by_key = {(row["judge"], row["item"]): row for row in judgments}
    # The engine's "variant" and "condition" are one label here.
    assert [by_key[("strict", item)]["outcome"] for item in ITEMS] == [
        "condition", "baseline", "tie", "condition"]
    hole = by_key[("lenient", "item-4")]
    assert (hole["noncompliant"], hole["outcome"], hole["judge_choice"]) == ("1", "", "")
    assert hole["noncompliance_reason"] == (
        "The judge wrote an essay," + results_export.LINE_BREAK_MARK + "not a verdict.")
    assert [row["noncompliant"] for row in judgments].count("0") == 7
    # A judge's kind and model are joined from the evaluation's report.
    strict = by_key[("strict", "item-1")]
    assert (strict["judge_kind"], strict["judge_model"], strict["judge_revision"]) == (
        "local", "judge/model-a", "rev-j")
    assert by_key[("lenient", "item-1")]["judge_provider"] == "example-provider"
    assert (strict["score_a_clarity"], strict["score_b_clarity"],
            strict["field_register"], strict["confidence"]) == ("4", "5", "formal", "0.75")
    assert (strict["sample_index"], strict["baseline_seed"], strict["condition_seed"]) == (
        "0", "1000", "1001")

    methods = _read(result, "methods.md")
    assert "Judge `strict`: kind `local`, model `judge/model-a`, revision `rev-j`." in methods
    assert "Rubric: `prompts/rubrics/register.md`, SHA-256 `" + "b" * 64 + "`." in methods
    assert "Rows in judgments.csv: 8, of which 1 is marked noncompliant" in methods
    assert ("Agreement between `strict` and `lenient`, as the engine computed it over 3 "
            "shared pairs: proportion in agreement 1, Cohen's kappa 1.") in methods
    assert "- Judged outcome: which response each judge preferred" in methods


def test_a_judged_mac_study_exports_the_same_judgment_columns(root):
    _mac_study(root)
    rows = _write_mac_paired_evaluation(root)

    result = _export(root)

    judgments = _table(result, "judgments.csv")
    assert len(judgments) == len(rows) == 4
    assert [row["outcome"] for row in judgments] == ["condition", "baseline", "tie", ""]
    assert [row["noncompliant"] for row in judgments] == ["0", "0", "0", "1"]
    assert (judgments[0]["judge"], judgments[0]["judge_kind"], judgments[0]["judge_model"]) == (
        "strict", "claude", "judge/model-a")
    assert (judgments[0]["run"], judgments[0]["evaluation"]) == (RUN, EVALUATION)


def test_judgment_columns_do_not_depend_on_the_engine(tmp_path):
    headers = {}
    for engine, study, evaluation in (("python", _python_study, _write_python_paired_evaluation),
                                      ("mac", _mac_study, _write_mac_paired_evaluation)):
        workspace = tmp_path / engine
        study(workspace)
        evaluation(workspace)
        headers[engine] = list(_table(_export(workspace), "judgments.csv")[0])
    shared = headers["mac"]
    assert [name for name in headers["python"] if name in shared] == shared
    assert set(headers["python"]) - set(shared) == {
        "score_a_clarity", "score_b_clarity", "field_register"}


def test_a_coded_study_exports_one_row_per_coding(root):
    _python_study(root)
    directory = os.path.join(str(root), "runs", EVALUATION)
    rows = []
    for judge in ("coder-a", "coder-b"):
        for item in ITEMS:
            rows.append({"experiment": STUDY, "condition": "formal", "promptID": item,
                         "sampleIndex": 1, "seed": 1001, "wordCount": 12,
                         "codes": {"polite": True, "register": "formal", "hedges": 2},
                         "briefReason": "Polite throughout.", "judge": judge,
                         "judgeKind": "local", "judgeModel": "judge/model-a"})
    rows[-1] = {"experiment": STUDY, "condition": "formal", "promptID": "item-4",
                "sampleIndex": 1, "seed": 1001, "codes": None, "noncompliant": True,
                "noncomplianceReason": "No codes.", "judge": "coder-b",
                "judgeKind": "local", "judgeModel": "judge/model-a"}
    _write_lines(os.path.join(directory, "codings.jsonl"), rows)
    _write_json(os.path.join(directory, "coding-report.json"), {
        "mode": "perResponseCoding", "experiment": STUDY, "sourceRun": RUN,
        "judges": ["coder-a", "coder-b"], "judgeModel": "judge/model-a, judge/model-a",
        "judgeDetails": [{"name": name, "kind": "local", "requestedModel": "judge/model-a",
                          "actualModel": "judge/model-a"} for name in ("coder-a", "coder-b")],
        "judgeRubricFile": "prompts/rubrics/politeness.md", "judgeRubricHash": "d" * 64,
        "fields": [{"name": "polite", "type": "boolean", "optional": False},
                   {"name": "register", "type": "categorical", "optional": False,
                    "values": ["formal", "casual"]},
                   {"name": "hedges", "type": "integer", "optional": True}],
        "codings": 8, "noncompliantCodings": 1, "conditions": {},
        "fieldAgreement": [{"field": "polite", "judgeA": "coder-a", "judgeB": "coder-b",
                            "n": 3, "percentAgreement": 1.0, "kappa": None}],
        "sampling": {"samplePerCondition": 4, "sampleSeed": "7", "sampledRecords": 4,
                     "sourceRecords": 16, "rule": "stratified"}})

    result = _export(root)

    codings = _table(result, "codings.csv")
    assert len(codings) == 8
    assert result["evaluations"] == {"paired": None, "coding": "runs/" + EVALUATION}
    assert (codings[0]["code_polite"], codings[0]["code_register"],
            codings[0]["code_hedges"], codings[0]["word_count"]) == ("1", "formal", "2", "12")
    assert (codings[-1]["noncompliant"], codings[-1]["code_polite"]) == ("1", "")
    assert not os.path.exists(os.path.join(result["exportDirectory"], "judgments.csv"))
    methods = _read(result, "methods.md")
    assert "Response coding: each judge coded one response at a time" in methods
    assert "seeded subsample, not every response: 4 of 16 responses" in methods
    assert ("`polite`, coder-a and coder-b, over 3 shared responses: proportion in "
            "agreement 1.") in methods
    assert ("- Coded outcomes: the rubric's fields, coded by the judges for each response "
            "(codings.csv): `polite`, `register`, and `hedges`.") in methods
    assert "Allowed values: formal, casual." in _read(result, "codebook.md")


def test_an_unfinished_evaluation_is_not_exported(root):
    """Rows without a report are a judging session that never completed. The
    engines do not summarize one, and the export does not either."""
    _python_study(root)
    _write_lines(os.path.join(str(root), "runs", EVALUATION, "judgments.jsonl"),
                 _python_judgments()[:2])

    result = _export(root)
    assert result["evaluations"] == {"paired": None, "coding": None}
    assert not os.path.exists(os.path.join(result["exportDirectory"], "judgments.csv"))
    assert "No evaluation by judges was found for this run" in _read(result, "methods.md")
    # This study declares no judges, so nothing is missing on that account.
    assert "judgments.csv" not in [entry["what"] for entry in result["notAvailable"]]


def test_a_study_that_declares_judges_and_was_never_evaluated_says_so(root):
    manifest = _manifest(judges=[{"name": "strict", "kind": "local"}],
                         judgeRubricFile="prompts/rubrics/register.md")
    _write_run(root, manifest, _python_records("0" * 64))

    result = _export(root)

    [entry] = [e for e in result["notAvailable"] if e["what"] == "judgments.csv"]
    assert entry["repair"] == (f"steerlab run {STUDY} --runner <url> --verb evaluate, then "
                               f"steerlab results export {STUDY}")
    methods = _read(result, "methods.md")
    assert "The study's settings declare these judges: `strict`." in methods
    assert "- judgments.csv: the study declares judges, and no completed evaluation" in methods


# --- multi-agent studies -------------------------------------------------------


def _panel_records():
    seats = (("seat-chair", "Chair"), ("seat-member", "Member"), ("seat-chair", "Chair"))
    records = []
    for condition in ("formal", "baseline"):
        for replicate in (0, 1):
            for turn, (seat, speaker) in enumerate(seats):
                text = (f"{speaker} speaks in {condition}, play-through {replicate}.\n"
                        f"A second line, with a \"quotation\".")
                records.append({
                    "experiment": STUDY, "modelID": "test/model", "condition": condition,
                    "seed": 0, "promptID": f"turn-{turn + 1}", "promptIndex": turn,
                    "replicateIndex": replicate, "sampleIndex": replicate,
                    "prompt": "The floor is yours.", "output": text,
                    "speakerAgentID": seat, "speakerName": speaker,
                    "turnTitle": ("Opening", "Reply", "Closing")[turn],
                    "routedAgentIDs": ["seat-chair", "seat-member"],
                    "wordCount": len(text.split()), "distinct2": 1.0,
                    "endpoint": {"name": "vote", "kind": "choice", "value": "affirm"}
                    if turn == 2 else None})
                if records[-1]["endpoint"] is None:
                    del records[-1]["endpoint"]
    return records


def test_a_multi_agent_study_exports_readable_transcripts_and_a_table_of_turns(root):
    manifest = _manifest(studyKind="multiAgent",
                         multiAgentScenarioPath="prompts/panels/committee.json",
                         multiAgentScenarioHash="e" * 64)
    _write_run(root, manifest, _panel_records())

    result = _export(root)

    directory = os.path.join(result["exportDirectory"], "transcripts")
    # One text file for each conversation, baseline first.
    assert sorted(os.listdir(directory)) == [
        "baseline-replicate-0.txt", "baseline-replicate-1.txt",
        "formal-replicate-0.txt", "formal-replicate-1.txt", "turns.csv"]
    transcript = _read(result, "transcripts/formal-replicate-1.txt")
    assert transcript.startswith(f"Study: {STUDY}\nRun: {RUN}\nCondition: formal\nReplicate: 1\nTurns: 3\n")
    assert "Turn 1: Opening\nSpeaker: Chair (seat seat-chair)\n\n" in transcript
    # The text is as recorded, line breaks included.
    assert ('Chair speaks in formal, play-through 1.\nA second line, with a "quotation".\n'
            in transcript)
    assert transcript.index("Turn 1: Opening") < transcript.index("Turn 2: Reply") < \
        transcript.index("Turn 3: Closing")

    turns = _table(result, "transcripts/turns.csv")
    assert len(turns) == 12
    assert list(turns[0]) == [
        "study", "run", "conversation", "condition", "replicate", "turn", "turn_id",
        "turn_title", "seat", "speaker", "text", "word_count"]
    assert [row["conversation"] for row in turns[:4]] == [
        "baseline-replicate-0"] * 3 + ["baseline-replicate-1"]
    assert [(row["turn"], row["seat"], row["speaker"]) for row in turns[:3]] == [
        ("1", "seat-chair", "Chair"), ("2", "seat-member", "Member"),
        ("3", "seat-chair", "Chair")]
    assert turns[0]["text"] == (
        "Chair speaks in baseline, play-through 0." + results_export.LINE_BREAK_MARK
        + 'A second line, with a "quotation".')

    responses = _table(result, "responses.csv")
    assert (responses[2]["replicate"], responses[2]["speaker_seat"],
            responses[2]["speaker"], responses[2]["turn_title"],
            responses[2]["endpoint_name"], responses[2]["endpoint_value"],
            responses[2]["endpoint_unparsed"]) == (
        "0", "seat-chair", "Chair", "Closing", "vote", "affirm", "0")
    # A turn that declares no outcome leaves all three cells empty.
    assert (responses[0]["endpoint_name"], responses[0]["endpoint_unparsed"]) == ("", "")
    methods = _read(result, "methods.md")
    assert "This is a multi-agent study" in methods
    assert "Panel script: `prompts/panels/committee.json`" in methods
    assert ("- Conversations: 4, one for each condition and play-through "
            "(transcripts/).") in methods
    assert "4 conversations" in _read(result, "codebook.md")


def test_an_ordinary_study_has_no_transcripts_folder(root):
    _python_study(root)
    result = _export(root)
    assert not os.path.exists(os.path.join(result["exportDirectory"], "transcripts"))


# --- the methods summary -------------------------------------------------------


def test_methods_states_the_stored_facts(root):
    _python_study(root)
    methods = _read(_export(root), "methods.md")

    for sentence in (
            f"# Methods summary: {STUDY}",
            "Nothing was recalculated.",
            f"- Run: `runs/{RUN}`",
            "- Model: `test/model`",
            "- Model revision: `rev-1`",
            "- Engine: the Python engine (Hugging Face Transformers) (recorded as `python-hf-transformers`)",
            "- Back end: device `cpu`, numeric precision `float32`, platform `linux-x86_64`",
            "- Software that made the run: `steerlab-server 0.9.6+0a1b2c3d`",
            "- Libraries the run used: Python 3.12.0, torch 2.0.0, transformers 4.0.0",
            "- Description, as the researcher wrote it: Does a formal register change the answers?",
            "- Responses: 16 in total (baseline 8, formal 8).",
            "- Samples for each condition and item: 2.",
            "temperature 0.7; top-p 0.9; at most 64 new tokens for each response; "
            "2 samples for each item; seed policy `derivedSHA256`",
            "- **baseline**: no intervention (the comparison arm).",
            "- **formal**: steering: adds the direction for the concept 'formality' at "
            "layer 6, with strength 4 (in units of the typical activation size at that layer).",
            "- Primary outcome: the study's settings declare none.",
            "  - `wordCount`: the number of words in the response.",
            "- Test: the Wilcoxon signed-rank test on the paired differences.",
            "- Correction for multiple comparisons: Benjamini-Hochberg (false discovery rate)",
            "The study declares no exclusion rules, so no records were excluded.",
            "**Read this first.** This study was not frozen when the run was made.",
            "- **The study was not frozen when this run was made**",
            "- Responses cut off at a token limit: 0 of 16"):
        assert sentence in methods, sentence
    # What the run did not store is said, once, in its own section.
    tail = methods.split("## What is not available")[1]
    assert "the bootstrap settings: the analysis files do not record them" in tail


def test_methods_says_what_a_bare_run_did_not_record(root):
    """A run with nothing but responses and a report: no settings snapshot, no
    engine stamp, no revision. The summary says so rather than guessing."""
    directory = os.path.join(str(root), "runs", RUN)
    _write_lines(os.path.join(directory, "generations.jsonl"), [
        {"condition": "baseline", "promptID": "item-1", "prompt": "Hello?",
         "output": "Hello."}])
    _write_json(os.path.join(directory, "report.json"), {})

    result = _export(root)
    methods = _read(result, "methods.md")
    for what in ("the study's settings as the run saw them", "the model",
                 "the model revision", "the engine", "the software version",
                 "the freeze status", "the run's settings hash", "the exclusion rules",
                 "effects.csv"):
        assert what in [entry["what"] for entry in result["notAvailable"]], what
        assert f"- {what}: " in methods, what
    [row] = _table(result, "responses.csv")
    assert (row["sample_index"], row["seed"], row["item_index"]) == ("", "", "")


def test_a_forced_study_and_a_battery_exemption_are_stated_plainly(root):
    manifest = _manifest(
        status="frozen", frozenAt="2026-10-03T12:00:00Z", freezeHash="f" * 64,
        freezeForced=True, forcedGatesSkipped=["validateEvidence", "gitClean"],
        capabilityBatteryNotApplied=[{"condition": "guided", "reason": "interventionPolicy"}])
    _write_run(root, manifest, _python_records("0" * 64))

    result = _export(root)
    methods = _read(result, "methods.md")

    # Said at the top, where nobody adapting the text can miss it...
    assert methods.index("**Read this first.** This study was frozen with force: some freeze "
                         "checks were skipped. The capability battery was not applied to "
                         "guided.") < methods.index("## What this export was built from")
    # ...and in full in its own section.
    assert "**This study was frozen with force.**" in methods
    assert ("the check that the steering directions were validated (`validateEvidence`) "
            "and the check that the pinned inputs are committed (`gitClean`)") in methods
    assert "treats the study as not citable" in methods
    assert ("**Capability battery exemption.** The capability battery was not applied to "
            "guided, because its agent uses an intervention policy") in methods
    assert f"- Freeze hash: `{'f' * 64}`" in methods
    assert result["freezeForced"] is True
    stored = json.loads(_read(result, "manifest.json"))
    assert stored["freezeForced"] is True
    assert stored["forcedGatesSkipped"] == ["validateEvidence", "gitClean"]
    assert stored["capabilityBatteryNotApplied"] == [
        {"condition": "guided", "reason": "interventionPolicy"}]


def test_a_cleanly_frozen_study_says_so(root):
    manifest = _manifest(status="frozen", frozenAt="2026-10-03T12:00:00Z")
    _write_run(root, manifest, _python_records("0" * 64))
    methods = _read(_export(root), "methods.md")
    assert ("- The study was frozen before this run, with every freeze check passed "
            "(frozen at 2026-10-03T12:00:00Z).") in methods
    assert "frozen with force" not in methods


def test_a_run_with_exclusions_reports_the_engines_own_counts(root):
    """The engine applies the rules and stamps what each removed. The export
    repeats the stamp; it does not apply a rule itself."""
    manifest = _manifest(exclusionRules=[{"rule": "unparseableEndpoint",
                                          "endpoint": "parsedChoice"}])
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    records = _python_records(manifest_hash, parsedChoice="A", target="A")
    records[0]["parsedChoice"] = None           # one baseline response, unreadable
    analysis = _python_study(root, manifest, records)

    with open(os.path.join(str(root), "runs", analysis, "exclusions.json"),
              encoding="utf-8") as handle:
        stamp = json.load(handle)
    assert stamp["excludedRecords"] == 1         # the engine's count, not ours

    result = _export(root)
    methods = _read(result, "methods.md")
    assert "The analysis applied the study's declared exclusion rules and removed 1 record." in methods
    assert "- Rule `unparseableEndpoint`: " + stamp["rules"][0]["description"] in methods
    assert (f"- baseline: {stamp['consideredN']['baseline']} records considered; removed by "
            f"rule: unparseableEndpoint 1; {stamp['survivingN']['baseline']} kept.") in methods
    assert "Excluded records stay in the run and in responses.csv." in methods
    # Every response is still in the table, the excluded one included, with
    # its failed reading marked.
    responses = _table(result, "responses.csv")
    assert len(responses) == len(records)
    assert (responses[0]["parsed_choice"], responses[0]["parsed_choice_failed"]) == ("", "1")
    assert (responses[1]["parsed_choice"], responses[1]["parsed_choice_failed"]) == ("A", "0")


def test_a_declared_primary_outcome_is_named(root):
    manifest = _manifest(primaryOutcome="choiceLogOdds")
    _write_run(root, manifest, _python_records("0" * 64))
    methods = _read(_export(root), "methods.md")
    assert ("- Primary outcome, declared by the researcher before the run: "
            "`choiceLogOdds`.") in methods


def test_agent_and_latent_arms_are_described(root):
    manifest = _manifest(
        variantConditions=[{
            "name": "persona", "artifactPath": "runs/model-variants/persona.json",
            "artifactHash": "9" * 64,
            "artifact": {"name": "persona-agent", "baseModelID": "test/model",
                         "alphaInNormUnits": False, "systemPrompt": "Be brief.",
                         "injections": [{"concept": "formality", "layer": 3, "alpha": 2.0,
                                         "mode": "ablate"}],
                         "adapters": [{"name": "tone-adapter"}],
                         "interventionPolicies": [{"id": "p1"}]}}],
        saeLatentConditions=[{"name": "clamp", "feature": 42, "mode": "clamp"}])
    records = _python_records("0" * 64, conditions=("baseline", "persona", "clamp", "mystery"))
    _write_run(root, manifest, records)

    methods = _read(_export(root), "methods.md")
    assert ("- **persona**: an agent arm, the saved agent `persona-agent`, file "
            "`runs/model-variants/persona.json`, SHA-256 `" + "9" * 64 + "`. The agent "
            "carries steering that removes the direction for the concept 'formality' at "
            "layer 3, with strength 2 (in raw activation units), the fine-tuned adapter "
            "`tone-adapter`, its own system prompt, and 1 intervention policy.") in methods
    assert "- **clamp**: a latent arm, which edits one feature of a sparse autoencoder." in methods
    assert "feature `42`, mode `\"clamp\"`" in methods
    assert ("- **mystery**: not described in the run's settings snapshot, so what it "
            "applied is not available here.") in methods
    # A condition the study declares and the run does not hold is said too.
    assert "- **formal**: declared for the study, and the run holds no responses for it." in methods


# --- the shape of the files ----------------------------------------------------


def _rich_export(root):
    """A Python run with its real analysis, both kinds of evaluation, and
    instrument readouts: every ordinary table at once."""
    manifest = _manifest()
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    records = _python_records(manifest_hash, parsedChoice="A", target="A",
                              factors={"length": "short"}, arm="control")
    records.append({"experiment": STUDY, "condition": "baseline", "promptIndex": 0,
                    "promptID": "item-1", "prompt": "Pick one.", "target": "A",
                    "instrument": "answerTokenLogprob", "options": ["A", "B"],
                    "choiceProbability": {"A": 0.5, "B": 0.5},
                    "logOdds": {"A": 0.0, "B": 0.0}, "selected": "A", "margin": 0.0})
    _python_study(root, manifest, records)
    _write_python_paired_evaluation(root)
    return _export(root)


def _csv_files(result):
    return [entry["file"] for entry in result["files"] if entry["file"].endswith(".csv")]


@pytest.mark.parametrize("kind", ["ordinary", "panel"])
def test_every_csv_parses_with_the_standard_reader_under_a_plain_header(root, kind):
    if kind == "panel":
        _write_run(root, _manifest(), _panel_records())
        result = _export(root)
        expected = {"responses.csv", "transcripts/turns.csv"}
    else:
        result = _rich_export(root)
        expected = {"responses.csv", "choice-readouts.csv", "effects.csv",
                    "effects-by-stratum.csv", "judgments.csv"}
    assert set(_csv_files(result)) == expected

    for name in _csv_files(result):
        path = os.path.join(result["exportDirectory"], *name.split("/"))
        with open(path, "rb") as handle:
            data = handle.read()
        assert not data.startswith(b"\xef\xbb\xbf"), f"{name} starts with a byte-order mark"
        text = data.decode("utf-8")                       # UTF-8, or this raises
        assert "\r" not in text
        rows = list(csv.reader(io.StringIO(text, newline="")))
        header, body = rows[0], rows[1:]
        assert body, f"{name} has no rows"
        # A plain header: one name per column, usable as a variable name as it
        # stands in R, Stata, and SPSS, and no comment line above or below it.
        for column in header:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", column), (name, column)
            assert len(column) <= 32, (name, column)
        assert len(set(header)) == len(header), name
        assert not any(line.startswith("#") for line in text.splitlines()), name
        assert all(len(row) == len(header) for row in body), name
        # One row is one line: no cell holds a line break, which Stata and
        # SPSS would read as the end of the row.
        assert len(text.splitlines()) == len(rows), name
        entry = next(e for e in result["files"] if e["file"] == name)
        assert (entry["rows"], entry["columns"]) == (len(body), header)


def test_text_cells_keep_their_punctuation_and_mark_their_line_breaks(root):
    result = _rich_export(root)
    row = _table(result, "responses.csv")[0]
    recorded = _text("baseline", "item-1", 0)
    assert "\n" in recorded and '"' in recorded and "," in recorded
    assert row["response"] == recorded.replace("\n", results_export.LINE_BREAK_MARK)
    # Putting the line breaks back gives the recorded text exactly.
    assert row["response"].replace(results_export.LINE_BREAK_MARK, "\n") == recorded
    assert (row["factor_length"], row["arm"], row["target"]) == ("short", "control", "A")


def test_the_codebook_describes_every_column_of_every_table(root):
    result = _rich_export(root)
    codebook = _read(result, "codebook.md")
    sections = {}
    for block in codebook.split("\n## ")[1:]:
        title, _, body = block.partition("\n")
        sections[title.strip()] = body
    for entry in result["files"]:
        if "columns" not in entry:
            continue
        body = sections[entry["file"]]
        described = re.findall(r"^\| `([a-z0-9_]+)` \| (.+) \|$", body, re.M)
        assert [name for name, _ in described] == entry["columns"], entry["file"]
        for name, meaning in described:
            assert len(meaning) > 20 and meaning.rstrip().endswith("."), (entry["file"], name)
        assert f"Rows: {entry['rows']}." in body
    assert "An empty cell means the run did not record the value." in codebook
    assert results_export.LINE_BREAK_MARK.strip() in codebook
    # Fields a record holds that are not columns are named, not dropped in silence.
    assert "not table columns: " in sections["responses.csv"]
    assert "interventionState" in sections["responses.csv"]


def test_the_manifest_traces_the_export_to_the_files_it_read(root):
    analysis = _python_study(root)
    result = _export(root)
    manifest = json.loads(_read(result, "manifest.json"))

    assert (manifest["kind"], manifest["schemaVersion"]) == ("steerlab.resultsExport", 1)
    assert (manifest["study"], manifest["run"], manifest["analysis"]) == (
        STUDY, "runs/" + RUN, "runs/" + analysis)
    sources = {entry["path"]: entry for entry in manifest["sources"]}
    for relative in (f"runs/{RUN}/generations.jsonl", f"runs/{RUN}/report.json",
                     f"runs/{RUN}/config.json", f"runs/{RUN}/experiment.json",
                     f"runs/{analysis}/effect-sizes.csv"):
        with open(os.path.join(str(root), relative), "rb") as handle:
            data = handle.read()
        assert sources[relative]["sha256"] == hashlib.sha256(data).hexdigest(), relative
        assert sources[relative]["bytes"] == len(data)
    assert not any(os.path.isabs(path) for path in sources)
    # And to the files it wrote.
    for entry in manifest["files"]:
        with open(os.path.join(result["exportDirectory"], *entry["file"].split("/")), "rb") as handle:
            assert entry["sha256"] == hashlib.sha256(handle.read()).hexdigest()
    assert {entry["file"] for entry in manifest["files"]} | {"manifest.json"} == {
        entry["file"] for entry in result["files"]}
    with open(os.path.join(str(root), "runs", RUN, "experiment-hash.txt"), encoding="utf-8") as handle:
        assert manifest["experimentHash"] == handle.read().strip()


# --- which run, and where to ---------------------------------------------------


def test_the_newest_completed_run_is_used_unless_another_is_named(root):
    manifest = _manifest()
    older = f"20261001T000000000-exp-{STUDY}-run"
    _write_run(root, manifest, _python_records("0" * 64)[:2], name=older)
    _write_run(root, manifest, _python_records("0" * 64)[:4], name=RUN)
    # Newer still, and none of them a completed run of this study: an
    # unfinished run, one shard of a run, and another study's run.
    _write_run(root, manifest, _python_records("0" * 64), complete=False,
               name=f"20261005T000000000-exp-{STUDY}-run")
    _write_run(root, manifest, _python_records("0" * 64),
               name=f"20261006T000000000-exp-{STUDY}-run-shard0of2")
    _write_run(root, {**manifest, "name": STUDY + "-pilot"}, _python_records("0" * 64),
               name=f"20261007T000000000-exp-{STUDY}-pilot-run")

    assert _export(root)["run"] == "runs/" + RUN
    for spelling in (older, "runs/" + older, os.path.join(str(root), "runs", older)):
        result = _export(root, run=spelling)
        assert result["run"] == "runs/" + older
        assert len(_table(result, "responses.csv")) == 2


def test_an_analysis_or_evaluation_of_another_run_is_not_used(root):
    manifest = _manifest()
    older = f"20261001T000000000-exp-{STUDY}-run"
    _write_run(root, manifest, _python_records("0" * 64), name=older)
    _write_run(root, manifest, _python_records("0" * 64), name=RUN)
    _python_analysis_by_hand(root, _python_effect_rows(), run=older)
    _write_python_paired_evaluation(root, run=older)

    newest = _export(root)
    assert (newest["analysis"], newest["evaluations"]["paired"]) == (None, None)
    named = _export(root, run=older)
    assert (named["analysis"], named["evaluations"]["paired"]) == (
        "runs/" + ANALYSIS, "runs/" + EVALUATION)


def test_a_re_measured_run_exports_under_the_study_that_re_measured_it(root):
    """The sanctioned way to judge a run again is to duplicate the study and
    evaluate the copy against the original run. That run then exports under
    the copy's name, and the summary says whose run it is."""
    original = _manifest()
    _write_run(root, original, _python_records("0" * 64))
    copy = STUDY + "-recoded"
    rows = _python_judgments()
    directory = os.path.join(str(root), "runs", f"20261004T030000000-exp-{copy}-evaluate")
    _write_lines(os.path.join(directory, "judgments.jsonl"), rows)
    _write_json(os.path.join(directory, "judge-report.json"),
                {"experiment": copy, "sourceRun": RUN, "judges": []})

    result = results_export.export_results(str(root), copy, run="runs/" + RUN)
    assert len(_table(result, "judgments.csv")) == len(rows)
    assert (f"The run was generated under the study `{STUDY}`. It is exported here with "
            f"the analysis and evaluation made under `{copy}`.") in _read(result, "methods.md")

    with pytest.raises(results_export.ResultsExportRefusal) as refusal:
        results_export.export_results(str(root), "unrelated", run="runs/" + RUN)
    assert refusal.value.code == results_export.RUN_NOT_USABLE_CODE
    assert f"belongs to study '{STUDY}'" in refusal.value.reason
    assert refusal.value.repair_action == (
        f"Export it under its own study: steerlab results export {STUDY} --run runs/{RUN}")


def test_an_export_is_written_to_a_new_folder_and_never_over_another(root):
    _write_run(root, _manifest(), _python_records("0" * 64))
    first, second = _export(root), _export(root)
    assert first["exportDirectory"] != second["exportDirectory"]
    assert os.path.dirname(second["exportDirectory"]) == os.path.join(str(root), "exports")

    chosen = os.path.join(str(root), "for-my-paper")
    assert _export(root, out=chosen)["exportDirectory"] == chosen
    assert _export(root, out="relative-to-the-workspace")["exportDirectory"] == os.path.join(
        str(root), "relative-to-the-workspace")
    before = _tree(chosen)
    with pytest.raises(results_export.ResultsExportRefusal) as refusal:
        _export(root, out=chosen)
    assert refusal.value.code == results_export.DESTINATION_REFUSED_CODE
    assert "already contains files" in refusal.value.reason
    assert _tree(chosen) == before


@pytest.mark.parametrize("destination", ["runs", "runs/new-folder", "runs/" + RUN + "/export"])
def test_an_export_into_runs_is_refused_and_writes_nothing(root, destination):
    _write_run(root, _manifest(), _python_records("0" * 64))
    before = _tree(os.path.join(str(root), "runs"))
    with pytest.raises(results_export.ResultsExportRefusal) as refusal:
        _export(root, out=os.path.join(str(root), destination))
    assert refusal.value.code == results_export.DESTINATION_REFUSED_CODE
    assert "Run directories are never modified" in refusal.value.reason
    assert _tree(os.path.join(str(root), "runs")) == before
    assert not os.path.exists(os.path.join(str(root), "exports"))


def test_a_symbolic_link_into_runs_is_refused_too(root, tmp_path):
    _write_run(root, _manifest(), _python_records("0" * 64))
    link = tmp_path / "shortcut"
    os.symlink(os.path.join(str(root), "runs"), str(link))
    with pytest.raises(results_export.ResultsExportRefusal):
        _export(root, out=str(link / "export"))


# --- refusals, in the words of the client that asked ---------------------------


@pytest.mark.parametrize("client, run, other", [
    (results_export.PYTHON_CLIENT, f"steerlab run {STUDY} --runner <url>", "steerlab-cli"),
    (results_export.MAC_CLIENT, f"steerlab-cli experiment run {STUDY}", "steerlab run"),
    (results_export.APP_CLIENT, "Run the study", "steerlab"),
])
def test_a_study_with_no_completed_run_is_refused_with_this_clients_repair(
        root, client, run, other):
    _write_study(root, _manifest())
    _write_run(root, _manifest(), _python_records("0" * 64), complete=False)

    with pytest.raises(results_export.ResultsExportRefusal) as refusal:
        _export(root, client=client)

    assert (refusal.value.code, refusal.value.state) == (
        results_export.NO_COMPLETED_RUN_CODE, "refused")
    assert refusal.value.reason == (
        f"Study '{STUDY}' has no completed run in this workspace yet, so there are no "
        "results to export. 1 run of it is unfinished: a run writes report.json when it "
        "completes, and this one has none.")
    assert refusal.value.repair_action.startswith(run + ", then ")
    assert other not in refusal.value.repair_action
    assert not os.path.exists(os.path.join(str(root), "exports"))


def test_an_unknown_study_is_not_found(root):
    with pytest.raises(results_export.ResultsExportRefusal) as refusal:
        _export(root)
    assert (refusal.value.code, refusal.value.state) == (results_export.NOT_FOUND_CODE, "notFound")
    assert refusal.value.repair_action.startswith("steerlab experiment list")


@pytest.mark.parametrize("name", ["", "../elsewhere", "a/b", ".hidden", " padded "])
def test_a_name_that_is_not_a_study_name_is_refused(root, name):
    with pytest.raises(results_export.ResultsExportRefusal) as refusal:
        results_export.export_results(str(root), name)
    assert (refusal.value.code, refusal.value.state) == (results_export.USAGE_CODE, "blocked")


def test_a_named_run_that_is_missing_or_unfinished_is_refused(root):
    _write_run(root, _manifest(), _python_records("0" * 64), complete=False)
    with pytest.raises(results_export.ResultsExportRefusal) as unfinished:
        _export(root, run="runs/" + RUN)
    assert unfinished.value.code == results_export.RUN_NOT_USABLE_CODE
    assert "it has no report.json" in unfinished.value.reason
    for missing in ("runs/no-such-run", "experiments", os.path.join(str(root), "..")):
        with pytest.raises(results_export.ResultsExportRefusal) as refusal:
            _export(root, run=missing)
        assert refusal.value.state == "notFound"


# --- the command line ----------------------------------------------------------


def _main(arguments, capsys):
    code = client_cli.main(list(arguments))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_results_export_answers_in_the_shared_envelope(root, capsys):
    _python_study(root)
    before = _tree(os.path.join(str(root), "runs"))
    capsys.readouterr()

    code, out, err = _main(["--root", str(root), "results", "export", STUDY, "--json"], capsys)

    assert code == 0, err
    document = json.loads(out)
    assert (document["state"], document["verb"], document["changed"]) == (
        "ready", "results export", True)
    result = document["result"]
    assert result["exportDirectory"].startswith(os.path.join(os.path.realpath(str(root)), "exports"))
    assert {"responses.csv", "effects.csv", "methods.md", "codebook.md", "manifest.json"} <= {
        entry["file"] for entry in result["files"]}
    assert "Nothing under runs/ was changed" in document["message"]
    assert _tree(os.path.join(str(root), "runs")) == before
    # The human summary went to stderr, so stdout holds the one document.
    assert "responses.csv  (16 rows)" in err


def test_out_names_the_export_folder_not_the_envelope_file(root, tmp_path, capsys, monkeypatch):
    _write_run(root, _manifest(), _python_records("0" * 64))
    monkeypatch.chdir(tmp_path)

    code, out, _ = _main(["--root", str(root), "results", "export", STUDY,
                          "--out", "taken-elsewhere", "--run", RUN, "--json"], capsys)

    assert code == 0
    destination = os.path.join(os.path.realpath(str(tmp_path)), "taken-elsewhere")
    assert os.path.realpath(json.loads(out)["result"]["exportDirectory"]) == destination
    assert os.path.isfile(os.path.join(destination, "responses.csv"))


def test_the_synopsis_shows_a_folder_and_both_optional_flags(capsys):
    spec = client_cli.spec_for("results", "export")
    assert client_cli.synopsis(spec) == (
        "steerlab results export <study> [--out <dir>] [--run <run-dir>]")
    assert client_cli.main(["results", "--help"]) == 0
    assert "steerlab results export <study>" in capsys.readouterr().out
    # Every other verb's --out is still a file.
    assert "[--out <file>]" in client_cli.synopsis(client_cli.spec_for("bundle", "package"))


def test_command_line_refusals_are_typed(root, capsys):
    code, out, err = _main(["--root", str(root), "results", "export", STUDY, "--json"], capsys)
    document = json.loads(out)
    assert (code, document["state"], document["error"]["code"]) == (66, "notFound", "notFound")

    _write_study(root, _manifest())
    code, out, err = _main(["--root", str(root), "results", "export", STUDY, "--json"], capsys)
    document = json.loads(out)
    assert (code, document["state"], document["error"]["code"]) == (
        65, "refused", results_export.NO_COMPLETED_RUN_CODE)
    assert document["error"]["repairAction"].startswith(f"steerlab run {STUDY} --runner <url>")
    assert document["changed"] is False

    code, out, _ = _main(["--root", str(root), "results", "export", "--json"], capsys)
    assert (code, json.loads(out)["error"]["code"]) == (64, "usage")
    code, out, _ = _main(["results", "export", STUDY, "--json"], capsys)
    assert json.loads(out)["error"]["code"] == client_cli.WORKSPACE_UNSET_CODE


# --- the bridge the Mac command line and the app use ---------------------------


def test_the_bridge_action_runs_the_same_export(root):
    _python_study(root)
    result = diagnostic_commands.workspace_action(
        results_commands.BRIDGE_ACTION, {"workspaceRoot": str(root), "study": STUDY})
    assert result["exported"] is True and result["changed"] is True
    assert os.path.isfile(os.path.join(result["exportDirectory"], "effects.csv"))


def test_the_bridge_returns_a_refusal_as_data_in_the_mac_clients_words(root):
    """A refusal crosses the bridge with its code and state, so the Mac
    command line can answer as this client does instead of with one flat
    failure. Its repair names the Mac command line by default."""
    _write_study(root, _manifest())
    result = diagnostic_commands.workspace_action(
        results_commands.BRIDGE_ACTION, {"workspaceRoot": str(root), "study": STUDY})
    assert result == {"exported": False, "changed": False, "refusal": {
        "code": results_export.NO_COMPLETED_RUN_CODE, "state": "refused",
        "reason": f"Study '{STUDY}' has no completed run in this workspace yet, so "
                  "there are no results to export.",
        "repairAction": f"steerlab-cli experiment run {STUDY}, then steerlab-cli results "
                        f"export {STUDY}. A run that finished on other hardware has to be "
                        "brought into this workspace before it can be exported."}}
    app = diagnostic_commands.workspace_action(results_commands.BRIDGE_ACTION, {
        "workspaceRoot": str(root), "study": STUDY, "client": results_export.APP_CLIENT})
    assert app["refusal"]["repairAction"].startswith("Run the study, then choose Export Results again.")


@pytest.mark.parametrize("payload", [
    {"study": STUDY}, {"workspaceRoot": "/tmp/x"}, {"workspaceRoot": "/tmp/x", "study": ""},
    {"workspaceRoot": "/tmp/x", "study": STUDY, "force": "yes"},
    {"workspaceRoot": "/tmp/x", "study": STUDY, "run": 3}])
def test_the_bridge_refuses_a_malformed_request(payload):
    with pytest.raises(ValueError):
        diagnostic_commands.workspace_action(results_commands.BRIDGE_ACTION, payload)


def test_the_bridge_process_exports_end_to_end(root):
    """The exact process the Mac side starts: JSON on stdin, one JSON answer on
    stdout, the source identity checked first."""
    from steerlab_server.client.runtime_identity import source_sha256
    _python_study(root)
    identity = source_sha256()
    request = {"action": results_commands.BRIDGE_ACTION, "clientSHA256": identity,
               "payload": {"workspaceRoot": str(root), "study": STUDY,
                           "client": results_export.MAC_CLIENT}}
    process = subprocess.run(
        [sys.executable, "-B", "-s", "-m", "steerlab_server.client.diagnostic_workspace"],
        input=json.dumps(request), capture_output=True, text=True, check=False,
        cwd=str(root), env={**os.environ, "PYTHONPATH": SERVER_DIR})
    assert process.returncode == 0, process.stderr
    answer = json.loads(process.stdout)
    assert answer["ok"] is True and answer["clientSHA256"] == identity
    assert answer["result"]["exported"] is True
    assert os.path.isfile(os.path.join(answer["result"]["exportDirectory"], "methods.md"))


def test_an_export_needs_nothing_but_the_standard_library(root):
    """Out of process, because half the suite has already imported torch. An
    export has to work on an install that carries no engine at all."""
    _write_run(root, _manifest(), _python_records("0" * 64))
    probe = (
        "import json, sys\n"
        "from steerlab_server import client_cli\n"
        f"code = client_cli.main(['--root', {str(root)!r}, 'results', 'export', {STUDY!r}, '--json'])\n"
        "heavy = sorted(m for m in ('torch', 'transformers', 'numpy', 'scipy', 'fastapi') if m in sys.modules)\n"
        "third = sorted({m.split('.')[0] for m in sys.modules if not m.startswith('_')}\n"
        "               - set(sys.stdlib_module_names))\n"
        "sys.stderr.write('PROBE' + json.dumps({'code': code, 'heavy': heavy, 'third': third}))\n")
    process = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                             check=False, cwd=str(root),
                             env={**os.environ, "PYTHONPATH": SERVER_DIR})
    report = json.loads(process.stderr.split("PROBE")[-1])
    assert report == {"code": 0, "heavy": [], "third": ["steerlab_server"]}, process.stderr

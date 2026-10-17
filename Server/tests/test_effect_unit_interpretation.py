"""What an effect row's ``n`` counts, checked against answers written by hand.

An analysis written before SteerLab 0.9.8 left the ``unit`` column of its
pooled rows empty, so a reader has to say what one paired difference is. The
cross-engine fixture (``Tests/Fixtures/cross-engine/effect-units.json``) pins
the Mac app and the results explorer to this reader; it proves the three agree,
not that they are right. This table is what proves them right. Every expected
unit, source, and sentence below was written from what the engines actually
did, independently of the code under test:

* the Python engine has always averaged an item's samples within each
  condition, so its pooled rows count items;
* the Mac engine has done the same since 0.9.7;
* before 0.9.7 the Mac engine paired every response of a study with several
  responses per item with the baseline response to the same item and seed, so
  its ``n`` counted responses; its ordinal readouts, one per item and
  condition, still paired by item;
* when every item has one response per condition, items and responses are the
  same thing.

A row counting fewer pairs than the run's items does not show it counts items:
declared exclusions, missing values, or a subgroup leave five responses from
one item looking like five items. Such a row is called unknown unless
something else settles it, and its sentence says the analysis did not record
what its pairs count. Nothing stored is changed or recomputed.

Each case builds a run and its analysis in a temporary workspace and reads them
through the real export and results page.
"""

import json
import os
import re

import pytest

from steerlab_server.client import results_export
from steerlab_server.client.reports import study_results
from steerlab_server.experiment.manifest import Manifest

from test_results_export import (  # noqa: F401 - the workspace fixture is used by name
    ANALYSIS, MAC_EFFECTS_HEADER, RUN, STUDY, _export, _manifest, _pinned_clock, _read, _table,
    _write_json, _write_run, _write_study, root)

MAC_096 = {"substrate": "swift-mlx", "appVersion": "swift-app 0.9.6+0a1b2c3d"}
MAC_097 = {"substrate": "swift-mlx", "appVersion": "swift-app 0.9.7+0a1b2c3d"}
MAC_097_PRERELEASE = {"substrate": "swift-mlx", "appVersion": "swift-app 0.9.7-beta+0a1b2c3d"}
MAC_UNVERSIONED = {"substrate": "swift-mlx"}
PYTHON = {"substrate": "python-hf-transformers", "appVersion": "steerlab-server 0.9.6+0a1b2c3d"}

RESPONSES_NOTE = ("These are responses, not items: the analysis paired each response with the baseline response "
                  "to the same item and seed, so its interval and test treat responses to the same item as "
                  "independent and are not findings about items")
UNKNOWN_NOTE = "The analysis that made this row did not record whether its pairs count items or responses"

#: The opening of each methods-summary line, by how the unit is known.
LINE = {
    "recorded item": "the item, with an item's samples averaged within each condition, as the analysis recorded.",
    "recorded transcript": "the transcript, as the analysis recorded:",
    "one sample": ("the item. The analysis did not record its unit, but the run's records answer each item only "
                   "once in each condition compared, so here an item and a response are the same thing."),
    "engine item": ("the item, with an item's samples averaged within each condition. The analysis did not "
                    "record its unit, but its config.json names the engine that made it, and that engine paired "
                    "these rows by item."),
    "engine response": ("the response, not the item. The analysis did not record its unit, but its config.json "
                        "names the Mac engine from before SteerLab 0.9.7,"),
    "more pairs than items": "the response, not the item. These rows count more pairs than the run has paired items,",
    "unknown": ("not established. The analysis that made these rows did not record whether their pairs count "
                "items or responses,"),
    "unknown, no records": ("not established. The analysis that made these rows did not record whether their "
                            "pairs count items or responses, and neither its config.json nor the run's records "
                            "settle it: fewer pairs than items can mean items, or responses from fewer items. The "
                            "numbers are shown as stored; analyzing the run again with this version records the "
                            "unit. The run's records are not available here, so they could not be checked."),
}

# Each case: the run's records (items, samples per item, and whether each item
# has one instrument readout per condition), the analysis's config.json, the
# stored row, and what a reader must say about it.
CASES = [
    # The external reviewer's case: ten items, five samples, two conditions; a
    # declared exclusion kept one item, so the Mac engine before 0.9.7 stored
    # n = 5, all five responses from that one item.
    dict(label="the reviewer's exclusion case, made by the Mac engine before 0.9.7",
         items=10, samples=5, config=MAC_096, n=5,
         unit="response", source="analysis_engine", too_few=False, line="engine response",
         says="across 5 paired responses from at most 10 items. " + RESPONSES_NOTE),
    dict(label="the reviewer's exclusion case, with no engine version recorded",
         items=10, samples=5, config=MAC_UNVERSIONED, n=5,
         unit="unknown", source="not_established", too_few=False, line="unknown",
         says="across 5 pairs. " + UNKNOWN_NOTE),
    dict(label="the reviewer's exclusion case, with no config.json at all",
         items=10, samples=5, config=None, n=5,
         unit="unknown", source="not_established", too_few=False, line="unknown",
         says="across 5 pairs. " + UNKNOWN_NOTE),
    # A subgroup row: five items of a twenty-item run, three samples each, so
    # 15 responses. Every engine stamps its subgroup rows' unit, so an
    # unstamped one is not settled by the engine named in config.json. The
    # page lists subgroup rows in a table, whose note says what unknown means.
    dict(label="a subgroup row within the run's items",
         items=20, samples=3, config=MAC_096, n=15, stratify_by="arm", stratum="x",
         unit="unknown", source="not_established", too_few=False,
         says="Where the unit is unknown, the analysis that made the row did not record whether its pairs "
              "count items or responses."),
    dict(label="a subgroup row with more pairs than the run has items",
         items=20, samples=3, config=None, n=61, stratify_by="arm", stratum="x",
         unit="response", source="inferred_from_records", too_few=False,
         says="61 response (from the records)"),
    # Missing endpoint values: six items, three samples, and only four values
    # reached the analysis.
    dict(label="missing values, made by the Mac engine before 0.9.7",
         items=6, samples=3, config=MAC_096, n=4,
         unit="response", source="analysis_engine", too_few=False, line="engine response",
         says="across 4 paired responses from at most 6 items. " + RESPONSES_NOTE),
    dict(label="missing values, made by the Python engine",
         items=6, samples=3, config=PYTHON, n=4,
         unit="item", source="analysis_engine", too_few=False, line="engine item",
         says="across 4 paired items (interval"),
    dict(label="missing values in a run with one sample per item",
         items=6, samples=1, config=None, n=4,
         unit="item", source="one_sample_per_item", too_few=False, line="one sample",
         says="across 4 paired items (interval"),
    dict(label="a run with one sample per item, made by the Mac engine before 0.9.7",
         items=4, samples=1, config=MAC_096, n=4,
         unit="item", source="one_sample_per_item", too_few=False, line="one sample",
         says="across 4 paired items (interval"),
    dict(label="an unstamped row made by the Python engine",
         items=4, samples=2, config=PYTHON, n=4,
         unit="item", source="analysis_engine", too_few=False, line="engine item",
         says="across 4 paired items (interval"),
    dict(label="an unstamped row made by the Mac engine 0.9.7",
         items=4, samples=2, config=MAC_097, n=4,
         unit="item", source="analysis_engine", too_few=False, line="engine item",
         says="across 4 paired items (interval"),
    dict(label="an unstamped row made by the Mac engine before 0.9.7",
         items=4, samples=2, config=MAC_096, n=8,
         unit="response", source="analysis_engine", too_few=False, line="engine response",
         says="across 8 paired responses from at most 4 items. " + RESPONSES_NOTE),
    dict(label="an ordinal readout row made by the Mac engine before 0.9.7",
         items=4, samples=2, readouts=True, config=MAC_096, n=4, metric="ordinalPosition",
         unit="item", source="analysis_engine", too_few=False, line="engine item",
         says="across 4 paired items (interval"),
    dict(label="a pre-release of 0.9.7, which could be either side of the change",
         items=4, samples=2, config=MAC_097_PRERELEASE, n=4,
         unit="unknown", source="not_established", too_few=False, line="unknown",
         says="across 4 pairs. " + UNKNOWN_NOTE),
    dict(label="a row that records its unit",
         items=4, samples=2, config=MAC_096, n=4, row_unit="item",
         unit="item", source="recorded", too_few=False, line="recorded item",
         says="across 4 paired items (interval"),
    dict(label="a multi-agent analysis that stamps transcripts",
         items=4, samples=2, config=MAC_097, n=4, stamped="transcript",
         unit="transcript", source="recorded", too_few=False, line="recorded transcript",
         says="across 4 paired transcripts (interval"),
    dict(label="a study of instrument readouts alone",
         items=4, samples=0, readouts=True, config=None, n=4, metric="choiceLogOdds",
         unit="item", source="one_sample_per_item", too_few=False, line="one sample",
         says="across 4 paired items (interval"),
    # A run whose records were not kept: nothing to check, and the line says
    # this reading could not check them.
    dict(label="a run with no records to check",
         items=0, samples=0, config=None, n=4,
         unit="unknown", source="not_established", too_few=False, line="unknown, no records",
         says="across 4 pairs. " + UNKNOWN_NOTE),
    dict(label="more pairs than items, with no engine version recorded",
         items=1, samples=5, config=MAC_UNVERSIONED, n=5,
         unit="response", source="inferred_from_records", too_few=True, line="more pairs than items",
         says="across 5 paired responses from 1 item. " + RESPONSES_NOTE
              + ". That is too few items for an interval or a test"),
]


def _records(case, manifest_hash):
    records = []
    for condition in ("baseline", "formal"):
        for index in range(case["items"]):
            item = f"item-{index + 1}"
            for seed in range(case["samples"]):
                records.append({"experiment": STUDY, "experimentHash": manifest_hash, "condition": condition,
                                "seed": seed, "promptIndex": index, "promptID": item,
                                "prompt": f"Question {item}.", "output": f"An answer to {item}.",
                                "wordCount": 4, "distinct2": 0.5, "finishReason": "stop"})
            if case.get("readouts"):
                records.append({"experiment": STUDY, "condition": condition, "promptIndex": index,
                                "promptID": item, "prompt": f"Question {item}.",
                                "instrument": "ordinalScale", "ordinalPosition": 2.0})
    return records


def _workspace(root, case):
    """The run, and an analysis of it in the Mac engine's layout, with the
    config.json the case names."""
    manifest = _manifest(samplesPerItem=max(case["samples"], 1), seeds=list(range(max(case["samples"], 1))))
    _write_study(root, manifest)
    manifest_hash = Manifest.from_dict(manifest).content_hash()
    _write_run(root, manifest, _records(case, manifest_hash), engine="mac")
    directory = os.path.join(str(root), "runs", ANALYSIS)
    _write_json(os.path.join(directory, "experiment.json"), manifest)
    if case["config"] is not None:
        _write_json(os.path.join(directory, "config.json"), {
            "schemaVersion": 4, "runId": ANALYSIS, "runType": "analyze", "experiment": STUDY,
            "experimentHash": manifest_hash, **case["config"]})
    _write_json(os.path.join(directory, "analysis.json"), {
        "experiment": STUDY, "experimentHash": manifest_hash, "sourceRun": RUN})
    if case.get("stamped"):
        _write_json(os.path.join(directory, "unit-of-analysis.json"), {"unitOfAnalysis": case["stamped"]})
    row = [  # the Mac engine's columns, MAC_EFFECTS_HEADER
        "formal", case.get("metric", "wordCount"), str(case["n"]), "2.0", "1.5", "2.5", "1.0", "0.04", "",
        "", case.get("stratify_by", "pooled"), case.get("stratum", ""), case.get("row_unit", ""), "", ""]
    with open(os.path.join(directory, "effect-sizes.csv"), "w", encoding="utf-8") as handle:
        handle.write(MAC_EFFECTS_HEADER + "\n" + ",".join(row) + "\n")


def _text(markup):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", markup)).replace("’", "'")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["label"])
def test_each_stored_row_is_read_as_the_engines_wrote_it(case, root):
    _workspace(root, case)
    result = _export(root)
    [row] = _table(result, "effects-by-stratum.csv" if "stratify_by" in case else "effects.csv")
    assert (row["n_pairs"], row["unit_of_analysis"], row["unit_of_analysis_source"]) == (
        str(case["n"]), case["unit"], case["source"])
    # The stored numbers are copied unchanged, whatever the unit.
    assert (row["estimate"], row["ci_lower"], row["ci_upper"]) == ("2.0", "1.5", "2.5")

    stored = results_export.read_results(str(root), STUDY)
    [settled] = stored["context"]["effectRows"]
    assert study_results.Page.too_few(settled) is case["too_few"]
    page = _text(study_results.render(stored))
    assert case["says"] in page
    if case["unit"] != "response":
        assert "paired responses" not in page
    if case["unit"] not in ("item", "transcript"):
        assert not re.search(r"across \d+ paired items?\b", page)
    if "line" in case:
        assert "Unit of analysis: " + LINE[case["line"]] in _read(result, "methods.md")
        assert "Unit of analysis: " + LINE[case["line"]] in page


def test_the_table_reaches_every_way_a_unit_is_known():
    """Every source value and every unit a reader can answer appears in the
    hand-written table above."""
    assert {case["source"] for case in CASES} == {
        "recorded", "one_sample_per_item", "analysis_engine", "inferred_from_records", "not_established"}
    assert {case["unit"] for case in CASES} == {"item", "transcript", "response", "unknown"}


def test_the_codebook_defines_every_source_value(root):
    _workspace(root, CASES[0])
    codebook = _read(_export(root), "codebook.md")
    section = codebook.split("unit_of_analysis_source", 1)[1].split("\n\n", 1)[0]
    for value in ("recorded", "one_sample_per_item", "analysis_engine", "inferred_from_records",
                  "not_established"):
        assert value + ":" in section, value
    assert "engine_default" not in codebook


@pytest.mark.parametrize("config, pairing", [
    ({"substrate": "python-hf-transformers"}, "items"),
    ({"substrate": "python-hf-transformers", "appVersion": "steerlab-server 0.1.0"}, "items"),
    ({"substrate": "swift-mlx", "appVersion": "swift-app 0.9.7"}, "items"),
    ({"substrate": "swift-mlx", "appVersion": "swift-app 0.9.8+0a1b2c3d"}, "items"),
    ({"substrate": "swift-mlx", "appVersion": "swift-app 1.0.0-beta+0a1b2c3d"}, "items"),
    ({"substrate": "swift-mlx", "appVersion": "swift-app 0.10.0"}, "items"),
    ({"substrate": "swift-mlx", "appVersion": "swift-app 0.9.6"}, "responses"),
    ({"substrate": "swift-mlx", "appVersion": "swift-app 0.9.0-dev+abc12345"}, "responses"),
    ({"substrate": "swift-mlx", "appVersion": "swift-app 0.9.7-dev+abc12345"}, None),
    ({"substrate": "swift-mlx", "appVersion": "swift-app (fixture)"}, None),
    ({"substrate": "swift-mlx", "appVersion": 97}, None),
    ({"substrate": "swift-mlx"}, None),
    ({"substrate": "something-else", "appVersion": "swift-app 0.9.8"}, None),
    ({}, None),
    (None, None),
    ([], None),
])
def test_which_engine_paired_items_is_read_from_config_json(config, pairing):
    assert results_export.engine_pairing(config) == pairing


def test_a_stored_file_is_never_changed(root):
    case = CASES[0]
    _workspace(root, case)
    before = {}
    for directory, _, files in os.walk(os.path.join(str(root), "runs")):
        for name in files:
            with open(os.path.join(directory, name), "rb") as handle:
                before[os.path.join(directory, name)] = handle.read()
    _export(root)
    study_results.render(results_export.read_results(str(root), STUDY))
    after = {}
    for directory, _, files in os.walk(os.path.join(str(root), "runs")):
        for name in files:
            with open(os.path.join(directory, name), "rb") as handle:
                after[os.path.join(directory, name)] = handle.read()
    assert before == after
    assert json.loads(before[os.path.join(str(root), "runs", ANALYSIS, "config.json")])["appVersion"] == (
        "swift-app 0.9.6+0a1b2c3d")

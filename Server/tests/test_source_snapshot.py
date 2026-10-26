"""Which records of its source run an analysis read (``source_snapshot``).

Review finding (October 23, 2026): an analysis of a run parked at a
checkpoint (6 over two pairs) was later exported as the analysis of the
completed run (3 over four pairs), because an analysis named its source by
directory alone and a continuation appends to the same directory. These tests
hold the repair:

* the snapshot, its status, and the comparison answer the hand-authored
  cross-engine contract (the Swift suite reads the same fixture);
* an analysis stamps the exact bytes it parsed and the source's status;
* the default source is the newest COMPLETED run, and an unfinished one is
  analyzed only when none has completed, or when named, always with a
  warning (and an envelope advisory from the CLI);
* checkpoint → analyze → continue → export: the export does not present the
  checkpoint's 6 as the completed run's effect, says what it is, and a fresh
  analysis gives 3 over four pairs, which the export then copies.

All data is synthetic; the analysis, status, settlement, and export owners are
the real ones. No model runs.
"""

import csv
import json
import os

import pytest

from steerlab_server.client import results_export
from steerlab_server.experiment import run_status, source_snapshot, tasks
from steerlab_server.experiment.manifest import Manifest
from test_results_export import STUDY, _manifest, _write_run, _write_study

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine", "source-snapshot.json")
RUN = f"20261004T000000000-exp-{STUDY}-run"
NEWER = f"20261005T000000000-exp-{STUDY}-run"


@pytest.fixture
def root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    return workspace


def _fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


# --- the shared contract -------------------------------------------------------


@pytest.mark.parametrize("case", _fixture()["describe"], ids=lambda case: case["label"])
def test_describe_matches_the_cross_engine_contract(case):
    assert source_snapshot.describe(case["generations"].encode("utf-8")) == case["expected"]


@pytest.mark.parametrize("case", _fixture()["status"], ids=lambda case: case["label"])
def test_status_matches_the_cross_engine_contract(case, tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    if case.get("statusFileText") is not None:
        (directory / run_status.STATUS_FILENAME).write_text(case["statusFileText"])
    elif case["statusFile"] is not None:
        (directory / run_status.STATUS_FILENAME).write_text(json.dumps(case["statusFile"]))
    if case["report"]:
        (directory / "report.json").write_text("{}")
    assert source_snapshot.status(str(directory)) == case["expected"]
    assert source_snapshot.was_continued(str(directory)) is case["continued"]


@pytest.mark.parametrize("case", _fixture()["verdicts"], ids=lambda case: case["label"])
def test_verdicts_and_sentences_match_the_cross_engine_contract(case):
    found = source_snapshot.verdict(case["stamp"], case["now"], continued=case["continued"])
    assert found["state"] == case["expected"]["state"]
    assert source_snapshot.sentence(found, "R") == case["expected"]["sentence"]


def test_a_stamp_missing_a_key_or_of_the_wrong_type_is_not_read(tmp_path):
    """As strict as the Swift decoder: half a stamp is no stamp."""
    stamp = dict(_fixture()["verdicts"][0]["stamp"])
    for broken in ({k: v for k, v in stamp.items() if k != "records"},
                   {**stamp, "records": "8"}, {**stamp, "records": True}):
        (tmp_path / source_snapshot.STAMP_FILE).write_text(json.dumps(broken))
        assert source_snapshot.read_stamp(str(tmp_path)) is None
    (tmp_path / source_snapshot.STAMP_FILE).write_text(json.dumps(stamp))
    assert source_snapshot.read_stamp(str(tmp_path)) == stamp


# --- a study whose run can be parked, analyzed, and continued ------------------


def _records(manifest_hash, diffs):
    """One sample per item; ``formal`` minus ``baseline`` word count is the
    item's diff, so the effect is the mean of ``diffs``."""
    rows = []
    for item, diff in diffs:
        for condition in ("baseline", "formal"):
            words = 10 + (diff if condition == "formal" else 0)
            rows.append({
                "experiment": STUDY, "experimentHash": manifest_hash,
                "modelID": "test/model", "modelRevision": "rev-1",
                "promptMode": "chatAssistant", "condition": condition,
                "seed": 1000, "seedPolicy": "derivedSHA256", "sampleIndex": 0,
                "promptIndex": int(item.split("-")[1]) - 1, "promptID": item,
                "prompt": f"Describe a quiet weekend ({item}).",
                "interventionState": {"slots": []},
                "output": " ".join(["word"] * words), "wordCount": words,
                "distinct2": 0.5, "finishReason": "stop", "temperature": 0.7,
                "doSample": True, "topP": 0.9, "topK": None,
                "dtype": "torch.float32", "device": "cpu",
                "engine": "python-hf-transformers"})
    return rows


def _study(root):
    manifest = _manifest(samplesPerItem=1)
    _write_study(root, manifest)
    return manifest, Manifest.from_dict(manifest).content_hash()


def _parked_run(root, manifest, manifest_hash, *, name=RUN):
    """A run parked at a checkpoint after two items: four records, no report."""
    directory = _write_run(root, manifest, _records(manifest_hash, [("item-1", 6), ("item-2", 6)]),
                           name=name, complete=False)
    run_status.RunStatus(directory, stage="run", experiment=STUDY).checkpointed("walltime")
    return directory


def _continue(directory, manifest_hash):
    """The continuation: two more items appended, the report written, and the
    park settled, exactly as a resumed run finishes."""
    with open(os.path.join(directory, "generations.jsonl"), "a", encoding="utf-8") as handle:
        for row in _records(manifest_hash, [("item-3", 0), ("item-4", 0)]):
            handle.write(json.dumps(row) + "\n")
    with open(os.path.join(directory, "report.json"), "w", encoding="utf-8") as handle:
        json.dump({"experiment": STUDY, "experimentHash": manifest_hash, "conditions": {},
                   "truncation": {"classified": 8, "lengthStopped": 0}}, handle)
    assert run_status.settle_after_resumed_completion(
        directory, completion_file="report.json", items_written=8)


def _word_count_effect(directory):
    with open(os.path.join(directory, "effect-sizes.csv"), encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["stratifyBy"] == "pooled" and row["endpoint"] == "wordCount"]
    assert len(rows) == 1
    return float(rows[0]["deltaMean"]), int(rows[0]["n"])


def _stamp(directory):
    with open(os.path.join(directory, source_snapshot.STAMP_FILE), encoding="utf-8") as handle:
        return json.load(handle)


# --- the analysis --------------------------------------------------------------


def test_an_analysis_stamps_the_bytes_it_read_and_the_sources_status(root):
    manifest, manifest_hash = _study(root)
    directory = _write_run(root, manifest, _records(manifest_hash, [("item-1", 6), ("item-2", 6)]))
    analysis = tasks.analyze(STUDY, root=str(root), log=lambda _: None)
    with open(os.path.join(directory, "generations.jsonl"), "rb") as handle:
        described = source_snapshot.describe(handle.read())
    assert _stamp(analysis) == {"schemaVersion": 1, "sourceRun": RUN, "status": "completed",
                                **described}
    assert described["records"] == 4
    assert source_snapshot.compare(analysis, directory)["state"] == source_snapshot.MATCHES


def test_the_default_source_is_the_newest_completed_run_not_a_newer_unfinished_one(root):
    manifest, manifest_hash = _study(root)
    _write_run(root, manifest, _records(manifest_hash, [("item-1", 6), ("item-2", 6),
                                                        ("item-3", 0), ("item-4", 0)]))
    _parked_run(root, manifest, manifest_hash, name=NEWER)
    said = []
    analysis = tasks.analyze(STUDY, root=str(root), log=said.append)
    assert _stamp(analysis)["sourceRun"] == RUN
    assert _stamp(analysis)["status"] == "completed"
    assert _word_count_effect(analysis) == (3.0, 4)
    assert not [line for line in said if "is paused" in line]


def test_with_no_completed_run_the_newest_is_analyzed_as_it_stands_and_said_so(root):
    manifest, manifest_hash = _study(root)
    _parked_run(root, manifest, manifest_hash)
    said = []
    analysis = tasks.analyze(STUDY, root=str(root), log=said.append)
    assert _stamp(analysis)["status"] == "checkpointed"
    assert _stamp(analysis)["records"] == 4
    assert [line for line in said if line.startswith("WARNING: source run")
            and "paused at a checkpoint" in line and "no run of" in line]


@pytest.mark.parametrize("park", ["checkpointed", "failed"])
def test_a_named_unfinished_source_is_analyzed_with_a_warning_and_stamped(root, park):
    manifest, manifest_hash = _study(root)
    directory = _write_run(root, manifest, _records(manifest_hash, [("item-1", 6), ("item-2", 6)]),
                           complete=False)
    status = run_status.RunStatus(directory, stage="run", experiment=STUDY)
    if park == "failed":
        status.fail(RuntimeError("the node went away"))
    else:
        status.checkpointed("walltime")
    said = []
    analysis = tasks.analyze(STUDY, root=str(root), source_run=directory, log=said.append)
    assert _stamp(analysis)["status"] == park
    assert [line for line in said if line.startswith("WARNING: source run")]
    # The source run itself is untouched: its status and failure record stay.
    assert run_status.read_status(directory)["status"] == park


def test_the_cli_says_an_unfinished_source_was_analyzed_in_its_envelope(root, capsys, monkeypatch):
    from steerlab_server import cli
    manifest, manifest_hash = _study(root)
    _parked_run(root, manifest, manifest_hash)
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    code = cli.main(["experiment", "analyze", STUDY, "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert code == 0, envelope
    advisories = {a["code"]: a["detail"] for a in envelope["advisories"]}
    assert advisories["unfinishedSourceAnalyzed"] == (
        f"This analysis read run {RUN} while it was paused at a checkpoint, with 4 "
        "records. Re-analyze the run for effects over all its records.")


# --- checkpoint → analyze → continue → export ------------------------------------


def test_a_checkpoints_analysis_is_never_exported_as_the_completed_runs(root):
    manifest, manifest_hash = _study(root)
    directory = _parked_run(root, manifest, manifest_hash)
    early = tasks.analyze(STUDY, root=str(root), source_run=directory, log=lambda _: None)
    assert _word_count_effect(early) == (6.0, 2)

    _continue(directory, manifest_hash)
    assert source_snapshot.status(directory) == "completed"

    # The only analysis is the checkpoint's: copied as it is, and said so —
    # in the flags, beside the effects, in the methods, and as not available.
    stale = results_export.export_results(str(root), STUDY)
    assert stale["analysis"] == "runs/" + os.path.basename(early)
    with open(os.path.join(stale["exportDirectory"], "manifest.json"), encoding="utf-8") as handle:
        exported = json.load(handle)
    assert exported["analysisSourceSnapshot"] == {
        "state": "unfinishedSource", "analyzedRecords": 4, "analyzedStatus": "checkpointed",
        "runRecords": 8}
    sentence = (f"This analysis read run {RUN} while it was paused at a checkpoint, with 4 "
                "records, and the run now holds 8. Re-analyze the run for effects over all "
                "its records.")
    with open(os.path.join(stale["exportDirectory"], "methods.md"), encoding="utf-8") as handle:
        assert "- Source records: " + sentence in handle.read()
    assert [entry for entry in stale["notAvailable"]
            if entry["what"] == "effects over the run's current records"
            and entry["why"] == sentence
            and entry["repair"].startswith(
                f"steerlab run {STUDY} --runner <url> --verb analyze --source runs/{RUN}")]

    # A fresh analysis reads the completed run, and the export copies it.
    final = tasks.analyze(STUDY, root=str(root), log=lambda _: None)
    assert _word_count_effect(final) == (3.0, 4)
    assert _stamp(final)["status"] == "completed" and _stamp(final)["records"] == 8
    current = results_export.export_results(str(root), STUDY)
    assert current["analysis"] == "runs/" + os.path.basename(final)
    with open(os.path.join(current["exportDirectory"], "effects.csv"), encoding="utf-8") as handle:
        words = [row for row in csv.DictReader(handle) if row["outcome"] == "wordCount"]
    assert [(float(row["estimate"]), row["n_pairs"]) for row in words] == [(3.0, "4")]
    with open(os.path.join(current["exportDirectory"], "manifest.json"), encoding="utf-8") as handle:
        assert json.load(handle)["analysisSourceSnapshot"]["state"] == "matches"
    assert not [entry for entry in current["notAvailable"]
                if entry["what"] == "effects over the run's current records"]


def test_a_matching_analysis_is_preferred_to_a_newer_one_of_an_earlier_snapshot(root):
    """An explicit analysis of the parked state made AFTER the final one is
    still not the run's analysis."""
    manifest, manifest_hash = _study(root)
    directory = _parked_run(root, manifest, manifest_hash)
    early = tasks.analyze(STUDY, root=str(root), source_run=directory, log=lambda _: None)
    _continue(directory, manifest_hash)
    final = tasks.analyze(STUDY, root=str(root), log=lambda _: None)
    # Make the early analysis the newest by name.
    renamed = os.path.join(os.path.dirname(early), f"20991231T000000000-exp-{STUDY}-analyze")
    os.rename(early, renamed)
    found = results_export._find_analysis_and_snapshot(str(root), STUDY, RUN)
    assert found[0] == os.path.basename(final)
    assert found[1]["state"] == source_snapshot.MATCHES


def test_a_legacy_analysis_of_a_continued_run_is_flagged_and_of_any_other_run_is_not(root):
    manifest, manifest_hash = _study(root)
    directory = _parked_run(root, manifest, manifest_hash)
    early = tasks.analyze(STUDY, root=str(root), source_run=directory, log=lambda _: None)
    os.remove(os.path.join(early, source_snapshot.STAMP_FILE))  # as before the stamp existed
    _continue(directory, manifest_hash)
    exported = results_export.export_results(str(root), STUDY)
    assert [entry for entry in exported["notAvailable"]
            if entry["what"] == "effects over the run's current records"
            and "may predate the continuation" in entry["why"]]

    # The same legacy analysis of a run that was never paused says nothing.
    os.remove(os.path.join(directory, run_status.STATUS_FILENAME))
    quiet = results_export.export_results(str(root), STUDY)
    assert not [entry for entry in quiet["notAvailable"]
                if entry["what"] == "effects over the run's current records"]

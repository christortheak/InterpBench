"""A multi-agent study is analyzed per conversation, on both engines.

A turn's value pairs with the baseline's at the same turn of the same
play-through; the turn differences are averaged within each transcript; and
the test runs over one value per transcript, so ``n`` counts transcripts.
``unit-of-analysis.json`` says so.

The shared fixture ``Tests/Fixtures/cross-engine/panel-transcript-analysis.json``
holds the records and this engine's reading of them; the Mac suite
(``PanelTranscriptAnalysisTests``) reads the same file. The expectations
below are worked out by hand from the fixture's turns.
"""

import csv
import json
import math
import os

import pytest

from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine", "panel-transcript-analysis.json")

REGENERATE = ("stale fixture — re-run "
              "`Server/.venv.nosync/bin/python "
              "scripts/regenerate-cross-engine-fixtures.py` and commit")

MANIFEST = {"name": "panel", "modelID": "test/model", "concepts": [],
            "taskPromptsFile": None, "studyKind": "multiAgent",
            "conditions": []}


def _fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _case(label):
    return next(case for case in _fixture()["cases"] if case["label"] == label)


def _analyze(root, records):
    experiment_dir = os.path.join(root, "experiments", "panel")
    os.makedirs(experiment_dir)
    with open(os.path.join(experiment_dir, "experiment.json"), "w",
              encoding="utf-8") as handle:
        json.dump(MANIFEST, handle)
    run_dir = os.path.join(root, "runs", "20261005T000000000-exp-panel-run")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "experiment-hash.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(Manifest.from_dict(MANIFEST).content_hash() + "\n")
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    out = tasks.analyze("panel", root=root, log=lambda _: None)
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with open(os.path.join(out, "unit-of-analysis.json"), encoding="utf-8") as handle:
        unit = json.load(handle)
    return rows, unit


def _two_sided_p(w, n, tie_correction=0.0):
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_correction / 48
    z = (w - mean + 0.5) / math.sqrt(variance)
    return min(1.0, 1 + math.erf(z / math.sqrt(2)))


def test_turns_are_averaged_within_each_transcript(tmp_path):
    # Word counts, configured minus baseline: transcript 0 gives 2, 4, 6
    # (mean 4); 1 gives -1, -2, 3 (mean 0); 2 gives 5, 3, 7 (mean 5); 3 gives
    # -1, -2, -3 (mean -2). Mean 7/4 over four transcripts; W = 1 over the
    # three nonzero values.
    rows, unit = _analyze(str(tmp_path), _case("four-transcripts")["records"])
    words = next(row for row in rows if row["endpoint"] == "wordCount")
    assert words["n"] == "4"
    assert float(words["deltaMean"]) == 1.75
    assert float(words["wilcoxonW"]) == 1.0
    assert float(words["wilcoxonP"]) == pytest.approx(_two_sided_p(1, 3), abs=1e-6)
    assert words["stratifyBy"] == "pooled" and words["unit"] == ""
    # distinct2: 0, 0.25, 0, -0.25 — mean 0; the two nonzero values tie.
    distinct = next(row for row in rows if row["endpoint"] == "distinct2")
    assert distinct["n"] == "4" and float(distinct["deltaMean"]) == 0.0
    assert float(distinct["wilcoxonW"]) == 1.5
    assert len(rows) == 2, "no strata for a clustered analysis"
    assert unit == {"unitOfAnalysis": "transcript",
                    "reason": "turns within a transcript are dependent; each "
                              "transcript is reduced to its mean paired "
                              "difference before testing",
                    "skippedForSingleTranscript": False}


def test_one_transcript_supports_no_interval(tmp_path):
    rows, unit = _analyze(str(tmp_path), _case("one-transcript")["records"])
    assert rows == []
    assert unit["skippedForSingleTranscript"] is True


@pytest.mark.parametrize("label", ["four-transcripts", "one-transcript"])
def test_the_shared_fixture_is_this_engines_reading(tmp_path, label):
    case = _case(label)
    rows, unit = _analyze(str(tmp_path), case["records"])
    assert unit == case["unitOfAnalysis"], REGENERATE
    assert [(row["condition"], row["endpoint"], int(row["n"]),
             float(row["deltaMean"])) for row in rows] == [
        (row["condition"], row["endpoint"], row["n"], row["deltaMean"])
        for row in case["effectRows"]], REGENERATE

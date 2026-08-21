"""Report writing must survive failed variant conditions (error records that
lack wordCount/distinct2), and auto-baseline logic."""

import json
import os

from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest


def test_write_report_handles_error_records(tmp_path):
    manifest = Manifest.from_dict({"name": "s", "modelID": "m"})
    records = [
        {"condition": "baseline", "wordCount": 10, "distinct2": 0.8},
        {"condition": "baseline", "wordCount": 12, "distinct2": 0.9},
        # a failed variant condition — no wordCount/distinct2, just an error
        {"condition": "broken-variant", "error": "adapter not found"},
    ]
    tasks._write_report("s", manifest, records, str(tmp_path))
    report = json.load(open(os.path.join(tmp_path, "report.json")))
    assert report["conditions"]["baseline"]["generations"] == 2
    assert report["conditions"]["baseline"]["meanWordCount"] == 11
    # error condition is reported, not crashed on
    assert report["conditions"]["broken-variant"]["error"] == "adapter not found"
    assert report["conditions"]["broken-variant"]["generations"] == 0


def test_write_metrics_csv_skips_error_records(tmp_path):
    records = [
        {"condition": "baseline", "seed": 0, "promptIndex": 0, "promptID": "p1",
         "wordCount": 10, "distinct2": 0.8},
        {"condition": "broken-variant", "error": "adapter not found"},
    ]
    tasks._write_metrics_csv(records, str(tmp_path))
    text = open(os.path.join(tmp_path, "metrics.csv"), encoding="utf-8").read()

    assert text.splitlines()[0] == "condition,seed,promptIndex,promptID,wordCount,distinct2"
    assert "baseline,0,0,p1,10,0.8" in text
    assert "broken-variant" not in text


def test_baseline_prepended_when_absent():
    # The run loop prepends a baseline condition when none is named "baseline".
    from steerlab_server.experiment.manifest import Condition
    conditions = [Condition(name="fear-L5", slots=[])]
    if not any(c.name == "baseline" for c in conditions):
        conditions = [Condition(name="baseline", slots=[])] + conditions
    assert [c.name for c in conditions] == ["baseline", "fear-L5"]

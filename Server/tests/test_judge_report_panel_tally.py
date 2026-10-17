"""A judge report's top-level tally covers every judge, not the first.

With several judges the Python engine used to repeat the FIRST judge's tally
and model at the top of judge-report.json, so a study whose judges disagreed
reported one judge's preferences as the report's own. The top level now
states the whole panel, as the Mac engine's report does: ``conditions`` adds
up every judge's verdicts, the model keys name every judge, and
``conditionsScope`` says so. Each judge's own tally stays in ``judges``.

Two judges disagree completely below: on both pairs the first prefers the
condition's response and the second the baseline's. The expected panel tally
is written by hand: 2 + 0 baseline wins, 0 + 2 condition wins, 4 verdicts.
"""

import json
import os

from steerlab_server.experiment import evaluation_evidence, paired_judge, tasks

from test_evaluate_deferred import _fixture, _judgments_for

CONDITION_RESPONSES = {"scared", "afraid"}


def _read(directory):
    with open(os.path.join(directory, "judge-report.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_the_inline_report_tallies_both_judges_that_disagree(tmp_path, monkeypatch):
    root, _run = _fixture(tmp_path, judges=[
        {"name": "eager", "kind": "claude", "model": "claude-model-one"},
        {"name": "wary", "kind": "claude", "model": "claude-model-two"}])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def judge(model, rubric, a, b, structured=None, task_prompt=None):
        condition_is_a = a in CONDITION_RESPONSES
        prefers_condition = model == "claude-model-one"
        return {"winner": "A" if condition_is_a == prefers_condition else "B", "confidence": 0.8}

    monkeypatch.setattr(paired_judge, "judge_pair", judge)
    report = _read(tasks.evaluate("ev", root=root, log=lambda *_: None))

    blocks = {block["name"]: block for block in report["judges"]}
    assert blocks["eager"]["conditions"] == {"fear": {"baselineWins": 0, "variantWins": 2, "ties": 0, "n": 2}}
    assert blocks["wary"]["conditions"] == {"fear": {"baselineWins": 2, "variantWins": 0, "ties": 0, "n": 2}}
    # The top level is the panel, not the first judge.
    assert report["conditions"] == {"fear": {"baselineWins": 2, "variantWins": 2, "ties": 0, "n": 4}}
    assert report["conditionsScope"] == "allJudges"
    assert report["pairs"] == 2
    assert report["judgeModel"] == report["requestedJudgeModel"] == "claude-model-one, claude-model-two"
    (agreement,) = report["agreement"]
    assert agreement["percentAgreement"] == 0.0


def test_the_deferred_report_tallies_both_judges_that_disagree(tmp_path):
    root, _run = _fixture(tmp_path)
    evaluation = tasks.evaluate("ev", root=root, log=lambda *_: None)
    judgments = (_judgments_for(evaluation, judges=["opus-judge"], variant_wins=True)
                 + _judgments_for(evaluation, judges=["or-judge"], variant_wins=False))
    report = _read(tasks.complete_evaluate_judgment(
        "ev", os.path.basename(evaluation), judgments, root=root, log=lambda *_: None))

    blocks = {block["name"]: block for block in report["judges"]}
    assert blocks["opus-judge"]["conditions"]["fear"]["variantWins"] == 2
    assert blocks["or-judge"]["conditions"]["fear"]["baselineWins"] == 2
    assert report["conditions"] == {"fear": {"baselineWins": 2, "variantWins": 2, "ties": 0, "n": 4}}
    assert report["conditionsScope"] == "allJudges"
    assert report["judgeModel"] == ", ".join(block["requestedModel"] for block in report["judges"])


def test_one_judge_reads_as_before():
    """A single judge's report keeps the same top-level values it always had."""
    block = {"name": "only", "requestedModel": "m", "actualModel": "m-actual", "pairs": 3,
             "conditions": {"fear": {"baselineWins": 1, "variantWins": 1, "ties": 1, "n": 3}}}
    top = evaluation_evidence.panel_top_level([block])
    assert (top["conditions"], top["pairs"], top["judgeModel"], top["actualJudgeModel"]) == (
        block["conditions"], 3, "m", "m-actual")
    assert evaluation_evidence.panel_top_level([]) == {
        "pairs": 0, "conditions": {}, "conditionsScope": "allJudges", "judgeModel": None,
        "requestedJudgeModel": None, "actualJudgeModel": None}

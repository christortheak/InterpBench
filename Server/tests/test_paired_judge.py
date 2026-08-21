"""Paired-judge pairing, A/B blinding, and aggregation (no network — the judge
function is injected)."""

import pytest

from steerlab_server.experiment import paired_judge


GENERATIONS = [
    {"promptID": "p0", "seed": 0, "condition": "baseline", "output": "calm reply"},
    {"promptID": "p0", "seed": 0, "condition": "fear-L10", "output": "scared reply"},
    {"promptID": "p1", "seed": 0, "condition": "baseline", "output": "neutral"},
    {"promptID": "p1", "seed": 0, "condition": "fear-L10", "output": "afraid"},
]


def test_pairs_each_condition_with_its_baseline():
    pairs = paired_judge._pair_generations(GENERATIONS)
    assert len(pairs) == 2
    assert all(p["condition"] == "fear-L10" for p in pairs)
    assert {p["baseline"] for p in pairs} == {"calm reply", "neutral"}


def test_instrument_and_error_records_never_form_pairs():
    # Answer-token instrument records (one per condition of a choice-bearing
    # study) have no "seed"/"output"; without filtering, a baseline and a
    # steered instrument record for the same promptID collapse into the same
    # (promptID, None) cell and get judged as an empty-vs-empty pair.
    instrument_records = [
        {"promptID": "p0", "condition": "baseline",
         "instrument": "answerTokenLogprob", "target": "yes"},
        {"promptID": "p0", "condition": "fear-L10",
         "instrument": "answerTokenLogprob", "target": "yes"},
        {"promptID": "p1", "condition": "baseline", "seed": 0,
         "error": "generation failed"},
        {"promptID": "p1", "condition": "fear-L10", "seed": 0,
         "output": "afraid"},  # sampled, but its baseline errored — no pair
    ]
    assert paired_judge._pair_generations(instrument_records) == []

    # Mixed in with sampled records, only the sampled records pair up.
    pairs = paired_judge._pair_generations(GENERATIONS + instrument_records)
    assert len(pairs) == 2
    assert all(p["baseline"] and p["variant"] for p in pairs)


def test_blinding_is_deterministic_and_unmaps_correctly():
    # Judge always says the VARIANT text wins, whichever slot it is in.
    def fake_judge(model, rubric, a, b, structured, task_prompt=None):
        variant_texts = {"scared reply", "afraid"}
        winner = "A" if a in variant_texts else "B"
        return {"winner": winner, "confidence": 0.9}

    judgments, report = paired_judge.evaluate(
        GENERATIONS, judge_model="x", judge_rubric="which is more fearful",
        judge=fake_judge)
    # Every judgment should resolve to the variant winning, regardless of A/B slot.
    assert all(j["outcome"] == "variant" for j in judgments)
    cond = report["conditions"]["fear-L10"]
    assert cond["variantWins"] == 2 and cond["baselineWins"] == 0 and cond["ties"] == 0


def test_tie_counts():
    def tie_judge(*a, **k):
        return {"winner": "tie", "confidence": 0.1}
    _, report = paired_judge.evaluate(GENERATIONS, judge_model="x",
                                      judge_rubric="r", judge=tie_judge)
    assert report["conditions"]["fear-L10"]["ties"] == 2


def test_parse_response_handles_fences():
    obj = paired_judge.parse_response('```json\n{"winner":"A","confidence":0.7}\n```')
    assert obj["winner"] == "A"


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert paired_judge.available() is False


def test_is_claude_model():
    assert paired_judge.is_claude_model("claude-opus-4-8")
    assert paired_judge.is_claude_model("anthropic/claude")
    assert not paired_judge.is_claude_model("mlx-community/gemma-3-27b-it-8bit")
    assert not paired_judge.is_claude_model("Qwen/Qwen3-0.6B")


def test_local_judge_parses_model_json():
    # The local model "generates" a verdict; make_local_judge wraps + parses it.
    judge = paired_judge.make_local_judge(
        lambda prompt: 'My verdict: {"winner":"A","confidence":0.8}')
    verdict = judge("local", "rubric", "resp A", "resp B", None)
    assert verdict["winner"] == "A"


def test_local_judge_raises_on_garbage_instead_of_inventing_a_tie():
    # Invalid-verdict closure (2026-07-20): a malformed local-judge response
    # used to be silently recorded as a substantive tie. It now raises (the
    # callers' valid_verdict wrapper retries once, then refuses).
    judge = paired_judge.make_local_judge(lambda prompt: "I cannot decide, sorry.")
    with pytest.raises(ValueError, match="no parseable JSON verdict"):
        judge("local", "rubric", "a", "b", None)


# --- valid_verdict: retry once, then refuse ----------------------------------

def test_valid_verdict_retries_once_then_succeeds():
    calls = []

    def flaky(model, rubric, a, b, structured, task_prompt=None):
        calls.append(1)
        if len(calls) == 1:
            return {"winner": "both", "confidence": 0.9}
        return {"winner": "B", "confidence": 0.8}

    verdict = paired_judge.valid_verdict(
        flaky, "m", "r", "a", "b", None,
        judge_label="'j1'", item_label="pair c/p0")
    assert verdict == {"winner": "B", "confidence": 0.8}
    assert len(calls) == 2  # retried exactly once


def test_valid_verdict_refuses_after_two_invalid_verdicts():
    calls = []

    def stubborn(model, rubric, a, b, structured, task_prompt=None):
        calls.append(1)
        return {"winner": "both"}

    with pytest.raises(RuntimeError) as excinfo:
        paired_judge.valid_verdict(
            stubborn, "m", "r", "a", "b", None,
            judge_label="'j1'", item_label="pair c/p0")
    message = str(excinfo.value)
    assert "judge 'j1'" in message
    assert "invalid verdict twice" in message
    assert "pair c/p0" in message
    assert "'both'" in message
    assert len(calls) == 2  # one retry, never a third call


def test_valid_verdict_retries_unparseable_local_responses():
    judge = paired_judge.make_local_judge(lambda prompt: "no json here")
    with pytest.raises(RuntimeError, match="no parseable JSON verdict"):
        paired_judge.valid_verdict(
            judge, "m", "r", "a", "b", None,
            judge_label="'local'", item_label="pair c/p0")


def test_valid_verdict_propagates_transport_errors_without_retry():
    calls = []

    def broken(model, rubric, a, b, structured, task_prompt=None):
        calls.append(1)
        raise RuntimeError("HTTP 500 from the judge API")

    with pytest.raises(RuntimeError, match="HTTP 500"):
        paired_judge.valid_verdict(
            broken, "m", "r", "a", "b", None,
            judge_label="'j1'", item_label="pair c/p0")
    assert len(calls) == 1  # only verdict problems retry


def test_evaluate_refuses_out_of_vocabulary_winner():
    # The old behavior was WORSE than a tie: "both" != "A" made one side win.
    # A judge that is garbage on EVERY pair is systemic failure (the
    # noncompliance cap), so the whole evaluation still refuses — but each
    # pair is first recorded as a noncompliant row (2026-08-09).
    def garbage_judge(model, rubric, a, b, structured, task_prompt=None):
        return {"winner": "both", "confidence": 0.9}

    persisted = []
    with pytest.raises(RuntimeError, match="noncompliant on 2 of 2"):
        paired_judge.evaluate(GENERATIONS, judge_model="x",
                              judge_rubric="r", judge=garbage_judge,
                              on_judgment=persisted.append)
    assert len(persisted) == 2
    assert all(row["noncompliant"] and row["outcome"] is None
               for row in persisted)
    assert all("invalid verdict twice" in row["noncomplianceReason"]
               for row in persisted)


def test_evaluate_survives_isolated_judge_noncompliance():
    # One flaky pair no longer kills hours of evaluation (Christian,
    # 2026-08-09): the failure becomes a recorded, classifiable row and the
    # run completes. Refusal-to-invent stands: no winner is recorded.
    many = []
    for i in range(8):
        many.append({"promptID": f"p{i}", "seed": 0, "condition": "baseline",
                     "output": f"base {i}"})
        many.append({"promptID": f"p{i}", "seed": 0, "condition": "fear-L10",
                     "output": f"steered {i}"})

    def mostly_fine(model, rubric, a, b, structured, task_prompt=None):
        if "3" in a or "3" in b:
            return {"winner": None}          # noncompliant, both attempts
        return {"winner": "A", "confidence": 0.8}

    persisted = []
    judgments, report = paired_judge.evaluate(
        many, judge_model="x", judge_rubric="r", judge=mostly_fine,
        on_judgment=persisted.append)
    assert report["noncompliantJudgments"] == 1
    assert report["pairs"] == 8
    bad = [j for j in judgments if j.get("noncompliant")]
    assert len(bad) == 1 and bad[0]["promptID"] == "p3"
    assert bad[0]["outcome"] is None and bad[0]["judgment"] is None
    # The tally only counts real verdicts.
    assert report["conditions"]["fear-L10"]["n"] == 7
    # Persist-then-continue held for the noncompliant row too.
    assert len(persisted) == 8


def test_evaluate_rejudges_noncompliant_rows_on_resume():
    # A noncompliant row from an earlier session is a recorded failure, not
    # a verdict — resume must re-judge that cell, not reuse (or refuse over)
    # the hole.
    def fine(model, rubric, a, b, structured, task_prompt=None):
        return {"winner": "A", "confidence": 0.9}

    prior = {("p0", "0", "fear-L10"): {
        "promptID": "p0", "sampleIndex": 0, "condition": "fear-L10",
        "baselineWas": "A", "outcome": None, "noncompliant": True,
        "noncomplianceReason": "invalid verdict twice", "judgment": None}}
    judgments, report = paired_judge.evaluate(
        GENERATIONS, judge_model="x", judge_rubric="r", judge=fine,
        existing=prior)
    assert not any(j.get("noncompliant") for j in judgments)
    assert "reusedJudgments" not in report
    assert report["conditions"]["fear-L10"]["n"] == 2

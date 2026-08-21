"""Paired judging over REAL sampled-path records (external review
2026-07-22, P0).

The defect this pins closed: ``derive_seed`` includes CONDITION identity by
documented policy, so under ``seedPolicy: derivedSHA256`` with
``temperature > 0`` a baseline and a variant draw of the same (prompt,
sampleIndex) never share a seed — and the old ``(promptID, seed)`` pairing
join therefore paired NOTHING. A judged evaluate then wrote a
successful-looking report with ``pairs: 0``, inline and deferred alike:
every judged positive-temperature comparison was silently void.

Contract now enforced (both engines): pairs join on
``(promptID, sampleIndex)`` (absent normalizes to 0, so greedy /
single-sample records pair exactly as before); every pair/judgment artifact
stamps both sides' seeds as ``baselineSeed``/``variantSeed`` (never a field
named ``seed``); and zero surviving pairs REFUSE with
``paired_judge.NO_PAIRS_MESSAGE`` — never a quiet empty report. Records are
produced by the REAL sampled run path (the test_variant_study_sampling
fixtures), not hand-built rows. Swift twin:
``Tests/ExperimentKitTests/PairedJudgePairingTests.swift``.
"""

import json
import os

import pytest

import test_variant_study_sampling as vss
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import paired_judge, tasks

# Non-baseline conditions in the shared fixture (baseline + 2 saved agents).
VARIANTS = ("agent-a", "agent-b")
EXPECTED_PAIRS = vss.PROMPTS * vss.SAMPLES * len(VARIANTS)


def _sampled_run(tmp_path, monkeypatch, name, *, evaluation=False):
    """A real sampled study run: 2 prompts x (baseline + 2 agents) x 3
    samples at temperature 0.7 under derivedSHA256, via the actual run
    loop with the fake model/generate seams."""
    root = str(tmp_path)
    prompts = vss._study_fixture(root, name)
    if evaluation:
        # Declared BEFORE the run so the run's experiment-hash stamp matches
        # the live manifest (the evaluate epoch guard). Draft study: the
        # inline judgePrompt is the (loud) rubric fallback.
        raw = es.load_raw(name, root)
        raw["evaluation"] = {"kind": "pairedJudge", "judgeModel": "claude-test",
                             "judgePrompt": "which response is better?"}
        raw["judges"] = [{"name": "opus-judge", "kind": "claude"}]
        es.save_raw(raw, root)
    vss._patch(monkeypatch, vss._fake_generate(), vss._fake_score_options())
    run_dir = tasks.run(name, prompts, root, model_provider=vss._fake_model,
                        log=lambda *_: None)
    return root, run_dir, vss._records(run_dir)


def test_sampled_records_pair_per_sample_cell_with_distinct_seeds(
        tmp_path, monkeypatch):
    _root, _run, records = _sampled_run(tmp_path, monkeypatch, "pjs-pairs")
    pairs = paired_judge._pair_generations(records)
    assert len(pairs) == EXPECTED_PAIRS == 12
    for condition in VARIANTS:
        mine = [p for p in pairs if p["condition"] == condition]
        # pair count == prompts × samples, per variant condition.
        assert len(mine) == vss.PROMPTS * vss.SAMPLES
        assert sorted((p["promptID"], p["sampleIndex"]) for p in mine) == [
            (f"p{i}", s) for i in range(vss.PROMPTS)
            for s in range(vss.SAMPLES)]
    for pair in pairs:
        assert "seed" not in pair  # never a single-seed field on a pair
        # Condition-identity derivation: the two sides NEVER share a seed.
        assert pair["baselineSeed"] != pair["variantSeed"]
        assert pair["baseline"] and pair["variant"]

    # And the refuted join: group the same records by the OLD (promptID,
    # seed) key — no cell ever contains two conditions, so the old rule
    # paired nothing. This is the P0, demonstrated on real records.
    sampled = [r for r in records if "instrument" not in r and "error" not in r]
    by_old_key = {}
    for r in sampled:
        by_old_key.setdefault((r["promptID"], r["seed"]), set()).add(
            r["condition"])
    assert by_old_key and all(len(c) == 1 for c in by_old_key.values())


def test_evaluate_judges_every_pair_and_stamps_seed_provenance(
        tmp_path, monkeypatch):
    _root, _run, records = _sampled_run(tmp_path, monkeypatch, "pjs-judged")
    judged = []

    def counting_judge(model, rubric, a, b, structured=None, task_prompt=None):
        judged.append((a, b))
        return {"winner": "A", "confidence": 0.9}

    judgments, report = paired_judge.evaluate(
        records, judge_model="x", judge_rubric="r", judge=counting_judge)
    assert report["pairs"] == EXPECTED_PAIRS
    assert len(judged) == EXPECTED_PAIRS  # one judge call per pair
    for condition in VARIANTS:
        assert report["conditions"][condition]["n"] == \
            vss.PROMPTS * vss.SAMPLES
    for j in judgments:
        assert "seed" not in j
        assert j["sampleIndex"] in range(vss.SAMPLES)
        assert j["baselineSeed"] != j["variantSeed"]


def test_zero_pairs_refuses_with_the_shared_message(tmp_path, monkeypatch):
    # Doctor the real record set back into the P0 shape: no variant record
    # shares a cell with a baseline record (the effect the seed join had).
    _root, _run, records = _sampled_run(tmp_path, monkeypatch, "pjs-none")
    doctored = []
    for r in records:
        r = dict(r)
        if r.get("condition") != "baseline" and "sampleIndex" in r:
            r["sampleIndex"] += 100
        doctored.append(r)
    with pytest.raises(RuntimeError, match=r"\(promptID, sampleIndex\)") \
            as excinfo:
        paired_judge.evaluate(doctored, judge_model="x", judge_rubric="r",
                              judge=lambda *a, **k: {"winner": "tie"})
    assert str(excinfo.value) == paired_judge.NO_PAIRS_MESSAGE


def test_no_pairs_message_names_join_key_and_likely_cause():
    # VALUE-pinned cross-engine (Swift: ExperimentTasks.noPairsMessage —
    # PairedJudgePairingTests asserts the same content).
    message = paired_judge.NO_PAIRS_MESSAGE
    assert "(promptID, sampleIndex)" in message
    assert "baseline" in message
    assert "instrument readouts and error records" in message
    assert "Refusing" in message


def test_full_evaluate_task_refuses_zero_pairs_before_any_judging(
        tmp_path, monkeypatch):
    # End-to-end inline evaluate: a judge panel is configured and the source
    # run pairs nothing → RuntimeError BEFORE any judge call, and no
    # evaluate run directory gets a judge-report.json.
    root, run_dir, records = _sampled_run(tmp_path, monkeypatch, "pjs-task",
                                          evaluation=True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # inline custody
    calls = []
    monkeypatch.setattr(
        paired_judge, "judge_pair",
        lambda *a, **k: calls.append(1) or {"winner": "tie",
                                            "confidence": 0.5})
    doctored = []
    for r in records:
        r = dict(r)
        if r.get("condition") != "baseline" and "sampleIndex" in r:
            r["sampleIndex"] += 100
        doctored.append(r)
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for r in doctored:
            handle.write(json.dumps(r) + "\n")
    with pytest.raises(RuntimeError, match="zero pairs"):
        tasks.evaluate("pjs-task", root=root, log=lambda *_: None)
    assert calls == []  # refused before judging, not after burning calls
    runs_root = os.path.join(root, "runs")
    for entry in os.listdir(runs_root):
        assert not os.path.exists(
            os.path.join(runs_root, entry, "judge-report.json"))


def test_deferred_evaluate_emits_one_packet_per_pair_with_seed_provenance(
        tmp_path, monkeypatch):
    # Keyless custody (the conftest default): the same sampled run defers —
    # packets emitted == pairs, and the judging map carries the pair-cell
    # key plus both sides' seeds (never "seed").
    root, _run, _records = _sampled_run(tmp_path, monkeypatch, "pjs-defer",
                                        evaluation=True)
    eval_dir = tasks.evaluate("pjs-defer", root=root, log=lambda *_: None)
    jm = json.load(open(os.path.join(eval_dir, "judging-manifest.json")))
    assert jm["packetCount"] == EXPECTED_PAIRS
    packets = [json.loads(line) for line in
               open(os.path.join(eval_dir, "judging-packets.jsonl"))]
    assert len(packets) == EXPECTED_PAIRS
    packet_map = json.load(
        open(os.path.join(eval_dir, "judging-map.json")))["packets"]
    assert len(packet_map) == EXPECTED_PAIRS
    for meta in packet_map.values():
        assert "seed" not in meta
        assert meta["sampleIndex"] in range(vss.SAMPLES)
        assert meta["baselineSeed"] != meta["variantSeed"]
        assert meta["condition"] in VARIANTS


def test_deferred_completion_carries_the_seed_provenance_through(
        tmp_path, monkeypatch):
    # Phase 2: completed judgment rows keep sampleIndex + both seeds, so the
    # deferred artifact set matches the inline path's shape.
    root, _run, _records = _sampled_run(tmp_path, monkeypatch, "pjs-complete",
                                        evaluation=True)
    eval_dir = tasks.evaluate("pjs-complete", root=root, log=lambda *_: None)
    packet_map = json.load(
        open(os.path.join(eval_dir, "judging-map.json")))["packets"]
    jm = json.load(open(os.path.join(eval_dir, "judging-manifest.json")))
    (judge_entry,) = jm["judges"]
    judgments = [
        {"packetID": pid, "judge": judge_entry["name"],
         "model": judge_entry["model"],
         "winner": "A" if meta["baselineIsA"] else "B", "confidence": 0.8}
        for pid, meta in packet_map.items()]
    out = tasks.complete_evaluate_judgment(
        "pjs-complete", os.path.basename(eval_dir), judgments, root=root,
        log=lambda *_: None)
    rows = [json.loads(line)
            for line in open(os.path.join(out, "judgments.jsonl"))]
    assert len(rows) == EXPECTED_PAIRS
    for row in rows:
        assert "seed" not in row
        assert row["sampleIndex"] in range(vss.SAMPLES)
        assert row["baselineSeed"] != row["variantSeed"]
        assert row["outcome"] == "baseline"  # the crafted winner unblinds
    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert report["pairs"] == EXPECTED_PAIRS

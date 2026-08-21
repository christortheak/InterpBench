"""Deferred (two-phase) evaluate — seamless pipeline stage 2, 2026-07-19.

An external judge panel with no cluster credential must not refuse OR judge:
the run's paired generations become blinded, hash-pinned packets; the Mac
judges them; `complete_evaluate_judgment` verifies every pin and aggregates
the SAME judgments.jsonl + judge-report.json the inline path writes.
Contract under test: emission artifacts + pins, inline-identical blinding
and outcome unblinding, full-coverage + pinned-model refusals, epoch gates,
idempotent completion, and the scanner's verified-marker suppression rule.
"""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import paired_judge, tasks
from steerlab_server.experiment.manifest import Manifest

RUBRIC = "Which response expresses more dread?"


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = (json.dumps(content, indent=2) if isinstance(content, dict)
            else content)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _fixture(tmp_path, *, judges=None, structured=None):
    root = str(tmp_path)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         RUBRIC)
    evaluation = {"kind": "pairedJudge", "judgeModel": "claude-opus-4-8"}
    if structured:
        evaluation["structuredPrompt"] = structured
    d = {"name": "ev", "modelID": "org/m", "status": "draft",
         "evaluation": evaluation,
         "judgeRubricFile": "prompts/rubrics/r.md",
         "judgeRubricHash": rubric_hash,
         "judges": judges if judges is not None else [
             {"name": "opus-judge", "kind": "claude"},
             {"name": "or-judge", "kind": "openrouter",
              "model": "anthropic/claude-opus-4.8",
              "provider": "Anthropic"}]}
    _write(os.path.join(root, "experiments", "ev", "experiment.json"), d)
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-ev-run")
    os.makedirs(run_dir)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(d).content_hash() + "\n")
    generations = [
        {"promptID": "p0", "seed": 0, "condition": "baseline",
         "prompt": "Describe the cellar.", "output": "calm"},
        {"promptID": "p0", "seed": 0, "condition": "fear",
         "prompt": "Describe the cellar.", "output": "scared"},
        {"promptID": "p1", "seed": 0, "condition": "baseline",
         "prompt": "Describe the attic.", "output": "fine"},
        {"promptID": "p1", "seed": 0, "condition": "fear",
         "prompt": "Describe the attic.", "output": "afraid"},
    ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for g in generations:
            handle.write(json.dumps(g) + "\n")
    return root, run_dir


def _judgments_for(eval_dir, *, judges, variant_wins=True):
    """Craft complete judgments by unblinding through the map — the variant
    wins (or loses) for every judge; every judgment stamps its judge's
    emission-pinned model, and openrouter judgments stamp the emission-pinned
    provider (completion verifies both)."""
    with open(os.path.join(eval_dir, "judging-map.json"),
              encoding="utf-8") as handle:
        packet_map = json.load(handle)["packets"]
    with open(os.path.join(eval_dir, "judging-manifest.json"),
              encoding="utf-8") as handle:
        jm = json.load(handle)
    entry_by_judge = {j["name"]: j for j in jm["judges"]}
    out = []
    for judge in judges:
        entry = entry_by_judge[judge]
        for pid, meta in packet_map.items():
            if variant_wins:
                winner = "B" if meta["baselineIsA"] else "A"
            else:
                winner = "A" if meta["baselineIsA"] else "B"
            row = {"packetID": pid, "judge": judge, "winner": winner,
                   "model": entry["model"], "confidence": 0.9}
            if entry.get("kind") == "openrouter":
                row["provider"] = entry["provider"]
            out.append(row)
    return out


def test_deferred_evaluate_emits_pinned_packets(tmp_path):
    root, run_dir = _fixture(tmp_path, structured="compare severity")
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)

    # No judging happened — the run is honestly awaiting.
    assert not os.path.exists(os.path.join(eval_dir, "judge-report.json"))
    assert os.path.exists(os.path.join(eval_dir, "awaiting-judgment.json"))
    with open(os.path.join(eval_dir, "judging-manifest.json"),
              encoding="utf-8") as handle:
        jm = json.load(handle)
    assert jm["kind"] == "evaluate"
    assert jm["sourceRun"] == os.path.basename(run_dir)
    assert jm["packetCount"] == 2
    # Structured prompt is pinned like the rubric — part of the criterion.
    assert jm["structuredPrompt"] == "compare severity"
    assert jm["structuredPromptSha256"] == hashlib.sha256(
        b"compare severity").hexdigest()
    assert jm["rubricTextSha256"] == hashlib.sha256(
        RUBRIC.encode()).hexdigest()
    # OpenRouter judge pins survive normalization verbatim.
    or_judge = next(j for j in jm["judges"] if j["kind"] == "openrouter")
    assert or_judge["provider"] == "Anthropic"
    # Packets are identity-free (prompt + responses only) and blinded with
    # the INLINE convention.
    packets = [json.loads(line) for line in
               open(os.path.join(eval_dir, "judging-packets.jsonl"))]
    assert all(set(p) == {"packetID", "prompt", "responseA", "responseB"}
               for p in packets)
    with open(os.path.join(eval_dir, "judging-map.json"),
              encoding="utf-8") as handle:
        packet_map = json.load(handle)["packets"]
    for pid, meta in packet_map.items():
        assert meta["baselineIsA"] == paired_judge._baseline_first(
            str(meta["promptID"]), meta["condition"])
    # Awaiting scanners: this run appears under evaluate, never under sweep.
    assert [a["run"] for a in tasks.list_awaiting_evaluate_judgment(
        "ev", root)] == [os.path.basename(eval_dir)]
    assert tasks.list_awaiting_judgment("ev", root) == []


def test_completion_aggregates_the_inline_shapes_and_is_idempotent(tmp_path):
    root, _run = _fixture(tmp_path)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)
    eval_run = os.path.basename(eval_dir)
    judgments = _judgments_for(eval_dir, judges=["opus-judge", "or-judge"])

    out = tasks.complete_evaluate_judgment("ev", eval_run, judgments,
                                           root=root, log=lambda *_: None)
    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert report["judgedOn"] == "client"
    assert report["pairs"] == 2
    assert [b["name"] for b in report["judges"]] == ["opus-judge", "or-judge"]
    for block in report["judges"]:
        assert block["conditions"]["fear"] == {
            "baselineWins": 0, "variantWins": 2, "ties": 0, "n": 2}
    (agreement,) = report["agreement"]
    assert agreement["percentAgreement"] == 1.0
    rows = [json.loads(line) for line in
            open(os.path.join(out, "judgments.jsonl"))]
    assert all(r["outcome"] == "variant" for r in rows)
    assert {r["judgeModel"] for r in rows if r["judge"] == "or-judge"} \
        == {"anthropic/claude-opus-4.8"}
    # The VERIFIED serving provider is stamped per judgment (openrouter
    # judges only) and onto the judge block.
    assert {r["judgeProvider"] for r in rows if r["judge"] == "or-judge"} \
        == {"anthropic"}
    assert all("judgeProvider" not in r for r in rows
               if r["judge"] == "opus-judge")
    or_block = next(b for b in report["judges"] if b["name"] == "or-judge")
    assert or_block["provider"] == "anthropic"
    # The completed run no longer awaits, and completion is idempotent.
    assert tasks.list_awaiting_evaluate_judgment("ev", root) == []
    again = tasks.complete_evaluate_judgment("ev", eval_run, judgments,
                                             root=root, log=lambda *_: None)
    assert again == out


def test_completion_refusals_fail_closed(tmp_path):
    root, _run = _fixture(tmp_path)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)
    eval_run = os.path.basename(eval_dir)
    good = _judgments_for(eval_dir, judges=["opus-judge", "or-judge"])

    # Incomplete coverage.
    with pytest.raises(ValueError, match="full coverage"):
        tasks.complete_evaluate_judgment("ev", eval_run, good[:-1],
                                         root=root, log=lambda *_: None)
    # Off-pin judge model.
    off_pin = [dict(j) for j in good]
    off_pin[0]["model"] = "claude-something-else"
    with pytest.raises(ValueError, match="off-pin"):
        tasks.complete_evaluate_judgment("ev", eval_run, off_pin,
                                         root=root, log=lambda *_: None)
    # Unknown judge.
    bad_judge = [dict(j) for j in good]
    bad_judge[0]["judge"] = "impostor"
    with pytest.raises(ValueError, match="unpinned judge"):
        tasks.complete_evaluate_judgment("ev", eval_run, bad_judge,
                                         root=root, log=lambda *_: None)
    # Provider evidence fails CLOSED (engineer review 2026-07-18): an
    # openrouter judgment with no provider stamp is refused, not assumed.
    unstamped = [dict(j) for j in good]
    for j in unstamped:
        j.pop("provider", None)
    with pytest.raises(ValueError, match="carries no provider"):
        tasks.complete_evaluate_judgment("ev", eval_run, unstamped,
                                         root=root, log=lambda *_: None)
    # Off-pin provider.
    off_provider = [dict(j) for j in good]
    for j in off_provider:
        if "provider" in j:
            j["provider"] = "SomeReseller"
    with pytest.raises(ValueError, match="off-pin"):
        tasks.complete_evaluate_judgment("ev", eval_run, off_provider,
                                         root=root, log=lambda *_: None)
    # A provider on a NON-openrouter judgment is a mix-up, refused.
    mixed_up = [dict(j) for j in good]
    next(j for j in mixed_up
         if j["judge"] == "opus-judge")["provider"] = "Anthropic"
    with pytest.raises(ValueError, match="not openrouter-kind"):
        tasks.complete_evaluate_judgment("ev", eval_run, mixed_up,
                                         root=root, log=lambda *_: None)
    # Tampered packets file.
    packets_path = os.path.join(eval_dir, "judging-packets.jsonl")
    with open(packets_path, "a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="drifted since emission"):
        tasks.complete_evaluate_judgment("ev", eval_run, good,
                                         root=root, log=lambda *_: None)


def test_completion_verifies_emitted_source_and_criterion_pins(tmp_path):
    # Emission pins are RE-CHECKED at completion (engineer review
    # 2026-07-18): the source generations the packets were cut from, the
    # structured prompt's own hash, and the run-directory identity — a
    # stamped-but-unverified pin is not provenance.
    root, run_dir = _fixture(tmp_path, structured="compare severity")
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)
    eval_run = os.path.basename(eval_dir)
    good = _judgments_for(eval_dir, judges=["opus-judge", "or-judge"])

    # Tampered SOURCE generations: the evidence chain back to raw outputs
    # is broken — refuse.
    with open(os.path.join(run_dir, "generations.jsonl"), "a",
              encoding="utf-8") as handle:
        handle.write(json.dumps({"promptID": "p9", "seed": 0,
                                 "condition": "fear", "prompt": "x",
                                 "output": "y"}) + "\n")
    with pytest.raises(ValueError, match="source run's generations"):
        tasks.complete_evaluate_judgment("ev", eval_run, good,
                                         root=root, log=lambda *_: None)

    # Fresh fixture: a structured prompt edited inside the judging manifest
    # no longer hashes to its own emission pin.
    root2, _ = _fixture(tmp_path / "two", structured="compare severity")
    eval_dir2 = tasks.evaluate("ev", root=root2, log=lambda *_: None)
    jm_path = os.path.join(eval_dir2, "judging-manifest.json")
    with open(jm_path, encoding="utf-8") as handle:
        jm = json.load(handle)
    jm["structuredPrompt"] = "compare tone instead"
    with open(jm_path, "w", encoding="utf-8") as handle:
        json.dump(jm, handle)
    with pytest.raises(ValueError, match="structured prompt does not hash"):
        tasks.complete_evaluate_judgment(
            "ev", os.path.basename(eval_dir2),
            _judgments_for(eval_dir2, judges=["opus-judge", "or-judge"]),
            root=root2, log=lambda *_: None)

    # Relocated awaiting run: the manifest names another run directory.
    root3, _ = _fixture(tmp_path / "three")
    eval_dir3 = tasks.evaluate("ev", root=root3, log=lambda *_: None)
    relocated = os.path.join(os.path.dirname(eval_dir3),
                             "20260103T000000000-exp-ev-relocated")
    os.rename(eval_dir3, relocated)
    with pytest.raises(ValueError, match="relocated awaiting run"):
        tasks.complete_evaluate_judgment(
            "ev", os.path.basename(relocated),
            _judgments_for(relocated, judges=["opus-judge", "or-judge"]),
            root=root3, log=lambda *_: None)


def test_completion_passes_full_verdicts_through(tmp_path):
    # Winner-only closure (2026-07-20): a judging client that sends the
    # judge's FULL verdict sees it land verbatim in judgments.jsonl (the
    # inline path's "judgment" key), instead of `judgment: null`; rows from
    # older winner-only clients still complete.
    root, _run = _fixture(tmp_path)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)
    eval_run = os.path.basename(eval_dir)
    judgments = _judgments_for(eval_dir, judges=["opus-judge", "or-judge"])
    for row in judgments:
        if row["judge"] == "opus-judge":
            row["judgment"] = {"winner": row["winner"], "confidence": 0.8,
                               "brief_reason": "more dread",
                               "a_scores": {"dread": 4}, "b_scores": {"dread": 1}}
            row.pop("confidence")  # confidence may ride inside the payload
    out = tasks.complete_evaluate_judgment("ev", eval_run, judgments,
                                           root=root, log=lambda *_: None)
    rows = [json.loads(line) for line in
            open(os.path.join(out, "judgments.jsonl"))]
    opus_rows = [r for r in rows if r["judge"] == "opus-judge"]
    assert opus_rows and all(
        r["judgment"]["brief_reason"] == "more dread"
        and r["judgment"]["a_scores"] == {"dread": 4}
        and r["confidence"] == 0.8
        for r in opus_rows)
    # Winner-only rows (or-judge) keep the inline shape with judgment: null.
    assert all(r["judgment"] is None for r in rows if r["judge"] == "or-judge")


def test_completion_refuses_a_verdict_contradicting_its_winner(tmp_path):
    # A payload is provenance only if verified: a verdict whose own winner
    # contradicts the judged winner is refused, never recorded.
    root, _run = _fixture(tmp_path)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)
    eval_run = os.path.basename(eval_dir)
    judgments = _judgments_for(eval_dir, judges=["opus-judge", "or-judge"])
    judgments[0]["judgment"] = {
        "winner": "tie" if judgments[0]["winner"] != "tie" else "A",
        "confidence": 0.9, "brief_reason": "x"}
    with pytest.raises(ValueError, match="contradicts the judged winner"):
        tasks.complete_evaluate_judgment("ev", eval_run, judgments,
                                         root=root, log=lambda *_: None)
    # A non-object payload refuses too.
    judgments = _judgments_for(eval_dir, judges=["opus-judge", "or-judge"])
    judgments[0]["judgment"] = "great response"
    with pytest.raises(ValueError, match="non-object verdict payload"):
        tasks.complete_evaluate_judgment("ev", eval_run, judgments,
                                         root=root, log=lambda *_: None)


def test_completion_refuses_a_drifted_manifest_epoch(tmp_path):
    root, _run = _fixture(tmp_path)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)
    eval_run = os.path.basename(eval_dir)
    good = _judgments_for(eval_dir, judges=["opus-judge", "or-judge"])
    d = es.load_raw("ev", root)
    d["description"] = "edited between emission and judgment"
    es.save_raw(d, root)
    with pytest.raises(ValueError, match="epoch mismatch"):
        tasks.complete_evaluate_judgment("ev", eval_run, good,
                                         root=root, log=lambda *_: None)


def test_split_panel_refuses_instead_of_deferring(tmp_path):
    root, _run = _fixture(tmp_path, judges=[
        {"name": "j-local", "kind": "local", "model": "org/judge"},
        {"name": "opus-judge", "kind": "claude"}])
    with pytest.raises(RuntimeError, match="split panel"):
        tasks.evaluate("ev", root=root, log=lambda *_: None)


def test_forged_completion_marker_never_suppresses_awaiting(tmp_path):
    root, _run = _fixture(tmp_path)
    eval_dir = tasks.evaluate("ev", root=root, log=lambda *_: None)
    eval_run = os.path.basename(eval_dir)
    forged = os.path.join(root, "runs", "20260102T000000000-exp-ev-forged")
    os.makedirs(forged)
    _write(os.path.join(forged, "judgment-source.json"),
           {"schema": 1, "kind": "evaluate", "experiment": "ev",
            "evaluateRun": eval_run, "packetsSha256": "0" * 64,
            "experimentHashAtJudgment": "0" * 64,
            "judgmentsSha256": "0" * 64, "judgeReportSha256": "0" * 64})
    # The forged record neither hides the awaiting run nor satisfies the
    # idempotency lookup — a matching-but-invalid marker RAISES there.
    assert [a["run"] for a in tasks.list_awaiting_evaluate_judgment(
        "ev", root)] == [eval_run]
    with pytest.raises(ValueError, match="failed verification"):
        tasks.complete_evaluate_judgment(
            "ev", eval_run,
            _judgments_for(eval_dir, judges=["opus-judge", "or-judge"]),
            root=root, log=lambda *_: None)


def test_inline_evaluate_passes_task_prompt_and_stamps_custody(
        tmp_path, monkeypatch):
    # With a credential the fork stays inline — and the canonical judge
    # contract now gives evaluate judges the task prompt, like sweeps.
    root, _run = _fixture(tmp_path, judges=[
        {"name": "opus-judge", "kind": "claude"}])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    seen_prompts = []

    def fake_judge(model, rubric, a, b, structured=None, task_prompt=None):
        seen_prompts.append(task_prompt)
        return {"winner": "tie", "confidence": 0.5}

    monkeypatch.setattr(paired_judge, "judge_pair", fake_judge)
    out = tasks.evaluate("ev", root=root, log=lambda *_: None)
    assert sorted(seen_prompts) == ["Describe the attic.",
                                    "Describe the cellar."]
    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert report["judgedOn"] == "server"
    assert report["judgeCredential"] == {"kind": "anthropic", "source": "env"}
    # Per-judge credential provenance (engineer review 2026-07-19).
    assert report["judges"][0]["credential"] == {"kind": "anthropic",
                                                "source": "env"}


def test_inline_openrouter_evaluate_stamps_verified_provider(
        tmp_path, monkeypatch):
    # Inline and deferred reports carry ONE provenance shape (engineer
    # review 2026-07-19): the inline path stamps the VERIFIED serving
    # provider per judgment and the provider + per-judge credential on the
    # judge block, exactly like deferred completion.
    root, _run = _fixture(tmp_path, judges=[
        {"name": "or-judge", "kind": "openrouter",
         "model": "anthropic/claude-opus-4.8", "provider": "Anthropic"}])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE",
                       os.path.join(root, "no-such-judge-key"))

    def fake_make(model, provider, transport=None):
        def judge(_m, rubric, a, b, structured, task_prompt=None):
            # The client stamps the SERVED-BY provider it verified.
            return {"winner": "tie", "confidence": 0.5, "provider": provider}
        return judge

    monkeypatch.setattr(paired_judge, "make_openrouter_judge", fake_make)
    out = tasks.evaluate("ev", root=root, log=lambda *_: None)
    rows = [json.loads(line) for line in
            open(os.path.join(out, "judgments.jsonl"))]
    assert rows and all(r["judgeProvider"] == "anthropic" for r in rows)
    report = json.load(open(os.path.join(out, "judge-report.json")))
    block = report["judges"][0]
    assert block["provider"] == "anthropic"
    assert block["credential"] == {"kind": "openrouter", "source": "env"}

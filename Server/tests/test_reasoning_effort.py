"""The declared reasoning effort and the reasoning token budget (2026-09-03).

``reasoningEffort`` ∈ off | low | medium | xhigh replaced the Qwen-specific
boolean ``qwenThinkingEnabled`` on the study protocol and on a concept's
extraction rendering, and ``reasoningMaxTokens`` gave the reasoning block a cap
of its own so it no longer competes with the answer for one budget.

These tests pin, on this engine:

1. the vocabulary and the legacy reading — a frozen manifest under the old
   spelling loads as off/xhigh, verifies unchanged, and is never rewritten;
2. the declaration rules — a non-off effort needs a budget and a family with a
   thinking mode, an off effort takes no budget, and every refusal is typed;
3. the two-phase budget — the pure rule, the stopping criterion built on it,
   and the ``lengthInReasoning`` finish reason read back from the ids;
4. the preflights — context reserves both budgets, walltime prices both;
5. the gate and the report — a reasoning-capped cell is incomplete, and the
   report and CSV say which cap was hit;
6. the preregistration line.
"""

import csv
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import lifecycle_gates, prompt_render, tasks
from steerlab_server.experiment import truncation_gate as tg
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import extraction_rendering as er

QWEN = "Qwen/Qwen3-0.6B"
GEMMA = "google/gemma-3-4b-it"
CLOSE = 7          # the fixture tokenizer's `</think>` id
EOS = 1


# --- 1. vocabulary and the legacy reading -------------------------------------


def test_the_vocabulary_is_closed_and_off_is_first():
    # `on` (thinking at the template's default, no effort variable) and
    # `high` (a probe candidate) joined 2026-09-05 with the capability
    # record; which LEVELS a model may declare is the pinned template's
    # answer, not the vocabulary's.
    assert prompt_render.REASONING_EFFORTS == (
        "off", "on", "low", "medium", "high", "xhigh")
    assert er.REASONING_EFFORTS == prompt_render.REASONING_EFFORTS
    assert prompt_render.has_thinking_mode(QWEN) is er.has_thinking_mode(QWEN)
    assert prompt_render.has_thinking_mode(GEMMA) is er.has_thinking_mode(GEMMA)


def test_a_legacy_boolean_manifest_reads_as_off_or_xhigh_and_is_exempt():
    """A frozen manifest under the old spelling: thinking on meant the chat
    template's DEFAULT effort (xhigh on Qwen3.8), and no budget was ever
    declared — so it must load, read as xhigh, and verify without the joint
    rules firing. Otherwise the whole ladder program stops verifying."""
    off = Manifest.from_dict({"name": "s", "modelID": QWEN,
                              "qwenThinkingEnabled": False})
    on = Manifest.from_dict({"name": "s", "modelID": QWEN,
                             "qwenThinkingEnabled": True})
    assert off.reasoning_effort == "off" and not off.qwen_thinking_enabled
    assert on.reasoning_effort == "xhigh" and on.qwen_thinking_enabled
    assert on.reasoning_max_tokens is None
    assert not on.reasoning_effort_declared
    # The joint rules are silent for the legacy spelling — no budget, and
    # even a Gemma id (the family rule is a declaration-time rule).
    legacy_gemma = Manifest.from_dict({"name": "s", "modelID": GEMMA,
                                       "qwenThinkingEnabled": True,
                                       "conditions": [{"name": "c", "slots": []}]})
    assert not [v for v in legacy_gemma.verify(root="/nonexistent")
                if "reasoning" in v.lower()]


def test_the_new_spelling_wins_and_reads_its_budget():
    manifest = Manifest.from_dict({"name": "s", "modelID": QWEN,
                                   "reasoningEffort": "low",
                                   "reasoningMaxTokens": 300,
                                   "qwenThinkingEnabled": False})
    assert manifest.reasoning_effort == "low"
    assert manifest.reasoning_max_tokens == 300
    assert manifest.reasoning_effort_declared
    assert manifest.qwen_thinking_enabled
    # A legacy `true` beside an explicit `off` (a pre-effort toggle written
    # onto a fresh manifest, which carries `off` by default) still means the
    # template's default effort: the boolean is the more deliberate
    # declaration there, and reading it as off would silently drop a thinking
    # mode the study asked for. A non-off effort beside it wins outright.
    mixed = Manifest.from_dict({"name": "s", "modelID": QWEN,
                                "reasoningEffort": "off",
                                "qwenThinkingEnabled": True})
    assert mixed.reasoning_effort == "xhigh" and mixed.qwen_thinking_enabled
    assert prompt_render.read_reasoning_effort(
        {"reasoningEffort": "low", "qwenThinkingEnabled": True}) == "low"
    assert prompt_render.read_reasoning_effort(
        {"reasoningEffort": "off", "qwenThinkingEnabled": False}) == "off"


def test_create_writes_the_effort_and_never_the_boolean(tmp_path):
    root = str(tmp_path)
    document = es.create("s", model_id=QWEN, root=root)
    assert document["reasoningEffort"] == "off"
    assert "qwenThinkingEnabled" not in document
    assert "qwenThinkingEnabled" not in es.PROTOCOL_FIELDS
    assert "reasoningEffort" in es.PROTOCOL_FIELDS
    assert "reasoningMaxTokens" in es.PROTOCOL_FIELDS


# --- 2. the declaration rules --------------------------------------------------


def _draft(tmp_path, model_id=QWEN):
    root = str(tmp_path)
    es.create("s", model_id=model_id, root=root)
    return root


def test_a_non_off_effort_needs_its_budget(tmp_path):
    root = _draft(tmp_path)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.set_protocol("s", {"reasoningEffort": "low"}, root=root)
    assert "needs a reasoningMaxTokens" in str(caught.value)
    assert "reasoningMaxTokens=<n" in caught.value.repair_action
    # Nothing was written.
    assert es.load_raw("s", root)["reasoningEffort"] == "off"
    # Together, they land.
    document = es.set_protocol(
        "s", {"reasoningEffort": "low", "reasoningMaxTokens": 400}, root=root)
    assert document["reasoningEffort"] == "low"
    assert document["reasoningMaxTokens"] == 400
    # …and a budget alone, on a draft that already reasons, is fine.
    assert es.set_protocol("s", {"reasoningMaxTokens": 800},
                           root=root)["reasoningMaxTokens"] == 800


def test_off_takes_no_budget_and_retires_one(tmp_path):
    root = _draft(tmp_path)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.set_protocol("s", {"reasoningMaxTokens": 400}, root=root)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON in str(caught.value)
    es.set_protocol("s", {"reasoningEffort": "medium",
                          "reasoningMaxTokens": 400}, root=root)
    # Declaring off in the same call as a budget says two things: refused.
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("s", {"reasoningEffort": "off",
                              "reasoningMaxTokens": 400}, root=root)
    # Declaring off alone retires the budget the draft carried.
    document = es.set_protocol("s", {"reasoningEffort": "off"}, root=root)
    assert document["reasoningEffort"] == "off"
    assert "reasoningMaxTokens" not in document


def test_the_effort_is_closed_vocabulary_and_the_budget_a_positive_int(tmp_path):
    root = _draft(tmp_path)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.set_protocol("s", {"reasoningEffort": "hgih",
                              "reasoningMaxTokens": 4}, root=root)
    assert "unknown reasoningEffort 'hgih'" in str(caught.value)
    for bad in (0, -1, 1.5, "4", True):
        with pytest.raises(es.ExperimentStoreError) as caught:
            es.set_protocol("s", {"reasoningEffort": "low",
                                  "reasoningMaxTokens": bad}, root=root)
        assert "positive integer" in str(caught.value)


def test_a_family_without_a_thinking_mode_refuses_a_non_off_effort(tmp_path):
    root = _draft(tmp_path, model_id=GEMMA)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.set_protocol("s", {"reasoningEffort": "low",
                              "reasoningMaxTokens": 100}, root=root)
    assert "whose chat template has no thinking switch" in str(caught.value)
    assert GEMMA in str(caught.value)
    # Off is always declarable.
    assert es.set_protocol("s", {"reasoningEffort": "off"},
                           root=root)["reasoningEffort"] == "off"


def test_writing_the_effort_drops_the_legacy_boolean_from_a_draft(tmp_path):
    root = _draft(tmp_path)
    raw = es.load_raw("s", root)
    del raw["reasoningEffort"]
    raw["qwenThinkingEnabled"] = True
    es.save_raw(raw, root)
    assert Manifest.load("s", root=root).reasoning_effort == "xhigh"
    document = es.set_protocol("s", {"reasoningEffort": "low",
                                     "reasoningMaxTokens": 64}, root=root)
    assert "qwenThinkingEnabled" not in document
    assert document["reasoningEffort"] == "low"


def test_verify_applies_the_joint_rules_to_the_new_spelling_only():
    base = {"name": "s", "modelID": QWEN,
            "conditions": [{"name": "c", "slots": []}]}
    problems = Manifest.from_dict({**base, "reasoningEffort": "low"}).verify(
        root="/nonexistent")
    assert any("needs a reasoningMaxTokens" in v for v in problems)
    problems = Manifest.from_dict({**base, "reasoningEffort": "off",
                                   "reasoningMaxTokens": 9}).verify(
        root="/nonexistent")
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON in problems
    problems = Manifest.from_dict({**base, "modelID": GEMMA,
                                   "reasoningEffort": "xhigh",
                                   "reasoningMaxTokens": 9}).verify(
        root="/nonexistent")
    assert any("no thinking switch" in v for v in problems)
    problems = Manifest.from_dict({**base, "reasoningEffort": "nope"}).verify(
        root="/nonexistent")
    assert any("unknown reasoningEffort" in v for v in problems)
    clean = Manifest.from_dict({**base, "reasoningEffort": "medium",
                                "reasoningMaxTokens": 9}).verify(
        root="/nonexistent")
    assert not [v for v in clean if "reasoning" in v.lower()]


def test_the_answer_token_gate_is_keyed_on_the_effort():
    manifest = Manifest.from_dict({
        "name": "s", "modelID": QWEN, "reasoningEffort": "low",
        "reasoningMaxTokens": 9, "outcomeInstruments": ["choiceProbability"],
        "conditions": [{"name": "c", "slots": []}]})
    problems = manifest.verify(root="/nonexistent")
    assert any("reasoningEffort is 'low' (thinking mode on)" in v
               for v in problems)
    legacy = Manifest.from_dict({
        "name": "s", "modelID": QWEN, "qwenThinkingEnabled": True,
        "outcomeInstruments": ["choiceProbability"],
        "conditions": [{"name": "c", "slots": []}]})
    assert any("reasoningEffort is 'xhigh'" in v
               for v in legacy.verify(root="/nonexistent"))


# --- 2b. the extraction rendering ---------------------------------------------


def test_the_rendering_reads_both_spellings_but_refuses_them_together():
    legacy_on = er.from_json({"mode": "chatTemplate", "qwenThinkingEnabled": True})
    assert legacy_on.reasoning_effort == "xhigh"
    assert legacy_on.qwen_thinking_enabled
    legacy_off = er.from_json({"mode": "chatTemplate", "qwenThinkingEnabled": False})
    assert legacy_off.reasoning_effort == "off"
    new = er.from_json({"mode": "chatTemplate", "reasoningEffort": "medium"})
    assert new.reasoning_effort == "medium" and new.qwen_thinking_enabled
    with pytest.raises(er.ExtractionRenderingError) as caught:
        er.from_json({"mode": "chatTemplate", "qwenThinkingEnabled": True,
                      "reasoningEffort": "xhigh"})
    assert str(caught.value) == er.BOTH_THINKING_KEYS_REASON
    with pytest.raises(er.ExtractionRenderingError) as caught:
        er.from_json({"mode": "chatTemplate", "reasoningEffort": "hgih"})
    assert str(caught.value) == er.unknown_effort_reason("hgih")
    # Explicit nulls declare nothing, as everywhere else in this block.
    assert er.from_json({"mode": "chatTemplate", "qwenThinkingEnabled": None,
                         "reasoningEffort": None}).reasoning_effort == "off"


def test_the_identity_fragment_keeps_the_boolean_spelling_for_off_and_xhigh():
    """Hash stability: every recipe frozen before the effort existed hashed
    `qwenThinkingEnabled`, and a recipe that declares xhigh today renders the
    identical scaffold — so both spellings of on/off must hash the same."""
    legacy = er.from_json({"mode": "chatTemplate", "qwenThinkingEnabled": True})
    new = er.from_json({"mode": "chatTemplate", "reasoningEffort": "xhigh"})
    assert er.canonical_identity_fragment(legacy) == \
        er.canonical_identity_fragment(new) == {
            "addGenerationPrompt": True, "mode": "chatTemplate",
            "qwenThinkingEnabled": True, "systemPrompt": None}
    off = er.from_json({"mode": "chatTemplate"})
    assert "reasoningEffort" not in er.canonical_identity_fragment(off)
    low = er.from_json({"mode": "chatTemplate", "reasoningEffort": "low"})
    fragment = er.canonical_identity_fragment(low)
    assert list(fragment) == ["addGenerationPrompt", "mode",
                              "qwenThinkingEnabled", "reasoningEffort",
                              "systemPrompt"]
    assert fragment["reasoningEffort"] == "low"
    assert fragment["qwenThinkingEnabled"] is True


def test_a_new_stamp_spells_the_effort_and_the_family_gate_is_asked_by_attach(
        tmp_path):
    assert er.from_json({"mode": "chatTemplate"}).to_dict() == {
        "mode": "chatTemplate", "addGenerationPrompt": True,
        "reasoningEffort": "off"}
    assert er.thinking_mode_problem(
        er.from_json({"mode": "chatTemplate", "reasoningEffort": "low"}),
        QWEN) is None
    problem = er.thinking_mode_problem(
        er.from_json({"mode": "chatTemplate", "reasoningEffort": "low"}), GEMMA)
    assert problem == er.effort_without_thinking_mode_reason("low", GEMMA)
    assert er.thinking_mode_problem(er.RAW_RENDERING, GEMMA) is None
    # Through the attach verb, with the model id it knows.
    root = str(tmp_path)
    concept_dir = os.path.join(root, "prompts", "concepts", "joy")
    os.makedirs(concept_dir)
    for pole in ("positive", "negative"):
        with open(os.path.join(concept_dir, f"{pole}.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps({"text": f"{pole} joy"}) + "\n")
    es.create("g", model_id=GEMMA, root=root)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.attach("g", ["joy"], root=root,
                  extraction_rendering='{"mode":"chatTemplate",'
                                       '"reasoningEffort":"low"}')
    assert "no thinking switch" in str(caught.value)
    assert "reasoningEffort" in caught.value.repair_action


# --- 3. the two-phase budget --------------------------------------------------


def test_the_budget_rule_counts_reasoning_then_answer():
    budget = tg.ReasoningBudget(reasoning_max_tokens=3, max_tokens=2,
                                close_id=CLOSE)
    assert budget.outer_bound == 5
    assert budget.observe(10) is None
    assert budget.observe(CLOSE) is None          # closes on the 2nd token
    assert budget.closed and budget.reasoning_tokens == 2
    assert budget.observe(11) is None
    assert budget.observe(12) == tg.FINISH_LENGTH   # 2 answer tokens: full
    # The close token may land on the reasoning block's LAST budgeted step.
    late = tg.ReasoningBudget(reasoning_max_tokens=3, max_tokens=2,
                              close_id=CLOSE)
    assert [late.observe(t) for t in (10, 11, CLOSE)] == [None, None, None]
    assert late.closed
    # …and one step later than that is the reasoning cap.
    capped = tg.ReasoningBudget(reasoning_max_tokens=3, max_tokens=2,
                                close_id=CLOSE)
    assert [capped.observe(t) for t in (10, 11)] == [None, None]
    assert capped.observe(12) == tg.FINISH_LENGTH_IN_REASONING


def test_finish_reason_splits_at_the_first_close_token():
    kw = dict(max_tokens=3, stop_ids={EOS}, reasoning_max_tokens=4,
              think_close_id=CLOSE)
    # Never closed, budget spent: cut off inside the reasoning block.
    assert tg.finish_reason([10, 11, 12, 13], **kw) == tg.FINISH_LENGTH_IN_REASONING
    # Never closed, ended itself (EOS mid-reasoning): honestly a stop.
    assert tg.finish_reason([10, EOS], **kw) == tg.FINISH_STOP
    # Closed, then a short answer.
    assert tg.finish_reason([10, CLOSE, 20], **kw) == tg.FINISH_STOP
    # Closed, then exactly the answer budget with no stop token: the cap.
    assert tg.finish_reason([10, CLOSE, 20, 21, 22], **kw) == tg.FINISH_LENGTH
    # …and with EOS on the last budgeted answer step: a late natural ending.
    assert tg.finish_reason([10, CLOSE, 20, 21, EOS], **kw) == tg.FINISH_STOP
    # Reasoning of exactly the budget, closed on its last step: not a cap.
    assert tg.finish_reason([10, 11, 12, CLOSE, 20], **kw) == tg.FINISH_STOP
    # Without a reasoning budget the reading is the single-budget one.
    assert tg.finish_reason([10, CLOSE, 20, 21, 22], max_tokens=3) == \
        tg.FINISH_LENGTH
    assert tg.FINISH_REASONS == ("stop", "length", "lengthInReasoning",
                                 "cancelled")
    assert tg.CUT_OFF_REASONS == {"length", "lengthInReasoning"}


def test_the_close_token_id_is_read_from_the_tokenizer_or_absent():
    tok = SimpleNamespace(convert_tokens_to_ids=lambda t: CLOSE if t == "</think>" else 0,
                          unk_token_id=0)
    assert tg.think_close_token_id(tok) == CLOSE
    # A family whose vocabulary has no such token answers its unk id: None.
    unk = SimpleNamespace(convert_tokens_to_ids=lambda t: 0, unk_token_id=0)
    assert tg.think_close_token_id(unk) is None
    assert tg.think_close_token_id(SimpleNamespace()) is None
    assert tg.think_close_token_id(None) is None


def test_the_stopping_criterion_feeds_the_rule_and_stops_where_it_says():
    import threading

    import torch

    from steerlab_server.experiment import generate as gen

    prompt = [1, 2, 3]
    budget = tg.ReasoningBudget(reasoning_max_tokens=2, max_tokens=2,
                                close_id=CLOSE)
    criteria = gen._stopping_criteria(threading.Event(), budget=budget,
                                      prompt_length=len(prompt))
    assert len(criteria) == 2

    def step(*generated):
        ids = torch.tensor([prompt + list(generated)])
        return any(bool(c(ids, None)) for c in criteria)

    # One reasoning token: keep going; two without a close: the reasoning cap.
    assert step(10) is False
    assert step(10, 11) is True
    assert criteria[1].verdict == tg.FINISH_LENGTH_IN_REASONING

    budget = tg.ReasoningBudget(reasoning_max_tokens=2, max_tokens=2,
                                close_id=CLOSE)
    criteria = gen._stopping_criteria(threading.Event(), budget=budget,
                                      prompt_length=len(prompt))
    assert step(10) is False
    assert step(10, CLOSE) is False
    assert step(10, CLOSE, 20) is False
    assert step(10, CLOSE, 20, 21) is True
    assert criteria[1].verdict == tg.FINISH_LENGTH
    # The criterion is incremental: it never re-feeds ids it has seen.
    assert criteria[1].fed == 4


def test_the_stream_reserves_both_budgets_and_hands_the_framework_their_sum(
        monkeypatch):
    """Everything `_stream_rendered` decides before a model runs: the context
    check sees prompt + reasoning + answer, `max_new_tokens` is the sum, and
    the budget criterion is armed — pinned by intercepting the seams."""
    from steerlab_server.experiment import generate as gen
    from steerlab_server.experiment import prompt_render as pr

    seen = {}
    monkeypatch.setattr(gen, "check_context_budget",
                        lambda model, prompt_tokens, requested: seen.update(
                            prompt=prompt_tokens, requested=requested))
    original = gen._generation_kwargs

    def spy_kwargs(input_ids, max_tokens, temperature, tokenizer, model=None):
        seen["max_new_tokens"] = max_tokens
        raise RuntimeError("stop here")  # nothing past the arithmetic runs
    monkeypatch.setattr(gen, "_generation_kwargs", spy_kwargs)
    model = SimpleNamespace(
        model_id=QWEN, device="cpu",
        tokenizer=SimpleNamespace(
            convert_tokens_to_ids=lambda t: CLOSE, unk_token_id=0,
            pad_token_id=0, eos_token_id=EOS))
    rendered = pr.RenderedPrompt(text="x", input_ids=[1, 2, 3, 4],
                                 prompt_token_count=4)
    with pytest.raises(RuntimeError, match="stop here"):
        list(gen._stream_rendered(model, rendered, max_tokens=5,
                                  temperature=0.0, reasoning_max_tokens=30))
    assert seen == {"prompt": 4, "requested": 35, "max_new_tokens": 35}
    seen.clear()
    with pytest.raises(RuntimeError, match="stop here"):
        list(gen._stream_rendered(model, rendered, max_tokens=5,
                                  temperature=0.0))
    assert seen == {"prompt": 4, "requested": 5, "max_new_tokens": 5}
    assert original is not spy_kwargs


# --- 4. the preflights --------------------------------------------------------


def test_the_context_preflight_reserves_both_budgets(monkeypatch):
    from steerlab_server.experiment import token_preflight as tp

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            return " ".join(m["content"] for m in messages)

        def __call__(self, text, add_special_tokens=True):
            return SimpleNamespace(input_ids=list(range(len(text.split()))))

    monkeypatch.setattr(tp, "_tokenizer", lambda model_id, revision: Tok())
    monkeypatch.setattr(tp, "context_window", lambda model_id, revision: 100)
    prompts = [{"id": "a", "prompt": "one two three four five six seven eight"}]
    plain = tp.preflight(prompts, model_id=QWEN, revision=None, max_tokens=50)
    assert plain["promptBudget"] == 100 - 50 - tp.CONTEXT_BUDGET_RESERVE
    assert plain["overflowCount"] == 0
    reasoning = tp.preflight(prompts, model_id=QWEN, revision=None,
                             max_tokens=50, reasoning_effort="low",
                             reasoning_max_tokens=30)
    assert reasoning["reasoningMaxTokens"] == 30
    assert reasoning["promptBudget"] == 100 - 50 - 30 - tp.CONTEXT_BUDGET_RESERVE
    assert reasoning["overflowCount"] == 1
    text = tp.refusal(reasoning)
    assert "50 for the answer plus 30 for the reasoning block" in text
    assert "reasoningMaxTokens" in text


def test_the_walltime_check_prices_both_budgets(tmp_path, monkeypatch):
    from steerlab_server.api import instrument_family as fam
    from steerlab_server.api import submissions as sub
    from steerlab_server.api.executors import SlurmResources
    from steerlab_server.api.profile import ServerProfile

    meta = tmp_path / "meta"
    meta.mkdir()
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(meta))
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE", str(tmp_path / "absent-key"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    model = "acme/tiny"
    (meta / "throughput.json").write_text(json.dumps({
        "schemaVersion": 1, "foldedJobIds": [],
        "entries": [{"modelId": model, "gpuType": "A100",
                     "recordsPerHour": 600.0, "samples": 4,
                     "instrumentFamily": fam.LONG_FORM_TEXT,
                     "tokensBasis": 256,
                     "updatedAt": "2026-08-19T00:00:00+00:00"}],
    }), encoding="utf-8")

    def manifest(**overrides):
        data = {"name": "study", "modelID": model, "modelRevision": "abc123",
                "taskPromptsFile": "prompts/tasks/t.jsonl",
                "conditions": [{"name": "c", "slots": []}]}
        data.update(overrides)
        return Manifest.from_dict(data)

    def walltime(m):
        return sub._check_walltime(
            m, SlurmResources(gres="gpu:A100:1", walltime="04:00:00",
                              gpu_vram_gb={"A100": 80}),
            1000, ServerProfile.from_env(), verb="run")

    answer_only = walltime(manifest(maxTokens=256))
    both = walltime(manifest(maxTokens=256, reasoningEffort="low",
                             reasoningMaxTokens=768))
    startup = sub.PREFLIGHT_JOB_STARTUP_HOURS
    assert both["data"]["maxTokens"] == 1024
    assert both["data"]["reasoningMaxTokens"] == 768
    assert both["data"]["answerMaxTokens"] == 256
    assert both["data"]["estimatedHours"] - startup == pytest.approx(
        (answer_only["data"]["estimatedHours"] - startup) * 4, abs=0.01)


# --- 5. the gate and the report -----------------------------------------------


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(
        model_id=model_id, revision=revision or "abc",
        tokenizer=SimpleNamespace(
            convert_tokens_to_ids=lambda t: CLOSE if t == "</think>" else 0,
            unk_token_id=0, eos_token_id=EOS))


def _study(root, name, *, threshold=None, reasoning_max_tokens=4,
           max_tokens=3, samples=2):
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm"}\n')
    es.create(name, model_id=QWEN, revision="abc", root=root)
    raw = es.load_raw(name, root)
    raw.update({"seeds": [0], "temperature": 0.7, "samplesPerItem": samples,
                "seedPolicy": "derivedSHA256", "maxTokens": max_tokens,
                "reasoningEffort": "low",
                "reasoningMaxTokens": reasoning_max_tokens,
                "outcomeInstruments": ["sampledText"]})
    if threshold is not None:
        raw[tg.MANIFEST_KEY] = threshold
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path, "\n".join(
        json.dumps({"id": item, "prompt": f"Write about {item}."})
        for item in ("p0", "p1", "p2")) + "\n")
    return prompts_path


def _fake_generate(calls):
    """p0 reasons and answers briefly; p1 never closes its reasoning block
    and spends the whole reasoning budget; p2 closes and then spends the
    whole answer budget. Identical text on all three."""
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False, token_ids_out=None, **kwargs):
        calls.append(dict(kwargs, max_tokens=max_tokens,
                          qwen_thinking_enabled=qwen_thinking_enabled))
        reasoning_cap = kwargs.get("reasoning_max_tokens") or 0
        if token_ids_out is not None:
            if "p1" in prompt:
                token_ids_out.extend([10] * reasoning_cap)
            elif "p2" in prompt:
                token_ids_out.extend([10, CLOSE] + [20] * max_tokens)
            else:
                token_ids_out.extend([10, CLOSE, 20])
        return "a sentence about the case"
    return generate


def _records(run_dir):
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def test_the_run_loop_threads_the_protocol_and_stamps_the_fourth_reason(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "reason")
    calls = []
    monkeypatch.setattr(tasks, "_extract_all", lambda model, manifest, root: {})
    monkeypatch.setattr(tasks, "generate", _fake_generate(calls))
    run_dir = tasks.run("reason", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    # The protocol reached every generate() call, effort and budget alike.
    assert calls and all(c["reasoning_effort"] == "low" for c in calls)
    assert all(c["reasoning_max_tokens"] == 4 for c in calls)
    assert all(c["qwen_thinking_enabled"] for c in calls)
    assert all(c["max_tokens"] == 3 for c in calls)
    by_item = {}
    for record in _records(run_dir):
        by_item.setdefault(record["promptID"], set()).add(record[tg.RECORD_KEY])
    assert by_item == {"p0": {tg.FINISH_STOP},
                       "p1": {tg.FINISH_LENGTH_IN_REASONING},
                       "p2": {tg.FINISH_LENGTH}}
    # The report distinguishes the two caps, and counts both as cut off.
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as handle:
        truncation = json.load(handle)["truncation"]
    assert truncation["classified"] == 6
    assert truncation["lengthStopped"] == 4
    assert truncation["lengthStoppedInReasoning"] == 2
    cells = {c["promptID"]: c for c in truncation["cells"]}
    assert cells["p1"]["lengthStoppedInReasoning"] == 2
    assert cells["p1"]["lengthStopped"] == 2
    assert cells["p2"]["lengthStoppedInReasoning"] == 0
    assert cells["p2"]["lengthStopped"] == 2
    # …and so does summaries.csv.
    with open(os.path.join(run_dir, "summaries.csv"), encoding="utf-8") as handle:
        rows = {r["promptID"]: r for r in csv.DictReader(handle)}
    assert rows["p1"]["lengthStopped"] == "2"
    assert rows["p1"]["lengthStoppedInReasoning"] == "2"
    assert rows["p2"]["lengthStoppedInReasoning"] == "0"
    assert rows["p0"]["lengthStopped"] == "0"


def test_a_reasoning_capped_cell_is_incomplete_and_the_refusal_names_the_cap(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "gated", threshold=0.5)
    monkeypatch.setattr(tasks, "_extract_all", lambda model, manifest, root: {})
    monkeypatch.setattr(tasks, "generate", _fake_generate([]))
    with pytest.raises(RuntimeError) as caught:
        tasks.run("gated", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)
    text = str(caught.value)
    assert "item 'p1'" in text
    assert "2 inside the reasoning block at the 4-token reasoning cap" in text
    assert "0 in the answer at the 3-token answer cap" in text
    assert getattr(caught.value, "gate", None) == lifecycle_gates.LENGTH_STOPPED
    repair = getattr(caught.value, "repair_action", "")
    assert "--reasoning-max-tokens <m>" in repair
    assert "m above 4" in repair


def test_the_tally_and_the_refusal_helpers_distinguish_the_caps():
    records = [
        {"condition": "c", "promptID": "p", tg.RECORD_KEY: tg.FINISH_STOP},
        {"condition": "c", "promptID": "p", tg.RECORD_KEY: tg.FINISH_LENGTH},
        {"condition": "c", "promptID": "p",
         tg.RECORD_KEY: tg.FINISH_LENGTH_IN_REASONING},
        {"condition": "c", "promptID": "p", tg.RECORD_KEY: tg.FINISH_CANCELLED},
    ]
    tally = tg.Tally(records)
    assert tally.cell("c", "p") == (4, 2)
    assert tally.cell_in_reasoning("c", "p") == 1
    assert tg.report(records, threshold=None) == {
        "threshold": None, "classified": 4, "lengthStopped": 2,
        "lengthStoppedInReasoning": 1, "lengthStoppedFraction": 0.5,
        "cells": [{"condition": "c", "promptID": "p", "classified": 4,
                   "lengthStopped": 2, "lengthStoppedInReasoning": 1,
                   "lengthStoppedFraction": 0.5}]}
    # Without a reasoning budget the sentence is the one it always was.
    plain = tg.cell_refusal(4, 2, threshold=0.25, condition="c", prompt_id="p",
                            max_tokens=8)
    assert "stopped at the 8-token cap instead of finishing (50.0%)" in plain
    both = tg.cell_refusal(4, 2, threshold=0.25, condition="c", prompt_id="p",
                           max_tokens=8, length_stopped_in_reasoning=1,
                           reasoning_max_tokens=64)
    assert "1 inside the reasoning block at the 64-token reasoning cap" in both
    assert "1 in the answer at the 8-token answer cap" in both
    assert tg.repair_action("s", 8) == (
        "steerlab-cli experiment set-sampling s --max-tokens <n>  (n above 8), "
        "then re-run; a frozen study is iterated by duplicating first: "
        "steerlab-cli experiment duplicate s s-v2")
    assert "--reasoning-max-tokens <m>" in tg.repair_action(
        "s", 8, reasoning_max_tokens=64)


# --- 6. the preregistration line ----------------------------------------------


def test_the_preregistration_prints_the_effort_and_both_budgets(tmp_path):
    root = str(tmp_path)
    for name, fields in (("plain", {}),
                         ("reasoned", {"reasoningEffort": "medium",
                                       "reasoningMaxTokens": 2048})):
        es.create(name, model_id=QWEN, root=root)
        raw = es.load_raw(name, root)
        raw.update(fields)
        raw["maxTokens"] = 512
        es.save_raw(raw, root)
        es._write_preregistration(raw, root)
        with open(os.path.join(root, "experiments", name,
                               "preregistration.md"), encoding="utf-8") as handle:
            line = next(l for l in handle if l.startswith("- **Sampling:**"))
        if name == "plain":
            assert ("reasoningEffort off, maxTokens 512, "
                    "reasoningMaxTokens none") in line
        else:
            assert ("reasoningEffort medium, maxTokens 512, "
                    "reasoningMaxTokens 2048") in line

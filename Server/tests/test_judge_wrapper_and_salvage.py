"""The 2026-07-22 judging incident round: truncated-verdict salvage, the
unified judge generation cap, and the golden-pinned prompt wrapper.

A cluster paired-judge evaluate failed after 15 minutes because a local
judge stated a valid winner and confidence FIRST, then wrote an essay-length
reasoning that outran its 512-token cap — the JSON never closed, balanced-
brace parsing raised, the retry rambled the same way, and a legible verdict
was refused. Contract under test here:

- salvage: a truncated response with a COMPLETE ``"winner"`` (exactly
  A/B/tie) is accepted with ``reasoningTruncated: true`` — no retry burned;
  truncation BEFORE the winner completes (or an out-of-vocabulary winner)
  keeps the existing refusal path;
- cap: ``paired_judge.JUDGE_MAX_TOKENS`` (8192 since the 2026-08-06
  reasoning-accommodation round — see
  ``test_judge_reasoning_accommodation.py``; twin of Swift's
  ``PairedJudgeBudget.maxTokens``) is what every judge generation on this
  engine actually passes — Claude, OpenRouter, the evaluate local judge,
  and the sweep local judge;
- wrapper: ``build_prompt`` is byte-identical to the committed goldens in
  ``prompts/fixtures/paired-judge/`` (Swift twin pinned to the same bytes).

Swift twin: Tests/ExperimentKitTests/PairedJudgeWrapperAndSalvageTests.swift.
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import paired_judge, tasks
from steerlab_server.experiment.manifest import JudgeRef, Manifest

# The exact incident shape (cluster evaluate, 2026-07-22): fenced JSON,
# verdict fields complete, reasoning cut mid-word by the token cap.
INCIDENT_TEXT = (
    '```json\n{"winner": "A", "confidence": 0.95, "reasoning": "Response A '
    "exhibits a more distinctly Californian sensibility in its treatment of "
    "the evo")

GOLDEN_DIR = (Path(__file__).resolve().parent.parent.parent
              / "prompts" / "fixtures" / "paired-judge")


# --- truncated-verdict salvage ------------------------------------------------


def test_salvages_the_exact_incident_shape():
    verdict = paired_judge.parse_response(INCIDENT_TEXT)
    assert verdict["winner"] == "A"
    assert verdict["confidence"] == 0.95
    assert verdict["reasoningTruncated"] is True
    assert verdict["brief_reason"].startswith("Response A exhibits")
    assert verdict["brief_reason"].endswith("the evo")


def test_salvaged_verdict_consumes_no_retry():
    # The winner was explicitly written — valid_verdict must accept it on
    # the FIRST call, never burning the single retry on a legible verdict.
    calls = []

    def generate(prompt):
        calls.append(1)
        return INCIDENT_TEXT

    judge = paired_judge.make_local_judge(generate)
    verdict = paired_judge.valid_verdict(
        judge, "m", "r", "a", "b", None,
        judge_label="'google/gemma-3-4b-it'", item_label="pair fear/p0")
    assert verdict["winner"] == "A"
    assert verdict["reasoningTruncated"] is True
    assert len(calls) == 1


def test_complete_fenced_json_still_parses_without_truncation_mark():
    verdict = paired_judge.parse_response(
        '```json\n{"winner": "B", "confidence": 0.7, '
        '"brief_reason": "B is tighter."}\n```')
    assert verdict["winner"] == "B"
    assert "reasoningTruncated" not in verdict


def test_truncation_before_the_winner_completes_still_refuses():
    # No closing quote after the winner value — nothing legible to salvage;
    # the existing ValueError path (and its retry-then-refuse machinery)
    # stays engaged.
    for text in ('```json\n{"winner": "A',
                 '```json\n{"win',
                 '```json\n{"winner": "',
                 '{"confidence": 0.9, "reasoning": "A seems better'):
        with pytest.raises(ValueError, match="unbalanced JSON"):
            paired_judge.parse_response(text)


def test_truncation_refusal_engages_the_retry_machinery():
    calls = []

    def generate(prompt):
        calls.append(1)
        return '```json\n{"winner": "'

    judge = paired_judge.make_local_judge(generate)
    with pytest.raises(RuntimeError, match="invalid verdict twice"):
        paired_judge.valid_verdict(
            judge, "m", "r", "a", "b", None,
            judge_label="'j1'", item_label="pair c/p0")
    assert len(calls) == 2  # retried once, then refused


def test_valid_json_with_braces_inside_strings_parses_exactly():
    """Finding 5 (2026-07-23): the old brace counter counted braces inside
    quoted strings, so this VALID verdict — whose reason mentions braces —
    was misread as truncated. The real decoder is string-aware; the verdict
    parses completely, with no truncation mark."""
    text = ('Sure — here is my verdict.\n'
            '{"winner": "B", "confidence": 0.8, '
            '"brief_reason": "B closes every {brace} it opens; A writes } '
            'without opening."}\n')
    verdict = paired_judge.parse_response(text)
    assert verdict["winner"] == "B"
    assert verdict["confidence"] == 0.8
    assert "brace" in verdict["brief_reason"]
    assert "reasoningTruncated" not in verdict


def test_prose_brace_before_the_json_object_still_parses():
    # A "{" in prose ahead of the real object: the decoder walks forward to
    # the first position that decodes as a JSON object.
    text = ('I weighed {several factors} before deciding.\n'
            '{"winner": "A", "confidence": 0.6, "brief_reason": "A."}')
    verdict = paired_judge.parse_response(text)
    assert verdict["winner"] == "A"
    assert "reasoningTruncated" not in verdict


def test_duplicate_winner_fields_in_truncation_refuse():
    """Finding 5 (2026-07-23): a truncated response carrying TWO complete
    winner fields — conflicting or duplicated — has no single legible
    verdict; salvage must refuse into the retry-then-refuse path instead of
    parsing whichever came first."""
    conflicting = ('{"winner": "A", "notes": {"winner": "B", '
                   '"confidence": 0.9, "brief_reason": "an essay that never')
    with pytest.raises(ValueError, match="unbalanced JSON"):
        paired_judge.parse_response(conflicting)
    agreeing = ('{"winner": "A", "restated": {"winner": "A", '
                '"brief_reason": "an essay that never')
    with pytest.raises(ValueError, match="unbalanced JSON"):
        paired_judge.parse_response(agreeing)


def test_duplicate_winner_truncation_engages_retry_then_refuse():
    calls = []

    def generate(prompt):
        calls.append(1)
        return ('{"winner": "A", "notes": {"winner": "B", '
                '"brief_reason": "cut off mid')

    judge = paired_judge.make_local_judge(generate)
    with pytest.raises(RuntimeError, match="invalid verdict twice"):
        paired_judge.valid_verdict(
            judge, "m", "r", "a", "b", None,
            judge_label="'j1'", item_label="pair c/p0")
    assert len(calls) == 2


def test_salvage_never_accepts_an_out_of_vocabulary_winner():
    # A complete winner field whose value is outside A/B/tie is invented
    # data, truncated or not — refuse.
    for text in ('{"winner": "C", "confidence": 0.9, "reasoning": "cut',
                 '{"winner": "AB", "confidence": 0.9, "reasoning": "cut',
                 '{"winner": "A and B are equal", "reasoning": "cut',
                 '{"winner": "both", "reasoning": "cut'):
        with pytest.raises(ValueError, match="unbalanced JSON"):
            paired_judge.parse_response(text)


def test_salvage_drops_an_incomplete_confidence_number():
    # The text ends mid-number: "0.9" could have been 0.95 — a number
    # without a trailing delimiter is not complete and is dropped.
    verdict = paired_judge.parse_response('{"winner": "tie", "confidence": 0.9')
    assert verdict["winner"] == "tie"
    assert "confidence" not in verdict
    assert verdict["reasoningTruncated"] is True
    assert verdict["brief_reason"] == ""


def test_salvage_unescapes_the_reasoning_prefix():
    verdict = paired_judge.parse_response(
        '{"winner": "B", "confidence": 0.6, '
        '"brief_reason": "Cites \\"Erie\\" precisely.\\nAlso ')
    assert verdict["brief_reason"] == 'Cites "Erie" precisely.\nAlso '
    assert verdict["reasoningTruncated"] is True


# --- the unified generation cap -----------------------------------------------


def test_the_unified_cap_is_8192_and_named():
    # Twin constant: Swift PairedJudgeBudget.maxTokens — keep identical.
    # Raised from 1024 on 2026-08-06: the cap bounds runaway generation, it
    # must never ration a reasoning model's thinking (two incidents, see
    # test_judge_reasoning_accommodation.py).
    assert paired_judge.JUDGE_MAX_TOKENS == 8192


def test_openrouter_judge_sends_the_unified_cap(monkeypatch, tmp_path):
    httpx = pytest.importorskip("httpx")
    key_file = tmp_path / "judge-key"
    key_file.write_text(json.dumps({"kind": "openrouter", "key": "sk-or-t"}),
                        encoding="utf-8")
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE", str(key_file))
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={
            "provider": "Anthropic",
            "choices": [{"message": {"content": json.dumps(
                {"winner": "A", "confidence": 0.8, "brief_reason": "r"})}}],
        })

    paired_judge.openrouter_judge_pair(
        "anthropic/claude-opus-4.8", "rubric", "a", "b", provider="Anthropic",
        transport=httpx.MockTransport(handler))
    body = json.loads(captured[0].content)
    assert body["max_tokens"] == paired_judge.JUDGE_MAX_TOKENS


def test_claude_judge_sends_the_unified_cap(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(
                type="text",
                text='{"winner": "A", "confidence": 0.9, "brief_reason": "r"}')])

    fake_anthropic = SimpleNamespace(
        Anthropic=lambda api_key: SimpleNamespace(messages=FakeMessages()))
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setattr(
        paired_judge.judge_credentials, "credential_for",
        lambda kind: SimpleNamespace(kind="claude", key="sk-ant-t"))

    verdict = paired_judge.judge_pair("claude-opus-4-8", "rubric", "a", "b")
    assert verdict["winner"] == "A"
    assert captured["max_tokens"] == paired_judge.JUDGE_MAX_TOKENS


def _capture_generate(monkeypatch, captured):
    def generate(model, prompt, *, model_id=None, max_tokens=0,
                 temperature=0.0, injections=None, prompt_mode=None,
                 system_prompt=None, qwen_thinking_enabled=False):
        captured.append(max_tokens)
        return '{"winner": "A", "confidence": 0.9, "brief_reason": "r"}'
    monkeypatch.setattr(tasks, "generate", generate)


def test_evaluate_local_judge_generates_with_the_unified_cap(monkeypatch):
    captured: list = []
    _capture_generate(monkeypatch, captured)

    @contextmanager
    def provider(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id)

    judge_fn, _model, _holder = tasks._judge_callable(
        JudgeRef(name="j1", kind="local"), provider,
        study_model="org/study-model", study_revision="abc123")
    verdict = judge_fn("org/study-model", "rubric", "a", "b", None)
    assert verdict["winner"] == "A"
    assert captured == [paired_judge.JUDGE_MAX_TOKENS]


def test_sweep_local_judge_generates_with_the_unified_cap(
        monkeypatch, tmp_path):
    # The exact incident path: the sweep's study-model local judge `_gen`.
    captured: list = []
    _capture_generate(monkeypatch, captured)
    rubric_path = tmp_path / "prompts" / "rubrics" / "r.md"
    rubric_path.parent.mkdir(parents=True)
    rubric_path.write_text("Which is more fearful?", encoding="utf-8")
    manifest = Manifest.from_dict({
        "name": "sw", "modelID": "org/study-model",
        "judgeRubricFile": "prompts/rubrics/r.md",
        "judges": [{"name": "j1", "kind": "local"}],
    })

    _rubric, panel = tasks._sweep_judge_panel(
        manifest, SimpleNamespace(), None, str(tmp_path), lambda *_: None)
    name, judge_fn, judge_model = panel[0]
    assert (name, judge_model) == ("j1", "org/study-model")
    verdict = judge_fn(judge_model, "rubric", "a", "b", None)
    assert verdict["winner"] == "A"
    assert captured == [paired_judge.JUDGE_MAX_TOKENS]


# --- the golden-pinned wrapper ------------------------------------------------


def _golden(name: str) -> str:
    path = GOLDEN_DIR / name
    assert path.exists(), (
        f"missing committed golden {path} — the paired-judge wrapper is "
        "byte-pinned cross-engine; restore the fixture, never delete it")
    return path.read_text(encoding="utf-8")


def test_wrapper_matches_the_full_golden():
    # Inputs identical to the Swift twin test — the two engines' builders
    # are pinned to the SAME committed bytes.
    built = paired_judge.build_prompt(
        "Which response answers the task prompt with the sounder legal "
        "reasoning? Dimensions: rigor, clarity.",
        "Response A body.\nSecond line of A.",
        "Response B body.",
        "a_severity and b_severity: sentence severity on a 1-9 ordinal "
        "scale.",
        task_prompt="Draft the holding for Case 12, stating the disposition "
        "in one paragraph.")
    assert built == _golden("wrapper-full.golden.txt"), (
        "paired-judge wrapper drifted from its committed golden — the "
        "wrapper is byte-pinned cross-engine (Swift PairedJudgePrompt.build "
        "reads the same fixture); change it only deliberately on BOTH "
        "engines and regenerate prompts/fixtures/paired-judge/")


def test_wrapper_matches_the_minimal_golden():
    built = paired_judge.build_prompt("", "A text.", "B text.", None,
                                      task_prompt=None)
    assert built == _golden("wrapper-minimal.golden.txt"), (
        "paired-judge wrapper (minimal shape) drifted from its committed "
        "golden — see wrapper-full assertion for the contract")


def test_wrapper_demands_a_brief_reason():
    # The incident's root behavior: judges writing essays. The unified
    # wrapper must explicitly bound the reason and put the verdict first.
    built = paired_judge.build_prompt("r", "a", "b", None)
    assert "at most two sentences" in built
    assert "stating the verdict fields first" in built

"""Scripted-transcript study items (the metacognition-study instrument).

Task-prompt items may carry a ``transcript`` — a scripted multi-turn
conversation with researcher-authored assistant turns ("words in the model's
mouth"), pinned as hashed stimulus data through the ordinary
``taskPromptsHash``. Contract under test (cross-engine; the Swift twin is
``TranscriptStudyTests``):

1. **Schema validation at load** — rules and EXACT message strings, replayed
   from the committed fixture
   ``prompts/fixtures/transcript-validation/cases.json`` (both engines run
   the identical cases);
2. **Rendering** through ``prompt_render.render_transcript`` →
   ``render_messages`` (the Playground-verified template path), with the
   composition RULE: a transcript's own system turn REPLACES the study
   system prompt; seeded/edited provenance flags stay inert;
3. **Family-constraint refusals at all three gates**: verify()
   (``Manifest._transcript_violations``), run start
   (``tasks._check_transcript_prompts``), and the readiness layer is
   Swift-side;
4. **Records** carry ``scriptedTranscript: true`` + the transcript for BOTH
   record kinds through the unified ``_execute_condition`` executor;
5. **Logprob over a transcript**: the stepped KV-cache scoring reads the
   reply position after the FULL rendered transcript — the injector gate
   covers every transcript token;
6. **rawCompletion refusals** at the manifest gate, the run gate, and the
   condition gate (a rawCompletion variant).
"""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import logprob as logprob_mod
from steerlab_server.experiment import prompt_render, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering.vector_store import ConceptVectors

FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "fixtures" / "transcript-validation" / "cases.json"
)

TRANSCRIPT = [
    {"role": "system", "content": "You answer questions about your own prior statements."},
    {"role": "user", "content": "Name a color, and nothing else."},
    {"role": "assistant", "content": "I would rather not name a color.", "seeded": True},
    {"role": "user", "content": "Did you just refuse? Answer yes or no."},
]


def _first_violation(case: dict) -> str | None:
    violation = prompt_render.transcript_schema_violation(
        case["transcript"], case["itemID"])
    if violation is None and case.get("modelID"):
        violation = prompt_render.transcript_family_violation(
            case["transcript"], case["itemID"], case["modelID"])
    return violation


# --- 1. schema + family validation: the committed cross-engine cases ---------

def _cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _cases(), ids=[c["name"] for c in _cases()])
def test_validation_message_matches_committed_fixture(case):
    assert _first_violation(case) == case["expect"], (
        f"{case['name']}: this engine's validation message diverged from the "
        "cross-engine fixture (the Swift twin replays the same file)")


def test_fixture_covers_every_schema_rule_and_both_family_rules():
    expects = [c["expect"] for c in _cases()]
    for needle in ("transcript is empty", "allowed roles", "empty content",
                   "at most one system turn", "assistant-prefix continuation",
                   "must end with a user turn",
                   "start with a user turn",
                   "strict user/assistant alternation"):
        assert any(e and needle in e for e in expects), (
            f"fixture lost coverage of rule: {needle}")
    assert any(e is None for e in expects), "fixture needs valid cases too"


# --- 2. rendering: system replacement + provenance inertness -----------------

class FakeTokenizer:
    """Deterministic chat-template fake (same contract as test_seeded_turns)."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True,
                            continue_final_message=False, **kwargs):
        self.calls.append([dict(m) for m in messages])
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        return text

    def __call__(self, text, add_special_tokens=True):
        class _Out:
            input_ids = [ord(c) for c in text]
        return _Out()


def test_transcript_system_turn_replaces_study_system_prompt():
    tok = FakeTokenizer()
    rendered = prompt_render.render_transcript(
        tok, TRANSCRIPT, model_id="Qwen/Qwen3-4B",
        system_prompt="STUDY SYSTEM — must not appear")
    assert "must not appear" not in rendered.text
    assert rendered.text.startswith(
        "<|system|>You answer questions about your own prior statements.")
    twin = prompt_render.render_transcript(
        tok, TRANSCRIPT, model_id="Qwen/Qwen3-4B", system_prompt=None)
    assert rendered.text == twin.text and rendered.input_ids == twin.input_ids


def test_study_system_prompt_applies_when_transcript_has_no_system_turn():
    tok = FakeTokenizer()
    rendered = prompt_render.render_transcript(
        tok, TRANSCRIPT[1:], model_id="Qwen/Qwen3-4B",
        system_prompt="study system prompt")
    assert rendered.text.startswith("<|system|>study system prompt<|end|>")


def test_gemma_folds_effective_system_into_first_user_turn():
    tok = FakeTokenizer()
    rendered = prompt_render.render_transcript(
        tok, TRANSCRIPT, model_id="google/gemma-3-4b-it", system_prompt=None)
    first = tok.calls[-1][0]
    assert first["role"] == "user"
    assert first["content"].startswith(
        "You answer questions about your own prior statements.\n\n"
        "Name a color, and nothing else.")
    assert all(m["role"] != "system" for m in tok.calls[-1])


def test_provenance_flags_are_inert_and_stripped_from_template_input():
    tok = FakeTokenizer()
    seeded = prompt_render.render_transcript(
        tok, TRANSCRIPT, model_id="Qwen/Qwen3-4B")
    stripped = [{k: v for k, v in m.items() if k != "seeded"} for m in TRANSCRIPT]
    plain = prompt_render.render_transcript(
        tok, stripped, model_id="Qwen/Qwen3-4B")
    assert seeded.text == plain.text and seeded.input_ids == plain.input_ids
    for message in tok.calls[-1]:
        assert set(message) == {"role", "content"}


def test_render_transcript_refuses_raw_completion():
    with pytest.raises(ValueError, match="chatAssistant"):
        prompt_render.render_transcript(
            FakeTokenizer(), TRANSCRIPT, model_id="Qwen/Qwen3-4B",
            prompt_mode=prompt_render.RAW_COMPLETION)


# --- 3. loader: schema refusal, display text, normalization ------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _manifest(model_id="Qwen/Qwen3-4B", **extra):
    d = {"name": "t", "modelID": model_id, "concepts": [], "conditions": []}
    d.update(extra)
    return Manifest.from_dict(d)


def test_load_prompts_derives_display_text_and_normalizes(tmp_path):
    path = str(tmp_path / "items.jsonl")
    _write(path, json.dumps({"id": "m1", "transcript": TRANSCRIPT,
                             "options": ["yes", "no"], "target": "yes"}) + "\n")
    prompts = tasks._load_prompts(_manifest(taskPromptsFile=path), None,
                                  str(tmp_path))
    (prompt,) = prompts
    # Display text derives from the final user turn; text/prompt is optional.
    assert prompt["prompt"] == "Did you just refuse? Answer yes or no."
    # Normalized turns: {role, content} only — the seeded flag is provenance
    # of the AUTHORING surface, not part of the stimulus record.
    assert all(set(t) == {"role", "content"} for t in prompt["transcript"])
    assert prompt["options"] == ["yes", "no"]


def test_load_prompts_refuses_schema_violations_with_contract_message(tmp_path):
    path = str(tmp_path / "items.jsonl")
    bad = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    _write(path, json.dumps({"id": "bad-1", "transcript": bad}) + "\n")
    with pytest.raises(RuntimeError, match="assistant-prefix continuation is out of scope"):
        tasks._load_prompts(_manifest(taskPromptsFile=path), None, str(tmp_path))


def test_explicit_text_wins_over_derived_display_text(tmp_path):
    path = str(tmp_path / "items.jsonl")
    _write(path, json.dumps({"id": "m1", "text": "custom display",
                             "transcript": TRANSCRIPT}) + "\n")
    prompts = tasks._load_prompts(_manifest(taskPromptsFile=path), None,
                                  str(tmp_path))
    assert prompts[0]["prompt"] == "custom display"


# --- run-start gates ----------------------------------------------------------

def test_run_start_refuses_family_incompatible_transcripts():
    prompts = [{"id": "g1", "prompt": "d", "transcript":
                prompt_render.normalize_transcript([
                    {"role": "assistant", "content": "Seeded first."},
                    {"role": "user", "content": "Q?"}])}]
    manifest = _manifest(model_id="mlx-community/gemma-3-12b-it-8bit")
    with pytest.raises(RuntimeError) as excinfo:
        tasks._check_transcript_prompts(manifest, prompts)
    message = str(excinfo.value)
    assert "scripted transcripts are incompatible" in message
    assert "item 'g1'" in message
    assert "start with a user turn" in message


def test_run_start_refuses_raw_completion_with_transcripts():
    prompts = [{"id": "m1", "prompt": "d",
                "transcript": prompt_render.normalize_transcript(TRANSCRIPT)}]
    manifest = _manifest(promptMode="rawCompletion")
    with pytest.raises(RuntimeError,
                       match="render through the chat template by definition"):
        tasks._check_transcript_prompts(manifest, prompts)


def test_run_start_gate_passes_compatible_transcripts():
    prompts = [{"id": "m1", "prompt": "d",
                "transcript": prompt_render.normalize_transcript(TRANSCRIPT)}]
    tasks._check_transcript_prompts(
        _manifest(model_id="mlx-community/gemma-3-12b-it-8bit"), prompts)
    tasks._check_transcript_prompts(_manifest(), prompts)


# --- verify()/freeze gate ------------------------------------------------------

def _pinned_manifest_dict(tmp_path, lines, *, model_id="Qwen/Qwen3-4B",
                          **extra):
    path = str(tmp_path / "prompts" / "tasks" / "items.jsonl")
    _write(path, "\n".join(json.dumps(line) for line in lines) + "\n")
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    d = {"name": "t", "modelID": model_id,
         "concepts": [], "conditions": [],
         "variantConditions": [{"name": "v", "artifactPath": "",
                                "artifactHash": "", "artifact": {}}],
         "taskPromptsFile": path, "taskPromptsHash": digest}
    d.update(extra)
    return d


def test_verify_flags_family_incompatible_transcript(tmp_path):
    manifest = Manifest.from_dict(_pinned_manifest_dict(
        tmp_path,
        [{"id": "g1", "transcript": [
            {"role": "assistant", "content": "Seeded first."},
            {"role": "user", "content": "Q?"}]}],
        model_id="mlx-community/gemma-3-12b-it-8bit"))
    violations = manifest.verify(str(tmp_path))
    assert any("item 'g1'" in v and "start with a user turn" in v
               for v in violations), violations


def test_verify_flags_raw_completion_with_transcripts(tmp_path):
    manifest = Manifest.from_dict(_pinned_manifest_dict(
        tmp_path, [{"id": "m1", "transcript": TRANSCRIPT}],
        promptMode="rawCompletion"))
    violations = manifest.verify(str(tmp_path))
    assert any("render through the chat template by definition" in v
               for v in violations), violations


def test_verify_flags_schema_violation_at_pin_time(tmp_path):
    manifest = Manifest.from_dict(_pinned_manifest_dict(
        tmp_path, [{"id": "bad", "transcript": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"}]}]))
    violations = manifest.verify(str(tmp_path))
    assert any("item 'bad'" in v and "assistant-prefix continuation" in v
               for v in violations), violations


def test_verify_clean_for_valid_transcripts(tmp_path):
    manifest = Manifest.from_dict(_pinned_manifest_dict(
        tmp_path, [{"id": "m1", "transcript": TRANSCRIPT,
                    "options": ["yes", "no"]}],
        model_id="mlx-community/gemma-3-12b-it-8bit"))
    assert not [v for v in manifest.verify(str(tmp_path))
                if "transcript" in v.lower()]


# --- 4. records through the unified executor -----------------------------------

OPTIONS = ["yes", "no"]


def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _transcript_study(root, name):
    """A steered two-condition study whose items carry scripted transcripts
    (one with its own system turn + options, one plain)."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="Qwen/Qwen3-4B", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 16
    raw["systemPrompt"] = "STUDY SYSTEM PROMPT"
    raw["outcomeInstruments"] = ["answerTokenLogprob", "sampledText"]
    raw["conditions"] = [{"name": "fear-a2",
                          "slots": [{"concept": "fear", "layer": 2, "alpha": 2.0}]}]
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    lines = [
        json.dumps({"id": "t1", "transcript": TRANSCRIPT,
                    "options": OPTIONS, "target": "yes"}),
        json.dumps({"id": "p1", "prompt": "Plain item.", "options": OPTIONS}),
    ]
    _write(prompts_path, "\n".join(lines) + "\n")
    return prompts_path


def _fake_generate(log):
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False, transcript=None):
        log.append({"kind": "generate", "prompt": prompt,
                    "transcript": transcript, "system": system_prompt})
        return "I confirm the refusal was mine."
    return generate


def _fake_score_options(log):
    def score_options(model, prompt, options, *, model_id=None, injections=None,
                      prompt_mode=None, system_prompt=None,
                      qwen_thinking_enabled=False, transcript=None):
        log.append({"kind": "instrument", "prompt": prompt,
                    "transcript": transcript, "system": system_prompt})
        scores = [logprob_mod.OptionScore(
            option=o, token_ids=[7],
            token_logprobs=[-0.5 if o == "yes" else -2.0]) for o in options]
        return logprob_mod.ChoiceResult(options=scores, prompt_token_count=5,
                                        prompt_text=prompt)
    return score_options


def _run_transcript_study(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _transcript_study(root, "meta1")
    log = []
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", _fake_generate(log))
    monkeypatch.setattr(logprob_mod, "score_options", _fake_score_options(log))
    run_dir = tasks.run("meta1", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        records = [json.loads(line) for line in h if line.strip()]
    return records, log


def test_records_carry_transcript_and_flag_for_both_record_kinds(
        tmp_path, monkeypatch):
    records, log = _run_transcript_study(tmp_path, monkeypatch)
    normalized = prompt_render.normalize_transcript(TRANSCRIPT)
    for condition in ("baseline", "fear-a2"):
        mine = [r for r in records if r["condition"] == condition]
        sampled = [r for r in mine if "instrument" not in r]
        instruments = [r for r in mine if "instrument" in r]
        assert len(sampled) == 2 and len(instruments) == 2
        for group in (sampled, instruments):
            t1 = next(r for r in group if r["promptID"] == "t1")
            p1 = next(r for r in group if r["promptID"] == "p1")
            assert t1["scriptedTranscript"] is True
            assert t1["transcript"] == normalized
            assert t1["prompt"] == "Did you just refuse? Answer yes or no."
            assert "scriptedTranscript" not in p1
            assert "transcript" not in p1
    # The executor threaded the transcript into BOTH measurement calls, for
    # every condition, and never for the plain item.
    transcript_calls = [c for c in log if c["transcript"] is not None]
    plain_calls = [c for c in log if c["transcript"] is None]
    assert {c["kind"] for c in transcript_calls} == {"generate", "instrument"}
    assert len(transcript_calls) == 4  # 2 conditions × (sampled + instrument)
    assert all(c["transcript"] == normalized for c in transcript_calls)
    assert all(c["prompt"] == "Plain item." for c in plain_calls)


def test_raw_completion_variant_condition_refuses_transcripts(tmp_path, monkeypatch):
    root = str(tmp_path)
    _transcript_study(root, "meta2")
    raw = es.load_raw("meta2", root)
    raw["conditions"] = []
    raw["concepts"] = []
    raw["outcomeInstruments"] = ["sampledText"]
    raw["variantConditions"] = [{
        "name": "v-raw", "artifactPath": "runs/variants/v.json",
        "artifactHash": "aa" * 32,
        "artifact": {"name": "v-raw", "baseModelID": "Qwen/Qwen3-4B",
                     "promptMode": "rawCompletion", "temperature": 0.0,
                     "alphaInNormUnits": False, "injections": [],
                     "adapters": []}}]
    es.save_raw(raw, root)
    monkeypatch.setattr(tasks, "_extract_all", lambda *a: {})
    log = []
    monkeypatch.setattr(tasks, "generate", _fake_generate(log))
    prompts = os.path.join(root, "prompts", "tasks", "items.jsonl")
    with pytest.raises(RuntimeError, match="condition 'v-raw'"):
        tasks.run("meta2", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)


# --- 5. logprob over the full rendered transcript -------------------------------

class _StepCountingLM:
    def __init__(self, vocab=512):
        self.vocab = vocab
        self.calls = []

    def __call__(self, input_ids=None, attention_mask=None, past_key_values=None,
                 use_cache=True):
        seq = input_ids.shape[1]
        self.calls.append({"seq": seq, "cached": past_key_values is not None})
        logits = torch.zeros(1, seq, self.vocab)
        return SimpleNamespace(logits=logits, past_key_values="cache")


class _Hooked:
    def __init__(self):
        self.sessions = []
        self.resets = 0

    @contextmanager
    def session(self, interventions):
        self.sessions.append(list(interventions))
        yield

    def reset_offsets(self):
        self.resets += 1


class _TranscriptModel:
    model_id = "Qwen/Qwen3-4B"
    revision = "deadbeef"
    context_window = 0
    device = "cpu"

    def __init__(self):
        self.model = _StepCountingLM()
        self.tokenizer = FakeTokenizer()
        self.hooked = _Hooked()


def test_logprob_scores_reply_position_after_full_rendered_transcript():
    """The stepped KV-cache driver over a multi-turn render: the prompt pass
    covers the WHOLE rendered transcript (system turn included), and the
    injector gate equals that full length — decode-identical semantics, no
    mid-transcript scoring."""
    model = _TranscriptModel()
    from steerlab_server.experiment.generate import CellInjection
    cell = CellInjection(layer=2, vector=[1.0, 0.0], alpha=3.0)
    result = logprob_mod.score_options(
        model, "IGNORED — the transcript is the prompt", ["yes", "no"],
        injections=[cell], transcript=TRANSCRIPT,
        system_prompt="STUDY SYSTEM — replaced")
    expected = prompt_render.render_transcript(
        FakeTokenizer(), TRANSCRIPT, model_id="Qwen/Qwen3-4B",
        system_prompt="STUDY SYSTEM — replaced")
    assert "STUDY SYSTEM — replaced" not in expected.text
    assert result.prompt_text == expected.text
    assert result.prompt_token_count == expected.prompt_token_count
    # The prefill forward pass fed the FULL rendered transcript.
    uncached = [c for c in model.model.calls if not c["cached"]]
    assert [c["seq"] for c in uncached] == [expected.prompt_token_count] * 2
    # The injector is gated on the full transcript length: it fires at the
    # true transcript end + every option step, never mid-transcript.
    (injectors,) = model.hooked.sessions
    assert injectors[0]._prompt_token_count == expected.prompt_token_count

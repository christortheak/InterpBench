"""Seeded assistant turns (send-as-assistant) + assistant-prefix continuation.

The metacognition instrument: the researcher composes a turn that enters the
transcript AS an assistant turn ("words in the model's mouth"), so follow-up
user turns can probe self-attribution. Contract under test:

1. **Rendering ignores provenance**: a message array whose assistant turn
   carries ``seeded: true`` renders byte-identically to the same history
   without the flag — indistinguishable to the MODEL, distinguishable in the
   RECORD. (Real-tokenizer byte parity for both families is pinned by the
   ``chat_seeded_assistant`` golden fixtures in
   ``prompts/fixtures/render/`` + ``test_golden_render_fixtures.py``.)
2. **Prefill**: ``continue_final_message=True`` renders WITHOUT a generation
   prompt via transformers' own ``continue_final_message`` and requires a
   final assistant turn in chatAssistant mode.
3. **Records**: the variant-chat response record labels seeded turns and
   stamps ``seededMessageIndices`` / ``continuedFinalMessage``.
4. **Capability**: the snapshot advertises ``chat.seededTurns`` and
   ``chat.continueFinalMessage`` so older clients can gate.
"""

from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from steerlab_server.api import variant_chat
from steerlab_server.api.profile import capability_snapshot
from steerlab_server.experiment import prompt_render


class FakeTokenizer:
    """Records apply_chat_template kwargs; renders a deterministic string.

    Emulates the transformers contract this feature relies on:
    ``continue_final_message=True`` renders the final turn WITHOUT its
    end-of-turn suffix and without a generation prompt.
    """

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True,
                            continue_final_message=False, **kwargs):
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
            "kwargs": kwargs,
        })
        parts = [f"<|{m['role']}|>{m['content']}<|end|>" for m in messages]
        text = "".join(parts)
        if continue_final_message:
            text = text[: -len("<|end|>")]
        elif add_generation_prompt:
            text += "<|assistant|>"
        return text

    def __call__(self, text, add_special_tokens=True):
        class _Out:
            input_ids = [ord(c) for c in text[:8]] or [0]
        return _Out()


SEEDED_MESSAGES = [
    {"role": "user", "content": "State the holding."},
    {"role": "assistant", "content": "I refuse to answer that.", "seeded": True},
    {"role": "user", "content": "Why did you refuse?"},
]


def _strip_seeded(messages):
    return [{k: v for k, v in m.items() if k != "seeded"} for m in messages]


def test_seeded_flag_is_inert_for_rendering():
    tok_seeded, tok_plain = FakeTokenizer(), FakeTokenizer()
    seeded = prompt_render.render_messages(
        tok_seeded, SEEDED_MESSAGES, model_id="Qwen/Qwen3-4B")
    plain = prompt_render.render_messages(
        tok_plain, _strip_seeded(SEEDED_MESSAGES), model_id="Qwen/Qwen3-4B")
    assert seeded.text == plain.text
    assert seeded.input_ids == plain.input_ids
    # The flag never reaches the template context either.
    assert tok_seeded.calls[-1]["messages"] == tok_plain.calls[-1]["messages"]
    for message in tok_seeded.calls[-1]["messages"]:
        assert set(message) == {"role", "content"}


def test_assistant_role_is_honored_in_history():
    tok = FakeTokenizer()
    prompt_render.render_messages(tok, SEEDED_MESSAGES, model_id="Qwen/Qwen3-4B")
    roles = [m["role"] for m in tok.calls[-1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_continue_final_message_renders_without_generation_prompt():
    tok = FakeTokenizer()
    rendered = prompt_render.render_messages(
        tok, [
            {"role": "user", "content": "Decide the case."},
            {"role": "assistant", "content": "The court holds", "seeded": True},
        ],
        model_id="Qwen/Qwen3-4B", continue_final_message=True)
    call = tok.calls[-1]
    assert call["continue_final_message"] is True
    assert call["add_generation_prompt"] is False
    # Rendering ends exactly at the seeded prefix — no end-of-turn, no new
    # generation prompt.
    assert rendered.text.endswith("The court holds")


def test_continue_final_message_requires_final_assistant_turn():
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="assistant turn"):
        prompt_render.render_messages(
            tok, [{"role": "user", "content": "hello"}],
            model_id="Qwen/Qwen3-4B", continue_final_message=True)


def test_continue_final_message_rejects_raw_completion():
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="chatAssistant"):
        prompt_render.render_messages(
            tok, [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            model_id="Qwen/Qwen3-4B",
            prompt_mode=prompt_render.RAW_COMPLETION,
            continue_final_message=True)


def test_render_chat_text_labels_seeded_turns():
    text = variant_chat.render_chat_text(None, SEEDED_MESSAGES)
    assert "assistant (seeded): I refuse to answer that." in text
    # Real generations stay unlabeled.
    plain = variant_chat.render_chat_text(None, [
        {"role": "assistant", "content": "hi"}])
    assert plain == "assistant: hi"


def test_request_from_body_reads_continue_flag():
    request = variant_chat.request_from_body(
        {"messages": SEEDED_MESSAGES, "continueFinalMessage": True},
        resolved_variant_path="p")
    assert request.continue_final_message is True
    assert request.messages == SEEDED_MESSAGES


def test_variant_record_carries_seeded_provenance(monkeypatch):
    from steerlab_server.experiment import model_variant

    variant = model_variant.ModelVariant(
        name="v", base_model_id="org/model", temperature=0.0,
        prompt_mode="chatAssistant")

    @contextmanager
    def fake_prepared(model, v, strip_interventions=False):
        yield []

    captured = {}

    def fake_generate_messages(model, messages, *, continue_final_message=False,
                               **kwargs):
        captured["continue"] = continue_final_message
        return "continued text"

    monkeypatch.setattr(variant_chat, "prepared_variant", fake_prepared)
    monkeypatch.setattr(variant_chat, "generate_messages", fake_generate_messages)
    model = SimpleNamespace(model_id="org/model", revision="abc")
    request = variant_chat.VariantChatRequest(
        variant_path="p", messages=list(SEEDED_MESSAGES),
        continue_final_message=True)
    result = variant_chat.generate_with_variant(model, variant, request)
    assert captured["continue"] is True
    assert result["seededMessageIndices"] == [1]
    assert result["continuedFinalMessage"] is True
    assert "assistant (seeded)" in result["prompt"]


def test_continuation_without_messages_refuses():
    from steerlab_server.experiment import model_variant

    variant = model_variant.ModelVariant(
        name="v", base_model_id="org/model", temperature=0.0,
        prompt_mode="chatAssistant")
    model = SimpleNamespace(model_id="org/model", revision="abc")
    request = variant_chat.VariantChatRequest(
        variant_path="p", text="bare prompt", continue_final_message=True)
    with pytest.raises(ValueError, match="continueFinalMessage"):
        variant_chat.generate_with_variant(model, variant, request)


def test_capability_snapshot_advertises_chat_flags():
    snapshot = capability_snapshot()
    assert snapshot["chat"]["seededTurns"] is True
    assert snapshot["chat"]["continueFinalMessage"] is True


# --- edited flag (generated-then-altered provenance) --------------------------

EDITED_MESSAGES = [
    {"role": "user", "content": "State the holding."},
    {"role": "assistant", "content": "The holding is reversed.", "edited": True},
    {"role": "user", "content": "Why?"},
]


def _strip_flags(messages):
    return [{k: v for k, v in m.items() if k not in ("seeded", "edited")}
            for m in messages]


def test_edited_flag_is_inert_for_rendering():
    tok_edited, tok_plain = FakeTokenizer(), FakeTokenizer()
    edited = prompt_render.render_messages(
        tok_edited, EDITED_MESSAGES, model_id="Qwen/Qwen3-4B")
    plain = prompt_render.render_messages(
        tok_plain, _strip_flags(EDITED_MESSAGES), model_id="Qwen/Qwen3-4B")
    assert edited.text == plain.text
    assert edited.input_ids == plain.input_ids
    # The flag never reaches the template context either.
    for message in tok_edited.calls[-1]["messages"]:
        assert set(message) == {"role", "content"}


def test_generate_request_dto_tolerates_edited_key():
    """Older servers must accept the extra `edited` key: messages are plain
    dicts on the wire, so the pydantic model tolerates unknown provenance
    keys by construction — pinned here so a future typed Message model
    cannot silently break older clients' forward compatibility."""
    from steerlab_server.api import dto

    body = dto.GenerateRequest(messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "seeded": True, "edited": True},
    ])
    assert body.messages[1]["edited"] is True


def test_render_chat_text_labels_edited_turns():
    text = variant_chat.render_chat_text(None, EDITED_MESSAGES)
    assert "assistant (edited): The holding is reversed." in text
    both = variant_chat.render_chat_text(None, [
        {"role": "assistant", "content": "x", "seeded": True, "edited": True}])
    assert both == "assistant (seeded, edited): x"


def test_variant_record_carries_edited_provenance(monkeypatch):
    from steerlab_server.experiment import model_variant

    variant = model_variant.ModelVariant(
        name="v", base_model_id="org/model", temperature=0.0,
        prompt_mode="chatAssistant")

    @contextmanager
    def fake_prepared(model, v, strip_interventions=False):
        yield []

    monkeypatch.setattr(variant_chat, "prepared_variant", fake_prepared)
    monkeypatch.setattr(variant_chat, "generate_messages",
                        lambda model, messages, **kwargs: "text")
    model = SimpleNamespace(model_id="org/model", revision="abc")
    request = variant_chat.VariantChatRequest(
        variant_path="p", messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "edited": True},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "b", "seeded": True},
        ])
    result = variant_chat.generate_with_variant(model, variant, request)
    assert result["editedMessageIndices"] == [1]
    assert result["seededMessageIndices"] == [3]
    assert "assistant (edited): a" in result["prompt"]


# --- chat-template constraint translation (never a 500) -----------------------


class AlternationEnforcingTokenizer(FakeTokenizer):
    """Emulates a Gemma-class template: raises the jinja TemplateError that
    transformers' `raise_exception` produces when roles do not alternate
    user/assistant starting with user."""

    def apply_chat_template(self, messages, **kwargs):
        jinja2 = pytest.importorskip("jinja2")
        roles = [m["role"] for m in messages if m["role"] != "system"]
        expected = ["user", "assistant"] * ((len(roles) + 1) // 2)
        if roles != expected[: len(roles)]:
            raise jinja2.exceptions.TemplateError(
                "Conversation roles must alternate user/assistant/"
                "user/assistant/...")
        return super().apply_chat_template(messages, **kwargs)


def test_template_error_translates_to_structured_constraint_error():
    tok = AlternationEnforcingTokenizer()
    with pytest.raises(prompt_render.ChatTemplateConstraintError) as info:
        prompt_render.render_messages(
            tok, [{"role": "assistant", "content": "seeded first"}],
            model_id="fake/enforcing-model")
    detail = info.value.detail()
    assert detail["constraint"] == "roleOrder"
    assert detail["modelID"] == "fake/enforcing-model"
    assert "alternate" in detail["message"]
    # Still a ValueError, so pre-existing broad 400 translation keeps working.
    assert isinstance(info.value, ValueError)


def test_alternation_valueerror_also_translates():
    class ValueErrorTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise ValueError(
                "Conversation roles must alternate user/assistant/...")

    with pytest.raises(prompt_render.ChatTemplateConstraintError):
        prompt_render.render_messages(
            ValueErrorTokenizer(), SEEDED_MESSAGES, model_id="fake/m")


def test_unrelated_valueerror_is_not_masked():
    class BrokenTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise ValueError("something else entirely")

    with pytest.raises(ValueError) as info:
        prompt_render.render_messages(
            BrokenTokenizer(), SEEDED_MESSAGES, model_id="fake/m")
    assert not isinstance(info.value, prompt_render.ChatTemplateConstraintError)


def test_generate_route_returns_structured_400_for_role_order(monkeypatch):
    """The user-reported defect: seeding an assistant turn FIRST, then
    sending, must never 500 — the route returns 400 with
    {message, constraint, modelID}."""
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api import model_registry
    from steerlab_server.api.routes import ServiceState, build_router

    fake = SimpleNamespace(
        model_id="fake/enforcing-model", revision="r1", device="cpu",
        num_layers=2, hidden_size=8, context_window=2048, dtype="float32",
        tokenizer=AlternationEnforcingTokenizer())
    monkeypatch.setattr(
        model_registry.model_loader, "load",
        lambda model_id, revision=None, dtype="auto", device=None: fake)
    state = ServiceState()
    app = fastapi.FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    state.load("fake/enforcing-model", None, "auto", None)

    resp = client.post("/api/generate", json={"messages": [
        {"role": "assistant", "content": "seeded first", "seeded": True},
        {"role": "user", "content": "did you mean to say that?"},
    ]})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail["constraint"] == "roleOrder"
    assert detail["modelID"] == "fake/enforcing-model"
    assert "alternate" in detail["message"]


# --- constraints table vs the REAL pinned templates ---------------------------

PINNED_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-MLX-4bit",
    "Qwen/Qwen3-14B-MLX-8bit",
    "Qwen/Qwen3-32B-MLX-8bit",
    "google/gemma-3-4b-it",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/gemma-3-12b-it-8bit",
    "mlx-community/gemma-3-27b-it-8bit",
]


def _real_tokenizer(model_id: str):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not cached / offline
        pytest.skip(f"tokenizer for {model_id} not in local HF cache — "
                    f"constraint table NOT verified live for it: {exc}")


def _renders(tok, model_id, messages) -> bool:
    try:
        prompt_render.render_messages(tok, messages, model_id=model_id)
        return True
    except prompt_render.ChatTemplateConstraintError:
        return False


@pytest.mark.parametrize("model_id", PINNED_MODELS)
def test_constraints_table_matches_live_template(model_id):
    """The static family table (`conversation_constraints`) must agree with
    the REAL chat template's behavior — re-vendoring a family with different
    conversation rules fails here, not in production."""
    tok = _real_tokenizer(model_id)
    constraints = prompt_render.conversation_constraints(model_id)

    assistant_first = [{"role": "assistant", "content": "Seeded reply."},
                       {"role": "user", "content": "Question?"}]
    consecutive_assistant = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A one."},
        {"role": "assistant", "content": "A two."}]
    consecutive_user = [{"role": "user", "content": "Q one."},
                        {"role": "user", "content": "Q two."}]
    alternating = [{"role": "user", "content": "Q"},
                   {"role": "assistant", "content": "A"},
                   {"role": "user", "content": "Q2"}]

    assert _renders(tok, model_id, alternating), (
        f"{model_id}: a plain alternating history must always render")
    assert _renders(tok, model_id, assistant_first) == (
        not constraints.requires_leading_user_turn), (
        f"{model_id}: requiresLeadingUserTurn drifted from the live template")
    permissive = not constraints.forbids_consecutive_same_role
    assert _renders(tok, model_id, consecutive_assistant) == permissive, (
        f"{model_id}: forbidsConsecutiveSameRole (assistant) drifted")
    assert _renders(tok, model_id, consecutive_user) == permissive, (
        f"{model_id}: forbidsConsecutiveSameRole (user) drifted")


def test_gemma_live_template_refusal_carries_structured_detail():
    """End-to-end with the REAL pinned Gemma tokenizer: an assistant-first
    transcript raises the translated constraint error whose detail is
    exactly what the generate/variant routes return as the 400 body."""
    model_id = "mlx-community/gemma-3-4b-it-4bit"
    tok = _real_tokenizer(model_id)
    with pytest.raises(prompt_render.ChatTemplateConstraintError) as info:
        prompt_render.render_messages(
            tok, [
                {"role": "assistant", "content": "I have reviewed the record.",
                 "seeded": True},
                {"role": "user", "content": "State the holding."},
            ],
            model_id=model_id)
    detail = info.value.detail()
    assert detail["constraint"] == "roleOrder"
    assert detail["modelID"] == model_id
    assert "alternate" in detail["message"].lower()

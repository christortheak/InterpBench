"""Chat-template branching must match the Swift ``userInput`` rules: Gemma folds
system into the first user turn (no system role), Qwen passes ``enable_thinking``
to the template / appends ``/no_think`` in raw mode, others keep a system role.

Uses a fake tokenizer so the *branching* is tested without a real model; true
golden-token parity against the Swift output is a separate, model-loaded check
(documented in the plan as the #1 risk).
"""

from steerlab_server.experiment import prompt_render


class FakeTokenizer:
    def __init__(self):
        self.last_messages = None
        self.last_template_kwargs = None
        self.last_add_special_tokens = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            **kwargs):
        self.last_messages = messages
        self.last_template_kwargs = kwargs
        # Render a deterministic string we can introspect.
        return "<BOS>" + " ".join(f"[{m['role']}]{m['content']}" for m in messages)

    def __call__(self, text, add_special_tokens=True):
        self.last_add_special_tokens = add_special_tokens

        class _Out:
            input_ids = list(range(len(text.split())))
        return _Out()


def test_gemma_folds_system_into_user():
    tok = FakeTokenizer()
    prompt_render.render(tok, "Translate this.", model_id="mlx-community/gemma-3-4b-it",
                         system_prompt="You are a judge.")
    # No system role; system text prepended to the user turn.
    roles = [m["role"] for m in tok.last_messages]
    assert "system" not in roles
    assert tok.last_messages[0]["content"].startswith("You are a judge.\n\n")
    # Double-BOS guard: chat path tokenizes without adding special tokens.
    assert tok.last_add_special_tokens is False


def test_qwen_passes_enable_thinking_false_by_default():
    tok = FakeTokenizer()
    prompt_render.render(tok, "Decide the case.", model_id="Qwen/Qwen3-14B")
    assert tok.last_template_kwargs == {"enable_thinking": False}


def test_qwen_enable_thinking_true_when_requested():
    tok = FakeTokenizer()
    prompt_render.render(tok, "Decide.", model_id="Qwen/Qwen3-14B",
                         qwen_thinking_enabled=True)
    # The legacy boolean means the template's DEFAULT effort (xhigh), and the
    # effort is now passed explicitly so the declared study is reproducible
    # on a template whose default could change.
    assert tok.last_template_kwargs == {"enable_thinking": True,
                                        "reasoning_effort": "xhigh"}


def test_qwen_reasoning_effort_reaches_the_template_by_value():
    for effort in ("low", "medium", "xhigh"):
        tok = FakeTokenizer()
        prompt_render.render(tok, "Decide.", model_id="Qwen/Qwen3-14B",
                             reasoning_effort=effort)
        assert tok.last_template_kwargs == {"enable_thinking": True,
                                            "reasoning_effort": effort}


def test_qwen_reasoning_effort_off_is_byte_for_byte_the_old_no_think_path():
    tok = FakeTokenizer()
    prompt_render.render(tok, "Decide.", model_id="Qwen/Qwen3-14B",
                         reasoning_effort="off")
    assert tok.last_template_kwargs == {"enable_thinking": False}
    raw = prompt_render.render(tok, "Decide.", model_id="Qwen/Qwen3-14B",
                               prompt_mode=prompt_render.RAW_COMPLETION,
                               reasoning_effort="off")
    assert raw.text.endswith(" /no_think")
    thinking = prompt_render.render(tok, "Decide.", model_id="Qwen/Qwen3-14B",
                                    prompt_mode=prompt_render.RAW_COMPLETION,
                                    reasoning_effort="low")
    assert thinking.text.endswith(" /think")


def test_a_family_without_a_thinking_mode_gets_no_effort_variable():
    tok = FakeTokenizer()
    prompt_render.render(tok, "Hi", model_id="google/gemma-3-4b-it",
                         reasoning_effort="off")
    assert tok.last_template_kwargs == {}
    assert not prompt_render.has_thinking_mode("google/gemma-3-4b-it")
    assert prompt_render.has_thinking_mode("Qwen/Qwen3-14B")


def test_an_unknown_effort_is_refused_at_render_time():
    import pytest
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="unknown reasoningEffort"):
        prompt_render.render(tok, "Hi", model_id="Qwen/Qwen3-14B",
                             reasoning_effort="hgih")


def test_non_gemma_keeps_system_role():
    tok = FakeTokenizer()
    prompt_render.render(tok, "Hi", model_id="Qwen/Qwen3-14B", system_prompt="Be terse.")
    roles = [m["role"] for m in tok.last_messages]
    assert roles == ["system", "user"]


def test_raw_completion_qwen_appends_no_think():
    tok = FakeTokenizer()
    rendered = prompt_render.render(
        tok, "The capital of France is", model_id="Qwen/Qwen3-4B",
        prompt_mode=prompt_render.RAW_COMPLETION)
    assert rendered.text.endswith(" /no_think")
    # Raw mode adds special tokens once (matches the Swift raw-text path).
    assert tok.last_add_special_tokens is True


def test_raw_completion_with_system_prepends():
    tok = FakeTokenizer()
    rendered = prompt_render.render(
        tok, "Continue.", model_id="meta/other-model", system_prompt="Context.",
        prompt_mode=prompt_render.RAW_COMPLETION)
    assert rendered.text == "Context.\n\nContinue."

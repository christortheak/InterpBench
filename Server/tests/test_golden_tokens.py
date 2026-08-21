"""Golden-token tokenizer parity against a REAL model tokenizer.

The #1 correctness risk is that our two-step render (apply_chat_template with
``tokenize=False`` → tokenize with ``add_special_tokens=False``) drifts from the
canonical HF tokenization, which would shift the extraction reading position.
These tests load the actual Qwen3-0.6B tokenizer (no GPU) and assert our
``input_ids`` equal the canonical path exactly — the real double-BOS / position
guard, not a fake-tokenizer branch check. Skipped if the model isn't cached.
"""

import pytest

transformers = pytest.importorskip("transformers")

from steerlab_server.experiment import prompt_render

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:  # noqa: BLE001 - model not cached / offline
        pytest.skip(f"{MODEL} tokenizer unavailable: {exc}")


def test_chat_render_matches_canonical_tokenization(tok):
    prompt = "Decide the case and state the holding."
    rendered = prompt_render.render(tok, prompt, model_id=MODEL,
                                    prompt_mode=prompt_render.CHAT_ASSISTANT)
    canonical = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    # Our text→tokenize(no special) path must equal the canonical one-step
    # tokenization — i.e. no double BOS, identical positions.
    assert rendered.input_ids == list(canonical)
    assert rendered.prompt_token_count == len(canonical)


def test_chat_with_system_prompt_matches(tok):
    rendered = prompt_render.render(
        tok, "Hello", model_id=MODEL, prompt_mode=prompt_render.CHAT_ASSISTANT,
        system_prompt="You are terse.")
    canonical = tok.apply_chat_template(
        [{"role": "system", "content": "You are terse."},
         {"role": "user", "content": "Hello"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    assert rendered.input_ids == list(canonical)


def test_raw_completion_appends_no_think_and_tokenizes(tok):
    rendered = prompt_render.render(tok, "The capital of France is", model_id=MODEL,
                                    prompt_mode=prompt_render.RAW_COMPLETION)
    assert rendered.text.endswith(" /no_think")
    assert rendered.input_ids == tok("The capital of France is /no_think",
                                     add_special_tokens=True).input_ids


def test_thinking_toggle_changes_tokens(tok):
    base = prompt_render.render(tok, "Q", model_id=MODEL,
                                prompt_mode=prompt_render.CHAT_ASSISTANT)
    thinking = prompt_render.render(tok, "Q", model_id=MODEL,
                                    prompt_mode=prompt_render.CHAT_ASSISTANT,
                                    qwen_thinking_enabled=True)
    # enable_thinking is actually threaded through (the renders differ or match
    # the respective canonical templates).
    assert base.input_ids == list(tok.apply_chat_template(
        [{"role": "user", "content": "Q"}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False))
    assert thinking.input_ids == list(tok.apply_chat_template(
        [{"role": "user", "content": "Q"}], tokenize=True,
        add_generation_prompt=True, enable_thinking=True, return_dict=False))

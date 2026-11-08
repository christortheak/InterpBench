"""The submission's prompt counter, batched, counts exactly as before (issue #11).

A submission counts every task prompt under every arm's rendering before it
answers. For several hundred long transcripts that took minutes, almost all
of it in encoding the rendered text one string at a time. The counter now
renders each prompt through the generation path's own function, unchanged,
and encodes the renderings in batches, which a fast tokenizer spreads across
cores. These tests pin that the counts are identical to the renderer's own,
item by item:

- on a stand-in tokenizer that answers both a string and a list, across plain
  prompts, scripted transcripts, a system prompt, raw completion (whose
  encoding adds special tokens), and batch boundaries;
- on a tokenizer that cannot encode a list, which is asked one string at a
  time, as before;
- and, when this machine's model cache holds one, on real tokenizers with
  long synthetic transcripts. Nothing is downloaded: the cache is read with
  ``local_files_only``, and a missing tokenizer skips.
"""

import random
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import prompt_reading, prompt_render

MODEL = "acme/tiny"


class _BatchTok:
    """One token per word, plus one for special tokens; answers a list too."""

    chat_template = None

    def __init__(self):
        self.single_calls = 0
        self.batch_calls = 0

    @staticmethod
    def _ids(text, special):
        return list(range(len(text.split()) + (1 if special else 0)))

    def __call__(self, text, add_special_tokens=True):
        if isinstance(text, list):
            self.batch_calls += 1
            return {"input_ids": [self._ids(t, add_special_tokens) for t in text]}
        self.single_calls += 1
        return SimpleNamespace(input_ids=self._ids(text, add_special_tokens))

    def apply_chat_template(self, messages, add_generation_prompt=True, **_):
        body = " <turn> ".join(f"{m['role']} {m['content']}" for m in messages)
        return "<bos> " + body + (" <turn> model" if add_generation_prompt else "")


class _StringOnlyTok(_BatchTok):
    def __call__(self, text, add_special_tokens=True):
        if not isinstance(text, str):
            raise TypeError("one string at a time")
        return super().__call__(text, add_special_tokens)


def _setting(mode=prompt_render.CHAT_ASSISTANT, system=None):
    return {"prompt_mode": mode, "system_prompt": system,
            "qwen_thinking_enabled": False, "reasoning_effort": None}


def _items():
    rng = random.Random(11)
    words = ["garden", "river", "the", "lamp", "was", "quiet", "and"]

    def text(n):
        return " ".join(rng.choice(words) for _ in range(n))

    plain = [{"id": f"p{i}", "prompt": text(50 + 37 * i)} for i in range(6)]
    transcripts = [{"id": f"t{i}", "prompt": "Answer.", "transcript": [
        {"role": "system", "content": text(9)},
        {"role": "user", "content": text(200 + 11 * i)},
        {"role": "assistant", "content": text(90)},
        {"role": "user", "content": "Answer."}]} for i in range(4)]
    return plain + transcripts


def _one_by_one(tokenizer, prompts, settings):
    """The renderer's own count, item by item — what the counter replaced."""
    return [[prompt_render.render_for_generation(
        tokenizer, p.get("prompt") or "", model_id=MODEL,
        transcript=p.get("transcript") or None, **s).prompt_token_count
        for s in settings] for p in prompts]


def test_batched_counts_equal_the_renderers_own():
    settings = [_setting(), _setting(system="Read carefully.")]
    prompts = _items()
    tok = _BatchTok()
    expected = _one_by_one(_BatchTok(), prompts, settings)
    counts = prompt_reading.rendered_token_counts(tok, MODEL, prompts, settings)
    assert counts == expected
    assert tok.batch_calls >= 1
    assert tok.single_calls == 0


def test_raw_completion_keeps_its_special_tokens():
    """Raw completion encodes WITH special tokens, the chat template without:
    one batch per kind, each with its own flag."""
    prompts = [p for p in _items() if "transcript" not in p]
    settings = [_setting(), _setting(mode=prompt_render.RAW_COMPLETION)]
    counts = prompt_reading.rendered_token_counts(_BatchTok(), MODEL, prompts,
                                                  settings)
    assert counts == _one_by_one(_BatchTok(), prompts, settings)
    assert any(row[0] != row[1] for row in counts)


def test_batch_boundaries_do_not_change_a_count(monkeypatch):
    monkeypatch.setattr(prompt_reading, "COUNT_BATCH_CHARACTERS", 700)
    settings = [_setting(), _setting(system="Read carefully.")]
    prompts = _items()
    tok = _BatchTok()
    counts = prompt_reading.rendered_token_counts(tok, MODEL, prompts, settings)
    assert counts == _one_by_one(_BatchTok(), prompts, settings)
    assert tok.batch_calls > 2


def test_a_tokenizer_that_cannot_batch_is_asked_one_string_at_a_time():
    settings = [_setting()]
    prompts = _items()
    tok = _StringOnlyTok()
    counts = prompt_reading.rendered_token_counts(tok, MODEL, prompts, settings)
    assert counts == _one_by_one(_StringOnlyTok(), prompts, settings)
    assert tok.single_calls == len(prompts)


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B", "google/gemma-3-4b-it"])
def test_real_tokenizers_count_identically(model_id, monkeypatch):
    """Long synthetic transcripts through a cached real tokenizer: the
    batched counter and the renderer's own count agree on every item."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from steerlab_server.experiment import token_preflight
    try:
        tokenizer = token_preflight._tokenizer(model_id, None,
                                               local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not in this machine's cache
        pytest.skip(f"{model_id} is not in this machine's model cache ({exc})")
    rng = random.Random(3)
    words = ["garden", "river", "lamp", "the", "window", "orange", "seven",
             "bridge", "paper", "cloud", "of", "to", "in", "a", "é", "—"]
    prompts = []
    for i in range(6):
        turns = [{"role": "system", "content": "You read short notes."}]
        for t in range(6):
            turns.append({"role": "user" if t % 2 == 0 else "assistant",
                          "content": " ".join(rng.choice(words)
                                              for _ in range(1500 + 400 * i))})
        turns.append({"role": "user", "content": "Answer A or B."})
        prompts.append({"id": f"p{i}", "prompt": "Answer A or B.",
                        "transcript": turns})
    prompts.append({"id": "plain", "prompt": " ".join(["lantern"] * 3000)})
    settings = [_setting(), _setting(system="Be brief.")]
    monkeypatch.setattr(prompt_reading, "COUNT_BATCH_CHARACTERS", 60_000)
    expected = [[prompt_render.render_for_generation(
        tokenizer, p.get("prompt") or "", model_id=model_id,
        transcript=p.get("transcript") or None, **s).prompt_token_count
        for s in settings] for p in prompts]
    assert prompt_reading.rendered_token_counts(
        tokenizer, model_id, prompts, settings) == expected

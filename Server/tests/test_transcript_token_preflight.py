"""The run's token check measures a scripted transcript whole (public issue #10).

The task-prompt loader fills a transcript item's ``prompt`` with its final user
turn, as display text. The token preflight rendered the transcript only when an
item had no text, so it measured every transcript item by its last question:
a run whose items replayed long conversations and asked one short question
logged its longest prompt as the length of that question, and a transcript
longer than the window passed the check. It now counts every item as
generation renders it, under every arm's settings, by the counter the run's
prompt-reading record and the submission's memory check use
(``prompt_reading.rendered_token_counts``).
"""

import json
import os
from types import SimpleNamespace

from steerlab_server.experiment import (prompt_reading, prompt_render,
                                        run_preflight, task_inputs)
from steerlab_server.experiment import token_preflight as tp
from steerlab_server.experiment.manifest import Manifest


class _Tokenizer:
    """One token per word; the chat template joins the turns' words."""

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=list(range(len(text.split()))))


LONG = " ".join(["word"] * 300)
QUESTION = "Why so?"
TRANSCRIPT_ROW = {"id": "replayed", "transcript": [
    {"role": "user", "content": LONG},
    {"role": "assistant", "content": LONG},
    {"role": "user", "content": QUESTION}]}
PLAIN_ROW = {"id": "plain", "prompt": "a short plain item"}


def _items():
    text = "".join(json.dumps(row) + "\n" for row in (TRANSCRIPT_ROW, PLAIN_ROW))
    return task_inputs.parse_prompts(text)


def _patch(monkeypatch, window):
    monkeypatch.setattr(tp, "_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(tp, "context_window", lambda *args, **kwargs: window)


def test_a_transcript_is_counted_whole_not_by_its_display_text(monkeypatch):
    _patch(monkeypatch, window=200)
    items = _items()
    # The loader's display text is the final question alone.
    assert items[0]["prompt"] == QUESTION
    report = tp.preflight(items, model_id="org/m", revision=None, max_tokens=10)
    counts = {item["id"]: item["promptTokens"] for item in report["items"]}
    whole = prompt_render.render_for_generation(
        _Tokenizer(), QUESTION, model_id="org/m",
        transcript=items[0]["transcript"]).prompt_token_count
    question_only = prompt_render.render(
        _Tokenizer(), QUESTION, model_id="org/m").prompt_token_count
    assert whole == 602 and question_only == 2  # written by hand: 300+300+2
    assert counts["replayed"] == whole
    assert counts["plain"] == 4
    # The verdict uses the whole rendering: the question alone would fit.
    assert report["promptBudget"] == 200 - 10 - tp.CONTEXT_BUDGET_RESERVE
    assert question_only <= report["promptBudget"] < whole
    assert [item["id"] for item in report["overflowItems"]] == ["replayed"]
    assert "replayed (602 tokens" in tp.refusal(report)


def test_the_count_is_the_longest_rendering_across_the_arms(monkeypatch):
    _patch(monkeypatch, window=10_000)
    items = _items()
    persona = " ".join(["frame"] * 50)
    settings = [
        {"prompt_mode": "chatAssistant", "system_prompt": None,
         "qwen_thinking_enabled": False, "reasoning_effort": None},
        {"prompt_mode": "chatAssistant", "system_prompt": persona,
         "qwen_thinking_enabled": False, "reasoning_effort": None},
    ]
    report = tp.preflight(items, model_id="org/m", revision=None,
                          max_tokens=10, settings=settings)
    counts = {item["id"]: item["promptTokens"] for item in report["items"]}
    rows = prompt_reading.rendered_token_counts(_Tokenizer(), "org/m", items,
                                                settings)
    assert [counts["replayed"], counts["plain"]] == [max(r) for r in rows]
    assert counts["plain"] == 4 + 50  # the agent's frame makes it longer


def test_the_run_preflight_logs_and_refuses_on_the_whole_rendering(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    path = os.path.join(root, "prompts", "tasks", "replayed.jsonl")
    os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(TRANSCRIPT_ROW) + "\n")
    manifest = Manifest.from_dict({
        "name": "replay", "modelID": "org/m", "concepts": [],
        "maxTokens": 10, "taskPromptsFile": "prompts/tasks/replayed.jsonl"})
    lines = []
    _patch(monkeypatch, window=1_000)
    run_preflight.token_preflight_or_warn(manifest, None, root, lines.append)
    assert lines == ["token preflight: 1 prompts, longest 602 tokens, budget "
                     f"{1_000 - 10 - tp.CONTEXT_BUDGET_RESERVE} — all fit"]
    _patch(monkeypatch, window=300)
    try:
        run_preflight.token_preflight_or_warn(manifest, None, root,
                                              lines.append)
    except RuntimeError as exc:
        assert "replayed (602 tokens" in str(exc)
    else:
        raise AssertionError("a transcript longer than the window passed")

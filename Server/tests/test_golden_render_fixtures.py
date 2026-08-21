"""Golden render-fixture parity for BOTH model families (qwen + gemma).

``prompts/fixtures/render/*.json`` are committed goldens generated from this
engine's own renderer (``prompt_render``) + the real HF tokenizers (see
``prompts/fixtures/render/README.md``). Each test re-renders the recorded
inputs and asserts byte-identical ``rendered`` text, identical ``tokenIDs``,
and the recorded ``bosCount`` (the double-BOS tripwire — Gemma must stay at
exactly one BOS, with or without a system prompt).

A tokenizer or chat-template upgrade that changes any of this must fail
LOUDLY here, never skip silently. The only permitted skip is a missing
tokenizer in the local HF cache, and that skip names the model.
"""

import json
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

from steerlab_server.experiment import prompt_render

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "prompts" / "fixtures" / "render"
)
FIXTURE_PATHS = sorted(FIXTURE_DIR.glob("*.json"))


def test_fixture_inventory_present():
    """The goldens are committed data — an empty directory is a failure
    (e.g. a bad checkout), not a skip."""
    assert FIXTURE_PATHS, f"no golden render fixtures found in {FIXTURE_DIR}"
    families = {p.name.split("--")[0] for p in FIXTURE_PATHS}
    assert {"qwen", "gemma"} <= families, (
        f"golden coverage must span both families, found only {families}")


_TOKENIZERS: dict[str, object] = {}


def _tokenizer(model_id: str):
    if model_id not in _TOKENIZERS:
        try:
            _TOKENIZERS[model_id] = transformers.AutoTokenizer.from_pretrained(
                model_id, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - not cached / offline
            _TOKENIZERS[model_id] = exc
    tok = _TOKENIZERS[model_id]
    if isinstance(tok, Exception):
        pytest.skip(
            f"tokenizer for {model_id} not in local HF cache — golden render "
            f"parity for this model NOT verified here: {tok}")
    return tok


def _bos_count(ids: list[int], bos_id) -> int:
    if bos_id is None:
        return 0
    n = 0
    for tok in ids:
        if tok != bos_id:
            break
        n += 1
    return n


def _rerender(tok, fx: dict) -> tuple[str, list[int]]:
    inputs = fx["inputs"]
    api = inputs["api"]
    model_id = fx["modelID"]
    if api == "render":
        r = prompt_render.render(
            tok, inputs["prompt"], model_id=model_id,
            prompt_mode=inputs["promptMode"],
            system_prompt=inputs["systemPrompt"],
            qwen_thinking_enabled=inputs["qwenThinkingEnabled"])
        return r.text, list(r.input_ids)
    if api == "render_messages":
        # `messages` may carry `seeded: true` provenance flags (the
        # send-as-assistant instrument) — rendering must ignore them — and
        # `continueFinalMessage` renders an incomplete final assistant turn
        # (assistant-prefix prefill; no generation prompt).
        r = prompt_render.render_messages(
            tok, inputs["messages"], model_id=model_id,
            prompt_mode=inputs["promptMode"],
            system_prompt=inputs["systemPrompt"],
            qwen_thinking_enabled=inputs["qwenThinkingEnabled"],
            continue_final_message=inputs.get("continueFinalMessage", False))
        return r.text, list(r.input_ids)
    if api == "render_transcript":
        # Scripted-transcript study items (`study_transcript` fixtures): the
        # recorded systemPrompt is a DECOY — the transcript carries its own
        # system turn, which must replace it (the composition rule is part of
        # the golden claim; the run loop's generate/score_options render
        # through this same function).
        r = prompt_render.render_transcript(
            tok, inputs["transcript"], model_id=model_id,
            prompt_mode=inputs["promptMode"],
            system_prompt=inputs["systemPrompt"],
            qwen_thinking_enabled=inputs["qwenThinkingEnabled"])
        return r.text, list(r.input_ids)
    if api == "render_reader":
        text = prompt_render.render_reader(inputs["text"], model_id=model_id)
        # Extractor convention: tokenizer defaults, single tokenizer-added BOS.
        ids = list(tok(text, add_special_tokens=True).input_ids)
        return text, ids
    raise AssertionError(f"unknown fixture api {api!r} in {fx['case']}")


@pytest.mark.parametrize(
    "path", FIXTURE_PATHS, ids=[p.stem for p in FIXTURE_PATHS])
def test_golden_render_fixture(path: Path):
    fx = json.loads(path.read_text(encoding="utf-8"))
    tok = _tokenizer(fx["modelID"])

    text, ids = _rerender(tok, fx)

    assert text == fx["rendered"], (
        f"{path.name}: rendered string drifted from golden.\n"
        f"golden:  {fx['rendered']!r}\n"
        f"current: {text!r}")
    assert ids == fx["tokenIDs"], (
        f"{path.name}: token ids drifted from golden "
        f"(golden {len(fx['tokenIDs'])} tokens, current {len(ids)})")
    assert len(ids) == fx["promptTokenCount"]

    # Double-BOS tripwire: the tokenizer's own bos id must agree with the
    # recorded one, and the leading-BOS count must be exactly as recorded
    # (gemma: 1 in every case, qwen: 0).
    assert tok.bos_token_id == fx["bosTokenID"], (
        f"{path.name}: tokenizer bos_token_id {tok.bos_token_id} != "
        f"recorded {fx['bosTokenID']}")
    assert _bos_count(ids, tok.bos_token_id) == fx["bosCount"], (
        f"{path.name}: leading-BOS count changed — double-BOS (or lost BOS) "
        f"regression")

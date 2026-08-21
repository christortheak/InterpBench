#!/usr/bin/env python3
"""Generate cross-engine golden RENDER fixtures for both model families.

Run from the project root with the server venv (the AUTHORITATIVE tokenizer):

    Server/.venv.nosync/bin/python prompts/fixtures/render/generate.py

Renders every case through the Python engine's prompt renderer
(``steerlab_server.experiment.prompt_render`` — the same module the server's
study runner uses) against tokenizers loaded from the local HF cache
(``local_files_only=True``; never downloads). One JSON file per
(model, case) is written next to this script; see README.md for the schema
and the case inventory.

The fixtures pin the documented #1 tokenizer risk:
  - Gemma 3 has no system role — system text folds into the first user turn.
  - Double-BOS when combining a chat template render with a second
    special-token-adding tokenization (``bosCount`` is the tripwire).
  - Qwen3 no-think template (``enable_thinking=False``).

Inputs are frozen constants below. Regenerating with the same tokenizer
revisions must be byte-identical (JSON is written with sorted keys and a
trailing newline). If a tokenizer/template upgrade changes any rendered
string or token ids, the fixture diff IS the finding — commit it consciously.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # prompts/fixtures/render -> project root
sys.path.insert(0, str(ROOT / "Server"))

from steerlab_server.experiment import prompt_render  # noqa: E402

import transformers  # noqa: E402

# --- frozen inputs (do not edit without regenerating every fixture) ----------

PROMPT_PLAIN = "Decide the case and state the holding in one sentence."
SYSTEM = "You are a terse appellate judge. Answer in plain language."
PROMPT_WITH_SYSTEM = "Summarize the controlling precedent for this appeal."
PROMPT_NOTHINK = "Was the evidence admissible? Answer yes or no."
PROMPT_RAW = "The court held that"
MULTITURN_MESSAGES = [
    {"role": "user", "content": "State the elements of common-law burglary."},
    {"role": "assistant", "content": (
        "Breaking, entering, the dwelling of another, at night, "
        "with intent to commit a felony inside.")},
    {"role": "user", "content": (
        "Now apply them to a daytime entry through an unlocked door.")},
]
READER_SCAFFOLD = (
    "Consider the fairness of the following decision.\n\n"
    "The defendant's sentence was doubled on appeal without new findings.\n\n"
    "The amount of fairness in that decision is"
)
# Send-as-assistant (metacognition instrument): the assistant turn is
# researcher-authored ("seeded": true). The flag is provenance for the
# transcript RECORD only — rendering must be byte-identical to the same
# history where that turn was a real assistant message (asserted at
# generation time below).
SEEDED_MESSAGES = [
    {"role": "user", "content": "State the holding in one sentence."},
    {"role": "assistant", "content": (
        "I will not analyze this case; discussing it makes me uncomfortable."),
     "seeded": True},
    {"role": "user", "content": (
        "Why did you refuse? Was that refusal your own decision?")},
]
# Assistant-prefix continuation ("prefill"): the final assistant turn is
# INCOMPLETE; the render must end exactly at the prefix (no end-of-turn
# marker, no new generation prompt) — transformers' continue_final_message.
PREFILL_MESSAGES = [
    {"role": "user", "content": "Decide the case and explain briefly."},
    {"role": "assistant", "content": "The court holds that the appeal",
     "seeded": True},
]
# Assistant-FIRST transcript (seeded assistant turn before any user turn).
# Only rendered where the family's chat template permits it (Qwen; Gemma's
# template enforces user-first alternation — see
# prompt_render.conversation_constraints) — the golden keeps assistant-first
# seeding working on the permissive family.
ASSISTANT_FIRST_MESSAGES = [
    {"role": "assistant", "content": (
        "Before you ask: I have already reviewed the record in this appeal."),
     "seeded": True},
    {"role": "user", "content": "Then state the holding in one sentence."},
]
# Scripted-transcript STUDY item (render_transcript — the metacognition-study
# instrument): a pinned transcript with its OWN system turn plus a seeded
# assistant turn. The study-level system prompt below must be REPLACED by the
# transcript's system turn (the composition rule) — asserted at generation
# time against the system_prompt=None twin.
STUDY_TRANSCRIPT = [
    {"role": "system", "content": (
        "You answer questions about your own prior statements accurately.")},
    {"role": "user", "content": "Name a color, and nothing else."},
    {"role": "assistant", "content": "I would rather not name a color.",
     "seeded": True},
    {"role": "user", "content": (
        "Did you just refuse my request? Answer with exactly one word: "
        "yes or no.")},
]
STUDY_SYSTEM_REPLACED = (
    "STUDY-LEVEL SYSTEM PROMPT — must never appear in the render; the "
    "transcript's own system turn replaces it.")

MODELS = {
    "qwen": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B-MLX-4bit"],
    "gemma": ["google/gemma-3-4b-it", "mlx-community/gemma-3-4b-it-4bit"],
}


def bos_count(ids: list[int], bos_id: int | None) -> int:
    """Consecutive BOS tokens at the sequence start — the double-BOS tripwire."""
    if bos_id is None:
        return 0
    n = 0
    for tok in ids:
        if tok != bos_id:
            break
        n += 1
    return n


def fixture(family: str, model_id: str, case: str, inputs: dict,
            rendered: str, ids: list[int], bos_id: int | None,
            tokenizer_class: str) -> dict:
    return {
        "family": family,
        "modelID": model_id,
        "case": case,
        "inputs": inputs,
        "rendered": rendered,
        "tokenIDs": ids,
        "promptTokenCount": len(ids),
        "bosCount": bos_count(ids, bos_id),
        "bosTokenID": bos_id,
        "generator": {
            "script": "prompts/fixtures/render/generate.py",
            "engine": "python-hf",
            "transformersVersion": transformers.__version__,
            "tokenizerClass": tokenizer_class,
            "generatedOn": date.today().isoformat(),
        },
    }


def cases_for(family: str, model_id: str, tok) -> list[dict]:
    bos_id = tok.bos_token_id
    cls = type(tok).__name__
    out: list[dict] = []

    def render_case(case: str, prompt: str, *, system: str | None = None,
                    mode: str = prompt_render.CHAT_ASSISTANT,
                    thinking: bool = False) -> dict:
        r = prompt_render.render(
            tok, prompt, model_id=model_id, prompt_mode=mode,
            system_prompt=system, qwen_thinking_enabled=thinking)
        inputs = {
            "api": "render",
            "prompt": prompt,
            "promptMode": mode,
            "systemPrompt": system,
            "qwenThinkingEnabled": thinking,
        }
        return fixture(family, model_id, case, inputs, r.text, r.input_ids,
                       bos_id, cls)

    # 1. plain single user prompt, chatAssistant
    out.append(render_case("chat_plain", PROMPT_PLAIN))

    # 2. user prompt WITH a system prompt (the Gemma folding case)
    out.append(render_case("chat_system", PROMPT_WITH_SYSTEM, system=SYSTEM))

    # 3. multi-turn (2 user + 1 assistant) with system prompt
    r = prompt_render.render_messages(
        tok, MULTITURN_MESSAGES, model_id=model_id,
        prompt_mode=prompt_render.CHAT_ASSISTANT, system_prompt=SYSTEM,
        qwen_thinking_enabled=False)
    out.append(fixture(
        family, model_id, "chat_multiturn_system",
        {
            "api": "render_messages",
            "messages": MULTITURN_MESSAGES,
            "promptMode": prompt_render.CHAT_ASSISTANT,
            "systemPrompt": SYSTEM,
            "qwenThinkingEnabled": False,
        },
        r.text, r.input_ids, bos_id, cls))

    # 4. Qwen thinking-mode OFF single prompt (the no-think template).
    if family == "qwen":
        out.append(render_case(
            "qwen_nothink", PROMPT_NOTHINK, thinking=False))

    # 5. rawCompletion single prompt (extra coverage: the raw path is where the
    #    Qwen " /no_think" suffix and the tokenizer-default BOS live).
    out.append(render_case(
        "raw_completion", PROMPT_RAW, mode=prompt_render.RAW_COMPLETION))

    # 6. Seeded assistant turn (send-as-assistant): the golden IS the render
    #    of the seeded history; the flags must be inert — asserted here against
    #    the flag-stripped twin AND an `edited: true` twin (the second
    #    provenance flag, generated-then-altered), so the fixture pins
    #    "indistinguishable to the MODEL, distinguishable in the RECORD" with
    #    the real tokenizer for BOTH flags.
    r = prompt_render.render_messages(
        tok, SEEDED_MESSAGES, model_id=model_id,
        prompt_mode=prompt_render.CHAT_ASSISTANT, system_prompt=None,
        qwen_thinking_enabled=False)
    stripped = [{k: v for k, v in m.items() if k != "seeded"}
                for m in SEEDED_MESSAGES]
    twin = prompt_render.render_messages(
        tok, stripped, model_id=model_id,
        prompt_mode=prompt_render.CHAT_ASSISTANT, system_prompt=None,
        qwen_thinking_enabled=False)
    assert r.text == twin.text and r.input_ids == twin.input_ids, (
        f"{model_id}: seeded flag perturbed the render")
    flagged = [dict(m, edited=True) if m["role"] == "assistant" else dict(m)
               for m in SEEDED_MESSAGES]
    edited_twin = prompt_render.render_messages(
        tok, flagged, model_id=model_id,
        prompt_mode=prompt_render.CHAT_ASSISTANT, system_prompt=None,
        qwen_thinking_enabled=False)
    assert (edited_twin.text == r.text
            and edited_twin.input_ids == r.input_ids), (
        f"{model_id}: edited flag perturbed the render")
    out.append(fixture(
        family, model_id, "chat_seeded_assistant",
        {
            "api": "render_messages",
            "messages": SEEDED_MESSAGES,
            "promptMode": prompt_render.CHAT_ASSISTANT,
            "systemPrompt": None,
            "qwenThinkingEnabled": False,
        },
        r.text, r.input_ids, bos_id, cls))

    # 6b. Assistant-FIRST seeded transcript — only where the family's
    #     template permits it (Qwen; conversation_constraints is the gate,
    #     mirroring the client pre-flight). The golden keeps assistant-first
    #     rendering working on the permissive family.
    if not prompt_render.conversation_constraints(
            model_id).requires_leading_user_turn:
        r = prompt_render.render_messages(
            tok, ASSISTANT_FIRST_MESSAGES, model_id=model_id,
            prompt_mode=prompt_render.CHAT_ASSISTANT, system_prompt=None,
            qwen_thinking_enabled=False)
        assert ASSISTANT_FIRST_MESSAGES[0]["content"] in r.text, (
            f"{model_id}: assistant-first turn missing from the render")
        out.append(fixture(
            family, model_id, "chat_assistant_first",
            {
                "api": "render_messages",
                "messages": ASSISTANT_FIRST_MESSAGES,
                "promptMode": prompt_render.CHAT_ASSISTANT,
                "systemPrompt": None,
                "qwenThinkingEnabled": False,
            },
            r.text, r.input_ids, bos_id, cls))

    # 6c. Scripted-transcript STUDY item (render_transcript): renders the
    #     pinned transcript through the SAME render_messages path, with the
    #     transcript's own system turn REPLACING the study system prompt.
    #     The generator asserts the replacement rule (a decoy study system
    #     prompt changes nothing) and seeded-flag inertness with the real
    #     tokenizer; the committed golden then pins byte parity for both
    #     engines' study run loops.
    r = prompt_render.render_transcript(
        tok, STUDY_TRANSCRIPT, model_id=model_id,
        prompt_mode=prompt_render.CHAT_ASSISTANT,
        system_prompt=STUDY_SYSTEM_REPLACED, qwen_thinking_enabled=False)
    assert STUDY_SYSTEM_REPLACED not in r.text, (
        f"{model_id}: study system prompt leaked into a transcript render "
        "that carries its own system turn")
    no_study_system = prompt_render.render_transcript(
        tok, STUDY_TRANSCRIPT, model_id=model_id,
        prompt_mode=prompt_render.CHAT_ASSISTANT,
        system_prompt=None, qwen_thinking_enabled=False)
    assert (r.text == no_study_system.text
            and r.input_ids == no_study_system.input_ids), (
        f"{model_id}: transcript system turn did not REPLACE the study "
        "system prompt")
    stripped_transcript = [{k: v for k, v in m.items() if k != "seeded"}
                           for m in STUDY_TRANSCRIPT]
    twin = prompt_render.render_transcript(
        tok, stripped_transcript, model_id=model_id,
        prompt_mode=prompt_render.CHAT_ASSISTANT,
        system_prompt=STUDY_SYSTEM_REPLACED, qwen_thinking_enabled=False)
    assert r.text == twin.text and r.input_ids == twin.input_ids, (
        f"{model_id}: seeded flag perturbed the transcript render")
    out.append(fixture(
        family, model_id, "study_transcript",
        {
            "api": "render_transcript",
            "transcript": STUDY_TRANSCRIPT,
            "promptMode": prompt_render.CHAT_ASSISTANT,
            "systemPrompt": STUDY_SYSTEM_REPLACED,
            "qwenThinkingEnabled": False,
        },
        r.text, r.input_ids, bos_id, cls))

    # 7. Assistant-prefix continuation (prefill): rendered via transformers'
    #    own continue_final_message; must end exactly at the seeded prefix.
    r = prompt_render.render_messages(
        tok, PREFILL_MESSAGES, model_id=model_id,
        prompt_mode=prompt_render.CHAT_ASSISTANT, system_prompt=None,
        qwen_thinking_enabled=False, continue_final_message=True)
    assert r.text.endswith(PREFILL_MESSAGES[-1]["content"]), (
        f"{model_id}: continue_final_message render does not end at the prefix")
    out.append(fixture(
        family, model_id, "chat_prefill_continue",
        {
            "api": "render_messages",
            "messages": PREFILL_MESSAGES,
            "promptMode": prompt_render.CHAT_ASSISTANT,
            "systemPrompt": None,
            "qwenThinkingEnabled": False,
            "continueFinalMessage": True,
        },
        r.text, r.input_ids, bos_id, cls))

    # 8. RepE reader scaffold (render_reader): plain text by design; token ids
    #    recorded with the extractor's tokenization convention
    #    (add_special_tokens=True — single tokenizer-added BOS, no template).
    scaffold = prompt_render.render_reader(READER_SCAFFOLD, model_id=model_id)
    ids = list(tok(scaffold, add_special_tokens=True).input_ids)
    out.append(fixture(
        family, model_id, "reader_scaffold",
        {
            "api": "render_reader",
            "text": READER_SCAFFOLD,
            "tokenization": "add_special_tokens=True (extractor default)",
        },
        scaffold, ids, bos_id, cls))

    return out


def main() -> None:
    # Optional case filter (`--case study_transcript`): write only the named
    # case's fixtures, leaving every other committed golden byte-untouched
    # (regenerating all files would churn `generatedOn` stamps without a
    # tokenizer change).
    only_case = None
    args = sys.argv[1:]
    if args[:1] == ["--case"] and len(args) >= 2:
        only_case = args[1]
    written = []
    for family, model_ids in MODELS.items():
        for model_id in model_ids:
            tok = transformers.AutoTokenizer.from_pretrained(
                model_id, local_files_only=True)
            for fx in cases_for(family, model_id, tok):
                if only_case is not None and fx["case"] != only_case:
                    continue
                name = "{}--{}--{}.json".format(
                    family, model_id.replace("/", "__"), fx["case"])
                path = HERE / name
                path.write_text(
                    json.dumps(fx, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n",
                    encoding="utf-8")
                written.append(name)
                print(f"wrote {name}  tokens={fx['promptTokenCount']} "
                      f"bosCount={fx['bosCount']}")
    print(f"{len(written)} fixtures")


if __name__ == "__main__":
    main()

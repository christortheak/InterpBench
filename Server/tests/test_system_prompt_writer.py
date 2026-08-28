"""``steerlab experiment set-system-prompt`` — the study's DEPLOYMENT FRAME.

Field discovery (2026-08-28): a replication study whose donor carries a
judge-persona system prompt could not be authored headlessly. On the Mac there
was no writer at all (the Study Setup panel's TextField was the only one); here
``systemPrompt`` was reachable as a ``set-protocol`` key, but nothing named it,
so an authoring agent looking for the verb found none — and running the study
without the persona would have been a different study, so it correctly refused
to improvise one. Both surfaces now have the verb, spelled identically.

This module tests the WRITE and the APPLICATION. A writer whose value never
reaches the model would satisfy the first and fail the point: what is written
must be inserted as a genuine system turn where the model's chat template has a
system role, and prepended to the first user turn where it does not. Both
branches are asserted through :func:`prompt_render.render` — the real render
path, not a re-implementation of its rule.

Swift twin: ``Tests/ExperimentKitTests/SystemPromptWriterTests.swift``.
"""

import json
import os

import pytest

from steerlab_server import cli_envelope, client_cli
from steerlab_server.experiment import experiment_store as store
from steerlab_server.experiment import prompt_render
from steerlab_server.experiment import system_prompt as system_prompt_mod


@pytest.fixture
def workspace(tmp_path):
    """An explicit ROOT for every call, never an ambient one: ``project_root``
    is process-global, and a suite that leaned on the environment would read a
    neighbouring test's workspace."""
    return str(tmp_path)


def _run(root, argv):
    return client_cli.main(list(argv) + ["--root", root])


def _document(capsys):
    return json.loads(capsys.readouterr().out)


def _create(root, name, model="Qwen/Qwen3-14B"):
    return store.create(name, model_id=model, root=root)


class FakeTokenizer:
    """The branching harness ``test_prompt_render_parity`` already uses: the
    rendering is tested without a model, because what is under test here is
    WHICH TURNS the written frame becomes."""

    def __init__(self):
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        self.last_messages = messages
        return "<BOS>" + " ".join(
            f"[{m['role']}]{m['content']}" for m in messages)

    def __call__(self, text, add_special_tokens=True):
        class _Out:
            input_ids = list(range(len(text.split())))
        return _Out()


# =============================================================================
# 1. The write
# =============================================================================


def test_the_verb_stores_the_frame_trimmed(workspace, capsys):
    _create(workspace, "demo")
    assert _run(workspace, ["experiment", "set-system-prompt", "demo",
                            "  You are a strict grader.  ", "--json"]) == 0
    document = _document(capsys)
    assert document["state"] == "ready"
    assert document["changed"] is True
    assert store.load_raw("demo", workspace)["systemPrompt"] == \
        "You are a strict grader."


def test_the_empty_string_clears_the_declaration(workspace, capsys):
    _create(workspace, "demo")
    _run(workspace, ["experiment", "set-system-prompt", "demo", "a frame"])
    capsys.readouterr()
    assert _run(workspace,
                ["experiment", "set-system-prompt", "demo", "", "--json"]) == 0
    document = _document(capsys)
    assert "systemPrompt" not in store.load_raw("demo", workspace)
    assert document["result"]["systemPrompt"] is None
    assert "cleared the system prompt" in document["message"]


def test_a_whitespace_only_frame_is_a_cleared_one(workspace):
    _create(workspace, "demo")
    store.set_system_prompt("demo", "a frame", workspace)
    store.set_system_prompt("demo", "   \n  ", workspace)
    assert "systemPrompt" not in store.load_raw("demo", workspace)


def test_a_frozen_study_refuses_with_the_immutability_line(workspace):
    _create(workspace, "demo")
    document = store.load_raw("demo", workspace)
    document["status"] = "frozen"
    store.save_raw(document, workspace, freeze_transition=True)
    with pytest.raises(store.ExperimentStoreError) as caught:
        store.set_system_prompt("demo", "a frame", workspace)
    assert "frozen" in str(caught.value)


def test_a_missing_value_is_a_usage_refusal_that_explains_delivery(
        workspace, capsys):
    """The help text has to say what setting a frame PHYSICALLY DOES: the
    delivery route is capability-dependent, and a researcher arming a persona
    needs to know which one this model gives them."""
    _create(workspace, "demo")
    assert _run(workspace,
                ["experiment", "set-system-prompt", "demo", "--json"]) == 64
    document = _document(capsys)
    assert document["error"]["code"] == "usage"
    spec = next(s for s in client_cli.CLIENT_VERB_SPECS
                if s.label == "experiment set-system-prompt")
    assert "genuine system turn" in spec.purpose
    assert "prepended to the first user turn" in spec.purpose


# =============================================================================
# 2. The application — what the model actually receives
# =============================================================================


def test_a_written_frame_becomes_a_system_turn_on_a_system_role_family(
        workspace):
    _create(workspace, "demo", model="Qwen/Qwen3-14B")
    store.set_system_prompt("demo", "You are a strict grader.", workspace)
    manifest = store.load_raw("demo", workspace)

    tokenizer = FakeTokenizer()
    prompt_render.render(
        tokenizer, "Decide the item.", model_id=manifest["modelID"],
        prompt_mode=manifest.get("promptMode") or prompt_render.CHAT_ASSISTANT,
        system_prompt=manifest["systemPrompt"])
    assert [m["role"] for m in tokenizer.last_messages] == ["system", "user"]
    assert tokenizer.last_messages[0]["content"] == "You are a strict grader."
    # …and the user turn is untouched: the frame did not eat it.
    assert tokenizer.last_messages[1]["content"] == "Decide the item."
    assert prompt_render.has_system_role(manifest["modelID"])


def test_a_written_frame_is_prepended_on_a_family_without_a_system_role(
        workspace):
    """Gemma has no system role, so the SAME bytes are prepended to the first
    user turn. The frame still reaches the model — which is the property the
    writer promises — by the only route the template allows."""
    _create(workspace, "demo", model="google/gemma-3-27b-it")
    store.set_system_prompt("demo", "You are a strict grader.", workspace)
    manifest = store.load_raw("demo", workspace)

    tokenizer = FakeTokenizer()
    prompt_render.render(
        tokenizer, "Decide the item.", model_id=manifest["modelID"],
        system_prompt=manifest["systemPrompt"])
    assert [m["role"] for m in tokenizer.last_messages] == ["user"]
    assert tokenizer.last_messages[0]["content"] == \
        "You are a strict grader.\n\nDecide the item."
    assert not prompt_render.has_system_role(manifest["modelID"])


def test_a_written_frame_is_prepended_under_raw_completion(workspace):
    """The third route: no template at all, and the frame is prepended to the
    prompt TEXT. There is no prompt mode on which a written frame is dropped,
    which is why the setter has no mode gate."""
    _create(workspace, "demo", model="meta/other-model")
    store.set_system_prompt("demo", "Context.", workspace)
    store.set_protocol("demo", {"promptMode": prompt_render.RAW_COMPLETION},
                       workspace)
    manifest = store.load_raw("demo", workspace)

    rendered = prompt_render.render(
        FakeTokenizer(), "Continue.", model_id=manifest["modelID"],
        prompt_mode=manifest["promptMode"],
        system_prompt=manifest["systemPrompt"])
    assert rendered.text == "Context.\n\nContinue."


def test_the_echo_names_the_delivery_route(workspace, capsys):
    _create(workspace, "qwen", model="Qwen/Qwen3-14B")
    _create(workspace, "gemma", model="google/gemma-3-27b-it")

    assert _run(workspace, ["experiment", "set-system-prompt", "qwen", "F.",
                 "--json"]) == 0
    on_qwen = _document(capsys)["result"]
    assert _run(workspace, ["experiment", "set-system-prompt", "gemma", "F.",
                 "--json"]) == 0
    on_gemma = _document(capsys)["result"]

    assert on_qwen["delivery"] == "systemTurn"
    assert on_gemma["delivery"] == "prependedToFirstUserTurn"
    # The DERIVED hash is the frame's own bytes, not the rendering, so the two
    # families agree on it.
    assert on_qwen["studyFrameHash"] == on_gemma["studyFrameHash"]
    assert on_qwen["studyFrameHash"] == system_prompt_mod.text_hash("F.")


def test_the_frame_composes_after_an_agent_persona(workspace):
    """An agent arm is not displaced: composition puts the persona first and
    this frame second, so declaring one ADDS a frame rather than replacing an
    identity."""
    _create(workspace, "demo")
    store.set_system_prompt("demo", "Answer in JSON.", workspace)
    manifest = store.load_raw("demo", workspace)
    assert system_prompt_mod.compose(
        "You are a cautious reviewer.", manifest["systemPrompt"]) == \
        "You are a cautious reviewer.\n\nAnswer in JSON."


# =============================================================================
# 3. The one path a frame does NOT reach
# =============================================================================


def _pin_prompts(workspace, name, lines):
    path = os.path.join(workspace, "prompts", "tasks", f"{name}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(line) + "\n" for line in lines))
    return f"prompts/tasks/{name}.jsonl"


def test_transcript_items_with_their_own_system_turn_raise_the_advisory(
        workspace, capsys):
    """A pinned item whose transcript opens with its OWN system turn replaces
    the study frame for that item (``prompt_render.render_transcript``). That
    is a declared cross-engine rule, not a bug — but it was silent, and a
    researcher who has just typed a persona is exactly the person who needs to
    hear it."""
    _create(workspace, "demo")
    relative = _pin_prompts(workspace, "t", [
        {"id": "a", "transcript": [{"role": "system", "content": "Be terse."},
                                   {"role": "user", "content": "Hi"}]},
        {"id": "b", "transcript": [{"role": "user", "content": "Hello"}]},
        {"id": "c", "prompt": "Plain item"},
    ])
    store.set_protocol("demo", {"taskPromptsFile": relative}, workspace)
    capsys.readouterr()

    assert _run(workspace, ["experiment", "set-system-prompt", "demo", "Frame.",
                 "--json"]) == 0
    document = _document(capsys)
    assert document["state"] == "okWithAdvisories"
    # Advisories NEVER change the exit code.
    assert cli_envelope.exit_code_for(document["state"]) == 0
    assert document["advisories"][0]["code"] == "systemPromptNotApplied"
    detail = document["advisories"][0]["detail"]
    assert "1 of 3 pinned item(s)" in detail
    assert "reaches the other 2 only" in detail
    assert document["result"]["itemsWithOwnSystemTurn"] == 1


def test_a_frame_that_reaches_every_item_is_silent(workspace, capsys):
    _create(workspace, "demo")
    relative = _pin_prompts(workspace, "plain", [
        {"id": "a", "prompt": "One"}, {"id": "b", "prompt": "Two"}])
    store.set_protocol("demo", {"taskPromptsFile": relative}, workspace)
    capsys.readouterr()

    assert _run(workspace, ["experiment", "set-system-prompt", "demo", "F.",
                 "--json"]) == 0
    declared = _document(capsys)
    assert declared["state"] == "ready"
    assert "advisories" not in declared or not declared["advisories"]

    # …and none when the frame is being CLEARED, where the substitution
    # cannot matter.
    assert _run(workspace,
                ["experiment", "set-system-prompt", "demo", "", "--json"]) == 0
    cleared = _document(capsys)
    assert "advisories" not in cleared or not cleared["advisories"]


def test_an_unreadable_prompts_pin_does_not_block_the_write(
        workspace, capsys):
    """An unreadable pin is ``verify()``'s business, never this counter's: the
    advisory input degrades to "nothing to say" rather than refusing a write
    that has nothing to do with the prompts file."""
    _create(workspace, "demo")
    relative = _pin_prompts(workspace, "gone", [{"id": "a", "prompt": "One"}])
    store.set_protocol("demo", {"taskPromptsFile": relative}, workspace)
    os.remove(os.path.join(workspace, relative))
    capsys.readouterr()

    assert _run(workspace, ["experiment", "set-system-prompt", "demo", "F.",
                 "--json"]) == 0
    document = _document(capsys)
    assert document["state"] == "ready"
    assert store.load_raw("demo", workspace)["systemPrompt"] == "F."


def test_the_counter_reads_the_same_file_the_renderer_would(workspace):
    """The count is not a guess about the file: the items it flags are exactly
    the ones ``render_transcript`` resolves a DIFFERENT system prompt for."""
    _create(workspace, "demo")
    relative = _pin_prompts(workspace, "mixed", [
        {"id": "a", "transcript": [{"role": "system", "content": "Own."},
                                   {"role": "user", "content": "Hi"}]},
        {"id": "b", "transcript": [{"role": "user", "content": "Hello"}]},
    ])
    store.set_protocol("demo", {"taskPromptsFile": relative}, workspace)
    document = store.load_raw("demo", workspace)
    assert store.transcript_system_turn_count(document, workspace) == (1, 2)

    # The claim, checked against the renderer itself.
    with open(os.path.join(workspace, relative), encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    own, inherited = items[0], items[1]
    tokenizer = FakeTokenizer()
    prompt_render.render_transcript(
        tokenizer, own["transcript"], model_id="Qwen/Qwen3-14B",
        system_prompt="Frame.")
    assert tokenizer.last_messages[0] == {"role": "system", "content": "Own."}
    prompt_render.render_transcript(
        tokenizer, inherited["transcript"], model_id="Qwen/Qwen3-14B",
        system_prompt="Frame.")
    assert tokenizer.last_messages[0] == {"role": "system",
                                          "content": "Frame."}


# =============================================================================
# 4. Cross-engine parity
# =============================================================================


def test_the_advisory_detail_twins_the_swift_sentence():
    """Copied from ``ExperimentStore.systemPromptNotAppliedDetail``
    (``Sources/ExperimentKit/ExperimentStore.swift``), written out
    independently so neither engine can quietly follow the other."""
    swift_literal = (
        "1 of 3 pinned item(s) in prompts/tasks/t.jsonl carry their own "
        "leading system turn, which REPLACES this study's system prompt for "
        "those items — the frame declared here reaches the other 2 only. "
        "Remove the transcripts' system turns to put every item under one "
        "frame, or keep them and say in METHODS that the frame is per-item.")
    assert store.system_prompt_not_applied_detail(
        1, 3, "prompts/tasks/t.jsonl") == swift_literal


def test_the_engine_redirects_the_verb_and_names_both_spellings():
    """The engine on compute hardware never authors, so it redirects — and the
    redirect names the CLIENT spelling too, because a Linux caller cannot run
    ``steerlab-cli``."""
    from steerlab_server import cli

    assert "set-system-prompt" in \
        cli_envelope.MAC_AUTHORITY_VERBS["experiment"]
    spelling = cli._client_spelling("experiment set-system-prompt")
    assert spelling.startswith(
        f"{client_cli.PROGRAM} experiment set-system-prompt ")


def test_the_field_stays_reachable_as_a_protocol_key():
    """Unlike the two measurement declarations, this IS a plain field
    assignment: nothing is derived from a workspace file, so ``set-protocol
    --set systemPrompt=…`` keeps working and the key stays in the vocabulary.
    The verb exists beside it for discoverability and for an echo that can say
    what the model will receive."""
    assert "systemPrompt" in store.PROTOCOL_FIELDS

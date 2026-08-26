"""System-prompt COMPOSITION (maintainer ruling, 2026-08-24).

Before this, the two system-prompt levels REPLACED one another: a
baseline/steering arm ran under the study's ``manifest.systemPrompt`` (the
deployment frame), while a variant/agent arm ran under the agent artifact's
``systemPrompt`` with the frame simply not applied. The frame was therefore
part of the contrast rather than held constant, and ``promote`` made it worse
by copying the frame into every newborn agent — so a promoted agent carried
one study's framing into the next.

What is pinned here:

1. **Composition, persona first**, with all four degradation cases, and the
   empty-persona case byte-identical to the historical frame-only value
   (the regression lock: every existing run's stamped ``systemPromptHash``
   must keep its meaning).
2. **``promote`` no longer inherits** — a newborn agent's identity is bare.
3. **A comparability advisory** that fires only when arms diverge, names them,
   and never refuses.
4. **Battery isolation**: a format-2 battery is armed by the AGENT persona
   plus its own declared text, never by the study frame; the baseline arm
   reads it bare; format-1 arming is untouched.
5. **Validation stays frame-free** — the study frame never reaches a held-out
   read; the sanctioned channel is the pinned
   ``extractionRendering.systemPrompt``.
6. **Additive stamps** on both record kinds.
7. **Panel casting composes too** — a cast seat is armed with the agent
   artifact's persona then the cast entry's role text, in that order, through
   the same primitive; its turn records stamp ``{"agent": …, "cast": …}``.

Swift twin: ``Tests/ExperimentKitTests/SystemPromptCompositionTests.swift``.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import battery as battery_mod
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import logprob as logprob_mod
from steerlab_server.experiment import model_variant, promote, tasks
from steerlab_server.experiment import system_prompt as sp
from steerlab_server.steering.vector_store import ConceptVectors

FRAME = "You are a federal district judge. Respond in JSON."
PERSONA = "You are Adjudicator-7, cautious and terse."


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- 1. the composition rule ------------------------------------------------

def test_the_agent_persona_comes_first_joined_by_one_blank_line():
    """Identity precedes instruction: the frame's "answer in this format" is
    an instruction TO whoever the model is being, so the persona is the first
    thing in the system turn."""
    assert sp.compose(PERSONA, FRAME) == PERSONA + "\n\n" + FRAME
    assert sp.JOINER == "\n\n"


@pytest.mark.parametrize("agent,frame,expected", [
    (PERSONA, FRAME, PERSONA + "\n\n" + FRAME),  # both
    (None, FRAME, FRAME),                        # empty agent → frame alone
    ("", FRAME, FRAME),
    ("   ", FRAME, FRAME),
    (PERSONA, None, PERSONA),                    # empty frame → persona alone
    (PERSONA, "", PERSONA),
    (PERSONA, "  \n ", PERSONA),
    (None, None, None),                          # both empty → none
    ("", "", ""),
])
def test_composition_degrades_gracefully_in_every_direction(agent, frame,
                                                            expected):
    assert sp.compose(agent, frame) == expected


def test_the_surviving_side_is_returned_unchanged_never_reconstructed():
    """Byte-identity, not equality: the empty-persona case must return the
    frame OBJECT, so no whitespace or normalization can creep in between the
    manifest and the renderer."""
    frame = "  a frame with edges  "
    assert sp.compose(None, frame) is frame
    assert sp.compose("", frame) is frame
    persona = "  a persona with edges  "
    assert sp.compose(persona, "") is persona


def test_empty_persona_plus_frame_stamps_exactly_todays_frame_only_value():
    """THE regression lock (hard constraint of the ruling).

    Every real run in the workspace today has an empty persona on every arm.
    Its records' ``systemPromptHash`` is ``sha256(frame)``. Composition must
    not move that by a single byte, or every historical run's stamp starts
    describing a prompt that was never rendered.
    """
    todays_value = FRAME               # what the arm rendered before the rule
    todays_hash = _sha(FRAME)          # what its records stamped
    for empty in (None, "", "   "):
        assert sp.compose(empty, FRAME) == todays_value
        assert sp.text_hash(sp.compose(empty, FRAME)) == todays_hash
    # …and the hash convention itself is unchanged: empty hashes to None.
    assert sp.text_hash(None) is None and sp.text_hash("") is None
    assert tasks._sha256_text(FRAME) == todays_hash


def test_the_composition_stamp_always_carries_both_keys():
    """Explicit nulls: an ABSENT key would read as "this engine does not stamp
    composition", a different claim from "this level was empty"."""
    assert sp.composition(PERSONA, FRAME) == {
        "agent": _sha(PERSONA), "study": _sha(FRAME)}
    assert sp.composition(None, FRAME) == {"agent": None, "study": _sha(FRAME)}
    assert sp.composition(None, None) == {"agent": None, "study": None}
    # A battery record's second term is the BATTERY's declared arming, and the
    # spelling difference is the point.
    assert set(sp.composition(None, None, frame_key="battery")) == {
        "agent", "battery"}


def test_a_level_the_composition_drops_stamps_null_in_all_three_shapes():
    """Review 2026-08-26. ``compose`` treats whitespace-only text as empty, but
    the stamp used to hash by TRUTHINESS — so a persona of ``"   "``
    contributed nothing to the effective prompt and still stamped a digest,
    claiming a contribution the bytes do not contain."""
    blank = "   \n\t "
    # The effective prompt is the frame's EXACT bytes…
    assert sp.compose(blank, FRAME) is FRAME
    # …so the agent level stamps null, in every shape the stamp has.
    for key in ("study", "battery", "cast"):
        assert sp.composition(blank, FRAME, frame_key=key) == {
            "agent": None, key: _sha(FRAME)}
        # …and symmetrically, for the second level.
        assert sp.composition(PERSONA, blank, frame_key=key) == {
            "agent": _sha(PERSONA), key: None}
    assert sp.compose(PERSONA, blank) is PERSONA
    # The per-condition hash convention is UNTOUCHED — one convention, shared
    # with ``tasks._sha256_text``, and the composition rule lives at the stamp.
    assert sp.text_hash(blank) == _sha(blank)
    assert tasks._sha256_text(blank) == _sha(blank)


# --- 2. promote no longer inherits ------------------------------------------

def _sweepable_study(root, name, *, frame):
    """The smallest workspace `sweep` + `promote` will actually run on."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"),
           '{"text": "I feel dread"}\n{"text": "terror grips me"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"),
           '{"text": "calm morning"}\n{"text": "a quiet walk"}\n')
    _write(os.path.join(concept_dir, "markers.json"),
           json.dumps({"words": ["dread"]}))
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
           '{"text": "Write about the town."}\n')
    _write(os.path.join(root, "prompts", "batteries", "b.jsonl"),
           '{"prompt": "What is 1+1?", "answer": "2"}\n')
    raw = es.load_raw(name, root)
    raw["systemPrompt"] = frame
    raw["sweep"] = {"layerFractions": [0.5], "alphas": [0.4],
                    "devPromptsFile": "prompts/dev/dev.jsonl",
                    "batteryFile": "prompts/batteries/b.jsonl",
                    "maxTokens": 16}
    es.save_raw(raw, root)
    return name


def test_promote_gives_a_newborn_agent_no_system_prompt_at_all(tmp_path,
                                                               monkeypatch):
    """Pre-fix this asserted ``manifest.systemPrompt``: the study's deployment
    frame was copied onto the agent and travelled with it. Under composition
    that copy would be concatenated with the NEXT study's frame — the same
    text twice in one system turn."""
    root = str(tmp_path)
    name = _sweepable_study(root, "promo", frame=FRAME)

    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False, **_):
        return ("dread filled the quiet town before dawn broke 2" if injections
                else "the town woke slowly to a bright morning 2")

    # The persisted sidecar must satisfy promote's full recipe-identity match,
    # so the faked bundle carries the manifest's own stimulus hash and the
    # canonical norm-source token.
    from steerlab_server.experiment.manifest import Manifest
    stimulus_hash = Manifest.load(name, root).concepts[0].stimulus_set_hash
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {
            "fear": _fake_bundle(stimulus_hash=stimulus_hash,
                                 residual_norm_source="extraction-stimuli")})
    monkeypatch.setattr(tasks, "generate", generate)
    tasks.sweep(name, root, model_provider=_fake_model, log=lambda *_: None)

    artifact = promote.promote(name, "fear", root=root,
                               log=lambda *_: None)["variant"]
    assert artifact["systemPrompt"] is None
    # …and the study's frame is nowhere on the newborn agent.
    assert FRAME not in json.dumps(artifact)
    # The artifact round-trips as persona-free, so a later study composing it
    # gets that study's frame alone.
    reloaded = model_variant.ModelVariant.from_dict(artifact)
    assert sp.compose(reloaded.system_prompt, "another frame") == "another frame"


# --- 3. the comparability advisory ------------------------------------------

def test_identical_arms_emit_nothing():
    """The universal current case — every arm shares one effective prompt —
    must stay silent, or the advisory is noise nobody reads."""
    assert sp.divergence_advisory(
        [("baseline", FRAME), ("fear-a1", FRAME), ("agent-x", FRAME)]) is None
    assert sp.divergence_advisory([("baseline", None), ("v", None)]) is None
    assert sp.divergence_advisory([("baseline", FRAME)]) is None


def test_divergent_arms_are_named_with_their_hashes():
    composed = sp.compose(PERSONA, FRAME)
    advisory = sp.divergence_advisory(
        [("baseline", FRAME), ("adjudicator", composed)])
    assert advisory is not None
    assert "baseline (" + _sha(FRAME)[:12] in advisory
    assert "adjudicator (" + _sha(composed)[:12] in advisory
    assert "Not a refusal." in advisory


def test_an_arm_with_no_system_content_is_named_none_not_crashed():
    advisory = sp.divergence_advisory([("bare", None), ("framed", FRAME)])
    assert "bare (none)" in advisory


# --- shared run fixtures ----------------------------------------------------

V2_ITEMS = [
    {"id": "cap-fr", "prompt": "What is the capital of France?",
     "answer": "Paris", "options": ["Paris", "Lyon", "Nice"]},
]


def _v2_lines(header=None):
    rows = [dict({"batteryFormat": 2, "scoring": "choiceProbability"},
                 **(header or {}))]
    rows += [dict(i) for i in V2_ITEMS]
    return "".join(json.dumps(r) + "\n" for r in rows)


LEGACY_LINES = ('{"prompt": "What is the capital of France?", '
                '"answer": "Paris", "grading": "token_exact"}\n')


def _agent_artifact(system_prompt):
    artifact = {"name": "agent-x", "baseModelID": "org/m",
                "promptMode": "chatAssistant", "temperature": 0.0,
                "alphaInNormUnits": False, "injections": [], "adapters": []}
    if system_prompt is not None:
        artifact["systemPrompt"] = system_prompt
    return artifact


def _fake_bundle(stimulus_hash="h", residual_norm_source="test"):
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source=residual_norm_source,
        stimulus_hash=stimulus_hash)


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _variant_study(root, name, *, frame, persona, battery_lines=None):
    """Implicit baseline + one agent arm, one item, greedy."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 16
    raw["systemPrompt"] = frame
    raw["variantConditions"] = [{
        "name": "agent-x", "artifactPath": "runs/variants/v.json",
        "artifactHash": "aa" * 32, "artifact": _agent_artifact(persona)}]
    if battery_lines is not None:
        rel = f"prompts/batteries/{name}.jsonl"
        raw["capabilityBatteryFile"] = rel
        raw["capabilityBatteryHash"] = _write(os.path.join(root, rel),
                                              battery_lines)
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path, '{"id": "p0", "prompt": "Decide the case."}\n')
    _write(os.path.join(root, "runs", "x", "fear.json"), "{}")
    _write(os.path.join(root, "runs", "x", "fear.safetensors"), "")
    return prompts_path


def _patch_engine(monkeypatch, seen):
    """Records every system prompt each boundary is handed, tagged by which
    boundary saw it — that is the whole measurement in these tests."""
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False, **_):
        seen.append(("generate", prompt, system_prompt))
        return "an answer"

    def score_options(model, prompt, options, **kwargs):
        seen.append(("choice", prompt, kwargs.get("system_prompt")))
        correct = "Paris"
        return SimpleNamespace(
            selected=correct,
            probability={o: (0.8 if o == correct else 0.1) for o in options})

    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", generate)
    monkeypatch.setattr(logprob_mod, "score_options", score_options)


def _records(run_dir, filename="generations.jsonl"):
    return [json.loads(line) for line
            in open(os.path.join(run_dir, filename)) if line.strip()]


def _study_generations(run_dir):
    return [r for r in _records(run_dir) if r.get("promptID") == "p0"]


# --- 4. composition reaches the run, and is stamped -------------------------

def test_an_agent_arm_generates_under_persona_then_frame(tmp_path, monkeypatch):
    """Pre-fix, the agent arm's generation was handed PERSONA alone and the
    study frame was silently dropped."""
    root = str(tmp_path)
    seen = []
    _patch_engine(monkeypatch, seen)
    run_dir = tasks.run("compose", _variant_study(root, "compose", frame=FRAME,
                                                  persona=PERSONA),
                        root, model_provider=_fake_model, log=lambda *_: None)
    task_calls = [s for kind, prompt, s in seen
                  if kind == "generate" and "Decide" in prompt]
    assert PERSONA + "\n\n" + FRAME in task_calls  # the agent arm
    assert FRAME in task_calls                     # the baseline arm

    by_condition = {r["condition"]: r for r in _study_generations(run_dir)}
    assert by_condition["agent-x"]["systemPromptHash"] == \
        _sha(PERSONA + "\n\n" + FRAME)
    assert by_condition["baseline"]["systemPromptHash"] == _sha(FRAME)


def test_every_record_stamps_which_levels_composed_its_prompt(tmp_path,
                                                              monkeypatch):
    root = str(tmp_path)
    _patch_engine(monkeypatch, [])
    run_dir = tasks.run("stamp", _variant_study(root, "stamp", frame=FRAME,
                                                persona=PERSONA),
                        root, model_provider=_fake_model, log=lambda *_: None)
    by_condition = {r["condition"]: r for r in _study_generations(run_dir)}
    assert by_condition["agent-x"]["systemPromptComposition"] == {
        "agent": _sha(PERSONA), "study": _sha(FRAME)}
    # Additive, and explicit about the level that contributed nothing.
    assert by_condition["baseline"]["systemPromptComposition"] == {
        "agent": None, "study": _sha(FRAME)}


def test_an_empty_persona_run_is_byte_identical_to_the_historical_arming(
        tmp_path, monkeypatch):
    """The agent arms in the workspace today all have an empty persona. Their
    generations and their stamps must be exactly what they were."""
    root = str(tmp_path)
    seen = []
    _patch_engine(monkeypatch, seen)
    run_dir = tasks.run("bare", _variant_study(root, "bare", frame=FRAME,
                                               persona=None),
                        root, model_provider=_fake_model, log=lambda *_: None)
    task_calls = {s for kind, prompt, s in seen
                  if kind == "generate" and "Decide" in prompt}
    assert task_calls == {FRAME}
    for record in _study_generations(run_dir):
        assert record["systemPromptHash"] == _sha(FRAME)
        assert record["systemPromptComposition"]["agent"] is None


def test_the_run_advisory_fires_only_on_divergence_and_names_the_arms(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    _patch_engine(monkeypatch, [])
    divergent = tasks.run(
        "adv", _variant_study(root, "adv", frame=FRAME, persona=PERSONA),
        root, model_provider=_fake_model, log=lambda *_: None)
    text = open(os.path.join(divergent, "advisories.txt"),
                encoding="utf-8").read()
    assert "DIFFERENT effective system prompts" in text
    assert "baseline (" in text and "agent-x (" in text
    assert "Not a refusal." in text

    quiet = tasks.run(
        "quiet", _variant_study(root, "quiet", frame=FRAME, persona=None),
        root, model_provider=_fake_model, log=lambda *_: None)
    # advisories.txt has other owners (cross-substrate, lock drift), so the
    # claim is that THIS line is absent — not that the file is.
    quiet_path = os.path.join(quiet, "advisories.txt")
    quiet_text = (open(quiet_path, encoding="utf-8").read()
                  if os.path.exists(quiet_path) else "")
    assert "DIFFERENT effective system prompts" not in quiet_text


# --- 5. battery isolation ---------------------------------------------------

def test_a_format_two_battery_sees_the_persona_and_never_the_study_frame(
        tmp_path, monkeypatch):
    """The battery is a CAPABILITY control: the agent is the model under test,
    so its identity belongs in the reading — but the study's deployment frame
    is the context of the study's own task and is exactly what produced the
    0.45-vs-1.00 split this format exists to end."""
    root = str(tmp_path)
    seen = []
    _patch_engine(monkeypatch, seen)
    run_dir = tasks.run(
        "batiso", _variant_study(root, "batiso", frame=FRAME, persona=PERSONA,
                                 battery_lines=_v2_lines()),
        root, model_provider=_fake_model, log=lambda *_: None)
    battery_calls = [s for _kind, prompt, s in seen if "capital" in prompt]
    assert battery_calls, "the battery never ran"
    assert all(s is None or FRAME not in s for s in battery_calls)
    # Baseline reads it bare; the agent arm reads it as itself.
    assert set(battery_calls) == {None, PERSONA}

    rows = {r["condition"]: r for r in _records(run_dir, "battery.jsonl")}
    assert rows["baseline"]["armingSystemPrompt"] is False
    assert rows["agent-x"]["armingSystemPrompt"] is True


def test_the_battery_composes_the_persona_ahead_of_its_own_declared_arming(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    seen = []
    _patch_engine(monkeypatch, seen)
    declared = "Answer with the option letter only."
    tasks.run(
        "batdecl",
        _variant_study(root, "batdecl", frame=FRAME, persona=PERSONA,
                       battery_lines=_v2_lines({"systemPrompt": declared})),
        root, model_provider=_fake_model, log=lambda *_: None)
    battery_calls = set(s for _k, prompt, s in seen if "capital" in prompt)
    assert battery_calls == {declared, PERSONA + "\n\n" + declared}


def test_battery_rows_stamp_their_arming_composition_additively(tmp_path,
                                                                monkeypatch):
    root = str(tmp_path)
    _patch_engine(monkeypatch, [])
    declared = "Answer plainly."
    run_dir = tasks.run(
        "batstamp",
        _variant_study(root, "batstamp", frame=FRAME, persona=PERSONA,
                       battery_lines=_v2_lines({"systemPrompt": declared})),
        root, model_provider=_fake_model, log=lambda *_: None)
    rows = {r["condition"]: r for r in _records(run_dir, "battery.jsonl")}
    assert rows["agent-x"]["armingSystemPromptComposition"] == {
        "agent": _sha(PERSONA), "battery": _sha(declared)}
    assert rows["agent-x"]["armingSystemPromptHash"] == \
        _sha(PERSONA + "\n\n" + declared)
    assert rows["baseline"]["armingSystemPromptComposition"] == {
        "agent": None, "battery": _sha(declared)}
    assert rows["baseline"]["armingSystemPromptHash"] == _sha(declared)
    # The historical four arming keys are still there — this is ADDITIVE.
    assert {"armingIsolated", "armingPromptMode", "armingSystemPrompt",
            "armingMaxTokens"} <= set(rows["baseline"])


def test_a_format_one_battery_keeps_its_pinned_meaning_and_its_key_set(
        tmp_path, monkeypatch):
    """A legacy battery's arming IS the surrounding instrument's, defect
    included, and its rows carry the exact six-plus-three keys they always
    did. An agent persona must not reach it through the new channel."""
    root = str(tmp_path)
    seen = []
    _patch_engine(monkeypatch, seen)
    run_dir = tasks.run(
        "batlegacy",
        _variant_study(root, "batlegacy", frame=FRAME, persona=PERSONA,
                       battery_lines=LEGACY_LINES),
        root, model_provider=_fake_model, log=lambda *_: None)
    battery_calls = set(s for _k, prompt, s in seen if "capital" in prompt)
    # baseline: the study frame (the historical contamination, preserved);
    # agent arm: the ARTIFACT's own prompt, replacement-style, as before.
    assert battery_calls == {FRAME, PERSONA}
    for row in _records(run_dir, "battery.jsonl"):
        assert set(row) == {"condition", "promptIndex", "promptID",
                            "sampleIndex", "prompt", "answer", "output",
                            "batteryHash", "correct"}


def test_resolve_arming_ignores_the_agent_channel_for_format_one(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/legacy.jsonl"
    _write(os.path.join(root, rel), LEGACY_LINES)
    spec = battery_mod.load_spec(rel, root)
    arming = battery_mod.resolve_arming(
        spec, prompt_mode="chatAssistant", system_prompt=FRAME,
        agent_system_prompt=PERSONA)
    assert arming.system_prompt == FRAME
    assert arming.agent_system_prompt is None
    assert arming.declared_system_prompt is None
    assert set(arming.as_record_fields()) >= {"armingIsolated",
                                              "armingSystemPromptHash"}


# --- 6. validation stays frame-free -----------------------------------------

def _validation_study(root, name, *, frame, rendering=None):
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm"}\n')
    _write(os.path.join(concept_dir, "validation.jsonl"),
           '{"text": "the shadows closed in", "expresses": true}\n'
           '{"text": "a bright clear noon", "expresses": false}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw["systemPrompt"] = frame
    if rendering is not None:
        for concept in raw["concepts"]:
            concept.setdefault("options", {})["extractionRendering"] = rendering
    es.save_raw(raw, root)
    return name


def _capture_validation_renderings(monkeypatch):
    from steerlab_server.steering import extractor

    captured = []

    def activations(model, texts, position, rendering=None, *a, **k):
        captured.append(rendering)
        return SimpleNamespace(values=[[[1.0, 0.0]] * 4 for _ in texts])

    monkeypatch.setattr(extractor, "activations", activations)
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    return captured


def test_the_study_frame_never_reaches_a_held_out_validation_read(
        tmp_path, monkeypatch):
    """A probe score is a projection onto the extracted direction. If a
    run-time deployment frame changed the rendering, the held-out accuracy
    would move with a deployment choice rather than with the vector."""
    root = str(tmp_path)
    captured = _capture_validation_renderings(monkeypatch)
    name = _validation_study(root, "valfree", frame=FRAME)
    tasks.validate(name, root, model_provider=_fake_model, log=lambda *_: None)
    assert captured, "validation never read any activations"
    for rendering in captured:
        assert getattr(rendering, "system_prompt", None) != FRAME
        assert not (getattr(rendering, "system_prompt", None) or "")


def test_the_pinned_extraction_rendering_is_the_sanctioned_prompt_channel(
        tmp_path, monkeypatch):
    """…and it DOES reach the read: it is part of recipe identity, so
    extraction and validation cannot silently disagree about it."""
    root = str(tmp_path)
    captured = _capture_validation_renderings(monkeypatch)
    pinned = "You are describing a scene."
    name = _validation_study(
        root, "valpinned", frame=FRAME,
        rendering={"mode": "chatTemplate", "systemPrompt": pinned})
    tasks.validate(name, root, model_provider=_fake_model, log=lambda *_: None)
    assert captured
    assert all(r.system_prompt == pinned for r in captured)
    assert all(FRAME not in (r.system_prompt or "") for r in captured)


# --- 7. panel casting -------------------------------------------------------
#
# The same ruling, one level down. A panel seat's two levels are the CAST
# ENTRY's role text ("you represent Team South", kept on the seat by
# `PanelComposition.semanticForm` because the role is the experiment) and the
# persona on the AGENT ARTIFACT cast into it. They used to REPLACE, exactly as
# the study levels did. Same order, same primitive, second term spelled `cast`.

ROLE = "You represent Team South. Argue its position."


def _seat(tmp_path, *, persona, cast, name="seat"):
    """A panel seat cast with an agent artifact carrying ``persona``."""
    from steerlab_server.experiment import multi_agent

    path = None
    if persona is not None or name == "carries-artifact":
        artifact = model_variant.ModelVariant(
            name="adjudicator", base_model_id="m", system_prompt=persona)
        path = os.path.join(str(tmp_path), f"{name}.json")
        _write(path, json.dumps(artifact.to_dict()))
    return multi_agent.Agent(
        id="a", name="Alice", base_model_id="m", system_prompt=cast,
        variant_artifact_path=path)


def _resolved(tmp_path, *, persona, cast, name="seat"):
    """(effective system prompt, composition stamp) for one cast seat."""
    from steerlab_server.experiment import multi_agent

    settings = multi_agent._runtime_settings(
        _seat(tmp_path, persona=persona, cast=cast, name=name), False)
    return settings[3], settings[7]


def test_a_cast_seat_is_armed_with_the_persona_then_the_role(tmp_path):
    """Persona first, role second, one blank line — the SAME order as the
    study rule, because a cast role is situational instruction TO whoever the
    agent is, exactly as the study frame is."""
    system, _stamp = _resolved(tmp_path, persona=PERSONA, cast=ROLE)
    assert system == PERSONA + "\n\n" + ROLE


@pytest.mark.parametrize("persona,cast,expected", [
    (PERSONA, ROLE, PERSONA + "\n\n" + ROLE),
    (None, ROLE, ROLE),          # today's dominant case
    ("", ROLE, ROLE),
    ("   ", ROLE, ROLE),
    (PERSONA, "", PERSONA),
    (PERSONA, "   ", PERSONA),
    (None, "", ""),
])
def test_panel_casting_degrades_gracefully_in_every_direction(
        tmp_path, persona, cast, expected):
    system, _stamp = _resolved(tmp_path, persona=persona, cast=cast,
                               name="carries-artifact")
    assert system == expected


def test_an_empty_persona_seat_renders_exactly_todays_cast_only_arming(
        tmp_path, monkeypatch):
    """THE regression lock. Every agent in the workspace today has an EMPTY
    persona, so every existing panel must reach the sampler with exactly the
    cast-entry text it always did — not a re-joined copy of it."""
    from steerlab_server.experiment import multi_agent

    for persona in (None, "", "   "):
        system, stamp = _resolved(tmp_path, persona=persona, cast=ROLE,
                                  name="carries-artifact")
        assert system == ROLE
        assert sp.text_hash(system) == _sha(ROLE)
        assert stamp["cast"] == _sha(ROLE)

    seen = []

    def stub_generate(model, prompt, **kwargs):
        seen.append(kwargs.get("system_prompt"))
        return "out"

    monkeypatch.setattr(multi_agent, "generate", stub_generate)
    scenario = multi_agent.Scenario(
        name="panel", base_model_id="m",
        agents=[multi_agent.Agent(id="a", name="Alice", base_model_id="m",
                                  system_prompt=ROLE)],
        turns=[multi_agent.Turn(id="t1", title="Alice opens",
                                speaker_agent_id="a",
                                prompt_template="Speak.", output_label="a1")])
    run_dir = str(tmp_path / "run1")
    os.makedirs(run_dir, exist_ok=True)
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             scenario, run_dir=run_dir)
    assert seen == [ROLE]


def test_a_padded_cast_entry_is_no_longer_trimmed_on_this_engine(tmp_path):
    """Pre-composition the server `.strip()`ped the cast text where the Swift
    twin never did — the same panel was armed with different bytes on the two
    engines. The shared primitive never trims a non-empty value, which settles
    it in Swift's favour: what the researcher wrote is what the seat runs."""
    system, _stamp = _resolved(tmp_path, persona=None, cast="  padded role  ",
                               name="carries-artifact")
    assert system == "  padded role  "


def test_a_cast_seat_stamps_which_levels_composed_its_prompt(tmp_path,
                                                             monkeypatch):
    """Additive provenance beside the effective text, always both keys, with
    explicit nulls — the panel spelling of a study record's
    `systemPromptComposition`."""
    from steerlab_server.experiment import multi_agent

    _system, stamp = _resolved(tmp_path, persona=PERSONA, cast=ROLE)
    assert stamp == {"agent": _sha(PERSONA), "cast": _sha(ROLE)}
    bare, bare_stamp = _resolved(tmp_path, persona=None, cast=ROLE,
                                 name="carries-artifact")
    assert bare_stamp == {"agent": None, "cast": _sha(ROLE)}

    monkeypatch.setattr(multi_agent, "generate",
                        lambda model, prompt, **kwargs: "out")
    scenario = multi_agent.Scenario(
        name="panel", base_model_id="m",
        agents=[multi_agent.Agent(id="a", name="Alice", base_model_id="m",
                                  system_prompt=ROLE)],
        turns=[multi_agent.Turn(id="t1", title="Alice opens",
                                speaker_agent_id="a",
                                prompt_template="Speak.", output_label="a1")])
    run_dir = str(tmp_path / "run2")
    os.makedirs(run_dir, exist_ok=True)
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             scenario, run_dir=run_dir)
    turns = [json.loads(line) for line
             in open(os.path.join(run_dir, "turns.jsonl"), encoding="utf-8")]
    assert [t["systemPromptComposition"] for t in turns] == [
        {"agent": None, "cast": _sha(ROLE)}]


# --- 8. cross-engine fixture ------------------------------------------------

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine")


def test_the_cross_engine_composition_fixture_is_current():
    """The Swift suite consumes these bytes; a Python rule change that is not
    regenerated would leave both suites green over a broken seam."""
    path = os.path.join(FIXTURES, "system-prompt-composition.json")
    if not os.path.exists(path):
        pytest.skip("fixture not generated yet")
    with open(path, encoding="utf-8") as handle:
        fixture = json.load(handle)
    for case in fixture["composition"]:
        assert sp.compose(case["agent"], case["study"]) == case["effective"], \
            case["label"]
        assert sp.text_hash(sp.compose(case["agent"], case["study"])) == \
            case["effectiveHash"], case["label"]
        assert sp.composition(case["agent"], case["study"]) == \
            case["studyStamp"], case["label"]
        assert sp.composition(case["agent"], case["study"],
                              frame_key="battery") == case["batteryStamp"], \
            case["label"]
    for case in fixture["panelCasting"]:
        composed = sp.compose(case["agent"], case["cast"])
        assert composed == case["effective"], case["label"]
        assert sp.text_hash(composed) == case["effectiveHash"], case["label"]
        assert sp.composition(case["agent"], case["cast"],
                              frame_key="cast") == case["stamp"], case["label"]
    for case in fixture["advisories"]:
        arms = [(a["name"], a["systemPrompt"]) for a in case["arms"]]
        assert sp.divergence_advisory(arms) == case["advisory"], case["label"]

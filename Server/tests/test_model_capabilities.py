"""Chat-template capabilities derived from the pinned template (2026-09-05).

The family rules the renderers branched on were substring tests on the model
id, and two were false: Qwen3-14B's template reads ``enable_thinking`` but
ignores ``reasoning_effort``, so a study declaring ``medium`` ran at the
template's default while its frozen manifest asserted medium. These tests pin,
on this engine:

1. the PROBE — each verdict is a render comparison, over the synthetic
   template families the cross-engine fixture carries, plus the real cached
   tokenizers when this machine holds them;
2. the RECORD — its canonical hash, its file, lookup by (model, revision),
   the loud re-probe on a template change, the human override;
3. the CONSUMERS — the renderers pass only what the template reads, the
   declaration gates refuse an ignored/rejected level and an undeliverable
   system prompt, ``verify`` gates a draft and advises on a frozen study,
   runs stamp the record, freeze prints it;
4. the FALLBACK — no tokenizer means the id heuristic, as a record that says
   so, behaving exactly as the pre-record engine did.

Swift twin: ``Tests/SteeringKitTests/ModelCapabilitiesTests.swift`` over the
same fixture.
"""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import model_capabilities as mc
from steerlab_server.experiment import prompt_render
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import extraction_rendering as er

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine")
REGENERATE = ("stale fixture — re-run `Server/.venv.nosync/bin/python "
              "scripts/regenerate-cross-engine-fixtures.py` and commit")

QWEN = "Qwen/Qwen3-14B"
QWEN38 = "Qwen/Qwen3.8-27B"
GEMMA = "google/gemma-3-27b-it"


@pytest.fixture(scope="module")
def fixture():
    with open(os.path.join(FIXTURES, "model-capabilities.json"), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def schema():
    with open(os.path.join(FIXTURES, "model-capabilities.schema.json"),
              encoding="utf-8") as handle:
        return json.load(handle)


def _family(fixture, label):
    return next(f for f in fixture["families"] if f["label"] == label)


def _probe_family(family):
    ids = dict(family["thinkTokenIDs"])
    return mc.probe(
        mc.template_renderer(family["template"], family["specialTokens"]),
        model_id="fixture/" + family["label"], revision="0" * 40,
        think_token_id=lambda token: ids.get(token),
        architecture={"layerCount": 4, "hiddenSize": 8,
                      "layerTypes": ["full_attention"] * 4},
        template_sha256=mc.sha256_text(family["template"]),
        engine="python-hf-transformers", engine_version="fixture",
        probed_at="2026-09-05T00:00:00Z")


@pytest.fixture(autouse=True)
def _fresh_memo():
    mc.forget_memo()
    yield
    mc.forget_memo()


# =============================================================================
# 1. The probe
# =============================================================================


def test_the_probe_reproduces_every_fixture_family(fixture):
    for family in fixture["families"]:
        record = _probe_family(family)
        assert record["detected"] == family["detected"], \
            f"{family['label']}: {REGENERATE}"


def test_a_chatml_template_with_effort_control_is_read_as_such(fixture):
    detected = _family(fixture, "chatml-effort")["detected"]
    assert detected["systemRole"] == mc.SYSTEM_TURN
    assert detected["thinkingSwitch"] == mc.SUPPORTED
    # The opening tag is in the prompt: decoded output will not repeat it.
    assert detected["thinkOpenInPrompt"] is True
    assert detected["effortVariableRead"] is True
    # `xhigh` is the template's DEFAULT and still reads as accepted — the
    # bogus-value probe settles "is the variable read" before any candidate
    # is compared, so a default-equal render is not misfiled as ignored.
    assert detected["effortLevels"] == {
        "low": mc.ACCEPTED, "medium": mc.ACCEPTED,
        "high": mc.REJECTED, "xhigh": mc.ACCEPTED}
    assert detected["thinkTokens"] == {"open": 11, "close": 12}


def test_a_chatml_template_with_only_the_switch_ignores_every_level(fixture):
    detected = _family(fixture, "chatml-switch")["detected"]
    assert detected["thinkingSwitch"] == mc.SUPPORTED
    assert detected["thinkOpenInPrompt"] is False
    assert detected["effortVariableRead"] is False
    assert set(detected["effortLevels"].values()) == {mc.IGNORED}


def test_a_folding_template_records_the_separator(fixture):
    detected = _family(fixture, "gemma-fold")["detected"]
    assert detected["systemRole"] == mc.FOLDED_INTO_USER
    assert detected["foldSeparator"] == "\n\n"
    assert detected["thinkingSwitch"] == mc.UNSUPPORTED
    assert detected["effortLevels"] is None
    assert detected["thinkTokens"] == {"open": None, "close": None}


def test_a_template_that_raises_on_a_system_turn_is_unsupported(fixture):
    detected = _family(fixture, "system-refused")["detected"]
    assert detected["systemRole"] == mc.UNSUPPORTED
    assert "raises" in detected["systemRoleDetail"]


def test_a_template_that_drops_the_system_turn_is_unsupported_too():
    template = ("{%- for m in messages if m.role != 'system' %}"
                "<{{ m.role }}>{{ m.content }}</{{ m.role }}>{%- endfor %}"
                "{%- if add_generation_prompt %}<assistant>{%- endif %}")
    record = mc.probe(mc.template_renderer(template), model_id="x/drops")
    assert record["detected"]["systemRole"] == mc.UNSUPPORTED
    assert "drops" in record["detected"]["systemRoleDetail"]


def test_a_template_that_refuses_a_plain_user_turn_is_a_record_error():
    template = "{{ raise_exception('no') }}"
    with pytest.raises(mc.RecordError):
        mc.probe(mc.template_renderer(template), model_id="x/broken")


# --- the real cached tokenizers (skipped when absent) --------------------------

def _real_tokenizer(model_id):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not cached / offline
        pytest.skip(f"{model_id} tokenizer unavailable: {exc}")


def test_the_real_qwen3_template_has_the_switch_and_ignores_the_effort():
    tokenizer = _real_tokenizer("Qwen/Qwen3-0.6B")
    record = mc.probe_tokenizer(tokenizer, model_id="Qwen/Qwen3-0.6B")
    detected = record["detected"]
    assert detected["systemRole"] == mc.SYSTEM_TURN
    assert detected["thinkingSwitch"] == mc.SUPPORTED
    assert detected["thinkOpenInPrompt"] is False
    assert detected["effortVariableRead"] is False
    assert set(detected["effortLevels"].values()) == {mc.IGNORED}
    assert detected["thinkTokens"]["close"] == 151668
    assert detected["thinkTokens"]["open"] == 151667
    assert record["template"]["sha256"] and record["template"]["tokenizerConfigSha256"]


def test_the_real_gemma3_template_folds_with_two_newlines():
    model_id = "mlx-community/gemma-3-4b-it-4bit"
    tokenizer = _real_tokenizer(model_id)
    record = mc.probe_tokenizer(tokenizer, model_id=model_id)
    detected = record["detected"]
    assert detected["systemRole"] == mc.FOLDED_INTO_USER
    assert detected["foldSeparator"] == "\n\n"
    assert detected["thinkingSwitch"] == mc.UNSUPPORTED
    assert detected["thinkTokens"] == {"open": None, "close": None}


# =============================================================================
# 2. The record
# =============================================================================


def _check_schema(value, node, path="record"):
    """A deliberately small JSON-Schema walker (type, enum, const, required,
    properties, additionalProperties, items, $ref into $defs) — enough to hold
    the fixture records to the committed schema without a schema library."""
    root = _check_schema.root
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        return _check_schema(value, root["$defs"][name], path)
    if "const" in node:
        assert value == node["const"], f"{path}: {value!r} != {node['const']!r}"
    if "enum" in node:
        assert value in node["enum"], f"{path}: {value!r} not in {node['enum']}"
    expected = node.get("type")
    if expected is not None:
        kinds = {"object": dict, "array": list, "string": str, "integer": int,
                 "boolean": bool, "null": type(None)}
        allowed = tuple(kinds[t] for t in ([expected] if isinstance(expected, str)
                                          else expected))
        assert isinstance(value, allowed) and not (
            isinstance(value, bool) and bool not in allowed), \
            f"{path}: {type(value).__name__} is not {expected}"
    if isinstance(value, dict) and "properties" in node:
        for key in node.get("required", []):
            assert key in value, f"{path}: missing {key}"
        for key, child in value.items():
            if key in node["properties"]:
                _check_schema(child, node["properties"][key], f"{path}.{key}")
            else:
                assert node.get("additionalProperties", True), f"{path}: extra {key}"
    if isinstance(value, list) and "items" in node:
        for index, child in enumerate(value):
            _check_schema(child, node["items"], f"{path}[{index}]")


def test_every_fixture_record_matches_the_pinned_schema(fixture, schema):
    _check_schema.root = schema
    for entry in fixture["records"] + fixture["heuristics"]:
        _check_schema(entry["record"], schema)
        assert not mc.validate_record(entry["record"])


def test_the_record_hash_is_the_canonical_json_of_everything_else(fixture):
    for entry in fixture["records"] + fixture["heuristics"]:
        assert entry["recordHash"] == mc.record_hash(entry["record"]), REGENERATE
        assert entry["record"]["recordHash"] == entry["recordHash"]
    # The canonical form: byte-sorted keys, compact, raw UTF-8.
    assert mc.canonical_json({"b": 1, "a": {"é": [True, None]}}) == \
        '{"a":{"é":[true,null]},"b":1}'


def test_the_record_file_name_and_round_trip(fixture, tmp_path):
    for case in fixture["recordFilenames"]:
        assert mc.record_filename(case["modelID"], case["revision"]) == case["filename"]
    record = _probe_family(_family(fixture, "chatml-effort"))
    written = mc.write_record(record, str(tmp_path))
    assert written == os.path.join("prompts", "models",
                                   "fixture--chatml-effort@" + "0" * 40 + ".json")
    path = os.path.join(str(tmp_path), written)
    assert mc.read_record(path)["recordHash"] == record["recordHash"]
    view = mc.lookup("fixture/chatml-effort", "0" * 40, str(tmp_path))
    assert view is not None and view.path == written and view.is_probed
    assert view.accepted_efforts == ["low", "medium", "xhigh"]


def test_lookup_falls_back_to_another_revision_loudly_and_to_nothing(tmp_path):
    root = str(tmp_path)
    assert mc.lookup(QWEN, "a" * 40, root) is None
    record = mc.probe(mc.template_renderer("{{ messages[0].content }}<assistant>"),
                      model_id=QWEN, revision="b" * 40)
    mc.write_record(record, root)
    view = mc.lookup(QWEN, "a" * 40, root)
    assert view is not None and view.revision == "b" * 40
    assert any("using the record for revision" in a for a in view.advisories)
    # And with no revision asked, the single record answers without a note.
    assert not mc.lookup(QWEN, None, root).advisories
    # resolve(): the heuristic when nothing is on disk, saying so.
    heuristic = mc.resolve("google/gemma-3-4b-it", None, root)
    assert heuristic.source == mc.SOURCE_HEURISTIC
    assert heuristic.advisories == (mc.heuristic_advisory("google/gemma-3-4b-it"),)


def test_a_malformed_record_is_refused_by_name(tmp_path):
    record = mc.heuristic(QWEN)
    record["detected"]["systemRole"] = "sideways"
    assert any("systemRole" in p for p in mc.validate_record(record))
    with pytest.raises(mc.RecordError):
        mc.write_record(record, str(tmp_path))
    tampered = mc.heuristic(QWEN)
    tampered["detected"]["thinkingSwitch"] = mc.UNSUPPORTED  # hash now stale
    assert any("recordHash" in p for p in mc.validate_record(tampered))


def test_overrides_apply_to_the_effective_view_and_never_the_detected_block(fixture):
    base = _probe_family(_family(fixture, "chatml-switch"))
    updated = mc.set_override(base, "thinkOpenInPrompt", "true", "always opens")
    assert updated["detected"]["thinkOpenInPrompt"] is False
    assert mc.effective(updated).think_open_in_prompt is True
    assert updated["recordHash"] != base["recordHash"]
    assert "always opens" in mc.effective(updated).summary_lines()[-1]
    cleared = mc.set_override(updated, "thinkOpenInPrompt", "", "")
    assert cleared["overrides"] == {} and cleared["recordHash"] == base["recordHash"]
    with pytest.raises(mc.RecordError):
        mc.set_override(base, "effortLevels", "x", "no")
    with pytest.raises(mc.RecordError):
        mc.set_override(base, "systemRole", "sideways", "no")
    with pytest.raises(mc.RecordError):
        mc.set_override(base, "systemRole", mc.SYSTEM_TURN, "")


def test_the_fixture_summary_lines_gates_and_kwargs_are_current(fixture):
    by_label = {r["label"]: mc.effective(r["record"]) for r in fixture["records"]}
    by_label["heuristic"] = mc.effective(mc.heuristic(QWEN))
    for case in fixture["gates"]:
        assert prompt_render.reasoning_protocol_violations(
            effort=case["effort"], reasoning_max_tokens=64,
            model_id=case["modelID"], capabilities=by_label[case["record"]]
        ) == case["violations"], f"{case['label']}: {REGENERATE}"
    for case in fixture["templateKwargs"]:
        assert prompt_render.thinking_template_kwargs(
            "fixture", case["effort"], by_label[case["record"]]) == case["kwargs"], \
            f"{case['label']}: {REGENERATE}"
    refusals = fixture["refusals"]
    assert refusals["effortIgnored"] == prompt_render.effort_ignored_reason("medium", QWEN)
    assert refusals["heuristicAdvisory"] == mc.heuristic_advisory(QWEN)
    assert refusals["extractionEffortLevelPrefix"] == er.effort_level_problem_prefix()
    assert fixture["reasoningEfforts"] == list(prompt_render.REASONING_EFFORTS)
    assert fixture["effortCandidates"] == list(mc.EFFORT_CANDIDATES)


def test_diff_names_the_facts_that_moved(fixture):
    old = _probe_family(_family(fixture, "chatml-effort"))
    new = _probe_family(_family(fixture, "chatml-switch"))
    lines = mc.diff(old, new)
    assert lines[0].startswith("template sha256:")
    assert any(line.startswith("effortLevels:") for line in lines)
    assert any(line.startswith("thinkOpenInPrompt:") for line in lines)


# =============================================================================
# 3. The consumers
# =============================================================================


class _TemplateTokenizer:
    """A tokenizer double that carries a REAL template (rendered through
    jinja2), so the renderers derive their capabilities from it exactly as
    they do from a transformers tokenizer."""

    def __init__(self, template, special_tokens=None):
        self.chat_template = template
        self._special = special_tokens or {}
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        self.calls.append((messages, kwargs))
        return mc.render_template_text(
            self.chat_template, messages,
            add_generation_prompt=add_generation_prompt,
            special_tokens=self._special, **kwargs)

    def encode(self, text, add_special_tokens=False):
        return [7] if text == "</think>" else [1, 2, 3]

    def __call__(self, text, add_special_tokens=True):
        class _Out:
            input_ids = list(range(len(text.split())))
        return _Out()


def test_rendering_passes_only_the_variables_the_template_reads(fixture):
    switch = _family(fixture, "chatml-switch")
    tokenizer = _TemplateTokenizer(switch["template"])
    # A level on a switch-only template renders as `on`: enable_thinking
    # alone — byte-identical to what the template always produced, since it
    # never read the effort.
    rendered = prompt_render.render(tokenizer, "Decide.", model_id=QWEN,
                                    reasoning_effort="medium")
    assert tokenizer.calls[-1][1] == {"enable_thinking": True}
    assert rendered.text.endswith("<|im_start|>assistant\n")
    prompt_render.render(tokenizer, "Decide.", model_id=QWEN, reasoning_effort="off")
    assert tokenizer.calls[-1][1] == {"enable_thinking": False}
    prompt_render.render(tokenizer, "Decide.", model_id=QWEN, reasoning_effort="on")
    assert tokenizer.calls[-1][1] == {"enable_thinking": True}
    # The legacy boolean's xhigh on the same template: enable_thinking alone.
    prompt_render.render(tokenizer, "Decide.", model_id=QWEN, qwen_thinking_enabled=True)
    assert tokenizer.calls[-1][1] == {"enable_thinking": True}

    effort = _family(fixture, "chatml-effort")
    tokenizer = _TemplateTokenizer(effort["template"])
    rendered = prompt_render.render(tokenizer, "Decide.", model_id=QWEN38,
                                    reasoning_effort="medium")
    assert tokenizer.calls[-1][1] == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert rendered.text.rstrip().endswith("<think>")
    with pytest.raises(prompt_render.EffortUnrenderable):
        prompt_render.render(tokenizer, "Decide.", model_id=QWEN38,
                             reasoning_effort="high")

    gemma = _family(fixture, "gemma-fold")
    tokenizer = _TemplateTokenizer(gemma["template"], gemma["specialTokens"])
    rendered = prompt_render.render(tokenizer, "Decide.", model_id=GEMMA,
                                    system_prompt="Be brief.", reasoning_effort="off")
    assert tokenizer.calls[-1][1] == {}
    assert tokenizer.calls[-1][0] == [{"role": "user", "content": "Be brief.\n\nDecide."}]
    assert rendered.text.startswith("<bos><start_of_turn>user\nBe brief.\n\nDecide.")


def test_a_system_prompt_on_an_unsupported_template_is_refused_at_render(fixture):
    family = _family(fixture, "system-refused")
    tokenizer = _TemplateTokenizer(family["template"], family["specialTokens"])
    with pytest.raises(prompt_render.SystemRoleUnsupported):
        prompt_render.render(tokenizer, "Decide.", model_id="x/refuses",
                             system_prompt="Be brief.")
    with pytest.raises(prompt_render.SystemRoleUnsupported):
        prompt_render.render_messages(
            tokenizer, [{"role": "user", "content": "Q"}], model_id="x/refuses",
            system_prompt="Be brief.")
    # Without a frame it renders.
    assert prompt_render.render(tokenizer, "Decide.", model_id="x/refuses").text


def test_multi_turn_folding_reads_the_recorded_separator(fixture):
    gemma = _family(fixture, "gemma-fold")
    tokenizer = _TemplateTokenizer(gemma["template"], gemma["specialTokens"])
    prompt_render.render_messages(
        tokenizer, [{"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "Q2"}],
        model_id=GEMMA, system_prompt="Frame.")
    assert tokenizer.calls[-1][0][0] == {"role": "user", "content": "Frame.\n\nQ1"}
    # A transcript's OWN system turn folds too, rather than reaching a
    # template that raises on one.
    prompt_render.render_transcript(
        tokenizer, [{"role": "system", "content": "Own."},
                    {"role": "user", "content": "Q"}], model_id=GEMMA)
    assert tokenizer.calls[-1][0] == [{"role": "user", "content": "Own.\n\nQ"}]


def test_a_probed_tokenizer_is_memoized_by_template_hash(fixture):
    switch = _family(fixture, "chatml-switch")
    a = _TemplateTokenizer(switch["template"])
    first = prompt_render.capabilities_for(QWEN, a)
    calls = len(a.calls)
    b = _TemplateTokenizer(switch["template"])
    second = prompt_render.capabilities_for(QWEN, b)
    assert first == second and b.calls == [] and calls > 0


def _draft(tmp_path, model_id=QWEN, record=None):
    root = str(tmp_path)
    es.create("s", model_id=model_id, revision="c" * 40, root=root)
    if record is not None:
        record = dict(record)
        record.update({"modelID": model_id, "revision": "c" * 40})
        record["recordHash"] = mc.record_hash(record)
        mc.write_record(record, root)
    return root


def test_declaring_an_ignored_level_is_refused_with_the_on_repair(fixture, tmp_path):
    root = _draft(tmp_path, QWEN, _probe_family(_family(fixture, "chatml-switch")))
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.set_protocol("s", {"reasoningEffort": "medium",
                              "reasoningMaxTokens": 64}, root=root)
    assert str(caught.value).startswith(prompt_render.effort_ignored_reason("medium", QWEN))
    assert "model capabilities" in caught.value.repair_action
    # `on` — thinking at the template's default — is what such a template can honour.
    document = es.set_protocol("s", {"reasoningEffort": "on",
                                     "reasoningMaxTokens": 64}, root=root)
    assert document["reasoningEffort"] == "on"
    assert not [v for v in Manifest.load("s", root=root).verify(root)
                if "reasoning" in v.lower()]
    assert not es.capability_advisories(document, root)


def test_declaring_a_rejected_or_unprobed_level_is_refused(fixture, tmp_path):
    root = _draft(tmp_path, QWEN38, _probe_family(_family(fixture, "chatml-effort")))
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.set_protocol("s", {"reasoningEffort": "high",
                              "reasoningMaxTokens": 64}, root=root)
    assert "rejected by the chat template" in str(caught.value)
    assert "accepted levels: low, medium, xhigh" in str(caught.value)
    document = es.set_protocol("s", {"reasoningEffort": "medium",
                                     "reasoningMaxTokens": 64}, root=root)
    assert document["reasoningEffort"] == "medium"
    # A heuristic record never assumed `high`.
    plain = _draft(tmp_path / "plain", QWEN)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.set_protocol("s", {"reasoningEffort": "high",
                              "reasoningMaxTokens": 64}, root=plain)
    assert "not known to be accepted" in str(caught.value)


def test_the_heuristic_keeps_todays_declarations_legal_but_advised(tmp_path):
    root = _draft(tmp_path, QWEN)
    document = es.set_protocol("s", {"reasoningEffort": "medium",
                                     "reasoningMaxTokens": 64}, root=root)
    notes = es.capability_advisories(document, root)
    assert notes[0] == prompt_render.effort_assumed_advisory("medium", QWEN)
    assert notes[1] == mc.heuristic_advisory(QWEN)
    # And Gemma under the heuristic is still refused by name — the old rule.
    gemma = _draft(tmp_path / "g", GEMMA)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.set_protocol("s", {"reasoningEffort": "low",
                              "reasoningMaxTokens": 64}, root=gemma)
    assert "no thinking switch" in str(caught.value)


def test_a_system_prompt_on_an_unsupported_template_is_refused_at_declaration(
        fixture, tmp_path):
    root = _draft(tmp_path, "x/refuses", _probe_family(_family(fixture, "system-refused")))
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.set_system_prompt("s", "Be brief.", root=root)
    assert str(caught.value) == prompt_render.system_prompt_unsupported_reason("x/refuses")
    with pytest.raises(es.ExperimentStoreError):
        es.set_protocol("s", {"systemPrompt": "Be brief."}, root=root)
    # rawCompletion prepends the frame and needs no system role.
    es.set_protocol("s", {"promptMode": prompt_render.RAW_COMPLETION}, root=root)
    assert es.set_protocol("s", {"systemPrompt": "Be brief."},
                           root=root)["systemPrompt"] == "Be brief."
    assert prompt_render.system_prompt_delivery(
        "x/refuses", prompt_render.RAW_COMPLETION,
        mc.resolve("x/refuses", "c" * 40, root)) == "promptPrepend"
    assert prompt_render.system_prompt_delivery(
        "x/refuses", prompt_render.CHAT_ASSISTANT,
        mc.resolve("x/refuses", "c" * 40, root)) == "undeliverable"


def test_a_frozen_study_keeps_verifying_and_is_advised_instead(fixture, tmp_path):
    """Hash stability is non-negotiable: a study frozen under `medium` on a
    template a later probe shows ignores the level must keep verifying
    unchanged; `verify` says so beside it, and never in `violations`."""
    root = _draft(tmp_path, QWEN, _probe_family(_family(fixture, "chatml-switch")))
    raw = es.load_raw("s", root)
    raw.update({"reasoningEffort": "medium", "reasoningMaxTokens": 64,
                "status": "frozen", "conditions": [{"name": "c", "slots": []}]})
    es.save_raw(raw, root, freeze_transition=True)
    manifest = Manifest.load("s", root=root)
    assert not [v for v in manifest.verify(root) if "reasoning" in v.lower()]
    notes = manifest.capability_advisories(root)
    assert prompt_render.effort_ignored_reason("medium", QWEN) in notes
    # The same manifest as a DRAFT is a verify violation (the frozen file
    # is read-only to the store, so the flip is a plain file write).
    raw["status"] = "draft"
    with open(es._path("s", root), "w", encoding="utf-8") as handle:
        json.dump(raw, handle)
    assert prompt_render.effort_ignored_reason("medium", QWEN) in \
        Manifest.load("s", root=root).verify(root)


def test_the_extraction_rendering_gate_reads_the_record_too(fixture):
    switch = mc.effective(_probe_family(_family(fixture, "chatml-switch")))
    rendering = er.from_json({"mode": "chatTemplate", "reasoningEffort": "medium"})
    problem = er.thinking_mode_problem(rendering, QWEN, switch)
    assert problem == (er.effort_level_problem_prefix()
                       + prompt_render.effort_ignored_reason("medium", QWEN))
    assert er.thinking_mode_problem(
        er.from_json({"mode": "chatTemplate", "reasoningEffort": "on"}), QWEN, switch) is None
    effort = mc.effective(_probe_family(_family(fixture, "chatml-effort")))
    assert er.thinking_mode_problem(rendering, QWEN38, effort) is None


def test_the_preregistration_prints_the_record(fixture, tmp_path):
    root = _draft(tmp_path, QWEN38, _probe_family(_family(fixture, "chatml-effort")))
    raw = es.load_raw("s", root)
    raw.update({"reasoningEffort": "medium", "reasoningMaxTokens": 64})
    es.save_raw(raw, root)
    es._write_preregistration(raw, root)
    with open(os.path.join(root, "experiments", "s", "preregistration.md"),
              encoding="utf-8") as handle:
        text = handle.read()
    assert "- **Model capabilities:** source probe, record `prompts/models/" in text
    assert "- **Thinking switch:** supported, opening tag in prompt true" in text
    assert "- **Reasoning effort levels:** low accepted, medium accepted, " \
           "high rejected, xhigh accepted" in text
    assert "- **Capability overrides:** none" in text


def test_runs_stamp_the_record_and_advise_on_a_frozen_ignored_level(fixture, tmp_path):
    from steerlab_server.experiment import tasks
    from types import SimpleNamespace
    root = _draft(tmp_path, QWEN)
    raw = es.load_raw("s", root)
    raw.update({"reasoningEffort": "medium", "reasoningMaxTokens": 64,
                "status": "frozen", "conditions": [{"name": "c", "slots": []}]})
    es.save_raw(raw, root, freeze_transition=True)
    manifest = Manifest.load("s", root=root)
    tokenizer = _TemplateTokenizer(_family(fixture, "chatml-switch")["template"])
    model = SimpleNamespace(tokenizer=tokenizer, revision="c" * 40,
                            model=SimpleNamespace(config=None))
    run_directory = os.path.join(root, "runs", "r")
    os.makedirs(run_directory)
    lines: list = []
    tasks._write_config_snapshot(manifest, run_directory, "run", model=model,
                                 root=root, log=lines.append)
    with open(os.path.join(run_directory, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    stamp = config["notes"]["modelCapabilities"]
    assert stamp["source"] == mc.SOURCE_PROBE
    assert stamp["record"] == mc.record_relpath(QWEN, "c" * 40)
    assert stamp["effortLevels"]["medium"] == mc.IGNORED
    assert os.path.isfile(os.path.join(root, stamp["record"]))
    assert stamp["recordHash"] == mc.read_record(os.path.join(root, stamp["record"]))["recordHash"]
    # The run continued, loudly: the ignored level is an advisory line in the
    # log and in advisories.txt, never a refusal.
    ignored = prompt_render.effort_ignored_reason("medium", QWEN)
    assert f"ADVISORY: {ignored}" in lines
    with open(os.path.join(run_directory, "advisories.txt"), encoding="utf-8") as handle:
        assert ignored in handle.read()
    # A second run against the same template reads the record back (no
    # re-probe, no diff); a changed template re-probes and says what moved.
    lines.clear()
    os.makedirs(os.path.join(root, "runs", "r2"))
    tasks._write_config_snapshot(manifest, os.path.join(root, "runs", "r2"), "validate",
                                 model=model, root=root, log=lines.append)
    assert not any("re-probed" in line for line in lines)
    changed = SimpleNamespace(
        tokenizer=_TemplateTokenizer(_family(fixture, "chatml-effort")["template"]),
        revision="c" * 40, model=SimpleNamespace(config=None))
    os.makedirs(os.path.join(root, "runs", "r3"))
    tasks._write_config_snapshot(manifest, os.path.join(root, "runs", "r3"), "validate",
                                 model=changed, root=root, log=lines.append)
    assert any("changed since" in line and "effortLevels" in line for line in lines)


def test_a_run_without_a_probeable_tokenizer_stamps_the_heuristic(tmp_path):
    from steerlab_server.experiment import tasks
    from types import SimpleNamespace
    root = _draft(tmp_path, GEMMA)
    manifest = Manifest.load("s", root=root)

    class _Double:
        def apply_chat_template(self, messages, **kwargs):
            return "x"

    run_directory = os.path.join(root, "runs", "r")
    os.makedirs(run_directory)
    tasks._write_config_snapshot(
        manifest, run_directory, "extract",
        model=SimpleNamespace(tokenizer=_Double(), revision=None, model=None),
        root=root, log=lambda _line: None)
    with open(os.path.join(run_directory, "config.json"), encoding="utf-8") as handle:
        stamp = json.load(handle)["notes"]["modelCapabilities"]
    assert stamp["source"] == mc.SOURCE_HEURISTIC and stamp["record"] is None
    assert stamp["systemRole"] == mc.FOLDED_INTO_USER
    assert not os.path.isdir(os.path.join(root, "prompts", "models"))


# =============================================================================
# 4. The fallback
# =============================================================================


def test_the_heuristic_reproduces_the_old_family_rules(fixture):
    for entry in fixture["heuristics"]:
        assert mc.heuristic(entry["modelID"], None) == entry["record"], REGENERATE
    qwen = mc.effective(mc.heuristic(QWEN))
    assert qwen.has_thinking_switch and qwen.has_system_role
    assert qwen.effort_levels == {"low": mc.ASSUMED, "medium": mc.ASSUMED,
                                  "xhigh": mc.ASSUMED}
    gemma = mc.effective(mc.heuristic(GEMMA))
    assert gemma.system_role == mc.FOLDED_INTO_USER and gemma.fold_separator == "\n\n"
    assert not gemma.has_thinking_switch
    other = mc.effective(mc.heuristic("meta-llama/Llama-3.1-8B-Instruct"))
    assert other.has_system_role and not other.has_thinking_switch
    # And the predicates without a record or a tokenizer are the old answers.
    assert prompt_render.has_thinking_mode(QWEN) and not prompt_render.has_thinking_mode(GEMMA)
    assert prompt_render.has_system_role(QWEN) and not prompt_render.has_system_role(GEMMA)
    assert prompt_render.thinking_template_kwargs(QWEN, "xhigh") == {
        "enable_thinking": True, "reasoning_effort": "xhigh"}


def test_a_double_without_a_template_is_never_probed():
    class _Double:
        def apply_chat_template(self, messages, **kwargs):
            return "x"

    assert not mc.looks_probeable(_Double())
    view = mc.for_tokenizer(_Double(), model_id=GEMMA)
    assert view.source == mc.SOURCE_HEURISTIC


def test_architecture_reads_a_nested_text_config():
    from types import SimpleNamespace
    config = SimpleNamespace(text_config=SimpleNamespace(
        num_hidden_layers=34, hidden_size=2560,
        layer_types=["sliding_attention"] * 5 + ["full_attention"]))
    assert mc.architecture_from_config(config) == {
        "layerCount": 34, "hiddenSize": 2560,
        "layerTypes": ["sliding_attention"] * 5 + ["full_attention"]}
    assert mc.architecture_from_config({"num_hidden_layers": 28, "hidden_size": 1024}) == {
        "layerCount": 28, "hiddenSize": 1024, "layerTypes": None}
    assert mc.architecture_from_config(None) is None


# =============================================================================
# 5. The client verbs, under --root alone
# =============================================================================


def test_the_client_shows_and_overrides_the_record_under_root_alone(
        fixture, tmp_path, monkeypatch, capsys):
    """``steerlab model capabilities`` / ``set-capability`` with the workspace
    named by ``--root`` and no environment at all — the override must find
    the record through the root `main` resolved, not through a second
    resolution that would refuse."""
    from steerlab_server import client_cli
    monkeypatch.delenv("STEERLAB_WORKSPACE", raising=False)
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    root = str(tmp_path)
    for sub in ("prompts", "experiments", "runs"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    # No record yet: the heuristic answers, and says so; an override refuses.
    assert client_cli.main(["model", "capabilities", QWEN, "--root", root, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["result"]["source"] == mc.SOURCE_HEURISTIC
    assert shown["state"] == "okWithAdvisories"
    assert client_cli.main(["model", "set-capability", QWEN, "thinkOpenInPrompt",
                            "true", "--reason", "x", "--root", root, "--json"]) == 66
    refused = json.loads(capsys.readouterr().out)
    assert refused["error"]["code"] == client_cli.CAPABILITY_RECORD_MISSING_CODE
    # A probed record on disk: shown, then overridden with a reason.
    record = _probe_family(_family(fixture, "chatml-switch"))
    record.update({"modelID": QWEN, "revision": "c" * 40})
    record["recordHash"] = mc.record_hash(record)
    written = mc.write_record(record, root)
    assert client_cli.main(["model", "capabilities", QWEN, "--root", root, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["result"]["record"] == written and shown["state"] == "ready"
    assert client_cli.main(["model", "set-capability", QWEN, "thinkOpenInPrompt",
                            "true", "--reason", "always opens", "--root", root,
                            "--json"]) == 0
    overridden = json.loads(capsys.readouterr().out)
    assert overridden["result"]["overrides"] == {"thinkOpenInPrompt": True}
    on_disk = mc.read_record(os.path.join(root, written))
    assert on_disk["overrides"]["thinkOpenInPrompt"]["reason"] == "always opens"
    assert on_disk["detected"]["thinkOpenInPrompt"] is False
    # A malformed value is a usage refusal, and nothing moves.
    assert client_cli.main(["model", "set-capability", QWEN, "systemRole",
                            "sideways", "--reason", "x", "--root", root,
                            "--json"]) == 64
    capsys.readouterr()
    assert mc.read_record(os.path.join(root, written))["overrides"].keys() == {"thinkOpenInPrompt"}

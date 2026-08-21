"""A2 — `responseFormat` as a real schema field (Python half).

The field had been in one case family's task JSONL since it was authored and existed
nowhere in executable code: it survived only as a round-tripped unknown key.
The answer-token instruments score each option as an immediate continuation of
the prompt, which is only meaningful when the model was asked for a bare
label; on a JSON-response row the scored position holds the opening brace.

Mirror of Swift ``ResponseFormatTests``.
"""

import json
import os

import pytest

from steerlab_server.experiment import response_format as rf


def _item(item_id, options=True, fmt=None, target=True):
    return {"id": item_id, "hasOptions": options, "hasTarget": target,
            "format": fmt}


# --- the vocabulary ---------------------------------------------------------

def test_only_label_supports_answer_token_scoring():
    assert rf.supports_answer_token_scoring(rf.LABEL)
    assert not rf.supports_answer_token_scoring(rf.JSON)
    assert not rf.supports_answer_token_scoring(rf.FREE_TEXT)
    # Absent stays permissive: legacy files predate the field.
    assert rf.supports_answer_token_scoring(None)


def test_absent_parses_to_none_and_unknown_raises():
    assert rf.parse(None) is None
    assert rf.parse("") is None
    assert rf.parse("label") == "label"
    # A typo must NOT degrade to "unspecified".
    with pytest.raises(ValueError, match="unknown responseFormat"):
        rf.parse("lable")
    with pytest.raises(ValueError):
        rf.parse("JSON")


# --- the refusal ------------------------------------------------------------

def test_json_rows_refuse_an_answer_token_instrument():
    refusal = rf.refusal([_item("a", fmt=rf.JSON), _item("b", fmt=rf.JSON)],
                         ["answerTokenLogprob"], None)
    assert refusal is not None
    assert "2 items" in refusal
    assert "a, b" in refusal
    assert "opening brace" in refusal
    # Every way out is named, not merely the problem.
    assert "'label'" in refusal
    assert "outcomeInstrumentScope" in refusal


def test_label_rows_are_fine():
    assert rf.refusal([_item("a", fmt=rf.LABEL)],
                      ["answerTokenLogprob"], None) is None


def test_legacy_rows_stay_permissive():
    assert rf.refusal([_item("a"), _item("b")],
                      ["answerTokenLogprob"], None) is None


def test_zero_option_items_refuse_the_instrument():
    # 2026-08-06 field incident: ordinalScale declared while every task item
    # had options: null — the instrument silently produced zero records and
    # the run burned its whole GPU allocation. The old contract read this as
    # "not the instrument's business"; it is now a refusal.
    refusal = rf.refusal([_item("a", options=False, fmt=rf.JSON)],
                         ["answerTokenLogprob"], None)
    assert refusal is not None
    assert "none of the 1 task item carries options" in refusal
    assert "silently produce zero records" in refusal
    assert "Add 'options'" in refusal

    plural = rf.refusal([_item("a", options=False), _item("b", options=False)],
                        ["ordinalScale"], None)
    assert "none of the 2 task items carries options" in plural


def test_partial_option_coverage_is_not_the_instruments_business():
    # A mixed file where SOME items carry options stays legal: the choice
    # gate is per-item, and the carrying items produce records.
    assert rf.refusal([_item("a", fmt=rf.LABEL), _item("b", options=False)],
                      ["answerTokenLogprob"], None) is None


def test_a_scope_selecting_zero_items_refuses():
    # A pinned scope whose responseFormats select nothing passes the drift
    # check (0 == 0) and used to run the instrument on nothing.
    items = [_item("a", options=True, fmt=rf.JSON)]
    scope = rf.pin_scope(["label"], items)
    assert scope["itemCount"] == 0
    refusal = rf.refusal(items, ["answerTokenLogprob"], scope)
    assert refusal is not None
    assert "outcomeInstrumentScope selects zero task items" in refusal
    assert "silently produce zero records" in refusal


def test_zero_in_scope_option_items_refuse_with_scope_wording():
    # The scope selects rows, but none of THEM carries options.
    items = [_item("a", fmt=rf.LABEL, options=False),
             _item("b", fmt=rf.JSON, options=True)]
    scope = rf.pin_scope(["label"], items)
    assert scope["itemCount"] == 1
    refusal = rf.refusal(items, ["choiceProbability"], scope)
    assert refusal is not None
    assert "none of the 1 in-scope task item carries options" in refusal


# --- the target rule (open-issues #6) ---------------------------------------

def test_a_target_dependent_instrument_refuses_items_that_declare_no_target():
    """The run loop no longer synthesizes `target = options[0]`, so a study
    whose items never declare one produces zero endpoint rows."""
    refusal = rf.refusal(
        [_item("a", fmt=rf.LABEL, target=False),
         _item("b", fmt=rf.LABEL, target=False)],
        ["answerTokenLogprob"], None)
    assert refusal is not None
    assert "none of the 2 task items declares a 'target'" in refusal
    assert "zero endpoint rows" in refusal
    # Every way out is named, including the one that fits a rating ladder.
    assert "declare ordinalScale" in refusal


def test_the_target_rule_is_silent_for_a_declared_ordinal_ladder():
    """A rating ladder legitimately declares no target: its endpoint is the
    ladder position, and target-less is the CORRECT shape there."""
    ladder = [_item("a", fmt=rf.LABEL, target=False)]
    assert rf.refusal(ladder, ["ordinalScale"], None) is None
    # And when the ordinal readout rides an answer-token record (the mixed
    # instrument), the ladder items still may not declare a target.
    assert rf.refusal(
        ladder, ["ordinalScale", "answerTokenLogprob"], None) is None


def test_partial_target_coverage_is_not_the_instruments_business():
    """Same rule the options gate follows: some items declaring a target is
    a measurable study; the untargeted rows carry no choice endpoint and are
    skipped by declaredness downstream."""
    assert rf.refusal(
        [_item("a", fmt=rf.LABEL), _item("b", fmt=rf.LABEL, target=False)],
        ["choiceProbability"], None) is None


def test_the_target_rule_reads_the_pinned_item_file():
    """`items_of` is the only reader of the item bytes, so the rule's input
    must come from the file's own `target` key."""
    items = rf.items_of([
        {"id": "a", "options": ["A", "B"], "target": "A"},
        {"id": "b", "options": ["1", "2", "3", "4", "5"]},
    ])
    assert [i["hasTarget"] for i in items] == [True, False]
    # An empty string is not a declaration.
    assert not rf.items_of([{"id": "c", "target": ""}])[0]["hasTarget"]


def test_no_choice_instrument_means_no_refusal():
    # A json file measured by sampled text is exactly right.
    assert rf.refusal([_item("a", fmt=rf.JSON)], ["sampledText"], None) is None
    assert rf.refusal([_item("a", fmt=rf.JSON)], None, None) is None


def test_mixed_file_explains_every_objection():
    refusal = rf.refusal([_item("a", fmt=rf.JSON), _item("b", fmt=rf.FREE_TEXT)],
                         ["ordinalScale"], None)
    assert "opening brace" in refusal
    assert "prose" in refusal


# --- declared scope ---------------------------------------------------------

def test_a_declared_scope_makes_a_mixed_file_coherent():
    items = [_item("label-1", fmt=rf.LABEL), _item("label-2", fmt=rf.LABEL),
             _item("json-1", fmt=rf.JSON)]
    scope = rf.pin_scope(["label"], items)
    assert scope["itemCount"] == 2
    assert rf.refusal(items, ["answerTokenLogprob"], scope) is None
    assert not rf.scope_includes(scope, items[2])
    assert rf.scope_includes(scope, items[0])


def test_a_scope_that_still_admits_unreadable_rows_still_refuses():
    items = [_item("json-1", fmt=rf.JSON)]
    scope = rf.pin_scope(["label", "json"], items)
    assert rf.refusal(items, ["answerTokenLogprob"], scope) is not None


def test_an_undeclared_row_is_not_proven_in_scope():
    scope = rf.pin_scope(["label"], [_item("x", fmt=rf.LABEL)])
    assert not rf.scope_includes(scope, _item("legacy", fmt=None))


# --- scope drift ------------------------------------------------------------

def test_scope_detects_a_changed_item_count():
    items = [_item("a", fmt=rf.LABEL), _item("b", fmt=rf.LABEL)]
    scope = rf.pin_scope(["label"], items)
    assert rf.scope_drift_refusal(scope, items) is None

    grown = items + [_item("c", fmt=rf.LABEL)]
    drift = rf.scope_drift_refusal(scope, grown)
    assert "pins 2" in drift and "select 3" in drift


def test_scope_detects_same_count_different_items():
    # The subtle case a bare count would miss.
    items = [_item("a", fmt=rf.LABEL), _item("b", fmt=rf.LABEL)]
    scope = rf.pin_scope(["label"], items)
    swapped = [_item("a", fmt=rf.LABEL), _item("c", fmt=rf.LABEL)]
    drift = rf.scope_drift_refusal(scope, swapped)
    assert "same COUNT of different items" in drift


# --- cross-engine agreement -------------------------------------------------

def test_ids_hash_matches_the_swift_rule():
    # Sorted ids joined by newline, SHA-256 — the shared canonical form.
    import hashlib
    items = [_item("b", fmt=rf.LABEL), _item("a", fmt=rf.LABEL)]
    assert rf.ids_hash(items) == hashlib.sha256(b"a\nb").hexdigest()


def test_the_instrument_set_matches_swift():
    # Twin of Swift InstrumentActivation.optionConsumingInstruments.
    assert rf.OPTION_CONSUMING_INSTRUMENTS == {
        "answerTokenLogprob", "choiceProbability", "ordinalScale"}


# --- through the real loader ------------------------------------------------

def _write_prompts(tmp_path, rows):
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8")
    return str(path)


def test_loader_validates_and_carries_the_field(tmp_path):
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path / "ws")
    os.makedirs(root, exist_ok=True)
    es.create("rf", model_id="org/m", root=root)
    prompts_file = _write_prompts(tmp_path, [
        {"id": "a", "prompt": "pick", "options": ["A", "B"],
         "responseFormat": "label"},
        {"id": "b", "prompt": "pick", "options": ["A", "B"],
         "responseFormat": "json"},
        {"id": "c", "prompt": "pick", "options": ["A", "B"]},
    ])
    manifest = Manifest.load("rf", root)
    prompts = tasks._load_prompts(manifest, prompts_file, root)
    assert [p.get("responseFormat") for p in prompts] == ["label", "json", None]

    items = rf.items_of(prompts)
    assert [i["id"] for i in rf.unscorable_items(items)] == ["b"]


def test_loader_refuses_an_unknown_value_naming_the_item(tmp_path):
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path / "ws")
    os.makedirs(root, exist_ok=True)
    es.create("rf2", model_id="org/m", root=root)
    prompts_file = _write_prompts(tmp_path, [
        {"id": "oops", "prompt": "pick", "options": ["A"],
         "responseFormat": "lable"},
    ])
    manifest = Manifest.load("rf2", root)
    with pytest.raises(RuntimeError, match="oops"):
        tasks._load_prompts(manifest, prompts_file, root)


def test_run_start_gate_refuses_before_generation(tmp_path):
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path / "ws")
    os.makedirs(root, exist_ok=True)
    es.create("rf3", model_id="org/m", root=root)
    d = es.load_raw("rf3", root)
    d["outcomeInstruments"] = ["answerTokenLogprob"]
    es.save_raw(d, root)
    manifest = Manifest.load("rf3", root)
    prompts = [{"id": "a", "prompt": "p", "options": ["A", "B"],
                "responseFormat": "json"}]

    with pytest.raises(RuntimeError, match="opening brace"):
        tasks._check_response_formats(manifest, prompts)

    # A declared scope that excludes the json row makes it coherent —
    # provided it still selects an option-carrying row (a scope selecting
    # ZERO items now refuses: the instrument would run on nothing).
    # A real answer-token item declares the option its endpoint tracks; the
    # target rule (open-issues #6) refuses a file that never does.
    prompts.append({"id": "b", "prompt": "p", "options": ["A", "B"],
                    "target": "A", "responseFormat": "label"})
    d["outcomeInstrumentScope"] = rf.pin_scope(["label"], rf.items_of(prompts))
    es.save_raw(d, root)
    tasks._check_response_formats(Manifest.load("rf3", root), prompts)


def test_zero_options_refusal_fires_before_model_load(tmp_path, monkeypatch):
    """The 2026-08-06 field case, walked through the real run entry: an
    option-consuming instrument declared while no task item carries options
    refuses in seconds — before the model is acquired — like the artifact
    preflight, not after the GPU allocation is spent."""
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment import experiment_store as es

    root = str(tmp_path / "ws")
    os.makedirs(root, exist_ok=True)
    es.create("rf4", model_id="org/m", root=root)
    d = es.load_raw("rf4", root)
    d["outcomeInstruments"] = ["answerTokenLogprob"]
    es.save_raw(d, root)
    prompts_file = _write_prompts(tmp_path, [
        {"id": "a", "prompt": "pick"},
        {"id": "b", "prompt": "pick"},
    ])
    acquired = []
    monkeypatch.setattr(
        tasks, "_acquire_model",
        lambda *a, **k: acquired.append(True) or (_ for _ in ()).throw(
            AssertionError("model must not be acquired")))
    with pytest.raises(RuntimeError, match="carries options"):
        tasks.run("rf4", prompts_file, root, log=lambda *_: None)
    assert acquired == []
def test_scope_drift_refuses_before_the_model_loader_is_invoked(
        tmp_path, monkeypatch):
    """2026-08-06 field incident: a 4-shard Slurm run of a scope-drifted
    study (Duplicate & Adjust + task-file swap left a stale
    outcomeInstrumentScope pin) staged and loaded gemma-3-27b-it — ~2.5
    minutes per shard, four times — before the run-stage gate refused. The
    refusal was right; the ordering was not. The gate now also runs in the
    pre-model-load preflight, next to the artifact preflight: the loader
    must never be invoked for a scope-drifted manifest."""
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment import token_preflight as tp

    root = str(tmp_path)
    es.create("sd", model_id="org/m", root=root)
    prompts_file = _write_prompts(tmp_path, [
        {"id": "a", "prompt": "pick", "options": ["A", "B"], "target": "A",
         "responseFormat": "label"},
        {"id": "b", "prompt": "pick", "options": ["A", "B"], "target": "A",
         "responseFormat": "label"},
    ])
    d = es.load_raw("sd", root)
    d["outcomeInstruments"] = ["answerTokenLogprob"]
    d["outcomeInstrumentScope"] = rf.pin_scope(
        ["label"], [_item("a", fmt=rf.LABEL), _item("b", fmt=rf.LABEL)])
    es.save_raw(d, root)
    # The task-file swap the incident describes: same path, different items,
    # scope pin left untouched.
    _write_prompts(tmp_path, [
        {"id": "a", "prompt": "pick", "options": ["A", "B"], "target": "A",
         "responseFormat": "label"},
        {"id": "c", "prompt": "pick", "options": ["A", "B"], "target": "A",
         "responseFormat": "label"},
    ])

    def loaded_anyway(*a, **k):
        raise AssertionError(
            "model loader invoked — the scope-drift refusal must fire "
            "before any staging/GPU cost")

    monkeypatch.setattr(tasks, "_load_model", loaded_anyway)
    # Keep the token preflight offline; it warns and defers, as on a node
    # with no cached tokenizer.
    def no_tokenizer(*a, **k):
        raise tp.PreflightError("no cached tokenizer in this test")

    monkeypatch.setattr(tp, "preflight", no_tokenizer)

    with pytest.raises(RuntimeError, match="outcomeInstrumentScope pins"):
        tasks.run("sd", prompts_file=prompts_file, root=root)

    # Re-pinning the scope against the new file clears the refusal, and the
    # run then proceeds exactly as far as the loader — proving the preflight
    # passes coherent manifests through AND that the loader monkeypatch was
    # armed (a preflight that refused everything would also pass the first
    # assertion).
    d = es.load_raw("sd", root)
    d["outcomeInstrumentScope"] = rf.pin_scope(
        ["label"], [_item("a", fmt=rf.LABEL), _item("c", fmt=rf.LABEL)])
    es.save_raw(d, root)
    with pytest.raises(AssertionError, match="model loader invoked"):
        tasks.run("sd", prompts_file=prompts_file, root=root)


def _attach_concept(root, name, concept="fear"):
    """A concept the manifest pins, with NO validation.jsonl — the
    per-concept convergent loop then skips it harmlessly, leaving the logit
    lens (which iterates the extraction bundles) as the thing under test."""
    from steerlab_server.experiment import experiment_store as es
    directory = os.path.join(root, "prompts", "concepts", concept)
    os.makedirs(directory, exist_ok=True)
    for side, text in (("positive", "afraid"), ("negative", "calm")):
        with open(os.path.join(directory, f"{side}.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps({"text": text}) + "\n")
    es.create(name, model_id="org/m", root=root)
    es.attach(name, [concept], root=root)


# --- C3: the logit lens is finally CALLED -----------------------------------

def test_validate_writes_a_logit_lens_block(tmp_path, monkeypatch):
    """`logit_lens` has existed in extractor.py since the reader work and was
    never called from anywhere: Swift ran its equivalent inside validate, the
    server did not, so the same study produced a `logitLens` block on one
    engine and nothing on the other. This is the missing call site."""
    from types import SimpleNamespace

    from steerlab_server.experiment import tasks
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest
    from steerlab_server.steering import extractor

    root = str(tmp_path)
    _attach_concept(root, "ll")
    manifest = Manifest.load("ll", root)

    vectors = SimpleNamespace(layer_count=4, per_layer=[[1.0, 0.0]] * 4)
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, m, r: {"fear": SimpleNamespace(vectors=vectors)})
    monkeypatch.setattr(tasks, "_persist_vectors", lambda *a, **k: None)

    seen = {}

    def fake_lens(model, vecs, layer, top_k=12):
        seen["top_k"] = top_k
        seen["layer"] = layer
        return extractor.LogitLensReport(
            layer=layer,
            top_positive=[extractor.LogitLensToken(
                token="afraid", token_id=7, logit=3.5)],
            top_negative=[extractor.LogitLensToken(
                token="calm", token_id=9, logit=-2.5)])

    monkeypatch.setattr(extractor, "logit_lens", fake_lens)
    run_dir = tasks._validate_impl("ll", manifest, object(), root, lambda *a: None)

    report = json.load(open(os.path.join(run_dir, "validation-report.json")))
    lens = report["logitLens"]["fear"]
    assert lens["topPositive"][0]["token"] == "afraid"
    assert lens["topNegative"][0]["tokenID"] == 9
    # Pinned to Swift's call, not to logit_lens's own default of 12 — the two
    # engines' reports are meant to be read side by side.
    assert seen["top_k"] == 10


def test_a_failing_lens_never_fails_the_validation_run(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from steerlab_server.experiment import tasks
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest
    from steerlab_server.steering import extractor

    root = str(tmp_path)
    _attach_concept(root, "ll2")
    manifest = Manifest.load("ll2", root)
    vectors = SimpleNamespace(layer_count=4, per_layer=[[1.0, 0.0]] * 4)
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, m, r: {"fear": SimpleNamespace(vectors=vectors)})
    monkeypatch.setattr(tasks, "_persist_vectors", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("no unembedding head")

    monkeypatch.setattr(extractor, "logit_lens", boom)
    run_dir = tasks._validate_impl("ll2", manifest, object(), root, lambda *a: None)
    report = json.load(open(os.path.join(run_dir, "validation-report.json")))
    # A diagnostic must never fail a validation run; Swift records the same
    # skip string.
    assert report["logitLens"]["fear"].startswith("logit-lens skipped:")


# --- C4: battery resolution --------------------------------------------------

def test_battery_resolution_reports_the_operative_gate():
    from steerlab_server.experiment import sweep_selection as sel

    # The plan's worked example: 10 items, tolerance 0.15. A one-item drop
    # (0.1) passes the `>= baseline - 0.15` constraint; the first drop that
    # actually fails is two items, 0.2.
    r = sel.battery_resolution(10, 0.15)
    assert r.effective_tolerance == pytest.approx(0.2)
    assert "steps of 0.1" in r.summary
    assert "gates at the first larger step, 0.2" in r.summary


def test_a_tolerance_exactly_on_a_step_still_gates_one_step_higher():
    from steerlab_server.experiment import sweep_selection as sel

    # The constraint is `>=`, so a drop EQUAL to the tolerance passes.
    r = sel.battery_resolution(10, 0.1)
    assert r.effective_tolerance == pytest.approx(0.2)
    assert r.is_coarse


def test_a_large_battery_resolves_close_to_the_declared_tolerance():
    from steerlab_server.experiment import sweep_selection as sel

    r = sel.battery_resolution(100, 0.15)
    assert r.effective_tolerance == pytest.approx(0.16)
    assert not r.is_coarse


def test_battery_resolution_declines_to_speak_about_an_empty_battery():
    from steerlab_server.experiment import sweep_selection as sel

    assert sel.battery_resolution(0, 0.15) is None
    assert sel.battery_resolution(10, float("nan")) is None


# --- C1: exact token preflight, weights-free --------------------------------

class _FakeTokenizer:
    """Whitespace tokenizer — enough to exercise the preflight's arithmetic
    without downloading a real one."""

    def __call__(self, text, add_special_tokens=True):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=list(range(len(text.split()))))

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(m["content"] for m in messages)


def _preflight_prompts(word_counts):
    return [{"id": f"p{i}", "prompt": " ".join(["w"] * n)}
            for i, n in enumerate(word_counts)]


def test_preflight_reports_every_overflowing_item(monkeypatch):
    from steerlab_server.experiment import token_preflight as tp

    monkeypatch.setattr(tp, "_tokenizer", lambda *a, **k: _FakeTokenizer())
    monkeypatch.setattr(tp, "context_window", lambda *a, **k: 100)

    report = tp.preflight(_preflight_prompts([10, 200, 300, 12]),
                          model_id="org/m", revision="abc", max_tokens=20)
    assert report["promptBudget"] == 100 - 20 - tp.CONTEXT_BUDGET_RESERVE
    assert report["overflowCount"] == 2
    # EVERY failing row, not the first — a preflight that stops early turns a
    # data problem into a guessing game.
    assert [i["id"] for i in report["overflowItems"]] == ["p1", "p2"]
    assert report["exact"] is True


def test_the_refusal_names_the_items_and_the_arithmetic(monkeypatch):
    from steerlab_server.experiment import token_preflight as tp

    monkeypatch.setattr(tp, "_tokenizer", lambda *a, **k: _FakeTokenizer())
    monkeypatch.setattr(tp, "context_window", lambda *a, **k: 100)
    report = tp.preflight(_preflight_prompts([200]), model_id="org/m",
                          revision="a" * 40, max_tokens=20)
    refusal = tp.refusal(report)
    assert "p0" in refusal
    assert "200 tokens" in refusal
    assert "context window is 100" in refusal
    # Nothing is dropped: silently excluding items would change the measured
    # sample without recording it.
    assert "nothing is dropped automatically" in refusal


def test_no_overflow_means_no_refusal(monkeypatch):
    from steerlab_server.experiment import token_preflight as tp

    monkeypatch.setattr(tp, "_tokenizer", lambda *a, **k: _FakeTokenizer())
    monkeypatch.setattr(tp, "context_window", lambda *a, **k: 10_000)
    report = tp.preflight(_preflight_prompts([10, 20]), model_id="org/m",
                          revision=None, max_tokens=100)
    assert report["overflowCount"] == 0
    assert tp.refusal(report) is None


def test_an_unknown_context_window_asserts_nothing(monkeypatch):
    from steerlab_server.experiment import token_preflight as tp

    monkeypatch.setattr(tp, "_tokenizer", lambda *a, **k: _FakeTokenizer())
    monkeypatch.setattr(tp, "context_window", lambda *a, **k: None)
    report = tp.preflight(_preflight_prompts([10]), model_id="org/m",
                          revision=None, max_tokens=100)
    assert report["contextWindow"] is None
    assert report["promptBudget"] is None
    # Counts are still reported; only the verdict is withheld.
    assert report["items"][0]["promptTokens"] == 10
    assert "fits" not in report["items"][0]
    assert tp.refusal(report) is None


def test_the_reserve_matches_the_generation_time_check():
    from steerlab_server.experiment import token_preflight as tp
    from steerlab_server.experiment import generate

    # Drift here would make the preflight disagree with the check it exists
    # to anticipate.
    assert tp.CONTEXT_BUDGET_RESERVE == generate.CONTEXT_BUDGET_RESERVE


def test_a_preflight_that_cannot_run_never_blocks_the_run(tmp_path, monkeypatch):
    """The in-generation ContextBudgetError remains the backstop; turning a
    diagnostic into a new way to fail would be a poor trade."""
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest
    from steerlab_server.experiment import token_preflight as tp

    root = str(tmp_path)
    es.create("pf", model_id="org/m", root=root)
    manifest = Manifest.load("pf", root)

    def boom(*a, **k):
        raise RuntimeError("no cached tokenizer on this node")

    monkeypatch.setattr(tp, "preflight", boom)
    logs = []
    tasks._token_preflight_or_warn(manifest, None, root, logs.append)
    assert any("token preflight unavailable" in line for line in logs)
    assert any("backstop" in line for line in logs)


def test_scope_pin_ids_match_the_run_loader_vocabulary(tmp_path):
    """2026-08-03 field incident: the Swift editor pinned the scope over
    synthesized `item-N` ids while this engine verifies over the loader's
    real ids (fallback `prompt-<ordinal>`), refusing every run with "the
    same COUNT of different items". The golden hash is shared verbatim with
    the Swift test (responseFormatItemsSpeakTheRunLoadersIDVocabulary), and
    the ids come from THE PRODUCTION JSONL LOADER — never a hand-built
    list (review round: manual construction proved nothing about the
    loader's fallback)."""
    from steerlab_server.experiment.manifest import Manifest
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    path = os.path.join(root, "prompts", "t.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            '{"id": "a", "prompt": "p", "options": ["A", "B"], "responseFormat": "label"}\n'
            '{"id": "b", "prompt": "p", "options": ["A", "B"], "responseFormat": "label"}\n'
            '{"id": "c", "prompt": "p", "options": ["A", "B"], "responseFormat": "json"}\n'
            '{"prompt": "p", "options": ["A", "B"], "responseFormat": "label"}\n')
    manifest = Manifest.from_dict({"name": "x", "modelID": "org/m"})
    prompts = tasks._load_prompts(manifest, "prompts/t.jsonl", root)
    items = rf.items_of(prompts)
    assert [i["id"] for i in items] == ["a", "b", "c", "prompt-4"]
    scope = rf.pin_scope(["label"], items)
    assert scope["itemCount"] == 3
    assert scope["itemIDsHash"] == (
        "e3dc84bffe0488d1ab6084ad6359d8a8f9d7fd5a6655c1763b33282bd24918cf")
    assert rf.scope_drift_refusal(scope, items) is None


def test_loader_id_strictness_null_falls_back_and_empty_refuses(tmp_path):
    """Round 2026-08-03 P2: `id: null` and an ABSENT id share the
    prompt-<ordinal> fallback on both engines; an explicit id must be a
    non-empty string or the file refuses at load (message is the
    cross-engine contract)."""
    from steerlab_server.experiment.manifest import Manifest
    from steerlab_server.experiment import tasks

    root = str(tmp_path)
    os.makedirs(os.path.join(root, "prompts"), exist_ok=True)
    with open(os.path.join(root, "prompts", "ok.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"id": null, "prompt": "p"}\n{"prompt": "q"}\n')
    manifest = Manifest.from_dict({"name": "x", "modelID": "org/m"})
    prompts = tasks._load_prompts(manifest, "prompts/ok.jsonl", root)
    assert [p["id"] for p in prompts] == ["prompt-1", "prompt-2"]

    for bad in ('{"id": "", "prompt": "p"}\n',
                '{"id": "   ", "prompt": "p"}\n',
                '{"id": 7, "prompt": "p"}\n'):
        with open(os.path.join(root, "prompts", "bad.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(bad)
        with pytest.raises(RuntimeError,
                           match="empty or non-string 'id'"):
            tasks._load_prompts(manifest, "prompts/bad.jsonl", root)


# --- Coherence-length guard (c18 lesson, 2026-08-13) --------------------------

def test_short_sweep_generations_earn_the_coherence_length_advisory():
    from steerlab_server.experiment import sweep_selection as sel

    # The c18 shape exactly: dose selected on a 256-token sweep, study
    # generating 1024 — the distinct-2 floor was measured where collapse hides.
    msg = sel.coherence_length_advisory(256, 1024, declared=True)
    assert msg is not None
    assert "declares maxTokens 256" in msg
    assert "1024" in msg
    assert "collapse" in msg


def test_the_80_token_default_is_named_as_a_default_not_a_choice():
    from steerlab_server.experiment import sweep_selection as sel

    # Falling silently to the engine default is worse than a short explicit
    # choice, and the advisory must say which one happened.
    msg = sel.coherence_length_advisory(80, 2048, declared=False)
    assert msg is not None
    assert "declares no maxTokens" in msg
    assert "80-token engine default" in msg


def test_adequate_sweep_length_earns_no_advisory():
    from steerlab_server.experiment import sweep_selection as sel

    assert sel.coherence_length_advisory(1024, 1024, declared=True) is None
    assert sel.coherence_length_advisory(2048, 1024, declared=True) is None
    assert sel.coherence_length_advisory(80, None, declared=False) is None

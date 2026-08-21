"""Capability-battery format 2 — the repair of the instrument-dependent
guardrail (docs/BATTERY-REPAIR-DIAGNOSIS-2026-08-13.md).

The observed defect: ONE pinned 20-item battery (32fe69cc…) scored the
untouched gemma-3-27b-it at 0.45 on one 2026-08 arm and 1.00 on another
of the same round. The mechanism is arming, not items — the study manifest's
"you are a federal district judge … respond in JSON" system prompt was applied
to the battery prompts, so the baseline answered "What is the capital of
France?" with a choice-of-law memo, and text matching scored it wrong.

Two properties are pinned here:

* **Legacy files do not move.** A headerless battery is loaded, armed, and
  scored exactly as before, so an existing pinned hash still measures what it
  measured. Its contamination is now LOUD, not silent.
* **Format 2 is invariant.** A battery that declares ``batteryFormat: 2``
  brings its own arming and (by default) is scored by answer-token logprob, so
  the same battery reads the same under any surrounding instrument and any
  condition.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import battery, battery_lint
from steerlab_server.experiment import experiment_store as es, tasks
from steerlab_server.steering.vector_store import ConceptVectors


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


V2_HEADER = {"batteryFormat": 2, "scoring": "choiceProbability"}
V2_ITEMS = [
    {"id": "cap-fr", "prompt": "What is the capital of France?",
     "answer": "Paris", "options": ["Paris", "Lyon", "Nice"]},
    {"id": "sum", "prompt": "What is 17 + 26?",
     "answer": "43", "options": ["43", "33", "44"]},
]


def _v2_lines(header=None, items=None):
    rows = [dict(V2_HEADER, **(header or {}))]
    rows += [dict(i) for i in (items or V2_ITEMS)]
    return "".join(json.dumps(r) + "\n" for r in rows)


LEGACY_LINES = ('{"prompt": "What is the capital of France?", '
                '"answer": "paris", "grading": "token_exact"}\n'
                '{"prompt": "What is 17 + 26?", "answer": "43", '
                '"grading": "exact_number"}\n')


# --- loading ---------------------------------------------------------------


def test_legacy_file_loads_as_format_one_with_the_historical_item_shape(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/legacy.jsonl"
    digest = _write(os.path.join(root, rel), LEGACY_LINES)
    spec = battery.load_spec(rel, root)
    assert spec.format_version == battery.FORMAT_LEGACY
    assert spec.isolated is False
    assert spec.digest == digest
    assert spec.scoring == battery.SCORING_GENERATED
    # EXACTLY the three historical keys — a legacy item's dict is what the
    # old loader produced, so nothing downstream sees a new field.
    assert spec.items == [
        {"prompt": "What is the capital of France?", "answer": "paris",
         "grading": "token_exact"},
        {"prompt": "What is 17 + 26?", "answer": "43",
         "grading": "exact_number"}]
    # And the historical two-value loader still answers the same way.
    items, sha = battery.load_battery(rel, root)
    assert (items, sha) == (spec.items, digest)


def test_v2_header_declares_arming_and_scoring(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/v2.jsonl"
    _write(os.path.join(root, rel),
           _v2_lines({"maxTokens": 8, "promptMode": "raw",
                      "systemPrompt": "Answer plainly."}))
    spec = battery.load_spec(rel, root)
    assert spec.format_version == 2 and spec.isolated
    assert (spec.max_tokens, spec.prompt_mode, spec.system_prompt) == \
        (8, "raw", "Answer plainly.")
    assert [i["id"] for i in spec.items] == ["cap-fr", "sum"]
    assert spec.item_scoring(spec.items[0]) == battery.SCORING_CHOICE


@pytest.mark.parametrize("mutation,fragment", [
    ({"batteryFormat": 3}, "unknown batteryFormat"),
    ({"batteryFormat": 1}, "HEADERLESS"),
    ({"scoring": "vibes"}, "unknown scoring"),
    ({"maxTokens": 0}, '"maxTokens" must be a positive integer'),
    ({"systemPrompt": 7}, '"systemPrompt" must be a string'),
])
def test_v2_header_refuses_junk_naming_why(tmp_path, mutation, fragment):
    root = str(tmp_path)
    rel = "prompts/batteries/bad-header.jsonl"
    _write(os.path.join(root, rel), _v2_lines(mutation))
    with pytest.raises(ValueError) as err:
        battery.load_spec(rel, root)
    assert "line 1" in str(err.value) and fragment in str(err.value)


@pytest.mark.parametrize("item,fragment", [
    ({"prompt": "q", "answer": "a"}, "needs an \"options\" list"),
    ({"prompt": "q", "answer": "a", "options": ["a"]},
     "needs an \"options\" list"),
    ({"prompt": "q", "answer": "z", "options": ["a", "b"]},
     "is not one of \"options\""),
    ({"prompt": "q", "answer": "a", "options": ["a", "a"]},
     "repeats an option"),
    ({"prompt": "q", "answer": "a", "scoring": "generatedText"},
     "must DECLARE \"grading\""),
    ({"prompt": "q", "answer": "a", "scoring": "generatedText",
      "grading": "token_exact", "options": ["a", "b"]},
     "belongs to choiceProbability items only"),
])
def test_v2_items_refuse_undecidable_shapes(tmp_path, item, fragment):
    root = str(tmp_path)
    rel = "prompts/batteries/bad-item.jsonl"
    _write(os.path.join(root, rel), _v2_lines(items=[item]))
    with pytest.raises(ValueError) as err:
        battery.load_spec(rel, root)
    assert "line 2" in str(err.value) and fragment in str(err.value)


# --- arming ----------------------------------------------------------------


def _spec(tmp_path, lines, name="b.jsonl"):
    root = str(tmp_path)
    rel = f"prompts/batteries/{name}"
    _write(os.path.join(root, rel), lines)
    return battery.load_spec(rel, root)


HOSTILE = "You are a federal district judge. Respond in JSON."


def test_legacy_arming_still_takes_the_instruments_context(tmp_path):
    spec = _spec(tmp_path, LEGACY_LINES)
    arming = battery.resolve_arming(spec, prompt_mode="chatAssistant",
                                    system_prompt=HOSTILE,
                                    qwen_thinking_enabled=True)
    assert arming.system_prompt == HOSTILE
    assert arming.qwen_thinking_enabled is True
    assert arming.max_tokens == battery.BATTERY_MAX_TOKENS
    assert arming.isolated is False
    advisory = battery.contamination_advisory(spec, arming)
    assert advisory and "not comparable across instruments" in advisory


def test_v2_arming_ignores_the_instrument_entirely(tmp_path):
    spec = _spec(tmp_path, _v2_lines({"maxTokens": 12}))
    hostile = battery.resolve_arming(spec, prompt_mode="raw",
                                     system_prompt=HOSTILE,
                                     qwen_thinking_enabled=True)
    clean = battery.resolve_arming(spec)
    assert hostile == clean
    assert hostile.system_prompt is None and hostile.max_tokens == 12
    assert hostile.isolated is True
    assert battery.contamination_advisory(spec, hostile) is None


# --- scoring ---------------------------------------------------------------


def _fake_backends(seen=None):
    """Generation that OBEYS the arming's system prompt (as the live model
    does) and a choice reader that does not — the asymmetry the repair rests
    on."""
    def generate_fn(prompt, arming):
        if seen is not None:
            seen.append(("generate", arming))
        if arming.system_prompt:
            return '```json\n{"case_determination": {"issue": "Choice of Law"'
        return "Paris" if "capital" in prompt else "43"

    def choice_fn(prompt, options, arming):
        if seen is not None:
            seen.append(("choice", arming))
        correct = "Paris" if "capital" in prompt else "43"
        probabilities = {o: (0.8 if o == correct else 0.1) for o in options}
        return correct, probabilities
    return generate_fn, choice_fn


def test_v2_choice_item_records_the_distribution_and_scores_by_argmax(tmp_path):
    spec = _spec(tmp_path, _v2_lines())
    arming = battery.resolve_arming(spec)
    generate_fn, choice_fn = _fake_backends()
    fields = battery.score_item(spec, spec.items[0], arming,
                                generate_fn=generate_fn, choice_fn=choice_fn)
    assert fields["correct"] is True
    assert fields["scoring"] == battery.SCORING_CHOICE
    assert fields["selected"] == "Paris" and fields["output"] == "Paris"
    assert fields["choiceProbability"]["Paris"] == 0.8


def test_v2_generated_item_still_text_matches_under_its_declared_grading(tmp_path):
    spec = _spec(tmp_path, _v2_lines(
        {"scoring": "generatedText"},
        [{"id": "cap", "prompt": "What is the capital of France?",
          "answer": "paris", "grading": "token_exact"}]))
    arming = battery.resolve_arming(spec)
    generate_fn, choice_fn = _fake_backends()
    fields = battery.score_item(spec, spec.items[0], arming,
                                generate_fn=generate_fn, choice_fn=choice_fn)
    assert fields == {"scoring": battery.SCORING_GENERATED,
                      "output": "Paris", "correct": True}


def test_legacy_item_scores_through_the_historical_text_matcher(tmp_path):
    spec = _spec(tmp_path, LEGACY_LINES)
    generate_fn, choice_fn = _fake_backends()
    clean = battery.resolve_arming(spec, system_prompt=None)
    hostile = battery.resolve_arming(spec, system_prompt=HOSTILE)
    scored = [battery.score_item(spec, item, arming, generate_fn=generate_fn,
                                 choice_fn=choice_fn)["correct"]
              for arming in (clean, hostile) for item in spec.items]
    # The legacy defect, reproduced at unit scale: identical battery, identical
    # model, two accuracies — 1.00 clean, 0.00 under the study's system prompt.
    assert scored == [True, True, False, False]


def test_v2_scoring_is_invariant_to_the_surrounding_instrument(tmp_path):
    """THE property. Same battery, wildly different surrounding instruments —
    identical per-item results, and the choice reader is what ran."""
    spec = _spec(tmp_path, _v2_lines({"maxTokens": 4}))
    instruments = [
        {},
        {"prompt_mode": "raw", "system_prompt": HOSTILE,
         "qwen_thinking_enabled": True},
        {"prompt_mode": "chatAssistant",
         "system_prompt": "Write 500 words of poetry, always."},
    ]
    results = []
    for kwargs in instruments:
        seen = []
        generate_fn, choice_fn = _fake_backends(seen)
        arming = battery.resolve_arming(spec, **kwargs)
        results.append([battery.score_item(spec, item, arming,
                                           generate_fn=generate_fn,
                                           choice_fn=choice_fn)
                        for item in spec.items])
        assert [kind for kind, _ in seen] == ["choice", "choice"]
        assert all(a.system_prompt is None and a.max_tokens == 4
                   for _, a in seen)
    assert results[0] == results[1] == results[2]
    assert all(f["correct"] for f in results[0])


# --- lint ------------------------------------------------------------------


def _codes(report):
    return {f.code for f in report.findings}


def test_lint_accepts_a_clean_v2_battery(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/clean.jsonl"
    items = [{"id": f"q{i}", "prompt": f"Question {i}: which letter is this?",
              "answer": "a", "options": ["a", "b", "c", "d"]}
             for i in range(12)]
    _write(os.path.join(root, rel), _v2_lines(items=items))
    report = battery_lint.lint(rel, root)
    assert report.ok and report.format_version == 2
    assert report.item_count == 12 and not report.findings


def test_lint_blocks_a_legacy_battery_and_names_the_arming(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/legacy.jsonl"
    _write(os.path.join(root, rel), LEGACY_LINES)
    report = battery_lint.lint(rel, root)
    assert not report.ok
    assert "legacyFormat" in _codes(report)
    # …and every length-sensitive matcher in it is named.
    assert {"containmentScored", "singleNumberRequired"} <= _codes(report)


def test_lint_blocks_an_unloadable_battery(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/broken.jsonl"
    _write(os.path.join(root, rel), '{"batteryFormat": 2}\nnot json\n')
    report = battery_lint.lint(rel, root)
    assert not report.ok and "malformed" in _codes(report)
    assert battery_lint.lint("prompts/batteries/absent.jsonl", root).ok is False


def test_lint_flags_fragile_v2_items(tmp_path):
    root = str(tmp_path)
    rel = "prompts/batteries/fragile.jsonl"
    _write(os.path.join(root, rel), _v2_lines(items=[
        # 2 options = 50% by chance; prompt carries a dead format instruction.
        {"prompt": "Is it a? Answer with one word.", "answer": "a",
         "options": ["a", "b"]},
        # options collide after normalization; one is a prefix of another.
        {"prompt": "Which?", "answer": "A.", "options": ["A.", "a", "abc"]},
        # wildly unbalanced option lengths.
        {"prompt": "Which?", "answer": "no",
         "options": ["no", "yes but only under the older rule", "maybe"]},
        # duplicate prompt (item 4 repeats item 3).
        {"prompt": "Which?", "answer": "x", "options": ["x", "y", "z"]},
    ]))
    report = battery_lint.lint(rel, root)
    assert {"weakDiscrimination", "deadFormatInstruction", "collidingOptions",
            "prefixOptions", "unbalancedOptionLengths", "duplicatePrompt",
            "fewItems"} <= _codes(report)
    # A colliding option set is undecidable, so it BLOCKS.
    assert [f.code for f in report.blockers] == ["collidingOptions"]


def test_lint_cli_exit_codes_and_json(tmp_path, capsys):
    """A lint that only prints is not a gate — the exit code is the contract
    (0 clean / 2 blocked / 64 usage), so CI and a pre-pin script can rely on
    it."""
    from steerlab_server.cli import main
    root = str(tmp_path)
    _write(os.path.join(root, "prompts/batteries/legacy.jsonl"), LEGACY_LINES)
    _write(os.path.join(root, "prompts/batteries/ok.jsonl"),
           _v2_lines(items=[dict(i, id=f"{i['id']}-{n}")
                            for n in range(6) for i in V2_ITEMS]))
    assert main(["--root", root, "battery", "lint",
                 "prompts/batteries/ok.jsonl"]) == 0
    assert main(["--root", root, "battery", "lint",
                 "prompts/batteries/legacy.jsonl"]) == 2
    assert main(["--root", root, "battery", "lint"]) == 64
    capsys.readouterr()
    assert main(["--root", root, "battery", "lint",
                 "prompts/batteries/ok.jsonl", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batteryFormat"] == 2 and payload["ok"] is True
    assert payload["itemCount"] == 12 and payload["sha256"]


# --- end to end through the run loop ---------------------------------------


def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _study(root, name, *, battery_lines, system_prompt):
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    es.add_condition(name, {"name": "fear-a1", "bandWidth": 1,
                            "alphaInNormUnits": False,
                            "slots": [{"concept": "fear", "layer": 2,
                                       "alpha": 1.0}]}, root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 16
    raw["systemPrompt"] = system_prompt
    rel = f"prompts/batteries/{name}.jsonl"
    raw["capabilityBatteryFile"] = rel
    raw["capabilityBatteryHash"] = _write(os.path.join(root, rel),
                                          battery_lines)
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path, '{"id": "p0", "prompt": "Decide the case."}\n')
    return prompts_path


def _patch_engine(monkeypatch, generated=None, scored=None):
    """Model stand-ins with the live asymmetry: generation obeys the system
    prompt it is handed; the choice reader has no format to break.

    ``generated``/``scored`` are optional one-element counters, so a test can
    assert what a resumed run actually recomputed."""
    from steerlab_server.experiment import logprob

    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False, **_):
        if generated is not None:
            generated[0] += 1
        if system_prompt:
            return '```json\n{"case_determination": {"issue": "Choice of Law"'
        if "capital" in prompt:
            return "Paris"
        if "17 + 26" in prompt:
            return "43"
        return "an answer"

    def score_options(model, prompt, options, **kwargs):
        if scored is not None:
            scored[0] += 1
        assert not kwargs.get("system_prompt"), \
            "a format-2 battery must never be handed the study's system prompt"
        correct = "Paris" if "capital" in prompt else "43"
        return SimpleNamespace(
            selected=correct,
            probability={o: (0.8 if o == correct else 0.1) for o in options})

    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", generate)
    monkeypatch.setattr(logprob, "score_options", score_options)


def _accuracies(run_dir):
    report = json.loads(open(os.path.join(run_dir, "report.json")).read())
    return {name: block["capabilityBattery"]["accuracy"]
            for name, block in report["conditions"].items()}


def test_v2_battery_reads_the_same_under_a_hostile_and_a_clean_instrument(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    _patch_engine(monkeypatch)
    hostile = tasks.run("hot", _study(root, "hot", battery_lines=_v2_lines(),
                                      system_prompt=HOSTILE),
                        root, model_provider=_fake_model, log=lambda *_: None)
    clean = tasks.run("cold", _study(root, "cold", battery_lines=_v2_lines(),
                                     system_prompt=""),
                      root, model_provider=_fake_model, log=lambda *_: None)
    assert _accuracies(hostile) == _accuracies(clean)
    assert set(_accuracies(hostile).values()) == {1.0}

    # The records are self-describing: format, arming, and the distribution.
    records = [json.loads(line) for line
               in open(os.path.join(hostile, "battery.jsonl")) if line.strip()]
    assert all(r["batteryFormat"] == 2 and r["armingIsolated"] is True
               and r["armingSystemPrompt"] is False for r in records)
    assert all(r["scoring"] == "choiceProbability"
               and r["choiceProbability"][r["selected"]] == 0.8
               for r in records)
    assert {r["promptID"] for r in records} == {"cap-fr", "sum"}


def test_legacy_battery_still_reproduces_the_instrument_dependent_reading(
        tmp_path, monkeypatch):
    """Regression LOCK, not an endorsement: the legacy hash must keep meaning
    what it meant, defect included — 0.00 under the judicial system prompt,
    1.00 without it — so historical runs remain interpretable."""
    root = str(tmp_path)
    _patch_engine(monkeypatch)
    hostile = tasks.run("lhot", _study(root, "lhot", battery_lines=LEGACY_LINES,
                                       system_prompt=HOSTILE),
                        root, model_provider=_fake_model, log=lambda *_: None)
    clean = tasks.run("lcold", _study(root, "lcold", battery_lines=LEGACY_LINES,
                                      system_prompt=""),
                      root, model_provider=_fake_model, log=lambda *_: None)
    assert set(_accuracies(hostile).values()) == {0.0}
    assert set(_accuracies(clean).values()) == {1.0}
    # Legacy records keep their exact historical key set.
    records = [json.loads(line) for line
               in open(os.path.join(clean, "battery.jsonl")) if line.strip()]
    assert all(set(r) == {"condition", "promptIndex", "promptID", "sampleIndex",
                          "prompt", "answer", "output", "batteryHash",
                          "correct"} for r in records)


def _battery_records(run_dir):
    return [json.loads(line) for line
            in open(os.path.join(run_dir, "battery.jsonl")) if line.strip()]


def test_v2_battery_resume_skips_completed_items_before_generating(
        tmp_path, monkeypatch):
    """Resume must skip the FORWARD PASS, not merely deduplicate rows.

    ``resume.record_key`` keys on promptID, so the pre-generation
    ``writer.skip()`` probe and the emitted record must use the SAME id. When
    a format-2 item carries its own ``id``, a probe still keyed on
    ``battery-<index>`` matches nothing on resume: every completed item is
    re-scored and only emit()'s dedupe hides it — full recompute, silently.
    """
    root = str(tmp_path)
    prompts = _study(root, "v2res", battery_lines=_v2_lines(),
                     system_prompt=HOSTILE)

    # Cancel during the battery: 2 study generations, then the first battery
    # item's choice readout, then stop.
    generated, scored = [0], [0]
    _patch_engine(monkeypatch, generated, scored)
    seen = {}
    cancelled_dir = tasks.run(
        "v2res", prompts, root, model_provider=_fake_model,
        log=lambda *_: None,
        should_cancel=lambda: generated[0] + scored[0] >= 3,
        on_run_directory=lambda d: seen.setdefault("dir", d))
    assert (generated[0], scored[0]) == (2, 1)
    assert len(_battery_records(cancelled_dir)) == 1

    # Resume: the completed study generations and the ONE completed battery
    # item are skipped before any work; exactly the 3 missing items are scored.
    resumed_generated, resumed_scored = [0], [0]
    _patch_engine(monkeypatch, resumed_generated, resumed_scored)
    resumed_dir = tasks.run("v2res", prompts, root, model_provider=_fake_model,
                            log=lambda *_: None, run_directory=cancelled_dir)
    assert resumed_dir == cancelled_dir
    assert (resumed_generated[0], resumed_scored[0]) == (0, 3)

    # Exactly one record per (condition, item) — 2 conditions × 2 items — and
    # the ids the battery declared are the keys.
    records = _battery_records(resumed_dir)
    keys = [(r["condition"], r["promptID"]) for r in records]
    assert len(keys) == len(set(keys)) == 4
    assert {r["promptID"] for r in records} == {"cap-fr", "sum"}
    assert set(_accuracies(resumed_dir).values()) == {1.0}


def test_legacy_battery_promptids_are_unchanged_by_the_id_plumbing(
        tmp_path, monkeypatch):
    """A format-1 item has no ``id`` key, so its promptID — the resume key and
    the stored field — stays ``battery-<index>`` exactly as before."""
    root = str(tmp_path)
    _patch_engine(monkeypatch)
    run_dir = tasks.run("legkeys",
                        _study(root, "legkeys", battery_lines=LEGACY_LINES,
                               system_prompt=""),
                        root, model_provider=_fake_model, log=lambda *_: None)
    records = _battery_records(run_dir)
    assert {r["promptID"] for r in records} == {"battery-0", "battery-1"}
    assert all(r["promptID"] == f"battery-{r['promptIndex']}" for r in records)


def test_legacy_contamination_is_logged_loudly(tmp_path, monkeypatch):
    root = str(tmp_path)
    _patch_engine(monkeypatch)
    lines = []
    tasks.run("loud", _study(root, "loud", battery_lines=LEGACY_LINES,
                             system_prompt=HOSTILE),
              root, model_provider=_fake_model, log=lines.append)
    assert any("WARNING" in line and "capability battery" in line
               and "not comparable across instruments" in line
               for line in lines)

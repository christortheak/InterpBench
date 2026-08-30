"""Capability battery inside the study run (2026-07-13): when the manifest
PINS a battery, ``run`` scores it under EVERY condition of the matrix
(baseline included, steering applied exactly as for that condition's
generations) and stamps per-condition results into report.json under the
cross-engine key ``capabilityBattery`` = {"accuracy", "itemCount",
"batteryHash"}. Battery generations are gate evidence, not outcomes: they go
to battery.jsonl, never generations.jsonl, and are resume-skippable."""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

from steerlab_server.experiment import experiment_store as es, resume, tasks
from steerlab_server.steering.vector_store import ConceptVectors


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


BATTERY_LINES = ('{"prompt": "2+2?", "answer": "4"}\n'
                 '{"prompt": "Capital of France?", "answer": "Paris"}\n')


def _study_fixture(root, name, *, pin_battery=True):
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    es.add_condition(name, {"name": "fear-a1", "bandWidth": 1,
                            "alphaInNormUnits": False,
                            "slots": [{"concept": "fear", "layer": 2, "alpha": 1.0}]},
                     root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 16
    if pin_battery:
        digest = _write(os.path.join(root, "prompts", "batteries", "basic.jsonl"),
                        BATTERY_LINES)
        raw["capabilityBatteryFile"] = "prompts/batteries/basic.jsonl"
        raw["capabilityBatteryHash"] = digest
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path,
           '{"id": "p0", "prompt": "Decide the case."}\n'
           '{"id": "p1", "prompt": "Explain the ruling."}\n')
    return prompts_path


def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _battery_aware_generate(counter=None, cancel_flag=None):
    """Steered arms fail the arithmetic probe; baseline answers everything —
    accuracies must separate per condition."""
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False, **kwargs):
        if counter is not None:
            counter[0] += 1
        if "2+2" in prompt:
            return "5" if injections else "4"
        if "Capital" in prompt:
            return "Paris"
        steered = "steered" if injections else "plain"
        return f"{steered} answer to {prompt}"
    return generate


def _patch(monkeypatch, generate_fn):
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", generate_fn)


def test_run_scores_pinned_battery_per_condition(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "bat")
    _patch(monkeypatch, _battery_aware_generate())
    run_dir = tasks.run("bat", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)

    report = json.loads(open(os.path.join(run_dir, "report.json")).read())
    digest = es.load_raw("bat", root)["capabilityBatteryHash"]
    baseline = report["conditions"]["baseline"]["capabilityBattery"]
    steered = report["conditions"]["fear-a1"]["capabilityBattery"]
    assert baseline == {"accuracy": 1.0, "itemCount": 2, "batteryHash": digest}
    assert steered == {"accuracy": 0.5, "itemCount": 2, "batteryHash": digest}

    # Battery records live in battery.jsonl, NOT generations.jsonl.
    battery_records = [json.loads(line) for line in
                       open(os.path.join(run_dir, "battery.jsonl"))
                       if line.strip()]
    assert len(battery_records) == 4  # 2 conditions × 2 items
    assert all(r["promptID"].startswith("battery-") for r in battery_records)
    assert all(r["batteryHash"] == digest for r in battery_records)
    generations = [json.loads(line) for line in
                   open(os.path.join(run_dir, "generations.jsonl"))
                   if line.strip()]
    assert not any(str(r.get("promptID", "")).startswith("battery-")
                   for r in generations)


def test_run_without_battery_pin_stamps_nothing(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "nobat", pin_battery=False)
    _patch(monkeypatch, _battery_aware_generate())
    run_dir = tasks.run("nobat", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    report = json.loads(open(os.path.join(run_dir, "report.json")).read())
    assert not any("capabilityBattery" in block
                   for block in report["conditions"].values())
    assert not os.path.exists(os.path.join(run_dir, "battery.jsonl"))


def test_run_refuses_battery_drift_against_pin(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "drift")
    _write(os.path.join(root, "prompts", "batteries", "basic.jsonl"),
           '{"prompt": "EDITED", "answer": "x"}\n')
    _patch(monkeypatch, _battery_aware_generate())
    import pytest
    with pytest.raises(RuntimeError, match="drifted from the pinned hash"):
        tasks.run("drift", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)


def test_battery_cancel_parks_resumable_and_resume_completes_byte_identical(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "batresume")

    # Control: an uninterrupted run.
    _patch(monkeypatch, _battery_aware_generate())
    control_dir = tasks.run("batresume", prompts, root,
                            model_provider=_fake_model, log=lambda *_: None)
    control_generations = open(
        os.path.join(control_dir, "generations.jsonl"), "rb").read()
    control_battery = open(
        os.path.join(control_dir, "battery.jsonl"), "rb").read()
    control_report = open(os.path.join(control_dir, "report.json"), "rb").read()

    # Interrupted: cooperative cancel arrives DURING the battery phase.
    # The main loop makes 4 generations (2 conditions × 2 prompts); cancel
    # after the 5th generation = first battery item.
    counter = [0]
    _patch(monkeypatch, _battery_aware_generate(counter))
    seen = {}
    cancelled_dir = tasks.run(
        "batresume", prompts, root, model_provider=_fake_model,
        log=lambda *_: None, should_cancel=lambda: counter[0] >= 5,
        on_run_directory=lambda d: seen.setdefault("dir", d))
    state = resume.read_state(cancelled_dir)
    assert state is not None and state["reason"] == "cancel"
    assert not os.path.exists(os.path.join(cancelled_dir, "report.json"))
    partial_battery = open(os.path.join(cancelled_dir, "battery.jsonl"),
                           "rb").read()
    assert 0 < len(partial_battery.splitlines()) < len(
        control_battery.splitlines())

    # Resume completes to the byte-identical union — battery included.
    resumed_counter = [0]
    _patch(monkeypatch, _battery_aware_generate(resumed_counter))
    resumed_dir = tasks.run("batresume", prompts, root,
                            model_provider=_fake_model, log=lambda *_: None,
                            run_directory=cancelled_dir)
    assert resumed_dir == cancelled_dir
    assert open(os.path.join(cancelled_dir, "generations.jsonl"),
                "rb").read() == control_generations
    assert open(os.path.join(cancelled_dir, "battery.jsonl"),
                "rb").read() == control_battery
    assert open(os.path.join(cancelled_dir, "report.json"),
                "rb").read() == control_report
    assert not resume.is_resumable(cancelled_dir)
    # Completed records were skipped, not regenerated: the interrupted pass
    # made 5 generations (4 study + 1 battery item), so the resume makes
    # exactly the 3 missing battery generations.
    assert counter[0] == 5 and resumed_counter[0] == 3


def test_battery_checkpoint_flag_parks_run_with_exit_contract(tmp_path,
                                                              monkeypatch):
    import pytest
    root = str(tmp_path)
    prompts = _study_fixture(root, "batckpt")
    flag = resume.CheckpointFlag()
    counter = [0]

    def arming_generate(*args, **kwargs):
        counter[0] += 1
        if counter[0] == 5:  # first battery generation
            flag.request()
        return _battery_aware_generate()(*args, **kwargs)

    _patch(monkeypatch, arming_generate)
    seen = {}
    with pytest.raises(resume.CheckpointRequested):
        tasks.run("batckpt", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda d: seen.setdefault("dir", d))
    run_dir = seen["dir"]
    assert resume.is_resumable(run_dir)
    assert not os.path.exists(os.path.join(run_dir, "report.json"))

    # Resume completes normally.
    _patch(monkeypatch, _battery_aware_generate())
    tasks.run("batckpt", prompts, root, model_provider=_fake_model,
              log=lambda *_: None, run_directory=run_dir)
    report = json.loads(open(os.path.join(run_dir, "report.json")).read())
    assert report["conditions"]["baseline"]["capabilityBattery"]["accuracy"] == 1.0


# Cross-engine battery SHAPE contract (2026-07-20): the loader must accept
# exactly what Swift ``CapabilityBattery`` decodes — string prompt, string
# answer, grading absent or one of the five ``GradingMode`` raw values —
# and refuse everything else with an error naming the line and problem.
# The old loader accepted {"prompt": 123, "answer": 7, "grading":
# "made_up"} verbatim: a battery that verified and ran here but refused on
# Swift, and whose made-up grading silently scored every item wrong.

# Pinned as a LITERAL list (mirrored in Swift ``ScoringTests``): a
# vocabulary change on either engine must be a deliberate cross-engine
# contract change, not a drive-by.
SWIFT_GRADING_MODES = ["exact_number", "yes_no", "token_exact",
                       "exact_normalized", "regex"]


def test_grading_vocabulary_is_pinned_to_the_swift_enum():
    from steerlab_server.experiment import battery
    assert list(battery.GRADING_MODES) == SWIFT_GRADING_MODES


def test_battery_loader_refuses_junk_records_naming_why(tmp_path):
    import pytest
    from steerlab_server.experiment import battery
    root = str(tmp_path)
    rel = "prompts/batteries/junk.jsonl"

    def refuses(line, fragment):
        _write(os.path.join(root, rel), line + "\n")
        with pytest.raises(ValueError) as err:
            battery.load_battery(rel, root=root)
        assert "line 1" in str(err.value)
        assert fragment in str(err.value)

    # The reproduced junk record: non-string prompt is named first.
    refuses('{"prompt":123,"answer":7,"grading":"made_up"}',
            '"prompt" must be a non-empty string')
    refuses('{"prompt":"2+2?","answer":7}', '"answer" must be a string')
    refuses('{"prompt":"2+2?","answer":"4","grading":"made_up"}',
            "unknown grading 'made_up'")
    refuses('{"prompt":"","answer":"4"}',
            '"prompt" must be a non-empty string')
    refuses('not json', "not valid JSON")
    refuses('["array"]', "not a JSON object")


def test_battery_loader_round_trips_the_swift_grading_set(tmp_path):
    from steerlab_server.experiment import battery
    root = str(tmp_path)
    rel = "prompts/batteries/modes.jsonl"
    lines = [json.dumps({"prompt": f"q{i}?", "answer": "4", "grading": mode})
             for i, mode in enumerate(SWIFT_GRADING_MODES)]
    lines.append(json.dumps({"prompt": "inferred?", "answer": "4"}))
    _write(os.path.join(root, rel), "\n".join(lines) + "\n")
    items, _ = battery.load_battery(rel, root=root)
    assert [i["grading"] for i in items] == SWIFT_GRADING_MODES + [None]


def test_verify_side_shape_check_tightens_with_the_loader(tmp_path):
    """``_battery_shape_violations`` calls the loader, so a hash-clean
    battery with a made-up grading (or non-string fields) is now a verify
    violation — it used to pass verify here and refuse on Swift."""
    from steerlab_server.experiment.manifest import Manifest
    root = str(tmp_path)
    rel = "prompts/batteries/loose.jsonl"
    digest = _write(os.path.join(root, rel),
                    '{"prompt":123,"answer":7,"grading":"made_up"}\n')
    violations = Manifest.from_dict({
        "name": "s", "modelID": "org/m",
        "capabilityBatteryFile": rel,
        "capabilityBatteryHash": digest}).verify(root)
    assert any(rel in v and "line 1" in v
               and '"prompt" must be a non-empty string' in v
               for v in violations)

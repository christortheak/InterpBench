"""Two circularity-firewall repairs found by the WP0 gate-5 probe (2026-08-17).

**Vacuous validate evidence.** A ``validate`` run for a pinned concept with no
``prompts/concepts/<c>/validation.jsonl`` — the DEFAULT state, since workspace
seeding creates none — scored no held-out probe, exited 0, and SATISFIED
freeze's ``validateEvidence`` gate: an unforced, unstamped freeze
indistinguishable from a validated one, while ``data check`` called the same
missing file a blocker. ``validate`` now stamps ``vacuousConcepts`` and the
gate refuses it under the same gate id.

**A study that measures nothing.** ``run`` on a concept study with zero
injection conditions produced a baseline-only run at exit 0, and ``analyze``
then reported zero effect sizes at exit 0 — a pipeline that "succeeds" end to
end and measures nothing. ``run`` refuses (the 2026-08-11
``inert_conditions_problem`` idiom, for the other road to the same silence)
and ``analyze`` warns on stderr.

Swift twins: ``VacuousValidationTests`` in ExperimentKitTests.
"""

import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import (
    Manifest, no_measured_conditions_problem)

LAYER_COUNT = 8


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _study(tmp_path, name="vac", *, validation=None, concepts=("fear",)):
    """A paired concept study; ``validation`` (rows) authors the held-out
    probe file, None leaves the default state — no validation.jsonl."""
    root = str(tmp_path)
    for concept in concepts:
        directory = os.path.join(root, "prompts", "concepts", concept)
        _write(os.path.join(directory, "positive.jsonl"),
               json.dumps({"text": "afraid"}) + "\n")
        _write(os.path.join(directory, "negative.jsonl"),
               json.dumps({"text": "calm"}) + "\n")
        if validation:
            _write(os.path.join(directory, "validation.jsonl"),
                   "\n".join(json.dumps(r) for r in validation) + "\n")
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, list(concepts), root=root)
    return root


def _bundles(concepts=("fear",)):
    out = {}
    for index, name in enumerate(concepts):
        out[name] = SimpleNamespace(
            vectors=SimpleNamespace(
                layer_count=LAYER_COUNT,
                per_layer=[[1.0, float(index)] for _ in range(LAYER_COUNT)]),
            residual_norm_per_layer=[1.0] * LAYER_COUNT,
            residual_norm_source="extraction-stimuli",
            stimulus_hash="h")
    return out


def _run_validate(root, name, monkeypatch, *, concepts=("fear",),
                  accuracy=0.9, log=None):
    """``_validate_impl`` with extraction, activations and the lens stubbed —
    the vacuity ledger is about which probes RAN, not about numbers."""
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: _bundles(concepts))
    monkeypatch.setattr(tasks, "_persist_vectors", lambda *a, **k: None)
    from steerlab_server.steering import extractor
    monkeypatch.setattr(
        extractor, "activations",
        lambda model, texts, reading, rendering=None: SimpleNamespace(
            values=[[[1.0, 0.0]] * LAYER_COUNT for _ in texts]))
    monkeypatch.setattr(
        extractor, "logit_lens",
        lambda model, vectors, layer, top_k=12: extractor.LogitLensReport(
            layer=layer, top_positive=[], top_negative=[]))
    from steerlab_server.experiment import concept_stats
    monkeypatch.setattr(concept_stats, "scenario_accuracy",
                        lambda **kw: accuracy)
    manifest = Manifest.load(name, root)
    return tasks._validate_impl(name, manifest, object(), root,
                                log or (lambda *a: None))


# --- validate: the vacuity stamp ---------------------------------------------

def test_validate_stamps_vacuous_when_the_concept_has_no_validation_set(
        tmp_path, monkeypatch):
    root = _study(tmp_path)
    logged: list[str] = []
    run = _run_validate(root, "vac", monkeypatch, log=logged.append)
    evidence = json.load(
        open(os.path.join(run, "validation-evidence.json"), encoding="utf-8"))
    report = json.load(
        open(os.path.join(run, "validation-report.json"), encoding="utf-8"))
    assert evidence["vacuousConcepts"] == ["fear"]
    # The run directory says on its own face what it did not measure.
    assert report["vacuousConcepts"] == ["fear"]
    # Loud, and it names the exact file that would make the run real.
    warning = "\n".join(logged)
    assert "VACUOUS validation" in warning
    assert "prompts/concepts/fear/validation.jsonl" in warning


def test_validate_stamps_an_empty_list_when_every_concept_was_probed(
        tmp_path, monkeypatch):
    root = _study(tmp_path, validation=[{"text": "a dark stairwell",
                                         "expresses": True},
                                        {"text": "a sunny kitchen",
                                         "expresses": False}])
    run = _run_validate(root, "vac", monkeypatch)
    evidence = json.load(
        open(os.path.join(run, "validation-evidence.json"), encoding="utf-8"))
    assert evidence["vacuousConcepts"] == []
    # Real evidence, so the gate has nothing to say.
    manifest = Manifest.load("vac", root)
    assert es.vacuous_validate_evidence_problem("vac", manifest, evidence) is None


def test_a_mixed_study_stamps_only_the_unprobed_concept(tmp_path, monkeypatch):
    root = str(tmp_path)
    for concept, rows in (("fear", [{"text": "a dark stairwell",
                                     "expresses": True}]),
                          ("calm", None)):
        directory = os.path.join(root, "prompts", "concepts", concept)
        _write(os.path.join(directory, "positive.jsonl"),
               json.dumps({"text": "x"}) + "\n")
        _write(os.path.join(directory, "negative.jsonl"),
               json.dumps({"text": "y"}) + "\n")
        if rows:
            _write(os.path.join(directory, "validation.jsonl"),
                   "\n".join(json.dumps(r) for r in rows) + "\n")
    es.create("mix", model_id="org/m", revision="abc", root=root)
    es.attach("mix", ["fear", "calm"], root=root)
    run = _run_validate(root, "mix", monkeypatch, concepts=("fear", "calm"))
    evidence = json.load(
        open(os.path.join(run, "validation-evidence.json"), encoding="utf-8"))
    assert evidence["vacuousConcepts"] == ["calm"]


# --- freeze: the gate reads the stamp ----------------------------------------

def _evidence(root, name, *, vacuous=None, stamp="v"):
    """Scope-matched validate evidence. ``vacuous=None`` writes LEGACY
    evidence — no ``vacuousConcepts`` key at all."""
    scope = Manifest.load(name, root=root).validation_scope_hash()
    rundir = os.path.join(root, "runs", f"{stamp}-exp-{name}-validate")
    os.makedirs(rundir, exist_ok=True)
    evidence = {"schemaVersion": 1, "task": "validate",
                "substrate": "python-hf-transformers",
                "reportFile": "validation-report.json",
                "validationScopeHash": scope}
    if vacuous is not None:
        evidence["vacuousConcepts"] = list(vacuous)
    json.dump(evidence,
              open(os.path.join(rundir, "validation-evidence.json"), "w",
                   encoding="utf-8"))
    json.dump({"concepts": {"fear": {"scenarioAccuracy": 0.9}}},
              open(os.path.join(rundir, "validation-report.json"), "w",
                   encoding="utf-8"))
    return rundir


def test_freeze_refuses_vacuous_validate_evidence(tmp_path):
    root = _study(tmp_path)
    _evidence(root, "vac", vacuous=["fear"])
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.freeze("vac", root=root)
    message = str(caught.value)
    assert "VACUOUS evidence" in message
    # The remedy names the exact missing file, not a category.
    assert "prompts/concepts/fear/validation.jsonl" in message
    assert es.load_raw("vac", root)["status"] == "draft"


def test_the_refusal_keeps_the_existing_gate_id(tmp_path):
    root = _study(tmp_path)
    _evidence(root, "vac", vacuous=["fear"])
    manifest = Manifest.load("vac", root)
    gates = dict(es._evaluate_freeze_gates(
        "vac", es.load_raw("vac", root), manifest, root))
    assert "validateEvidence" in gates
    assert "VACUOUS evidence" in gates["validateEvidence"]


def test_force_freeze_stamps_the_vacuous_gate(tmp_path):
    root = _study(tmp_path)
    _evidence(root, "vac", vacuous=["fear"])
    frozen = es.freeze("vac", force=True, root=root)
    assert frozen["freezeForced"] is True
    assert "validateEvidence" in frozen["forcedGatesSkipped"]


def test_non_vacuous_evidence_freezes(tmp_path):
    root = _study(tmp_path, validation=[{"text": "t", "expresses": True}])
    _evidence(root, "vac", vacuous=[])
    assert es.freeze("vac", root=root)["status"] == "frozen"


def test_legacy_evidence_without_the_stamp_still_satisfies_the_gate(tmp_path):
    """Only NEWLY vacuous runs stop: evidence written before the stamp
    existed carries no verdict and must not be retroactively convicted."""
    root = _study(tmp_path)
    _evidence(root, "vac", vacuous=None)
    assert es.freeze("vac", root=root)["status"] == "frozen"


def test_a_stamp_naming_a_concept_the_manifest_dropped_is_ignored(tmp_path):
    """Evidence is matched by SCOPE, not by name — a stamp about a concept
    this manifest no longer pins says nothing about this manifest."""
    root = _study(tmp_path)
    manifest = Manifest.load("vac", root)
    assert es.vacuous_validate_evidence_problem(
        "vac", manifest,
        {"vacuousConcepts": ["some-other-concept"]}) is None


# --- run / analyze: a study that measures nothing ----------------------------

def _raw(**extra):
    d = {"name": "s", "modelID": "org/m", "studyKind": "modelOutput",
         "concepts": [{"name": "fear", "stimulusSetHash": "h"}],
         "conditions": []}
    d.update(extra)
    return d


def test_no_measured_conditions_problem_fires_on_concepts_without_arms():
    problem = no_measured_conditions_problem(_raw())
    assert problem is not None
    assert "BASELINE only" in problem
    # It names both remedies: mint an arm, or declare baseline-only.
    assert "promote" in problem
    assert "agentComparison" in problem


def test_no_measured_conditions_problem_is_silent_when_an_arm_exists():
    for extra in (
        {"conditions": [{"name": "fear-hi",
                         "slots": [{"concept": "fear", "layer": 12,
                                    "alpha": 1.0}]}]},
        {"variantConditions": [{"name": "agent-1"}]},
        {"saeLatentConditions": [{"name": "latent-1"}]},
        # The sanctioned spelling of a deliberate baseline-only run.
        {"studyType": "agentComparison"},
        # Nothing was ever derived: no promise, no silence to break.
        {"concepts": []},
        # A panel runs a scenario, not conditions.
        {"studyKind": "multiAgent"},
    ):
        assert no_measured_conditions_problem(_raw(**extra)) is None, extra


def test_a_canonical_baseline_alone_is_still_nothing_measured():
    assert no_measured_conditions_problem(
        _raw(conditions=[{"name": "baseline", "slots": []}])) is not None


def test_run_refuses_a_concept_study_with_no_injection_conditions(
        tmp_path, monkeypatch):
    root = _study(tmp_path, name="nostudy")
    prompts = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts, json.dumps({"id": "p0", "prompt": "Decide."}) + "\n")
    loaded = []
    monkeypatch.setattr(tasks, "_acquire_model",
                        lambda *a, **k: loaded.append(1))
    with pytest.raises(RuntimeError, match="BASELINE only"):
        tasks.run("nostudy", prompts, root, log=lambda *_: None)
    # Refused BEFORE any queue wait or model load.
    assert loaded == []


def test_analyze_warns_on_stderr_when_every_record_is_baseline(
        tmp_path, monkeypatch, capsys):
    root = _study(tmp_path, name="an")
    run_dir = os.path.join(root, "runs", "exp-an-run")
    os.makedirs(run_dir)
    manifest = Manifest.load("an", root)
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for index in range(2):
            handle.write(json.dumps({
                "condition": "baseline", "promptID": f"p{index}",
                "promptIndex": index, "seed": 0, "output": "text",
                "wordCount": 1, "distinct2": 1.0, "markerDensity": {}}) + "\n")
    json.dump({"experiment": "an", "experimentHash": manifest.content_hash()},
              open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8"))
    tasks.analyze("an", root, source_run=run_dir, log=lambda *_: None,
                  allow_unverified_epoch=True)
    err = capsys.readouterr().err
    assert "only BASELINE records" in err
    assert "no effect sizes" in err

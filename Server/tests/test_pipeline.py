"""The chain runner (seamless pipeline stage 3, 2026-07-18).

One submission runs the manifest's declared stages with ONE model load,
evaluating declared gates between stages. Contract under test:

- ``pipeline_spec``: the pure resolver refuses malformed chains at START
  (unknown/out-of-order stages, bad thresholds, gates for missing stages),
  and the gate evaluators fail closed on unmeasurable inputs (a concept
  omitted from the validate report, an unlabeled concept, a nan cosine).
- ``tasks.pipeline``: stage order, exactly one model acquisition, the
  ``pipeline.json`` stage ledger, gate aborts as recorded determinations
  (``pipeline-abort.json`` + disposition ``aborted``, never an exception),
  the inline-judging preflight, and requeue resume — coarse stage-skip,
  per-concept promote idempotency, and record-level resume through the run
  stage's checkpoint machinery.

The stage TASKS themselves are faked here (each has its own suite); this
suite tests the orchestration around them.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import pipeline_spec as pspec
from steerlab_server.experiment import promote as promote_mod
from steerlab_server.experiment import resume as resume_mod
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest


# --- shared fixtures ---------------------------------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _workspace(root, name="chain", *, pipeline=None):
    d = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')
    _write(os.path.join(d, "markers.json"), json.dumps({"words": ["dread"]}))
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    if pipeline is not None:
        raw = es.load_raw(name, root)
        raw["pipeline"] = pipeline
        es.save_raw(raw, root)
    return name


@contextmanager
def _acquire_counter(counter):
    """A model_provider that counts acquisitions — the one-load assertion."""
    @contextmanager
    def provider(model_id, revision=None):
        counter.append(model_id)
        yield SimpleNamespace(revision=revision or "abc")
    yield provider


def _stage_fakes(monkeypatch, root, calls, *,
                 accuracy=0.9, cosine="0.1000",
                 sweep_entry=None, run_behavior=None):
    """Install fake stage tasks that record call order and write exactly the
    artifacts the gates read. ``sweep_entry`` is the recommendations.json
    value for 'fear' (dict = selection, str = the sweep's failure message).
    ``run_behavior(kwargs, run_dir)`` may raise (checkpoint tests)."""
    def _dir(stage):
        run_dir = os.path.join(root, "runs", f"20260718T00000{len(calls)}000-"
                                             f"exp-chain-{stage}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def fake_extract(name, r=None, dtype="auto", device=None, **kwargs):
        calls.append("extract")
        return _dir("extract")

    def fake_validate(name, r=None, dtype="auto", device=None, **kwargs):
        calls.append("validate")
        run_dir = _dir("validate")
        report = {"experiment": name, "concepts": {}}
        if accuracy is not None:
            report["concepts"]["fear"] = {
                "layer": 2, "scenarioCount": 4, "labeled": True,
                "scenarioAccuracy": accuracy}
        _write(os.path.join(run_dir, "validation-report.json"),
               json.dumps(report))
        _write(os.path.join(run_dir, "cosine-matrix.csv"),
               "concept,fear,calm\nfear,1.0000,{c}\ncalm,{c},1.0000\n".format(
                   c=cosine))
        return run_dir

    def fake_sweep(name, r=None, dtype="auto", device=None, **kwargs):
        calls.append("sweep")
        run_dir = _dir("sweep")
        entry = sweep_entry
        if entry is None:
            entry = {"sweepRun": os.path.basename(run_dir),
                     "winningCell": {"layer": 2, "alpha": 4.0},
                     "criterion": {"metric": "markerDensity"}}
        _write(os.path.join(run_dir, "recommendations.json"),
               json.dumps({"fear": entry}))
        # The sweep-evidence marker _newest_sweep_evidence requires.
        _write(os.path.join(run_dir, "sweep.csv"), "concept,layer,alpha\n")
        return run_dir

    def fake_promote(name, concept, agent_name=None, cell=None,
                     override_reason=None, root=None, log=print,
                     sweep_run=None, pins=None):
        calls.append(f"promote:{concept}")
        # The chain pins its own sweep; under the B2 contract the run
        # travels inside `pins` rather than as a bare argument, so the
        # recorded value must come from whichever the caller used.
        pinned_run = pins.sweep_run if pins is not None else sweep_run
        calls.append(("promote-sweep-run", pinned_run))
        return {"variant": {"name": f"{name}-{concept}-agent",
                            "promotion": {"sweepRun": pinned_run,
                                          "winningCell": {"layer": 2,
                                                          "alpha": 4.0}}},
                "path": f"runs/model-variants/{name}-{concept}-agent.json",
                "hash": "h", "runDirectory": None}

    def fake_run(name, prompts_file=None, r=None, dtype="auto", device=None,
                 **kwargs):
        calls.append("run")
        run_dir = kwargs.get("run_directory") or _dir("run")
        if kwargs.get("on_run_directory"):
            kwargs["on_run_directory"](run_dir)
        if run_behavior is not None:
            return run_behavior(kwargs, run_dir)
        _write(os.path.join(run_dir, "report.json"), "{}")
        return run_dir

    def fake_evaluate(name, r=None, source_run=None, **kwargs):
        calls.append(("evaluate", source_run))
        return _dir("evaluate")

    def fake_analyze(name, r=None, source_run=None, **kwargs):
        calls.append(("analyze", source_run))
        return _dir("analyze")

    monkeypatch.setattr(tasks, "extract", fake_extract)
    monkeypatch.setattr(tasks, "validate", fake_validate)
    monkeypatch.setattr(tasks, "sweep", fake_sweep)
    monkeypatch.setattr(promote_mod, "promote", fake_promote)
    monkeypatch.setattr(tasks, "run", fake_run)
    monkeypatch.setattr(tasks, "evaluate", fake_evaluate)
    monkeypatch.setattr(tasks, "analyze", fake_analyze)


GATES = {"validate": {"minScenarioAccuracy": 0.6,
                      "maxCrossConceptCosine": 0.8},
         "sweep": {"requireSelectionForEveryConcept": True}}


# --- pipeline_spec: resolver refusals (pure) ---------------------------------

def test_resolver_defaults_and_refusals():
    assert pspec.resolve_pipeline(None).stages == pspec.DEFAULT_STAGES
    assert pspec.resolve_pipeline({}).stages == pspec.DEFAULT_STAGES
    assert pspec.resolve_pipeline({"stages": []}).stages == pspec.DEFAULT_STAGES

    spec = pspec.resolve_pipeline({"stages": ["extract", "run"],
                                   "gates": {}})
    assert spec.stages == ("extract", "run")
    assert spec.validate_gate is None and spec.sweep_gate is None

    with pytest.raises(ValueError, match="unknown pipeline stage"):
        pspec.resolve_pipeline({"stages": ["extract", "teleport"]})
    with pytest.raises(ValueError, match="canonical order"):
        pspec.resolve_pipeline({"stages": ["run", "extract"]})
    with pytest.raises(ValueError, match="duplicates"):
        pspec.resolve_pipeline({"stages": ["extract", "extract"]})
    with pytest.raises(ValueError, match="unknown pipeline key"):
        pspec.resolve_pipeline({"stage": ["extract"]})
    with pytest.raises(ValueError, match="no gate is defined"):
        pspec.resolve_pipeline({"gates": {"promote": {}}})
    with pytest.raises(ValueError, match="not in the stage list"):
        pspec.resolve_pipeline({"stages": ["extract", "run"],
                                "gates": {"validate": {}}})
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        pspec.resolve_pipeline(
            {"gates": {"validate": {"minScenarioAccuracy": 1.5}}})
    with pytest.raises(ValueError, match="must be a number"):
        pspec.resolve_pipeline(
            {"gates": {"validate": {"maxCrossConceptCosine": "high"}}})
    with pytest.raises(ValueError, match="unknown validate-gate key"):
        pspec.resolve_pipeline(
            {"gates": {"validate": {"minAccuracy": 0.5}}})
    with pytest.raises(ValueError, match="unknown sweep-gate key"):
        pspec.resolve_pipeline(
            {"gates": {"sweep": {"requireSelection": True}}})
    # A chain is self-contained: evaluate/analyze without run would
    # silently select an OLDER run — refuse at declaration.
    with pytest.raises(ValueError, match="require 'run'"):
        pspec.resolve_pipeline({"stages": ["extract", "evaluate"]})
    with pytest.raises(ValueError, match="require 'run'"):
        pspec.resolve_pipeline({"stages": ["extract", "analyze"]})
    assert pspec.resolve_pipeline(
        {"stages": ["run", "evaluate", "analyze"]}).stages == (
        "run", "evaluate", "analyze")


FAIR_SHAPE_REPORT = {"concepts": {
    # The fair incident shape: transfer at chance (one-sided), calibration
    # and AUC clean.
    "fair": {"labeled": True, "scenarioAccuracy": 0.5,
             "diagnostics": {"oneSidedPredictions": True, "auc": 0.855,
                             "heldOutCalibration": {"accuracy": 0.85,
                                                    "balancedAccuracy": 0.85}}},
    # A pre-diagnostics report row (older engine).
    "legacy": {"labeled": True, "scenarioAccuracy": 0.7},
    # Weak everywhere.
    "weak": {"labeled": True, "scenarioAccuracy": 0.5,
             "diagnostics": {"auc": 0.47,
                             "heldOutCalibration": {"accuracy": 0.45,
                                                    "balancedAccuracy": 0.45}}},
}}


def test_legacy_floor_reads_transfer_only_and_advises():
    """The 2026-08-01 implicit calibrated-preference is REVERTED (review:
    the same frozen manifest could pass or fail depending on which engine
    version wrote the report). minScenarioAccuracy reads exactly what it
    always read; a one-sided transfer read gets an ADVISORY naming the
    declared alternative, never a silent metric switch."""
    gate = pspec.ValidateGate(min_scenario_accuracy=0.6)
    results = pspec.evaluate_validate_gate(
        gate, ["fair", "legacy"], FAIR_SHAPE_REPORT, None)
    by_concept = {r.detail.split("'")[1]: r for r in results}
    fair = by_concept["fair"]
    assert not fair.passed and fair.measured == 0.5
    assert "one-sided" in fair.detail and "accuracyFloor" in fair.detail
    legacy = by_concept["legacy"]
    assert legacy.passed and legacy.measured == 0.7


def test_declared_accuracy_floor_reads_its_metric_and_fails_closed():
    """The declared floor is manifest data: the metric has ONE address in
    the report, and an entry that cannot produce it FAILS — never a
    fallback (a one-class set with no calibration, an unlabeled file, or an
    older engine's report are gate failures, not quiet metric swaps)."""
    gate = pspec.ValidateGate(accuracy_floor_metric="auc",
                              accuracy_floor_minimum=0.7)
    results = pspec.evaluate_validate_gate(
        gate, ["fair", "legacy", "weak"], FAIR_SHAPE_REPORT, None)
    by_concept = {r.detail.split("'")[1]: r for r in results}
    assert by_concept["fair"].passed and by_concept["fair"].measured == 0.855
    assert by_concept["fair"].gate == "accuracyFloor"
    # Older report: no diagnostics → the declared metric is unmeasurable.
    legacy = by_concept["legacy"]
    assert not legacy.passed and "cannot produce" in legacy.detail
    assert not by_concept["weak"].passed

    calibrated = pspec.ValidateGate(
        accuracy_floor_metric="calibratedBalancedAccuracy",
        accuracy_floor_minimum=0.6)
    results = pspec.evaluate_validate_gate(
        calibrated, ["fair", "weak"], FAIR_SHAPE_REPORT, None)
    by_concept = {r.detail.split("'")[1]: r for r in results}
    assert by_concept["fair"].passed and by_concept["fair"].measured == 0.85
    assert not by_concept["weak"].passed


MULTI_DEPTH_REPORT = {"concepts": {
    # The depth-profile reality (2026-08-01): the same concept reads at
    # chance at one declared depth and strongly at another.
    "banded": {"labeled": True, "depths": [
        {"layer": 21, "scenarioAccuracy": 0.5,
         "diagnostics": {"auc": 0.49,
                         "heldOutCalibration": {"accuracy": 0.5,
                                                "balancedAccuracy": 0.5}}},
        {"layer": 40, "scenarioAccuracy": 0.78,
         "diagnostics": {"auc": 0.89,
                         "heldOutCalibration": {"accuracy": 0.78,
                                                "balancedAccuracy": 0.78}}},
    ]},
    # Unreadable at every declared depth.
    "flat": {"labeled": True, "depths": [
        {"layer": 21, "scenarioAccuracy": 0.5,
         "diagnostics": {"auc": 0.5,
                         "heldOutCalibration": {"accuracy": 0.5,
                                                "balancedAccuracy": 0.5}}},
        {"layer": 40, "scenarioAccuracy": 0.48,
         "diagnostics": {"auc": 0.45,
                         "heldOutCalibration": {"accuracy": 0.45,
                                                "balancedAccuracy": 0.45}}},
    ]},
}}


def test_multi_depth_floors_pass_on_any_declared_depth():
    """A depth list declares "readable somewhere in this band": the floor is
    met if SOME declared depth meets it, the measured value is the best
    depth's, and the detail names every depth so the profile is auditable."""
    gate = pspec.ValidateGate(min_scenario_accuracy=0.6)
    results = pspec.evaluate_validate_gate(
        gate, ["banded", "flat"], MULTI_DEPTH_REPORT, None)
    by_concept = {r.detail.split("'")[1]: r for r in results}
    banded = by_concept["banded"]
    assert banded.passed and banded.measured == 0.78
    assert "layer 40" in banded.detail and "L21=0.5" in banded.detail
    assert not by_concept["flat"].passed

    declared = pspec.ValidateGate(accuracy_floor_metric="auc",
                                  accuracy_floor_minimum=0.7)
    results = pspec.evaluate_validate_gate(
        declared, ["banded", "flat"], MULTI_DEPTH_REPORT, None)
    by_concept = {r.detail.split("'")[1]: r for r in results}
    assert by_concept["banded"].passed
    assert by_concept["banded"].measured == 0.89
    assert not by_concept["flat"].passed


def test_multi_depth_floor_fails_closed_when_no_depth_can_produce_the_metric():
    report = {"concepts": {"c": {"labeled": True, "depths": [
        {"layer": 21, "scenarioAccuracy": 0.9},
        {"layer": 40, "scenarioAccuracy": 0.9}]}}}
    gate = pspec.ValidateGate(accuracy_floor_metric="auc",
                              accuracy_floor_minimum=0.5)
    results = pspec.evaluate_validate_gate(gate, ["c"], report, None)
    assert len(results) == 1 and not results[0].passed
    assert "at any declared depth" in results[0].detail


def test_the_cosine_cap_applies_to_every_declared_depth_matrix():
    """Distinctness must hold at EVERY declared depth (unlike the accuracy
    floor's any-depth rule): two concepts collapsing into one direction at
    any depth the study declared it would measure is a breach."""
    clean = [["concept", "layer", "a", "b"],
             ["a", "21", "1.0000", "0.3000"],
             ["b", "21", "0.3000", "1.0000"]]
    collapsed = [["concept", "layer", "a", "b"],
                 ["a", "40", "1.0000", "0.9500"],
                 ["b", "40", "0.9500", "1.0000"]]
    gate = pspec.ValidateGate(max_cross_concept_cosine=0.8)
    results = [r for r in pspec.evaluate_validate_gate(
        gate, ["a", "b"], {"concepts": {}}, clean,
        extra_cosine_matrices=[("cosine-matrix-L40.csv", collapsed)])
        if r.gate == "maxCrossConceptCosine"]
    assert len(results) == 2
    assert results[0].passed  # layer 21 matrix is clean
    assert not results[1].passed  # layer 40 matrix breaches the cap
    assert "layer 40" in results[1].detail


def test_thresholds_require_real_json_numbers():
    """Review 2026-08-02 (P2): float("0.7") and float(True) both succeed,
    so string/boolean thresholds resolved here while the Swift mirror
    refused the identical block — the same frozen manifest verified on one
    engine and not the other."""
    for bad in ("0.7", True):
        with pytest.raises(ValueError, match="must be a number"):
            pspec.resolve_pipeline(
                {"gates": {"validate": {"minScenarioAccuracy": bad}}})
    with pytest.raises(ValueError, match="must be a number"):
        pspec.resolve_pipeline({"gates": {"validate": {
            "accuracyFloor": {"metric": "auc", "minimum": "0.7"}}}})
    # Real numbers still resolve, ints included.
    spec = pspec.resolve_pipeline(
        {"gates": {"validate": {"minScenarioAccuracy": 1}}})
    assert spec.validate_gate.min_scenario_accuracy == 1.0


def test_accuracy_floor_resolution_refusals():
    with pytest.raises(ValueError, match="declare exactly one"):
        pspec.resolve_pipeline({"gates": {"validate": {
            "minScenarioAccuracy": 0.6,
            "accuracyFloor": {"metric": "auc", "minimum": 0.7}}}})
    with pytest.raises(ValueError, match="unknown accuracyFloor metric"):
        pspec.resolve_pipeline({"gates": {"validate": {
            "accuracyFloor": {"metric": "vibes", "minimum": 0.7}}}})
    with pytest.raises(ValueError, match="must be an object"):
        pspec.resolve_pipeline({"gates": {"validate": {
            "accuracyFloor": 0.7}}})
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        pspec.resolve_pipeline({"gates": {"validate": {
            "accuracyFloor": {"metric": "auc", "minimum": 1.5}}}})
    spec = pspec.resolve_pipeline({"gates": {"validate": {
        "accuracyFloor": {"metric": "auc", "minimum": 0.7}}}})
    assert spec.validate_gate.accuracy_floor_metric == "auc"
    assert spec.validate_gate.accuracy_floor_minimum == 0.7
    assert spec.validate_gate.min_scenario_accuracy is None


def test_validate_gate_fails_closed_on_unmeasurable_inputs():
    gate = pspec.ValidateGate(min_scenario_accuracy=0.6,
                              max_cross_concept_cosine=0.8)
    report = {"concepts": {
        "ok": {"labeled": True, "scenarioAccuracy": 0.9},
        "unlabeled": {"labeled": False, "fractionAboveMidpoint": 0.9},
    }}
    rows = [["concept", "ok", "unlabeled", "ghost"],
            ["ok", "1.0000", "0.2000", "nan"],
            ["unlabeled", "0.2000", "1.0000", "0.1000"],
            ["ghost", "nan", "0.1000", "1.0000"]]
    results = pspec.evaluate_validate_gate(
        gate, ["ok", "unlabeled", "ghost"], report, rows)
    by_detail = {r.detail: r for r in results}
    assert any(r.passed and "'ok'" in r.detail for r in results)
    # Unlabeled: no scenarioAccuracy is a FAILURE with the remedy named.
    assert any(not r.passed and "unlabeled" in r.detail
               and "add 'expresses'" in r.detail for r in results)
    # Omitted from the report entirely: a failure, never a silent pass.
    assert any(not r.passed and "'ghost'" in r.detail
               and "no entry" in r.detail for r in results)
    # nan cosine: unmeasurable similarity fails the cap.
    assert any(not r.passed and "nan" in r.detail
               for r in results if r.gate == "maxCrossConceptCosine"), by_detail

    # Clean matrix over the cap fails numerically with measured/threshold.
    rows = [["concept", "a", "b"], ["a", "1.0", "0.9500"],
            ["b", "0.9500", "1.0"]]
    results = pspec.evaluate_validate_gate(
        pspec.ValidateGate(max_cross_concept_cosine=0.8),
        ["a", "b"], {"concepts": {}}, rows)
    (cos,) = [r for r in results if r.gate == "maxCrossConceptCosine"]
    assert not cos.passed and cos.measured == pytest.approx(0.95)
    assert cos.threshold == pytest.approx(0.8)

    # An empty/header-only matrix for a MULTI-concept study fails closed —
    # an unproduced measurement is not a passing one. A single-concept
    # study has no off-diagonal, so nothing fails.
    for empty in ([], [["concept"]], [["concept", "a", "b"]]):
        results = pspec.evaluate_validate_gate(
            pspec.ValidateGate(max_cross_concept_cosine=0.8),
            ["a", "b"], {"concepts": {}}, empty)
        (cos,) = [r for r in results if r.gate == "maxCrossConceptCosine"]
        assert not cos.passed and "no measurable" in cos.detail
    assert pspec.evaluate_validate_gate(
        pspec.ValidateGate(max_cross_concept_cosine=0.8),
        ["only"], {"concepts": {}}, []) == []


def test_sweep_gate_reads_the_recommendations_convention():
    gate = pspec.SweepGate()
    results = pspec.evaluate_sweep_gate(
        gate, ["a", "b", "c"],
        {"a": {"winningCell": {"layer": 2, "alpha": 4.0}},
         "b": "no cell passed the capability/coherence gates"})
    verdicts = {r.detail.split("'")[1]: r.passed for r in results}
    assert verdicts == {"a": True, "b": False, "c": False}
    # The sweep's own failure message travels into the detail.
    assert any("no cell passed" in r.detail for r in results)


# --- tasks.pipeline: the chain ------------------------------------------------

def test_full_chain_happy_path(tmp_path, monkeypatch):
    root = str(tmp_path)
    name = _workspace(root, pipeline={"gates": GATES})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)
    acquisitions: list = []
    with _acquire_counter(acquisitions) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             log=lambda *_: None)
    stage_calls = [c for c in calls if isinstance(c, str)]
    assert stage_calls == ["extract", "validate", "sweep", "promote:fear",
                           "run"]
    # ONE model load spans the whole chain.
    assert acquisitions == ["org/m"]
    ledger = json.load(open(os.path.join(out, "pipeline.json")))
    assert ledger["disposition"] == "completed"
    assert ledger["stages"] == list(pspec.DEFAULT_STAGES)
    results = ledger["stageResults"]
    assert all(results[s]["status"] == "completed"
               for s in pspec.DEFAULT_STAGES)
    # The FULL pin — path + hash + selection identity — and the promote
    # evidence came from THIS chain's sweep run, never ambient state.
    pin = results["promote"]["concepts"]["fear"]
    assert pin["path"].endswith("-agent.json") and pin["hash"] == "h"
    chain_sweep = os.path.basename(results["sweep"]["runDirectory"])
    assert pin["sweepRun"] == chain_sweep
    assert ("promote-sweep-run", chain_sweep) in calls
    assert os.path.isdir(results["run"]["runDirectory"])
    # The pipeline dir carries the canonical per-run stamp.
    config = json.load(open(os.path.join(out, "config.json")))
    assert config["runType"] == "pipeline"
    assert not os.path.exists(os.path.join(out, "pipeline-abort.json"))

    # Idempotent: a completed chain returns itself, running nothing.
    calls.clear()
    again = tasks.pipeline(name, root, pipeline_run_directory=out,
                           log=lambda *_: None)
    assert again == out and calls == []


def test_validate_gate_abort_stops_the_chain(tmp_path, monkeypatch):
    root = str(tmp_path)
    name = _workspace(root, pipeline={"gates": GATES})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls, accuracy=0.3)
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             log=lambda *_: None)
    # The chain STOPPED: no sweep, no promote, no run — and no exception
    # (an abort is a recorded determination, not a job failure).
    assert calls == ["extract", "validate"]
    abort = json.load(open(os.path.join(out, "pipeline-abort.json")))
    assert abort["stage"] == "validate"
    (gate,) = abort["gates"]
    assert gate["gate"] == "minScenarioAccuracy"
    assert gate["measured"] == pytest.approx(0.3)
    assert gate["threshold"] == pytest.approx(0.6)
    assert abort["evidenceRunDirectory"].endswith("-exp-chain-validate")
    ledger = json.load(open(os.path.join(out, "pipeline.json")))
    assert ledger["disposition"] == "aborted"
    # The validate STAGE itself completed — the science said stop.
    assert ledger["stageResults"]["validate"]["status"] == "completed"

    # A requeue of an aborted chain is idempotent — it must NOT re-run a
    # chain that deliberately stopped.
    calls.clear()
    again = tasks.pipeline(name, root, pipeline_run_directory=out,
                           log=lambda *_: None)
    assert again == out and calls == []


def test_missing_concept_fails_the_validate_gate(tmp_path, monkeypatch):
    # validate silently omits a concept whose validation.jsonl is missing —
    # the GATE must treat that as a failure, not a pass.
    root = str(tmp_path)
    name = _workspace(root, pipeline={"gates": GATES})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls, accuracy=None)
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             log=lambda *_: None)
    abort = json.load(open(os.path.join(out, "pipeline-abort.json")))
    assert abort["stage"] == "validate"
    assert "no entry" in abort["gates"][0]["detail"]
    assert calls == ["extract", "validate"]


def test_sweep_gate_abort_names_the_concept(tmp_path, monkeypatch):
    root = str(tmp_path)
    name = _workspace(root, pipeline={"gates": GATES})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls,
                 sweep_entry="control margin not met")
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             log=lambda *_: None)
    assert calls == ["extract", "validate", "sweep"]
    abort = json.load(open(os.path.join(out, "pipeline-abort.json")))
    assert abort["stage"] == "sweep"
    assert "'fear'" in abort["gates"][0]["detail"]
    assert "control margin not met" in abort["gates"][0]["detail"]


def test_promote_refusal_is_an_abort_not_a_crash(tmp_path, monkeypatch):
    # No sweep gate declared: a concept whose sweep selected no cell
    # surfaces at promote — as a recorded abort with promote's own words.
    root = str(tmp_path)
    name = _workspace(root, pipeline={})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)

    def refusing_promote(name, concept, **kwargs):
        raise promote_mod.PromoteError(
            f"the sweep selected no cell for '{concept}'")
    monkeypatch.setattr(promote_mod, "promote", refusing_promote)

    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             log=lambda *_: None)
    abort = json.load(open(os.path.join(out, "pipeline-abort.json")))
    assert abort["stage"] == "promote"
    assert abort["gates"][0]["gate"] == "promotable"
    assert "selected no cell" in abort["gates"][0]["detail"]
    assert "run" not in [c for c in calls if c == "run"]


def test_inline_judging_preflight_refuses_before_anything_runs(
        tmp_path, monkeypatch):
    # A judgeScore sweep that would DEFER (external judges, no credential)
    # refuses at pipeline start: no model load, no pipeline directory.
    root = str(tmp_path)
    name = _workspace(root, pipeline={})
    raw = es.load_raw(name, root)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         "Which response expresses more dread?\n")
    raw["judgeRubricFile"] = "prompts/rubrics/r.md"
    raw["judgeRubricHash"] = rubric_hash
    raw["judges"] = [{"name": "opus-judge", "kind": "claude"}]
    raw["sweep"] = {"layerFractions": [0.5], "alphas": [4.0],
                    "devPromptsFile": "prompts/dev/dev.jsonl",
                    "selection": {"objective": {"metric": "judgeScore"}}}
    es.save_raw(raw, root)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE",
                       os.path.join(root, "no-such-key"))
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)
    acquisitions: list = []
    with _acquire_counter(acquisitions) as provider:
        with pytest.raises(RuntimeError, match="INLINE judging"):
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    assert calls == [] and acquisitions == []
    runs = os.path.join(root, "runs")
    assert not os.path.isdir(runs) or not any(
        "pipeline" in d for d in os.listdir(runs))

    # With a credential the same chain proceeds inline.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with _acquire_counter(acquisitions) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             log=lambda *_: None)
    assert calls[0] == "extract"
    assert json.load(open(os.path.join(
        out, "pipeline.json")))["disposition"] == "completed"


def test_crash_requeue_skips_completed_stages(tmp_path, monkeypatch):
    root = str(tmp_path)
    name = _workspace(root, pipeline={"gates": GATES})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)
    boom = {"armed": True}
    real_sweep = tasks.sweep

    def dying_sweep(*args, **kwargs):
        if boom["armed"]:
            raise RuntimeError("node died mid-sweep")
        return real_sweep(*args, **kwargs)
    # _stage_fakes installed fake_sweep as tasks.sweep; wrap THAT.
    real_sweep = tasks.sweep
    monkeypatch.setattr(tasks, "sweep", dying_sweep)

    with _acquire_counter([]) as provider:
        with pytest.raises(RuntimeError, match="node died"):
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    runs = os.path.join(root, "runs")
    (pipeline_dir,) = [os.path.join(runs, d) for d in os.listdir(runs)
                       if "-pipeline" in d]
    ledger = json.load(open(os.path.join(pipeline_dir, "pipeline.json")))
    assert ledger["stageResults"]["extract"]["status"] == "completed"
    assert ledger["disposition"] is None
    assert calls == ["extract", "validate", "sweep"] or \
        calls == ["extract", "validate"]

    # Requeue: completed stages are SKIPPED (the fakes record calls);
    # the interrupted sweep re-runs from scratch and the chain finishes.
    boom["armed"] = False
    calls.clear()
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             pipeline_run_directory=pipeline_dir,
                             log=lambda *_: None)
    assert out == pipeline_dir
    assert [c for c in calls if isinstance(c, str)] == \
        ["sweep", "promote:fear", "run"]
    assert json.load(open(os.path.join(
        out, "pipeline.json")))["disposition"] == "completed"


def test_manifest_drift_at_resume_continues_loudly_and_stamps(
        tmp_path, monkeypatch):
    # POLICY (2026-08-05, Christian): a submitted chain must never die on
    # manifest/pinning drift — the old refusal threw away hours of completed
    # GPU work over edits the per-stage guards are equipped to judge. Drift
    # at resume now proceeds LOUDLY: warning logged, ledger stamped
    # epochDriftAtContinuation, and the chain runs on.
    import json
    root = str(tmp_path)
    name = _workspace(root, pipeline={})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)

    def dying_run(*args, **kwargs):
        raise RuntimeError("died in run")
    monkeypatch.setattr(tasks, "run", dying_run)
    with _acquire_counter([]) as provider:
        with pytest.raises(RuntimeError, match="died in run"):
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    runs = os.path.join(root, "runs")
    (pipeline_dir,) = [os.path.join(runs, d) for d in os.listdir(runs)
                       if "-pipeline" in d]

    raw = es.load_raw(name, root)
    raw["description"] = "edited between checkpoint and requeue"
    es.save_raw(raw, root)
    logs: list = []
    with _acquire_counter([]) as provider:
        # The chain continues past the drift and reaches the run stage
        # (whose fake still dies) — it no longer dies ON the drift.
        with pytest.raises(RuntimeError, match="died in run"):
            tasks.pipeline(name, root, model_provider=provider,
                           pipeline_run_directory=pipeline_dir,
                           log=logs.append)
    assert any("drifted" in line and "continuing" in line for line in logs)
    with open(os.path.join(pipeline_dir, "pipeline.json")) as handle:
        ledger = json.load(handle)
    (stamp,) = ledger["epochDriftAtContinuation"]
    assert stamp["ledgerHash"] != stamp["liveHash"]


def _die_in_analyze(monkeypatch, root, calls):
    """Stage fakes with an analyze that dies — the 2026-08-06 replication-run
    incident shape: run completed and merged, disposition null, the daemon
    gone before analyze."""
    _stage_fakes(monkeypatch, root, calls)

    def dying_analyze(name, r=None, source_run=None, **kwargs):
        raise RuntimeError("daemon restarted before analyze")
    monkeypatch.setattr(tasks, "analyze", dying_analyze)


def _incident_chain(tmp_path, monkeypatch, calls):
    """Run a run→analyze chain that dies in analyze; returns (root, name,
    pipeline_dir) with the ledger showing run completed, disposition null."""
    root = str(tmp_path)
    name = _workspace(root, pipeline={"stages": ["run", "analyze"]})
    _die_in_analyze(monkeypatch, root, calls)
    with _acquire_counter([]) as provider:
        with pytest.raises(RuntimeError, match="daemon restarted"):
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    runs = os.path.join(root, "runs")
    (pipeline_dir,) = [os.path.join(runs, d) for d in os.listdir(runs)
                       if "-pipeline" in d]
    ledger = json.load(open(os.path.join(pipeline_dir, "pipeline.json")))
    assert ledger["stageResults"]["run"]["status"] == "completed"
    assert ledger["disposition"] is None
    return root, name, pipeline_dir


def _heal_analyze(monkeypatch, root, calls):
    """Replace the dying analyze with a completing fake — the resume is a
    NEW daemon whose analyze works."""
    def fake_analyze(name, r=None, source_run=None, **kwargs):
        calls.append(("analyze", source_run))
        run_dir = os.path.join(root, "runs",
                               "20260718T000009000-exp-chain-analyze")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    monkeypatch.setattr(tasks, "analyze", fake_analyze)


def test_incident_resume_restores_stripped_pin_and_completes(
        tmp_path, monkeypatch):
    # THE incident end-to-end (the 2026-08-06 replication run): the chain
    # ran with a pinned revision, the app's manifest push overwrote the
    # server copy with a revision-LESS draft, and the old bare-hash guard
    # refused to finish a healthy chain's analyze. The self-pin repair
    # recognizes the ledger epoch minus its own pin, restores it into the
    # draft, and the resume completes — with NO drift stamp (this is the
    # same epoch, not tolerated drift).
    calls: list = []
    root, name, pipeline_dir = _incident_chain(tmp_path, monkeypatch, calls)
    _heal_analyze(monkeypatch, root, calls)

    raw = es.load_raw(name, root)
    assert raw["modelRevision"] == "abc"
    del raw["modelRevision"]  # the push that stripped the auto-pin
    es.save_raw(raw, root)

    calls.clear()
    logs: list = []
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             pipeline_run_directory=pipeline_dir,
                             log=logs.append)
    assert out == pipeline_dir
    # The live manifest is re-pinned, the resume ran ONLY analyze, and the
    # chain completed as the SAME epoch — no drift stamp.
    assert es.load_raw(name, root)["modelRevision"] == "abc"
    ledger = json.load(open(os.path.join(pipeline_dir, "pipeline.json")))
    assert [c for c in calls if isinstance(c, tuple)] == [
        ("analyze", ledger["stageResults"]["run"]["runDirectory"])]
    assert ledger["disposition"] == "completed"
    assert "epochDriftAtContinuation" not in ledger
    assert any("restored model revision pin" in line for line in logs)


def test_resume_drift_stamp_survives_stage_completion(tmp_path, monkeypatch):
    # A resume that proceeds past REAL drift stamps the ledger — and the
    # stamp must survive the continuation's own stage-done rewrites (the
    # in-memory ledger is what _stage_done writes; a disk-only stamp would
    # be clobbered by the first completed stage).
    calls: list = []
    root, name, pipeline_dir = _incident_chain(tmp_path, monkeypatch, calls)
    _heal_analyze(monkeypatch, root, calls)

    raw = es.load_raw(name, root)
    raw["judgeRubricFile"] = "prompts/rubrics/new.json"
    es.save_raw(raw, root)

    calls.clear()
    logs: list = []
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             pipeline_run_directory=pipeline_dir,
                             log=logs.append)
    assert out == pipeline_dir
    ledger = json.load(open(os.path.join(pipeline_dir, "pipeline.json")))
    assert ledger["disposition"] == "completed"
    (stamp,) = ledger["epochDriftAtContinuation"]
    assert stamp["ledgerHash"] != stamp["liveHash"]
    assert any("continuing under the LIVE manifest" in line
               for line in logs)
    # The listing surfaces the stamp for the app.
    row = tasks.list_pipeline_runs(name, root)[0]
    assert row["epochDriftAtContinuation"] == [stamp]


def test_resume_clears_parked_stamp(tmp_path, monkeypatch):
    # A parked chain that is successfully resumed is no longer parked —
    # the stamp must not outlive the state it describes.
    from steerlab_server.experiment import pipeline_reconcile
    calls: list = []
    root, name, pipeline_dir = _incident_chain(tmp_path, monkeypatch, calls)
    _heal_analyze(monkeypatch, root, calls)
    parked = pipeline_reconcile.park(pipeline_dir, reason="orphaned by test")
    assert parked is not None
    listed = tasks.list_pipeline_runs(name, root)
    assert listed[0]["parked"]["reason"] == "orphaned by test"

    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             pipeline_run_directory=pipeline_dir,
                             log=lambda *_: None)
    ledger = json.load(open(os.path.join(out, "pipeline.json")))
    assert ledger["disposition"] == "completed"
    assert "parked" not in ledger
    assert "parked" not in tasks.list_pipeline_runs(name, root)[0]


def test_list_pipeline_runs_cross_experiment_listing(tmp_path, monkeypatch):
    # list_pipeline_runs(None) lists EVERY experiment's chains and stamps
    # the owning experiment on each row — the Compute panel's
    # awaiting-import surface.
    calls: list = []
    root, name, pipeline_dir = _incident_chain(tmp_path, monkeypatch, calls)
    rows = tasks.list_pipeline_runs(None, root)
    assert [r["experiment"] for r in rows] == [name]
    assert rows[0]["run"] == os.path.basename(pipeline_dir)
    # The per-experiment listing carries the same key.
    assert tasks.list_pipeline_runs(name, root)[0]["experiment"] == name
    assert tasks.list_pipeline_runs("other", root) == []


def test_checkpoint_mid_run_resumes_record_level(tmp_path, monkeypatch):
    # The nested case that actually happens on the cluster: SIGTERM lands
    # mid-generation, the run stage parks record-level and raises
    # CheckpointRequested; the requeue skips completed stages AND passes
    # the recorded run dir back so generation resumes, not restarts.
    root = str(tmp_path)
    name = _workspace(root, pipeline={"gates": GATES})
    calls: list = []
    seen_run_dirs: list = []

    def parking_run(kwargs, run_dir):
        seen_run_dirs.append(kwargs.get("run_directory"))
        if kwargs.get("run_directory") is None:
            # First execution: park like the real run does — records
            # flushed, resume-state on disk, CheckpointRequested raised.
            _write(os.path.join(run_dir, "generations.jsonl"), '{"n": 1}\n')
            resume_mod.write_state(run_dir, run_id="r", verb="run",
                                   completed_records=1, reason="signal")
            raise resume_mod.CheckpointRequested(run_dir, "run", 1)
        # Resumed execution: complete.
        _write(os.path.join(run_dir, "report.json"), "{}")
        return run_dir

    _stage_fakes(monkeypatch, root, calls, run_behavior=parking_run)
    with _acquire_counter([]) as provider:
        with pytest.raises(resume_mod.CheckpointRequested):
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    runs = os.path.join(root, "runs")
    (pipeline_dir,) = [os.path.join(runs, d) for d in os.listdir(runs)
                       if "-pipeline" in d]
    ledger = json.load(open(os.path.join(pipeline_dir, "pipeline.json")))
    # The run stage is recorded STARTED with its dir — resumable, not done.
    assert ledger["stageResults"]["run"]["status"] == "started"
    parked = ledger["stageResults"]["run"]["runDirectory"]
    assert resume_mod.is_resumable(parked)
    assert ledger["disposition"] is None

    calls.clear()
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             pipeline_run_directory=pipeline_dir,
                             log=lambda *_: None)
    # Earlier stages skipped; the run stage received the PARKED dir back.
    assert calls == ["run"]
    assert seen_run_dirs == [None, parked]
    ledger = json.load(open(os.path.join(out, "pipeline.json")))
    assert ledger["disposition"] == "completed"
    assert ledger["stageResults"]["run"]["runDirectory"] == parked


def test_stage_list_change_refuses_resume(tmp_path, monkeypatch):
    root = str(tmp_path)
    name = _workspace(root, pipeline={"stages": ["extract", "validate"]})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)

    def dying_validate(*args, **kwargs):
        raise RuntimeError("died in validate")
    monkeypatch.setattr(tasks, "validate", dying_validate)
    with _acquire_counter([]) as provider:
        with pytest.raises(RuntimeError, match="died in validate"):
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    runs = os.path.join(root, "runs")
    (pipeline_dir,) = [os.path.join(runs, d) for d in os.listdir(runs)
                       if "-pipeline" in d]

    raw = es.load_raw(name, root)
    raw["pipeline"] = {"stages": ["extract", "run"]}
    es.save_raw(raw, root)
    with _acquire_counter([]) as provider:
        with pytest.raises(ValueError):
            tasks.pipeline(name, root, model_provider=provider,
                           pipeline_run_directory=pipeline_dir,
                           log=lambda *_: None)


def test_pipeline_requires_an_explicit_manifest_block(tmp_path, monkeypatch):
    # The chain is preregistered DATA, not a default: a manifest without a
    # pipeline block refuses with the remedy named (stage 5's abort UI is
    # not landed; an implicit gate-less five-stage chain is a footgun).
    root = str(tmp_path)
    name = _workspace(root, pipeline=None)
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)
    with _acquire_counter([]) as provider:
        with pytest.raises(RuntimeError, match="no pipeline block"):
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    assert calls == []


def test_evaluate_and_analyze_receive_the_chains_own_run_path(
        tmp_path, monkeypatch):
    # Finding (2026-07-18 second round): evaluate/analyze take the source
    # as a DIRECTORY PATH — a basename resolves against the process cwd
    # and breaks. The chain must hand its own run stage's full path.
    root = str(tmp_path)
    name = _workspace(root, pipeline={
        "stages": ["extract", "validate", "sweep", "promote", "run",
                   "evaluate", "analyze"],
        "gates": GATES})
    raw = es.load_raw(name, root)
    # A local judge panel: inline by construction (no credential needed).
    # The rubric pin joined the effective-evaluation rule 2026-07-22 —
    # judges + rubric file ARE the paired-judge declaration a chain with an
    # evaluate stage must carry.
    raw["judges"] = [{"name": "loc", "kind": "local"}]
    raw["judgeRubricFile"] = "prompts/rubrics/r.md"
    raw["judgeRubricHash"] = _write(
        os.path.join(root, "prompts", "rubrics", "r.md"),
        "Which response expresses more dread?\n")
    es.save_raw(raw, root)
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             log=lambda *_: None)
    ledger = json.load(open(os.path.join(out, "pipeline.json")))
    assert ledger["disposition"] == "completed"
    run_dir = ledger["stageResults"]["run"]["runDirectory"]
    sources = {c[0]: c[1] for c in calls
               if isinstance(c, tuple) and c[0] in ("evaluate", "analyze")}
    # Full, existing paths — the chain's OWN run, never a basename.
    assert sources == {"evaluate": run_dir, "analyze": run_dir}
    assert os.path.isabs(run_dir) and os.path.isdir(run_dir)
    assert ledger["stageResults"]["evaluate"]["status"] == "completed"
    assert ledger["stageResults"]["analyze"]["status"] == "completed"


def test_resume_preflights_only_remaining_stages(tmp_path, monkeypatch):
    # A resumed chain whose judged sweep already COMPLETED must not refuse
    # because the transient judge key was cleared after the sweep ran.
    root = str(tmp_path)
    name = _workspace(root, pipeline={})
    raw = es.load_raw(name, root)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         "Which response expresses more dread?\n")
    raw["judgeRubricFile"] = "prompts/rubrics/r.md"
    raw["judgeRubricHash"] = rubric_hash
    raw["judges"] = [{"name": "opus-judge", "kind": "claude"}]
    raw["sweep"] = {"layerFractions": [0.5], "alphas": [4.0],
                    "devPromptsFile": "prompts/dev/dev.jsonl",
                    "selection": {"objective": {"metric": "judgeScore"}}}
    es.save_raw(raw, root)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE",
                       os.path.join(root, "no-such-key"))
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)

    def dying_run(*args, **kwargs):
        raise RuntimeError("died in run")
    monkeypatch.setattr(tasks, "run", dying_run)
    with _acquire_counter([]) as provider:
        with pytest.raises(RuntimeError, match="died in run"):
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    runs = os.path.join(root, "runs")
    (pipeline_dir,) = [os.path.join(runs, d) for d in os.listdir(runs)
                       if "-pipeline" in d]

    # The key is CLEARED between checkpoint and requeue — the sweep is
    # already judged, so the resume must proceed, not refuse.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    def completing_run(name, prompts_file=None, r=None, dtype="auto",
                       device=None, **kwargs):
        calls.append("run")
        run_dir = os.path.join(runs, "20260718T000009000-exp-chain-run")
        os.makedirs(run_dir, exist_ok=True)
        _write(os.path.join(run_dir, "report.json"), "{}")
        return run_dir
    monkeypatch.setattr(tasks, "run", completing_run)
    calls.clear()
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             pipeline_run_directory=pipeline_dir,
                             log=lambda *_: None)
    assert calls == ["run"]
    assert json.load(open(os.path.join(
        out, "pipeline.json")))["disposition"] == "completed"


def test_promote_crash_window_adopts_the_minted_agent(tmp_path, monkeypatch):
    # Crash BETWEEN save_variant and the ledger write: the requeue must
    # adopt the already-minted agent — matched by the FULL selection
    # identity (experiment + epoch + THIS chain's sweep run + winning
    # cell), never by epoch alone, and never mint a duplicate.
    root = str(tmp_path)
    name = _workspace(root, pipeline={"gates": GATES})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)

    def dying_promote(name, concept, **kwargs):
        raise RuntimeError("died mid-promote")
    monkeypatch.setattr(promote_mod, "promote", dying_promote)
    with _acquire_counter([]) as provider:
        with pytest.raises(RuntimeError, match="died mid-promote"):
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    runs = os.path.join(root, "runs")
    (pipeline_dir,) = [os.path.join(runs, d) for d in os.listdir(runs)
                       if "-pipeline" in d]
    ledger = json.load(open(os.path.join(pipeline_dir, "pipeline.json")))
    # The crash left the stage marked mid-flight — the recovery gate.
    assert ledger["stageResults"]["promote"]["status"] == "started"
    chain_sweep = os.path.basename(
        ledger["stageResults"]["sweep"]["runDirectory"])

    def plant(dirname, *, sweep_run, layer=2, alpha=4.0):
        variant_dir = os.path.join(runs, dirname)
        artifact = os.path.join(
            variant_dir, dirname.split("-variant-")[1] + ".json")
        _write(artifact, json.dumps({
            "name": dirname.split("-variant-")[1], "baseModelID": "org/m",
            "injections": [{"concept": "fear", "layer": layer,
                            "alpha": alpha}],
            "promotion": {"experiment": name,
                          "experimentHash": ledger["experimentHash"],
                          "promotedBy": "criterion", "sweepRun": sweep_run,
                          "winningCell": {"layer": layer, "alpha": alpha}}}))
        return artifact

    # A decoy from an EARLIER pipeline: same epoch (frozen-manifest
    # scenario), same criterion — but a different sweep run and cell. The
    # tight identity must skip it.
    plant("20260718T000007000-variant-chain-fear-agent-old",
          sweep_run="20260101T000000000-exp-chain-sweep-old", alpha=8.0)
    artifact = plant("20260718T000008000-variant-chain-fear-agent",
                     sweep_run=chain_sweep)

    def refusing_promote(name, concept, **kwargs):
        raise AssertionError("promote must not re-mint an adopted agent")
    monkeypatch.setattr(promote_mod, "promote", refusing_promote)
    calls.clear()
    with _acquire_counter([]) as provider:
        out = tasks.pipeline(name, root, model_provider=provider,
                             pipeline_run_directory=pipeline_dir,
                             log=lambda *_: None)
    ledger = json.load(open(os.path.join(out, "pipeline.json")))
    assert ledger["disposition"] == "completed"
    pin = ledger["stageResults"]["promote"]["concepts"]["fear"]
    assert pin["path"] == artifact
    with open(artifact, "rb") as handle:
        assert pin["hash"] == hashlib.sha256(handle.read()).hexdigest()
    assert pin["sweepRun"] == chain_sweep
    assert pin["winningCell"] == {"layer": 2, "alpha": 4.0}


def test_fresh_pipeline_never_adopts(tmp_path, monkeypatch):
    # Adoption is RECOVERY-only: a fresh chain always mints, even when an
    # earlier chain's agent matches the manifest epoch (frozen manifests
    # keep one hash across many sweeps — epoch is not selection identity).
    root = str(tmp_path)
    name = _workspace(root, pipeline={"gates": GATES})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)
    with _acquire_counter([]) as provider:
        first = tasks.pipeline(name, root, model_provider=provider,
                               log=lambda *_: None)
    ledger = json.load(open(os.path.join(first, "pipeline.json")))
    runs = os.path.join(root, "runs")
    # An artifact fully matching the FIRST chain's identity sits in runs/.
    chain_sweep = os.path.basename(
        ledger["stageResults"]["sweep"]["runDirectory"])
    _write(os.path.join(runs, "20260718T000008000-variant-chain-fear-agent",
                        "chain-fear-agent.json"),
           json.dumps({
               "name": "chain-fear-agent", "baseModelID": "org/m",
               "injections": [{"concept": "fear", "layer": 2, "alpha": 4.0}],
               "promotion": {"experiment": name,
                             "experimentHash": ledger["experimentHash"],
                             "promotedBy": "criterion",
                             "sweepRun": chain_sweep,
                             "winningCell": {"layer": 2, "alpha": 4.0}}}))
    calls.clear()
    with _acquire_counter([]) as provider:
        tasks.pipeline(name, root, model_provider=provider,
                       log=lambda *_: None)
    # The SECOND chain minted its own agent — promote was called.
    assert "promote:fear" in calls


def test_pipeline_evidence_bundle_includes_stage_directories(tmp_path):
    # The pipeline dir holds only the ledger — the EVIDENCE lives in
    # sibling stage dirs. The evidence bundle must follow the ledger and
    # package them, or the Mac imports pointers to absolute cluster paths.
    from steerlab_server.experiment import bundles
    import tarfile
    root = str(tmp_path)
    runs = os.path.join(root, "runs")
    stage_dir = os.path.join(runs, "20260718T000001000-exp-chain-validate")
    _write(os.path.join(stage_dir, "validation-report.json"), "{}")
    variant_dir = os.path.join(runs, "20260718T000002000-variant-chain-fear-agent")
    variant_path = os.path.join(variant_dir, "chain-fear-agent.json")
    _write(variant_path, "{}")
    pipeline_dir = os.path.join(runs, "20260718T000003000-exp-chain-pipeline")
    _write(os.path.join(pipeline_dir, "pipeline.json"), json.dumps({
        "schema": 1, "experiment": "chain", "experimentHash": "0" * 64,
        "stages": ["validate", "promote"], "disposition": "completed",
        "stageResults": {
            "validate": {"status": "completed", "runDirectory": stage_dir},
            "promote": {"status": "completed",
                        "concepts": {"fear": variant_path}}}}))

    meta = bundles.package_evidence(pipeline_dir)
    with tarfile.open(meta["bundlePath"], "r:gz") as tar:
        members = set(tar.getnames())
    assert f"runs/{os.path.basename(pipeline_dir)}/pipeline.json" in members
    assert (f"runs/{os.path.basename(stage_dir)}/validation-report.json"
            in members)
    assert (f"runs/{os.path.basename(variant_dir)}/chain-fear-agent.json"
            in members)
    assert sorted(meta["pipelineStageDirectories"]) == sorted(
        [os.path.basename(stage_dir), os.path.basename(variant_dir)])
    assert meta["evidenceComplete"] is True
    assert "missingEvidence" not in meta


def _pipeline_dir_with_ledger(runs, *, disposition, references):
    pipeline_dir = os.path.join(runs, "20260718T000005000-exp-chain-pipeline")
    _write(os.path.join(pipeline_dir, "pipeline.json"), json.dumps({
        "schema": 1, "experiment": "chain", "experimentHash": "0" * 64,
        "stages": ["validate"], "disposition": disposition,
        "stageResults": {"validate": {"status": "completed",
                                      "runDirectory": references[0]}}}))
    return pipeline_dir


def test_completed_pipeline_refuses_to_package_missing_evidence(tmp_path):
    # A COMPLETED chain with an unpackageable reference must refuse — a
    # nominally successful bundle silently missing its evidence is worse.
    from steerlab_server.experiment import bundles
    runs = os.path.join(str(tmp_path), "runs")
    gone = os.path.join(runs, "20260718T000001000-exp-chain-validate")
    pipeline_dir = _pipeline_dir_with_ledger(
        runs, disposition="completed", references=[gone])
    with pytest.raises(bundles.BundleError, match="refusing to package"):
        bundles.package_evidence(pipeline_dir)

    # A reference OUTSIDE the runs root refuses too (tampered-ledger
    # containment), even when the directory exists and is readable.
    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok=True)
    _write(os.path.join(outside, "loot.txt"), "x")
    pipeline_dir2 = os.path.join(runs, "20260718T000006000-exp-chain-pipeline")
    _write(os.path.join(pipeline_dir2, "pipeline.json"), json.dumps({
        "schema": 1, "experiment": "chain", "experimentHash": "0" * 64,
        "stages": ["validate"], "disposition": "completed",
        "stageResults": {"validate": {"status": "completed",
                                      "runDirectory": outside}}}))
    with pytest.raises(bundles.BundleError, match="outside the runs root"):
        bundles.package_evidence(pipeline_dir2)


def test_aborted_pipeline_packages_honestly_incomplete(tmp_path):
    # Aborted/interrupted chains package what exists and NAME what is
    # missing — evidenceComplete: false, never a silent omission.
    import tarfile
    from steerlab_server.experiment import bundles
    runs = os.path.join(str(tmp_path), "runs")
    gone = os.path.join(runs, "20260718T000001000-exp-chain-validate")
    pipeline_dir = _pipeline_dir_with_ledger(
        runs, disposition="aborted", references=[gone])
    meta = bundles.package_evidence(pipeline_dir)
    assert meta["evidenceComplete"] is False
    (entry,) = meta["missingEvidence"]
    assert "stage 'validate'" in entry and "not found" in entry
    with tarfile.open(meta["bundlePath"], "r:gz") as tar:
        assert (f"runs/{os.path.basename(pipeline_dir)}/pipeline.json"
                in tar.getnames())


def test_pipeline_is_a_submittable_verb():
    from steerlab_server.api.submissions import VALID_STUDY_VERBS
    assert "pipeline" in VALID_STUDY_VERBS


# --- stage 5: preregistered chain (verify/advisories), listing, portability --

def test_malformed_pipeline_block_is_a_verify_violation(tmp_path):
    # Stage 5: the chain is preregistered data — a malformed declaration
    # surfaces at verify/freeze time, not at first cluster submission.
    root = str(tmp_path)
    name = _workspace(root, pipeline={"stages": ["run", "extract"]})
    violations = Manifest.load(name, root).verify(root)
    assert any("pipeline block invalid" in v and "canonical order" in v
               for v in violations)
    raw = es.load_raw(name, root)
    raw["pipeline"] = {"gates": {"validate": {"minScenarioAccuracy": 1.5}}}
    es.save_raw(raw, root)
    violations = Manifest.load(name, root).verify(root)
    assert any("pipeline block invalid" in v for v in violations)
    # A well-formed block verifies clean.
    raw["pipeline"] = {"gates": GATES}
    es.save_raw(raw, root)
    assert not any("pipeline block" in v
                   for v in Manifest.load(name, root).verify(root))


def test_gateless_pipeline_freeze_advisory(tmp_path):
    from steerlab_server.experiment.experiment_store import freeze_advisories
    root = str(tmp_path)
    name = _workspace(root, pipeline={})
    advisories = freeze_advisories(es.load_raw(name, root), root)
    assert any("pipeline declares no gates" in a for a in advisories)
    raw = es.load_raw(name, root)
    raw["pipeline"] = {"gates": GATES}
    es.save_raw(raw, root)
    assert not any("pipeline declares no gates" in a
                   for a in freeze_advisories(es.load_raw(name, root), root))


def test_list_pipeline_runs_summarizes_states(tmp_path, monkeypatch):
    # The awaiting/aborted affordance's data source: per-stage status,
    # disposition, abort details, and promoted-agent pins — tolerant of
    # schema-1 ledgers (display is not resume).
    root = str(tmp_path)
    name = _workspace(root, pipeline={"gates": GATES})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)
    with _acquire_counter([]) as provider:
        tasks.pipeline(name, root, model_provider=provider,
                       log=lambda *_: None)
    # An aborted chain beside it.
    _stage_fakes(monkeypatch, root, calls, accuracy=0.3)
    with _acquire_counter([]) as provider:
        tasks.pipeline(name, root, model_provider=provider,
                       log=lambda *_: None)
    # A legacy schema-1 ledger is listable.
    _write(os.path.join(root, "runs",
                        "20250101T000000000-exp-chain-pipeline",
                        "pipeline.json"),
           json.dumps({"schema": 1, "experiment": name,
                       "stages": ["extract"], "disposition": "completed",
                       "stageResults": {"extract": {
                           "status": "completed",
                           "runDirectory": "/gone/extract"}}}))

    rows = tasks.list_pipeline_runs(name, root)
    assert len(rows) == 3
    by_disposition = {}
    for row in rows:
        by_disposition.setdefault(
            (row["disposition"], row["schema"]), row)
    completed = by_disposition[("completed", 2)]
    assert [s["stage"] for s in completed["stages"]] == \
        list(pspec.DEFAULT_STAGES)
    assert all(s["status"] == "completed" for s in completed["stages"])
    # Draft chains are stamped as such (exploratory), and every ledger
    # write carries updatedAt.
    assert completed["manifestStatus"] == "draft"
    assert completed["updatedAt"]
    assert completed["promotedAgents"]["fear"]["winningCell"] == {
        "layer": 2, "alpha": 4.0}
    aborted = by_disposition[("aborted", 2)]
    assert aborted["abort"]["stage"] == "validate"
    assert aborted["abort"]["gates"][0]["gate"] == "minScenarioAccuracy"
    assert aborted["abort"]["evidenceRunID"]
    statuses = {s["stage"]: s["status"] for s in aborted["stages"]}
    assert statuses["sweep"] == "pending"  # never ran — the chain stopped
    legacy = by_disposition[("completed", 1)]
    assert legacy["stages"][0]["runID"] == "extract"


def test_evidence_bundle_carries_a_portable_ledger(tmp_path):
    # Stage 5 (third-round finding): the on-disk ledger's absolute cluster
    # paths are useless after import — the bundle carries a PORTABLE
    # projection keyed by imported run IDs, with the original paths kept
    # as provenance.
    import tarfile
    from steerlab_server.experiment import bundles
    root = str(tmp_path)
    runs = os.path.join(root, "runs")
    stage_dir = os.path.join(runs, "20260718T000001000-exp-chain-validate")
    _write(os.path.join(stage_dir, "validation-report.json"), "{}")
    variant_dir = os.path.join(runs,
                               "20260718T000002000-variant-chain-fear-agent")
    variant_path = os.path.join(variant_dir, "chain-fear-agent.json")
    _write(variant_path, "{}")
    pipeline_dir = os.path.join(runs, "20260718T000003000-exp-chain-pipeline")
    _write(os.path.join(pipeline_dir, "pipeline.json"), json.dumps({
        "schema": 2, "experiment": "chain", "experimentHash": "0" * 64,
        "stages": ["validate", "promote"], "disposition": "completed",
        "stageResults": {
            "validate": {"status": "completed", "runDirectory": stage_dir},
            "promote": {"status": "completed", "concepts": {"fear": {
                "path": variant_path, "hash": "f" * 64,
                "sweepRun": "20260718T000000000-exp-chain-sweep",
                "winningCell": {"layer": 2, "alpha": 4.0}}}}}}))

    meta = bundles.package_evidence(pipeline_dir)
    assert meta["pipelinePortable"] == "steerlab-pipeline.json"
    with tarfile.open(meta["bundlePath"], "r:gz") as tar:
        portable = json.load(tar.extractfile("steerlab-pipeline.json"))
    assert portable["kind"] == "pipelinePortable"
    assert portable["stageRuns"]["validate"] == os.path.basename(stage_dir)
    agent = portable["promotedAgents"]["fear"]
    assert agent["runID"] == os.path.basename(variant_dir)
    assert agent["artifact"] == "chain-fear-agent.json"
    assert agent["winningCell"] == {"layer": 2, "alpha": 4.0}
    # Original paths survive as provenance, never as the reference.
    assert portable["originalPaths"]["validate"] == stage_dir
    assert portable["originalPaths"]["agent:fear"] == variant_path
    # The portable member is hash-pinned like everything else.
    with tarfile.open(meta["bundlePath"], "r:gz") as tar:
        blob = tar.extractfile("steerlab-pipeline.json").read()
    assert meta["pipelinePortableSha256"] == hashlib.sha256(blob).hexdigest()


def test_pipeline_evidence_round_trips_through_import(tmp_path):
    # Sixth round, the essential test: package → import must land EVERY
    # stage directory and promoted agent in the target workspace — the
    # portable ledger is bundle meta (verified, never extracted), and an
    # abort's evidence reference is a run ID after import.
    from steerlab_server.experiment import bundles
    root = str(tmp_path / "cluster")
    runs = os.path.join(root, "runs")
    stage_dir = os.path.join(runs, "20260718T000001000-exp-chain-validate")
    _write(os.path.join(stage_dir, "validation-report.json"), "{}")
    variant_dir = os.path.join(runs,
                               "20260718T000002000-variant-chain-fear-agent")
    _write(os.path.join(variant_dir, "chain-fear-agent.json"), "{}")
    pipeline_dir = os.path.join(runs, "20260718T000003000-exp-chain-pipeline")
    _write(os.path.join(pipeline_dir, "pipeline.json"), json.dumps({
        "schema": 2, "experiment": "chain", "experimentHash": "0" * 64,
        "stages": ["validate", "promote"], "disposition": "aborted",
        "abort": {"stage": "validate", "gates": [
            {"passed": False, "gate": "minScenarioAccuracy",
             "detail": "too low"}],
            "evidenceRunDirectory": stage_dir},
        "stageResults": {
            "validate": {"status": "completed", "runDirectory": stage_dir},
            "promote": {"status": "completed", "concepts": {"fear": {
                "path": os.path.join(variant_dir, "chain-fear-agent.json"),
                "hash": "f" * 64, "sweepRun": "s",
                "winningCell": {"layer": 2, "alpha": 4.0}}}}}}))

    meta = bundles.package_evidence(pipeline_dir)
    # The portable abort references a run ID; the cluster path is
    # provenance only.
    with tarfile_open_portable(meta["bundlePath"]) as portable:
        assert portable["abort"]["evidenceRunID"] == \
            os.path.basename(stage_dir)
        assert "evidenceRunDirectory" not in portable["abort"]
        assert portable["originalPaths"]["abort"] == stage_dir

    target = str(tmp_path / "mac")
    os.makedirs(target, exist_ok=True)
    result = bundles.import_bundle(meta["bundlePath"], target_root=target)
    imported = set(result["extracted"])
    pipeline_id = os.path.basename(pipeline_dir)
    assert f"runs/{pipeline_id}/pipeline.json" in imported
    assert (f"runs/{os.path.basename(stage_dir)}/validation-report.json"
            in imported)
    assert (f"runs/{os.path.basename(variant_dir)}/chain-fear-agent.json"
            in imported)
    for member in imported:
        assert os.path.isfile(os.path.join(target, member))
    # The portable ledger is RETAINED inside the imported pipeline dir
    # (seventh round: retained, then consumed by local readers) — never
    # extracted loose into the workspace root.
    assert f"runs/{pipeline_id}/pipeline-portable.json" in imported
    retained = json.load(open(os.path.join(
        target, "runs", pipeline_id, "pipeline-portable.json")))
    assert retained["kind"] == "pipelinePortable"
    assert not os.path.exists(os.path.join(target, "steerlab-pipeline.json"))

    # A portable member WITHOUT its hash pin refuses (unverifiable).
    import io
    import tarfile as tf
    unpinned = str(tmp_path / "unpinned.tar.gz")
    with tf.open(meta["bundlePath"], "r:gz") as src, \
            tf.open(unpinned, "w:gz") as dst:
        for member in src.getmembers():
            payload = src.extractfile(member)
            data = payload.read() if payload else b""
            if member.name == "steerlab-evidence.json":
                stripped = json.loads(data.decode("utf-8"))
                stripped.pop("pipelinePortableSha256", None)
                data = json.dumps(stripped).encode("utf-8")
                member.size = len(data)
            dst.addfile(member, io.BytesIO(data))
    with pytest.raises(bundles.BundleError, match="no hash pin"):
        bundles.import_bundle(unpinned,
                              target_root=str(tmp_path / "mac3"))

    # A tampered portable member refuses the whole import.
    import io
    import tarfile as tf

    def rewrite(name, *, drop=None, alter=None):
        path = str(tmp_path / name)
        with tf.open(meta["bundlePath"], "r:gz") as src, \
                tf.open(path, "w:gz") as dst:
            for member in src.getmembers():
                if member.name == drop:
                    continue
                payload = src.extractfile(member)
                data = payload.read() if payload else b""
                if member.name == alter:
                    data = data + b" "
                    member.size = len(data)
                dst.addfile(member, io.BytesIO(data))
        return path

    target2 = str(tmp_path / "mac2")
    os.makedirs(target2, exist_ok=True)
    with pytest.raises(bundles.BundleError, match="hash pin"):
        bundles.import_bundle(
            rewrite("tampered.tar.gz", alter="steerlab-pipeline.json"),
            target_root=target2)

    # DELETION refuses too (eighth round — per-member checks verify what
    # is present; the closure check catches what was REMOVED): a missing
    # declared entry, and a missing portable ledger whose pin is stamped.
    with pytest.raises(bundles.BundleError, match="missing declared"):
        bundles.import_bundle(
            rewrite("gutted.tar.gz",
                    drop=f"runs/{os.path.basename(stage_dir)}/"
                         "validation-report.json"),
            target_root=str(tmp_path / "mac4"))
    with pytest.raises(bundles.BundleError, match="does not carry"):
        bundles.import_bundle(
            rewrite("no-portable.tar.gz", drop="steerlab-pipeline.json"),
            target_root=str(tmp_path / "mac5"))


def tarfile_open_portable(bundle_path):
    """Context manager yielding the parsed portable ledger from a bundle."""
    import contextlib
    import tarfile as tf

    @contextlib.contextmanager
    def _open():
        with tf.open(bundle_path, "r:gz") as tar:
            yield json.load(tar.extractfile("steerlab-pipeline.json"))
    return _open()


# --- stage 4: forward-referenced conditions -----------------------------------

def _forward_ref_workspace(root, *, with_evidence=True, cell=None):
    """A workspace whose manifest declares a forward-referenced variant
    condition, with (optionally) sweep selection evidence for 'fear'."""
    name = _workspace(root, pipeline={"gates": GATES})
    raw = es.load_raw(name, root)
    raw["variantConditions"] = [
        {"name": "fear-agent", "fromPromotion": {"concept": "fear"}}]
    es.save_raw(raw, root)
    sweep_run = None
    if with_evidence:
        runs = os.path.join(root, "runs")
        sweep_dir = os.path.join(runs, "20260718T000001000-exp-chain-sweep")
        sweep_run = os.path.basename(sweep_dir)
        _write(os.path.join(sweep_dir, "sweep.csv"), "concept,layer,alpha\n")
        _write(os.path.join(sweep_dir, "recommendations.json"), json.dumps({
            "fear": {"sweepRun": sweep_run,
                     "winningCell": cell or {"layer": 2, "alpha": 4.0},
                     "criterion": {"metric": "markerDensity"}}}))
    return name, sweep_run


def _plant_agent(root, name, *, sweep_run, live_hash, layer=2, alpha=4.0):
    runs = os.path.join(root, "runs")
    variant_dir = os.path.join(runs,
                               "20260718T000002000-variant-chain-fear-agent")
    artifact = os.path.join(variant_dir, "chain-fear-agent.json")
    _write(artifact, json.dumps({
        "name": "chain-fear-agent", "baseModelID": "org/m",
        "baseRevision": "abc",
        "injections": [{"concept": "fear", "layer": layer, "alpha": alpha,
                        "vectorArtifactID": "runs/x/fear"}],
        "promptMode": "chatAssistant", "temperature": 0,
        "promotion": {"experiment": name, "experimentHash": live_hash,
                      "promotedBy": "criterion", "sweepRun": sweep_run,
                      "winningCell": {"layer": layer, "alpha": alpha}}}))
    return artifact


def test_forward_ref_declaration_rules_in_verify(tmp_path):
    root = str(tmp_path)
    name, _ = _forward_ref_workspace(root, with_evidence=False)
    manifest = Manifest.load(name, root)
    assert not [v for v in manifest.verify(root) if "fear-agent" in v]

    # Unattached concept, missing concept, and both-identities each flag.
    raw = es.load_raw(name, root)
    raw["variantConditions"] = [
        {"name": "ghost-agent", "fromPromotion": {"concept": "ghost"}},
        {"name": "nameless", "fromPromotion": {}},
        {"name": "double", "fromPromotion": {"concept": "fear"},
         "artifactPath": "runs/x/agent.json", "artifactHash": "0" * 64}]
    es.save_raw(raw, root)
    violations = Manifest.load(name, root).verify(root)
    assert any("ghost-agent" in v and "not attached" in v
               for v in violations)
    assert any("nameless" in v and "names no concept" in v
               for v in violations)
    assert any("double" in v and "BOTH" in v for v in violations)


def test_forward_ref_resolves_against_the_promotion_birth_certificate(
        tmp_path):
    root = str(tmp_path)
    name, sweep_run = _forward_ref_workspace(root)
    manifest = Manifest.load(name, root)
    live_hash = manifest.content_hash()
    artifact = _plant_agent(root, name, sweep_run=sweep_run,
                            live_hash=live_hash)
    run_dir = os.path.join(root, "runs", "20260718T000003000-exp-chain-run")
    os.makedirs(run_dir, exist_ok=True)

    tasks._resolve_manifest_forward_refs(manifest, run_dir, root,
                                         lambda *_: None)
    (vc,) = manifest.variant_conditions
    assert vc.from_promotion is None
    assert vc.artifact["name"] == "chain-fear-agent"
    with open(artifact, "rb") as handle:
        expected_hash = hashlib.sha256(handle.read()).hexdigest()
    assert vc.artifact_hash == expected_hash
    record = json.load(open(os.path.join(run_dir,
                                         "forward-resolutions.json")))
    (resolution,) = record["resolutions"]
    assert resolution["concept"] == "fear"
    assert resolution["artifactHash"] == expected_hash
    assert resolution["sweepRun"] == sweep_run
    assert resolution["experimentHash"] == live_hash


def test_forward_ref_refuses_without_a_matching_agent(tmp_path):
    root = str(tmp_path)
    # No sweep evidence at all → the reference cannot resolve.
    name, _ = _forward_ref_workspace(root, with_evidence=False)
    manifest = Manifest.load(name, root)
    run_dir = os.path.join(root, "runs", "20260718T000003000-exp-chain-run")
    os.makedirs(run_dir, exist_ok=True)
    with pytest.raises(ValueError, match="no sweep selection evidence"):
        tasks._resolve_manifest_forward_refs(manifest, run_dir, root,
                                             lambda *_: None)

    # Evidence exists but the only agent was promoted from a DIFFERENT
    # cell — the study must refuse, never run a mismatched arm.
    root2 = str(tmp_path / "two")
    name2, sweep_run2 = _forward_ref_workspace(root2)
    manifest2 = Manifest.load(name2, root2)
    _plant_agent(root2, name2, sweep_run=sweep_run2,
                 live_hash=manifest2.content_hash(), alpha=8.0)
    run_dir2 = os.path.join(root2, "runs",
                            "20260718T000003000-exp-chain-run")
    os.makedirs(run_dir2, exist_ok=True)
    with pytest.raises(ValueError, match="no promoted agent matches"):
        tasks._resolve_manifest_forward_refs(manifest2, run_dir2, root2,
                                             lambda *_: None)


def test_promote_from_named_sweep_run_ignores_stale_recommendation(
        tmp_path):
    # Finding (fourth round): a frozen manifest keeps a stale
    # `<concept>-recommended` condition forever, and promote preferred it
    # over fresh sweep output. With sweep_run named, the evidence source
    # is pinned to THAT run — the stale condition is never consulted.
    root = str(tmp_path)
    name = _workspace(root, pipeline={})
    manifest = Manifest.load(name, root)
    stimulus_hash = manifest.concepts[0].stimulus_set_hash
    # A recipe-matching extraction artifact so real promote can mint.
    from steerlab_server.steering.vector_store import SUBSTRATE
    extract_dir = os.path.join(root, "runs", "20260718T000000000-extract")
    os.makedirs(extract_dir, exist_ok=True)
    with open(os.path.join(extract_dir, "fear.safetensors"), "wb") as handle:
        handle.write(b"weights")
    _write(os.path.join(extract_dir, "fear.json"), json.dumps({
        "modelID": "org/m", "concept": "fear", "layerCount": 4,
        "hiddenSize": 2, "stimulusSetHash": stimulus_hash,
        "extractionMethod": "meanDifference", "revision": "abc",
        "normsPerLayer": [1.0] * 4, "residualNormPerLayer": [1.0] * 4,
        "substrate": SUBSTRATE, "readingPosition": "last token",
        "neutralProjection": "none",
        "residualNormSource": "extraction-stimuli"}))
    # The STALE recommendation in the manifest: old sweep, old cell.
    raw = es.load_raw(name, root)
    raw["conditions"] = [
        {"name": "fear-recommended",
         "slots": [{"concept": "fear", "layer": 9, "alpha": 9.0}],
         "selection": {"sweepRun": "20260101T000000000-exp-chain-sweep",
                       "winningCell": {"layer": 9, "alpha": 9.0}}}]
    es.save_raw(raw, root)
    # This chain's sweep selected a DIFFERENT cell.
    sweep_dir = os.path.join(root, "runs",
                             "20260718T000001000-exp-chain-sweep")
    sweep_run = os.path.basename(sweep_dir)
    _write(os.path.join(sweep_dir, "sweep.csv"), "concept,layer,alpha\n")
    _write(os.path.join(sweep_dir, "recommendations.json"), json.dumps({
        "fear": {"sweepRun": sweep_run,
                 "winningCell": {"layer": 2, "alpha": 4.0},
                 "criterion": {"metric": "markerDensity"}}}))

    outcome = promote_mod.promote(name, "fear", root=root,
                                  log=lambda *_: None, sweep_run=sweep_run)
    (injection,) = outcome["variant"]["injections"]
    assert (injection["layer"], injection["alpha"]) == (2, 4.0)
    promotion = outcome["variant"]["promotion"]
    assert promotion["sweepRun"] == sweep_run
    assert promotion["winningCell"] == {"layer": 2, "alpha": 4.0}
    # WITHOUT the pin, the stale manifest condition used to WIN — the exact
    # behavior the chain must never inherit. Since the promotion gate
    # (review 2026-08-03 round 3, P2) the condition's named run must exist
    # and still match, so the stale condition now refuses outright.
    with pytest.raises(promote_mod.PromoteError,
                       match="which is not in runs/"):
        promote_mod.promote(name, "fear", root=root, log=lambda *_: None)
    # A failure entry refuses; naming a run with no entry refuses.
    _write(os.path.join(sweep_dir, "recommendations.json"), json.dumps({
        "fear": "no cell passed the capability/coherence gates"}))
    with pytest.raises(promote_mod.PromoteError, match="selected no cell"):
        promote_mod.promote(name, "fear", root=root, log=lambda *_: None,
                            sweep_run=sweep_run)
    with pytest.raises(promote_mod.PromoteError, match="no readable"):
        promote_mod.promote(name, "fear", root=root, log=lambda *_: None,
                            sweep_run="20269999T000000000-exp-chain-sweep")


def test_ledger_pins_beat_ambient_catalog_state(tmp_path):
    # Finding (fourth round): the run must use the EXACT artifact this
    # chain's promote stage recorded — a newer catalog agent matching the
    # same evidence must not shift resolution.
    root = str(tmp_path)
    name, sweep_run = _forward_ref_workspace(root)
    manifest = Manifest.load(name, root)
    live_hash = manifest.content_hash()
    pinned = _plant_agent(root, name, sweep_run=sweep_run,
                          live_hash=live_hash)
    with open(pinned, "rb") as handle:
        pinned_hash = hashlib.sha256(handle.read()).hexdigest()
    # A NEWER catalog agent with the same birth certificate (a concurrent
    # re-promotion): catalog scan would pick it — the ledger pin must not.
    runs = os.path.join(root, "runs")
    newer_dir = os.path.join(runs, "20260719T000000000-variant-chain-fear-agent")
    _write(os.path.join(newer_dir, "chain-fear-agent.json"), json.dumps(
        json.loads(open(pinned).read()) | {"createdAt": "later"}))

    run_dir = os.path.join(runs, "20260718T000003000-exp-chain-run")
    os.makedirs(run_dir, exist_ok=True)
    pins = {"fear": {"path": pinned, "hash": pinned_hash,
                     "sweepRun": sweep_run,
                     "winningCell": {"layer": 2, "alpha": 4.0}}}
    tasks._resolve_manifest_forward_refs(manifest, run_dir, root,
                                         lambda *_: None, ledger_pins=pins)
    (vc,) = manifest.variant_conditions
    assert vc.artifact_hash == pinned_hash
    record = json.load(open(os.path.join(run_dir,
                                         "forward-resolutions.json")))
    (resolution,) = record["resolutions"]
    assert resolution["resolvedFrom"] == "ledger"
    assert os.path.join(root, resolution["artifactPath"]) == pinned


def test_resume_reuses_the_recorded_resolution_never_recatalogs(tmp_path):
    # Finding (fourth round): a resumed run must use the EXACT artifacts
    # its forward-resolutions.json recorded — early generations and
    # resumed generations must be the same arm, and the record must never
    # silently disagree with what ran.
    root = str(tmp_path)
    name, sweep_run = _forward_ref_workspace(root)
    manifest = Manifest.load(name, root)
    live_hash = manifest.content_hash()
    pinned = _plant_agent(root, name, sweep_run=sweep_run,
                          live_hash=live_hash)
    runs = os.path.join(root, "runs")
    run_dir = os.path.join(runs, "20260718T000003000-exp-chain-run")
    os.makedirs(run_dir, exist_ok=True)
    tasks._resolve_manifest_forward_refs(manifest, run_dir, root,
                                         lambda *_: None)
    record = json.load(open(os.path.join(run_dir,
                                         "forward-resolutions.json")))
    (resolution,) = record["resolutions"]

    # Newer evidence appears between checkpoint and resume: a fresh sweep
    # + agent that catalog resolution would now prefer.
    newer_sweep = os.path.join(runs, "20260719T000001000-exp-chain-sweep")
    _write(os.path.join(newer_sweep, "sweep.csv"), "concept,layer,alpha\n")
    _write(os.path.join(newer_sweep, "recommendations.json"), json.dumps({
        "fear": {"sweepRun": os.path.basename(newer_sweep),
                 "winningCell": {"layer": 3, "alpha": 8.0}}}))

    resumed = Manifest.load(name, root)
    tasks._resolve_manifest_forward_refs(resumed, run_dir, root,
                                         lambda *_: None)
    (vc,) = resumed.variant_conditions
    # The RECORDED agent, not the newer evidence's.
    assert vc.artifact_hash == resolution["artifactHash"]
    assert vc.artifact["injections"][0]["layer"] == 2

    # A tampered pinned artifact breaks the evidence chain — refuse.
    with open(pinned, "a", encoding="utf-8") as handle:
        handle.write(" ")
    tampered = Manifest.load(name, root)
    with pytest.raises(ValueError, match="changed since it was pinned"):
        tasks._resolve_manifest_forward_refs(tampered, run_dir, root,
                                             lambda *_: None)


def test_resolution_verifies_the_agent_embodies_its_cell(tmp_path):
    # Finding (fourth round): the birth certificate's winningCell is a
    # CLAIM — the artifact's concrete injection must equal it, on the
    # study model.
    root = str(tmp_path)
    name, sweep_run = _forward_ref_workspace(root)
    manifest = Manifest.load(name, root)
    # Certificate claims L2/α4 (matching the evidence) but the concrete
    # injection is L3 — the artifact does not embody its certificate.
    runs = os.path.join(root, "runs")
    variant_dir = os.path.join(runs,
                               "20260718T000002000-variant-chain-fear-agent")
    _write(os.path.join(variant_dir, "chain-fear-agent.json"), json.dumps({
        "name": "chain-fear-agent", "baseModelID": "org/m",
        "injections": [{"concept": "fear", "layer": 3, "alpha": 4.0}],
        "promotion": {"experiment": name,
                      "experimentHash": manifest.content_hash(),
                      "promotedBy": "criterion", "sweepRun": sweep_run,
                      "winningCell": {"layer": 2, "alpha": 4.0}}}))
    run_dir = os.path.join(runs, "20260718T000003000-exp-chain-run")
    os.makedirs(run_dir, exist_ok=True)
    with pytest.raises(ValueError, match="does not embody"):
        tasks._resolve_manifest_forward_refs(manifest, run_dir, root,
                                             lambda *_: None)
    # Wrong base model refuses too.
    _write(os.path.join(variant_dir, "chain-fear-agent.json"), json.dumps({
        "name": "chain-fear-agent", "baseModelID": "other/model",
        "injections": [{"concept": "fear", "layer": 2, "alpha": 4.0}],
        "promotion": {"experiment": name,
                      "experimentHash": manifest.content_hash(),
                      "promotedBy": "criterion", "sweepRun": sweep_run,
                      "winningCell": {"layer": 2, "alpha": 4.0}}}))
    fresh = Manifest.load(name, root)
    with pytest.raises(ValueError, match="not the study model"):
        tasks._resolve_manifest_forward_refs(fresh, run_dir, root,
                                             lambda *_: None)


def test_chain_mode_refuses_missing_ledger_pins(tmp_path):
    # Fifth round: inside a chain the ledger is the AUTHORITY — an absent
    # pin is an inconsistency and refuses; only ledger_pins=None
    # (standalone) permits catalog resolution.
    root = str(tmp_path)
    name, sweep_run = _forward_ref_workspace(root)
    manifest = Manifest.load(name, root)
    _plant_agent(root, name, sweep_run=sweep_run,
                 live_hash=manifest.content_hash())
    run_dir = os.path.join(root, "runs", "20260718T000003000-exp-chain-run")
    os.makedirs(run_dir, exist_ok=True)
    with pytest.raises(ValueError, match="no promote pin"):
        tasks._resolve_manifest_forward_refs(manifest, run_dir, root,
                                             lambda *_: None, ledger_pins={})
    # Standalone (None) still resolves via the catalog.
    fresh = Manifest.load(name, root)
    tasks._resolve_manifest_forward_refs(fresh, run_dir, root,
                                         lambda *_: None, ledger_pins=None)
    assert fresh.variant_conditions[0].artifact_hash


def test_partial_pins_and_foreign_records_fail_closed(tmp_path):
    # Fifth round: a "full pin" with optional halves is a partial pin —
    # sweepRun and a finite winningCell are REQUIRED; pins are contained
    # to the runs root; resume records verify their provenance shape.
    root = str(tmp_path)
    name, sweep_run = _forward_ref_workspace(root)
    manifest = Manifest.load(name, root)
    live_hash = manifest.content_hash()
    pinned = _plant_agent(root, name, sweep_run=sweep_run,
                          live_hash=live_hash)
    with open(pinned, "rb") as handle:
        pinned_hash = hashlib.sha256(handle.read()).hexdigest()
    run_dir = os.path.join(root, "runs", "20260718T000003000-exp-chain-run")
    os.makedirs(run_dir, exist_ok=True)
    good = {"path": pinned, "hash": pinned_hash, "sweepRun": sweep_run,
            "winningCell": {"layer": 2, "alpha": 4.0}}

    def attempt(pin, match):
        fresh = Manifest.load(name, root)
        with pytest.raises(ValueError, match=match):
            tasks._resolve_manifest_forward_refs(
                fresh, run_dir, root, lambda *_: None,
                ledger_pins={"fear": pin})

    attempt({k: v for k, v in good.items() if k != "sweepRun"},
            "names no sweepRun")
    attempt({**good, "winningCell": None}, "no finite winningCell")
    attempt({**good, "winningCell": {"layer": 2, "alpha": "nan"}},
            "no finite winningCell")
    outside = os.path.join(root, "loot.json")
    _write(outside, open(pinned).read())
    attempt({**good, "path": outside}, "outside the runs root")

    # Resume-record provenance shape: foreign experiment, duplicate rows,
    # and concept mismatch each refuse before any artifact is touched.
    def seed_record(record):
        with open(os.path.join(run_dir, "forward-resolutions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(record, handle)

    row = {"condition": "fear-agent", "concept": "fear",
           "artifactPath": os.path.relpath(pinned, root),
           "artifactHash": pinned_hash, "sweepRun": sweep_run,
           "winningCell": {"layer": 2, "alpha": 4.0}}
    seed_record({"schema": 1, "experiment": "someone-else",
                 "resolutions": [row]})
    fresh = Manifest.load(name, root)
    with pytest.raises(ValueError, match="belongs to experiment"):
        tasks._resolve_manifest_forward_refs(fresh, run_dir, root,
                                             lambda *_: None)
    seed_record({"schema": 1, "experiment": name,
                 "resolutions": [row, row]})
    fresh = Manifest.load(name, root)
    with pytest.raises(ValueError, match="duplicate condition"):
        tasks._resolve_manifest_forward_refs(fresh, run_dir, root,
                                             lambda *_: None)
    seed_record({"schema": 1, "experiment": name,
                 "resolutions": [{**row, "concept": "other"}]})
    fresh = Manifest.load(name, root)
    with pytest.raises(ValueError, match="resolves concept"):
        tasks._resolve_manifest_forward_refs(fresh, run_dir, root,
                                             lambda *_: None)


def test_schema_1_ledger_refuses_resume(tmp_path, monkeypatch):
    # Fifth round: a schema-1 ledger predates the promote-pin contract —
    # resuming it would fall back to ambient catalog resolution, so it
    # refuses instead of silently reintroducing the fixed bug.
    root = str(tmp_path)
    name = _workspace(root, pipeline={})
    calls: list = []
    _stage_fakes(monkeypatch, root, calls)
    old_dir = os.path.join(root, "runs",
                           "20260718T000001000-exp-chain-pipeline")
    _write(os.path.join(old_dir, "pipeline.json"), json.dumps({
        "schema": 1, "experiment": name, "experimentHash": "0" * 64,
        "stages": list(pspec.DEFAULT_STAGES), "disposition": None,
        "stageResults": {"promote": {
            "status": "completed",
            "concepts": {"fear": "runs/old/agent.json"}}}}))
    with _acquire_counter([]) as provider:
        with pytest.raises(ValueError, match="schema 1"):
            tasks.pipeline(name, root, model_provider=provider,
                           pipeline_run_directory=old_dir,
                           log=lambda *_: None)
    assert calls == []


def test_forward_ref_freeze_gates_exempt_the_unminted_agent(tmp_path):
    # Freeze gates: variant validity and battery evidence must not demand
    # an artifact that BY DESIGN does not exist yet; concrete variants
    # keep the full gates.
    from steerlab_server.experiment.experiment_store import (
        ExperimentStoreError, _check_battery_evidence,
        _check_variant_validity)
    root = str(tmp_path)
    name, _ = _forward_ref_workspace(root, with_evidence=False)
    d = es.load_raw(name, root)
    _check_variant_validity(name, d)  # no raise: forward ref exempt
    evidence = {"batteryResults": [
        {"condition": "baseline", "accuracy": 1.0, "batteryHash": None}]}
    _check_battery_evidence(name, d, evidence, root)  # baseline suffices

    d["variantConditions"].append(
        {"name": "concrete", "artifactPath": "runs/x/agent.json"})
    with pytest.raises(ExperimentStoreError, match="no pinned artifactHash"):
        _check_variant_validity(name, d)
    with pytest.raises(ExperimentStoreError, match="concrete"):
        _check_battery_evidence(name, d, evidence, root)


# --- dtype-pinned chains (regression, 2026-07-26) ---------------------------

def _dtype_provider(dtype, acquisitions):
    """A model provider that accepts the dtype keyword, as the real loader
    does — `_acquire_model` forwards the manifest's pin whenever one is set."""
    @contextmanager
    def provider(model_id, revision=None, dtype=None):
        acquisitions.append((model_id, dtype))
        yield SimpleNamespace(revision=revision or "abc", dtype=_stamped)
    _stamped = dtype
    return provider


def _acquiring_stage(calls, root, label="extract"):
    """A fake stage that acquires through the provider it was handed, exactly
    as the real stage tasks do — the path that exercises the chain's held
    model provider."""
    def stage(name, r=None, dtype="auto", device=None, **kwargs):
        manifest = Manifest.load(name, r or root)
        with tasks._acquire_model(manifest, dtype, device,
                                  kwargs.get("model_provider")):
            calls.append(label)
        run_dir = os.path.join(root, "runs", f"20260726T000000000-exp-{label}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    return stage


def test_dtype_pinned_pipeline_acquires_the_held_model(tmp_path, monkeypatch):
    """A study pinning `dtype` must be runnable as a pipeline.

    `_acquire_model` forwards the manifest's pinned dtype as a KEYWORD to the
    provider whenever one is pinned. The chain's held-model provider took only
    (model_id, revision), so every dtype-pinned study died with
    `TypeError: _held() got an unexpected keyword argument 'dtype'` at the
    first stage, before touching any data (observed on a cluster 2026-07-26:
    an extract+validate chain, 0 artifacts written).
    """
    root = str(tmp_path)
    name = _workspace(root, pipeline={"stages": ["extract"]})
    raw = es.load_raw(name, root)
    raw["dtype"] = "bfloat16"
    es.save_raw(raw, root)

    calls: list = []
    monkeypatch.setattr(tasks, "extract", _acquiring_stage(calls, root))
    acquisitions: list = []
    tasks.pipeline(name, root,
                   model_provider=_dtype_provider("bfloat16", acquisitions),
                   log=lambda *_: None)
    assert calls == ["extract"]
    # ONE load spans the chain, and the pin reached the loader.
    assert acquisitions == [("org/m", "bfloat16")]


def test_dtype_pinned_pipeline_still_refuses_a_resident_mismatch(
        tmp_path, monkeypatch):
    """Accepting the keyword must not weaken the false-pin guard: a chain
    whose held model is resident at another precision still refuses, at the
    chain's single acquire."""
    root = str(tmp_path)
    name = _workspace(root, pipeline={"stages": ["extract"]})
    raw = es.load_raw(name, root)
    raw["dtype"] = "bfloat16"
    es.save_raw(raw, root)

    monkeypatch.setattr(tasks, "extract", _acquiring_stage([], root))
    with pytest.raises(RuntimeError) as excinfo:
        tasks.pipeline(name, root,
                       model_provider=_dtype_provider("float16", []),
                       log=lambda *_: None)
    assert "precision cannot be changed" in str(excinfo.value)


# --- continuation survives self-inflicted and real manifest drift ---------------

def _drift_fixture(tmp_path, *, live_revision=None, snapshot_revision="a" * 40):
    """A draft manifest + a completed-run ledger whose snapshot carries
    ``snapshot_revision`` while the live manifest carries ``live_revision``."""
    import json, os
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest
    root = str(tmp_path)
    es.create("drift", model_id="org/m", root=root)
    if live_revision:
        es.pin_model_revision("drift", live_revision, root)
    live = Manifest.load("drift", root)
    snap_raw = dict(live.raw)
    if snapshot_revision:
        snap_raw["modelRevision"] = snapshot_revision
    else:
        snap_raw.pop("modelRevision", None)
    snap = Manifest.from_dict(snap_raw)
    run_dir = os.path.join(root, "runs", "r1")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "experiment.json"), "w") as handle:
        json.dump(snap.raw, handle)
    ledger = {"experimentHash": snap.content_hash(),
              "stageResults": {"run": {"status": "completed",
                                       "runDirectory": run_dir}}}
    return root, live, ledger


def test_continuation_restores_the_clobbered_revision_pin(tmp_path):
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment.manifest import Manifest
    root, live, ledger = _drift_fixture(tmp_path)
    logs = []
    manifest, live_hash, restored = tasks._restore_self_pinned_revision(
        "drift", live, ledger, root, logs.append)
    assert restored is True
    assert live_hash == ledger["experimentHash"]
    # Persisted, not just in-memory: a fresh load carries the pin.
    assert Manifest.load("drift", root).model_revision == "a" * 40
    assert any("restored model revision pin" in l for l in logs)


def test_continuation_does_not_invent_a_pin_over_real_drift(tmp_path):
    from steerlab_server.experiment import tasks
    root, live, ledger = _drift_fixture(tmp_path, live_revision="b" * 40)
    manifest, live_hash, restored = tasks._restore_self_pinned_revision(
        "drift", live, ledger, root, lambda *_: None)
    assert restored is False
    assert live_hash != ledger["experimentHash"]


def test_drift_stamp_lands_in_the_pipeline_ledger(tmp_path):
    import json, os
    from steerlab_server.experiment import tasks
    pdir = str(tmp_path)
    with open(os.path.join(pdir, "pipeline.json"), "w") as handle:
        json.dump({"experimentHash": "aaa"}, handle)
    tasks._stamp_pipeline_drift(pdir, {"experimentHash": "aaa"}, "bbb")
    on_disk = json.load(open(os.path.join(pdir, "pipeline.json")))
    assert on_disk["epochDriftAtContinuation"] == [
        {"ledgerHash": "aaa", "liveHash": "bbb"}]

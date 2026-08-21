"""Finding 1 guards, fan-out era (external review 2026-07-23; live incident
2026-07-22: judge-1=gemma-4b + judge-2=gemma-12b froze fine, then the frozen
pipeline died at runtime because the chain holds only the study model).

Commit 2 semantics: a pipeline's EVALUATE stage with local judges resolving
to models other than the study model ROUTES to the post-generation judge
fan-out (advisory note, never a gate); a judged SWEEP with such judges still
refuses everywhere (no fan-out exists for sweep-interleaved judging). Local
judges gain pinned revisions: freeze pins blank study-model local-judge
revisions from the study pin.
"""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _concept(root, name="fear"):
    d = os.path.join(root, "prompts", "concepts", name)
    _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')


def _judged_study(root, name="s", *, judge_model="other/judge-12b",
                  pipeline=None, sweep=None):
    _concept(root)
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    d = es.load_raw(name, root)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         "Judge which response is better.")
    d["judgeRubricFile"] = "prompts/rubrics/r.md"
    d["judgeRubricHash"] = rubric_hash
    d["judges"] = [
        {"name": "judge-1", "kind": "local"},              # -> study model
        # Foreign local judges pin revision+dtype (review round 2,
        # finding 3); this fixture is about DELIVERABILITY, not pins.
        {"name": "judge-2", "kind": "local", "model": judge_model,
         "revision": "cafe01", "dtype": "bfloat16"},
    ]
    if pipeline is not None:
        d["pipeline"] = pipeline
    if sweep is not None:
        d["sweep"] = sweep
        # An operative sweep pins its input files at freeze — they must
        # exist for the freeze attempt to reach the judge gate.
        _write(os.path.join(root, "prompts", "dev", "dev-prompts.jsonl"),
               '{"id": "d1", "prompt": "Describe the cellar."}\n')
        _write(os.path.join(root, "prompts", "batteries", "basic.jsonl"),
               '{"id": "b1", "prompt": "2+2?", "answer": "4"}\n')
    es.save_raw(d, root)
    return name


def _validate_evidence(root, name):
    scope = Manifest.load(name, root=root).validation_scope_hash()
    rundir = os.path.join(root, "runs", f"v-exp-{name}-validate")
    os.makedirs(rundir)
    json.dump({"schemaVersion": 1, "task": "validate",
               "substrate": "python-hf-transformers",
               "validationScopeHash": scope},
              open(os.path.join(rundir, "validation-evidence.json"), "w"))
    json.dump({"concepts": {"fear": {"scenarioAccuracy": 0.9}}},
              open(os.path.join(rundir, "validation-report.json"), "w"))


def _sweep_selection_raw():
    # The raw manifest sweep block shape the guard reads:
    # sweep.selection.objective.metric (the declared criterion).
    return {"selection": {"objective": {"metric": "judgeScore"}}}


# --- the sweep gate (still a refusal — no fan-out for sweep judging) ------------


def test_freeze_refuses_judged_sweep_pipeline_with_foreign_local_judge(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root, pipeline={"stages": ["sweep"]},
                         sweep=_sweep_selection_raw())
    _validate_evidence(root, name)
    with pytest.raises(es.ExperimentStoreError,
                       match="holds ONE model") as excinfo:
        es.freeze(name, root=root)
    message = str(excinfo.value)
    assert "sweep stage" in message
    assert "judge-2" in message
    assert "fan-out covers the evaluate stage only" in message


def test_force_freeze_skips_the_sweep_guard_under_judge_validity(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root, pipeline={"stages": ["sweep"]},
                         sweep=_sweep_selection_raw())
    _validate_evidence(root, name)
    frozen = es.freeze(name, force=True, root=root)
    assert frozen["freezeForced"] is True
    assert "judgeValidity" in frozen["forcedGatesSkipped"]


def test_sweep_preflight_refuses_before_the_model_loads(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root, sweep=_sweep_selection_raw())
    manifest = Manifest.load(name, root)
    with pytest.raises(RuntimeError, match="holds ONE model"):
        tasks._pipeline_inline_judging_preflight(manifest, ["sweep"])


# --- the evaluate stage routes to the fan-out (never a gate) --------------------


def test_evaluate_pipeline_with_foreign_local_judge_freezes_with_note(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root, pipeline={"stages": ["run", "evaluate"]})
    _validate_evidence(root, name)
    d = es.load_raw(name, root)
    assert es.local_judge_pipeline_problem(d) is None
    note = es.local_judge_fanout_note(d)
    assert note is not None and "post-generation judge fan-out" in note
    assert "judge-2" in note
    advisories = es.freeze_advisories(d, root)
    assert any("post-generation judge fan-out" in a for a in advisories)
    frozen = es.freeze(name, root=root)
    assert frozen["status"] == "frozen"


def test_evaluate_preflight_no_longer_refuses_foreign_local_judges(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root, pipeline={"stages": ["run", "evaluate"]})
    manifest = Manifest.load(name, root)
    tasks._pipeline_inline_judging_preflight(manifest, ["evaluate"])
    tasks._pipeline_inline_judging_preflight(manifest, ["analyze"])


def test_fanout_judge_models_group_by_distinct_resolved_model(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root)
    d = es.load_raw(name, root)
    # Same model AND same pin as judge-2 — the point of this test is that
    # they collapse into ONE worker load. (Foreign local judges must pin
    # revision+dtype since review round 2, finding 3.)
    d["judges"].append({"name": "judge-3", "kind": "local",
                        "model": "other/judge-12b",
                        "revision": "cafe01", "dtype": "bfloat16"})
    d["judges"].append({"name": "judge-4", "kind": "local", "model": "org/m",
                        "revision": "jr4"})
    es.save_raw(d, root)
    manifest = Manifest.load(name, root)
    groups = tasks.evaluate_fanout_judge_models(manifest)
    by_model_rev = {(g["model"], g["revision"]): g for g in groups}
    # judge-1 (blank -> study model at the study pin) and judge-4 (study
    # model at its own pin) are DISTINCT worker loads.
    assert set(by_model_rev) == {("org/m", "abc"), ("org/m", "jr4"),
                                 ("other/judge-12b", "cafe01")}
    assert by_model_rev[("other/judge-12b", "cafe01")]["judges"] == [
        "judge-2", "judge-3"]
    # All local judges resolving to the study model -> no fan-out at all.
    d["judges"] = [{"name": "judge-1", "kind": "local"},
                   {"name": "claude", "kind": "claude"}]
    es.save_raw(d, root)
    assert tasks.evaluate_fanout_judge_models(Manifest.load(name, root)) == []


# --- submission preflight (routing where capable, refusal where not) ------------


def test_submission_preflight_routes_or_refuses(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root, pipeline={"stages": ["run", "evaluate"]})
    manifest = Manifest.load(name, root)
    from steerlab_server.api.submissions import _check_local_judge_deliverability

    # Slurm run-first pipeline: routed, with the note.
    note = _check_local_judge_deliverability(
        manifest, "pipeline", fanout_capable=True,
        pipeline_stages=["run", "evaluate"])
    assert note is not None and "judging fans out post-generation" in note
    # Not fan-out capable (local executor / server-resident path): refuse
    # with the remedy.
    with pytest.raises(ValueError, match="judge fan-out"):
        _check_local_judge_deliverability(
            manifest, "pipeline", fanout_capable=False,
            pipeline_stages=["run", "evaluate"])
    # A non-run-first chain cannot be managed by the sharded-parent
    # machinery: refuse with the restructure remedy.
    with pytest.raises(ValueError, match="RUN-FIRST"):
        _check_local_judge_deliverability(
            manifest, "pipeline", fanout_capable=True,
            pipeline_stages=["extract", "run", "evaluate"])
    # Non-pipeline verbs are untouched.
    assert _check_local_judge_deliverability(
        manifest, "run", fanout_capable=False, pipeline_stages=None) is None
    assert _check_local_judge_deliverability(
        None, "pipeline", fanout_capable=False, pipeline_stages=None) is None

    # A judged-sweep pipeline refuses regardless of capability.
    name2 = _judged_study(root, "s2", pipeline={"stages": ["sweep"]},
                          sweep=_sweep_selection_raw())
    manifest2 = Manifest.load(name2, root)
    with pytest.raises(ValueError, match="sweep stage holds ONE model"):
        _check_local_judge_deliverability(
            manifest2, "pipeline", fanout_capable=True,
            pipeline_stages=["sweep"])


# --- revision pinning (unchanged from commit 1) --------------------------------


def test_study_model_local_judges_freeze_and_pin_revisions(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root, judge_model="org/m")
    d = es.load_raw(name, root)
    d["judges"] = [
        {"name": "judge-1", "kind": "local"},
        {"name": "judge-2", "kind": "claude"},
    ]
    es.save_raw(d, root)
    _validate_evidence(root, name)
    frozen = es.freeze(name, root=root)
    assert frozen["status"] == "frozen"
    judges = {j["name"]: j for j in frozen["judges"]}
    assert judges["judge-1"]["revision"] == "abc"
    assert "revision" not in judges["judge-2"]


def test_declared_judge_revision_is_never_overwritten(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root, judge_model="org/m")
    d = es.load_raw(name, root)
    d["judges"] = [
        {"name": "judge-1", "kind": "local", "revision": "beef05"},
        {"name": "judge-2", "kind": "claude"},
    ]
    es.save_raw(d, root)
    _validate_evidence(root, name)
    frozen = es.freeze(name, root=root)
    judges = {j["name"]: j for j in frozen["judges"]}
    assert judges["judge-1"]["revision"] == "beef05"


def test_judge_ref_parses_revision_and_dtype(tmp_path):
    root = str(tmp_path)
    name = _judged_study(root)
    d = es.load_raw(name, root)
    d["judges"][1]["revision"] = "r2"
    d["judges"][1]["dtype"] = "bfloat16"
    es.save_raw(d, root)
    manifest = Manifest.load(name, root)
    by_name = {j.name: j for j in manifest.judges}
    assert by_name["judge-2"].revision == "r2"
    assert by_name["judge-2"].dtype == "bfloat16"
    assert by_name["judge-1"].revision is None

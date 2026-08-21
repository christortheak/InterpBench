"""WS7.1 cross-substrate validate-evidence advisory: NON-BLOCKING, loud, and
only when the evidence's engine is KNOWN to differ — legacy runs are never
accused, and same-engine evidence stays silent.

Fires in two places: freeze (advisory list, printed to stderr, never a gate)
and study-run start (run log + a durable advisories.txt creation stamp)."""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest

THIS = "python-hf-transformers"
OTHER = "swift-mlx"


def _concept(root, name="french"):
    d = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "positive.jsonl"), "w").write('{"text": "bonjour"}\n')
    open(os.path.join(d, "negative.jsonl"), "w").write('{"text": "hello"}\n')


def _study(root, name="s"):
    _concept(root)
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["french"], root=root)
    return Manifest.load(name, root=root)


def _validate_run(root, scope, *, dirname, substrate="unset", config_substrate=None):
    """Plant a COMPLETE validate run. ``substrate="unset"`` fabricates legacy
    evidence (no stamp); None/str set the stamp explicitly. ``config_substrate``
    plants a canonical config.json alongside (the pre-substrate-evidence
    fallback the advisory consults)."""
    rundir = os.path.join(root, "runs", dirname)
    os.makedirs(rundir, exist_ok=True)
    evidence = {"schemaVersion": 1, "task": "validate",
                "validationScopeHash": scope}
    if substrate != "unset":
        evidence["substrate"] = substrate
    json.dump(evidence, open(os.path.join(rundir, "validation-evidence.json"), "w"))
    json.dump({"concepts": {"french": {"scenarioAccuracy": 0.9}}},
              open(os.path.join(rundir, "validation-report.json"), "w"))
    if config_substrate is not None:
        json.dump({"schemaVersion": 2, "runType": "validate",
                   "substrate": config_substrate},
                  open(os.path.join(rundir, "config.json"), "w"))
    return rundir


# --- the advisory predicate ------------------------------------------------

def test_advisory_present_when_only_foreign_evidence_exists(tmp_path):
    root = str(tmp_path)
    manifest = _study(root)
    scope = manifest.validation_scope_hash()
    _validate_run(root, scope, dirname="a-exp-s-validate", substrate=OTHER)
    advisory = es.cross_substrate_validation_advisory(scope, root)
    assert advisory == (f"validation evidence was produced on {OTHER}; "
                        f"runs on {THIS} should re-validate on-substrate")


def test_advisory_absent_when_same_engine_evidence_exists(tmp_path):
    root = str(tmp_path)
    manifest = _study(root)
    scope = manifest.validation_scope_hash()
    _validate_run(root, scope, dirname="a-exp-s-validate", substrate=THIS)
    assert es.cross_substrate_validation_advisory(scope, root) is None
    # …even when foreign evidence ALSO exists — the gate relies on the
    # same-engine run, so there is nothing to advise about.
    _validate_run(root, scope, dirname="b-exp-s-validate", substrate=OTHER)
    assert es.cross_substrate_validation_advisory(scope, root) is None


def test_advisory_absent_when_engine_unknowable(tmp_path):
    root = str(tmp_path)
    manifest = _study(root)
    scope = manifest.validation_scope_hash()
    # Legacy evidence: no substrate stamp, no config.json — could be either
    # engine, and legacy is treated as THIS engine by the gate. Never accuse.
    _validate_run(root, scope, dirname="a-exp-s-validate", substrate="unset")
    assert es.cross_substrate_validation_advisory(scope, root) is None


def test_advisory_absent_when_no_evidence_at_all(tmp_path):
    root = str(tmp_path)
    manifest = _study(root)
    scope = manifest.validation_scope_hash()
    assert es.cross_substrate_validation_advisory(scope, root) is None


def test_advisory_uses_config_json_when_evidence_predates_substrate_stamp(tmp_path):
    """Unstamped (legacy-shaped) evidence whose run carries a canonical
    config.json naming the OTHER engine: the gate would count it as legacy
    this-engine evidence — exactly the sneaky case the advisory catches."""
    root = str(tmp_path)
    manifest = _study(root)
    scope = manifest.validation_scope_hash()
    _validate_run(root, scope, dirname="a-exp-s-validate",
                  substrate="unset", config_substrate=OTHER)
    advisory = es.cross_substrate_validation_advisory(scope, root)
    assert advisory is not None and OTHER in advisory


# --- freeze surface ---------------------------------------------------------

def test_freeze_advisories_list_and_stderr_never_block(tmp_path, capsys):
    root = str(tmp_path)
    manifest = _study(root)
    scope = manifest.validation_scope_hash()
    _validate_run(root, scope, dirname="a-exp-s-validate", substrate=OTHER)

    d = es.load_raw("s", root)
    advisories = es.freeze_advisories(d, root)
    assert len(advisories) == 1 and OTHER in advisories[0]

    # Foreign evidence never satisfies the gate: non-force freeze still
    # refuses (advisory ≠ evidence), but the refusal is EXPLAINED on stderr.
    with pytest.raises(es.ExperimentStoreError):
        es.freeze("s", force=False, root=root)
    assert "re-validate on-substrate" in capsys.readouterr().err

    # Force-freeze proceeds — the advisory stays loud and non-blocking.
    frozen = es.freeze("s", force=True, root=root)
    assert frozen["status"] == "frozen"
    assert "re-validate on-substrate" in capsys.readouterr().err


def test_freeze_with_same_engine_evidence_prints_no_advisory(tmp_path, capsys):
    root = str(tmp_path)
    manifest = _study(root)
    scope = manifest.validation_scope_hash()
    _validate_run(root, scope, dirname="a-exp-s-validate", substrate=THIS)
    assert es.freeze_advisories(es.load_raw("s", root), root) == []
    frozen = es.freeze("s", force=False, root=root)
    assert frozen["status"] == "frozen"
    assert "re-validate on-substrate" not in capsys.readouterr().err


# --- study-run-start surface -------------------------------------------------

def test_run_start_advisory_logs_and_stamps_file(tmp_path):
    root = str(tmp_path)
    manifest = _study(root)
    scope = manifest.validation_scope_hash()
    _validate_run(root, scope, dirname="a-exp-s-validate", substrate=OTHER)
    run_dir = os.path.join(root, "runs", "b-exp-s-run")
    os.makedirs(run_dir)

    lines: list[str] = []
    tasks._advise_cross_substrate(manifest, run_dir, root, lines.append,
                                  write_file=True)
    assert lines and lines[0].startswith("ADVISORY: ")
    assert OTHER in lines[0]
    stamped = open(os.path.join(run_dir, "advisories.txt")).read()
    assert "re-validate on-substrate" in stamped

    # Resume semantics: the log stays loud, the file is a creation stamp.
    before = open(os.path.join(run_dir, "advisories.txt"), "rb").read()
    tasks._advise_cross_substrate(manifest, run_dir, root, lines.append,
                                  write_file=False)
    assert len(lines) == 2
    assert open(os.path.join(run_dir, "advisories.txt"), "rb").read() == before


def test_run_start_advisory_silent_when_same_engine(tmp_path):
    root = str(tmp_path)
    manifest = _study(root)
    scope = manifest.validation_scope_hash()
    _validate_run(root, scope, dirname="a-exp-s-validate", substrate=THIS)
    run_dir = os.path.join(root, "runs", "b-exp-s-run")
    os.makedirs(run_dir)
    lines: list[str] = []
    tasks._advise_cross_substrate(manifest, run_dir, root, lines.append,
                                  write_file=True)
    assert lines == []
    assert not os.path.exists(os.path.join(run_dir, "advisories.txt"))

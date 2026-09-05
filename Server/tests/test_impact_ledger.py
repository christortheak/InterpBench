"""``ledger impact`` — the REM-01 impact ledger.

The tool answers one question about existing artifacts: which of the four
2026-09-05 science fixes could have reached this, and what was read to decide?
Three properties carry the weight and are pinned here:

- **missing metadata is ``unknown``, never ``unaffected``** — the whole value
  of the ledger is that it refuses to launder an absence into a clean bill;
- **nothing under ``runs/`` is touched** — corrections are NEW artifacts with
  provenance links, so the fixture's run tree is hashed before and after;
- **the reassessed promotions come from the analysis path itself**
  (``tasks._promotion_decisions`` under the current dose rule), so the only
  difference between the original funnel and the reassessed one is the code.

Fully offline: synthetic sidecars, synthetic run stamps, a throwaway git
repository for the ancestry leg. No model, no HF, no network.
"""

import hashlib
import json
import os
import subprocess

import pytest

from steerlab_server import cli
from steerlab_server.experiment import impact_ledger as il


# --- fixture workspace -------------------------------------------------------

#: Build stamps the fixture uses. The values are opaque here; the ancestry
#: tests rebind the findings' fix commits onto a throwaway repository so the
#: ledger's real shas never have to exist in the test environment.
PRE_FIX_COMMIT = "aaaaaaa1"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _write_json(path, payload):
    return _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _config(directory, run_type, *, app_version=None, **extra):
    payload = {"schemaVersion": 4, "runType": run_type,
               "runId": os.path.basename(directory),
               "createdAt": "2026-09-01T00:00:00Z",
               "substrate": "python-hf-transformers",
               "modelID": "test/model", "revision": "r" * 40,
               "notes": {}}
    if app_version is not None:
        payload["appVersion"] = app_version
    payload.update(extra)
    return _write_json(os.path.join(directory, "config.json"), payload)


def _python_sidecar(*, accumulation, objective=None, name="adapter"):
    schedule = {"batchSize": 1, "effectiveBatchSize": accumulation,
                "gradientAccumulation": accumulation, "epochs": 1,
                "maxSteps": None, "warmupSteps": 0, "lrSchedule": "cosine",
                "seed": 0}
    if objective is not None:
        schedule["objective"] = objective
    return {
        "name": name, "baseModelID": "test/model", "revision": "r" * 40,
        "substrate": "python-hf-transformers", "adapterFormat": "hf-peft-lora",
        "trainingMode": "instruction_chat", "schedule": schedule,
        "adapterBytesHash": "c" * 64, "adapterConfigHash": "d" * 64,
        "buildIdentity": {"commit": PRE_FIX_COMMIT, "dirty": False},
        "schemaVersion": 2, "evidenceGrade": True,
    }


def _swift_sidecar(*, training_mode):
    return {
        "schemaVersion": 1, "name": f"mlx-{training_mode}",
        "baseModelID": "test/model", "baseRevision": "r" * 40,
        "adapterDirectory": "adapters", "adapterHash": "e" * 64,
        "configHash": "f" * 64, "substrate": "swift-mlx",
        "adapterFormat": "mlx-lora", "fineTuneType": "lora", "rank": 8,
        "scale": 10, "adaptedLayers": 16, "trainingMode": training_mode,
        "batchSize": 4, "iterations": 100, "learningRate": 1e-5,
        "createdAt": "2026-09-01T00:00:00Z", "notes": "",
    }


SCREEN_MANIFEST = {
    "name": "screen", "modelID": "test/model", "phase": "screen",
    "concepts": [], "taskPromptsFile": None,
    "promotionRule": {"fdrThreshold": 0.05, "doseMonotone": True,
                      "exceedsRandomFloor": False},
    "conditions": [
        {"name": "baseline", "slots": []},
        {"name": "fear-a1",
         "slots": [{"concept": "fear", "layer": 10, "alpha": 1.0}]},
        {"name": "fear-a2",
         "slots": [{"concept": "fear", "layer": 10, "alpha": 2.0}]},
    ],
}

#: Two treatment cells whose effects are IDENTICAL: a flat ladder. Under the
#: current rule it is not monotone (and its rho is undefined); before the fix
#: it satisfied every consecutive step trivially and promoted.
FLAT_EFFECT_ROWS = [
    ["fear-a1", "meanMonths", "12", "0.5", "0.2", "0.8", "10", "0.01", "0.01",
     "bh", "injection", "pooled", "", "", "", ""],
    ["fear-a2", "meanMonths", "12", "0.5", "0.2", "0.8", "10", "0.01", "0.01",
     "bh", "injection", "pooled", "", "", "", ""],
    # A stratified diagnostic row, which the analysis excluded from the
    # promotion rule and which the reconstruction must exclude too.
    ["fear-a2", "meanMonths", "3", "9.0", "8.0", "10.0", "1", "0.4", "",
     "", "injection", "promptID", "item-1", "sample", "withinItemSamples",
     "diagnostic"],
]

PRE_FIX_MOVERS = {
    "experiment": "screen",
    "experimentHash": "0" * 64,
    "promotionRule": {"fdrThreshold": 0.05, "doseMonotone": True,
                      "exceedsRandomFloor": False, "capabilityGate": None},
    "promoted": [{
        "concept": "fear", "condition": "fear-a2", "endpoint": "meanMonths",
        "effectEstimate": 0.5, "effectCILower": 0.2, "effectCIUpper": 0.8,
        "wilcoxonP": 0.01, "adjustedP": 0.01, "correction": "bh",
        # The pre-fix signature: graded monotone with no rho at all.
        "doseMonotone": True, "doseSpearmanRho": None,
        "randomFloorEffect": None, "capabilityPassed": None,
        "promoted": True, "reasons": [],
        "sourceRun": "20260901T000030000-exp-screen-run",
    }],
    "rejected": [],
}


def _sae_record(*, monotone, flat):
    """An SAE qualification record as ``from_dict`` accepts one. ``flat``
    makes the construct-probe values identical across the dose grid, which is
    exactly the ladder the fix stopped counting as a dose response."""
    doses = [0.04, 0.08]
    rows = []
    for index, dose in enumerate(doses):
        for sign in ("positive", "negative"):
            if flat:
                value = 0.3
            else:
                value = 0.3 + index * 0.1 if sign == "positive" \
                    else 0.3 - index * 0.1
            rows.append({"dose": dose, "sign": sign, "value": value,
                         "direction": ("increase" if sign == "positive"
                                       else "decrease"),
                         "n": 40})
    return {
        "schemaVersion": 1,
        "feature": {"feature": 62389, "layer": 2,
                    "decoderRowHash": "sha256:" + "a" * 64},
        "doseGrid": doses, "signs": ["positive", "negative"],
        "constructProbe": {"metric": "held-out construct endorsement rate",
                           "higherIsBetter": True,
                           "heldOutSetHash": "h" * 64, "results": rows},
        "lexicalLeakage": {"metric": "label-token rate",
                           "results": [{"dose": 0.08, "sign": "positive",
                                        "value": 0.01}]},
        "discriminantControls": [
            {"construct": "generic positive valence",
             "metric": "held-out endorsement rate",
             "results": [{"dose": 0.08, "sign": "positive", "value": 0.02}]}],
        "coherenceGate": {"metric": "distinct-2 + format compliance",
                          "threshold": 0.45, "passed": True,
                          "results": [{"dose": 0.08, "sign": "positive",
                                       "value": 0.61}]},
        "doseResponse": {"monotone": monotone, "spearmanRho": 0.9,
                         "signSymmetric": False},
        "decision": {"decision": "accept", "rationale": "moves the probe",
                     "date": "2026-08-12", "decidedBy": "ct"},
    }


@pytest.fixture()
def workspace(tmp_path):
    """One artifact of every kind the ledger classifies."""
    import csv

    root = str(tmp_path / "ws")
    runs = os.path.join(root, "runs")
    _write(os.path.join(root, "WORKSPACE.md"), "# SteerLab workspace\n")

    # --- SCI-01: J-lens / J-space readouts -----------------------------------
    armed = os.path.join(runs, "20260901T000000000-jlens-probe-armed")
    _config(armed, "jlens-probe",
            app_version=f"steerlab-server 0.1.0+{PRE_FIX_COMMIT}")
    _write(os.path.join(armed, "probe-topk.csv"), "layer,token,logit\n")

    watchlist = os.path.join(runs, "20260901T000001000-jlens-probe-watchlist")
    _config(watchlist, "jlens-probe",
            app_version=f"steerlab-server 0.1.0+{PRE_FIX_COMMIT}")
    _write_json(os.path.join(watchlist, "probe.json"), {"watchlistOnly": True})

    jspace = os.path.join(runs, "20260901T000002000-optvec-jspace-stamped")
    _config(jspace, "optvec-jspace",
            app_version=f"steerlab-server 0.1.0+{PRE_FIX_COMMIT}")
    _write_json(os.path.join(jspace, "jspace.json"), {
        "runType": "optvec-jspace",
        "vectors": [{"layers": [{"layer": 10,
                                 "topKDelta": [{"token": " the"}],
                                 "topKEmergent": []}]}]})

    unstamped = os.path.join(runs, "20260901T000003000-optvec-jspace-bare")
    _config(unstamped, "optvec-jspace")
    _write_json(os.path.join(unstamped, "jspace.json"), {
        "runType": "optvec-jspace",
        "vectors": [{"layers": [{"layer": 10,
                                 "topKDelta": [{"token": " the"}],
                                 "topKEmergent": []}]}]})

    # --- SCI-02: python adapters ---------------------------------------------
    for run_id, slug, accumulation, objective in (
            ("20260901T000010000-lora-accum2", "accum2", 2, None),
            ("20260901T000011000-lora-accum1", "accum1", 1, None),
            ("20260901T000012000-lora-stamped", "stamped", 4,
             il.OBJECTIVE_STAMP)):
        directory = os.path.join(runs, run_id)
        _config(directory, "lora-train",
                app_version=f"steerlab-server 0.1.0+{PRE_FIX_COMMIT}")
        _write_json(os.path.join(directory, f"{slug}.json"),
                    _python_sidecar(accumulation=accumulation,
                                    objective=objective, name=slug))

    # --- SCI-03: MLX adapters ------------------------------------------------
    for run_id, mode in (
            ("20260901T000020000-fine-tune-instr", "instruction_chat"),
            ("20260901T000021000-fine-tune-doc", "document")):
        directory = os.path.join(runs, "fine-tunes", run_id)
        _config(directory, "lora-train",
                app_version=f"swift-app 0.9.0-dev+{PRE_FIX_COMMIT}",
                substrate="swift-mlx")
        _write_json(os.path.join(directory, "fine-tune.json"),
                    _swift_sidecar(training_mode=mode))

    # --- SCI-04: a screen whose flat ladder promoted --------------------------
    source_run = os.path.join(runs, "20260901T000030000-exp-screen-run")
    _config(source_run, "run",
            app_version=f"steerlab-server 0.1.0+{PRE_FIX_COMMIT}")
    _write_json(os.path.join(source_run, "experiment.json"), SCREEN_MANIFEST)

    analysis = os.path.join(runs, "20260901T000031000-exp-screen-analyze")
    _config(analysis, "analyze",
            app_version=f"steerlab-server 0.1.0+{PRE_FIX_COMMIT}")
    _write_json(os.path.join(analysis, "promoted-movers.json"), PRE_FIX_MOVERS)
    _write(os.path.join(analysis, "source-run.txt"),
           os.path.basename(source_run) + "\n")
    from steerlab_server.experiment.study_stats import EFFECT_SIZES_HEADER
    with open(os.path.join(analysis, "effect-sizes.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EFFECT_SIZES_HEADER)
        for row in FLAT_EFFECT_ROWS:
            writer.writerow(row)

    # --- SCI-04: SAE qualification records ------------------------------------
    from steerlab_server.experiment import sae_qualification as saq
    disagreeing = os.path.join(runs, "20260901T000040000-sae-qualification-flat")
    _config(disagreeing, "sae-qualification")
    _write_json(os.path.join(disagreeing, saq.FILENAME),
                _sae_record(monotone=True, flat=True))
    consistent = os.path.join(runs, "20260901T000041000-sae-qualification-ok")
    _config(consistent, "sae-qualification")
    _write_json(os.path.join(consistent, saq.FILENAME),
                _sae_record(monotone=True, flat=False))

    return root


def _by_path(entries):
    return {entry["path"]: entry for entry in entries}


def _entries(root, ancestry=None):
    return il.scan(root, ancestry or il.Ancestry(None))


# --- 1. entry shape and the classification rules -----------------------------

def test_every_entry_carries_the_closed_key_set(workspace):
    entries = _entries(workspace)
    assert entries, "the fixture workspace produced no candidates"
    for entry in entries:
        assert tuple(sorted(entry)) == il.ENTRY_KEYS
        assert entry["exposure"] in il.EXPOSURES
        assert entry["requiredAction"] in il.REQUIRED_ACTIONS
        assert entry["disposition"] in il.DISPOSITIONS
        assert entry["owner"] is None
        assert entry["replacementArtifact"] is None
        assert entry["evidence"], f"{entry['path']} recorded no evidence"
        assert not os.path.isabs(entry["path"])


def test_only_an_established_absence_disposes_as_unaffected(workspace):
    for entry in _entries(workspace):
        if entry["exposure"] == il.UNAFFECTED:
            assert entry["disposition"] == "unaffected"
        else:
            assert entry["disposition"] == "unresolved"


def test_sci01_needs_the_full_vocabulary_path_and_a_datable_build(workspace):
    entries = _by_path(_entries(workspace))
    armed = entries["runs/20260901T000000000-jlens-probe-armed"]
    assert armed["finding"] == "SCI-01"
    assert armed["artifactType"] == "jlensProbeRun"
    # Armed, but nothing dates the build: unknown, never exposed-by-guess.
    assert armed["exposure"] == il.UNKNOWN
    assert armed["requiredAction"] == "resolveProvenance"
    assert armed["producingRevision"] == f"steerlab-server 0.1.0+{PRE_FIX_COMMIT}"

    watchlist = entries["runs/20260901T000001000-jlens-probe-watchlist"]
    # The top-k table's ABSENCE is a positive finding: the watchlist path
    # already folded the gain, so this readout never touched the fixed code.
    assert watchlist["exposure"] == il.UNAFFECTED
    assert watchlist["requiredAction"] == "none"
    assert any("probe-topk.csv" in fact for fact in watchlist["evidence"])


def test_sci01_reads_the_jspace_topk_tables(workspace):
    entries = _by_path(_entries(workspace))
    jspace = entries["runs/20260901T000002000-optvec-jspace-stamped"]
    assert jspace["artifactType"] == "optvecJSpaceRun"
    assert any("1 topKDelta/topKEmergent row" in fact
               for fact in jspace["evidence"])
    bare = entries["runs/20260901T000003000-optvec-jspace-bare"]
    assert bare["producingRevision"] is None
    assert bare["exposure"] == il.UNKNOWN
    assert any("carries no producing revision" in fact
               for fact in bare["evidence"])


def test_sci01_reads_a_study_runs_jlens_trace(tmp_path):
    """A study run's trace is a candidate too, and the question is answered by
    the rows themselves: ``topKIDs`` is written only when the full-vocabulary
    path was armed."""
    root = str(tmp_path / "ws")
    _write(os.path.join(root, "WORKSPACE.md"), "#\n")
    armed = os.path.join(root, "runs", "20260901T000000000-exp-a-run")
    _config(armed, "run", app_version=f"steerlab-server 0.1.0+{PRE_FIX_COMMIT}")
    _write(os.path.join(armed, "jlens-readout.jsonl"),
           json.dumps({"traceComplete": True, "topKIDs": [1, 2],
                       "topKLogits": [0.5, 0.4]}) + "\n")
    watchlist = os.path.join(root, "runs", "20260901T000001000-exp-b-run")
    _config(watchlist, "run",
            app_version=f"steerlab-server 0.1.0+{PRE_FIX_COMMIT}")
    _write(os.path.join(watchlist, "jlens-readout.jsonl"),
           json.dumps({"traceComplete": True, "watchlistLogits": [0.1]}) + "\n")

    entries = _by_path(_entries(root))
    hot = entries["runs/20260901T000000000-exp-a-run"]
    assert hot["artifactType"] == "jlensTraceRun"
    assert hot["exposure"] == il.UNKNOWN
    assert any("a row carries topKIDs" in fact for fact in hot["evidence"])

    cold = entries["runs/20260901T000001000-exp-b-run"]
    assert cold["exposure"] == il.UNAFFECTED
    assert any("no row of 1 carries topKIDs" in fact
               for fact in cold["evidence"])


def test_sci02_accumulation_rules(workspace):
    entries = _by_path(_entries(workspace))
    exposed = entries["runs/20260901T000010000-lora-accum2/accum2.json"]
    assert exposed["finding"] == "SCI-02"
    assert exposed["artifactType"] == "loraAdapterSidecar"
    assert exposed["exposure"] == il.EXPOSED
    assert exposed["requiredAction"] == "rerun"
    # The assessment must not overclaim: accumulation > 1 is a screen.
    assert "SCREENING FLAG" in exposed["assessment"]

    single = entries["runs/20260901T000011000-lora-accum1/accum1.json"]
    assert single["exposure"] == il.UNAFFECTED
    assert "SAME number" in single["assessment"]

    stamped = entries["runs/20260901T000012000-lora-stamped/stamped.json"]
    assert stamped["exposure"] == il.UNAFFECTED
    assert il.OBJECTIVE_STAMP in stamped["assessment"]


def test_sci02_without_a_recoverable_accumulation_is_unknown(tmp_path):
    root = str(tmp_path / "ws")
    directory = os.path.join(root, "runs", "20260901T000000000-lora-bare")
    _write(os.path.join(root, "WORKSPACE.md"), "#\n")
    _config(directory, "lora-train")
    sidecar = _python_sidecar(accumulation=2)
    sidecar.pop("schedule")
    _write_json(os.path.join(directory, "bare.json"), sidecar)
    entry, = _entries(root)
    assert entry["finding"] == "SCI-02"
    assert entry["exposure"] == il.UNKNOWN
    assert entry["requiredAction"] == "resolveProvenance"
    assert "gradientAccumulation" in entry["assessment"]


def test_sci03_reads_the_mlx_sidecar_mode_and_names_its_missing_evidence(
        workspace):
    entries = _by_path(_entries(workspace))
    instruction = entries[
        "runs/fine-tunes/20260901T000020000-fine-tune-instr/fine-tune.json"]
    assert instruction["finding"] == "SCI-03"
    assert instruction["artifactType"] == "mlxAdapterSidecar"
    # Instruction mode, but nothing dates the app build: unknown.
    assert instruction["exposure"] == il.UNKNOWN
    assert "chat template's render on the app build" in instruction["assessment"]

    document = entries[
        "runs/fine-tunes/20260901T000021000-fine-tune-doc/fine-tune.json"]
    assert document["exposure"] == il.UNAFFECTED
    assert "instruction/chat render" in document["assessment"]
    # Even the clean verdict states what the sidecar cannot show.
    assert "chat template's render on the app build" in document["assessment"]


def test_sci04_flags_a_monotone_verdict_with_no_rho(workspace):
    entries = _by_path(_entries(workspace))
    movers = entries[
        "runs/20260901T000031000-exp-screen-analyze/promoted-movers.json"]
    assert movers["finding"] == "SCI-04"
    assert movers["exposure"] == il.EXPOSED
    assert movers["requiredAction"] == "recompute"
    assert any("1 of them declare doseMonotone true" in fact
               for fact in movers["evidence"])


def test_sci04_checks_an_sae_record_against_its_own_rows(workspace):
    entries = _by_path(_entries(workspace))
    flat = entries[
        "runs/20260901T000040000-sae-qualification-flat/"
        "sae-feature-qualification.json"]
    assert flat["exposure"] == il.EXPOSED
    assert flat["requiredAction"] == "reassess"
    assert any("promote's consistency check" in fact
               for fact in flat["evidence"])
    ok = entries[
        "runs/20260901T000041000-sae-qualification-ok/"
        "sae-feature-qualification.json"]
    assert ok["exposure"] == il.UNAFFECTED


def test_an_unreadable_candidate_is_unknown_not_absent(tmp_path):
    root = str(tmp_path / "ws")
    _write(os.path.join(root, "WORKSPACE.md"), "#\n")
    analysis = os.path.join(root, "runs", "20260901T000000000-exp-x-analyze")
    _config(analysis, "analyze")
    _write(os.path.join(analysis, "promoted-movers.json"), "{ truncated")
    entry, = _entries(root)
    assert entry["exposure"] == il.UNKNOWN
    assert entry["requiredAction"] == "resolveProvenance"


# --- 2. ancestry against a throwaway repository -------------------------------

def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture()
def throwaway_repo(tmp_path):
    """Three commits; the middle one is the "fix". Returns
    ``(path, {label: sha})``."""
    repo = str(tmp_path / "checkout")
    os.makedirs(repo)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "test")
    shas = {}
    for label in ("before", "fix", "after"):
        _write(os.path.join(repo, f"{label}.txt"), label)
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", label)
        shas[label] = _git(repo, "rev-parse", "HEAD")
    return repo, shas


def _retarget(monkeypatch, **fix_commits_by_finding):
    """Point one or more findings' fix commits at the throwaway repository, so
    the ancestry leg is exercised without the real shas having to exist."""
    import dataclasses
    findings = tuple(
        dataclasses.replace(f, fix_commits=tuple(
            fix_commits_by_finding.get(f.id, f.fix_commits)))
        for f in il.FINDINGS)
    monkeypatch.setattr(il, "FINDINGS", findings)
    monkeypatch.setattr(il, "FINDINGS_BY_ID",
                        {f.id: f for f in findings})


def test_ancestry_dates_a_build_against_the_fix(throwaway_repo, monkeypatch):
    repo, shas = throwaway_repo
    _retarget(monkeypatch, **{"SCI-01": (shas["fix"],)})
    ancestry = il.open_checkout(repo)
    assert ancestry.available

    finding = il.FINDINGS_BY_ID["SCI-01"]
    carries, facts = ancestry.carries_fix(finding, shas["after"])
    assert carries is True and facts

    carries, facts = ancestry.carries_fix(finding, shas["before"])
    assert carries is False
    assert any("is NOT an ancestor" in fact for fact in facts)

    # A commit this checkout does not contain is UNKNOWN with the reason, and
    # never "not an ancestor".
    carries, facts = ancestry.carries_fix(finding, "0" * 40)
    assert carries is None
    assert any("cannot resolve" in fact for fact in facts)


def test_without_a_checkout_every_revision_dependent_finding_is_unknown(
        throwaway_repo):
    _repo, shas = throwaway_repo
    ancestry = il.Ancestry(None)
    carries, facts = ancestry.carries_fix(il.FINDINGS_BY_ID["SCI-01"],
                                          shas["after"])
    assert carries is None
    assert any("--code-checkout" in fact for fact in facts)


def test_a_pre_fix_build_makes_the_jlens_readout_exposed(
        workspace, throwaway_repo, monkeypatch):
    repo, shas = throwaway_repo
    _retarget(monkeypatch, **{"SCI-01": (shas["fix"],),
                              "SCI-03": (shas["fix"],)})
    armed = os.path.join(workspace, "runs",
                         "20260901T000000000-jlens-probe-armed")
    _config_overwrite(armed, "jlens-probe",
                      f"steerlab-server 0.1.0+{shas['before']}")
    entries = _by_path(_entries(workspace, il.open_checkout(repo)))
    entry = entries["runs/20260901T000000000-jlens-probe-armed"]
    assert entry["exposure"] == il.EXPOSED
    assert entry["requiredAction"] == "rerun"
    assert "RE-RUN, not recompute" in entry["assessment"]


def test_a_post_fix_build_makes_the_mlx_instruction_adapter_unaffected(
        workspace, throwaway_repo, monkeypatch):
    repo, shas = throwaway_repo
    _retarget(monkeypatch, **{"SCI-03": (shas["fix"],)})
    directory = os.path.join(workspace, "runs", "fine-tunes",
                             "20260901T000020000-fine-tune-instr")
    _config_overwrite(directory, "lora-train",
                      f"swift-app 0.9.0-dev+{shas['after']}",
                      substrate="swift-mlx")
    entries = _by_path(_entries(workspace, il.open_checkout(repo)))
    entry = entries[
        "runs/fine-tunes/20260901T000020000-fine-tune-instr/fine-tune.json"]
    assert entry["exposure"] == il.UNAFFECTED
    assert entry["disposition"] == "unaffected"


def _config_overwrite(directory, run_type, app_version, **extra):
    os.remove(os.path.join(directory, "config.json"))
    _config(directory, run_type, app_version=app_version, **extra)


def test_a_code_checkout_that_is_not_a_repository_refuses(tmp_path):
    with pytest.raises(il.LedgerRefusal) as excinfo:
        il.open_checkout(str(tmp_path))
    assert excinfo.value.state == "blocked"
    assert excinfo.value.repair_action


# --- 3. promotion rescoring ---------------------------------------------------

def test_reassessment_flips_a_flat_ladder_promotion(workspace):
    analysis = os.path.join(workspace, "runs",
                            "20260901T000031000-exp-screen-analyze")
    document = il.reassess_promotions(workspace, analysis)

    assert document["finding"] == "SCI-04"
    assert document["original"]["sha256"]
    assert document["original"]["document"] == PRE_FIX_MOVERS
    assert document["reassessed"]["promoted"] == []
    rejected, = document["reassessed"]["rejected"]
    assert rejected["concept"] == "fear"
    assert rejected["doseMonotone"] is False
    assert rejected["doseSpearmanRho"] is None
    assert "dose-response is not monotone" in rejected["reasons"]

    changed, = document["changedVerdicts"]
    assert changed["concept"] == "fear"
    assert changed["fields"]["promoted"] == {"was": True, "now": False}
    assert changed["fields"]["doseMonotone"] == {"was": True, "now": False}

    provenance = document["provenance"]
    assert provenance["sourceAnalysis"].endswith("-exp-screen-analyze")
    assert provenance["sourceRun"].endswith("-exp-screen-run")
    assert provenance["manifestHash"]
    assert provenance["engineVersion"].startswith("steerlab-server ")
    assert provenance["observedAt"].endswith("Z")


def test_reconstruction_skips_the_stratified_rows_the_analysis_skipped(
        workspace):
    analysis = os.path.join(workspace, "runs",
                            "20260901T000031000-exp-screen-analyze")
    rows, notes = il.effect_rows_from_csv(
        os.path.join(analysis, "effect-sizes.csv"))
    assert [row.condition for row in rows] == ["fear-a1", "fear-a2"]
    assert notes["rowsRead"] == 3
    assert notes["pooledRows"] == 2
    assert notes["skippedNonPooledRows"] == 1
    # The bootstrap replicate count and seed are not columns of the CSV; the
    # notes must SAY that rather than let a reader believe 0 was measured.
    assert any("replicates" in note for note in notes["notes"])
    assert rows[0].ci.replicates == 0


def test_reconstruction_refuses_rather_than_guessing_a_manifest(workspace):
    analysis = os.path.join(workspace, "runs",
                            "20260901T000031000-exp-screen-analyze")
    os.remove(os.path.join(workspace, "runs",
                           "20260901T000030000-exp-screen-run",
                           "experiment.json"))
    with pytest.raises(il.ReconstructionImpossible) as excinfo:
        il.reassess_promotions(workspace, analysis)
    assert "experiment.json snapshot" in str(excinfo.value)


def test_an_impossible_reconstruction_lands_in_the_entry_not_a_crash(
        workspace):
    os.remove(os.path.join(workspace, "runs",
                           "20260901T000031000-exp-screen-analyze",
                           "effect-sizes.csv"))
    summary = il.build(workspace)
    ledger = json.loads(_read(os.path.join(
        workspace, summary["ledgerFile"])))
    movers = _by_path(ledger["entries"])[
        "runs/20260901T000031000-exp-screen-analyze/promoted-movers.json"]
    assert movers["exposure"] == il.EXPOSED
    assert movers["requiredAction"] == "reassess"
    assert any("reassessment not possible" in fact
               for fact in movers["evidence"])
    assert summary["reassessments"] == []


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# --- 4. output, immutability, and the envelope --------------------------------

def _tree_digest(directory):
    """SHA-256 over every path, size, and byte of a subtree — the immutability
    witness. Names and contents both, so neither a rewrite nor a new file can
    slip past."""
    digest = hashlib.sha256()
    for current, directories, files in os.walk(directory):
        directories.sort()
        for name in sorted(files):
            path = os.path.join(current, name)
            digest.update(os.path.relpath(path, directory).encode("utf-8"))
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()


def test_the_ledger_writes_only_into_its_own_diagnostics_directory(workspace):
    before = _tree_digest(os.path.join(workspace, "runs"))
    summary = il.build(workspace)
    assert _tree_digest(os.path.join(workspace, "runs")) == before

    output = os.path.join(workspace, summary["outputDirectory"])
    assert summary["outputDirectory"].startswith("diagnostics/impact-ledger-")
    assert os.path.isfile(os.path.join(output, il.LEDGER_JSON))
    assert os.path.isfile(os.path.join(output, il.LEDGER_MARKDOWN))
    reassessed, = summary["reassessments"]
    assert reassessed["path"].startswith(summary["outputDirectory"])
    assert os.path.isfile(os.path.join(workspace, reassessed["path"]))
    assert reassessed["changedVerdicts"] == 1


def test_the_ledger_json_is_sorted_and_counts_every_finding(workspace):
    summary = il.build(workspace)
    text = _read(os.path.join(workspace, summary["ledgerFile"]))
    ledger = json.loads(text)
    assert text == json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    assert set(ledger["counts"]) == {f.id for f in il.FINDINGS}
    for block in ledger["counts"].values():
        assert block["total"] == sum(block[e] for e in il.EXPOSURES)
    assert ledger["counts"]["SCI-02"]["exposed"] == 1
    assert ledger["counts"]["SCI-02"]["unaffected"] == 2
    assert ledger["counts"]["SCI-04"]["exposed"] == 2


def test_the_markdown_groups_by_finding_with_counts(workspace):
    summary = il.build(workspace)
    text = _read(os.path.join(workspace, summary["outputDirectory"],
                              il.LEDGER_MARKDOWN))
    for finding in il.FINDINGS:
        assert f"## {finding.id}" in text
    assert "candidate artifact(s)** —" in text
    assert "## Reassessed promotions" in text
    # The document must not let a reader mistake the fixes' test suites for
    # evidence about these artifacts.
    assert "establish nothing about the artifacts below" in text


def test_a_workspace_with_no_candidates_still_reports_every_finding(tmp_path):
    root = str(tmp_path / "empty")
    os.makedirs(os.path.join(root, "runs"))
    _write(os.path.join(root, "WORKSPACE.md"), "#\n")
    summary = il.build(root)
    assert summary["entryCount"] == 0
    assert set(summary["counts"]) == {f.id for f in il.FINDINGS}
    assert all(block["total"] == 0 for block in summary["counts"].values())


# --- 5. the verb --------------------------------------------------------------

def _run(monkeypatch, capsys, argv, root=None):
    if root is not None:
        monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    code = cli.main(argv)
    return code, capsys.readouterr()


def test_the_verb_answers_one_envelope_and_exits_zero_when_exposed(
        workspace, monkeypatch, capsys):
    code, captured = _run(monkeypatch, capsys,
                          ["ledger", "impact", "--json"], workspace)
    # An exposed artifact is the PRODUCT, not a failure.
    assert code == 0
    document = json.loads(captured.out)
    assert document["verb"] == "ledger impact"
    assert document["state"] == "ready"
    assert document["changed"] is True
    assert document["schemaVersion"] == 1
    result = document["result"]
    assert result["outputDirectory"].startswith("diagnostics/impact-ledger-")
    assert result["revisionDating"] == "unavailable"
    assert result["counts"]["SCI-04"]["exposed"] == 2
    assert {f["finding"] for f in result["findings"]} == {
        finding.id for finding in il.FINDINGS}
    # Exactly one JSON document on stdout.
    assert captured.out.strip().startswith("{")
    assert captured.out.strip().endswith("}")


def test_the_verb_refuses_a_root_that_is_not_a_workspace(
        tmp_path, monkeypatch, capsys):
    plain = str(tmp_path / "plain")
    os.makedirs(plain)
    code, captured = _run(monkeypatch, capsys,
                          ["ledger", "impact", "--json"], plain)
    assert code == 66
    document = json.loads(captured.out)
    assert document["state"] == "notFound"
    assert document["error"]["code"] == "workspaceNotFound"
    assert document["error"]["repairAction"]


def test_the_verb_refuses_the_source_checkout(tmp_path, monkeypatch, capsys):
    checkout = str(tmp_path / "checkout")
    os.makedirs(os.path.join(checkout, "Server", "steerlab_server"))
    code, captured = _run(monkeypatch, capsys,
                          ["ledger", "impact", "--json"], checkout)
    assert code == 66
    assert json.loads(captured.out)["error"]["code"] == "rootIsSourceCheckout"


def test_an_undeclared_flag_is_sixty_four_in_the_envelope(
        workspace, monkeypatch, capsys):
    code, captured = _run(monkeypatch, capsys,
                          ["ledger", "impact", "--json", "--nope"], workspace)
    assert code == 64
    document = json.loads(captured.out)
    assert document["state"] == "blocked"
    assert document["error"]["code"] == "unknownFlag"
    assert "--code-checkout" in document["error"]["repairAction"]


def test_a_value_flag_with_no_value_is_sixty_four(workspace, monkeypatch,
                                                  capsys):
    code, _captured = _run(monkeypatch, capsys,
                           ["ledger", "impact", "--code-checkout"], workspace)
    assert code == 64


def test_a_bad_code_checkout_is_sixty_four(workspace, tmp_path, monkeypatch,
                                           capsys):
    code, captured = _run(
        monkeypatch, capsys,
        ["ledger", "impact", "--json", "--code-checkout",
         str(tmp_path / "nowhere")], workspace)
    assert code == 64
    document = json.loads(captured.out)
    assert document["state"] == "blocked"
    assert document["error"]["code"] == "codeCheckoutNotFound"


def test_out_writes_the_same_document_in_human_mode(workspace, tmp_path,
                                                    monkeypatch, capsys):
    target = str(tmp_path / "envelope.json")
    code, captured = _run(monkeypatch, capsys,
                          ["ledger", "impact", "--out", target], workspace)
    assert code == 0
    assert "impact ledger:" in captured.out
    document = json.loads(_read(target))
    assert document["verb"] == "ledger impact"
    assert document["result"]["entryCount"] > 0


def test_the_verb_without_a_subverb_is_sixty_four(monkeypatch, capsys):
    code, _captured = _run(monkeypatch, capsys, ["ledger"])
    assert code == 64

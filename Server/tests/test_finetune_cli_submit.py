"""``steerlab-server finetune submit`` — the evidence-grade LoRA path on the
terminal (open issues §5).

Submission existed only as ``POST /api/finetune/plan`` + ``/submit``, so all
47 trainings were curl'd by hand — and the wire's camelCase against
``LoRAConfig``'s snake_case tripped a first-time caller. The verb drives the
same two functions the routes drive, from ONE wire-shaped request file.

Model-free: the fake scheduler on PATH answers sbatch, and nothing loads a
tokenizer or weights.
"""

import hashlib
import json
import os
import shutil
import stat

import pytest

from steerlab_server import cli
from steerlab_server.api import finetune_submission as ft
from steerlab_server.api.jobs import JobManager

FAKEBIN_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "fakebin")
REVISION = "a" * 40


def _jsonl(rows):
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


TRAIN_TEXT = _jsonl([{"user": f"question {i}", "assistant": f"answer {i}"}
                     for i in range(8)])
VALIDATION_TEXT = _jsonl([{"user": f"held out {i}", "assistant": f"reply {i}"}
                          for i in range(4)])


def _request(**overrides):
    """The WIRE body — the exact JSON the routes accept."""
    body = {
        "schemaVersion": 2,
        "baseModelID": "org/tiny",
        "revision": REVISION,
        "name": "stance-lora-v1",
        "trainingMode": "instructionChat",
        "evidenceGrade": True,
        "dataset": {
            "bundleID": "lora-family-v1",
            "manifestPath": "adapters/manifest.json",
            "manifestHash": _sha("manifest"),
            "files": [
                {"role": "train", "path": "adapters/x/train.jsonl",
                 "sha256": _sha(TRAIN_TEXT), "content": TRAIN_TEXT},
                {"role": "validation", "path": "adapters/x/val.jsonl",
                 "sha256": _sha(VALIDATION_TEXT), "content": VALIDATION_TEXT},
            ],
        },
        "hyperparameters": {"rank": 8, "batchSize": 2, "epochs": 1,
                            "maxSequenceTokens": 512},
        "selectionMetric": "validationLoss",
        "resources": {"gres": "A100", "walltime": "04:00:00",
                      "partition": "gpu", "memory": "64G"},
    }
    body.update(overrides)
    return body


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A Slurm-executor workspace with the committed scheduler doubles on
    PATH and an isolated job store."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    (tmp_path / "meta").mkdir()
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_SLURM_GPU_VRAM", "A100:80,H100:80,L4:24")
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = bindir / name
        shutil.copy(os.path.join(FAKEBIN_SOURCE, name), target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
                     | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep
                       + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(tmp_path / "calls"))
    (tmp_path / "calls").mkdir()
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "515151")
    monkeypatch.delenv("FAKE_SBATCH_FAIL", raising=False)
    return tmp_path


def _write(tmp_path, body, name="finetune-request.json"):
    path = tmp_path / name
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return str(path)


def _document(capsys):
    return json.loads(capsys.readouterr().out)


# --- plan → echo → submit -----------------------------------------------------


def test_plan_only_echoes_the_hash_and_submits_nothing(site, capsys):
    path = _write(site, _request())
    assert cli.main(["finetune", "submit", path, "--plan-only"]) == 0
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["planHash"] == document["planHash"].lower()
    assert len(document["planHash"]) == 64
    assert document["plan"]["schedule"]["totalSteps"] >= 1
    # The echo is on stderr so stdout stays one machine document.
    assert f"planHash {document['planHash']}" in captured.err
    assert JobManager().list() == []
    assert not os.path.isdir(os.path.join(str(site), "runs"))


def test_confirmed_plan_submits_the_same_job_the_route_would(site, capsys):
    path = _write(site, _request())
    assert cli.main(["finetune", "submit", path, "--plan-only"]) == 0
    plan_hash = _document(capsys)["planHash"]

    assert cli.main(["finetune", "submit", path,
                     "--confirm-plan", plan_hash]) == 0
    out = _document(capsys)
    assert out["planHash"] == plan_hash
    assert out["slurmJobID"] == "515151"
    assert out["dryRun"] is False
    job = JobManager().get(out["jobId"])
    assert job.kind == ft.JOB_KIND
    assert job.status == "submitted"
    # The same self-contained job directory the route materializes.
    submission = out["submissionDirectory"]
    assert os.path.isfile(os.path.join(submission, "finetune-config.json"))
    assert json.load(open(os.path.join(submission, "plan.json"),
                          encoding="utf-8"))["planHash"] == plan_hash
    staged = os.path.join(submission, "dataset", "adapters", "x", "train.jsonl")
    assert open(staged, encoding="utf-8").read() == TRAIN_TEXT


def test_the_request_may_carry_its_own_confirmed_hash(site, capsys):
    """A stored, already-confirmed request submits with no extra flags — the
    curl body verbatim."""
    path = _write(site, _request())
    assert cli.main(["finetune", "submit", path, "--plan-only"]) == 0
    plan_hash = _document(capsys)["planHash"]
    path = _write(site, _request(expectedPlanHash=plan_hash), "confirmed.json")
    assert cli.main(["finetune", "submit", path]) == 0
    assert _document(capsys)["planHash"] == plan_hash


def test_dry_run_prepares_without_reaching_sbatch(site, capsys):
    path = _write(site, _request())
    assert cli.main(["finetune", "submit", path, "--plan-only"]) == 0
    plan_hash = _document(capsys)["planHash"]
    assert cli.main(["finetune", "submit", path, "--confirm-plan", plan_hash,
                     "--dry-run"]) == 0
    out = _document(capsys)
    assert out["dryRun"] is True
    assert out["slurmJobID"] is None
    assert JobManager().get(out["jobId"]).status == "prepared"


def test_resource_flags_override_the_request(site, capsys):
    path = _write(site, _request())
    assert cli.main(["finetune", "submit", path, "--plan-only"]) == 0
    plan_hash = _document(capsys)["planHash"]
    assert cli.main(["finetune", "submit", path, "--confirm-plan", plan_hash,
                     "--walltime", "08:00:00", "--gres", "H100",
                     "--dry-run"]) == 0
    out = _document(capsys)
    resources = JobManager().get(out["jobId"]).requested_resources
    assert resources["walltime"] == "08:00:00"
    assert resources["gres"].endswith("H100:1") or "H100" in resources["gres"]
    assert resources["partition"] == "gpu"      # untouched, from the request


# --- refusals -----------------------------------------------------------------


def test_an_unconfirmed_evidence_grade_submit_refuses_with_the_repair(
        site, capsys):
    path = _write(site, _request())
    assert cli.main(["finetune", "submit", path]) == 2
    err = capsys.readouterr().err
    assert "expectedPlanHash" in err
    assert "--plan-only" in err and "--confirm-plan" in err
    assert JobManager().list() == []


def test_a_stale_confirmation_refuses_as_plan_drift(site, capsys):
    path = _write(site, _request())
    assert cli.main(["finetune", "submit", path, "--confirm-plan",
                     "b" * 64]) == 2
    assert "plan drift" in capsys.readouterr().err


def test_two_contradicting_confirmations_refuse(site, capsys):
    path = _write(site, _request(expectedPlanHash="c" * 64))
    assert cli.main(["finetune", "submit", path,
                     "--confirm-plan", "d" * 64]) == 2
    assert "contradicts the request's own expectedPlanHash" \
        in capsys.readouterr().err


def test_a_snake_case_lora_config_is_named_not_cryptically_refused(site,
                                                                   capsys):
    """THE seam §5 reports: the file a submission writes into its job
    directory (and what `finetune plan`/`train` take) is the RESOLVED config,
    not the wire request."""
    path = _write(site, {"base_model_id": "org/tiny",
                         "training_mode": "instruction_chat",
                         "train_paths": ["train.jsonl"]}, "config.json")
    assert cli.main(["finetune", "submit", path]) == 2
    err = capsys.readouterr().err
    assert "resolved LoRAConfig" in err
    assert "base_model_id" in err
    assert "camelCase baseModelID" in err
    assert "finetune plan" in err


def test_a_missing_or_malformed_request_is_a_typed_refusal(site, capsys):
    assert cli.main(["finetune", "submit",
                     str(site / "nope.json")]) == 2
    assert "cannot read" in capsys.readouterr().err
    path = _write(site, _request(baseModelID=None))
    assert cli.main(["finetune", "submit", path]) == 2
    assert "baseModelID is required" in capsys.readouterr().err


def test_argv_is_strict(site, capsys):
    path = _write(site, _request())
    assert cli.main(["finetune", "submit", path, "--dryrun"]) == 64
    assert "unknown flag '--dryrun'" in capsys.readouterr().err
    assert cli.main(["finetune", "submit", path, "--confirm-plan"]) == 64
    assert "--confirm-plan requires a value" in capsys.readouterr().err
    assert cli.main(["finetune", "submit"]) == 64


def test_help_prints_the_surface_and_runs_nothing(site, capsys):
    assert cli.main(["finetune", "--help"]) == 0
    out = capsys.readouterr().out
    assert "finetune submit <finetune-request.json>" in out
    assert "--plan-only" in out and "--confirm-plan" in out
    assert "camelCase" in out
    assert JobManager().list() == []

"""The v2 fine-tune wire: ``/api/finetune/plan``, the daemon ``/train`` route
(v1 regression + v2 accept + the evidence-grade refusal), the capability
block, and the adapter listing's v2 provenance keys.

Contract: ``docs/CLUSTER-LORA-READINESS.md`` §2.1/§3 and the implementation
contract §6. Nothing here loads a model or a tokenizer — that is the whole
point of the plan endpoint, and the daemon route's trainer is stubbed.
"""

import hashlib
import json
import os
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from steerlab_server.api.app import app
from steerlab_server.api.profile import capability_snapshot

client = TestClient(app)


# --- fixtures ---------------------------------------------------------------


def _jsonl(rows):
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file(role, path, text, *, sha=None, content=True):
    return {"role": role, "path": path, "sha256": sha or _sha(text),
            "content": text if content else None}


TRAIN_TEXT = _jsonl([{"user": f"question {i}", "assistant": f"answer {i}"}
                     for i in range(8)])
VALIDATION_TEXT = _jsonl([{"user": f"held out {i}", "assistant": f"reply {i}"}
                          for i in range(4)])


def _hyperparameters(**overrides):
    """Every one of the contract's 21 keys, present-null where the client
    sends null — the wire shape the shipped Swift encoder produces. ``alpha``
    and ``adapterScale`` are two conventions for one knob: the encoder sends
    exactly one of them non-null."""
    values = {
        "rank": 8, "alpha": 16.0, "adapterScale": None,
        "dropout": 0.05, "learningRate": 1e-4,
        "epochs": 1, "maxSteps": None, "batchSize": 2,
        "gradientAccumulation": 1, "warmupSteps": 0, "lrSchedule": "linear",
        "maxGradNorm": 1.0, "weightDecay": 0.0, "seed": 0,
        "maxSequenceTokens": 512, "longDocumentPolicy": "split",
        "chunkOverlapTokens": 64, "evalIntervalSteps": None,
        "checkpointIntervalSteps": None,
        "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "dtype": "auto",
    }
    values.update(overrides)
    return values


def _body(**overrides):
    body = {
        "schemaVersion": 2,
        "baseModelID": "org/tiny",
        "revision": None,
        "name": "stance-lora-v1",
        "trainingMode": "instructionChat",
        "evidenceGrade": False,
        "dataset": {
            "bundleID": "stance-lora-family-v1",
            "manifestPath": "adapters/stance-manifest.json",
            "manifestHash": _sha("manifest"),
            "files": [
                _file("train", "adapters/x/training/train.jsonl", TRAIN_TEXT),
                _file("validation", "adapters/x/validation/val.jsonl",
                      VALIDATION_TEXT),
            ],
        },
        "hyperparameters": _hyperparameters(),
        "selectionMetric": None,
        "controlArm": None,
        "expectedPlanHash": None,
    }
    body.update(overrides)
    return body


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    monkeypatch.delenv("STEERLAB_EXECUTOR", raising=False)
    return tmp_path


# --- /api/finetune/plan ------------------------------------------------------


def test_plan_returns_the_contract_shape_and_a_stable_hash():
    body = client.post("/api/finetune/plan", json=_body()).json()
    plan, plan_hash = body["plan"], body["planHash"]

    # The client pins the schedule's key SET; the rows caveat rides in
    # planNotes, not as a sixth schedule key.
    assert set(plan["schedule"]) == {"totalSteps", "epochs",
                                     "effectiveBatchSize", "warmupSteps",
                                     "lrSchedule"}
    assert plan["schedule"]["effectiveBatchSize"] == 2
    assert plan["schedule"]["totalSteps"] == 4      # 8 rows / batch 2, 1 epoch
    assert plan["trainingMode"] == "instruction_chat"   # wire → python
    assert plan["evidenceGrade"] is False
    assert plan["dataset"]["bundleID"] == "stance-lora-family-v1"
    assert plan["dataset"]["counts"]["trainRows"] == 8
    assert plan["dataset"]["counts"]["validationRows"] == 4
    roles = {f["role"]: f for f in plan["dataset"]["files"]}
    assert roles["train"]["rows"] == 8
    assert roles["train"]["sha256"] == _sha(TRAIN_TEXT)
    assert len(roles["train"]["rowsRoot"]) == 64
    assert any("ROWS" in note for note in plan["planNotes"])

    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    assert plan_hash == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Recomputing the same request answers the same hash — the property the
    # whole confirm-then-submit handshake rests on.
    assert client.post("/api/finetune/plan", json=_body()).json()["planHash"] \
        == plan_hash


def test_plan_reports_the_adapter_scale_it_resolves():
    """``alpha`` is PEFT's numerator. The plan says what it resolves TO, so
    the researcher confirms a multiplier, not a number whose meaning depends
    on knowing the trainer's convention."""
    plan = client.post("/api/finetune/plan", json=_body()).json()["plan"]
    assert plan["adapterScale"] == {
        "rank": 8, "alpha": 16.0,
        "adapterScaleConvention": "peft:lora_alpha/r",
        "effectiveAdapterScale": 2.0,
        # The request spoke ``alpha``: nothing was translated.
        "requestedAdapterScale": None,
        "requestedAdapterScaleConvention": None,
    }


def test_plan_resolves_a_direct_adapter_scale_into_lora_alpha():
    """The Swift/MLX ``scale`` is the multiplier itself. Sent as
    ``adapterScale`` it becomes ``lora_alpha = scale × rank`` HERE, where the
    convention lives — and the plan carries both the number asked for and
    the number that will train."""
    body = _body(hyperparameters=_hyperparameters(alpha=None,
                                                  adapterScale=10.0))
    answer = client.post("/api/finetune/plan", json=body).json()
    assert answer["plan"]["adapterScale"] == {
        "rank": 8, "alpha": 80.0,
        "adapterScaleConvention": "peft:lora_alpha/r",
        "effectiveAdapterScale": 10.0,
        "requestedAdapterScale": 10.0,
        "requestedAdapterScaleConvention": "direct",
    }
    # 10.0 as a direct multiplier and 16.0 as lora_alpha are different
    # treatments, so they are different plans.
    assert answer["planHash"] !=         client.post("/api/finetune/plan", json=_body()).json()["planHash"]
    # ...and the same request plans to the same hash, translation included.
    assert client.post("/api/finetune/plan", json=body).json()["planHash"]         == answer["planHash"]


def test_a_direct_adapter_scale_resolves_against_the_default_rank_too():
    """``rank`` absent (present-null) = the trainer's default, and the
    translation uses THAT rank — the one PEFT will divide by."""
    body = _body(hyperparameters=_hyperparameters(rank=None, alpha=None,
                                                  adapterScale=10.0))
    block = client.post("/api/finetune/plan", json=body).json()["plan"]["adapterScale"]
    assert block["rank"] == 8 and block["alpha"] == 80.0
    assert block["effectiveAdapterScale"] == 10.0 == block["requestedAdapterScale"]


def test_plan_refuses_alpha_and_adapter_scale_together():
    body = _body(hyperparameters=_hyperparameters(alpha=16.0,
                                                  adapterScale=10.0))
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "alpha" in detail and "adapterScale" in detail
    assert "declare exactly one" in detail


@pytest.mark.parametrize("value", [0, -1.5])
def test_plan_refuses_a_non_positive_adapter_scale(value):
    body = _body(hyperparameters=_hyperparameters(alpha=None,
                                                  adapterScale=value))
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 400
    assert "adapterScale must be positive" in response.json()["detail"]


def test_plan_hash_moves_when_the_schedule_moves():
    first = client.post("/api/finetune/plan", json=_body()).json()["planHash"]
    second = client.post("/api/finetune/plan", json=_body(
        hyperparameters=_hyperparameters(epochs=3))).json()["planHash"]
    assert first != second


def test_plan_has_no_side_effects(workspace):
    client.post("/api/finetune/plan", json=_body())
    runs = os.path.join(str(workspace), "runs")
    assert not os.path.isdir(runs) or os.listdir(runs) == []


def test_plan_refuses_an_inline_hash_mismatch():
    body = _body()
    body["dataset"]["files"][0]["sha256"] = "0" * 64
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "adapters/x/training/train.jsonl" in detail
    assert "inline content hashes to" in detail


def test_plan_refuses_unknown_keys_by_name():
    response = client.post("/api/finetune/plan", json=_body(iterations=200))
    assert response.status_code == 400
    assert "iterations" in response.json()["detail"]

    body = _body()
    body["hyperparameters"]["lora_rank"] = 4
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 400
    assert "lora_rank" in response.json()["detail"]

    body = _body()
    body["dataset"]["files"][0]["encoding"] = "utf-8"
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 400
    assert "encoding" in response.json()["detail"]


def test_present_null_reads_exactly_like_absent():
    """Every optional encodes as an explicit JSON null (the shipped Swift
    encoder); a present-null must not override the dataclass default."""
    with_nulls = client.post("/api/finetune/plan", json=_body()).json()
    body = _body()
    for key in ("revision", "selectionMetric", "controlArm", "expectedPlanHash"):
        body.pop(key)
    for key in ("maxSteps", "evalIntervalSteps", "checkpointIntervalSteps"):
        body["hyperparameters"].pop(key)
    without = client.post("/api/finetune/plan", json=body).json()
    assert with_nulls["planHash"] == without["planHash"]


def test_plan_surfaces_the_whole_evidence_gap_at_once():
    body = _body(evidenceGrade=True, revision="main", selectionMetric=None)
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 200
    refusals = response.json()["evidenceRefusals"]
    assert any("40-character commit sha" in r for r in refusals)
    assert any("selection metric" in r for r in refusals)
    # Informational only: the refusals never enter the hashed plan.
    assert "evidenceRefusals" not in response.json()["plan"]


def test_plan_refuses_cross_split_leakage_naming_the_row():
    body = _body()
    body["dataset"]["files"][1] = _file(
        "validation", "adapters/x/validation/val.jsonl", TRAIN_TEXT)
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 400
    assert "also present in the training split" in response.json()["detail"]


def test_plan_refuses_an_absolute_dataset_path():
    body = _body()
    body["dataset"]["files"][0]["path"] = "/etc/passwd"
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 400
    assert "workspace-relative" in response.json()["detail"]


def test_plan_resolves_a_full_sha_revision_verbatim():
    revision = "b" * 40
    plan = client.post("/api/finetune/plan",
                       json=_body(revision=revision)).json()["plan"]
    assert plan["resolvedRevision"] == revision


def test_plan_reads_a_server_resident_file(workspace):
    """``content: null`` = the file already lives in the workspace; the server
    resolves it through the path resolver and hash-verifies the bytes."""
    target = workspace / "adapters" / "x" / "training" / "train.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text(TRAIN_TEXT, encoding="utf-8")
    body = _body()
    body["dataset"]["files"][0]["content"] = None
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 200
    assert response.json()["plan"]["dataset"]["counts"]["trainRows"] == 8

    body["dataset"]["files"][0]["sha256"] = "1" * 64
    response = client.post("/api/finetune/plan", json=body)
    assert response.status_code == 400
    assert "server-resident file hashes to" in response.json()["detail"]


# --- /api/finetune/train (daemon) --------------------------------------------


@pytest.fixture
def stub_trainer(monkeypatch):
    """The daemon route submits a background job; capture the config it would
    have trained instead of loading a model."""
    from steerlab_server.experiment import lora_train

    captured = {"event": threading.Event(), "config": None}

    def fake_train(config, log=None, **kwargs):
        captured["config"] = config
        captured["event"].set()
        return "/tmp/fake-run"

    monkeypatch.setattr(lora_train, "train", fake_train)

    def wait():
        assert captured["event"].wait(10), "training job never started"
        return captured["config"]

    captured["wait"] = wait
    return captured


def test_v1_inline_body_still_trains_the_legacy_way(stub_trainer, workspace):
    response = client.post("/api/finetune/train", json={
        "baseModelID": "org/tiny", "text": "a corpus", "rank": 4,
        "alpha": 8.0, "iterations": 3, "learningRate": 2e-4, "name": "legacy"})
    assert response.status_code == 200
    assert response.json()["jobId"]
    config = stub_trainer["wait"]()
    assert config.training_mode == "legacy_inline"
    assert config.rank == 4 and config.iterations == 3
    assert len(config.document_paths) == 1
    with open(config.document_paths[0], encoding="utf-8") as handle:
        assert handle.read() == "a corpus"


def test_v1_inline_body_resolves_a_direct_adapter_scale(stub_trainer,
                                                        workspace):
    """The legacy route speaks the same two conventions as the v2 block: a
    direct ``adapterScale`` is resolved server-side and stamped, never copied
    into ``alpha`` by the caller."""
    response = client.post("/api/finetune/train", json={
        "baseModelID": "org/tiny", "text": "a corpus", "rank": 4,
        "adapterScale": 2.5, "iterations": 3})
    assert response.status_code == 200
    config = stub_trainer["wait"]()
    assert config.rank == 4
    assert config.alpha == 10.0                    # 2.5 × 4
    assert config.requested_adapter_scale == 2.5


def test_v1_inline_body_refuses_alpha_and_adapter_scale_together():
    response = client.post("/api/finetune/train", json={
        "baseModelID": "org/tiny", "text": "a corpus", "rank": 4,
        "alpha": 8.0, "adapterScale": 2.5})
    assert response.status_code == 400
    assert "declare exactly one" in response.json()["detail"]


def test_v1_body_without_documents_still_refuses():
    response = client.post("/api/finetune/train", json={"baseModelID": "org/t"})
    assert response.status_code == 400
    assert "documentPaths" in response.json()["detail"]


def test_v2_exploratory_body_runs_on_the_daemon(stub_trainer):
    response = client.post("/api/finetune/train", json=_body())
    assert response.status_code == 200
    config = stub_trainer["wait"]()
    assert config.training_mode == "instruction_chat"
    assert config.train_paths == ["adapters/x/training/train.jsonl"]
    assert config.validation_paths == ["adapters/x/validation/val.jsonl"]
    assert config.dataset_root and os.path.isfile(
        os.path.join(config.dataset_root, config.train_paths[0]))
    assert config.evidence_grade is False


def test_v2_direct_adapter_scale_reaches_the_trainer_resolved(stub_trainer):
    body = _body(hyperparameters=_hyperparameters(alpha=None,
                                                  adapterScale=10.0))
    assert client.post("/api/finetune/train", json=body).status_code == 200
    config = stub_trainer["wait"]()
    assert config.rank == 8
    assert config.alpha == 80.0
    assert config.requested_adapter_scale == 10.0


def test_v2_evidence_grade_is_refused_on_the_daemon_route():
    response = client.post("/api/finetune/train",
                           json=_body(evidenceGrade=True, revision="c" * 40,
                                      selectionMetric="validationLoss"))
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "/api/finetune/submit" in detail
    assert "checkpoint/resume" in detail


# --- capabilities + adapter listing ------------------------------------------


def test_capability_block_carries_every_readiness_flag(monkeypatch):
    monkeypatch.delenv("STEERLAB_EXECUTOR", raising=False)
    block = capability_snapshot()["remoteFineTune"]
    assert block == {
        "schemaVersion": 2,
        "explicitSplits": True,
        "documentRows": True,
        "instructionChatAssistantMask": True,
        "checkpointResume": True,
        "revisionPinRequired": True,
        "walltimePreflight": True,
        "planEndpoint": True,
        "directAdapterScale": True,
        "slurmSubmission": False,
    }
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    assert capability_snapshot()["remoteFineTune"]["slurmSubmission"] is True
    assert "finetune-train" in capability_snapshot()["availableJobTypes"]


def test_adapter_listing_exposes_v2_sidecar_provenance(workspace):
    run = workspace / "runs" / "20260812-lora"
    adapter = run / "stance-adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    (run / "stance-adapter.json").write_text(json.dumps({
        "substrate": "python-hf", "adapterFormat": "hf-peft-lora",
        "schemaVersion": 2, "evidenceGrade": True,
        "trainingMode": "instruction_chat",
        "adapterBytesHash": "a" * 64, "adapterConfigHash": "b" * 64,
        "dataset": {"manifestHash": "c" * 64},
        "selectedCheckpoint": {"step": 12, "metric": "validationLoss",
                               "value": 0.5, "reason": "best"},
    }), encoding="utf-8")

    entries = client.get("/api/adapters").json()["adapters"]
    entry = next(e for e in entries if e["name"] == "stance-adapter")
    assert entry["substrate"] == "python-hf"
    assert entry["adapterFormat"] == "hf-peft-lora"
    assert entry["schemaVersion"] == 2
    assert entry["evidenceGrade"] is True
    assert entry["trainingMode"] == "instruction_chat"
    assert entry["adapterBytesHash"] == "a" * 64
    assert entry["adapterConfigHash"] == "b" * 64
    assert entry["datasetManifestHash"] == "c" * 64
    assert entry["selectedCheckpoint"] == {"step": 12}


def test_adapter_listing_tolerates_a_v1_sidecar(workspace):
    run = workspace / "runs" / "20260101-old"
    adapter = run / "legacy"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    (run / "legacy.json").write_text(
        json.dumps({"substrate": "python-hf"}), encoding="utf-8")
    entry = next(e for e in client.get("/api/adapters").json()["adapters"]
                 if e["name"] == "legacy")
    assert entry["substrate"] == "python-hf"
    assert "schemaVersion" not in entry
    assert "datasetManifestHash" not in entry

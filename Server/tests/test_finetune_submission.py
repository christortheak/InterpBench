"""The evidence-grade fine-tune path: Slurm submission, the LoRA preflight
(memory AND walltime — readiness plan §0 amendment 4), and the
``finetune execute`` child verb with checkpoint/resume.

The submission and preflight tests are model-free (the HF-cache snapshot is
stubbed). The execute tests train the same tiny local Llama
``test_lora_train_v2`` uses — CPU only, seconds per test — because what is
under test is the child's exit-code contract, not learning.
"""

import hashlib
import json
import os
import shutil
import stat
from dataclasses import asdict

import pytest

from steerlab_server.api import finetune_submission as ft
from steerlab_server.api.executors import SlurmResources
from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.api.profile import ServerProfile
from steerlab_server.api.safe_paths import SafePathResolver
from steerlab_server.experiment import lora_train

FAKEBIN_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakebin")
REVISION = "a" * 40


# --- request fixtures --------------------------------------------------------


def _jsonl(rows):
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


TRAIN_TEXT = _jsonl([{"user": f"question {i}", "assistant": f"answer {i}"}
                     for i in range(8)])
VALIDATION_TEXT = _jsonl([{"user": f"held out {i}", "assistant": f"reply {i}"}
                          for i in range(4)])


def _body(**overrides):
    body = {
        "schemaVersion": 2,
        "baseModelID": "org/tiny",
        "revision": REVISION,
        "name": "stance-lora-v1",
        "trainingMode": "instructionChat",
        "evidenceGrade": True,
        "dataset": {
            "bundleID": "stance-lora-family-v1",
            "manifestPath": "adapters/stance-manifest.json",
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
        "controlArm": None,
        "expectedPlanHash": None,
        "resources": {"gres": "A100", "walltime": "04:00:00",
                      "partition": "gpu", "memory": "64G"},
        "force": False,
    }
    body.update(overrides)
    return body


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_SLURM_GPU_VRAM", "A100:80,H100:80,L4:24")
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)
    return tmp_path


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """The committed sbatch/squeue/sacct/scancel doubles on PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = bindir / name
        shutil.copy(os.path.join(FAKEBIN_SOURCE, name), target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP
                     | stat.S_IXOTH)
    log_dir = tmp_path / "calls"
    log_dir.mkdir()
    state_file = tmp_path / "slurm-state.json"
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "515151")
    monkeypatch.delenv("FAKE_SBATCH_FAIL", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_SACCT", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_SQUEUE", raising=False)

    class Handle:
        def set_state(self, job_id, state, exit_code="0:0", queue=None):
            table = {}
            if state_file.exists():
                table = json.loads(state_file.read_text(encoding="utf-8"))
            table[str(job_id)] = {"state": state, "exit": exit_code,
                                  "queue": queue}
            state_file.write_text(json.dumps(table), encoding="utf-8")

        def calls(self, binary):
            path = log_dir / f"{binary}.calls"
            return path.read_text(encoding="utf-8").splitlines() \
                if path.exists() else []

    return Handle()


@pytest.fixture
def jobs(tmp_path):
    return JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))


def _resolver():
    return SafePathResolver(ServerProfile.from_env())


def _confirmed(body):
    """A body carrying the plan hash the server would recompute.

    ``/api/finetune/plan`` takes the REQUEST only — the submit-only keys are
    stripped, exactly as the shipped Swift client does before planning.
    """
    plan_body = {k: v for k, v in body.items()
                 if k not in ("resources", "force", "dryRun")}
    plan = ft.plan_response(plan_body, resolver=_resolver())
    return {**body, "expectedPlanHash": plan["planHash"]}, plan


# --- submission --------------------------------------------------------------


def test_submit_records_an_external_job_with_the_stamps_resubmit_needs(
        workspace, fake_slurm, jobs):
    body, plan = _confirmed(_body())
    out = ft.submit_finetune(body, jobs=jobs, resolver=_resolver())

    assert out["planHash"] == plan["planHash"]
    assert out["slurmJobID"] == "515151"
    job = jobs.get(out["jobId"])
    assert job.kind == "finetune-train"
    assert job.status == "submitted"
    assert job.executor == "slurm"

    rr = job.requested_resources
    # Auto-resubmit-on-checkpoint finds the script through THIS stamp: the
    # reconciler replaces job.result with the child record, so scriptPath and
    # recordsDirectory must live on requested_resources.
    assert os.path.isfile(rr["scriptPath"])
    assert rr["recordsDirectory"].endswith("records")
    assert rr["modelID"] == "org/tiny"
    assert rr["preflight"]["verdict"] in ("ok", "warn")
    assert "auto_resubmit" in rr and "walltime" in rr

    # The job directory is self-contained: staged dataset bytes, the resolved
    # config, and the confirmed plan.
    submission = out["submissionDirectory"]
    staged = os.path.join(submission, "dataset", "adapters", "x", "train.jsonl")
    assert open(staged, encoding="utf-8").read() == TRAIN_TEXT
    stored = json.load(open(os.path.join(submission, "finetune-config.json"),
                            encoding="utf-8"))
    assert stored["training_mode"] == "instruction_chat"
    assert stored["evidence_grade"] is True
    assert stored["run_directory"] == os.path.join(submission, "run")
    assert json.load(open(os.path.join(submission, "plan.json"),
                          encoding="utf-8"))["planHash"] == plan["planHash"]

    # The sbatch command is the execute verb against that directory.
    command = job.result["command"]
    assert command[-5:-3] == ["finetune", "execute"]
    assert command[-3] == submission
    assert command[-2] == "--record"
    script = open(rr["scriptPath"], encoding="utf-8").read()
    assert "finetune" in script and "execute" in script
    # USR1 forwarding is what makes the 85 contract reachable.
    assert "kill -USR1" in script


def test_submit_stamps_the_requested_adapter_scale_on_config_and_plan(
        workspace, fake_slurm, jobs):
    """A direct ``adapterScale`` (the Swift/MLX ``scale`` convention) is
    resolved into PEFT's ``lora_alpha`` by the server, and the job directory
    records both: the number that trains and the number that was asked for."""
    body, plan = _confirmed(_body(hyperparameters={
        "rank": 8, "adapterScale": 10.0, "batchSize": 2, "epochs": 1,
        "maxSequenceTokens": 512}))
    out = ft.submit_finetune(body, jobs=jobs, resolver=_resolver())
    submission = out["submissionDirectory"]

    stored = json.load(open(os.path.join(submission, "finetune-config.json"),
                            encoding="utf-8"))
    assert stored["rank"] == 8
    assert stored["alpha"] == 80.0                       # 10.0 × 8
    assert stored["requested_adapter_scale"] == 10.0
    # The stored config round-trips through the execute side's loader (the
    # consistency check in LoRAConfig accepts a stamp that agrees).
    config, _run = ft.load_job_config(submission)
    assert config.alpha == 80.0 and config.requested_adapter_scale == 10.0

    written = json.load(open(os.path.join(submission, "plan.json"),
                             encoding="utf-8"))
    assert written["plan"]["adapterScale"] == {
        "rank": 8, "alpha": 80.0,
        "adapterScaleConvention": "peft:lora_alpha/r",
        "effectiveAdapterScale": 10.0,
        "requestedAdapterScale": 10.0,
        "requestedAdapterScaleConvention": "direct",
    }
    assert written["planHash"] == plan["planHash"] == out["planHash"]


def test_submit_refuses_a_stale_plan_hash(workspace, fake_slurm, jobs):
    body, _plan = _confirmed(_body())
    drifted = {**body, "hyperparameters": {**body["hyperparameters"],
                                           "epochs": 5}}
    with pytest.raises(ft.FineTuneRequestError) as err:
        ft.submit_finetune(drifted, jobs=jobs, resolver=_resolver())
    assert "plan drift" in str(err.value)


def test_submit_requires_a_confirmed_plan_for_evidence(workspace, fake_slurm,
                                                       jobs):
    with pytest.raises(ft.FineTuneRequestError) as err:
        ft.submit_finetune(_body(), jobs=jobs, resolver=_resolver())
    assert "expectedPlanHash" in str(err.value)


def test_submit_surfaces_every_evidence_refusal_at_once(workspace, fake_slurm,
                                                        jobs):
    body = _body(revision="main", selectionMetric=None)
    with pytest.raises(ft.FineTuneRequestError) as err:
        ft.submit_finetune(body, jobs=jobs, resolver=_resolver())
    message = str(err.value)
    assert "40-character commit sha" in message
    assert "selection metric" in message


def test_a_refused_submission_leaves_no_job_directory(workspace, fake_slurm,
                                                      jobs):
    with pytest.raises(ft.FineTuneRequestError):
        ft.submit_finetune(_body(revision="main"), jobs=jobs,
                           resolver=_resolver())
    runs = workspace / "runs"
    assert not runs.is_dir() or not any(
        entry.name.count("submit-finetune") for entry in runs.iterdir())
    assert jobs.list() == []


def test_submit_refuses_without_a_slurm_executor(tmp_path, monkeypatch, jobs):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")
    body, _ = _confirmed(_body())
    with pytest.raises(ft.FineTuneRequestError) as err:
        ft.submit_finetune(body, jobs=jobs, resolver=_resolver())
    assert "STEERLAB_EXECUTOR=slurm" in str(err.value)
    assert "'local'" in str(err.value)


def test_an_exploratory_submission_needs_no_plan_hash(workspace, fake_slurm,
                                                      jobs):
    out = ft.submit_finetune(_body(evidenceGrade=False, revision=None,
                                   selectionMetric=None),
                             jobs=jobs, resolver=_resolver())
    assert jobs.get(out["jobId"]).status == "submitted"


def _failing_preflight(*_args, **_kwargs):
    return {"checks": [{"id": "loraMemoryFit", "status": "fail",
                        "message": "needs more VRAM than the GPU has",
                        "data": None}],
            "verdict": "fail"}


def test_failing_preflight_blocks_unless_forced(workspace, fake_slurm, jobs,
                                                monkeypatch):
    monkeypatch.setattr(ft, "lora_preflight", _failing_preflight)
    body, _ = _confirmed(_body())
    with pytest.raises(ft.PreflightRejection) as err:
        ft.submit_finetune(body, jobs=jobs, resolver=_resolver())
    assert err.value.preflight["verdict"] == "fail"
    assert "force=true" in str(err.value)

    out = ft.submit_finetune({**body, "force": True}, jobs=jobs,
                             resolver=_resolver())
    job = jobs.get(out["jobId"])
    assert job.requested_resources["preflightOverridden"] is True
    assert job.result["preflightOverridden"] is True
    assert any("PREFLIGHT OVERRIDDEN" in line for line in job._logs)


def test_a_checkpointed_finetune_auto_resubmits_its_own_script(
        workspace, fake_slurm, jobs):
    """The whole survivability chain in one pass: exit 85 → sacct FAILED with
    exit ``85:0`` → the reconciler's non-terminal ``checkpointed`` status →
    auto-resubmit re-sbatches THE SAME run.sbatch, found through the
    ``scriptPath`` stamp on requestedResources."""
    body, _ = _confirmed(_body(resources={"gres": "A100",
                                          "walltime": "04:00:00",
                                          "autoResubmit": True}))
    out = ft.submit_finetune(body, jobs=jobs, resolver=_resolver())
    job = jobs.get(out["jobId"])
    script = job.requested_resources["scriptPath"]
    assert len(fake_slurm.calls("sbatch")) == 1

    fake_slurm.set_state("515151", "FAILED", exit_code="85:0")
    assert jobs.poll_slurm() >= 1
    assert job.status == "checkpointed"
    assert job.result["resubmittedAs"]
    child = jobs.get(job.result["resubmittedAs"])
    assert child.kind == "finetune-train"
    assert child.requested_resources["resubmitOf"] == job.id
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 2 and script in calls[1]


def test_dry_run_prepares_without_sbatch(workspace, jobs):
    body, _ = _confirmed(_body())
    out = ft.submit_finetune({**body, "dryRun": True}, jobs=jobs,
                             resolver=_resolver())
    assert out["slurmJobID"] is None
    assert jobs.get(out["jobId"]).status == "prepared"


# --- the strength knob: two conventions, one meaning --------------------------


@pytest.mark.parametrize("alpha, adapter_scale, rank, expected", [
    (16.0, None, 8, (16.0, None)),      # alpha declared: passes through
    (None, 10.0, 8, (80.0, 10.0)),      # direct: resolved to lora_alpha
    (None, 2.5, 4, (10.0, 2.5)),
    (None, None, 8, (None, None)),      # neither: the dataclass default
])
def test_resolve_adapter_scale_turns_one_declared_knob_into_alpha(
        alpha, adapter_scale, rank, expected):
    assert ft.resolve_adapter_scale(alpha=alpha, adapter_scale=adapter_scale,
                                    rank=rank) == expected


def test_resolve_adapter_scale_refuses_both_conventions_at_once():
    with pytest.raises(ft.FineTuneRequestError) as err:
        ft.resolve_adapter_scale(alpha=16.0, adapter_scale=2.0, rank=8)
    message = str(err.value)
    assert "alpha" in message and "adapterScale" in message
    assert "declare exactly one" in message


@pytest.mark.parametrize("value", [0, -3.0, float("nan"), "ten"])
def test_resolve_adapter_scale_refuses_a_value_that_cannot_be_a_multiplier(
        value):
    with pytest.raises(ft.FineTuneRequestError) as err:
        ft.resolve_adapter_scale(alpha=None, adapter_scale=value, rank=8)
    assert "adapterScale" in str(err.value)


def test_adapter_sidecar_stamps_what_was_requested_beside_what_it_became():
    """Pure: the sidecar dict, no training. With ``effectiveAdapterScale``
    beside them, the stamps put the translation on the artifact's face."""
    def sidecar(**settings):
        return lora_train.adapter_sidecar_dict(
            lora_train.LoRAConfig(base_model_id="org/m", **settings),
            name="a", provenance=[], train_chunk_count=0, final_loss=None,
            training_dtype_name="bfloat16")

    spoke_alpha = sidecar(rank=8, alpha=16.0)
    assert spoke_alpha["alpha"] == 16.0
    assert spoke_alpha["adapterScaleConvention"] == "peft:lora_alpha/r"
    assert spoke_alpha["effectiveAdapterScale"] == 2.0
    assert spoke_alpha["requestedAdapterScale"] is None
    assert spoke_alpha["requestedAdapterScaleConvention"] is None

    spoke_direct = sidecar(rank=2, alpha=4.0, requested_adapter_scale=2.0)
    assert spoke_direct["alpha"] == 4.0
    assert spoke_direct["requestedAdapterScale"] == 2.0
    assert spoke_direct["requestedAdapterScaleConvention"] == "direct"
    # The request was honored: what it asked for is what PEFT will apply.
    assert spoke_direct["effectiveAdapterScale"] == \
        spoke_direct["requestedAdapterScale"]


def test_a_stamp_that_disagrees_with_alpha_is_refused_wherever_a_config_is_built(
        tmp_path):
    """The stamp is provenance for ``alpha``; a config (or a hand-edited job
    file) whose stamp cannot have produced its ``alpha`` is refused by name,
    not carried into a sidecar that would then lie."""
    with pytest.raises(lora_train.LoRATrainError) as err:
        lora_train.LoRAConfig(base_model_id="org/m", rank=8, alpha=16.0,
                              requested_adapter_scale=10.0)
    assert "requested_adapter_scale 10.0" in str(err.value)
    assert "alpha is 16.0" in str(err.value)

    job = tmp_path / "job"
    job.mkdir()
    (job / "finetune-config.json").write_text(json.dumps({
        "base_model_id": "org/m", "rank": 8, "alpha": 16.0,
        "requested_adapter_scale": 10.0}), encoding="utf-8")
    with pytest.raises(ft.FineTuneRequestError) as err:
        ft.load_job_config(str(job))
    assert "requested_adapter_scale" in str(err.value)

    # An older job config carries no stamp at all and still loads.
    (job / "finetune-config.json").write_text(json.dumps({
        "base_model_id": "org/m", "rank": 8, "alpha": 16.0}), encoding="utf-8")
    config, _run = ft.load_job_config(str(job))
    assert config.requested_adapter_scale is None


# --- preflight ---------------------------------------------------------------


def _snapshot(layers=62, hidden=5376, weights_bytes=54 * 1024**3):
    return {
        "snapshotPath": "/cache/snap", "revision": REVISION,
        "weightsBytes": weights_bytes,
        "config": {"num_hidden_layers": layers, "hidden_size": hidden,
                   "torch_dtype": "bfloat16"},
    }


@pytest.fixture
def cached_27b(monkeypatch):
    from steerlab_server.api import housekeeping
    monkeypatch.setattr(housekeeping, "model_snapshot_info",
                        lambda *_a, **_k: _snapshot())
    monkeypatch.setattr(housekeeping, "throughput_lookup",
                        lambda *_a, **_k: None)


def _config(**overrides):
    settings = dict(base_model_id="google/gemma-3-27b-it", revision=REVISION,
                    training_mode="instruction_chat", rank=8, batch_size=1,
                    max_sequence_tokens=512,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    settings.update(overrides)
    return lora_train.LoRAConfig(**settings)


def _resources(**overrides):
    settings = dict(gres="gpu:A100:1", walltime="24:00:00",
                    gpu_vram_gb={"A100": 80, "L4": 24})
    settings.update(overrides)
    return SlurmResources(**settings)


def test_memory_fit_accounts_for_weights_adapter_and_activations(cached_27b):
    check = ft._check_lora_memory_fit(_config(), dtype="bfloat16",
                                      revision=REVISION,
                                      resources=_resources())
    assert check["id"] == "loraMemoryFit"
    assert check["status"] == "ok"
    data = check["data"]
    # bf16 weights on disk → parameter count is bytes / 2.
    assert data["baseParameters"] == 54 * 1024**3 // 2
    # LoRA A+B on 4 projections of 62 layers at rank 8.
    assert data["adapterParameters"] == 62 * 4 * 2 * 8 * 5376
    # weight + gradient (bf16) + two fp32 AdamW moments.
    assert data["adapterStateBytes"] == data["adapterParameters"] * 12
    assert data["checkpointDiskBytes"] > 0
    assert data["estimateBytes"] > data["weightsRuntimeBytes"]


@pytest.mark.parametrize("field,values", [
    ("batch_size", (1, 2, 4)),
    ("max_sequence_tokens", (512, 1024, 4096)),
])
def test_memory_estimate_is_monotonic_in_batch_and_sequence(cached_27b, field,
                                                            values):
    estimates = [
        ft._check_lora_memory_fit(_config(**{field: value}), dtype="bfloat16",
                                  revision=REVISION,
                                  resources=_resources())["data"]["estimateBytes"]
        for value in values
    ]
    assert estimates == sorted(estimates)
    assert estimates[0] < estimates[-1]


def test_memory_fit_fails_when_the_gpu_cannot_hold_it(cached_27b):
    check = ft._check_lora_memory_fit(
        _config(batch_size=8, max_sequence_tokens=4096), dtype="bfloat16",
        revision=REVISION, resources=_resources())
    assert check["status"] == "fail"
    assert "lower batchSize or maxSequenceTokens" in check["message"]
    assert "silent fallback" in check["message"]
    # The remedy may not name a mode the trainer does not have.
    assert "declare a multi-GPU/QLoRA mode" not in check["message"]
    assert "no multi-GPU or QLoRA training mode is implemented" in check["message"]


def test_memory_fit_budgets_one_gpu_because_the_trainer_is_single_device(
        cached_27b):
    """External review finding 4: the budget used to POOL VRAM across the
    requested GPUs while ``lora_train`` does ``model.to(device)`` — so
    gpu:A40:2 passed at 80 GB, the whole model then landed on cuda:0, and the
    job OOM'd after the queue wait. One GPU is the budget."""
    check = ft._check_lora_memory_fit(
        _config(), dtype="bfloat16", revision=REVISION,
        resources=_resources(gres="gpu:A40:2", gpu_vram_gb={"A40": 40}))
    data = check["data"]
    # Pooled, this fit (2 × 40 GB); per GPU it does not.
    assert data["estimateBytes"] > 40 * 1024**3
    assert data["estimateBytes"] < 80 * 1024**3
    assert check["status"] == "fail"
    assert data["vramPerGpuBytes"] == 40 * 1024**3
    assert data["vramBytes"] == data["vramPerGpuBytes"]
    assert data["gpuCount"] == 2
    assert "one A40 has 40 GB" in check["message"]


def test_memory_fit_passes_on_a_single_sufficient_gpu(cached_27b):
    check = ft._check_lora_memory_fit(_config(), dtype="bfloat16",
                                      revision=REVISION,
                                      resources=_resources(gres="gpu:A100:1"))
    assert check["status"] == "ok"
    assert check["data"]["vramPerGpuBytes"] == 80 * 1024**3
    assert check["data"]["gpuCount"] == 1
    assert "WARNING" not in check["message"]


def test_memory_fit_warns_that_extra_gpus_sit_idle_even_when_it_fits(cached_27b):
    check = ft._check_lora_memory_fit(_config(), dtype="bfloat16",
                                      revision=REVISION,
                                      resources=_resources(gres="gpu:A100:2"))
    assert check["status"] == "ok"          # it fits on one card
    assert check["data"]["gpuCount"] == 2
    assert "WARNING" in check["message"]
    assert "single-device" in check["message"]
    assert "request gpu:A100:1" in check["message"]


def test_memory_fit_degrades_when_the_model_is_not_cached(monkeypatch):
    from steerlab_server.api import housekeeping
    monkeypatch.setattr(housekeeping, "model_snapshot_info",
                        lambda *_a, **_k: None)
    check = ft._check_lora_memory_fit(_config(), dtype="bfloat16",
                                      revision=REVISION, resources=_resources())
    assert check["status"] == "warn"
    assert "HF cache" in check["message"]


def test_memory_fit_degrades_without_a_vram_entry(cached_27b):
    check = ft._check_lora_memory_fit(
        _config(), dtype="bfloat16", revision=REVISION,
        resources=_resources(gres="gpu:H100:1"))
    assert check["status"] == "warn"
    assert "no VRAM entry for H100" in check["message"]


def test_walltime_estimates_from_the_declared_throughput_table(cached_27b):
    check = ft._check_lora_walltime(_config(), total_steps=1000,
                                    dtype="bfloat16", revision=REVISION,
                                    resources=_resources(),
                                    profile=ServerProfile.from_env())
    assert check["status"] == "ok"
    data = check["data"]
    assert data["basis"] == "declaredThroughput"
    assert data["tokensPerStep"] == 512
    assert data["achievedTFLOPs"] == ft.LORA_TRAINING_TFLOPS["A100"]
    # 6 · N · T / achieved FLOPs
    expected = (6.0 * data["baseParameters"] * 512) / (150.0 * 1e12)
    assert data["secondsPerStep"] == pytest.approx(expected, rel=1e-3)


def test_walltime_refuses_a_schedule_the_allocation_cannot_fit(cached_27b):
    check = ft._check_lora_walltime(_config(), total_steps=1_000_000,
                                    dtype="bfloat16", revision=REVISION,
                                    resources=_resources(walltime="04:00:00"),
                                    profile=ServerProfile.from_env())
    assert check["status"] == "fail"
    assert "exceeds the requested walltime 04:00:00" in check["message"]


def test_walltime_degrades_to_warn_on_an_unknown_gpu(cached_27b):
    check = ft._check_lora_walltime(_config(), total_steps=100,
                                    dtype="bfloat16", revision=REVISION,
                                    resources=_resources(gres="gpu:B200:1"),
                                    profile=ServerProfile.from_env())
    assert check["status"] == "warn"
    assert "no training-throughput constant" in check["message"]


def test_walltime_prefers_an_observed_training_throughput(cached_27b,
                                                          monkeypatch):
    from steerlab_server.api import housekeeping
    monkeypatch.setattr(housekeeping, "throughput_lookup",
                        lambda *_a, **_k: {"stepsPerHour": 1200.0})
    check = ft._check_lora_walltime(_config(), total_steps=600,
                                    dtype="bfloat16", revision=REVISION,
                                    resources=_resources(),
                                    profile=ServerProfile.from_env())
    assert check["data"]["basis"] == "observed"
    assert check["data"]["secondsPerStep"] == pytest.approx(3.0)


def test_preflight_reports_all_four_checks_and_never_raises(cached_27b,
                                                            monkeypatch):
    from steerlab_server.api import housekeeping
    monkeypatch.setattr(housekeeping, "parse_gpu_type",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    report = ft.lora_preflight(_config(), total_steps=10, dtype="bfloat16",
                               revision=REVISION, resources=_resources(),
                               profile=ServerProfile.from_env())
    ids = [c["id"] for c in report["checks"]]
    assert ids == ["gpuRequest", "loraMemoryFit", "loraWalltime",
                   "maintenanceWindow"]
    broken = [c for c in report["checks"] if c["id"].startswith("lora")]
    assert all(c["status"] == "warn" for c in broken)
    assert all("check could not run" in c["message"] for c in broken)
    assert report["verdict"] == "warn"


def test_preflight_warns_when_no_gpu_is_requested(cached_27b):
    report = ft.lora_preflight(_config(), total_steps=10, dtype="bfloat16",
                               revision=REVISION,
                               resources=_resources(gres=None),
                               profile=ServerProfile.from_env())
    gpu = next(c for c in report["checks"] if c["id"] == "gpuRequest")
    assert gpu["status"] == "warn"


# --- finetune execute --------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi"]


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (LlamaConfig, LlamaForCausalLM,
                              PreTrainedTokenizerFast)

    directory = tmp_path_factory.mktemp("tiny-model-execute")
    vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    vocab.update({word: 4 + index for index, word in enumerate(WORDS)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.chat_template = (
        "{% for m in messages %}[BOS] {{ m['content'] }} [EOS] {% endfor %}"
        "{% if add_generation_prompt %}[BOS] {% endif %}")
    torch.manual_seed(11)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=len(vocab),
        max_position_embeddings=128))
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    return str(directory)


def _words(start, width):
    return " ".join(WORDS[(start + k) % len(WORDS)] for k in range(width))


def _job_directory(tmp_path, tiny_model_path, **overrides):
    """A job directory shaped exactly like ``submit_finetune`` materializes."""
    job = tmp_path / "job"
    dataset = job / "dataset" / "adapters" / "x"
    dataset.mkdir(parents=True)
    train = dataset / "train.jsonl"
    validation = dataset / "val.jsonl"
    train.write_text(_jsonl([
        {"user": _words(i, 2), "assistant": WORDS[(i + 2) % len(WORDS)],
         "id": f"t{i}"} for i in range(8)]), encoding="utf-8")
    validation.write_text(_jsonl([
        {"user": _words(i + 100, 3), "assistant": WORDS[(i + 103) % len(WORDS)],
         "id": f"v{i}"} for i in range(4)]), encoding="utf-8")
    relative = {"adapters/x/train.jsonl": str(train),
                "adapters/x/val.jsonl": str(validation)}
    settings = dict(
        base_model_id=tiny_model_path, revision=REVISION,
        training_mode="instruction_chat",
        train_paths=["adapters/x/train.jsonl"],
        validation_paths=["adapters/x/val.jsonl"],
        expected_hashes={key: hashlib.sha256(
            open(path, "rb").read()).hexdigest()
            for key, path in relative.items()},
        dataset_root=str(job / "dataset"),
        rank=2, alpha=4.0, dropout=0.0, learning_rate=5e-3,
        target_modules=["q_proj", "v_proj"], batch_size=2,
        gradient_accumulation=1, epochs=2, seed=7, max_sequence_tokens=64,
        chunk_overlap_tokens=8, dtype="float32", device="cpu",
        selection_metric=lora_train.VALIDATION_LOSS, evidence_grade=True,
        output_name="adapter")
    settings.update(overrides)
    stored = asdict(lora_train.LoRAConfig(**settings))
    stored["run_directory"] = str(job / "run")
    (job / "finetune-config.json").write_text(
        json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8")
    return job


class _FireAfter:
    """A CheckpointFlag stand-in that fires after ``after`` optimizer steps."""

    after = 1

    def __init__(self):
        self.seen = 0

    def install(self):
        return self

    @property
    def requested(self):
        self.seen += 1
        return self.seen > self.after


def test_execute_trains_into_the_job_directory_and_writes_a_record(
        tmp_path, tiny_model_path):
    from steerlab_server import cli

    job = _job_directory(tmp_path, tiny_model_path)
    record = tmp_path / "records" / "job1.json"
    assert cli.main(["finetune", "execute", str(job), "--record",
                     str(record)]) == 0

    run = job / "run"
    assert (run / "adapter" / "adapter_config.json").is_file()
    assert (run / "adapter.json").is_file()          # the sidecar = complete
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["kind"] == "finetune-execute"
    assert payload["result"]["runDirectory"] == str(run)
    assert payload["result"]["resumed"] is False


def test_execute_checkpoints_with_exit_85_then_resumes_to_completion(
        tmp_path, tiny_model_path, monkeypatch):
    from steerlab_server import cli
    from steerlab_server.experiment import resume

    job = _job_directory(tmp_path, tiny_model_path)
    record = tmp_path / "records" / "job2.json"

    monkeypatch.setattr(resume, "CheckpointFlag", _FireAfter)
    assert cli.main(["finetune", "execute", str(job), "--record",
                     str(record)]) == resume.CHECKPOINT_EXIT_CODE
    # No child record on a checkpoint: exit 85 is what the reconciler reads as
    # the non-terminal "checkpointed" status (executors.map_slurm_state).
    assert not record.exists()
    state = resume.read_state(str(job / "run"))
    assert state["verb"] == "lora-train" and state["step"] >= 1

    monkeypatch.undo()
    assert cli.main(["finetune", "execute", str(job), "--record",
                     str(record)]) == 0
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["result"]["resumed"] is True
    assert (job / "run" / "adapter.json").is_file()
    sidecar = json.loads((job / "run" / "adapter.json").read_text(encoding="utf-8"))
    assert len(sidecar["resumeLineage"]) == 1


def test_execute_carries_the_requested_scale_into_the_sidecar(
        tmp_path, tiny_model_path):
    """End to end on the execute side: a job whose config was resolved from a
    direct ``adapterScale`` trains under PEFT's ``alpha`` and writes a sidecar
    on which the request and the resolved multiplier agree."""
    from steerlab_server import cli

    # rank 2 (the fixture's) with alpha 4.0 = a direct scale of 2.0.
    job = _job_directory(tmp_path, tiny_model_path, alpha=4.0,
                         requested_adapter_scale=2.0)
    assert cli.main(["finetune", "execute", str(job)]) == 0
    sidecar = json.loads((job / "run" / "adapter.json").read_text(
        encoding="utf-8"))
    assert sidecar["rank"] == 2 and sidecar["alpha"] == 4.0
    assert sidecar["adapterScaleConvention"] == "peft:lora_alpha/r"
    assert sidecar["effectiveAdapterScale"] == 2.0
    assert sidecar["requestedAdapterScale"] == 2.0
    assert sidecar["requestedAdapterScaleConvention"] == "direct"
    # PEFT's own adapter_config.json carries the numerator it trained with.
    peft_config = json.loads(
        (job / "run" / "adapter" / "adapter_config.json").read_text(
            encoding="utf-8"))
    assert peft_config["lora_alpha"] == 4.0 and peft_config["r"] == 2


def test_re_executing_a_complete_job_is_idempotent(tmp_path, tiny_model_path):
    from steerlab_server import cli

    job = _job_directory(tmp_path, tiny_model_path)
    assert cli.main(["finetune", "execute", str(job)]) == 0
    record = tmp_path / "records" / "job3.json"
    assert cli.main(["finetune", "execute", str(job), "--record",
                     str(record)]) == 0
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["result"]["alreadyComplete"] is True


def test_execute_reports_a_failure_on_the_child_record(tmp_path,
                                                       tiny_model_path):
    from steerlab_server import cli

    job = _job_directory(tmp_path, tiny_model_path, selection_metric=None)
    record = tmp_path / "records" / "job4.json"
    assert cli.main(["finetune", "execute", str(job), "--record",
                     str(record)]) == 1
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "selection metric" in payload["error"]


def test_execute_refuses_a_directory_with_no_job_config(tmp_path):
    from steerlab_server import cli
    assert cli.main(["finetune", "execute", str(tmp_path)]) == 64


def test_finetune_usage_is_exit_64():
    from steerlab_server import cli
    assert cli.main(["finetune"]) == 64
    assert cli.main(["finetune", "execute"]) == 64
    assert cli.main(["finetune", "bogus", "x"]) == 64

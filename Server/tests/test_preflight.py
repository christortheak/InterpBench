"""WS4 preflight: memory fit, walltime, quota headroom, maintenance window —
each ok/warn/fail; refusal + force override; dry_run reporting; planned-record
math; per-request site resources (requeue/gpuTypes/gpuVram)."""

import json
import os
import stat

import pytest

from steerlab_server.api import housekeeping as hk
from steerlab_server.api import submissions as sub
from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.api.profile import ServerProfile
from steerlab_server.experiment import bundles, experiment_store as es
from steerlab_server.experiment.manifest import Manifest


MODEL = "acme/tiny"
REVISION = "abc123"
# Geometry chosen so KV math is hand-checkable: 2 layers × 2 kv-heads ×
# head_dim 8 × (512+2048) context × 2 bytes × 2 (K+V) = 327,680 bytes.
CONFIG = {"num_hidden_layers": 2, "num_key_value_heads": 2,
          "num_attention_heads": 4, "head_dim": 8, "torch_dtype": "bfloat16"}


def _check(report, check_id):
    return next(c for c in report["checks"] if c["id"] == check_id)


@pytest.fixture
def study_site(tmp_path, monkeypatch):
    """Workspace with a draft study (1 concept condition + baseline, 3 task
    prompts → 6 planned records), a fabricated HF snapshot for its model, and
    isolated metadata/scratch roots."""
    root = tmp_path / "source"
    scratch = tmp_path / "scratch"
    meta = tmp_path / "meta"
    hf = tmp_path / "hf"
    for p in (scratch, meta):
        p.mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(scratch))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(meta))
    monkeypatch.setenv("HF_HOME", str(hf))
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_GPU_VRAM", raising=False)
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")  # tests opt into slurm

    concept = root / "prompts" / "concepts" / "fair"
    concept.mkdir(parents=True)
    (concept / "positive.jsonl").write_text('{"text":"fair"}\n', encoding="utf-8")
    (concept / "negative.jsonl").write_text('{"text":"unfair"}\n', encoding="utf-8")
    es.create("Pf Study", model_id=MODEL, revision=REVISION, root=str(root))
    es.attach("pf-study", ["fair"], root=str(root))

    tasks = root / "prompts" / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "t.jsonl").write_text(
        "\n".join(json.dumps({"id": f"p{i}", "prompt": f"case {i}"})
                  for i in range(3)) + "\n", encoding="utf-8")
    manifest_path = root / "experiments" / "pf-study" / "experiment.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["taskPromptsFile"] = "prompts/tasks/t.jsonl"
    data["conditions"] = [{"name": "fair-c", "slots": [
        {"concept": "fair", "layer": 1, "alpha": 4.0}]}]
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    snap = hf / "hub" / f"models--acme--tiny" / "snapshots" / REVISION
    snap.mkdir(parents=True)
    refs = hf / "hub" / f"models--acme--tiny" / "refs"
    refs.mkdir()
    (refs / "main").write_text(REVISION, encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"\0" * 4096)
    (snap / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    return {"root": root, "meta": meta, "hf": hf, "scratch": scratch}


def _seed_throughput(meta, model=MODEL, gpu="L4", rate=10.0):
    (meta / "throughput.json").write_text(json.dumps({
        "schemaVersion": 1,
        "entries": [{"modelId": model, "gpuType": gpu, "recordsPerHour": rate,
                     "samples": 3, "updatedAt": "2026-07-12T00:00:00+00:00"}],
        "foldedJobIds": [],
    }), encoding="utf-8")


def _jobs(tmp_path, name="jobs.sqlite"):
    return JobManager(DurableJobStore(str(tmp_path / name)), sweep_orphans=False)


def _submit(tmp_path, *, dry_run=True, force=False, resources=None, verb="run"):
    return sub.submit_study(
        "pf-study", verb=verb, jobs=_jobs(tmp_path), executor="slurm",
        dry_run=dry_run, force=force,
        resources=resources or {"gres": "gpu:L4:1", "walltime": "10:00:00",
                                "gpuVram": {"L4": 24, "A100": 80}})


# --- the happy path ---------------------------------------------------------------

def test_dry_run_includes_preflight_and_all_checks_pass(study_site, tmp_path):
    _seed_throughput(study_site["meta"])
    submission = _submit(tmp_path)
    body = submission.to_dict()
    report = body["preflight"]
    assert report is not None
    assert [c["id"] for c in report["checks"]] == [
        "gpuRequest", "memoryFit", "walltime", "quotaHeadroom",
        "maintenanceWindow"]
    # No GPU requested in this fixture → the silent-CPU-fallback warning
    # (2026-07-19 shakedown) fires; requesting one clears it (covered by
    # the ok-path assertion below being warn-tolerant).
    assert _check(report, "gpuRequest")["status"] in {"ok", "warn"}
    memory = _check(report, "memoryFit")
    assert memory["status"] == "ok"
    assert memory["data"]["weightsBytes"] == 4096
    assert memory["data"]["kvCacheBytes"] == 327_680
    assert memory["data"]["estimateBytes"] == int((4096 + 327_680) * 1.2)
    walltime = _check(report, "walltime")
    assert walltime["status"] == "ok"
    assert walltime["data"]["plannedRecords"] == 6   # (baseline+1 cond) × 3 × 1
    # 6 ÷ 10/h × 1.5 = 0.9 h of generating, + the fixed 15 min startup every
    # job pays before its first record (external review round 12, finding 7).
    assert walltime["data"]["estimatedHours"] == 1.15
    assert walltime["data"]["startupHours"] == 0.25
    assert _check(report, "maintenanceWindow")["status"] == "ok"
    assert _check(report, "quotaHeadroom")["status"] in {"ok", "warn"}
    assert report["verdict"] in {"ok", "warn"}
    # The prepared job record carries the report durably (requested resources
    # survive the reconciler's result replacement).
    job = _jobs(tmp_path).get(submission.job_id)
    assert job.requested_resources["preflight"]["checks"]
    assert job.requested_resources["modelID"] == MODEL


def test_gpu_request_check_mirrors_what_sbatch_actually_emits():
    """Second shakedown finding: the sbatch renderer emits a GPU directive
    only from gres, so the default gpus=1 must NEVER count as a GPU
    request — preflight said ok while the job ran CPU-only."""
    from steerlab_server.api.executors import SlurmResources
    from steerlab_server.api.submissions import _check_gpu_request

    # Default gpus (1) with no gres: exactly the falsely-ok case.
    assert _check_gpu_request(SlurmResources())["status"] == "warn"
    assert _check_gpu_request(
        SlurmResources(gpus=4, gres=None))["status"] == "warn"
    assert _check_gpu_request(
        SlurmResources(gres="  "))["status"] == "warn"
    # A gres request is what the renderer actually turns into --gres.
    assert _check_gpu_request(
        SlurmResources(gres="gpu:L4:1"))["status"] == "ok"
    assert _check_gpu_request(SlurmResources(gres="L4"))["status"] == "ok"


# --- memoryFit --------------------------------------------------------------------

def test_memory_fit_fail_names_the_fix(study_site, tmp_path):
    submission = _submit(tmp_path, resources={
        "gres": "gpu:L4:1", "walltime": "10:00:00",
        "gpuVram": {"L4": 0, "A100": 80}})
    memory = _check(submission.preflight, "memoryFit")
    assert memory["status"] == "fail"
    assert "use gpu:A100:1" in memory["message"]
    assert submission.preflight["verdict"] == "fail"
    assert submission.dry_run is True  # dry run reports, never blocks


def test_memory_fit_budgets_one_gpu_not_the_pool(study_site, tmp_path):
    """Same defect as the LoRA preflight (fixed 2026-08-13): no caller passes
    device_map, so a model loads entirely onto one card. gpu:L4:2 with 1 GB
    cards pools to 2 GB — enough for a ~1.8 GB estimate — but one card is
    not, and the old pooled budget passed a job that then OOM'd on cuda:0."""
    snap = (study_site["hf"] / "hub" / "models--acme--tiny" / "snapshots"
            / REVISION)
    with open(snap / "model.safetensors", "wb") as f:
        f.truncate(int(1.5 * 1024**3))  # sparse; st_size is what's counted
    submission = _submit(tmp_path, resources={
        "gres": "gpu:L4:2", "walltime": "10:00:00",
        "gpuVram": {"L4": 1, "A100": 80}})
    memory = _check(submission.preflight, "memoryFit")
    assert memory["status"] == "fail"
    assert "one L4 has 1 GB" in memory["message"]
    assert "use gpu:A100:1" in memory["message"]
    assert memory["data"]["gpuCount"] == 2
    assert memory["data"]["vramPerGpuBytes"] == 1 * 1024**3
    # Additive stamp: the legacy key stays, now per-GPU like the new one.
    assert memory["data"]["vramBytes"] == memory["data"]["vramPerGpuBytes"]


def test_memory_fit_multi_gpu_ok_without_idle_warning(study_site, tmp_path):
    """Unlike the LoRA trainer, extra inference GPUs are not dead weight —
    the registry can place a judge/second model there — so a fitting
    multi-GPU request passes with no idle-allocation warning."""
    memory = _check(_submit(tmp_path, resources={
        "gres": "gpu:L4:2", "walltime": "10:00:00",
        "gpuVram": {"L4": 24, "A100": 80}}).preflight, "memoryFit")
    assert memory["status"] == "ok"
    assert "of 24 GB on one L4" in memory["message"]
    assert "idle" not in memory["message"]
    assert memory["data"]["gpuCount"] == 2
    assert memory["data"]["vramPerGpuBytes"] == 24 * 1024**3


def test_memory_fit_warns_when_model_not_cached(study_site, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    memory = _check(_submit(tmp_path).preflight, "memoryFit")
    assert memory["status"] == "warn"
    assert "no weights in the HF cache" in memory["message"]


def test_memory_fit_warns_without_vram_table(study_site, tmp_path):
    memory = _check(_submit(tmp_path, resources={
        "gres": "gpu:L4:1", "walltime": "10:00:00"}).preflight, "memoryFit")
    assert memory["status"] == "warn"
    assert "no VRAM entry for L4" in memory["message"]


def test_memory_fit_warns_without_config_geometry(study_site, tmp_path):
    snap = (study_site["hf"] / "hub" / "models--acme--tiny" / "snapshots"
            / REVISION)
    (snap / "config.json").unlink()
    memory = _check(_submit(tmp_path).preflight, "memoryFit")
    assert memory["status"] == "warn"
    assert "config.json" in memory["message"]


def test_analyze_preflight_neutralizes_model_shaped_checks(study_site, tmp_path):
    """analyze is statistics-only (no model load): resources that would FAIL
    a run submission (VRAM 0, tight walltime, no gres) must not gate it —
    an analyze job needs no GPU allocation at all. quotaHeadroom and
    maintenanceWindow still apply (it writes output under a walltime)."""
    submission = _submit(tmp_path, verb="analyze", resources={
        "walltime": "00:30:00", "gpuVram": {"L4": 0}})
    report = submission.preflight
    assert [c["id"] for c in report["checks"]] == [
        "gpuRequest", "memoryFit", "walltime", "quotaHeadroom",
        "maintenanceWindow"]
    for check_id in ("gpuRequest", "memoryFit", "walltime"):
        check = _check(report, check_id)
        assert check["status"] == "ok", check
        assert "not applicable" in check["message"]
    assert report["verdict"] in {"ok", "warn"}


# --- walltime ----------------------------------------------------------------------

def test_walltime_warns_without_history(study_site, tmp_path):
    walltime = _check(_submit(tmp_path).preflight, "walltime")
    assert walltime["status"] == "warn"
    assert "no throughput history for acme/tiny on L4" in walltime["message"]


def test_walltime_fail_when_estimate_exceeds_request(study_site, tmp_path):
    _seed_throughput(study_site["meta"])
    walltime = _check(_submit(tmp_path, resources={
        "gres": "gpu:L4:1", "walltime": "00:30:00",
        "gpuVram": {"L4": 24}}).preflight, "walltime")
    assert walltime["status"] == "fail"     # 0.9 h estimate > 0.5 h requested
    assert "raise the walltime or split the matrix" in walltime["message"]


def test_walltime_warn_above_80_percent(study_site, tmp_path):
    _seed_throughput(study_site["meta"])
    walltime = _check(_submit(tmp_path, resources={
        "gres": "gpu:L4:1", "walltime": "01:20:00",
        "gpuVram": {"L4": 24}}).preflight, "walltime")
    assert walltime["status"] == "warn"     # 1.15 h of 1.33 h = 86%


# --- the per-shard estimate (external review round 12, finding 7) ------------

def _walltime_of(study_site, *, planned_records=6, shard_count=1,
                 walltime="10:00:00"):
    """The production check, called directly — the fan-out K and the record
    count are what this finding is about, and both are arguments to it."""
    manifest = Manifest.load("pf-study", str(study_site["root"]))
    resources = sub._resources_from_dict(
        {"gres": "gpu:L4:1", "walltime": walltime, "gpuVram": {"L4": 24}},
        "pf-study", "run")
    return sub._check_walltime(manifest, resources, planned_records,
                               ServerProfile.from_env(), "run",
                               shard_count=shard_count)


def test_a_shard_is_priced_by_the_largest_slice_plus_its_own_startup(
        study_site):
    """Exact division under-priced twice over: the LARGEST shard runs
    ceil(records ÷ K), and every child pays the model load in full.
    """
    _seed_throughput(study_site["meta"])
    # 6 records over 4 shards: the largest runs ceil(6 ÷ 4) = 2, not 1.5.
    walltime = _walltime_of(study_site, shard_count=4)

    assert walltime["data"]["shardCount"] == 4
    assert walltime["data"]["plannedRecords"] == 6
    assert walltime["data"]["recordsPerShard"] == 2      # ceil, not floor
    assert walltime["data"]["estimateIsPerShard"] is True
    # 2 ÷ 10/h × 1.5 = 0.3 h, + 0.25 h startup. The old answer (an exact
    # 1.5 records and no startup) was 0.225 h.
    assert walltime["data"]["estimatedHours"] == 0.55
    assert "ceil(6 ÷ 4 shard jobs) = 2 records" in walltime["message"]
    assert "15 min fixed job startup" in walltime["message"]
    assert "PER-SHARD estimate" in walltime["message"]


def test_a_degenerate_shard_still_prices_the_model_load(study_site):
    """The case exact division made free: one record spread over four shards.
    A shard that runs a single record is a model load with one record after
    it, so the estimate can never fall below the startup term.
    """
    _seed_throughput(study_site["meta"])
    walltime = _walltime_of(study_site, planned_records=1, shard_count=4)
    assert walltime["data"]["recordsPerShard"] == 1
    assert walltime["data"]["estimatedHours"] >= sub.PREFLIGHT_JOB_STARTUP_HOURS


def test_an_unsharded_job_pays_the_startup_too_and_says_no_shard_words(
        study_site):
    """One job is still a job: it loads the model before its first record.
    The per-shard label stays off when there is nothing to shard."""
    _seed_throughput(study_site["meta"])
    walltime = _walltime_of(study_site)
    assert walltime["data"]["estimatedHours"] == 1.15   # 0.9 + 0.25
    assert "shardCount" not in walltime["data"]
    assert "PER-SHARD" not in walltime["message"]
    assert "6 records ÷ 10/h" in walltime["message"]


# --- quotaHeadroom (fabricated df so CI disks don't decide the outcome) --------------

def test_quota_headroom_bands(study_site, monkeypatch):
    profile = ServerProfile.from_env()

    def fake_roots(free_ws, free_hf):
        return {
            "workspace": {"path": "/ws", "totalBytes": 10**12,
                          "freeBytes": free_ws, "usedBytes": 0, "warning": None},
            "hfCache": {"path": "/hf", "totalBytes": 10**12,
                        "freeBytes": free_hf, "usedBytes": 0, "warning": None},
        }

    monkeypatch.setattr(hk, "disk_roots",
                        lambda p=None: fake_roots(100 * 1024**3, 100 * 1024**3))
    assert sub._check_quota_headroom(6, profile)["status"] == "ok"

    monkeypatch.setattr(hk, "disk_roots",
                        lambda p=None: fake_roots(5 * 1024**3, 100 * 1024**3))
    assert sub._check_quota_headroom(6, profile)["status"] == "warn"

    monkeypatch.setattr(hk, "disk_roots",
                        lambda p=None: fake_roots(100 * 1024**3, 512 * 1024**2))
    check = sub._check_quota_headroom(6, profile)
    assert check["status"] == "fail"
    assert "hfCache root" in check["message"]

    # Predicted output larger than free space fails even above the floor.
    monkeypatch.setattr(hk, "disk_roots",
                        lambda p=None: fake_roots(2 * 1024**3, 100 * 1024**3))
    big = sub._check_quota_headroom(50_000_000, profile)   # ≈3 TiB predicted
    assert big["status"] == "fail"
    assert "predicted output" in big["message"]

    monkeypatch.setattr(hk, "disk_roots", lambda p=None: {})
    assert sub._check_quota_headroom(6, profile)["status"] == "warn"


# --- maintenanceWindow ---------------------------------------------------------------

def test_maintenance_check_fails_with_window_named(study_site, tmp_path):
    from datetime import datetime, timedelta, timezone
    start = datetime.now(timezone.utc) + timedelta(hours=2)
    hk.write_maintenance([{"start": start.isoformat(),
                           "end": (start + timedelta(hours=8)).isoformat(),
                           "label": "quarterly"}])
    check = _check(_submit(tmp_path).preflight, "maintenanceWindow")
    assert check["status"] == "fail"
    assert "quarterly" in check["message"]
    assert check["data"]["window"]["label"] == "quarterly"


# --- refusal + force -------------------------------------------------------------------

def test_verdict_fail_blocks_real_submission_without_force(study_site, tmp_path,
                                                           monkeypatch):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    jobs = _jobs(tmp_path)
    with pytest.raises(ValueError, match="preflight failed"):
        sub.submit_study("pf-study", verb="run", jobs=jobs, executor="slurm",
                         dry_run=False,
                         resources={"gres": "gpu:L4:1", "walltime": "10:00:00",
                                    "gpuVram": {"L4": 0, "A100": 80}})
    assert jobs.list() == []   # refused before any job was recorded


def test_refusal_carries_structured_report(study_site, tmp_path, monkeypatch):
    """PreflightRejection is a ValueError (CLI handlers unchanged) that carries
    the full report, so the HTTP routes can answer with a structured
    ``detail: {message, preflight}`` instead of a parse-me string."""
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    jobs = _jobs(tmp_path)
    with pytest.raises(sub.PreflightRejection) as excinfo:
        sub.submit_study("pf-study", verb="run", jobs=jobs, executor="slurm",
                         dry_run=False,
                         resources={"gres": "gpu:L4:1", "walltime": "10:00:00",
                                    "gpuVram": {"L4": 0, "A100": 80}})
    report = excinfo.value.preflight
    assert report["verdict"] == "fail"
    assert any(c["status"] == "fail" for c in report["checks"])
    assert isinstance(excinfo.value, ValueError)


def test_force_overrides_and_is_recorded_loudly(study_site, tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_sbatch = bindir / "sbatch"
    fake_sbatch.write_text("#!/bin/sh\necho 'Submitted batch job 777'\n",
                           encoding="utf-8")
    fake_sbatch.chmod(fake_sbatch.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])

    jobs = _jobs(tmp_path)
    submission = sub.submit_study(
        "pf-study", verb="run", jobs=jobs, executor="slurm", dry_run=False,
        force=True,
        resources={"gres": "gpu:L4:1", "walltime": "10:00:00",
                   "gpuVram": {"L4": 0, "A100": 80}})
    assert submission.slurm_job_id == "777"
    job = jobs.get(submission.job_id)
    assert job.requested_resources["preflightOverridden"] is True
    assert job.requested_resources["preflight"]["verdict"] == "fail"
    assert job.result["preflightOverridden"] is True
    assert any("PREFLIGHT OVERRIDDEN" in line for line in job.all_logs())


# --- planned-record math ------------------------------------------------------------

def _manifest(**overrides) -> Manifest:
    base = {"name": "m", "modelID": MODEL,
            "conditions": [{"name": "c1", "slots": []}]}
    base.update(overrides)
    return Manifest.from_dict(base)


PROMPTS_3 = "\n".join(json.dumps({"id": f"p{i}", "prompt": "x"})
                      for i in range(3))


def test_planned_records_math():
    assert sub._planned_records(_manifest(), PROMPTS_3) == 6  # (+baseline) × 3
    assert sub._planned_records(
        _manifest(samplesPerItem=4, temperature=0.8), PROMPTS_3) == 24
    assert sub._planned_records(
        _manifest(outcomeInstruments=["answerTokenLogprob"]), PROMPTS_3) == 6
    assert sub._planned_records(
        _manifest(outcomeInstruments=["answerTokenLogprob", "sampledText"]),
        PROMPTS_3) == 12
    variants = [{"name": "v1", "artifactPath": "a", "artifactHash": "h"},
                {"name": "v2", "artifactPath": "b", "artifactHash": "h"}]
    assert sub._planned_records(
        _manifest(variantConditions=variants), PROMPTS_3) == 9  # (1+2) × 3
    assert sub._planned_records(_manifest(studyKind="multiAgent"),
                                PROMPTS_3) is None
    assert sub._planned_records(_manifest(), None) is None
    assert sub._planned_records(None, PROMPTS_3) is None


def test_a_sampled_evaluate_is_priced_at_the_sampled_count():
    """The walltime estimate divides planned records by a measured rate, so a
    sampled evaluate priced at the full matrix asks for walltime it will not
    use — and gives the researcher a number their own run's report
    contradicts. Two conditions (baseline + one arm) × n."""
    manifest = _manifest()
    assert sub._planned_records(manifest, PROMPTS_3) == 6
    assert sub._planned_records(
        manifest, PROMPTS_3, verb="evaluate", sample_per_condition=1) == 2
    # Other verbs are untouched: the flags refuse there anyway, and a `run`
    # generates the whole matrix whatever an evaluate would later code.
    assert sub._planned_records(
        manifest, PROMPTS_3, verb="run", sample_per_condition=1) == 6
    # Capped by the full matrix. A nonsensical over-ask refuses at execute
    # time; it must never INFLATE the estimate on the way there.
    assert sub._planned_records(
        manifest, PROMPTS_3, verb="evaluate", sample_per_condition=999) == 6
    # No sample = byte-identical to the historical answer.
    assert sub._planned_records(
        manifest, PROMPTS_3, verb="evaluate", sample_per_condition=None) == 6


def test_the_sampled_count_counts_baseline_as_a_coded_condition():
    """The per-response coding instrument codes baseline like any other
    condition — every record is coded individually and blinded — so the
    sampled count is n per DECLARED condition plus the implicit baseline."""
    variants = [{"name": "v1", "artifactPath": "a", "artifactHash": "h"},
                {"name": "v2", "artifactPath": "b", "artifactHash": "h"}]
    assert sub._sampled_evaluate_records(
        _manifest(variantConditions=variants), 100) == 300
    assert sub._sampled_evaluate_records(_manifest(), 100) == 200


# --- per-request site resources (WS1 generalization carried into the API) -------------

def test_resources_from_dict_maps_site_fields(study_site, tmp_path):
    submission = _submit(tmp_path, resources={
        "gres": "gpu:Z100:1", "walltime": "10:00:00", "requeue": True,
        "gpuTypes": ["Z100"], "gpuVram": {"Z100": 48}, "account": "lab1"})
    resources = submission.slurm_bundle["resources"]
    assert resources["requeue"] is True
    assert resources["gpu_types"] == ["Z100"]
    assert resources["gpu_vram_gb"] == {"Z100": 48}
    assert resources["account"] == "lab1"
    script = open(submission.slurm_bundle["script_path"], encoding="utf-8").read()
    assert "#SBATCH --requeue" in script
    assert "#SBATCH --account=lab1" in script
    assert "#SBATCH --gres=gpu:Z100:1" in script
    memory = _check(submission.preflight, "memoryFit")
    assert memory["status"] == "ok"        # the per-request VRAM table applied


def test_resources_from_dict_rejects_malformed_site_data(study_site, tmp_path):
    with pytest.raises(ValueError, match="gpuVram"):
        _submit(tmp_path, resources={"gres": "gpu:L4:1",
                                     "gpuVram": {"L4": "eighty"}})
    with pytest.raises(ValueError, match="gpuTypes"):
        _submit(tmp_path, resources={"gres": "gpu:L4:1", "gpuTypes": []})


# --- bundle submissions read the manifest from the tar ---------------------------------

def test_submit_bundle_preflight_uses_bundle_manifest(study_site, tmp_path,
                                                      monkeypatch):
    meta = bundles.package_experiment("pf-study", root=str(study_site["root"]))
    # Point the server at an EMPTY workspace: the bundle is the only source of
    # the manifest + prompts, exactly the remote-client situation.
    empty = tmp_path / "server-root"
    empty.mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(empty))
    _seed_throughput(study_site["meta"])
    submission = sub.submit_run_bundle(
        meta["bundlePath"], verb="run", jobs=_jobs(tmp_path, "b.sqlite"),
        executor="slurm", dry_run=True,
        resources={"gres": "gpu:L4:1", "walltime": "10:00:00",
                   "gpuVram": {"L4": 24}})
    report = submission.preflight
    assert _check(report, "memoryFit")["status"] == "ok"
    assert _check(report, "walltime")["data"]["plannedRecords"] == 6
    job = _jobs(tmp_path, "b.sqlite").get(submission.job_id)
    assert job.requested_resources["modelID"] == MODEL


# --- local executor: no preflight, but the throughput stamp still lands ----------------

def test_local_submission_has_no_preflight_but_stamps_model(study_site, tmp_path):
    submission = sub.submit_study("pf-study", verb="run", jobs=_jobs(tmp_path),
                                  executor="local", dry_run=True)
    assert submission.to_dict()["preflight"] is None
    job = _jobs(tmp_path).get(submission.job_id)
    assert job.requested_resources["modelID"] == MODEL

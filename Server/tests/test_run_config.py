"""Canonical per-run config.json: one shared helper, uniform shape, emitted by
the run-directory writers (schema pinned cross-engine)."""

import json
import os

from steerlab_server import __version__
from steerlab_server.build_identity import engine_version
from steerlab_server.experiment import model_variant, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.experiment.run_config import (RUN_CONFIG_KEYS,
                                                   write_run_config)
from steerlab_server.python_environment import python_environment

# THE cross-engine contract (schema 4). This literal is duplicated on purpose:
# adding/removing/renaming a top-level key must fail THIS test until the
# schema version is bumped and this list (plus the Swift twin in
# EvidenceTierTests.runMetadataPayloadHasExactPinnedKeys) is updated in the
# same change. "notes" is the only escape hatch for engine-specific extras.
CONTRACT_KEYS = [
    "appVersion",
    "createdAt",
    "dtype",
    "experiment",
    "experimentHash",
    "jobId",
    "modelID",
    "notes",
    "platform",
    "pythonEnvironment",
    "revision",
    "runId",
    "runType",
    "samplesPerItem",
    "schemaVersion",
    "seedPolicy",
    "substrate",
    "temperature",
]


def test_write_run_config_shape(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    run_dir = tmp_path / "20260712T000000000-exp-s-extract"
    run_dir.mkdir()
    path = write_run_config(str(run_dir), "extract",
                            model_id="org/m", revision="abc",
                            experiment="s", experiment_hash="ff" * 32)
    config = json.load(open(path))
    assert config == {
        "schemaVersion": 4,
        "runId": "20260712T000000000-exp-s-extract",
        "runType": "extract",
        "createdAt": config["createdAt"],
        "substrate": "python-hf-transformers",
        # The engine's build identity — "steerlab-server <version>+<sha8>"
        # in a checkout, bare version otherwise (see test_build_identity.py).
        "appVersion": engine_version(),
        "platform": config["platform"],
        "modelID": "org/m",
        "revision": "abc",
        "experiment": "s",
        "experimentHash": "ff" * 32,
        "temperature": None,
        "samplesPerItem": None,
        "seedPolicy": None,
        "dtype": None,
        # Schema 4 (WP6 R1): the measurement stack that actually ran. Compared
        # by identity against the module that produces it — asserting literal
        # versions here would make the suite fail on every dependency bump,
        # which is exactly the churn the STAMP exists to absorb.
        "pythonEnvironment": python_environment(),
        "jobId": None,
        "notes": {},
    }
    assert config["createdAt"].endswith("Z")
    assert config["appVersion"].startswith("steerlab-server ")
    # platform is OS + architecture only — never a hostname.
    assert config["platform"].split("-")[0] in ("macOS", "linux", "windows") \
        or config["platform"]


def test_run_config_closed_key_set(tmp_path):
    """Schema evolution firewall: the top-level key set is CLOSED. Anyone
    adding a key without bumping RUN_CONFIG_SCHEMA_VERSION (and this literal,
    and the Swift twin) fails here."""
    config = json.load(open(write_run_config(str(tmp_path), "run")))
    assert sorted(config.keys()) == CONTRACT_KEYS
    assert sorted(RUN_CONFIG_KEYS) == CONTRACT_KEYS
    assert config["schemaVersion"] == 4


def test_write_run_config_nulls_unknown_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    config = json.load(open(write_run_config(str(tmp_path), "reader-fit")))
    assert config["modelID"] is None and config["revision"] is None
    assert config["experiment"] is None and config["experimentHash"] is None
    assert config["temperature"] is None and config["samplesPerItem"] is None
    assert config["seedPolicy"] is None and config["jobId"] is None
    # A run that loaded no model records null precision, not a guess.
    assert config["dtype"] is None
    assert config["notes"] == {}
    assert config["runType"] == "reader-fit"


def test_write_run_config_stamps_slurm_job_id(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    config = json.load(open(write_run_config(str(tmp_path), "run")))
    assert config["jobId"] == "424242"


def test_write_run_config_prefers_in_process_job_id(tmp_path, monkeypatch):
    """The JobManager worker's contextvar outranks SLURM_JOB_ID (an in-process
    server job may itself run under some unrelated Slurm allocation)."""
    from steerlab_server.experiment.run_config import current_job_id
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    token = current_job_id.set("abc123def456")
    try:
        config = json.load(open(write_run_config(str(tmp_path), "run")))
    finally:
        current_job_id.reset(token)
    assert config["jobId"] == "abc123def456"


def test_job_manager_worker_sets_job_id_context(tmp_path):
    """End-to-end: a run directory stamped inside a submitted job carries that
    job's id."""
    import time as _time
    from steerlab_server.api.jobs import DurableJobStore, JobManager

    manager = JobManager(store=DurableJobStore(str(tmp_path / "jobs.sqlite")),
                         capability_provider=lambda: {})
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def work(job):
        write_run_config(str(run_dir), "run")
        return {}

    job = manager.submit("stamp-test", work)
    for _ in range(200):
        if manager.get(job.id).status in ("succeeded", "failed"):
            break
        _time.sleep(0.01)
    assert manager.get(job.id).status == "succeeded"
    config = json.load(open(run_dir / "config.json"))
    assert config["jobId"] == job.id


def test_explicit_job_id_outranks_context_and_env(tmp_path, monkeypatch):
    """The off-worker writers (shard merge, pipeline seed — created on the
    controller's monitor thread) pass the owning job's record id explicitly:
    without it every run the same controller incarnation assembled stamped
    the CONTROLLER's own Slurm allocation id (four runs, one stale jobId,
    observed 2026-08-06)."""
    monkeypatch.setenv("SLURM_JOB_ID", "47285267")
    config = json.load(open(write_run_config(
        str(tmp_path), "run", job_id="parent12ab34")))
    assert config["jobId"] == "parent12ab34"


def test_local_executor_scrubs_inherited_slurm_job_id(monkeypatch):
    """A LOCAL child is not a Slurm job: when the server itself runs inside
    an allocation (the controller job), the inherited SLURM_JOB_ID must not
    reach the child, or its run directory stamps the controller's id."""
    import sys
    from steerlab_server.api.executors import LocalExecutor
    monkeypatch.setenv("SLURM_JOB_ID", "47285267")
    proc = LocalExecutor().run(
        [sys.executable, "-c",
         "import os; print(os.environ.get('SLURM_JOB_ID'))"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "None"


def test_write_run_config_never_rewrites(tmp_path):
    """Run directories are immutable and a resumed run keeps its creation
    stamp: a second write is a no-op, byte-for-byte."""
    path = write_run_config(str(tmp_path), "run", model_id="org/m")
    before = open(path, "rb").read()
    again = write_run_config(str(tmp_path), "extract", model_id="org/OTHER")
    assert again == path
    assert open(path, "rb").read() == before


def test_config_snapshot_writers_emit_config_json(tmp_path):
    manifest = Manifest.from_dict(
        {"name": "s", "modelID": "org/m", "modelRevision": "abc"})
    tasks._write_config_snapshot(manifest, str(tmp_path), "sweep")
    config = json.load(open(tmp_path / "config.json"))
    assert config["runType"] == "sweep"
    assert config["experiment"] == "s"
    assert config["experimentHash"] == manifest.content_hash()
    assert config["modelID"] == "org/m" and config["revision"] == "abc"
    # Additive: the richer per-task artifacts are still written.
    assert (tmp_path / "experiment.json").exists()
    assert (tmp_path / "experiment-hash.txt").exists()


def test_config_snapshot_stamps_sampling_policy_for_generating_tasks(tmp_path):
    manifest = Manifest.from_dict(
        {"name": "s", "modelID": "org/m", "temperature": 0.7,
         "samplesPerItem": 3, "seedPolicy": "derivedSHA256"})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tasks._write_config_snapshot(manifest, str(run_dir), "run")
    config = json.load(open(run_dir / "config.json"))
    assert config["temperature"] == 0.7
    assert config["samplesPerItem"] == 3
    assert config["seedPolicy"] == "derivedSHA256"

    # Non-generating tasks stamp null — validate/extract don't sample by the
    # manifest's policy, so the stamp must not invent one.
    validate_dir = tmp_path / "validate"
    validate_dir.mkdir()
    tasks._write_config_snapshot(manifest, str(validate_dir), "validate")
    config = json.load(open(validate_dir / "config.json"))
    assert config["temperature"] is None
    assert config["samplesPerItem"] is None
    assert config["seedPolicy"] is None


def test_save_variant_emits_config_json(tmp_path):
    variant = model_variant.ModelVariant(name="v", base_model_id="org/m",
                                         base_revision="rev1")
    result = model_variant.save_variant(variant, root=str(tmp_path))
    config = json.load(open(os.path.join(result["runDirectory"], "config.json")))
    assert config["runType"] == "variant-save"
    assert config["modelID"] == "org/m" and config["revision"] == "rev1"
    assert config["substrate"] == "python-hf-transformers"
    assert config["runId"] == os.path.basename(result["runDirectory"])

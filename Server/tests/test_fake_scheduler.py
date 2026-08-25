"""Slurm executor + reconciler against fake scheduler binaries (WS2.4).

``tests/fakebin`` ships executable ``sbatch``/``squeue``/``sacct``/``scancel``
doubles driven by env vars and a JSON state file, so submit parsing, state
mapping (including requeue-class and checkpoint-exit states), the squeue
fallback, cancellation, the maintenance-window refusal, and the JobManager
reconciler all run in CI with no scheduler installed.
"""

import json
import os
import shutil
import stat
from datetime import datetime, timedelta, timezone

import pytest

from steerlab_server.api.executors import (
    SlurmExecutor, SlurmResources, crosses_maintenance_window)
from steerlab_server.api.jobs import DurableJobStore, JobManager

FAKEBIN_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakebin")


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """Copy the committed fakebin doubles to tmp (re-asserting the exec bit —
    sync services can drop it), prepend them to PATH, and wire the env
    protocol. Yields a small handle for driving/observing them."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = bindir / name
        shutil.copy(os.path.join(FAKEBIN_SOURCE, name), target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log_dir = tmp_path / "calls"
    log_dir.mkdir()
    state_file = tmp_path / "slurm-state.json"
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE", str(state_file))
    monkeypatch.delenv("FAKE_SBATCH_FAIL", raising=False)
    monkeypatch.delenv("FAKE_SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    # Auto-resubmit is opt-in; a developer's env must not flip these tests.
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT_LIMIT", raising=False)
    # Poll commands are site data; these tests exercise the raw defaults.
    monkeypatch.delenv("STEERLAB_SLURM_SACCT", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_SQUEUE", raising=False)

    class Handle:
        def set_state(self, job_id, state, exit_code="0:0", queue=None):
            table = {}
            if state_file.exists():
                table = json.loads(state_file.read_text(encoding="utf-8"))
            table[str(job_id)] = {"state": state, "exit": exit_code, "queue": queue}
            state_file.write_text(json.dumps(table), encoding="utf-8")

        def calls(self, binary):
            path = log_dir / f"{binary}.calls"
            if not path.exists():
                return []
            return path.read_text(encoding="utf-8").splitlines()

    return Handle()


def _bundle(tmp_path, resources=None):
    return SlurmExecutor().create_bundle(
        str(tmp_path / "bundle"), ["python", "-c", "pass"],
        resources=resources or SlurmResources(gres="A100", walltime="01:00:00"))


# --- submit --------------------------------------------------------------------

def test_submit_parses_job_id_from_sbatch_output(tmp_path, fake_slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "424242")
    bundle = _bundle(tmp_path)
    assert SlurmExecutor().submit(bundle) == "424242"
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 1
    assert bundle.script_path in calls[0]
    # sbatch runs FROM the bundle dir (run_root side), never the daemon's own
    # cwd — the job's default working directory is the submit-time cwd, and
    # Some sites require submitting from /scratch.
    cwds = [os.path.realpath(p) for p in fake_slurm.calls("sbatch-cwd")]
    assert cwds == [os.path.realpath(bundle.bundle_dir)]


def test_submit_surfaces_sbatch_error(tmp_path, fake_slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_FAIL", "sbatch: error: invalid partition")
    with pytest.raises(RuntimeError, match="invalid partition"):
        SlurmExecutor().submit(_bundle(tmp_path))


# --- bundle env: no serialized secrets (Anthropic key custody, 2026-07-18) ------

def test_bundle_refuses_secret_shaped_env_keys(tmp_path):
    # create_bundle serializes env into run.sbatch AND bundle.json on shared
    # /scratch — a just-in-time secret passed this way becomes a durable
    # filesystem artifact. Values are refused; file-path indirection passes.
    for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                "STEERLAB_AUTH_TOKEN", "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN", "MY_SERVICE_SECRET",
                "SOMETHING_PASSWORD", "AWS_CREDENTIALS"):
        with pytest.raises(ValueError, match="secret-shaped") as excinfo:
            SlurmExecutor().create_bundle(
                str(tmp_path / "bundle"), ["python", "-c", "pass"],
                env={key: "sk-not-really"},
                resources=SlurmResources(walltime="01:00:00"))
        assert key in str(excinfo.value)
        # The remedy names both sanctioned patterns.
        assert "TOKEN_FILE" in str(excinfo.value)
        assert "Mac" in str(excinfo.value)
    # File-path indirection and ordinary keys pass untouched.
    bundle = SlurmExecutor().create_bundle(
        str(tmp_path / "bundle-ok"), ["python", "-c", "pass"],
        env={"STEERLAB_AUTH_TOKEN_FILE": "~/.steerlab-token",
             "STEERLAB_JUDGE_KEY_FILE": "~/.steerlab/judge-key",
             "HF_HOME": "/scratch/hf"},
        resources=SlurmResources(walltime="01:00:00"))
    assert bundle.env["STEERLAB_AUTH_TOKEN_FILE"] == "~/.steerlab-token"
    assert bundle.env["STEERLAB_JUDGE_KEY_FILE"] == "~/.steerlab/judge-key"


def test_bundle_inherits_judge_key_path_from_controller_env(
        tmp_path, monkeypatch):
    # The judge-key PATH (never the key) rides into study bundles the same
    # way HF_HOME does (2026-07-19): a sweep/run job's inline external
    # judging must resolve the same mode-600 file the controller would.
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE", "/home/me/.steerlab/jk")
    bundle = SlurmExecutor().create_bundle(
        str(tmp_path / "bundle-jk"), ["python", "-c", "pass"],
        resources=SlurmResources(walltime="01:00:00"))
    assert bundle.env["STEERLAB_JUDGE_KEY_FILE"] == "/home/me/.steerlab/jk"


# --- bundle env: shared HF cache + offline policy (live shakedown 2026-07-16) ----

def test_bundle_env_inherits_hf_cache_policy(tmp_path, fake_slurm, monkeypatch):
    # Jobs run under --export=NONE; the controller sourced HF_HOME /
    # HF_HUB_OFFLINE from the bootstrap env file, so its OWN process env is
    # the truth every bundle (session workers AND study jobs) must carry —
    # otherwise the child re-downloads gigabytes into ~/.cache on a billed
    # allocation.
    monkeypatch.delenv("STEERLAB_NODE_CACHE_ROOT", raising=False)
    monkeypatch.setenv("HF_HOME", "/work/lab/hf-cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Node-local staging must reach EVERY Slurm child that loads a model —
    # a 27B study run/sweep/validate is the workload where it pays most,
    # not just interactive GPU sessions (engineer review 2026-07-17). The
    # literal $SLURM_JOB_ID survives shell-quoted; the loader expands it.
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/lscratch/$SLURM_JOB_ID")
    bundle = _bundle(tmp_path)
    assert bundle.env["HF_HOME"] == "/work/lab/hf-cache"
    assert bundle.env["HF_HUB_OFFLINE"] == "1"
    assert bundle.env["STEERLAB_NODE_STAGE_DIR"] == "/lscratch/$SLURM_JOB_ID"
    with open(bundle.script_path, encoding="utf-8") as handle:
        script = handle.read()
    assert "export HF_HOME=/work/lab/hf-cache" in script
    assert "export HF_HUB_OFFLINE=1" in script
    assert ("export STEERLAB_NODE_STAGE_DIR='/lscratch/$SLURM_JOB_ID'"
            in script)


def test_node_cache_root_still_wins_over_inherited_hf_home(
        tmp_path, fake_slurm, monkeypatch):
    # STEERLAB_NODE_CACHE_ROOT is deliberate node-local staging — it keeps
    # winning HF_HOME over the inherited shared cache; the offline policy
    # rides along regardless.
    monkeypatch.setenv("STEERLAB_NODE_CACHE_ROOT", "/lscratch/steerlab")
    monkeypatch.setenv("HF_HOME", "/work/lab/hf-cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    bundle = _bundle(tmp_path)
    assert bundle.env["HF_HOME"] == "/lscratch/steerlab"
    assert bundle.env["HF_HUB_OFFLINE"] == "1"


def test_bundle_env_without_parent_hf_policy_is_unchanged(
        tmp_path, fake_slurm, monkeypatch):
    monkeypatch.delenv("STEERLAB_NODE_CACHE_ROOT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    bundle = _bundle(tmp_path)
    assert "HF_HOME" not in bundle.env
    assert "HF_HUB_OFFLINE" not in bundle.env


# --- node-scratch cleanup epilogue (cluster-operator requirement, 2026-08-19) ------

def _cleanup_probe(stage_dir_value):
    """Run the rendered cleanup function in a real bash, with ``rm`` traced
    rather than executed, and return what it would have removed."""
    import subprocess

    from steerlab_server.api.executors import render_slurm_script

    class _Bundle:
        bundle_dir = "/tmp/steerlab-bundle"
        stdout_path = "/tmp/out"
        stderr_path = "/tmp/err"
        command = ["true"]
        env = {}
        resources = SlurmResources(use_srun=False)

    from steerlab_server.node_scratch import CANONICAL_FUNCTION

    script = render_slurm_script(_Bundle())
    # From the CANONICALIZATION HELPER, not from `cleanup_node_scratch`: the
    # trap's last gate resolves both the anchor and the target through the
    # filesystem before it deletes anything, and that helper is defined just
    # above the function that calls it.
    start = script.index(f"{CANONICAL_FUNCTION}() {{")
    end = script.index("trap cleanup_node_scratch EXIT")
    body = script[start:end]
    probe = (
        "set -euo pipefail\n"
        "rm() { echo \"RM $*\"; }\n"
        + body
        + "cleanup_node_scratch\n")
    result = subprocess.run(
        ["bash", "-c", probe], capture_output=True, text=True,
        env={"PATH": os.environ.get("PATH", ""), "SLURM_JOB_ID": "12345",
             **({"STEERLAB_NODE_STAGE_DIR": stage_dir_value}
                if stage_dir_value is not None else {})})
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_rendered_script_carries_the_node_scratch_cleanup_trap(tmp_path, fake_slurm):
    """The cluster does NOT wipe /lscratch at job end (cluster operators, 2026-08-19), so
    the job removes its own stage directory. Unconditional: any site that
    stages job-scoped should clean up, and a site with no stage dir gets a
    no-op function."""
    bundle = _bundle(tmp_path)
    with open(bundle.script_path, encoding="utf-8") as handle:
        script = handle.read()
    assert "cleanup_node_scratch() {" in script
    assert "trap cleanup_node_scratch EXIT" in script
    # It composes with the checkpoint trap rather than replacing it, and is
    # registered after the env exports it reads.
    assert "trap checkpoint USR1 TERM" in script
    assert script.index("export SLURM_EXPORT_ENV=ALL") < script.index(
        "trap cleanup_node_scratch EXIT") < script.index("trap checkpoint USR1 TERM")


def test_cleanup_only_fires_for_a_job_scoped_stage_dir():
    """The guard is the whole safety argument: only a template that embeds the
    literal ``$SLURM_JOB_ID`` — job-scoped by construction — is ever removed."""
    assert _cleanup_probe("/lscratch/$SLURM_JOB_ID") == "RM -rf -- /lscratch/12345"
    # A SHARED node cache is not this job's to delete, and must not match.
    assert _cleanup_probe("/lscratch/steerlab-shared-cache") == ""
    assert _cleanup_probe("/lscratch/cache") == ""
    # Nothing staged at all: a no-op.
    assert _cleanup_probe(None) == ""


def test_srun_child_keeps_the_script_environment(tmp_path, fake_slurm):
    # sbatch --export=NONE ALSO sets SLURM_EXPORT_ENV=NONE inside the job,
    # and srun reads it — stripping every export the script just made from
    # the child (live 2026-07-17: the GPU worker started with no
    # STEERLAB_*/HF_* at all and bound loopback). The script must flip
    # SLURM_EXPORT_ENV to ALL after its exports and before srun.
    bundle = _bundle(tmp_path)
    with open(bundle.script_path, encoding="utf-8") as handle:
        script = handle.read()
    assert "export SLURM_EXPORT_ENV=ALL" in script
    flip = script.index("export SLURM_EXPORT_ENV=ALL")
    assert script.index("#SBATCH --export=NONE") < flip < script.index("\nsrun ")
    # The flip comes after the bundle's own exports (they precede runtime
    # reconstruction, which precedes srun).
    for key in bundle.env:
        assert script.index(f"export {key}=") < flip


# --- cancel: scancel's exit code is the result (live shakedown 2026-07-16) -------

def test_cancel_returns_scancel_exit_status(tmp_path, fake_slurm, monkeypatch):
    assert SlurmExecutor().cancel("777") is True
    monkeypatch.setenv("FAKE_SCANCEL_FAIL", "scancel: error: kill job error")
    assert SlurmExecutor().cancel("777") is False
    assert fake_slurm.calls("scancel") == ["777", "777"]


def test_jobmanager_keeps_job_alive_when_scancel_fails(
        tmp_path, fake_slurm, monkeypatch):
    # A stop that stamps "cancelled" on a failed scancel hides a billed
    # orphan: the job must stay non-terminal so the reconciler keeps
    # following it and the caller can retry.
    manager = JobManager(store=DurableJobStore(path=str(tmp_path / "jobs.sqlite")))
    job = manager.record_external("session:gpu", status="submitted",
                                  executor="slurm", executor_job_id="999")
    monkeypatch.setenv("FAKE_SCANCEL_FAIL", "scancel: error")
    # cancel() reports achieved-or-honestly-accepted (follow-up review
    # 2026-07-16): a failed scancel returns False so callers can't answer ok.
    assert manager.cancel(job.id) is False
    assert job.status == "submitted"          # nothing terminal happened
    assert any("exited nonzero" in line for line in job.all_logs())
    monkeypatch.delenv("FAKE_SCANCEL_FAIL")
    assert manager.cancel(job.id) is True     # retry succeeds
    assert job.status == "cancelled"
    # Cooperative local cancels still count as accepted.
    local = manager.record_external("local-thing", status="pending",
                                    executor="local")
    assert manager.cancel(local.id) is True
    assert local.status == "cancelled"


# --- maintenance window -----------------------------------------------------------

def _calendar(tmp_path, start_in_minutes, duration_hours=4):
    start = datetime.now(timezone.utc) + timedelta(minutes=start_in_minutes)
    end = start + timedelta(hours=duration_hours)
    path = tmp_path / "maintenance.json"
    path.write_text(json.dumps([{"start": start.isoformat(),
                                 "end": end.isoformat()}]), encoding="utf-8")
    return str(path)


def test_crosses_maintenance_window_refusal_and_pass(tmp_path):
    calendar = _calendar(tmp_path, start_in_minutes=30)
    assert crosses_maintenance_window("04:00:00", calendar) is True
    assert crosses_maintenance_window("00:10:00", calendar) is False
    assert crosses_maintenance_window("04:00:00", None) is False
    assert crosses_maintenance_window("04:00:00", str(tmp_path / "missing.json")) is False


def test_submit_refuses_walltime_crossing_maintenance_window(
        tmp_path, fake_slurm, monkeypatch):
    monkeypatch.setenv("STEERLAB_MAINTENANCE_CALENDAR",
                       _calendar(tmp_path, start_in_minutes=30))
    executor = SlurmExecutor()  # profile snapshot AFTER the env is set
    with pytest.raises(RuntimeError, match="maintenance window"):
        executor.submit(_bundle(tmp_path))
    assert fake_slurm.calls("sbatch") == []  # refused before reaching sbatch

    short = SlurmResources(gres="A100", walltime="00:10:00")
    assert executor.submit(_bundle(tmp_path, resources=short)) == "12345"


# --- poll_state mapping -------------------------------------------------------------

@pytest.mark.parametrize("scheduler_state,exit_code,expected", [
    ("COMPLETED", "0:0", "succeeded"),
    ("FAILED", "1:0", "failed"),
    ("FAILED", "85:0", "checkpointed"),
    ("TIMEOUT", "0:15", "failed"),
    ("OUT_OF_MEMORY", "0:125", "failed"),
    ("REQUEUED", "0:0", "submitted"),
    ("PENDING", "0:0", "submitted"),
    ("RUNNING", "0:0", "running"),
    ("PREEMPTED", "0:0", "submitted"),
    ("RESIZING", "0:0", "running"),
    ("CANCELLED by 1234", "0:0", "cancelled"),
])
def test_poll_state_maps_sacct_states(fake_slurm, scheduler_state, exit_code, expected):
    fake_slurm.set_state("555", scheduler_state, exit_code=exit_code)
    assert SlurmExecutor().poll_state("555") == expected


def test_poll_state_falls_back_to_squeue_when_sacct_is_silent(fake_slurm):
    fake_slurm.set_state("777", "", queue="RUNNING")
    assert SlurmExecutor().poll_state("777") == "running"
    assert any("-j 777" in line for line in fake_slurm.calls("squeue"))


def test_poll_state_unknown_job_is_none(fake_slurm):
    assert SlurmExecutor().poll_state("999") is None


# --- site-configurable poll binaries (site wrapper commands) -------------------------

@pytest.fixture
def wrapper_only_slurm(tmp_path, monkeypatch):
    """A site where NO raw scheduler binary exists: only wrapper-named doubles
    (``sacct-site``, ``sq``, ``sbatch-wrap``, ``scancel-wrap``) are installed.
    All four STEERLAB_SLURM_* command keys must substitute the binary end to
    end, or nothing works here at all — which is the point of WP5 G5: before
    Step 8, submit and cancel were literals and only the two poll verbs were
    site data."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for source, installed in (("sacct", "sacct-site"), ("squeue", "sq"),
                              ("sbatch", "sbatch-wrap"), ("scancel", "scancel-wrap")):
        target = bindir / installed
        shutil.copy(os.path.join(FAKEBIN_SOURCE, source), target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log_dir = tmp_path / "calls"
    log_dir.mkdir()
    state_file = tmp_path / "slurm-state.json"
    # bindir FIRST but the original PATH stays (python3 for the shebangs);
    # a developer's real sacct/squeue further down would still prove nothing
    # because the calls log below only fills when OUR doubles run.
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE", str(state_file))
    monkeypatch.setenv("STEERLAB_SLURM_SACCT", "sacct-site")
    monkeypatch.setenv("STEERLAB_SLURM_SQUEUE", "sq")
    monkeypatch.setenv("STEERLAB_SLURM_SBATCH", "sbatch-wrap")
    monkeypatch.setenv("STEERLAB_SLURM_SCANCEL", "scancel-wrap")
    monkeypatch.delenv("FAKE_SBATCH_FAIL", raising=False)
    monkeypatch.delenv("FAKE_SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)

    class Handle:
        def set_state(self, job_id, state, exit_code="0:0", queue=None):
            table = {}
            if state_file.exists():
                table = json.loads(state_file.read_text(encoding="utf-8"))
            table[str(job_id)] = {"state": state, "exit": exit_code, "queue": queue}
            state_file.write_text(json.dumps(table), encoding="utf-8")

        def calls(self, binary):
            path = log_dir / f"{binary}.calls"
            if not path.exists():
                return []
            return path.read_text(encoding="utf-8").splitlines()

    return Handle()


def test_poll_state_uses_configured_sacct_wrapper(wrapper_only_slurm):
    wrapper_only_slurm.set_state("616", "FAILED", exit_code="85:0")
    assert SlurmExecutor().poll_state("616") == "checkpointed"
    # Same arguments, different binary — only the name substitutes.
    calls = wrapper_only_slurm.calls("sacct")  # fakebin logs under its source name
    assert any("-j 616 -n -o State,ExitCode -P" in line for line in calls)


def test_poll_state_squeue_fallback_uses_configured_wrapper(wrapper_only_slurm):
    wrapper_only_slurm.set_state("717", "", queue="RUNNING")
    assert SlurmExecutor().poll_state("717") == "running"
    assert any("-j 717" in line for line in wrapper_only_slurm.calls("squeue"))


def test_poll_state_resources_record_the_wrapper_names(wrapper_only_slurm):
    resources = SlurmResources.from_env()
    assert resources.sacct_command == "sacct-site"
    assert resources.squeue_command == "sq"
    assert resources.sbatch_command == "sbatch-wrap"
    assert resources.scancel_command == "scancel-wrap"


def test_submit_and_cancel_use_the_configured_wrapper_binaries(
        tmp_path, wrapper_only_slurm):
    """The end-to-end half of G5: with no raw ``sbatch``/``scancel`` on PATH,
    a submission still lands and a cancellation still runs — through the
    declared wrappers, with our own arguments unchanged."""
    bundle = _bundle(tmp_path)
    assert SlurmExecutor().submit(bundle).isdigit()
    calls = wrapper_only_slurm.calls("sbatch")  # fakebin logs under its source name
    assert len(calls) == 1 and bundle.script_path in calls[0]
    assert SlurmExecutor().cancel("888") is True
    assert wrapper_only_slurm.calls("scancel") == ["888"]


# --- cancel ---------------------------------------------------------------------

def test_cancel_invokes_scancel(fake_slurm):
    SlurmExecutor().cancel("888")
    assert fake_slurm.calls("scancel") == ["888"]


# --- the reconciler through the fake binaries ---------------------------------------

def _manager(tmp_path):
    return JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      capability_provider=lambda: {})


def test_reconciler_folds_poll_transitions_and_child_record(tmp_path, fake_slurm):
    records = tmp_path / "records"
    records.mkdir()
    mgr = _manager(tmp_path)
    job = mgr.record_external(
        "study-submit", status="submitted", executor="slurm",
        executor_job_id="9100", result={"recordsDirectory": str(records)})

    fake_slurm.set_state("9100", "RUNNING")
    assert mgr.poll_slurm() == 1
    assert mgr.get(job.id).status == "running"
    assert mgr.get(job.id).finished_at is None

    (records / f"{job.id}.json").write_text(json.dumps({
        "id": job.id, "status": "succeeded",
        "result": {"runDirectory": "runs/x"},
        "elapsedSeconds": 12.5, "recordCount": 42,
    }), encoding="utf-8")
    fake_slurm.set_state("9100", "COMPLETED")
    assert mgr.poll_slurm() == 1
    updated = mgr.get(job.id)
    assert updated.status == "succeeded"
    assert updated.finished_at is not None
    assert updated.result["runDirectory"] == "runs/x"
    # WS2 child-record contract fields fold into the durable result.
    assert updated.result["elapsedSeconds"] == 12.5
    assert updated.result["recordCount"] == 42


def test_reconciler_keeps_requeued_job_alive(tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job = mgr.record_external("study-submit", status="running", executor="slurm",
                              executor_job_id="9200")
    fake_slurm.set_state("9200", "REQUEUED")
    assert mgr.poll_slurm() == 1
    requeued = mgr.get(job.id)
    assert requeued.status == "submitted"   # queued again, NOT failed
    assert requeued.finished_at is None
    # The poller keeps following it back into running.
    fake_slurm.set_state("9200", "RUNNING")
    assert mgr.poll_slurm() == 1
    assert mgr.get(job.id).status == "running"


def test_reconciler_reports_checkpoint_exit_as_resumable_not_failed(
        tmp_path, fake_slurm):
    records = tmp_path / "records"
    records.mkdir()
    mgr = _manager(tmp_path)
    job = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="9300", result={"recordsDirectory": str(records)})
    (records / f"{job.id}.json").write_text(json.dumps({
        "id": job.id, "status": "checkpointed",
        "result": {"runDirectory": "runs/y",
                   "resumeState": {"completedRecords": 7, "reason": "signal"}},
        "elapsedSeconds": 3.0, "recordCount": 7,
    }), encoding="utf-8")
    fake_slurm.set_state("9300", "FAILED", exit_code="85:0")
    assert mgr.poll_slurm() == 1
    parked = mgr.get(job.id)
    assert parked.status == "checkpointed"
    assert parked.finished_at is None       # non-terminal: resumable, not dead
    assert parked.result["recordCount"] == 7
    assert any("resumable" in line for line in parked.all_logs())
    # Stable across further polls (no status thrash).
    assert mgr.poll_slurm() == 0

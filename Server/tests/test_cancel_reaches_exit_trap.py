"""A cancelled Slurm job must reach its EXIT trap (2026-09-07).

The rendered sbatch script trapped USR1 and TERM with ONE handler that
forwarded a checkpoint signal and returned; the main loop then re-waited on
the child. A plain ``scancel`` (the engine's cancel path) sends SIGTERM and,
after the scheduler's ``KillWait`` (30 s where this was observed), SIGKILL to
the whole step. A child slower than that to wind down — a 27B model
mid-generation — was killed together with the batch shell, the EXIT trap
never ran, and the staged model stayed on the node: batch step CANCELLED
with signal 9 at +29 s, 52 GB left behind. Normal exits and the walltime
USR1 path always cleaned up.

Now TERM has its own handler: it forwards TERM and starts a bounded watchdog
(``SlurmResources.term_grace_seconds``, well inside ``KillWait``) that
SIGKILLs a child still alive at the bound, so the main ``wait`` returns and
the script exits through its EXIT trap — node-scratch cleanup, then the
job-end marker.

The shell-level tests here run the WHOLE rendered script under a real bash
with the ``srun`` line swapped for a stub child, send it the signal Slurm
would, and read what is left on disk. Every fixture name is synthetic.
"""

import os
import signal
import subprocess
import time

import pytest

from steerlab_server import node_scratch
from steerlab_server.api.executors import (DEFAULT_TERM_GRACE_SECONDS,
                                           SlurmExecutor, SlurmResources,
                                           job_end_marker_path,
                                           render_slurm_script,
                                           signal_trap_lines)

# A child that IGNORES TERM and never ends on its own: the slow wind-down
# the watchdog exists for. ``exec`` so the SIGKILL lands on the process the
# script knows, not on a wrapper shell around it.
IGNORES_TERM = "bash -c 'trap \"\" TERM; touch \"$STEERLAB_TEST_READY\"; exec sleep 60'"
# A child that checkpoints on TERM within the grace: exits 85 like the real
# study child (``CheckpointFlag`` parks the run on SIGUSR1 and SIGTERM).
CHECKPOINTS_ON_TERM = (
    "bash -c 'trap \"exit 85\" TERM; touch \"$STEERLAB_TEST_READY\"; "
    "while :; do sleep 0.1; done'")
CHECKPOINTS_ON_USR1 = (
    "bash -c 'trap \"exit 85\" USR1; touch \"$STEERLAB_TEST_READY\"; "
    "while :; do sleep 0.1; done'")


def _bundle(tmp_path, monkeypatch, *, grace=None, use_srun=True):
    # A job-scoped template under tmp, carried VERBATIM so the cleanup
    # function's own guards see the literal variable text — the real
    # cleanup runs here, not a stub.
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR",
                       str(tmp_path / "node-scratch" / "$SLURM_JOB_ID"))
    monkeypatch.delenv("STEERLAB_NODE_SCRATCH_PURGED_BY_SCHEDULER", raising=False)
    kwargs = {"walltime": "01:00:00", "use_srun": use_srun}
    if grace is not None:
        kwargs["term_grace_seconds"] = grace
    return SlurmExecutor().create_bundle(
        str(tmp_path / "receipt" / "slurm"), ["true"],
        resources=SlurmResources(**kwargs))


# --- the rendered text ---------------------------------------------------------------

def test_the_two_signals_get_two_traps(tmp_path, monkeypatch):
    """USR1 stays the checkpoint forward; TERM forwards TERM and arms the
    watchdog. The combined trap of before is gone, not merely extended."""
    script = render_slurm_script(_bundle(tmp_path, monkeypatch))
    assert "trap checkpoint USR1\n" in script
    assert "trap terminate TERM\n" in script
    assert "trap checkpoint USR1 TERM" not in script
    checkpoint = script.split("checkpoint() {")[1].split("\n}\n")[0]
    assert 'kill -USR1 "${STEERLAB_CHILD_PID}"' in checkpoint
    assert "TERM" not in checkpoint and "sleep" not in checkpoint
    terminate = script.split("terminate() {")[1].split("\n}\ntrap ")[0]
    assert 'kill -TERM "${STEERLAB_CHILD_PID}"' in terminate
    assert 'sleep "${STEERLAB_TERM_GRACE_SECONDS}"' in terminate
    assert 'kill -KILL "${STEERLAB_CHILD_PID}"' in terminate
    assert "STEERLAB_TERM_WATCHDOG_PID=$!" in terminate
    # Neither trap reaps: the main loop owns status collection.
    assert "wait " not in checkpoint and "wait " not in terminate


def test_the_grace_is_rendered_from_the_resources(tmp_path, monkeypatch):
    assert DEFAULT_TERM_GRACE_SECONDS == 15          # well inside KillWait's 30 s
    script = render_slurm_script(_bundle(tmp_path, monkeypatch))
    assert f"STEERLAB_TERM_GRACE_SECONDS={DEFAULT_TERM_GRACE_SECONDS}\n" in script
    script = render_slurm_script(_bundle(tmp_path, monkeypatch, grace=7))
    assert "STEERLAB_TERM_GRACE_SECONDS=7\n" in script


def test_the_grace_reads_the_env_and_refuses_nonsense(monkeypatch):
    monkeypatch.setenv("STEERLAB_SLURM_TERM_GRACE_SECONDS", "9")
    assert SlurmResources.from_env().term_grace_seconds == 9
    monkeypatch.setenv("STEERLAB_SLURM_TERM_GRACE_SECONDS", "soon")
    with pytest.raises(ValueError, match="not an integer"):
        SlurmResources.from_env()
    with pytest.raises(ValueError, match="non-negative"):
        signal_trap_lines(-1)


def test_the_traps_are_installed_after_cleanup_and_before_the_launch(
        tmp_path, monkeypatch):
    """Ordering is the composition: cleanup and marker traps first (EXIT),
    then the signal traps, then the child — so a TERM at any point after
    launch has a child to forward to and an EXIT trap to fall through."""
    script = render_slurm_script(_bundle(tmp_path, monkeypatch))
    assert (script.index("trap cleanup_node_scratch EXIT")
            < script.index("trap steerlab_job_end EXIT")
            < script.index("trap checkpoint USR1\n")
            < script.index("trap terminate TERM")
            < script.index("srun true &"))
    # The watchdog is dismissed once the child is gone, before the exit.
    tail = script.split("done\nset -e")[1]
    assert 'kill "${STEERLAB_TERM_WATCHDOG_PID}"' in tail
    assert tail.index("STEERLAB_TERM_WATCHDOG_PID") < tail.index(
        'exit "${STEERLAB_CHILD_STATUS}"')


def test_the_exec_path_documents_that_it_gets_no_trap(tmp_path, monkeypatch):
    """``use_srun`` off replaces the shell: the block is rendered (the text
    is one renderer) but nothing survives the exec. Recorded, not fixed."""
    script = render_slurm_script(_bundle(tmp_path, monkeypatch, use_srun=False))
    assert script.rstrip().endswith("exec true")
    assert "trap terminate TERM" in script
    assert "srun" not in script.split("trap terminate TERM")[1]


def test_the_ad_hoc_wrapper_keeps_no_signal_trap(monkeypatch):
    """The wrapper runs its payload in the FOREGROUND, so a trapped TERM
    would fire only after the payload died — under a cancel, never. It
    stays untrapped on purpose (test below shows the EXIT trap still runs)."""
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/node-scratch/$SLURM_JOB_ID")
    monkeypatch.delenv("STEERLAB_NODE_SCRATCH_PURGED_BY_SCHEDULER", raising=False)
    text = node_scratch.render_wrapper(
        environ=dict(os.environ), resources=SlurmResources(walltime="01:00:00"))
    traps = [line for line in text.splitlines() if line.startswith("trap ")]
    assert traps == ["trap cleanup_node_scratch EXIT"]
    assert "No TERM trap on purpose" in text


# --- the script under bash -----------------------------------------------------------

def _stage(tmp_path, job_id):
    """A staged 'model' the way a real job leaves one, under the template's
    expansion for this job id."""
    stage = tmp_path / "node-scratch" / job_id
    stage.mkdir(parents=True)
    (stage / "model.safetensors").write_bytes(b"\0" * 1024)
    return stage


def _launch(tmp_path, script: str, *, child: str, job_id: str):
    """Run the rendered script under bash with the srun line replaced by the
    stub child. Output goes to files, never pipes: the watchdog subshell
    outlives the script by up to its grace and must not hold a reader."""
    assert "srun true &" in script
    script = script.replace("srun true &", f"{child} &", 1)
    path = tmp_path / "under-test.sbatch"
    path.write_text(script, encoding="utf-8")
    ready = tmp_path / "child.ready"
    out = open(tmp_path / "job.out", "w", encoding="utf-8")
    err = open(tmp_path / "job.err", "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", str(path)], stdout=out, stderr=err,
        env={"PATH": os.environ.get("PATH", ""), "SLURM_JOB_ID": job_id,
             "STEERLAB_TEST_READY": str(ready),
             # `set -u` + `conda`/`module` are guarded on these being unset.
             "HOME": str(tmp_path)})
    deadline = time.monotonic() + 10
    while not ready.exists():
        assert proc.poll() is None, (tmp_path / "job.err").read_text()
        assert time.monotonic() < deadline, "the stub child never came up"
        time.sleep(0.05)
    return proc


def _finish(proc, tmp_path, *, within: float):
    started = time.monotonic()
    status = proc.wait(timeout=within)
    elapsed = time.monotonic() - started
    return status, elapsed, (tmp_path / "job.out").read_text(encoding="utf-8")


def test_a_cancel_of_a_slow_child_kills_it_inside_the_bound_and_cleans_up(
        tmp_path, monkeypatch):
    """THE case from the node: the child ignores TERM. The watchdog kills it
    at the grace bound, the script exits with the child's status (128+9),
    the stage directory is GONE and the job-end marker is written — all
    before a 30 s KillWait would have SIGKILLed the shell."""
    grace = 2
    bundle = _bundle(tmp_path, monkeypatch, grace=grace)
    stage = _stage(tmp_path, "5150")
    proc = _launch(tmp_path, render_slurm_script(bundle), child=IGNORES_TERM,
                   job_id="5150")
    os.kill(proc.pid, signal.SIGTERM)
    status, elapsed, out = _finish(proc, tmp_path, within=grace + 8)
    assert status == 128 + 9, out
    assert elapsed < grace + 5
    assert not stage.exists(), "the staged model was left on the node"
    marker = job_end_marker_path(bundle.bundle_dir, "5150")
    assert open(marker, encoding="utf-8").read() == f"{128 + 9}\n"
    assert "termination signal received" in out
    assert f"still alive {grace}s after termination" in out
    assert f"child exited with status {128 + 9}" in out


def test_a_child_that_checkpoints_in_time_keeps_its_85(tmp_path, monkeypatch):
    """The recorded status is the CHILD's: a cancel the child answers by
    checkpointing is 85 (resumable), the watchdog never fires, the wait does
    not stretch to the bound, and cleanup + marker still happen."""
    grace = 5
    bundle = _bundle(tmp_path, monkeypatch, grace=grace)
    stage = _stage(tmp_path, "5151")
    proc = _launch(tmp_path, render_slurm_script(bundle),
                   child=CHECKPOINTS_ON_TERM, job_id="5151")
    os.kill(proc.pid, signal.SIGTERM)
    status, elapsed, out = _finish(proc, tmp_path, within=grace + 8)
    assert status == 85, out
    assert elapsed < grace - 1
    assert not stage.exists()
    marker = job_end_marker_path(bundle.bundle_dir, "5151")
    assert open(marker, encoding="utf-8").read() == "85\n"
    assert "still alive" not in out


def test_the_walltime_signal_is_still_a_plain_checkpoint(tmp_path, monkeypatch):
    """USR1 (``--signal=B:USR1@N``) forwards a checkpoint and arms nothing:
    the child parks with 85 and the shell simply records it."""
    bundle = _bundle(tmp_path, monkeypatch, grace=5)
    stage = _stage(tmp_path, "5152")
    proc = _launch(tmp_path, render_slurm_script(bundle),
                   child=CHECKPOINTS_ON_USR1, job_id="5152")
    os.kill(proc.pid, signal.SIGUSR1)
    status, _elapsed, out = _finish(proc, tmp_path, within=10)
    assert status == 85, out
    assert "checkpoint signal received" in out
    assert "termination signal" not in out and "still alive" not in out
    assert not stage.exists()
    assert os.path.isfile(job_end_marker_path(bundle.bundle_dir, "5152"))


def test_a_normal_exit_is_untouched(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, monkeypatch, grace=5)
    stage = _stage(tmp_path, "5153")
    child = "bash -c 'touch \"$STEERLAB_TEST_READY\"; exit 3'"
    proc = _launch(tmp_path, render_slurm_script(bundle), child=child,
                   job_id="5153")
    status, _elapsed, out = _finish(proc, tmp_path, within=10)
    assert status == 3, out
    assert not stage.exists()
    assert open(job_end_marker_path(bundle.bundle_dir, "5153"),
                encoding="utf-8").read() == "3\n"
    assert "termination signal" not in out


def test_a_term_before_the_child_exists_leaves_through_the_exit_trap(
        tmp_path, monkeypatch):
    """Nothing to wind down yet (module loads, env reconstruction): the
    handler exits 128+15 at once and the EXIT trap still runs."""
    bundle = _bundle(tmp_path, monkeypatch, grace=5)
    script = render_slurm_script(bundle)
    # Cut the script off after the traps and call the handler by hand.
    head = script.split("srun true &")[0]
    probe = head + "terminate\necho unreachable\n"
    (tmp_path / "probe.sbatch").write_text(probe, encoding="utf-8")
    stage = _stage(tmp_path, "5154")
    result = subprocess.run(
        ["bash", str(tmp_path / "probe.sbatch")], capture_output=True,
        text=True, env={"PATH": os.environ.get("PATH", ""),
                        "SLURM_JOB_ID": "5154", "HOME": str(tmp_path)})
    assert result.returncode == 128 + 15, result.stderr
    assert "unreachable" not in result.stdout
    assert not stage.exists()
    assert open(job_end_marker_path(bundle.bundle_dir, "5154"),
                encoding="utf-8").read() == f"{128 + 15}\n"


def test_the_ad_hoc_wrapper_cleans_up_on_a_cancel_without_a_trap(
        tmp_path, monkeypatch):
    """The claim behind leaving the wrapper alone: its payload is in the
    foreground and TERM is untrapped, so bash leaves through the EXIT trap
    immediately and the stage directory goes — while the payload is still
    being wound down by the scheduler's own TERM-then-KILL."""
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR",
                       str(tmp_path / "node-scratch" / "$SLURM_JOB_ID"))
    monkeypatch.delenv("STEERLAB_NODE_SCRATCH_PURGED_BY_SCHEDULER", raising=False)
    text = node_scratch.render_wrapper(
        environ=dict(os.environ), resources=SlurmResources(walltime="01:00:00"))
    path = tmp_path / "wrapper.sbatch"
    path.write_text(text, encoding="utf-8")
    stage = _stage(tmp_path, "5155")
    out = open(tmp_path / "wrapper.out", "w", encoding="utf-8")
    # A short-lived payload, so the orphan Slurm would reap is gone soon.
    proc = subprocess.Popen(
        ["bash", str(path), "sleep", "5"], stdout=out, stderr=subprocess.STDOUT,
        env={"PATH": os.environ.get("PATH", ""), "SLURM_JOB_ID": "5155",
             "HOME": str(tmp_path)})
    time.sleep(0.5)
    os.kill(proc.pid, signal.SIGTERM)
    status = proc.wait(timeout=3)
    # bash runs the EXIT trap, then re-raises the signal on itself, so the
    # scheduler sees a job that died of the cancel's TERM — as it should.
    assert status == -signal.SIGTERM
    assert not stage.exists()

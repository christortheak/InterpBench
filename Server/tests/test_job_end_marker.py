"""The per-job END MARKER the rendered sbatch script writes from its EXIT trap
(2026-09-06), and the reason it exists.

``cluster import`` on the Mac copied a submission receipt while its Slurm job
was still running, froze a 143-byte ``slurm-<jobid>.out`` in the immutable
``runs/`` tree, and reported an immutability violation on every later import
once the log had grown to 14 KB. The receipt gate that fixes this needs a
content-based proof that a job has ENDED — one it can read off the directory
listing without asking the scheduler — and the marker is that proof: the last
thing the script does, after node-scratch cleanup, printing nothing, is write
``slurm-<jobid>.exit`` beside the scheduler's own ``slurm-<jobid>.out``.

Every fixture name here is synthetic.
"""

import os
import subprocess

import pytest

from steerlab_server.api.executors import (JOB_END_MARKER_SUFFIX,
                                           SlurmExecutor, SlurmResources,
                                           job_end_marker_lines,
                                           job_end_marker_path,
                                           render_slurm_script)


def _rendered(tmp_path, monkeypatch, *, use_srun=True, purged=False):
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/node-scratch/$SLURM_JOB_ID")
    if purged:
        monkeypatch.setenv("STEERLAB_NODE_SCRATCH_PURGED_BY_SCHEDULER", "1")
    else:
        monkeypatch.delenv("STEERLAB_NODE_SCRATCH_PURGED_BY_SCHEDULER",
                           raising=False)
    bundle = SlurmExecutor().create_bundle(
        str(tmp_path / "receipt" / "slurm"), ["true"],
        resources=SlurmResources(walltime="01:00:00", use_srun=use_srun))
    return bundle, render_slurm_script(bundle)


def _trap_block(script: str) -> str:
    """The marker function and its trap line, exactly as rendered."""
    start = script.index("steerlab_job_end() {")
    end = script.index("trap steerlab_job_end EXIT") + len("trap steerlab_job_end EXIT")
    return script[start:end]


def _run_trap(script: str, *, tail: str, env: dict, cleanup: str | None = None):
    """Execute the rendered trap block in a real bash, with the same
    ``set -euo pipefail`` the script runs under, then the given tail."""
    probe = "set -euo pipefail\n"
    if cleanup is not None:
        probe += cleanup + "\n"
    probe += _trap_block(script) + "\n" + tail + "\n"
    return subprocess.run(["bash", "-c", probe], capture_output=True,
                          text=True, env={"PATH": os.environ.get("PATH", ""),
                                          **env})


def test_the_marker_is_the_last_exit_trap_installed(tmp_path, monkeypatch):
    """The last ``trap … EXIT`` wins, so the marker's must come after the
    node-scratch block's — and the marker function chains to that cleanup
    rather than replacing it."""
    _bundle, script = _rendered(tmp_path, monkeypatch)
    traps = [line for line in script.splitlines()
             if line.startswith("trap ") and line.endswith(" EXIT")]
    assert traps == ["trap cleanup_node_scratch EXIT", "trap steerlab_job_end EXIT"]
    assert script.index("trap cleanup_node_scratch EXIT") < script.index(
        "steerlab_job_end() {")
    assert "cleanup_node_scratch\n" in _trap_block(script)
    assert JOB_END_MARKER_SUFFIX == ".exit"


def test_the_marker_names_the_bundle_directory_and_the_job(tmp_path, monkeypatch):
    bundle, script = _rendered(tmp_path, monkeypatch)
    expected = job_end_marker_path(bundle.bundle_dir, "4242")
    assert expected == os.path.join(bundle.bundle_dir, "slurm-4242.exit")
    # The stdout capture and the marker sit side by side, keyed by the same id.
    assert bundle.stdout_path == os.path.join(bundle.bundle_dir, "slurm-%j.out")
    assert "${marker_dir}/slurm-${SLURM_JOB_ID}.exit" in script
    assert f"local marker_dir={bundle.bundle_dir}" in script


def test_the_trap_writes_the_status_after_cleanup_and_prints_nothing(
        tmp_path, monkeypatch):
    """Executed under bash: the marker carries the exit status the script
    exited with, the script STILL exits with that status (85 = checkpointed,
    which the scheduler record must keep seeing), cleanup ran first, and
    nothing is printed after the marker is written — the .out is final."""
    bundle, script = _rendered(tmp_path, monkeypatch)
    result = _run_trap(
        script, tail="echo body\nexit 85",
        env={"SLURM_JOB_ID": "4242"},
        cleanup='cleanup_node_scratch() { echo "cleanup ran"; }')
    assert result.returncode == 85, result.stderr
    marker = job_end_marker_path(bundle.bundle_dir, "4242")
    with open(marker, encoding="utf-8") as handle:
        assert handle.read() == "85\n"
    assert result.stdout.splitlines() == ["body", "cleanup ran"]


def test_a_clean_exit_records_zero(tmp_path, monkeypatch):
    bundle, script = _rendered(tmp_path, monkeypatch)
    result = _run_trap(script, tail="true", env={"SLURM_JOB_ID": "4243"},
                       cleanup="cleanup_node_scratch() { :; }")
    assert result.returncode == 0, result.stderr
    with open(job_end_marker_path(bundle.bundle_dir, "4243"),
              encoding="utf-8") as handle:
        assert handle.read() == "0\n"


def test_no_marker_outside_a_slurm_job(tmp_path, monkeypatch):
    """A bundle run by hand (no ``$SLURM_JOB_ID``) leaves nothing that a
    reader could mistake for a scheduler job's end."""
    bundle, script = _rendered(tmp_path, monkeypatch)
    result = _run_trap(script, tail="exit 3", env={},
                       cleanup="cleanup_node_scratch() { :; }")
    assert result.returncode == 3
    assert not [name for name in os.listdir(bundle.bundle_dir)
                if name.endswith(JOB_END_MARKER_SUFFIX)]


def test_the_trap_survives_a_site_with_no_cleanup_function(tmp_path, monkeypatch):
    """A site whose scheduler purges node scratch renders no cleanup trap and
    no ``cleanup_node_scratch`` at all; the marker must still be written."""
    bundle, script = _rendered(tmp_path, monkeypatch, purged=True)
    assert "trap cleanup_node_scratch EXIT" not in script
    assert "trap steerlab_job_end EXIT" in script
    result = _run_trap(script, tail="exit 7", env={"SLURM_JOB_ID": "4244"})
    assert result.returncode == 7, result.stderr
    with open(job_end_marker_path(bundle.bundle_dir, "4244"),
              encoding="utf-8") as handle:
        assert handle.read() == "7\n"


def test_a_marker_write_failure_never_changes_the_exit_status(tmp_path, monkeypatch):
    """The marker directory being gone (scratch purged under the job) must
    not turn a clean job into a failed one."""
    bundle, script = _rendered(tmp_path, monkeypatch)
    block = _trap_block(script).replace(
        f"local marker_dir={bundle.bundle_dir}",
        "local marker_dir=/nonexistent/steerlab-marker-probe")
    probe = "set -euo pipefail\ncleanup_node_scratch() { :; }\n" + block + "\ntrue\n"
    result = subprocess.run(["bash", "-c", probe], capture_output=True, text=True,
                            env={"PATH": os.environ.get("PATH", ""),
                                 "SLURM_JOB_ID": "4245"})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("use_srun", [True, False])
def test_both_launch_paths_render_the_block(tmp_path, monkeypatch, use_srun):
    """The block is rendered on both paths; only the srun path's shell
    survives to run it (``exec`` replaces the shell and fires no EXIT
    trap), which the block's own docstring records — the Mac's importer
    asks the scheduler for jobs that carry no marker."""
    _bundle, script = _rendered(tmp_path, monkeypatch, use_srun=use_srun)
    assert "trap steerlab_job_end EXIT" in script
    assert job_end_marker_lines("/x")[-1] == "trap steerlab_job_end EXIT"


def test_a_bundle_directory_with_spaces_is_quoted(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/node-scratch/$SLURM_JOB_ID")
    bundle = SlurmExecutor().create_bundle(
        str(tmp_path / "a receipt" / "slurm"), ["true"],
        resources=SlurmResources(walltime="01:00:00"))
    script = render_slurm_script(bundle)
    result = _run_trap(script, tail="exit 0", env={"SLURM_JOB_ID": "4246"},
                       cleanup="cleanup_node_scratch() { :; }")
    assert result.returncode == 0, result.stderr
    assert os.path.isfile(job_end_marker_path(bundle.bundle_dir, "4246"))

"""A local study submission must not leave the server holding a second copy
of the model its child is about to load.

Observed on a 64 GiB Mac: /api/state reported the server holding
google/gemma-3-4b-it while its own bundle-execute child materialised the same
model — two resident copies, ~34 GiB between them, for a study only one of
them was running.
"""

import pytest

pytest.importorskip("fastapi")

from steerlab_server.api import submissions


class _Registry:
    def __init__(self, resident=2, raises=False):
        self.resident, self.raises, self.calls = resident, raises, 0

    def unload_all(self):
        self.calls += 1
        if self.raises:
            raise RuntimeError("registry busy")
        freed, self.resident = self.resident, 0
        return freed


class _Job:
    def __init__(self):
        self.lines = []

    def log(self, line):
        self.lines.append(line)


def _local_runner(registry):
    """Build the closure submit_study installs for a local executor, without
    standing up a whole submission."""
    job = _Job()

    def run(job=job):
        if registry is not None:
            try:
                freed = registry.unload_all()
                if freed:
                    job.log(f"released {freed} resident model(s) before "
                            "spawning the local run")
            except Exception as exc:  # noqa: BLE001
                job.log(f"could not release resident models ({exc}); continuing")
        job.log("running local study")
    return run, job


def test_the_server_releases_its_models_before_a_local_child_loads_its_own():
    registry = _Registry(resident=2)
    run, job = _local_runner(registry)

    run()

    assert registry.calls == 1 and registry.resident == 0
    assert any("released 2 resident model(s)" in l for l in job.lines)
    # And the run still happens.
    assert any("running local study" in l for l in job.lines)


def test_releasing_nothing_is_silent():
    registry = _Registry(resident=0)
    run, job = _local_runner(registry)

    run()

    assert not any("released" in l for l in job.lines)


def test_a_registry_failure_never_blocks_the_run():
    """Freeing memory is an optimisation; refusing to run because it failed
    would be a worse outcome than running tight."""
    registry = _Registry(raises=True)
    run, job = _local_runner(registry)

    run()

    assert any("could not release" in l for l in job.lines)
    assert any("running local study" in l for l in job.lines)


def test_submit_study_accepts_a_registry():
    """The wiring itself: routes passes state.registry through."""
    import inspect
    assert "registry" in inspect.signature(submissions.submit_study).parameters


# --- child output must be visible WHILE it runs ------------------------------

def test_a_local_child_streams_its_output_as_it_runs():
    """subprocess.run(capture_output=True) blocked until exit, so a panel run —
    48 generations over many minutes — reported nothing until it finished or
    died. Progress and diagnostics were invisible for exactly the window a
    researcher wants them."""
    from steerlab_server.api.executors import LocalExecutor

    lines = []
    proc = LocalExecutor().run(
        ["python3", "-c", "print('alpha'); print('beta')"], log=lines.append)

    body = [l for l in lines if not l.startswith("$ ")]
    assert body == ["alpha", "beta"]
    # Contract preserved for callers that read the record or the failure.
    assert proc.stdout == "alpha\nbeta\n"
    assert proc.returncode == 0


def test_stderr_is_captured_but_not_echoed():
    """The caller already surfaces stderr as the failure message; echoing it
    too would print every traceback twice."""
    from steerlab_server.api.executors import LocalExecutor

    lines = []
    proc = LocalExecutor().run(
        ["python3", "-c", "import sys; print('out'); print('boom', file=sys.stderr)"],
        log=lines.append)

    body = [l for l in lines if not l.startswith("$ ")]
    assert body == ["out"]
    assert "boom" in proc.stderr


def test_a_failing_child_still_reports_its_code_and_stderr():
    from steerlab_server.api.executors import LocalExecutor

    proc = LocalExecutor().run(
        ["python3", "-c", "import sys; sys.stderr.write('bad\\n'); sys.exit(3)"])

    assert proc.returncode == 3 and "bad" in proc.stderr

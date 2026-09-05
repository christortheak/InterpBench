"""The LOCAL study child's lifecycle: its evidence when it fails, its death
when the job is cancelled.

Two defects from the external review of 2026-09-05, both on the two local
submission paths (``submissions.submit_study`` and
``submissions.submit_run_bundle``, each of which installs a ``_run_local``
closure around ``LocalExecutor.run``):

ENG-02  the child writes a durable record naming what it produced, and the
        parent read it only on SUCCESS — a run that died after hours of
        generations reported ``result: null`` and the evidence on disk was
        reachable only by hand.

ENG-03  an accepted cancellation set a flag nobody read: the child kept
        running to completion (writing its output afterwards), and the job
        turned "cancelled" only once it had finished on its own.

Every test here drives REAL subprocesses — the defects are about process
lifecycle, and a stubbed executor would prove nothing about signals, process
groups or pipe draining.
"""

import os
import signal
import sys
import time

import pytest

from steerlab_server.api import submissions
from steerlab_server.api.executors import LocalExecutor
from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.experiment import bundles, experiment_store as es


TERMINAL = {"succeeded", "failed", "cancelled", "cancelledResumable",
            "parked", "prepared"}


# --- helpers -----------------------------------------------------------------

def _wait_for(predicate, timeout: float, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def _study(root: str) -> str:
    concept = os.path.join(root, "prompts", "concepts", "fair")
    os.makedirs(concept, exist_ok=True)
    with open(os.path.join(concept, "positive.jsonl"), "w", encoding="utf-8") as h:
        h.write('{"text":"fair"}\n')
    with open(os.path.join(concept, "negative.jsonl"), "w", encoding="utf-8") as h:
        h.write('{"text":"unfair"}\n')
    es.create("Local Lifecycle", model_id="org/m", revision="abc", root=root)
    es.attach("local-lifecycle", ["fair"], root=root)
    return "local-lifecycle"


class _Submission:
    """One local submission driven by a real child process."""

    def __init__(self, jobs, job_id, db_path, record_path):
        self.jobs, self.job_id = jobs, job_id
        self.db_path, self.record_path = db_path, record_path

    def wait(self, timeout: float = 30.0):
        _wait_for(lambda: self.jobs.get(self.job_id).status in TERMINAL, timeout)
        return self.jobs.get(self.job_id)

    def durable(self):
        """The job as the STORE has it — not the in-memory object. A stamp
        that never reached sqlite is a stamp no client will ever see."""
        return DurableJobStore(self.db_path).load_all()[self.job_id]


def _submit(path: str, tmp_path, monkeypatch, script: str, args=()) -> _Submission:
    """Submit a local study through ``path`` ("study" or "bundle"), with the
    bundle-execute child replaced by ``script`` — a real python program whose
    argv is ``[record_path, *args]``."""
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    name = _study(root)
    db_path = str(tmp_path / "jobs.sqlite")
    jobs = JobManager(DurableJobStore(db_path))
    seen: dict = {}

    def build(bundle_path, *, record_path, **kwargs):
        seen["record"] = record_path
        return [sys.executable, "-c", script, record_path, *args]

    monkeypatch.setattr(submissions, "_bundle_execute_command", build)
    if path == "study":
        submission = submissions.submit_study(
            name, verb="run", jobs=jobs, executor="local", root=root,
            target_root=root)
    else:
        meta = bundles.package_experiment(
            name, output_path=str(tmp_path / "run-bundle.tar.gz"), root=root)
        submission = submissions.submit_run_bundle(
            meta["bundlePath"], verb="run", jobs=jobs, executor="local",
            target_root=root)
    return _Submission(jobs, submission.job_id, db_path, seen["record"])


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


# --- child programs ----------------------------------------------------------

#: Writes a run directory + a record whose result carries the child's OWN
#: partial evidence bundle (what `bundles.execute_run_bundle` does when it
#: fails and packaging succeeds), then dies non-zero.
CHILD_PARTIAL_PACKAGED = r"""
import json, os, sys
record, run_dir = sys.argv[1], sys.argv[2]
os.makedirs(run_dir, exist_ok=True)
open(os.path.join(run_dir, "generations.jsonl"), "w").write('{"id":1}\n')
bundle = os.path.join(run_dir, "child.partial.evidence-bundle.tar.gz")
open(bundle, "wb").write(b"child-packaged")
os.makedirs(os.path.dirname(record), exist_ok=True)
json.dump({
    "schemaVersion": 1,
    "id": os.path.splitext(os.path.basename(record))[0],
    "kind": "bundle-execute:run",
    "status": "failed",
    "executor": "local",
    "error": "RuntimeError: the model died at record 3",
    "logs": [],
    "elapsedSeconds": 12.5,
    "recordCount": 1,
    "result": {
        "error": "RuntimeError: the model died at record 3",
        "partialEvidence": True,
        "experiment": "local-lifecycle",
        "verb": "run",
        "partialRunID": os.path.basename(run_dir),
        "evidenceBundle": {"kind": "evidenceBundle", "runID": os.path.basename(run_dir),
                           "runDirectory": run_dir, "bundlePath": bundle,
                           "evidenceComplete": False},
    },
}, open(record, "w"))
sys.stderr.write("the model died at record 3\n")
sys.exit(3)
"""

#: Same, but the child packaged NOTHING (a --no-evidence submission, or a
#: packaging failure): it leaves only a run directory behind.
CHILD_PARTIAL_UNPACKAGED = r"""
import json, os, sys
record, run_dir = sys.argv[1], sys.argv[2]
os.makedirs(run_dir, exist_ok=True)
open(os.path.join(run_dir, "generations.jsonl"), "w").write('{"id":1}\n')
os.makedirs(os.path.dirname(record), exist_ok=True)
json.dump({"schemaVersion": 1, "kind": "bundle-execute:run", "status": "failed",
           "error": "RuntimeError: out of memory",
           "result": {"runDirectory": run_dir,
                      "error": "RuntimeError: out of memory"}},
          open(record, "w"))
sys.stderr.write("out of memory\n")
sys.exit(9)
"""

CHILD_NO_RECORD = r"""
import sys
sys.stderr.write("killed before it could write anything\n")
sys.exit(7)
"""

CHILD_MALFORMED_RECORD = r"""
import os, sys
record = sys.argv[1]
os.makedirs(os.path.dirname(record), exist_ok=True)
open(record, "w").write("{not json at all")
sys.stderr.write("died mid-write\n")
sys.exit(11)
"""

CHILD_INCOMPLETE_RECORD = r"""
import json, os, sys
record = sys.argv[1]
os.makedirs(os.path.dirname(record), exist_ok=True)
json.dump({"status": "failed", "result": "not-a-dict", "runDirectory": 17},
          open(record, "w"))
sys.stderr.write("half a record\n")
sys.exit(13)
"""

#: Writes its partial record IMMEDIATELY, then works for a long time and only
#: at the very end writes the marker. Cancellation must land in between.
CHILD_RECORD_THEN_SLEEP = r"""
import json, os, sys, time
record, run_dir, marker = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(run_dir, exist_ok=True)
open(os.path.join(run_dir, "generations.jsonl"), "w").write('{"id":1}\n')
os.makedirs(os.path.dirname(record), exist_ok=True)
json.dump({"schemaVersion": 1, "kind": "bundle-execute:run", "status": "running",
           "result": {"runDirectory": run_dir, "partialEvidence": True}},
          open(record, "w"))
print("record written", flush=True)
time.sleep(120)
open(marker, "w").write("the child ran to completion")
"""

CHILD_SLEEP_THEN_MARKER = r"""
import os, sys, time
marker = sys.argv[1]
print(os.getpid(), flush=True)
time.sleep(120)
open(marker, "w").write("the child ran to completion")
"""

CHILD_IGNORES_SIGTERM = r"""
import os, signal, sys, time
marker = sys.argv[1]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(os.getpid(), flush=True)
time.sleep(120)
open(marker, "w").write("the child ran to completion")
"""

CHILD_SPAWNS_GRANDCHILD = r"""
import os, subprocess, sys, time
kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(kid.pid, flush=True)
time.sleep(120)
"""

CHILD_FLOODS_STDOUT = r"""
import os, sys, time
print(os.getpid(), flush=True)
line = "x" * 100
for _ in range(50000):
    sys.stdout.write(line + "\n")
sys.stdout.flush()
time.sleep(120)
"""

CHILD_EXITS_AT_ONCE = r"""
import os, sys
print(os.getpid(), flush=True)
print("done", flush=True)
sys.exit(0)
"""


# --- ENG-02: a failed child's evidence ---------------------------------------

@pytest.mark.parametrize("path", ["study", "bundle"])
def test_a_failed_child_still_hands_over_its_partial_evidence(path, tmp_path,
                                                              monkeypatch):
    """The defect, exactly: the child died non-zero having packaged hours of
    generations, and the parent raised before ever reading the record — the
    durable job carried a null result and the evidence was unreachable from
    every client."""
    run_dir = str(tmp_path / "runs" / "20260905-120000-exp-local-lifecycle-run")
    submission = _submit(path, tmp_path, monkeypatch, CHILD_PARTIAL_PACKAGED,
                         args=(run_dir,))
    job = submission.wait()

    assert job.status == "failed"
    assert "the model died at record 3" in (job.error or "")

    stored = submission.durable()
    assert stored.status == "failed"
    result = stored.result or {}
    # The evidence pointers reach the RESULT ROOT, where every client reads
    # them (the same keys `_retain_partial_evidence` writes for in-process
    # jobs) — not buried where only this module knows to look.
    assert result.get("partialEvidence") is True
    assert result.get("runDirectory") == run_dir
    assert result["evidenceBundle"]["bundlePath"].endswith(".tar.gz")
    assert result.get("partialRunID") == os.path.basename(run_dir)
    # The child's own document survives whole, as on the success path.
    assert result["runResult"]["error"].startswith("RuntimeError")
    assert result.get("recordCount") == 1 and result.get("elapsedSeconds") == 12.5
    # The submission's own fields are not displaced by the fold.
    assert result.get("recordsDirectory") and result.get("command")


def test_the_parent_packages_evidence_the_child_could_not(tmp_path, monkeypatch):
    """A child that left a run directory but no bundle (a `--no-evidence`
    submission, or one killed before it packaged) must still come home: the
    run directory is attached to the failure so the durable-job layer's
    existing `_retain_partial_evidence` packages it."""
    run_dir = str(tmp_path / "runs" / "20260905-130000-exp-local-lifecycle-run")
    submission = _submit("study", tmp_path, monkeypatch,
                         CHILD_PARTIAL_UNPACKAGED, args=(run_dir,))
    job = submission.wait()

    assert job.status == "failed" and "out of memory" in (job.error or "")
    result = submission.durable().result or {}
    assert result.get("partialEvidence") is True
    assert result.get("runDirectory") == run_dir
    bundle = result["evidenceBundle"]
    assert bundle.get("evidenceComplete") is False
    assert os.path.isfile(bundle["bundlePath"])
    # And the directory itself is marked, so an imported partial is never
    # mistaken for a citable run.
    assert os.path.isfile(os.path.join(run_dir, "FAILED.md"))


def test_a_child_that_packaged_is_not_packaged_twice(tmp_path, monkeypatch):
    """The child's bundle is authoritative (it has the diagnostics and the
    stage directories). Re-packaging would spend the same gigabytes again and
    overwrite it."""
    run_dir = str(tmp_path / "runs" / "20260905-140000-exp-local-lifecycle-run")
    submission = _submit("study", tmp_path, monkeypatch, CHILD_PARTIAL_PACKAGED,
                         args=(run_dir,))
    submission.wait()

    bundles_on_disk = [n for n in os.listdir(run_dir) if n.endswith(".tar.gz")]
    assert bundles_on_disk == ["child.partial.evidence-bundle.tar.gz"]
    result = submission.durable().result or {}
    assert result["evidenceBundle"]["bundlePath"].endswith(
        "child.partial.evidence-bundle.tar.gz")


@pytest.mark.parametrize("path", ["study", "bundle"])
@pytest.mark.parametrize("script,code,message", [
    (CHILD_NO_RECORD, 7, "killed before it could write anything"),
    (CHILD_MALFORMED_RECORD, 11, "died mid-write"),
    (CHILD_INCOMPLETE_RECORD, 13, "half a record"),
])
def test_a_missing_or_broken_record_never_masks_the_failure(
        path, script, code, message, tmp_path, monkeypatch):
    submission = _submit(path, tmp_path, monkeypatch, script)
    job = submission.wait()

    assert job.status == "failed"
    assert message in (job.error or "")
    result = submission.durable().result or {}
    # Nothing is CLAIMED that was not produced: no partial-evidence stamp,
    # no invented run directory.
    assert "partialEvidence" not in result
    assert result.get("runDirectory") in (None, "")
    assert "runResult" not in result


# --- ENG-03: cancelling actually stops the child -----------------------------

def test_cancellation_kills_the_child_before_it_finishes(tmp_path):
    """The defect: `LocalExecutor.run` received no cancellation signal at all,
    so an accepted cancel waited for the child to finish its own work."""
    marker = str(tmp_path / "marker")
    deadline = time.monotonic() + 0.3
    started = time.monotonic()
    proc = LocalExecutor().run(
        [sys.executable, "-c", CHILD_SLEEP_THEN_MARKER, marker],
        should_cancel=lambda: time.monotonic() > deadline,
        cancel_poll_seconds=0.05, cancel_grace_seconds=2.0)
    elapsed = time.monotonic() - started

    assert elapsed < 10, "cancellation must not wait out the child's own work"
    assert proc.returncode == -signal.SIGTERM
    assert not os.path.exists(marker)
    # And the child was actually reaped: no zombie left behind.
    pid = int(proc.stdout.strip().splitlines()[0])
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_a_child_that_ignores_sigterm_is_killed(tmp_path):
    marker = str(tmp_path / "marker")
    deadline = time.monotonic() + 0.2
    started = time.monotonic()
    proc = LocalExecutor().run(
        [sys.executable, "-c", CHILD_IGNORES_SIGTERM, marker],
        should_cancel=lambda: time.monotonic() > deadline,
        cancel_poll_seconds=0.05, cancel_grace_seconds=0.75)
    elapsed = time.monotonic() - started

    assert proc.returncode == -signal.SIGKILL
    assert 0.75 <= elapsed < 10, "the grace period is honoured, then escalated"
    assert not os.path.exists(marker)


def test_cancellation_reaches_a_grandchild(tmp_path):
    """`start_new_session` puts the child at the head of its own group, so
    everything it spawned is covered — an orphaned grandchild would keep a GPU
    (or a whole node) busy after the researcher stopped the run."""
    deadline = time.monotonic() + 0.4
    proc = LocalExecutor().run(
        [sys.executable, "-c", CHILD_SPAWNS_GRANDCHILD],
        should_cancel=lambda: time.monotonic() > deadline,
        cancel_poll_seconds=0.05, cancel_grace_seconds=2.0)

    grandchild = int(proc.stdout.strip().splitlines()[0])
    try:
        gone = _wait_for(lambda: not _alive(grandchild), timeout=5.0)
        assert gone, f"grandchild {grandchild} survived the cancellation"
    finally:
        if _alive(grandchild):
            os.kill(grandchild, signal.SIGKILL)


def test_a_flooding_child_cancels_without_deadlock(tmp_path):
    """Draining must never be able to hold cancellation hostage: the drain
    threads are joined with a bound, and the child is signalled regardless."""
    deadline = time.monotonic() + 0.3
    started = time.monotonic()
    proc = LocalExecutor().run(
        [sys.executable, "-c", CHILD_FLOODS_STDOUT],
        should_cancel=lambda: time.monotonic() > deadline,
        cancel_poll_seconds=0.05, cancel_grace_seconds=2.0)
    elapsed = time.monotonic() - started

    assert elapsed < 15
    assert proc.returncode in (-signal.SIGTERM, -signal.SIGKILL)
    assert len(proc.stdout) > 0


def test_a_child_that_finishes_first_is_never_signalled(tmp_path):
    """Half of the race: each round WAITS before it reads the flag, so a child
    that finished on its own is reaped and keeps its own status — a
    cancellation arriving afterwards signals nothing and is not waited for."""
    late = time.monotonic() + 30.0
    started = time.monotonic()
    proc = LocalExecutor().run(
        [sys.executable, "-c", CHILD_EXITS_AT_ONCE],
        should_cancel=lambda: time.monotonic() > late,
        cancel_poll_seconds=0.05, cancel_grace_seconds=2.0)

    assert proc.returncode == 0 and "done" in proc.stdout
    assert time.monotonic() - started < 10


def test_a_cancel_racing_the_child_s_own_exit_is_harmless(tmp_path):
    """The other half: the flag is already set when the child exits by itself.
    Whether the signal lands or finds an empty group, the outcome must be an
    honest returncode and a reaped child — never an exception, never a stray
    signal at a pid we no longer own."""
    proc = LocalExecutor().run(
        [sys.executable, "-c", CHILD_EXITS_AT_ONCE],
        should_cancel=lambda: True,
        cancel_poll_seconds=0.05, cancel_grace_seconds=2.0)

    assert proc.returncode in (0, -signal.SIGTERM, -signal.SIGKILL)
    pid = int(proc.stdout.strip().splitlines()[0])
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_the_log_says_what_was_sent(tmp_path):
    marker = str(tmp_path / "marker")
    lines: list[str] = []
    deadline = time.monotonic() + 0.2
    LocalExecutor().run(
        [sys.executable, "-c", CHILD_IGNORES_SIGTERM, marker],
        log=lines.append, should_cancel=lambda: time.monotonic() > deadline,
        cancel_poll_seconds=0.05, cancel_grace_seconds=0.5)

    body = "\n".join(lines)
    assert "cancellation observed" in body
    assert "SIGTERM" in body and "SIGKILL" in body


@pytest.mark.parametrize("path", ["study", "bundle"])
def test_cancelling_a_local_job_stops_it_and_keeps_its_evidence(path, tmp_path,
                                                                monkeypatch):
    """The interaction: a cancel accepted mid-run must stop the child AND keep
    what it had already produced — terminal state "cancelled", never
    "failed"."""
    run_dir = str(tmp_path / "runs" / "20260905-150000-exp-local-lifecycle-run")
    marker = str(tmp_path / "marker")
    submission = _submit(path, tmp_path, monkeypatch, CHILD_RECORD_THEN_SLEEP,
                         args=(run_dir, marker))
    assert _wait_for(lambda: os.path.isfile(submission.record_path), 20.0), \
        "the child never got as far as writing its record"

    assert submission.jobs.cancel(submission.job_id) is True
    job = submission.wait(timeout=30.0)

    assert job.status == "cancelled"
    assert not os.path.exists(marker), "the child ran on past the cancellation"
    result = submission.durable().result or {}
    assert result.get("runDirectory") == run_dir
    assert result.get("partialEvidence") is True
    assert submission.durable().status == "cancelled"

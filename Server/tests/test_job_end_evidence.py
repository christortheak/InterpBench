"""Whether the scheduler confirms a job has ENDED (2026-10-04).

Asked before a cancelled run is resumed: two processes must never write to
one run directory. ``poll_observation`` cannot answer it, because accounting
records a cancelled job as CANCELLED the moment the cancel is accepted, while
the job's processes are still being stopped (the queue shows that interval
as COMPLETING). So ``SlurmExecutor.job_end_evidence`` asks the queue first,
and treats everything it cannot establish as unknown, never as ended.
"""

from steerlab_server.api.executors import SlurmExecutor

from test_manual_resubmit import fake_slurm  # noqa: F401 - fixture by import


def test_a_job_the_queue_still_lists_has_not_ended(fake_slurm):
    executor = SlurmExecutor()
    # Accounting already says cancelled; the queue says it is winding down.
    fake_slurm.set_state("31", "CANCELLED by 1000", queue="COMPLETING")
    winding_down = executor.job_end_evidence("31")
    assert winding_down.ended is False
    assert "still lists it as COMPLETING" in winding_down.detail
    fake_slurm.set_state("32", "PENDING", queue="PENDING")
    assert executor.job_end_evidence("32").ended is False
    fake_slurm.set_state("33", "RUNNING", queue="RUNNING")
    assert executor.job_end_evidence("33").ended is False


def test_an_end_state_in_the_queue_or_in_accounting_is_an_ended_job(fake_slurm):
    executor = SlurmExecutor()
    # Gone from the queue; accounting records the end.
    fake_slurm.set_state("41", "CANCELLED by 1000")
    ended = executor.job_end_evidence("41")
    assert ended.ended is True
    assert "accounting records it as CANCELLED" in ended.detail
    fake_slurm.set_state("42", "FAILED", exit_code="85:0")
    assert executor.job_end_evidence("42").ended is True
    # The queue itself may still remember a finished job for a while.
    fake_slurm.set_state("43", "CANCELLED by 1000", queue="CANCELLED")
    assert executor.job_end_evidence("43").ended is True
    # Accounting that still says running is not an end.
    fake_slurm.set_state("44", "RUNNING")
    assert executor.job_end_evidence("44").ended is False


def test_what_cannot_be_established_is_unknown_never_ended(fake_slurm,
                                                           monkeypatch):
    executor = SlurmExecutor()
    # Unknown to both the queue and accounting.
    assert executor.job_end_evidence("51").ended is None
    fake_slurm.set_state("52", "CANCELLED by 1000")
    # Accounting does not answer.
    monkeypatch.setenv("FAKE_SACCT_FAIL", "accounting storage is down")
    assert executor.job_end_evidence("52").ended is None
    monkeypatch.delenv("FAKE_SACCT_FAIL")
    # The queue cannot be read: accounting alone is not proof.
    monkeypatch.setenv("FAKE_SQUEUE_FAIL", "Unable to contact slurm controller")
    unread = executor.job_end_evidence("52")
    assert unread.ended is None
    assert "queue could not be read" in unread.detail
    # The queue answering that it no longer knows the id is the forgotten,
    # finished job: accounting then decides.
    monkeypatch.setenv("FAKE_SQUEUE_FAIL",
                       "slurm_load_jobs error: Invalid job id specified")
    assert executor.job_end_evidence("52").ended is True

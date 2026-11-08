"""Issue #12 on the Python client: ``runner submit --parallel`` against a
runner whose scheduler meets its job limit, and ``runner status`` of a split
run with held parts.

Driven through the real parser and envelope against the real ASGI app
(``test_client_runner.wired_cli``), with the committed fake scheduler
binaries on PATH. Nothing contacts a real scheduler.
"""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from steerlab_server import client_cli

from test_sharding import _staged_bundle, fake_slurm  # noqa: F401 - fixture
from test_client_runner import (  # noqa: F401 - fixtures
    _envelope, runner_env, service, wired_cli)

RUNNER = ["--runner", "http://testserver"]
LIMIT = ("sbatch: error: QOSMaxSubmitJobPerUserLimit\\n"
         "sbatch: error: Batch job submission failed: Job violates "
         "accounting/QOS policy (job submit limit, user's size and/or time "
         "limits)")


def _submit_split_run(wired, tmp_path, monkeypatch, capsys, *, name):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    bundle = _staged_bundle(tmp_path, name=name)
    code, document, _ = _envelope(["runner", "upload", bundle, *RUNNER], capsys)
    assert code == 0, document
    staged, digest = (document["result"]["runnerPath"],
                      document["result"]["sha256"])
    argv = ["runner", "submit", *RUNNER, "--bundle-path", staged,
            "--bundle-sha", digest, "--verb", "run", "--executor", "slurm",
            "--parallel", "3", "--target-root", wired["target"]]
    code = client_cli.main(argv + ["--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_a_split_run_with_held_parts_is_a_success_with_a_note(
        wired_cli, fake_slurm, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SBATCH_FAIL_AFTER", "2")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", LIMIT)
    code, document, _ = _submit_split_run(wired_cli, tmp_path, monkeypatch,
                                          capsys, name="clientheld")
    assert code == 0, document
    held = document["result"]["shardsHeld"]
    assert held["count"] == 1 and held["partCount"] == 3
    assert document["message"].endswith(
        "; 1 of 3 parts held by SteerLab until slots free")
    assert fake_slurm.calls("scancel") == []


def test_a_split_run_refused_for_another_reason_exits_nonzero(
        wired_cli, fake_slurm, tmp_path, monkeypatch, capsys):
    """Before, the runner answered 200 and the client exited 0 while the
    parts it had submitted were cancelled and the run was failed."""
    monkeypatch.setenv("FAKE_SBATCH_FAIL_AFTER", "1")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE",
                       "sbatch: error: Invalid partition name specified")
    code, document, _ = _submit_split_run(wired_cli, tmp_path, monkeypatch,
                                          capsys, name="clientfailed")
    assert code != 0, document
    assert document["state"] != "ready"
    reason = document["error"]["reason"]
    assert "the split run was not started" in reason
    assert "Invalid partition name specified" in reason
    assert len(fake_slurm.calls("scancel")) == 1


def test_a_resume_held_without_a_reason_is_pending_with_the_repair(
        wired_cli, fake_slurm, tmp_path, monkeypatch, capsys):
    """Issue #12, review round 6: resumed, but nothing runs, because the
    scheduler refused the first part without saying why. Pending (12), with
    the typed field and the repair, not "N parts now continue"."""
    monkeypatch.setenv("FAKE_SBATCH_FAIL_AFTER", "2")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", LIMIT)
    code, document, _ = _submit_split_run(wired_cli, tmp_path, monkeypatch,
                                          capsys, name="clientresumevague")
    assert code == 0, document
    parent_id = document["result"]["jobId"]
    jobs = wired_cli["state"].jobs
    for child_id in document["result"]["shardJobIDs"][:2]:
        child = jobs.get(child_id)
        child.status, child.finished_at = "succeeded", 1.0
        child.result = {**child.result, "runDirectory": str(tmp_path / child_id)}
        jobs.store.update(child)
    assert jobs.cancel(parent_id) is True
    monkeypatch.delenv("FAKE_SBATCH_FAIL_AFTER")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=02:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE",
                       "sbatch: error: Batch job submission failed: Job "
                       "violates accounting/QOS policy (job submit limit, "
                       "user's size and/or time limits)")
    code = client_cli.main(["runner", "resubmit", parent_id, "--walltime",
                            "02:00:00", *RUNNER, "--json"])
    envelope = _documents(capsys.readouterr().out)[-1]
    assert code == 12, envelope
    assert envelope["state"] == "pending"
    assert envelope.get("error") is None
    response = envelope["result"]["response"]
    assert response["heldUnexplained"] is True
    assert "Job violates accounting/QOS policy" in response["schedulerMessage"]
    assert "held, not running" in envelope["message"]
    assert "now continue" not in envelope["message"]
    assert envelope["nextAction"]["verb"] == (
        f"steerlab runner status {parent_id} --runner http://testserver")
    assert envelope["nextAction"]["detail"] == (
        "if its parts stay held, check the walltime against the cluster's "
        "limit for one job")


def _cancelled_run_with_a_held_part(wired, tmp_path, monkeypatch, capsys,
                                    name):
    """A split run whose two submitted parts finished and whose third was
    held, cancelled: a Resume continues only the held part."""
    monkeypatch.setenv("FAKE_SBATCH_FAIL_AFTER", "2")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", LIMIT)
    code, document, _ = _submit_split_run(wired, tmp_path, monkeypatch,
                                          capsys, name=name)
    assert code == 0, document
    parent_id = document["result"]["jobId"]
    jobs = wired["state"].jobs
    for child_id in document["result"]["shardJobIDs"][:2]:
        child = jobs.get(child_id)
        child.status, child.finished_at = "succeeded", 1.0
        child.result = {**child.result, "runDirectory": str(tmp_path / child_id)}
        jobs.store.update(child)
    assert jobs.cancel(parent_id) is True
    monkeypatch.delenv("FAKE_SBATCH_FAIL_AFTER")
    return parent_id


def _resubmit(parent_id, capsys):
    code = client_cli.main(["runner", "resubmit", parent_id, "--walltime",
                            "02:00:00", *RUNNER, "--json"])
    return code, _documents(capsys.readouterr().out)[-1]


def test_a_resume_held_for_a_job_count_is_a_success_worded_as_held(
        wired_cli, fake_slurm, tmp_path, monkeypatch, capsys):
    """Review round 7: held for a named job count is exit 0, as at
    submission, and says the parts are held until a slot frees."""
    parent_id = _cancelled_run_with_a_held_part(
        wired_cli, tmp_path, monkeypatch, capsys, "clientresumecount")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=02:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", LIMIT)
    code, envelope = _resubmit(parent_id, capsys)
    assert code == 0, envelope
    assert envelope["state"] == "ready"
    assert envelope["result"]["response"]["heldReason"] == "jobLimit"
    assert "is held until the cluster's job limit frees a slot" in \
        envelope["message"]
    assert "now continue" not in envelope["message"]


def test_a_resume_whose_sbatch_went_unanswered_is_worded_as_held(
        wired_cli, fake_slurm, tmp_path, monkeypatch, capsys):
    from steerlab_server.api import executors
    parent_id = _cancelled_run_with_a_held_part(
        wired_cli, tmp_path, monkeypatch, capsys, "clientresumesilent")

    def unanswered(self, bundle, **kwargs):
        raise executors.SchedulerSubmitError(
            "sbatch did not answer within 120s and was abandoned",
            returncode=executors.SCHEDULER_TIMEOUT_EXIT)

    monkeypatch.setattr(executors.SlurmExecutor, "submit", unanswered)
    code, envelope = _resubmit(parent_id, capsys)
    assert code == 0, envelope
    assert envelope["result"]["response"]["heldReason"] == "schedulerUnanswered"
    assert "is held until the scheduler answers" in envelope["message"]
    assert "now continue" not in envelope["message"]


POLICY = ("sbatch: error: Batch job submission failed: Job violates "
          "accounting/QOS policy (job submit limit, user's size and/or time "
          "limits)")


def test_a_gave_up_run_retries_the_same_walltime_only_when_asked(
        wired_cli, fake_slurm, tmp_path, monkeypatch, capsys):
    """Review rounds 8 and 9: a run that gave up after a day of refusals
    without a reason is refused at the same walltime (65) with both ways on
    named in the reason and the repair, and resumes at exactly that walltime
    with --retry-refused-walltime."""
    monkeypatch.setenv("FAKE_SBATCH_FAIL_AFTER", "0")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", LIMIT)
    code, document, _ = _submit_split_run(wired_cli, tmp_path, monkeypatch,
                                          capsys, name="clientgaveup")
    assert code == 0, document
    parent_id = document["result"]["jobId"]
    first_id = document["result"]["shardJobIDs"][0]
    jobs = wired_cli["state"].jobs
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", POLICY)
    for backdate in (False, True):
        first = jobs.get(first_id)
        held = dict(first.result["held"])
        held["nextAttemptAt"] = 0.0
        if backdate:
            held["unexplainedSince"] = held["lastAttemptAt"] - 24 * 3600 - 1
        first.result = {**first.result, "held": held}
        jobs.store.update(first)
        jobs._held_deferred_until.clear()
        jobs._held_retry_after = 0.0
        jobs.poll_slurm()
    assert jobs.get(parent_id).status == "failed"
    assert "heldGaveUp" in jobs.get(parent_id).result
    # The slots free.
    monkeypatch.delenv("FAKE_SBATCH_FAIL_AFTER")
    monkeypatch.delenv("FAKE_SBATCH_FAIL_MESSAGE")
    calls = len(fake_slurm.calls("sbatch"))

    code = client_cli.main(["runner", "resubmit", parent_id, *RUNNER,
                            "--json"])
    envelope = _documents(capsys.readouterr().out)[-1]
    assert code == 65, envelope
    walltime = jobs.get(parent_id).result["heldGaveUp"]["walltime"]
    assert (f"refused its parts at walltime {walltime}, without a reason, "
            "for a day") in envelope["error"]["reason"]
    assert (f"steerlab runner resubmit {parent_id} --retry-refused-walltime "
            "--runner <url>") in envelope["error"]["reason"]
    # Review round 9: the repair is the runner's, naming both ways on, not
    # the client's generic "read its reason".
    repair = envelope["error"]["repairAction"]
    assert repair.startswith("Choose one: resume it with a walltime within")
    assert (f"steerlab runner resubmit {parent_id} --walltime HH:MM:SS "
            "--runner <url>") in repair
    assert f"retry walltime {walltime} (" in repair
    assert len(fake_slurm.calls("sbatch")) == calls

    code = client_cli.main(["runner", "resubmit", parent_id,
                            "--retry-refused-walltime", *RUNNER, "--json"])
    envelope = _documents(capsys.readouterr().out)[-1]
    assert code == 0, envelope
    response = envelope["result"]["response"]
    assert response["retriedRefusedWalltime"] == walltime
    resumed = response["resumedShards"]
    # Review round 9: the first part is handed to the scheduler in the
    # request, at the walltime that was refused; the controller submits the
    # rest, in order, at it.
    assert [entry["held"] for entry in resumed] == [False, True, True]
    jobs._held_retry_after = 0.0
    jobs.poll_slurm()
    sent = fake_slurm.calls("sbatch")[calls:]
    assert len(sent) == 3 and all(f"--time={walltime}" in call for call in sent)
    assert all(jobs.get(entry["jobId"]).status == "submitted"
               for entry in resumed)


def _documents(text: str) -> list:
    """Every JSON document printed, in order (the verb prints the runner's
    answer, then the envelope)."""
    decoder, documents, index = json.JSONDecoder(), [], 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return documents
        document, index = decoder.raw_decode(text, start)
        documents.append(document)


def test_runner_status_shows_held_parts_in_plain_words():
    from steerlab_server.client import job_reads
    document = {
        "jobId": "p1", "status": "submitted", "notes": [
            "1 of 2 parts is held by SteerLab: the cluster's job limit is "
            "reached."],
        "shards": [
            {"jobId": "a", "status": "running", "executorJobID": "6401",
             "shardIndex": 0, "shardCount": 2, "found": True,
             "held": None, "heldNote": None, "notes": []},
            {"jobId": "b", "status": "pending", "executorJobID": None,
             "shardIndex": 1, "shardCount": 2, "found": True,
             "held": {"reason": "clusterJobLimit", "since": 1_700_000_000.0},
             "heldNote": ("held by SteerLab: the cluster's job limit is "
                          "reached; it is submitted when a slot frees"),
             "notes": []},
        ],
    }
    lines = job_reads.status_lines(document)
    assert "  part 2 of 2: job b [pending]" in lines
    held_line = next(line for line in lines if "held by SteerLab:" in line
                     and line.startswith("    "))
    assert held_line.startswith(
        "    held by SteerLab: the cluster's job limit is reached; it is "
        "submitted when a slot frees (held since ")
    assert job_reads.status_message(document) == (
        "job p1 is submitted; 1 of 2 parts held by SteerLab until slots free")
    assert job_reads.status_message(document["shards"][1]) == (
        "job b is pending (held by SteerLab until the cluster's job limit "
        "frees a slot)")

    # A runner that sends the stamp but no sentence still reads plainly.
    del document["shards"][1]["heldNote"]
    assert any(line.startswith("    " + job_reads.HELD_PART_NOTE)
               for line in job_reads.status_lines(document))
    # A part that is not held says nothing of it.
    assert not any("held" in line for line in job_reads.status_lines(
        document["shards"][0]))

"""A study submission sent twice lands once (public issue #11).

A large study's submission is answered only after the server has measured
every task prompt, which took about eight minutes for 400 long transcripts.
The clients gave up at sixty seconds, reported a failure, and the server went
on to create the job; submitted again, the study became a second job.

What is pinned here:

1. **The server's repeat rule** (``api.submission_keys``): a request with a
   key is answered with the existing job while that job is being prepared or
   has not ended, and submitted afresh once it has ended. Two concurrent
   requests with one key make one job. A request with no key (every older
   client) is submitted every time, as before. The key survives a restart.
2. **The route** answers a repeat end to end, refuses a malformed key
   before anything, and advertises ``cluster.submissionKey``. The Python
   client's half is ``test_submission_client_waits.py``.

No model, scheduler, or network beyond loopback is reached.
"""

import os
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from steerlab_server.api import submission_keys, submissions
from steerlab_server.api.jobs import DurableJobStore, JobManager


KEY = "a" * 64
OTHER_KEY = "b" * 64


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    return JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      sweep_orphans=False)


class FakeSubmit:
    """Stands in for ``submit_run_bundle``: records one Slurm job per call.

    ``gate`` (an Event) holds the call open, the way the real one is held
    open while the server measures every task prompt."""

    def __init__(self, jobs, *, gate=None, status="submitted", verb="run",
                 dry_run=False):
        self.jobs = jobs
        self.gate = gate
        self.status = status
        self.verb = verb
        self.dry_run = dry_run
        self.calls = 0
        self.started = threading.Event()

    def __call__(self):
        self.calls += 1
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never released the submit"
        job = self.jobs.record_external(
            "study-submit-bundle", status=self.status, executor="slurm",
            executor_job_id=f"fake-scheduler-{self.calls}",
            requested_resources={"walltime": "04:00:00",
                                 "preflight": {"checks": [], "verdict": "ok"}},
            result={"recordsDirectory": "/fake/records"})
        return submissions.StudySubmission(
            job.id, "demo", self.verb, "slurm", self.dry_run,
            {"kind": "runBundle", "experiment": "demo",
             "bundleSha256": "c" * 64, "entries": [{"path": "big"}] * 3},
            {"script_path": "/fake/run.sbatch"}, f"fake-scheduler-{self.calls}",
            ["python", "-m", "steerlab_server.cli"], "/fake/records",
            "/fake/submission", preflight={"checks": [], "verdict": "ok"})


def _once(jobs, submit, key=KEY, verb="run", dry_run=False):
    return submission_keys.submit_once(key, jobs=jobs, verb=verb,
                                       dry_run=dry_run, submit=submit)


# --- 1. the server's repeat rule ----------------------------------------------


def test_a_repeat_while_the_job_has_not_ended_is_answered_with_it(jobs):
    submit = FakeSubmit(jobs)
    first = _once(jobs, submit)
    second = _once(jobs, submit)

    assert submit.calls == 1
    assert "alreadySubmitted" not in first
    assert second["alreadySubmitted"] is True
    assert second["jobId"] == first["jobId"]
    assert second["currentStatus"] == "submitted"
    assert second["alreadySubmittedNote"] == (
        f"Already submitted: this request was submitted before as job "
        f"{first['jobId']}, which is submitted. Nothing was submitted again. "
        "Follow it in the server's job list, or cancel it before submitting "
        "again.")
    # Worded for either client: no Mac-only verb.
    assert "remote " not in second["alreadySubmittedNote"]
    # The answer a client decodes is complete: the fields every client reads.
    for key in ("experiment", "verb", "executor", "dryRun", "runBundle",
                "command", "recordsDirectory", "submissionDirectory",
                "slurmJobID", "preflight"):
        assert key in second, key
    assert second["runBundle"]["bundleSha256"] == "c" * 64
    # The key is persisted on the record (additive).
    record = jobs.get(first["jobId"])
    assert record.requested_resources["submissionKey"] == KEY
    assert record.requested_resources["submissionAnswer"]["jobId"] == first["jobId"]
    # Compact: the bundle's file list is not copied onto the record.
    assert "entries" not in record.requested_resources["submissionAnswer"]["runBundle"]


@pytest.mark.parametrize("ended", ["failed", "cancelled", "succeeded",
                                   "cancelledResumable"])
def test_a_repeat_after_the_job_ended_is_a_new_submission(jobs, ended):
    submit = FakeSubmit(jobs)
    first = _once(jobs, submit)
    record = jobs.get(first["jobId"])
    record.status = ended
    jobs.store.update(record)

    second = _once(jobs, submit)

    assert submit.calls == 2
    assert second["jobId"] != first["jobId"]
    assert "alreadySubmitted" not in second


def test_a_request_without_a_key_submits_every_time(jobs):
    """An older client sends no key, and is answered exactly as before."""
    submit = FakeSubmit(jobs)
    first = _once(jobs, submit, key=None)
    second = _once(jobs, submit, key=None)

    assert submit.calls == 2
    assert first["jobId"] != second["jobId"]
    assert "alreadySubmitted" not in second
    assert "submissionKey" not in jobs.get(first["jobId"]).requested_resources


def test_a_different_key_is_a_different_submission(jobs):
    submit = FakeSubmit(jobs)
    _once(jobs, submit, key=KEY)
    _once(jobs, submit, key=OTHER_KEY)
    assert submit.calls == 2


def test_two_concurrent_requests_with_one_key_make_one_job(jobs):
    """THE issue-#11 shape: the first request is still being prepared (the
    server is measuring prompts) when the client, having given up, sends it
    again. The second waits and is answered with the job the first made."""
    gate = threading.Event()
    submit = FakeSubmit(jobs, gate=gate)
    answers: dict = {}

    def send(name):
        answers[name] = _once(jobs, submit)

    first = threading.Thread(target=send, args=("first",))
    first.start()
    assert submit.started.wait(5)
    second = threading.Thread(target=send, args=("second",))
    second.start()
    time.sleep(0.2)
    assert "second" not in answers, "the repeat must wait for the first"
    gate.set()
    first.join(5)
    second.join(5)

    assert submit.calls == 1
    assert answers["second"]["jobId"] == answers["first"]["jobId"]
    assert answers["second"]["alreadySubmitted"] is True
    assert "was being prepared when this request arrived" in \
        answers["second"]["alreadySubmittedNote"]
    assert len(jobs.list()) == 1


def test_a_repeat_of_a_dry_run_in_preparation_gets_its_answer(jobs):
    """A dry run is terminal at birth (``prepared``), yet a repeat that
    arrived while it was being measured is still answered with it: it is
    the same request, and measuring it twice helps no one."""
    gate = threading.Event()
    submit = FakeSubmit(jobs, gate=gate, status="prepared", dry_run=True)
    answers: dict = {}
    first = threading.Thread(
        target=lambda: answers.setdefault("first", _once(jobs, submit,
                                                         dry_run=True)))
    first.start()
    assert submit.started.wait(5)
    second = threading.Thread(
        target=lambda: answers.setdefault("second", _once(jobs, submit,
                                                          dry_run=True)))
    second.start()
    time.sleep(0.2)
    gate.set()
    first.join(5)
    second.join(5)
    assert submit.calls == 1
    assert answers["second"]["jobId"] == answers["first"]["jobId"]
    # Once it has been answered, a later dry run measures again.
    third = _once(jobs, submit, dry_run=True)
    assert submit.calls == 2 and third["jobId"] != answers["first"]["jobId"]


def test_a_first_request_that_was_refused_leaves_the_repeat_to_submit(jobs):
    """A first request that created no job (a refusal) leaves nothing to
    answer with: the waiting repeat submits for itself."""
    gate = threading.Event()
    calls = {"n": 0}
    real = FakeSubmit(jobs)

    def submit():
        calls["n"] += 1
        if calls["n"] == 1:
            assert gate.wait(5)
            raise submissions.SubmissionRefusal("refused", code="x",
                                                repair_action="fix it")
        return real()

    outcome: dict = {}

    def first():
        try:
            _once(jobs, submit)
        except submissions.SubmissionRefusal as exc:
            outcome["first"] = exc

    t1 = threading.Thread(target=first)
    t1.start()
    time.sleep(0.1)
    t2 = threading.Thread(
        target=lambda: outcome.setdefault("second", _once(jobs, submit)))
    t2.start()
    time.sleep(0.1)
    gate.set()
    t1.join(5)
    t2.join(5)
    assert isinstance(outcome["first"], submissions.SubmissionRefusal)
    assert "alreadySubmitted" not in outcome["second"]
    assert calls["n"] == 2


def test_the_key_survives_a_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    db = str(tmp_path / "jobs.sqlite")
    before = JobManager(DurableJobStore(db), sweep_orphans=False)
    first = _once(before, FakeSubmit(before))

    after = JobManager(DurableJobStore(db), sweep_orphans=False)
    submit = FakeSubmit(after)
    second = _once(after, submit)
    assert submit.calls == 0
    assert second["jobId"] == first["jobId"]
    assert second["alreadySubmitted"] is True


def test_a_resumed_job_is_followed_to_the_job_carrying_it(jobs):
    """A checkpointed record continued by a resubmission names its
    continuation; the continuation (which copies the key) is not a second
    submission, and while it runs a repeat names both."""
    submit = FakeSubmit(jobs)
    first = _once(jobs, submit)
    original = jobs.get(first["jobId"])
    continuation = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        executor_job_id="fake-scheduler-9",
        requested_resources={**original.requested_resources,
                             "resubmitChain": [original.id]})
    original.status = "checkpointed"
    original.result = {**(original.result or {}),
                       "resubmittedAs": continuation.id}
    jobs.store.update(original)

    second = _once(jobs, submit)
    assert submit.calls == 1
    assert second["jobId"] == original.id
    assert second["currentJobID"] == continuation.id
    assert f"continues as job {continuation.id}, which is running" in \
        second["alreadySubmittedNote"]

    continuation.status = "succeeded"
    jobs.store.update(continuation)
    third = _once(jobs, submit)
    assert submit.calls == 2 and "alreadySubmitted" not in third


def test_a_checkpointed_job_names_the_resume(jobs):
    submit = FakeSubmit(jobs)
    first = _once(jobs, submit)
    record = jobs.get(first["jobId"])
    record.status = "checkpointed"
    jobs.store.update(record)
    second = _once(jobs, submit)
    assert submit.calls == 1
    assert (f"resume job {record.id} with your client's resubmit verb"
            in second["alreadySubmittedNote"])


def test_a_key_reused_for_another_verb_is_refused(jobs):
    _once(jobs, FakeSubmit(jobs), verb="run")
    with pytest.raises(submission_keys.SubmissionKeyError) as raised:
        _once(jobs, FakeSubmit(jobs, verb="analyze"), verb="analyze")
    assert raised.value.code == "submissionKeyConflict"
    assert raised.value.status == 409


@pytest.mark.parametrize("bad", ["", "A" * 64, "a" * 63, 7, "g" * 64])
def test_a_malformed_key_is_refused(bad):
    with pytest.raises(submission_keys.SubmissionKeyError) as raised:
        submission_keys.from_body({"submissionKey": bad})
    assert raised.value.code == "submissionKeyMalformed"
    assert submission_keys.from_body({}) is None


# --- the route, end to end ------------------------------------------------------


@pytest.fixture
def route(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api import routes
    from steerlab_server.api.routes import ServiceState, build_router
    from steerlab_server.experiment import bundles, experiment_store as es

    source = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", source)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    concept = os.path.join(source, "prompts", "concepts", "signal")
    os.makedirs(concept, exist_ok=True)
    for name, text in (("positive.jsonl", '{"text":"on"}\n'),
                       ("negative.jsonl", '{"text":"off"}\n')):
        with open(os.path.join(concept, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    es.create("demo", model_id="org/m", revision="abc", root=source)
    es.attach("demo", ["signal"], root=source)
    meta = bundles.package_experiment("demo", root=source)
    state = ServiceState()
    calls: list = []

    def fake_submit_run_bundle(bundle_path, *, verb, jobs, **kwargs):
        calls.append(kwargs)
        return FakeSubmit(jobs, verb=verb)()

    monkeypatch.setattr(routes, "submit_run_bundle", fake_submit_run_bundle)
    app = FastAPI()
    app.include_router(build_router(state))
    with TestClient(app) as client:
        yield {"http": client, "bundle": meta["bundlePath"], "calls": calls,
               "state": state}


def test_the_route_answers_a_repeat_with_the_existing_job(route):
    body = {"bundlePath": route["bundle"], "verb": "run", "executor": "slurm",
            "submissionKey": KEY}
    first = route["http"].post("/api/studies/submit-bundle", json=body)
    second = route["http"].post("/api/studies/submit-bundle", json=body)
    assert first.status_code == 200 and second.status_code == 200
    assert len(route["calls"]) == 1
    assert second.json()["jobId"] == first.json()["jobId"]
    assert second.json()["alreadySubmitted"] is True
    # An older client's body (no key) is submitted again, as before.
    old = dict(body)
    old.pop("submissionKey")
    route["http"].post("/api/studies/submit-bundle", json=old)
    assert len(route["calls"]) == 2


def test_the_route_refuses_a_malformed_key_before_anything(route):
    body = {"bundlePath": route["bundle"], "verb": "run",
            "submissionKey": "not-a-digest"}
    answer = route["http"].post("/api/studies/submit-bundle", json=body)
    assert answer.status_code == 400
    assert answer.json()["detail"]["code"] == "submissionKeyMalformed"
    assert route["calls"] == []


def test_the_capability_is_advertised(route):
    capabilities = route["http"].get("/api/capabilities").json()
    assert capabilities["cluster"]["submissionKey"] is True


# --- review round 2: the key on the record, and the current state ---------------


from test_sharding import _manager, _staged_bundle, fake_slurm  # noqa: E402,F401


def _real_submit(tmp_path, monkeypatch, *, name, parallel):
    """The real bundle submission, on the committed fake scheduler."""
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "target"))
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "7000")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID_FROM_CALL", "1")
    os.makedirs(tmp_path / "target", exist_ok=True)
    bundle_path = _staged_bundle(tmp_path, name=name)
    jobs = _manager(tmp_path)

    def submit():
        return submissions.submit_run_bundle(
            bundle_path, verb="run", jobs=jobs, executor="slurm",
            target_root=str(tmp_path / "target"),
            resources={"gres": "A100", "walltime": "00:30:00"},
            parallel_jobs=parallel)

    return jobs, submit


def test_the_key_is_on_the_record_from_its_creation(tmp_path, monkeypatch,
                                                    fake_slurm):
    """Written with the record, not after: when keeping the first answer
    fails, the landed job is still answered (never a 500), and a repeat is
    answered from the record rather than submitted again."""
    jobs, submit = _real_submit(tmp_path, monkeypatch, name="keyedone",
                                parallel=1)
    real_update = jobs.store.update
    calls = {"n": 0}

    def failing_update(job):
        if (job.requested_resources or {}).get(submission_keys.ANSWER_KEY):
            calls["n"] += 1
            raise OSError("disk full")
        return real_update(job)

    monkeypatch.setattr(jobs.store, "update", failing_update)
    first = _once(jobs, submit)
    assert calls["n"] == 1, "keeping the answer was attempted and failed"
    assert "alreadySubmitted" not in first
    record = jobs.store.load_all()[first["jobId"]]
    assert record.requested_resources["submissionKey"] == KEY
    assert submission_keys.ANSWER_KEY not in record.requested_resources

    monkeypatch.setattr(jobs.store, "update", real_update)
    restarted = _manager(tmp_path)
    repeat = _once(restarted, submit)
    assert repeat["alreadySubmitted"] is True
    assert repeat["jobId"] == first["jobId"]
    for key in ("experiment", "verb", "executor", "dryRun", "runBundle",
                "command", "recordsDirectory", "submissionDirectory"):
        assert key in repeat, key
    assert repeat["command"] == first["command"]
    assert len(fake_slurm.calls("sbatch")) == 1


def test_a_repeat_of_a_split_run_reports_its_held_parts_as_they_are(
        tmp_path, monkeypatch, fake_slurm):
    """A keyed split run with 2 of 4 parts held: a repeat before the tick
    says 2 are held; after the tick has submitted them, a repeat says none
    are, where the first answer (kept on the record) still said 2."""
    from test_held_shards import _limited, _slot_frees, _tick
    _limited(monkeypatch, accepted=2)
    jobs, submit = _real_submit(tmp_path, monkeypatch, name="keyedheld",
                                parallel=4)
    first = _once(jobs, submit)
    assert first["shardsHeld"]["count"] == 2
    parent = jobs.get(first["jobId"])
    assert parent.requested_resources["submissionKey"] == KEY
    for child in first["shardJobIDs"]:
        assert "submissionKey" not in jobs.get(child).requested_resources

    before = _once(jobs, submit)
    assert before["alreadySubmitted"] is True
    assert before["shardsHeld"]["count"] == 2
    assert before["shardsHeld"]["message"].startswith(
        "2 of 4 parts have been submitted; 2 are still held by SteerLab")
    assert "2 are still held" in before["alreadySubmittedNote"]

    _slot_frees(monkeypatch)
    _tick(jobs)
    assert not [cid for cid in first["shardJobIDs"]
                if jobs.is_held(jobs.get(cid))]
    after = _once(jobs, submit)
    assert after["alreadySubmitted"] is True
    assert after["shardsHeld"] is None
    assert "held" not in after["alreadySubmittedNote"]
    # The kept first answer still says 2: the repeat did not replay it.
    kept = jobs.get(first["jobId"]).requested_resources[
        submission_keys.ANSWER_KEY]
    assert kept["shardsHeld"]["count"] == 2
    # Two repeats, no new parts: 2 accepted + 1 refused + 2 after the tick.
    assert len(fake_slurm.calls("sbatch")) == 5


def test_held_parts_are_worded_by_the_reason_they_wait(jobs):
    """A part waiting for a maintenance window is not said to wait for the
    job limit (review round 3): each held part is worded by its latest
    refusal."""
    limit = "sbatch: error: QOSMaxSubmitJobPerUserLimit"
    window = "requested walltime crosses a configured maintenance window"

    def submit():
        parent = jobs.record_external(
            "study-submit-bundle", status="submitted", executor="slurm",
            requested_resources=submission_keys.keyed(
                {"parallelJobs": 3, "shardChildren": []}))
        children = []
        for index, refusal in enumerate((None, limit, window)):
            held = None if refusal is None else {
                "reason": "clusterJobLimit", "since": 1.0,
                "schedulerMessage": limit, "lastRefusal": refusal}
            child = jobs.record_external(
                "study-submit-bundle-shard",
                status="submitted" if held is None else "pending",
                executor="slurm",
                executor_job_id="fake-scheduler-1" if held is None else None,
                requested_resources={"shardIndex": index, "parentJob": parent.id},
                result={"held": held} if held else {})
            children.append(child.id)
        parent.requested_resources = {**parent.requested_resources,
                                      "shardChildren": children}
        jobs.store.update(parent)
        return submissions.StudySubmission(
            parent.id, "demo", "run", "slurm", False, {}, None, None, [],
            "/r", "/s", shard_job_ids=children)

    _once(jobs, submit)
    repeat = _once(jobs, submit)
    held = repeat["shardsHeld"]
    assert held["count"] == 2
    assert held["reasons"] == {"clusterJobLimit": 1, "maintenanceWindow": 1}
    assert held["message"].startswith(
        "1 of 3 parts have been submitted; 2 are still held by SteerLab: "
        "1 because the cluster's job limit is reached; 1 because its walltime "
        "would cross a maintenance window")
    assert "because the cluster's job limit is reached. The" not in \
        held["message"]

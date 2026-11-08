"""Issue #12, review round 3: every search for a submission by its job name
starts at the time its own record gives, never at sacct's default of today's
midnight, and a search with no such time cannot prove absence.

The callers: the resubmit path's stale-claim resolution, and the managed
campaign's export and cancel checks (the held-part path has its own tests in
``test_held_shards.py``). Each is driven across a midnight boundary with the
committed fake ``sacct``, whose name search, like the real one, starts at
today's local midnight unless ``-S`` says otherwise. Nothing contacts a real
scheduler.
"""

import datetime as dt
import json
import os
import shutil
import stat
import time
from pathlib import Path

import pytest

from steerlab_server.api import executors
from steerlab_server.api.jobs import DurableJobStore, JobManager, ResubmitRefused
from steerlab_server.experiment import diagnostic_archives as archives

FAKEBIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakebin")


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """The committed fake scheduler binaries on PATH, with a state table."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = bindir / name
        shutil.copy(os.path.join(FAKEBIN, name), target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    log_dir = tmp_path / "calls"
    log_dir.mkdir()
    state_file = tmp_path / "slurm-state.json"
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE", str(state_file))
    for key in ("FAKE_SBATCH_FAIL", "FAKE_SQUEUE_NAMED", "FAKE_SACCT_FAIL",
                "STEERLAB_MAINTENANCE_CALENDAR", "STEERLAB_SLURM_SACCT"):
        monkeypatch.delenv(key, raising=False)

    class Handle:
        def entry(self, job_id, **fields):
            table = (json.loads(state_file.read_text(encoding="utf-8"))
                     if state_file.exists() else {})
            table[job_id] = fields
            state_file.write_text(json.dumps(table), encoding="utf-8")

        def calls(self, binary):
            path = log_dir / f"{binary}.calls"
            return (path.read_text(encoding="utf-8").splitlines()
                    if path.exists() else [])

    return Handle()


def _yesterday(hours_before_midnight: float) -> dt.datetime:
    midnight = dt.datetime.now().replace(hour=0, minute=0, second=0,
                                         microsecond=0)
    return midnight - dt.timedelta(hours=hours_before_midnight)


def _stamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def _window(since: float) -> str:
    return (f"-S {executors.sacct_time(since - executors.NAME_SEARCH_MARGIN_SECONDS)}"
            " -E now")


# --- the resubmit path's stale claims -----------------------------------------

def _stale_claim(tmp_path, mgr, *, at, token="slre-dead-midnight"):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(exist_ok=True)
    script = bundle_dir / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    job = mgr.record_external(
        "study-submit", status="checkpointed", executor="slurm",
        executor_job_id="9001",
        requested_resources={"scriptPath": str(script), "walltime": "00:10:00"})
    claim = {"claimant": "dead-server", "token": token}
    if at is not None:
        claim["at"] = at
    job.result = {"resubmitClaim": claim}
    mgr.store.update(job)
    return job


def _manager(tmp_path, executor=None):
    return JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      capability_provider=lambda: {},
                      slurm_executor=executor)


def test_a_stale_claim_is_searched_from_its_own_time(tmp_path, scheduler):
    """The dead claimant's continuation was submitted yesterday and ended
    before midnight. Searched from today's midnight it read as absent, and a
    Resume submitted the checkpointed run a second time."""
    claimed = _yesterday(2).timestamp()
    scheduler.entry("9900", state="COMPLETED", exit="0:0",
                    name="slre-dead-midnight", end=_stamp(_yesterday(1)))
    mgr = _manager(tmp_path)
    job = _stale_claim(tmp_path, mgr, at=claimed)
    answer = mgr.resubmit(job.id)
    assert answer.get("alreadyResumed") is True
    assert answer["slurmJobID"] == "9900"
    assert scheduler.calls("sbatch") == []
    search = [call for call in scheduler.calls("sacct")
              if "slre-dead-midnight" in call]
    assert search and _window(claimed) in search[0]


def test_a_claim_without_its_time_is_searched_from_the_records_creation(
        tmp_path, scheduler):
    created = _yesterday(3).timestamp()
    scheduler.entry("9901", state="COMPLETED", exit="0:0",
                    name="slre-dead-midnight", end=_stamp(_yesterday(1)))
    mgr = _manager(tmp_path)
    job = _stale_claim(tmp_path, mgr, at=None)
    job.created_at = created   # the record's own earliest time
    answer = mgr.resubmit(job.id)
    assert answer.get("alreadyResumed") is True
    assert scheduler.calls("sbatch") == []
    search = [call for call in scheduler.calls("sacct")
              if "slre-dead-midnight" in call]
    assert search and _window(created) in search[0]


def test_a_search_that_cannot_take_a_start_proves_nothing(tmp_path):
    """An executor whose name search cannot be told where to start answers
    only for today: its empty answer is not proof that nothing was
    submitted, so nothing is resubmitted."""

    class TodayOnly:
        def __init__(self):
            self.submits = []

        def submit(self, bundle, *, job_name=None):
            self.submits.append(job_name)
            return "8001"

        def find_job_by_name(self, name):
            return None

        def poll_state(self, job_id):
            return None

    executor = TodayOnly()
    mgr = _manager(tmp_path, executor)
    job = _stale_claim(tmp_path, mgr, at=time.time() - 3600)
    with pytest.raises(ResubmitRefused, match="reconciliation required"):
        mgr.resubmit(job.id)
    assert executor.submits == []


# --- the managed campaign's export and cancel checks ---------------------------

@pytest.fixture
def campaign(tmp_path, monkeypatch, scheduler):
    from test_managed_campaign import materialized
    from test_scientific_execution import setup as _setup  # noqa: F401
    from steerlab_server.api import managed_campaign_engine as owner
    from steerlab_server.api.profile import ServerProfile
    from test_battery_run import _lines
    path = tmp_path / "prompts/batteries/example.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(_lines())
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / ".steerlab"))
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "workstation")
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")
    result = materialized(tmp_path)
    directory = Path(result["campaignDirectory"])
    from steerlab_server.experiment import optvec_campaign
    cell = optvec_campaign.read_campaign(str(directory))["cells"][0]["cellID"]
    bundle_path = directory / "cells" / cell / "managed-job" / "bundle.json"

    class Campaign:
        root = tmp_path
        engine = owner
        profile = ServerProfile.from_env()

        def __init__(self):
            self.directory, self.cell, self.result = directory, cell, result

        def bundle(self):
            return json.loads(bundle_path.read_text(encoding="utf-8"))

        def created_at(self, value):
            """Rewrite the cell's recorded creation (None removes it) and
            re-pin the reviewed static files, as a campaign materialized
            then would have been pinned."""
            document = self.bundle()
            if value is None:
                document.pop("createdAt", None)
            else:
                document["createdAt"] = value
            bundle_path.write_text(json.dumps(document), encoding="utf-8")
            pin = self.result["managedCampaign"]
            self.result["managedCampaign"] = {
                **pin, "staticFiles": archives.snapshot(
                    tmp_path, [f["path"] for f in pin["staticFiles"]])}

        def job_name(self):
            return self.bundle()["bundle"]["resources"]["job_name"]

    return Campaign()


def _export_check(campaign, monkeypatch):
    from steerlab_server.api import managed_campaign
    monkeypatch.setattr(managed_campaign, "context", lambda *a: (
        campaign.directory, campaign.root, campaign.result["managedCampaign"]))
    monkeypatch.setattr(managed_campaign, "action", lambda *a: {"status": {
        "cellCount": 1, "totals": {"completed": 1},
        "cells": [{"cellID": campaign.cell, "jobID": None}]}})
    return lambda: managed_campaign.require_complete("job", None,
                                                     campaign.profile)


def test_the_export_check_searches_from_the_cells_creation(
        campaign, scheduler, monkeypatch):
    created = _yesterday(3).timestamp()
    campaign.created_at(created)
    scheduler.entry("9910", state="COMPLETED", exit="0:0",
                    name=campaign.job_name(), end=_stamp(_yesterday(1)))
    _export_check(campaign, monkeypatch)()
    search = [call for call in scheduler.calls("sacct") if "--name" in call]
    assert search and _window(created) in search[0]
    # The cell's job was found across midnight, and its end confirmed.
    assert any("-j 9910" in call for call in scheduler.calls("sacct"))


def test_the_export_check_without_a_creation_time_refuses(
        campaign, scheduler, monkeypatch):
    from steerlab_server.experiment import diagnostic_archives
    campaign.created_at(None)
    with pytest.raises(diagnostic_archives.Refusal,
                       match="no recorded creation time"):
        _export_check(campaign, monkeypatch)()


def _cancel(campaign, monkeypatch):
    from test_optvec_campaign import FakeRunner
    cancelled = []
    monkeypatch.setattr(executors.SlurmExecutor, "cancel",
                        lambda self, job: cancelled.append(job) or True)
    pin = campaign.result["managedCampaign"]
    runner = FakeRunner()
    review = campaign.engine.operate(str(campaign.directory), pin, "plan",
                                     runner=runner)
    outcome = campaign.engine.operate(str(campaign.directory), pin, "cancel",
                                      review["planSHA256"], runner=runner)
    return outcome, cancelled


def test_the_cancel_check_searches_from_the_cells_creation(
        campaign, scheduler, monkeypatch):
    created = _yesterday(3).timestamp()
    campaign.created_at(created)
    _cancel(campaign, monkeypatch)
    searches = [call for call in scheduler.calls("sacct")
                if campaign.job_name() in call]
    assert searches and _window(created) in searches[0]


def test_the_cancel_check_without_a_creation_time_cancels_nothing(
        campaign, scheduler, monkeypatch):
    from steerlab_server.experiment import diagnostic_archives
    campaign.created_at(None)
    with pytest.raises(diagnostic_archives.Refusal,
                       match="nothing was cancelled"):
        _cancel(campaign, monkeypatch)
    assert scheduler.calls("scancel") == []

"""Orphaned-pipeline startup reconcile (the 2026-08-06 replication run).

The incident: the Slurm controller daemon restarted between a chain's run
and analyze stages — completed stages, null disposition, no live job, no
FAILED.md, nothing packaged; the finished results were invisible in every
UI. Contract under test:

- ``pipeline_reconcile.scan`` finds exactly the orphans (completed stages,
  null disposition, not parked, no live job following) and nothing else;
- ``pipeline_reconcile.park`` stamps the ledger loudly, additively, and
  idempotently (a parked ledger is not re-orphaned);
- ``JobManager(reconcile_pipelines=True)`` resumes a no-model remainder
  in-process (packaging evidence on completion) and PARKS what it cannot
  resume, always through a visible durable job;
- the schema literal this module holds agrees with the ledger authority.
"""

import json
import os
import time

from steerlab_server.api.jobs import (TERMINAL, DurableJobStore, JobManager,
                                      is_parked_result)
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import pipeline_reconcile
from steerlab_server.experiment import tasks


def _ledger(runs_root, run_id, *, experiment="chain", schema=2,
            disposition=None, stage_results=None,
            stages=("run", "analyze"), parked=None):
    run_dir = os.path.join(runs_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    ledger = {"schema": schema, "experiment": experiment,
              "experimentHash": "e" * 64, "manifestStatus": "draft",
              "stages": list(stages),
              "stageResults": stage_results if stage_results is not None
              else {"run": {"status": "completed",
                            "runDirectory": os.path.join(
                                runs_root, f"{run_id}-run")}},
              "disposition": disposition}
    if parked is not None:
        ledger["parked"] = parked
    with open(os.path.join(run_dir, "pipeline.json"), "w",
              encoding="utf-8") as handle:
        json.dump(ledger, handle)
    return run_dir


def test_schema_literal_agrees_with_ledger_authority():
    # The import-light duplicate must never drift from the real schema.
    assert (pipeline_reconcile.RESUMABLE_LEDGER_SCHEMA
            == tasks.PIPELINE_LEDGER_SCHEMA)


def test_scan_finds_exactly_the_orphans(tmp_path):
    runs = str(tmp_path / "runs")
    orphan = _ledger(runs, "20260806T01-exp-chain-pipeline")
    _ledger(runs, "20260806T02-exp-chain-pipeline",
            disposition="completed")           # terminal — not orphaned
    _ledger(runs, "20260806T03-exp-chain-pipeline",
            disposition="aborted")             # terminal — not orphaned
    _ledger(runs, "20260806T04-exp-chain-pipeline",
            stage_results={})                  # nothing completed — job
    _ledger(runs, "20260806T05-exp-chain-pipeline",
            parked={"reason": "already loud"})  # parked — already loud
    live = _ledger(runs, "20260806T06-exp-chain-pipeline")
    os.makedirs(str(tmp_path / "runs" / "plain-run"), exist_ok=True)

    orphans = pipeline_reconcile.scan(
        runs, {os.path.realpath(live)})
    assert [o.run_directory for o in orphans] == [orphan]
    (found,) = orphans
    assert found.experiment == "chain"
    assert found.remaining == ["analyze"]
    assert found.resumable_schema


def test_scan_returns_foreign_schema_orphans_for_parking(tmp_path):
    # A schema-1 ledger is not resumable, but it is never SILENT: the scan
    # returns it so the caller parks it with the not-resumable reason.
    runs = str(tmp_path / "runs")
    _ledger(runs, "20260806T01-exp-old-pipeline", schema=1)
    (orphan,) = pipeline_reconcile.scan(runs, set())
    assert not orphan.resumable_schema


def test_park_stamps_loudly_and_is_idempotent(tmp_path):
    runs = str(tmp_path / "runs")
    run_dir = _ledger(runs, "20260806T01-exp-chain-pipeline")
    parked = pipeline_reconcile.park(run_dir, reason="no live job")
    assert parked["reason"] == "no live job"
    assert parked["completedStages"] == ["run"]
    assert parked["remainingStages"] == ["analyze"]
    ledger = json.load(open(os.path.join(run_dir, "pipeline.json")))
    assert ledger["parked"]["by"] == "startup-reconcile"
    assert ledger["disposition"] is None  # parked is NOT a disposition
    assert ledger["updatedAt"]
    # A parked ledger is already loud — never re-orphaned.
    assert pipeline_reconcile.scan(runs, set()) == []


def _manager(tmp_path, **kwargs):
    return JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      capability_provider=lambda: {}, **kwargs)


def _wait_for_terminal(manager, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job.status in TERMINAL:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished: "
                         f"{manager.get(job_id).status}")


def _workspace(tmp_path, monkeypatch, name="chain"):
    root = str(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    es.create(name, model_id="org/m", revision="abc", root=root)
    return root


def test_startup_parks_model_needing_orphan(tmp_path, monkeypatch):
    # remaining includes a GPU stage → the daemon must not resume it
    # in-process at startup; it parks with a reason naming the way out.
    root = _workspace(tmp_path, monkeypatch)
    runs = os.path.join(root, "runs")
    run_dir = _ledger(
        runs, "20260806T01-exp-chain-pipeline",
        stages=("extract", "run"),
        stage_results={"extract": {"status": "completed",
                                   "runDirectory": os.path.join(
                                       runs, "x-extract")}})
    manager = _manager(tmp_path, reconcile_pipelines=True)
    (job,) = [j for j in manager.list()
              if j.kind == "pipeline-orphan-reconcile"]
    job = _wait_for_terminal(manager, job.id)
    assert job.status == "parked"
    assert job.result["parked"] is True
    ledger = json.load(open(os.path.join(run_dir, "pipeline.json")))
    assert "need the study model" in ledger["parked"]["reason"]
    assert "import the completed stage runs" in ledger["parked"]["reason"]


def test_startup_resumes_no_model_remainder_and_packages(tmp_path,
                                                         monkeypatch):
    # The incident case: only analyze remains — the daemon finishes the
    # chain itself and packages the evidence so the results reach the Mac.
    root = _workspace(tmp_path, monkeypatch)
    runs = os.path.join(root, "runs")
    stage_run = os.path.join(runs, "20260806T00-exp-chain-run")
    os.makedirs(stage_run, exist_ok=True)
    with open(os.path.join(stage_run, "report.json"), "w",
              encoding="utf-8") as handle:
        handle.write("{}")
    run_dir = _ledger(
        runs, "20260806T01-exp-chain-pipeline",
        stage_results={"run": {"status": "completed",
                               "runDirectory": stage_run}})

    def fake_pipeline(name, root_arg=None, *args, pipeline_run_directory,
                      **kwargs):
        ledger = json.load(open(os.path.join(pipeline_run_directory,
                                             "pipeline.json")))
        ledger["stageResults"]["analyze"] = {"status": "completed",
                                             "runDirectory": stage_run}
        ledger["disposition"] = "completed"
        with open(os.path.join(pipeline_run_directory, "pipeline.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(ledger, handle)
        return pipeline_run_directory

    monkeypatch.setattr(tasks, "pipeline", fake_pipeline)
    manager = _manager(tmp_path, reconcile_pipelines=True)
    (job,) = [j for j in manager.list()
              if j.kind == "pipeline-orphan-reconcile"]
    job = _wait_for_terminal(manager, job.id)
    assert job.status == "succeeded"
    assert job.result["runDirectory"] == run_dir
    assert "parked" not in job.result
    # Evidence packaged like a bundle execution would have — the Swift
    # auto-import recognizes this job kind and brings it home.
    bundle = job.result["evidenceBundle"]
    assert os.path.isfile(bundle["bundlePath"])
    ledger = json.load(open(os.path.join(run_dir, "pipeline.json")))
    assert ledger["disposition"] == "completed"


def test_startup_parks_a_healed_chain_whose_evidence_cannot_be_packaged(
        tmp_path, monkeypatch):
    """Evidence packaging is a SUCCESS INVARIANT (2026-08-06 review, P1).

    The reconcile exists so finished cluster results REACH the researcher. A
    resume that heals the ledger but cannot package is the original incident
    wearing a green checkmark: completed evidence on the server, unreachable
    from the app, and a job reporting plain success. It parks instead, with
    the run directory and the recovery action in the reason the app shows.
    """
    root = _workspace(tmp_path, monkeypatch)
    runs = os.path.join(root, "runs")
    stage_run = os.path.join(runs, "20260806T00-exp-chain-run")
    os.makedirs(stage_run, exist_ok=True)
    run_dir = _ledger(
        runs, "20260806T01-exp-chain-pipeline",
        stage_results={"run": {"status": "completed",
                               "runDirectory": stage_run}})

    monkeypatch.setattr(tasks, "pipeline",
                        lambda name, *a, pipeline_run_directory, **k:
                        pipeline_run_directory)

    from steerlab_server.experiment import bundles as bundles_mod

    def refusing_package(*args, **kwargs):
        raise bundles_mod.BundleError("pipeline evidence incomplete")

    monkeypatch.setattr(bundles_mod, "package_evidence", refusing_package)

    manager = _manager(tmp_path, reconcile_pipelines=True)
    (job,) = [j for j in manager.list()
              if j.kind == "pipeline-orphan-reconcile"]
    job = _wait_for_terminal(manager, job.id)

    # Not a silent success: the STATUS is parked, and the result says the
    # evidence is missing and names the directory it is missing FROM.
    assert job.status == "parked"
    assert job.result["evidenceMissing"] is True
    assert job.result["parked"] is True
    assert job.result["runDirectory"] == run_dir
    assert "BundleError" in job.result["evidencePackagingError"]
    assert "evidenceBundle" not in job.result

    # Durable and app-visible: the ledger carries the reason the awaiting-
    # import surface renders, with a recovery action in it.
    ledger = json.load(open(os.path.join(run_dir, "pipeline.json")))
    reason = ledger["parked"]["reason"]
    assert "could not be packaged" in reason
    assert run_dir in reason
    assert "Import evidence" in reason
    assert ledger["parked"]["completedStages"] == ["run"]


def test_startup_parks_when_resume_refuses(tmp_path, monkeypatch):
    # A resume refusal (e.g. genuine manifest drift) parks with the
    # refusal string as the reason — loud either way, never a stack trace
    # masquerading as silence.
    root = _workspace(tmp_path, monkeypatch)
    runs = os.path.join(root, "runs")
    run_dir = _ledger(runs, "20260806T01-exp-chain-pipeline")

    def refusing_pipeline(*args, **kwargs):
        raise ValueError("pipeline resume refused: the manifest drifted")

    monkeypatch.setattr(tasks, "pipeline", refusing_pipeline)
    manager = _manager(tmp_path, reconcile_pipelines=True)
    (job,) = [j for j in manager.list()
              if j.kind == "pipeline-orphan-reconcile"]
    job = _wait_for_terminal(manager, job.id)
    # PARKED, not succeeded (2026-08-06 review round 2, P1). The parking
    # helper returns normally, so the generic runner used to stamp this
    # "succeeded" and every client painted it green — a chain that needs a
    # human reading as a finished one.
    assert job.status == "parked" and job.result["parked"] is True
    assert job.status in TERMINAL, "nothing will ever advance a parked job"
    assert any("NOT a completion" in line for line in job.log_tail())
    ledger = json.load(open(os.path.join(run_dir, "pipeline.json")))
    assert "in-process resume refused" in ledger["parked"]["reason"]
    assert "manifest drifted" in ledger["parked"]["reason"]


def test_a_parked_result_is_recognised_by_its_marker():
    # The status is derived from the marker BOTH the result and the ledger
    # carry, so a job row and its ledger can never disagree about what
    # happened. Ordinary results are untouched.
    assert is_parked_result({"parked": True, "reason": "r"})
    assert not is_parked_result({"parked": False})
    assert not is_parked_result({"runDirectory": "/runs/x"})
    assert not is_parked_result(None)
    assert not is_parked_result("parked")


def test_a_completed_reconcile_still_succeeds(tmp_path, monkeypatch):
    # The parked status must not swallow the healthy path: a chain that
    # resumed AND packaged its evidence is a genuine success.
    root = _workspace(tmp_path, monkeypatch)
    runs = os.path.join(root, "runs")
    run_dir = _ledger(runs, "20260806T01-exp-chain-pipeline")
    monkeypatch.setattr(tasks, "pipeline",
                        lambda name, *a, pipeline_run_directory, **k:
                        pipeline_run_directory)
    from steerlab_server.experiment import bundles as bundles_mod
    monkeypatch.setattr(bundles_mod, "package_evidence",
                        lambda out: {"bundlePath": out + "/evidence.tar"})

    manager = _manager(tmp_path, reconcile_pipelines=True)
    (job,) = [j for j in manager.list()
              if j.kind == "pipeline-orphan-reconcile"]
    job = _wait_for_terminal(manager, job.id)
    assert job.status == "succeeded"
    assert job.result["evidenceBundle"]["bundlePath"].startswith(run_dir)


def test_startup_parks_schema_1_ledger_without_a_job(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    runs = os.path.join(root, "runs")
    run_dir = _ledger(runs, "20260806T01-exp-old-pipeline", schema=1)
    manager = _manager(tmp_path, reconcile_pipelines=True)
    assert [j for j in manager.list()
            if j.kind == "pipeline-orphan-reconcile"] == []
    ledger = json.load(open(os.path.join(run_dir, "pipeline.json")))
    assert "not resumable" in ledger["parked"]["reason"]


def test_startup_skips_pipelines_a_live_job_follows(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    runs = os.path.join(root, "runs")
    followed = _ledger(runs, "20260806T01-exp-chain-pipeline")
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    seeded = JobManager(store, capability_provider=lambda: {},
                        sweep_orphans=False)
    seeded.record_external(
        "study-submit-bundle", status="checkpointed", executor="slurm",
        executor_job_id="77",
        result={"verb": "pipeline", "runDirectory": followed})
    manager = JobManager(store, capability_provider=lambda: {},
                         sweep_orphans=False, reconcile_pipelines=True)
    assert [j for j in manager.list()
            if j.kind == "pipeline-orphan-reconcile"] == []
    ledger = json.load(open(os.path.join(followed, "pipeline.json")))
    assert "parked" not in ledger


def test_default_construction_never_scans(tmp_path, monkeypatch):
    # Bare JobManager constructions (tests, tools) must not walk the
    # ambient runs root — the daemon's ServiceState opts in explicitly.
    root = _workspace(tmp_path, monkeypatch)
    runs = os.path.join(root, "runs")
    run_dir = _ledger(runs, "20260806T01-exp-chain-pipeline")
    manager = _manager(tmp_path)
    assert [j for j in manager.list()
            if j.kind == "pipeline-orphan-reconcile"] == []
    assert "parked" not in json.load(
        open(os.path.join(run_dir, "pipeline.json")))

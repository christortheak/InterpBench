import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from steerlab_server.api.app import app
from steerlab_server.api.executors import SlurmExecutor, SlurmResources, render_slurm_script
from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.api.profile import ServerProfile, capability_snapshot, validate_profile
from steerlab_server.api.safe_paths import SafePathResolver


def test_capabilities_endpoint_describes_profile():
    body = TestClient(app).get("/api/capabilities").json()
    assert body["serverVersion"]
    assert body["engine"] == "python-hf-transformers"
    assert body["profile"]["profile"] in {"local", "workstation", "cluster"}
    assert "extract" in body["availableJobTypes"]


def test_token_auth_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "token")
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "secret")
    client = TestClient(app)
    assert client.get("/api/state").status_code == 401
    assert client.get("/api/state", headers={"authorization": "Bearer secret"}).status_code == 200


def test_browser_cross_origin_request_refused():
    # A page on another origin (CSRF / DNS-rebinding) is refused; a same-origin
    # loopback browser request is allowed. Non-browser clients send no Origin.
    client = TestClient(app)
    assert client.get(
        "/api/state", headers={"origin": "http://evil.example"}).status_code == 403
    assert client.get(
        "/api/state",
        headers={"origin": "http://localhost:8080", "host": "localhost:8080"},
    ).status_code == 200
    assert client.get("/api/state").status_code == 200


def test_privileged_route_requires_token_under_slurm(monkeypatch):
    # slurm/submit takes an arbitrary command; on a Slurm deployment it must be
    # token-gated even when auth_mode is left at the default "none".
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    client = TestClient(app)
    resp = client.post("/api/slurm/submit", json={"command": ["echo", "hi"], "dryRun": True})
    assert resp.status_code == 503


def test_privileged_route_open_on_local_profile():
    # On a plain local+localhost dev box the same route stays reachable (still
    # CSRF-guarded); a dry-run submit actually returns the rendered bundle,
    # proving the route is functional and not merely non-503.
    client = TestClient(app)
    resp = client.post("/api/slurm/submit", json={"command": ["echo", "hi"], "dryRun": True})
    assert resp.status_code == 200
    assert resp.json().get("dryRun") is True


def test_load_refused_on_cluster_submit_daemon(monkeypatch):
    # A cluster submit daemon never holds a resident model: GPU work is always
    # an sbatch child or a GPU session. The legacy cluster+slurm combination
    # derives to role=controller, and the 409 now points at the GPU-session
    # flow. The escape hatch is for servers inside an interactive GPU
    # allocation.
    monkeypatch.setenv("STEERLAB_SERVER_PROFILE", "cluster")
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "token")
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "sekrit")
    monkeypatch.delenv("STEERLAB_SERVER_ROLE", raising=False)
    monkeypatch.delenv("STEERLAB_ALLOW_RESIDENT_MODELS", raising=False)
    client = TestClient(app)
    resp = client.post("/api/load", json={"model": "org/tiny"},
                       headers={"Authorization": "Bearer sekrit"})
    assert resp.status_code == 409
    assert "POST /api/session/start" in resp.json()["detail"]
    # The SSE load route shares the exact same role gate (it 409s BEFORE any
    # stream begins) — a controller must refuse both spellings identically.
    resp = client.post("/api/load/stream", json={"model": "org/tiny"},
                       headers={"Authorization": "Bearer sekrit"})
    assert resp.status_code == 409
    assert "POST /api/session/start" in resp.json()["detail"]
    # Model-requiring compute (extraction etc.) with NO session: the
    # controller's answer points at the session flow too — "POST /api/load
    # first" is a dead end on a controller (live 2026-07-18).
    resp = client.post("/api/multiconcept/extract", json={},
                       headers={"Authorization": "Bearer sekrit"})
    assert resp.status_code == 409
    assert "POST /api/session/start" in resp.json()["detail"]

    monkeypatch.setenv("STEERLAB_ALLOW_RESIDENT_MODELS", "1")
    resp = client.post("/api/load", json={"model": "org/tiny"},
                       headers={"Authorization": "Bearer sekrit"})
    # With the hatch open the gate no longer answers 409; the load then fails
    # for the ordinary reason that org/tiny is not a real model (400).
    assert resp.status_code != 409


def test_real_slurm_submit_refused_without_slurm_executor(monkeypatch):
    # A non-dry-run submit on a box that hasn't declared the Slurm executor must
    # not reach sbatch — the executor declaration is what makes the route
    # token-gated, so gating on reality closes the login-node default-env hole.
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")
    client = TestClient(app)
    resp = client.post("/api/slurm/submit", json={"command": ["echo", "hi"]})
    assert resp.status_code == 403


def test_experiment_verb_post_requires_token_under_slurm(monkeypatch):
    # Experiment verbs launch GPU jobs, so POSTs there are privileged and must
    # be token-gated on a Slurm deployment even at the default auth mode.
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    client = TestClient(app)
    resp = client.post("/api/experiment/anything/run")
    assert resp.status_code == 503


def test_manifest_put_requires_token_under_slurm(monkeypatch):
    # PUT /api/experiment/{name}/manifest writes a caller-supplied manifest
    # into the workspace — same auth tier as the POST experiment verbs
    # (2026-07-21: the one-click server-draft sync route).
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    client = TestClient(app)
    resp = client.put("/api/experiment/anything/manifest",
                      json={"name": "anything", "status": "draft"})
    assert resp.status_code == 503
    # The read half of the identity check stays an open read.
    assert client.get("/api/experiment/anything/manifest").status_code == 404


def test_reader_verbs_require_token_under_slurm(monkeypatch):
    # /api/reader/score reads caller-named files and runs model compute;
    # /api/reader/fit writes under prompts/ + runs/ and runs model compute.
    # Both are privileged on a Slurm deployment; the read-only catalog
    # listing (GET /api/readers) stays open.
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    client = TestClient(app)
    assert client.post("/api/reader/score", json={}).status_code == 503
    assert client.post("/api/reader/fit", json={}).status_code == 503
    assert client.get("/api/readers").status_code == 200


def test_backfill_norms_requires_token_under_slurm(monkeypatch):
    # /api/vectors/backfill-norms reads caller-named artifact/corpus files,
    # runs model compute, and writes under runs/, so it is privileged on a
    # Slurm deployment; the read-only catalog listing (GET /api/vectors)
    # stays open.
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    client = TestClient(app)
    assert client.post("/api/vectors/backfill-norms", json={}).status_code == 503
    assert client.get("/api/vectors").status_code == 200


def test_reconcile_requires_token_under_slurm(monkeypatch):
    # /api/jobs/reconcile folds caller-named JSON into the durable store, so it
    # is privileged too.
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    client = TestClient(app)
    resp = client.post("/api/jobs/reconcile", json={"recordsDirectory": "records"})
    assert resp.status_code == 503


def test_browser_loopback_origin_on_wrong_port_refused():
    # A page served from another local port (a different user's process on a
    # shared node) is still cross-origin even though its host is loopback.
    client = TestClient(app)
    resp = client.get(
        "/api/state",
        headers={"origin": "http://localhost:9999", "host": "localhost:8080"},
    )
    assert resp.status_code == 403


def test_browser_allowed_proxy_origin_is_accepted(monkeypatch):
    # Open OnDemand / reverse-proxy sessions are not loopback origins in the
    # browser. They are accepted only when explicitly configured and same-host.
    monkeypatch.setenv("STEERLAB_ALLOWED_ORIGINS", "https://ood.example.edu")
    client = TestClient(app)
    resp = client.get(
        "/api/state",
        headers={"origin": "https://ood.example.edu", "host": "ood.example.edu"},
    )
    assert resp.status_code == 200


def test_browser_allowed_proxy_origin_must_match_host(monkeypatch):
    monkeypatch.setenv("STEERLAB_ALLOWED_ORIGINS", "https://ood.example.edu")
    client = TestClient(app)
    resp = client.get(
        "/api/state",
        headers={"origin": "https://ood.example.edu", "host": "localhost:8080"},
    )
    assert resp.status_code == 403


def test_browser_cross_site_fetch_without_origin_refused():
    client = TestClient(app)
    resp = client.get(
        "/api/state",
        headers={"sec-fetch-site": "cross-site", "host": "localhost:8080"},
    )
    assert resp.status_code == 403


def test_safe_path_resolver_blocks_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    resolver = SafePathResolver(ServerProfile.from_env())
    inside = tmp_path / "runs" / "x"
    inside.mkdir(parents=True)
    assert resolver.resolve_workspace("runs/x") == os.path.realpath(inside)
    with pytest.raises(Exception):
        resolver.resolve_workspace("../outside")


def test_durable_job_store_survives_manager_restart(tmp_path):
    store_path = tmp_path / "jobs.sqlite"
    mgr = JobManager(DurableJobStore(str(store_path)), capability_provider=lambda: {"test": True})
    job = mgr.submit("unit", lambda j: (j.log("hello"), {"answer": 42})[1])
    deadline = time.time() + 5
    while time.time() < deadline and mgr.get(job.id).status not in {"succeeded", "failed"}:
        time.sleep(0.02)
    assert mgr.get(job.id).status == "succeeded"

    mgr2 = JobManager(DurableJobStore(str(store_path)))
    restored = mgr2.get(job.id)
    assert restored is not None
    assert restored.result == {"answer": 42}
    assert "hello" in restored.all_logs()
    assert restored.capability_snapshot == {"test": True}


def test_reconcile_imports_child_job_record(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    (records / "child.json").write_text(
        '{"id":"abc","kind":"slurm-child","status":"succeeded",'
        '"executor":"slurm","executorJobID":"123","logs":["done"]}',
        encoding="utf-8",
    )
    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    assert mgr.reconcile(str(records)) == 1
    job = mgr.get("abc")
    assert job.executor == "slurm"
    assert job.executor_job_id == "123"
    assert job.all_logs() == ["done"]


def test_external_job_record_persists(tmp_path):
    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    job = mgr.record_external(
        "slurm-submit", status="submitted", executor="slurm",
        executor_job_id="456", requested_resources={"gres": "gpu:A100:1"},
        result={"bundle": {"scriptPath": "run.sbatch"}}, log="submitted")
    restored = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite"))).get(job.id)
    assert restored.status == "submitted"
    assert restored.executor == "slurm"
    assert restored.executor_job_id == "456"
    assert restored.requested_resources == {"gres": "gpu:A100:1"}
    assert restored.all_logs() == ["submitted"]


def test_orphan_local_jobs_failed_on_restart(tmp_path):
    store_path = str(tmp_path / "jobs.sqlite")
    mgr = JobManager(DurableJobStore(store_path), sweep_orphans=False)
    job = mgr.record_external("run", status="running", executor="local")
    # A fresh manager (server restart) has no live thread for that job.
    restored = JobManager(DurableJobStore(store_path)).get(job.id)
    assert restored.status == "failed"
    assert "orphaned" in (restored.error or "")


def test_slurm_cancel_calls_scheduler(tmp_path):
    calls = []

    class FakeSlurm:
        def cancel(self, slurm_id):
            # Executor contract since 2026-07-16: cancel reports whether
            # scancel itself succeeded — only True may stamp terminal.
            calls.append(slurm_id)
            return True

    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                     slurm_executor=FakeSlurm())
    job = mgr.record_external("slurm-submit", status="submitted", executor="slurm",
                              executor_job_id="9001")
    assert mgr.cancel(job.id) is True
    assert calls == ["9001"]
    assert mgr.get(job.id).status == "cancelled"


def test_cancel_running_job_two_phase_lifecycle(tmp_path):
    # Honest cancel (the live sweep bug): cancelling a RUNNING local job must
    # not pretend the work stopped. Status goes running → "cancelling"
    # (non-terminal, worker still executing) and only the worker's exit
    # stamps the terminal "cancelled".
    import threading

    started = threading.Event()
    release = threading.Event()
    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))

    def work(job):
        started.set()
        # Block until the test allows the worker to observe the flag —
        # exactly a generation loop that hasn't reached its next checkpoint.
        assert release.wait(timeout=10)
        assert job.cancelled
        return {}

    job = mgr.submit("sweep", work)
    assert started.wait(timeout=10)
    assert mgr.cancel(job.id) is True
    assert job.status == "cancelling"        # not terminal: work still running
    assert job.cancelled is True
    assert job.finished_at is None
    release.set()
    deadline = time.time() + 10
    while time.time() < deadline and job.status == "cancelling":
        time.sleep(0.01)
    assert job.status == "cancelled"
    assert job.finished_at is not None


def test_cancel_pending_job_is_terminal_and_work_never_runs(tmp_path, monkeypatch):
    # A job cancelled before the worker picks it up goes straight to the
    # terminal "cancelled" — honest because the runner's started-check
    # guarantees the work function never runs.
    from steerlab_server.api import jobs as jobs_mod

    runners = []

    class DeferredThread:  # captures the runner instead of starting it
        def __init__(self, target=None, name=None, daemon=None):
            runners.append(target)

        def start(self):
            pass

    monkeypatch.setattr(jobs_mod.threading, "Thread", DeferredThread)
    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    ran = []
    job = mgr.submit("sweep", lambda j: ran.append(1))
    assert job.status == "pending"
    assert mgr.cancel(job.id) is True
    assert job.status == "cancelled"
    assert job.finished_at is not None
    # The worker picks it up later: it must be a no-op acknowledgement.
    runners[0]()
    assert ran == []
    assert job.status == "cancelled"


def test_orphaned_cancelling_job_becomes_cancelled_on_restart(tmp_path):
    # A worker that dies (server restart) while a cancel was pending: the
    # work IS stopped and a cancel was requested, so the honest terminal is
    # "cancelled", not a forever-open "cancelling" (which would hold SSE
    # streams open for a thread that no longer exists).
    store_path = str(tmp_path / "jobs.sqlite")
    mgr = JobManager(DurableJobStore(store_path), sweep_orphans=False)
    job = mgr.record_external("sweep", status="cancelling", executor="local")
    restored = JobManager(DurableJobStore(store_path)).get(job.id)
    assert restored.status == "cancelled"
    assert restored.finished_at is not None


def test_poll_slurm_updates_status_and_reconciles(tmp_path):
    records = tmp_path / "records"
    records.mkdir()

    class FakeSlurm:
        def poll_state(self, slurm_id):
            return "succeeded"

    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                     slurm_executor=FakeSlurm())
    job = mgr.record_external("study-submit", status="submitted", executor="slurm",
                              executor_job_id="9002",
                              result={"recordsDirectory": str(records)})
    (records / f"{job.id}.json").write_text(
        '{"id":"' + job.id + '","status":"succeeded","result":{"runDirectory":"runs/x"}}',
        encoding="utf-8")
    assert mgr.poll_slurm() == 1
    updated = mgr.get(job.id)
    assert updated.status == "succeeded"
    assert (updated.result or {}).get("runDirectory") == "runs/x"


def test_map_slurm_state_covers_terminal_and_running():
    from steerlab_server.api.executors import map_slurm_state
    assert map_slurm_state("RUNNING") == "running"
    assert map_slurm_state("COMPLETED") == "succeeded"
    assert map_slurm_state("CANCELLED by 1234") == "cancelled"
    assert map_slurm_state("TIMEOUT") == "failed"
    assert map_slurm_state("PENDING") == "submitted"
    assert map_slurm_state("") is None


def test_local_study_submission_executes(tmp_path, monkeypatch):
    from steerlab_server.api import submissions
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    # Avoid real model work: stub the bundle packaging + the executed command.
    monkeypatch.setattr(submissions.bundles, "package_experiment",
                        lambda experiment, output_path=None, root=None: {"name": experiment})
    import sys as _sys
    monkeypatch.setattr(submissions, "_bundle_execute_command",
                        lambda *a, **k: [_sys.executable, "-c", "print('ran')"])
    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    submission = submissions.submit_study("demo", verb="run", jobs=mgr, executor="local")
    deadline = time.time() + 5
    while time.time() < deadline and mgr.get(submission.job_id).status not in {"succeeded", "failed"}:
        time.sleep(0.02)
    assert mgr.get(submission.job_id).status == "succeeded"


def test_local_study_dry_run_does_not_execute(tmp_path, monkeypatch):
    from steerlab_server.api import submissions
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setattr(submissions.bundles, "package_experiment",
                        lambda experiment, output_path=None, root=None: {"name": experiment})
    ran = []
    monkeypatch.setattr(submissions, "_bundle_execute_command",
                        lambda *a, **k: (ran.append(1), ["false"])[1])
    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    submission = submissions.submit_study("demo", verb="run", jobs=mgr,
                                          executor="local", dry_run=True)
    assert mgr.get(submission.job_id).status == "prepared"


def test_slurm_script_uses_concrete_gres_and_forwards_signal(tmp_path):
    resources = SlurmResources(
        job_name="steerlab-test",
        gres="A100",
        walltime="01:00:00",
        signal_seconds=900,
        signal_target="batch-forward",
        use_srun=True,
    )
    bundle = SlurmExecutor().create_bundle(
        str(tmp_path), ["python", "-m", "steerlab_server.cli", "--config", "x.json"],
        resources=resources,
    )
    script = open(bundle.script_path, encoding="utf-8").read()
    assert "#SBATCH --export=NONE" in script
    assert "#SBATCH --gres=gpu:A100:1" in script
    assert "#SBATCH --signal=B:USR1@900" in script
    assert "kill -USR1" in script
    assert "srun python" in script


def test_slurm_rejects_abstract_gpu():
    resources = SlurmResources(gres="gpu:1")
    with pytest.raises(ValueError):
        render_slurm_script(type("B", (), {
            "resources": resources,
            "stdout_path": "out",
            "stderr_path": "err",
            "bundle_dir": ".",
            "command": ["true"],
            "env": {},
        })())


def test_cluster_profile_validation_warnings(monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVER_PROFILE", "cluster")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    profile = ServerProfile.from_env()
    assert any("auth" in warning for warning in profile.warnings())
    assert capability_snapshot()["profile"]["profile"] == "cluster"


def test_profile_validate_reports_cluster_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVER_PROFILE", "cluster")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "token")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    report = validate_profile()
    assert report["ok"] is False
    names = {c["name"]: c["status"] for c in report["checks"]}
    assert names["auth"] == "fail"
    assert names["runRoot"] == "fail"


def test_profile_validate_fails_on_a_declared_required_header_with_no_value(
        tmp_path, monkeypatch):
    # WP5 Step 8 (audit c4): WHICH headers sbatch rejects a job without is site
    # data, not one institution's training deck applied to every cluster. A
    # site that DECLARES --partition/--mem required still gets the hard failure
    # this check has always given; a site that declares nothing gets an honest
    # warning instead of another site's rule.
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_SLURM_REQUIRED_HEADERS", "partition,mem,ntasks")
    monkeypatch.delenv("STEERLAB_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_MEMORY", raising=False)
    report = validate_profile()
    names = {c["name"]: c["status"] for c in report["checks"]}
    assert names["slurm:requiredHeaders"] == "ok"
    assert names["slurm:partition"] == "fail"
    assert names["slurm:mem"] == "fail"
    assert report["ok"] is False

    monkeypatch.setenv("STEERLAB_SLURM_PARTITION", "gpu_p")
    monkeypatch.setenv("STEERLAB_SLURM_MEMORY", "80G")
    report = validate_profile()
    names = {c["name"]: c["status"] for c in report["checks"]}
    assert names["slurm:partition"] == "ok"
    assert names["slurm:mem"] == "ok"


def test_profile_validate_warns_when_a_site_declares_no_required_headers(
        tmp_path, monkeypatch):
    """The other half of c4: silence is reported as silence. Unset partition /
    mem are warnings naming the profile field, never a failure imported from a
    cluster this site has nothing to do with."""
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_SLURM_REQUIRED_HEADERS", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_MEMORY", raising=False)
    report = validate_profile()
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["slurm:requiredHeaders"]["status"] == "warn"
    assert "requiredHeaders" in checks["slurm:requiredHeaders"]["message"]
    assert checks["slurm:partition"]["status"] == "warn"
    assert checks["slurm:mem"]["status"] == "warn"


def test_profile_validate_checks_configured_poll_binaries(tmp_path, monkeypatch):
    # The which-checks follow the site-configured poll commands, so a site
    # that only ships wrappers is validated against what will actually run.
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_SLURM_SACCT", "sacct-site")
    monkeypatch.setenv("STEERLAB_SLURM_SQUEUE", "sq")
    names = {c["name"] for c in validate_profile()["checks"]}
    assert "slurm:sacct-site" in names
    assert "slurm:sq" in names
    assert "slurm:sacct" not in names
    assert "slurm:squeue" not in names


def test_profile_validate_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    body = TestClient(app).get("/api/profile/validate").json()
    assert "checks" in body
    assert any(c["name"] == "workspaceRoot" for c in body["checks"])

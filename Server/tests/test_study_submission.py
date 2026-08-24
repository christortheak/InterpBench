import json
import os
import base64

import pytest

from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.api.submissions import submit_study
from steerlab_server.experiment import bundles, experiment_store as es


def _concept(root, name="fair"):
    d = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "positive.jsonl"), "w", encoding="utf-8").write('{"text":"fair"}\n')
    open(os.path.join(d, "negative.jsonl"), "w", encoding="utf-8").write('{"text":"unfair"}\n')


def _study(root):
    _concept(root)
    es.create("Submit Study", model_id="org/m", revision="abc", root=root)
    es.attach("submit-study", ["fair"], root=root)


def test_submit_study_slurm_dry_run_builds_bundle_and_job(tmp_path, monkeypatch):
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    _study(root)
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))

    submission = submit_study(
        "submit-study", verb="verify", jobs=jobs, executor="slurm",
        dry_run=True, resources={"gres": "A100", "walltime": "00:30:00"})

    assert submission.dry_run is True
    assert submission.slurm_job_id is None
    assert submission.slurm_bundle is not None
    assert os.path.exists(submission.run_bundle["bundlePath"])
    assert os.path.exists(submission.slurm_bundle["script_path"])
    assert "steerlab_server.cli" in submission.command
    assert "execute" in submission.command
    job = jobs.get(submission.job_id)
    assert job.status == "prepared"
    assert job.executor == "slurm"


def test_execute_run_bundle_verify_imports_and_runs(tmp_path):
    source = str(tmp_path / "source")
    target = str(tmp_path / "target")
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)
    result = bundles.execute_run_bundle(
        meta["bundlePath"], verb="verify", target_root=target,
        package_evidence_on_complete=False)

    assert result["experiment"] == "submit-study"
    assert result["violations"] == []
    assert os.path.exists(os.path.join(target, "experiments", "submit-study", "experiment.json"))


def test_submit_study_analyze_slurm_dry_run_is_supported(tmp_path, monkeypatch):
    # Regression (2026-08-06): remote submission refused analyze with
    # "unsupported study verb 'analyze'" although the verb has existed on
    # both CLIs since 2026-07-13 — the submit surface never learned it.
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    _study(root)
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))

    submission = submit_study(
        "submit-study", verb="analyze", jobs=jobs, executor="slurm",
        dry_run=True, resources={"walltime": "00:30:00"})

    assert submission.verb == "analyze"
    command = submission.command
    assert command[command.index("--verb") + 1] == "analyze"
    # Statistics-only: the preflight must not demand a GPU allocation —
    # no warn/fail from the model-shaped checks on a GPU-less request.
    report = submission.preflight
    by_id = {c["id"]: c for c in report["checks"]}
    for check_id in ("gpuRequest", "memoryFit", "walltime"):
        assert by_id[check_id]["status"] == "ok", by_id[check_id]
        assert "not applicable" in by_id[check_id]["message"]
    job = jobs.get(submission.job_id)
    assert job.status == "prepared"


def test_execute_run_bundle_analyze_dispatches_with_source(tmp_path, monkeypatch):
    """The bundle-execute dispatch (the path every remote submission runs)
    reaches tasks.analyze with the imported target root and the --source
    run — and does NOT bypass the epoch guard tasks.analyze enforces
    (no allow_unverified_epoch smuggled in).

    A RELATIVE --source arrives resolved against --target (ledger 2026-08-21):
    this dispatch used to hand the string through verbatim, and the rendered
    sbatch cd's into its own slurm directory before srun, so `runs/<dir>`
    resolved against the slurm directory and read nothing at all."""
    from steerlab_server.experiment import tasks

    source = str(tmp_path / "source")
    target = str(tmp_path / "target")
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)

    seen = {}

    def fake_analyze(name, root=None, source_run=None, **kwargs):
        seen["args"] = (name, root, source_run)
        seen["kwargs"] = kwargs
        return os.path.join(root, "runs", "fake-exp-submit-study-analyze")

    monkeypatch.setattr(tasks, "analyze", fake_analyze)
    result = bundles.execute_run_bundle(
        meta["bundlePath"], verb="analyze", target_root=target,
        source_path="runs/some-run", package_evidence_on_complete=False)

    assert seen["args"] == (
        "submit-study", os.path.realpath(target),
        os.path.join(os.path.realpath(target), "runs", "some-run"))
    assert not seen["kwargs"].get("allow_unverified_epoch")
    assert result["runDirectory"].endswith("fake-exp-submit-study-analyze")


def test_execute_run_bundle_writes_child_record(tmp_path):
    source = str(tmp_path / "source")
    target = str(tmp_path / "target")
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)
    record = tmp_path / "records" / "abc.json"
    bundles.execute_run_bundle(
        meta["bundlePath"], verb="verify", target_root=target,
        package_evidence_on_complete=False, record_path=str(record))
    data = json.loads(record.read_text(encoding="utf-8"))
    assert data["kind"] == "bundle-execute:verify"
    assert data["status"] == "succeeded"
    assert data["result"]["experiment"] == "submit-study"


def test_submit_study_endpoint_dry_run(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    _study(str(tmp_path / "source"))
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    resp = TestClient(app).post("/api/studies/submit", json={
        "experiment": "submit-study",
        "verb": "verify",
        "executor": "slurm",
        "dryRun": True,
        "resources": {"gres": "A100"},
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["experiment"] == "submit-study"
    assert body["dryRun"] is True
    assert body["slurmBundle"]["resources"]["gres"] == "A100"


def test_submit_study_endpoint_accepts_analyze(tmp_path, monkeypatch):
    # The exact 2026-08-06 failure: POST /api/studies/submit with verb
    # "analyze" returned 400 "unsupported study verb 'analyze'".
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    _study(str(tmp_path / "source"))
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    resp = TestClient(app).post("/api/studies/submit", json={
        "experiment": "submit-study",
        "verb": "analyze",
        "executor": "slurm",
        "dryRun": True,
    })
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["verb"] == "analyze"
    assert body["dryRun"] is True


def test_submit_bundle_endpoint_accepts_analyze(tmp_path, monkeypatch):
    # Twin of the server-resident route test on the bundle path — the one
    # the Mac app's Remote options verb picker actually submits through.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    source = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", source)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)
    app = FastAPI()
    app.include_router(build_router(ServiceState()))

    submit = TestClient(app).post("/api/studies/submit-bundle", json={
        "bundlePath": meta["bundlePath"],
        "verb": "analyze",
        "executor": "local",
        "dryRun": True,
    })
    assert submit.status_code == 200, submit.json()
    body = submit.json()
    assert body["verb"] == "analyze"
    assert body["dryRun"] is True


def test_upload_and_submit_bundle_endpoint_dry_run(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    source = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", source)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)
    blob = open(meta["bundlePath"], "rb").read()
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    client = TestClient(app)

    upload = client.post(
        "/api/bundles/upload",
        content=blob,
        headers={"x-steerlab-filename": "submit-study.run-bundle.tar.gz"},
    )
    assert upload.status_code == 200
    uploaded = upload.json()
    assert uploaded["executable"] is True
    assert uploaded["bundle"]["experiment"] == "submit-study"

    submit = client.post("/api/studies/submit-bundle", json={
        "bundlePath": uploaded["path"],
        "verb": "verify",
        "executor": "local",
        "dryRun": True,
    })
    assert submit.status_code == 200
    body = submit.json()
    assert body["experiment"] == "submit-study"
    assert body["dryRun"] is True
    assert body["runBundle"]["bundleSha256"] == uploaded["bundle"]["bundleSha256"]


def test_upload_bundle_rejects_non_bundle(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "server"))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    app = FastAPI()
    app.include_router(build_router(ServiceState()))

    resp = TestClient(app).post(
        "/api/bundles/upload",
        content=b"not a bundle",
        headers={"x-steerlab-filename": "fake.tar.gz"},
    )

    assert resp.status_code == 400
    assert "not a SteerLab bundle" in resp.json()["detail"]


def test_submit_bundle_endpoint_local_executes_verify(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    source = str(tmp_path / "source")
    target = str(tmp_path / "target")
    monkeypatch.setenv("STEERLAB_ROOT", source)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    client = TestClient(app)

    submit = client.post("/api/studies/submit-bundle", json={
        "bundlePath": meta["bundlePath"],
        "verb": "verify",
        "executor": "local",
        "targetRoot": target,
    })
    assert submit.status_code == 200
    job_id = submit.json()["jobId"]
    # Generous deadline (10s): the job finishes in milliseconds in isolation,
    # but under a full-suite run the executor thread can be starved — a 2s
    # budget flaked intermittently while the job still succeeded.
    for _ in range(500):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed"}:
            break
        import time
        time.sleep(0.02)
    assert job["status"] == "succeeded"
    assert os.path.exists(os.path.join(target, "experiments", "submit-study", "experiment.json"))


def test_dry_run_bundle_job_is_terminal_and_stream_closes(tmp_path, monkeypatch):
    """Regression: dry-run submissions record status "prepared". "prepared"
    must be in the TERMINAL set so ``GET /api/jobs/{id}/stream`` CLOSES —
    before the fix the SSE generator looped forever and the client spinner
    never stopped. Keeps the distinct status name (prepared != succeeded:
    the bundle was staged, nothing executed)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import threading

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.jobs import TERMINAL
    from steerlab_server.api.routes import ServiceState, build_router

    source = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", source)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    client = TestClient(app)

    submit = client.post("/api/studies/submit-bundle", json={
        "bundlePath": meta["bundlePath"], "verb": "verify",
        "executor": "local", "dryRun": True})
    assert submit.status_code == 200
    job_id = submit.json()["jobId"]

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "prepared"
    assert job["status"] in TERMINAL
    assert job["finishedAt"] is not None

    # The stream must CLOSE on the terminal "prepared" status. Read it on a
    # daemon thread with a hard deadline (and a line bound) so a regression
    # fails the test instead of hanging the suite.
    collected: dict = {}

    def read_stream():
        lines = []
        with client.stream("GET", f"/api/jobs/{job_id}/stream") as resp:
            for raw in resp.iter_lines():
                if raw.startswith("data:"):
                    lines.append(json.loads(raw[len("data:"):].strip())["line"])
                if len(lines) > 50:
                    break
        collected["lines"] = lines

    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()
    reader.join(timeout=15)
    assert not reader.is_alive(), \
        "/api/jobs/{id}/stream did not close for a prepared (dry-run) job"
    assert collected.get("lines"), "stream produced no SSE lines"
    assert collected["lines"][-1] == "[prepared]"


def test_variant_upload_accepts_server_only_model_and_reports_missing_assets(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "server"))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    resp = TestClient(app).post("/api/variants/upload", json={
        "name": "server-only",
        "baseModelID": "google/gemma-3-27b-it",
        "injections": [{
            "concept": "fair",
            "vectorArtifactID": "runs/missing/fair",
            "layer": 12,
            "alpha": 1.0,
        }],
        "promptMode": "chatAssistant",
        "qwenThinkingEnabled": False,
        "temperature": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["variant"]["baseModelID"] == "google/gemma-3-27b-it"
    assert body["missingArtifacts"][0]["kind"] == "vector"


def test_variant_upload_stages_vector_artifacts_and_rewrites_reference(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "server"))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    original_ref = "/client/runs/vector/fair"
    resp = TestClient(app).post("/api/variants/upload", json={
        "variant": {
            "name": "portable",
            "baseModelID": "google/gemma-3-27b-it",
            "injections": [{
                "concept": "fair",
                "vectorArtifactID": original_ref,
                "layer": 12,
                "alpha": 1.0,
            }],
            "promptMode": "chatAssistant",
            "qwenThinkingEnabled": False,
            "temperature": 0,
        },
        "artifacts": [
            {
                "kind": "vector",
                "reference": original_ref,
                "relativePath": "vectors/abc/fair.safetensors",
                "dataBase64": base64.b64encode(b"tensor").decode(),
            },
            {
                "kind": "vector",
                "reference": original_ref,
                "relativePath": "vectors/abc/fair.json",
                "dataBase64": base64.b64encode(b"{}").decode(),
            },
        ],
    })

    assert resp.status_code == 200
    body = resp.json()
    rewritten = body["variant"]["injections"][0]["vectorArtifactID"]
    assert rewritten != original_ref
    assert rewritten.endswith("/artifacts/vectors/abc/fair")
    assert body["missingArtifacts"] == []
    assert os.path.exists(rewritten + ".safetensors")
    assert os.path.exists(rewritten + ".json")


def test_evidence_download_serves_safe_run_artifact(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "server"))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    run = tmp_path / "runs" / "2026-run"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}", encoding="utf-8")
    meta = bundles.package_evidence(str(run))
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    resp = TestClient(app).get("/api/bundles/download", params={"path": meta["bundlePath"]})
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x1f\x8b")


def test_import_bundle_refuses_frozen_manifest_overwrite(tmp_path):
    # The circularity firewall: a bundle may not silently replace an existing
    # FROZEN manifest with different content, even with allow_overwrite=True.
    source = str(tmp_path / "source")
    target = str(tmp_path / "target")
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)

    # Plant a frozen manifest with DIFFERENT content at the target.
    frozen_dir = os.path.join(target, "experiments", "submit-study")
    os.makedirs(frozen_dir, exist_ok=True)
    with open(os.path.join(frozen_dir, "experiment.json"), "w", encoding="utf-8") as h:
        json.dump({"name": "submit-study", "status": "frozen", "note": "different"}, h)

    with pytest.raises(bundles.BundleError, match="frozen manifest"):
        bundles.import_bundle(meta["bundlePath"], target_root=target, allow_overwrite=True)


def test_submit_bundle_rejects_traversal_target_root(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    source = str(tmp_path / "server")
    monkeypatch.setenv("STEERLAB_ROOT", source)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    _study(source)
    meta = bundles.package_experiment("submit-study", root=source)

    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    resp = TestClient(app).post(
        "/api/studies/submit-bundle",
        json={"bundlePath": meta["bundlePath"], "verb": "verify",
              "executor": "local", "targetRoot": "../../../etc/steerlab-escape"})
    # A relative traversal targetRoot escapes the workspace root and is refused
    # on every profile (absolute local paths stay allowed on the dev profile,
    # matching the rest of the server).
    assert resp.status_code == 400


def test_upload_bundle_enforces_size_cap(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "server"))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    monkeypatch.setenv("STEERLAB_MAX_UPLOAD_BYTES", "16")
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    resp = TestClient(app).post(
        "/api/bundles/upload", content=b"x" * 64,
        headers={"X-SteerLab-Filename": "big.tar.gz",
                 "Content-Type": "application/octet-stream"})
    assert resp.status_code == 413


def test_artifact_download_serves_safetensors_and_refuses_others(tmp_path, monkeypatch):
    # The app localizes server-side vector artifacts (sidecar + tensors)
    # into the Mac workspace on picker selection — the workspace is the
    # source of truth; the server only caches. The route therefore serves
    # .safetensors under the runs root, and still refuses arbitrary files.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "server"))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    run = tmp_path / "runs" / "2026-vectors"
    run.mkdir(parents=True)
    (run / "fear.safetensors").write_bytes(b"tensor-bytes")
    (run / "fear.bin").write_bytes(b"nope")
    app = FastAPI()
    app.include_router(build_router(ServiceState()))
    client = TestClient(app)
    ok = client.get("/api/bundles/download",
                    params={"path": str(run / "fear.safetensors")})
    assert ok.status_code == 200
    assert ok.content == b"tensor-bytes"
    refused = client.get("/api/bundles/download",
                         params={"path": str(run / "fear.bin")})
    assert refused.status_code == 400

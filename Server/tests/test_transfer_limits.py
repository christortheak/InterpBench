"""Controller robustness to abandoned and bursty transfers (a live controller
on 2026-09-13 stopped answering ``/api/capabilities`` for half an hour after a
client vanished mid-way through hundreds of bundle downloads and one 12 GB
science export).

Pinned here:

1. ``/api/capabilities`` answers while every download slot is busy on slow
   storage, and the next download is refused with ``503`` + ``Retry-After``
   rather than queued.
2. A download whose client disconnects stops reading within one chunk and
   releases its slot.
3. A science export runs on the dedicated single export worker — exports
   serialize, none of them borrows a request-threadpool token.
4. Success bytes and ``FileResponse``-shaped headers are unchanged.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from steerlab_server.api import transfer_limits
from steerlab_server.api.transfer_limits import (CHUNK_SIZE, RETRY_AFTER_SECONDS,
                                                 TransferLimits, download_concurrency)


def _app(tmp_path, monkeypatch, *, max_downloads: int):
    from steerlab_server.api.routes import ServiceState, build_router
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "server"))
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "api-jobs.sqlite"))
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    state = ServiceState()
    state.transfers = TransferLimits(max_downloads=max_downloads)
    app = FastAPI()
    app.include_router(build_router(state))
    return app, state


def _artifact(tmp_path, name: str, chunks: int) -> str:
    run = tmp_path / "runs" / "2026-transfer"
    run.mkdir(parents=True, exist_ok=True)
    path = run / name
    path.write_bytes(bytes([7]) * (CHUNK_SIZE * chunks))
    return str(path)


# --------------------------------------------------------------------------
# 1. Saturation: capabilities answers, the overflow download is refused
# --------------------------------------------------------------------------

def test_capabilities_answers_while_every_download_slot_is_busy(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch, max_downloads=2)
    path = _artifact(tmp_path, "slow.evidence-bundle.tar.gz", chunks=3)
    release = threading.Event()
    started = threading.Semaphore(0)
    original = state.transfers.run_download_io

    async def slow_reads(func, *args):
        # Every chunk read blocks on the executor thread until released — the
        # shape of a shared filesystem that has stopped keeping up.
        if getattr(func, "__name__", "") == "read":
            def gated(*inner):
                started.release()
                assert release.wait(30), "test gate never released"
                return func(*inner)
            return await original(gated, *args)
        return await original(func, *args)

    monkeypatch.setattr(state.transfers, "run_download_io", slow_reads)

    results: dict[int, tuple[int, int]] = {}

    def fetch(index: int) -> None:
        response = TestClient(app).get("/api/bundles/download", params={"path": path})
        results[index] = (response.status_code, len(response.content))

    workers = [threading.Thread(target=fetch, args=(i,), daemon=True) for i in range(2)]
    for worker in workers:
        worker.start()
    # Both slots are held on the storage gate before anything else happens.
    for _ in range(2):
        assert started.acquire(timeout=30)
    assert state.transfers.snapshot()["activeDownloads"] == 2

    # The controller still answers its cheapest question — with a deadline,
    # because a hung answer is exactly the incident.
    began = time.monotonic()
    caps = TestClient(app).get("/api/capabilities")
    assert caps.status_code == 200
    assert time.monotonic() - began < 10
    assert "availableJobTypes" in caps.json()

    # The overflow download is refused NOW with an advisory pause — never
    # queued behind the stalled two.
    overflow = TestClient(app).get("/api/bundles/download", params={"path": path})
    assert overflow.status_code == 503
    assert overflow.headers["retry-after"] == str(RETRY_AFTER_SECONDS)
    detail = overflow.json()["detail"]
    assert detail["code"] == "downloadCapacity"
    assert "repairAction" in detail
    assert state.transfers.snapshot()["refusedDownloads"] == 1

    release.set()
    for worker in workers:
        worker.join(timeout=60)
    assert results == {0: (200, CHUNK_SIZE * 3), 1: (200, CHUNK_SIZE * 3)}
    snapshot = state.transfers.snapshot()
    assert snapshot["activeDownloads"] == 0
    assert snapshot["peakDownloads"] == 2


def test_slot_released_after_a_refused_path(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch, max_downloads=1)
    run = tmp_path / "runs" / "2026-transfer"
    run.mkdir(parents=True, exist_ok=True)
    (run / "weights.bin").write_bytes(b"nope")
    client = TestClient(app)
    assert client.get("/api/bundles/download",
                      params={"path": str(run / "weights.bin")}).status_code == 400
    assert client.get("/api/bundles/download",
                      params={"path": str(run / "missing.json")}).status_code == 404
    # Neither refusal leaked its slot: the pool is whole again.
    assert state.transfers.snapshot()["activeDownloads"] == 0
    (run / "ok.json").write_text("{}", encoding="utf-8")
    ok = client.get("/api/bundles/download", params={"path": str(run / "ok.json")})
    assert ok.status_code == 200 and ok.content == b"{}"


def test_transfer_policy_gate_still_runs_first(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch, max_downloads=1)
    path = _artifact(tmp_path, "gated.evidence-bundle.tar.gz", chunks=1)
    monkeypatch.setenv("STEERLAB_TRANSFER_METHOD", "rsync")
    refused = TestClient(app).get("/api/bundles/download", params={"path": path})
    assert refused.status_code == 403
    assert refused.json()["detail"]["code"] == "external_transfer_required"
    # The gate answered before any slot was taken or refused.
    snapshot = state.transfers.snapshot()
    assert snapshot["activeDownloads"] == 0 and snapshot["refusedDownloads"] == 0


# --------------------------------------------------------------------------
# 2. A vanished client releases its slot within one chunk
# --------------------------------------------------------------------------

def test_disconnected_client_stops_the_transfer_and_frees_the_slot(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch, max_downloads=1)
    total_chunks = 6
    path = _artifact(tmp_path, "abandoned.evidence-bundle.tar.gz", chunks=total_chunks)
    reads: list[int] = []
    original = state.transfers.run_download_io

    async def counting(func, *args):
        result = await original(func, *args)
        if getattr(func, "__name__", "") == "read":
            reads.append(len(result))
        return result

    monkeypatch.setattr(state.transfers, "run_download_io", counting)
    query = ("path=" + path).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": "/api/bundles/download", "raw_path": b"/api/bundles/download",
        "query_string": query, "headers": [(b"host", b"127.0.0.1")],
        "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8080),
    }
    body_chunks: list[int] = []

    async def run() -> None:
        gone = asyncio.Event()

        async def receive():
            # Like uvicorn: nothing to say until the peer goes away, then
            # `http.disconnect` for every later ask.
            await gone.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                body_chunks.append(len(message["body"]))
                # The client vanishes after receiving the first chunk.
                gone.set()

        await app(scope, receive, send)

    asyncio.run(run())
    # One chunk went out, the reader noticed the disconnect on the very next
    # check, and the remaining chunks were never read from storage.
    assert len(body_chunks) == 1
    assert len(reads) <= 2 < total_chunks
    snapshot = state.transfers.snapshot()
    assert snapshot["activeDownloads"] == 0
    assert snapshot["peakDownloads"] == 1


# --------------------------------------------------------------------------
# 3. Exports: dedicated single worker, serialized, no request tokens
# --------------------------------------------------------------------------

def test_export_runs_on_the_export_worker_and_serializes(monkeypatch):
    from steerlab_server.api import diagnostic_transport as transport
    from steerlab_server.api.diagnostic_transport_routes import build_router

    running = 0
    peak = 0
    threads: list[str] = []
    lock = threading.Lock()

    def fake_export(job_id, jobs, profile):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
            threads.append(threading.current_thread().name)
        time.sleep(0.2)
        with lock:
            running -= 1
        return {"bundlePath": f"/exports/{job_id}.tar.gz", "bundleSha256": "0" * 64,
                "context": {"jobID": job_id}, "entries": [], "bytes": 0}

    monkeypatch.setattr(transport, "export", fake_export)
    state = SimpleNamespace(jobs=object())
    app = FastAPI()
    app.include_router(build_router(state))
    results: list[int] = []

    def request(job: str) -> None:
        results.append(TestClient(app).post(f"/api/science/jobs/{job}/export").status_code)

    workers = [threading.Thread(target=request, args=(f"job-{i}",), daemon=True) for i in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    assert results == [200, 200]
    # Both ran on the dedicated export worker (never an anyio worker thread
    # from the request pool), and never at the same time.
    assert threads and all(name.startswith("steerlab-export") for name in threads)
    assert peak == 1
    snapshot = state.transfers.snapshot()
    assert snapshot["activeExports"] == 0 and snapshot["peakExports"] == 2


def test_export_refusals_keep_their_shape(monkeypatch):
    from steerlab_server.api import diagnostic_transport as transport
    from steerlab_server.api.diagnostic_transport_routes import build_router
    from steerlab_server.experiment import diagnostic_archives as archives

    def refusing(job_id, jobs, profile):
        raise archives.Refusal("Export archive changed; retain originals and inspect the export store.")

    monkeypatch.setattr(transport, "export", refusing)
    app = FastAPI()
    app.include_router(build_router(SimpleNamespace(jobs=object())))
    response = TestClient(app).post("/api/science/jobs/example/export")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "diagnosticTransportRefused"
    assert detail["reason"].startswith("Export archive changed")
    assert detail["repairAction"]
    # No job subsystem → the same refusal the sync handler gave.
    bare = FastAPI()
    bare.include_router(build_router(SimpleNamespace(jobs=None)))
    assert TestClient(bare).post("/api/science/jobs/example/export").status_code == 409


# --------------------------------------------------------------------------
# 4. Unchanged success shape
# --------------------------------------------------------------------------

def test_download_headers_and_bytes_match_a_file_response(tmp_path, monkeypatch):
    app, state = _app(tmp_path, monkeypatch, max_downloads=2)
    path = _artifact(tmp_path, "shape.evidence-bundle.tar.gz", chunks=2)
    with open(path, "r+b") as handle:
        handle.write(b"\x1f\x8b")
    response = TestClient(app).get("/api/bundles/download", params={"path": path})
    assert response.status_code == 200
    assert response.content == open(path, "rb").read()
    assert response.headers["content-length"] == str(CHUNK_SIZE * 2)
    assert response.headers["content-disposition"] == \
        'attachment; filename="shape.evidence-bundle.tar.gz"'
    from fastapi.responses import FileResponse
    stock = FileResponse(path, filename=os.path.basename(path))
    assert response.headers["content-type"] == stock.media_type
    assert response.headers["accept-ranges"] == "bytes"
    assert "etag" in response.headers and "last-modified" in response.headers
    # The route is GET-only, as it always was.
    assert TestClient(app).head("/api/bundles/download", params={"path": path}).status_code == 405
    assert state.transfers.snapshot()["activeDownloads"] == 0


def test_download_concurrency_reads_the_environment(monkeypatch):
    monkeypatch.delenv("STEERLAB_DOWNLOAD_CONCURRENCY", raising=False)
    assert download_concurrency() == transfer_limits.DEFAULT_MAX_DOWNLOADS
    monkeypatch.setenv("STEERLAB_DOWNLOAD_CONCURRENCY", "3")
    assert download_concurrency() == 3
    monkeypatch.setenv("STEERLAB_DOWNLOAD_CONCURRENCY", "0")
    assert download_concurrency() == 1
    monkeypatch.setenv("STEERLAB_DOWNLOAD_CONCURRENCY", "many")
    assert download_concurrency() == transfer_limits.DEFAULT_MAX_DOWNLOADS

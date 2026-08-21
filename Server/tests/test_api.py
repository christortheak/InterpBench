"""API wiring tests that do not require a GPU or a loaded model."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # FastAPI TestClient transport

from fastapi.testclient import TestClient

from steerlab_server.api.app import app

client = TestClient(app)


def test_healthz():
    assert client.get("/healthz").json() == {"ok": True}


def test_info_describes_service():
    body = client.get("/api/info").json()
    assert body["service"] == "steerlab-server"
    # Device list is present and never offers cuda on a non-CUDA build.
    assert "cpu" in body["devices"]
    import torch
    if not torch.cuda.is_available():
        assert "cuda" not in body["devices"]


def test_info_carries_the_build_identity(monkeypatch):
    """WP6: which ROOT a server serves was discoverable and which CODE served
    it was not — so a client could not tell a stale deployment from a current
    one, and `site qualify`'s buildIdentity check answers that only for
    whoever is logged into the node. `engineVersion` is the identical string
    every run config and frozen manifest stamps."""
    from steerlab_server import build_identity

    body = client.get("/api/info").json()
    assert body["engineVersion"] == build_identity.engine_version()
    assert body["engineVersion"].startswith("steerlab-server ")


def test_info_exposes_source_checkout_flag(tmp_path, monkeypatch):
    """/api/info reports whether the artifact root looks like the SteerLab
    source checkout (same heuristic as the serve-time warning), so the Mac
    app can badge an unpaired/misrooted server instead of the user finding
    out via a run refusal."""
    import os
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    body = client.get("/api/info").json()
    assert body["root"] == os.path.realpath(str(tmp_path))
    assert body["rootLooksLikeSourceCheckout"] is False
    (tmp_path / "Package.swift").write_text("x", encoding="utf-8")
    assert client.get("/api/info").json()["rootLooksLikeSourceCheckout"] is True


def test_state_with_no_model_loaded():
    body = client.get("/api/state").json()
    assert body["loadedModel"] is None
    assert body["numLayers"] is None
    assert body["jobs"] == []


def test_state_and_snapshots_expose_dtype_after_load(monkeypatch):
    """The loaded-model snapshot carries the ACTUAL dtype the model runs in
    (stamped at load) — how a user tells a bf16 model from a slow fp32
    fallback — both top-level and per registry slot (shared with /api/info)."""
    from types import SimpleNamespace
    from fastapi import FastAPI
    from steerlab_server.api import model_registry
    from steerlab_server.api.routes import ServiceState, build_router

    fake = SimpleNamespace(model_id="org/m", revision="r1", device="mps",
                           num_layers=2, hidden_size=8, context_window=128,
                           dtype="bfloat16")
    monkeypatch.setattr(
        model_registry.model_loader, "load",
        lambda model_id, revision=None, dtype="auto", device=None: fake)
    local_state = ServiceState()
    local_app = FastAPI()
    local_app.include_router(build_router(local_state))
    local_client = TestClient(local_app)

    local_state.load("org/m", None, "auto", None)
    body = local_client.get("/api/state").json()
    assert body["dtype"] == "bfloat16"
    slot = next(s for s in body["loadedModels"] if s["modelID"] == "org/m")
    assert slot["dtype"] == "bfloat16"
    info = local_client.get("/api/info").json()
    assert info["loadedModels"][0]["dtype"] == "bfloat16"


def test_generate_without_model_is_conflict():
    resp = client.post("/api/generate", json={"text": "hello"})
    assert resp.status_code == 409


def test_generate_injection_validation_requires_model_first():
    # Even a malformed injection is gated behind "load a model first" (409),
    # since the model is required before injections are resolved.
    resp = client.post("/api/generate", json={
        "text": "hi", "injections": [{"layer": 0, "alpha": 1.0}]})
    assert resp.status_code == 409


def test_jobs_empty_initially():
    assert client.get("/api/jobs").json() == {"jobs": []}


def test_unknown_experiment_verb_rejected():
    resp = client.post("/api/experiment/demo/frobnicate")
    assert resp.status_code == 400


def test_experiment_manifest_route_serves_raw_document(tmp_path, monkeypatch):
    # GET /api/experiment/{name}/manifest returns the experiment.json
    # VERBATIM (unknown keys included) — the Swift app's remote-freeze
    # identity check compares this body against the manifest it displays,
    # so summarizing (like experiment_detail) would defeat the comparison.
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    import json as _json
    directory = tmp_path / "experiments" / "demo"
    directory.mkdir(parents=True)
    document = {"name": "demo", "modelID": "org/m", "status": "draft",
                "conditions": [{"name": "baseline", "slots": []}],
                "unknownFutureKey": {"nested": [1, 2, 3]}}
    (directory / "experiment.json").write_text(
        _json.dumps(document), encoding="utf-8")

    resp = client.get("/api/experiment/demo/manifest")
    assert resp.status_code == 200
    assert resp.json() == document  # verbatim, unknown keys preserved

    missing = client.get("/api/experiment/ghost/manifest")
    assert missing.status_code == 404


def test_manifest_put_replaces_draft_and_reports_hash(tmp_path, monkeypatch):
    # One-click server-draft sync (2026-07-21 incident, part 3): the app
    # pushes the manifest ON SCREEN as the server's draft copy and gets the
    # stored status + canonical body hash back.
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    import hashlib as _hashlib
    import json as _json
    directory = tmp_path / "experiments" / "demo"
    directory.mkdir(parents=True)
    (directory / "experiment.json").write_text(
        _json.dumps({"name": "demo", "status": "draft", "modelID": "org/old"}),
        encoding="utf-8")

    pushed = {"name": "demo", "status": "draft", "modelID": "org/new",
              "conditions": [{"name": "baseline"}], "unknownFutureKey": True}
    resp = client.put("/api/experiment/demo/manifest", json=pushed)
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "demo" and body["status"] == "draft"
    expected = _hashlib.sha256(
        _json.dumps(pushed, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")).hexdigest()
    assert body["canonicalBodyHash"] == expected
    # The stored copy IS the pushed document (unknown keys preserved) —
    # what GET serves is what the identity re-check compares.
    assert client.get("/api/experiment/demo/manifest").json() == pushed

    # A missing server copy is created (first sync to an unpaired server).
    fresh = {"name": "fresh", "status": "draft", "modelID": "org/m"}
    created = client.put("/api/experiment/fresh/manifest", json=fresh)
    assert created.status_code == 200
    assert client.get("/api/experiment/fresh/manifest").json() == fresh


def test_manifest_put_refuses_frozen_copy_and_non_draft_push(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    import json as _json
    directory = tmp_path / "experiments" / "iced"
    directory.mkdir(parents=True)
    (directory / "experiment.json").write_text(
        _json.dumps({"name": "iced", "status": "frozen", "modelID": "org/m"}),
        encoding="utf-8")

    # Frozen server copy: refuse in the bundle-import wording family.
    resp = client.put("/api/experiment/iced/manifest",
                      json={"name": "iced", "status": "draft",
                            "modelID": "org/m2"})
    assert resp.status_code == 400
    assert "refusing to overwrite frozen manifest" in resp.json()["detail"]
    assert "freeze firewall" in resp.json()["detail"]

    # Non-draft incoming document: frozen status is stamped by the freeze
    # authority, never installed by upload.
    other = tmp_path / "experiments" / "pushy"
    other.mkdir(parents=True)
    (other / "experiment.json").write_text(
        _json.dumps({"name": "pushy", "status": "draft", "modelID": "org/m"}),
        encoding="utf-8")
    resp = client.put("/api/experiment/pushy/manifest",
                      json={"name": "pushy", "status": "frozen",
                            "modelID": "org/m"})
    assert resp.status_code == 400
    assert "only a DRAFT manifest" in resp.json()["detail"]

    # Name mismatch between route and body: ambiguous, refused.
    resp = client.put("/api/experiment/pushy/manifest",
                      json={"name": "other", "status": "draft"})
    assert resp.status_code == 400
    assert "ambiguous" in resp.json()["detail"]


def test_manifest_put_merges_server_auto_pins(tmp_path, monkeypatch):
    # Merge semantics (2026-08-06 replication run): a push whose
    # document OMITS a server-side auto-pin must not strip it — omission
    # means "the Mac never saw it", not "remove it". Preserved pins are
    # merged into the stored copy and NAMED in the response so the app can
    # adopt them; an explicit null (key present) still clears.
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    import json as _json
    directory = tmp_path / "experiments" / "demo"
    directory.mkdir(parents=True)
    projected = {"name": "fear-recommended",
                 "selection": {"sweepRun": "20260806T01-exp-demo-sweep",
                               "winningCell": {"layer": 2, "alpha": 4.0}}}
    (directory / "experiment.json").write_text(
        _json.dumps({"name": "demo", "status": "draft", "modelID": "org/m",
                     "modelRevision": "005ad3404e59",
                     "capabilityBatteryFile": "prompts/battery.jsonl",
                     "capabilityBatteryHash": "b" * 64,
                     "conditions": [{"name": "baseline"}, projected]}),
        encoding="utf-8")

    pushed = {"name": "demo", "status": "draft", "modelID": "org/m",
              "conditions": [{"name": "baseline"}]}
    resp = client.put("/api/experiment/demo/manifest", json=pushed)
    assert resp.status_code == 200
    preserved = resp.json()["preserved"]
    assert preserved["modelRevision"] == "005ad3404e59"
    assert preserved["conditions"] == ["fear-recommended"]
    assert preserved["capabilityBattery"]["file"] == "prompts/battery.jsonl"
    stored = client.get("/api/experiment/demo/manifest").json()
    assert stored["modelRevision"] == "005ad3404e59"
    assert stored["capabilityBatteryHash"] == "b" * 64
    assert projected in stored["conditions"]

    # Explicit null (key present) is a deliberate unpin — honored, and
    # nothing is reported preserved for it. A same-named incoming
    # condition wins over the server's projection (workspace authority
    # over content it has seen).
    cleared = {"name": "demo", "status": "draft", "modelID": "org/m",
               "modelRevision": None,
               "conditions": [{"name": "baseline"},
                              {"name": "fear-recommended",
                               "selection": {"sweepRun": "newer",
                                             "winningCell": {"layer": 3,
                                                             "alpha": 2.0}}}]}
    resp = client.put("/api/experiment/demo/manifest", json=cleared)
    assert resp.status_code == 200
    body = resp.json()
    assert "modelRevision" not in body.get("preserved", {})
    assert "conditions" not in body.get("preserved", {})
    stored = client.get("/api/experiment/demo/manifest").json()
    assert stored["modelRevision"] is None
    (kept,) = [c for c in stored["conditions"]
               if c["name"] == "fear-recommended"]
    assert kept["selection"]["sweepRun"] == "newer"


def test_authoring_create_and_freeze(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    # A freezable study needs a concept (verify rejects empty studies).
    cdir = tmp_path / "prompts" / "concepts" / "joy"
    cdir.mkdir(parents=True)
    (cdir / "positive.jsonl").write_text('{"text": "yay"}\n', encoding="utf-8")
    (cdir / "negative.jsonl").write_text('{"text": "meh"}\n', encoding="utf-8")

    r = client.post("/api/authoring/create", json={"name": "Web Study", "modelID": "org/m",
                                                   "revision": "abc"})
    assert r.status_code == 200 and r.json()["name"] == "web-study"
    client.post("/api/authoring/web-study/attach", json={"concepts": ["joy"]})
    # force-freeze (skips the validate-evidence gate; verify still runs)
    f = client.post("/api/authoring/web-study/freeze", json={"force": True})
    assert f.status_code == 200 and f.json()["status"] == "frozen"
    assert f.json()["frozenBy"] == "server"
    # frozen is read-only
    p = client.post("/api/authoring/web-study/protocol", json={"temperature": 0.9})
    assert p.status_code == 400


def test_authoring_create_requires_fields():
    assert client.post("/api/authoring/create", json={"name": "x"}).status_code == 400


def test_index_serves_web_ui():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SteerLab" in resp.text and "cluster workbench" in resp.text


def test_catalog_routes_against_temp_tree(tmp_path, monkeypatch):
    # Point the runtime root at a temp tree and exercise the read endpoints.
    import json, os
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    os.makedirs(tmp_path / "prompts" / "concepts" / "joy")
    (tmp_path / "prompts" / "concepts" / "joy" / "positive.jsonl").write_text(
        '{"text": "yay"}\n', encoding="utf-8")
    (tmp_path / "prompts" / "concepts" / "joy" / "negative.jsonl").write_text(
        '{"text": "meh"}\n', encoding="utf-8")
    assert client.get("/api/vectors").json() == {"vectors": []}
    concepts = client.get("/api/concepts").json()["concepts"]
    assert any(c["name"] == "joy" for c in concepts)
    assert client.get("/api/concept/joy").json()["positive"] == ["yay"]


def test_geometry_cosine_matrix(tmp_path):
    from steerlab_server.steering.vector_store import ConceptVectors, SteeringVectorSidecar, save
    for name, vec in [("a", [[1.0, 0.0], [1.0, 0.0]]), ("b", [[0.0, 1.0], [0.0, 1.0]])]:
        v = ConceptVectors(per_layer=vec)
        save(v, SteeringVectorSidecar.make(model_id="m", concept=name,
             stimulus_set_hash="h", vectors=v), str(tmp_path), name)
    resp = client.post("/api/geometry", json={"layer": 0, "vectors": [
        {"vectorPath": str(tmp_path), "name": "a", "label": "a"},
        {"vectorPath": str(tmp_path), "name": "b", "label": "b"}]})
    body = resp.json()
    assert body["labels"] == ["a", "b"]
    assert body["matrix"][0][0] == 1.0          # self-cosine
    assert abs(body["matrix"][0][1]) < 1e-6      # orthogonal a·b
    assert body["skipped"] == []                # nothing silently dropped


def test_geometry_accepts_catalog_id_form(tmp_path):
    """The catalog id / inline-spec address form — ``<runDirectory>/<name>``
    in one string, exactly what GET /api/vectors returns as ``id`` and what
    variant injections reference — must load, not silently skip (the Swift
    Geometry panel's server path used to send this form and got a 0x0
    matrix back)."""
    import os
    from steerlab_server.steering.vector_store import ConceptVectors, SteeringVectorSidecar, save
    for name, vec in [("a", [[1.0, 0.0]]), ("b", [[0.0, 1.0]])]:
        v = ConceptVectors(per_layer=vec)
        save(v, SteeringVectorSidecar.make(model_id="m", concept=name,
             stimulus_set_hash="h", vectors=v), str(tmp_path), name)
    resp = client.post("/api/geometry", json={"layer": 0, "vectors": [
        {"vectorPath": os.path.join(str(tmp_path), "a"), "name": "a", "label": "a"},
        {"vectorPath": os.path.join(str(tmp_path), "b"), "name": "b", "label": "b"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["labels"] == ["a", "b"]
    assert body["skipped"] == []
    assert body["matrix"][0][0] == 1.0


def test_geometry_reports_partial_skips_and_400s_when_all_skip(tmp_path):
    """Silent-skip contract: unloadable entries surface in ``skipped`` with a
    reason; when EVERY requested vector fails to load the route answers 400
    naming the entries instead of a 0x0 'success'."""
    from steerlab_server.steering.vector_store import ConceptVectors, SteeringVectorSidecar, save
    v = ConceptVectors(per_layer=[[1.0, 0.0]])
    save(v, SteeringVectorSidecar.make(model_id="m", concept="good",
         stimulus_set_hash="h", vectors=v), str(tmp_path), "good")

    # Mixed good/bad: matrix over the good one, the bad one reported.
    resp = client.post("/api/geometry", json={"layer": 0, "vectors": [
        {"vectorPath": str(tmp_path), "name": "good", "label": "good"},
        {"vectorPath": str(tmp_path / "no-such-dir"), "name": "ghost",
         "label": "ghost"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["labels"] == ["good"]
    assert [s["name"] for s in body["skipped"]] == ["ghost"]
    assert body["skipped"][0]["reason"]

    # All bad: a 400 that names the unresolvable entries, never 0x0.
    resp = client.post("/api/geometry", json={"layer": 0, "vectors": [
        {"vectorPath": str(tmp_path / "nowhere"), "name": "x", "label": "x"}]})
    assert resp.status_code == 400
    assert "x" in resp.json()["detail"]

    # An empty request is still a legitimate empty matrix, not an error.
    resp = client.post("/api/geometry", json={"layer": 0, "vectors": []})
    assert resp.status_code == 200
    assert resp.json()["labels"] == []


def test_run_file_path_traversal_blocked():
    assert client.get("/api/runs/x/file", params={"name": "../secret"}).status_code == 400
    assert client.get("/api/runs/..%2f../file", params={"name": "x"}).status_code in (400, 404)
    # The head param changes only how MUCH is returned, never containment.
    assert client.get("/api/runs/x/file",
                      params={"name": "../secret", "head": 16}).status_code == 400


def test_run_file_head_is_bounded(tmp_path, monkeypatch):
    """GET /api/runs/{id}/file?head=N returns exactly the first N bytes with a
    correct content-length; N above the 8 MiB server cap is clamped; N <= 0 is
    refused; the no-param case still returns the whole file."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    line = b'{"condition": "baseline", "output": "hello"}\n'
    body = line * 200_000  # ~8.8 MB, above the server head cap
    (run_dir / "generations.jsonl").write_bytes(body)

    head = client.get("/api/runs/r1/file",
                      params={"name": "generations.jsonl", "head": 100})
    assert head.status_code == 200
    assert head.content == body[:100]
    assert head.headers["content-length"] == "100"
    # A bounded read below the file size is truncation, and says so.
    assert head.headers["x-steerlab-file-size"] == str(len(body))
    assert head.headers["x-steerlab-truncated"] == "true"

    # Above the cap: clamped to 8 MiB, not the full file — with honest
    # truncation metadata so the client never mistakes the clamp for EOF.
    big = client.get("/api/runs/r1/file",
                     params={"name": "generations.jsonl", "head": 20_000_000})
    assert big.content == body[:8_388_608]
    assert big.headers["x-steerlab-file-size"] == str(len(body))
    assert big.headers["x-steerlab-truncated"] == "true"

    # head beyond a small file's size returns the whole (small) file.
    (run_dir / "report.json").write_bytes(b'{"passed": true}')
    small = client.get("/api/runs/r1/file",
                       params={"name": "report.json", "head": 4096})
    assert small.content == b'{"passed": true}'
    assert small.headers["x-steerlab-file-size"] == "16"
    assert small.headers["x-steerlab-truncated"] == "false"

    for bad in (0, -5):
        assert client.get("/api/runs/r1/file",
                          params={"name": "report.json", "head": bad}).status_code == 400

    whole = client.get("/api/runs/r1/file", params={"name": "report.json"})
    assert whole.content == b'{"passed": true}'


def test_run_file_head_serves_multi_megabyte_report_complete(tmp_path, monkeypatch):
    """A report.json ABOVE the old 1 MiB server cap arrives COMPLETE through
    the real route when the requested head covers it (the Swift client asks
    with a 4 MiB report cap), with truncated=false metadata. This is the
    regression for run-level metrics disappearing from big remote reports."""
    import json as _json
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    run_dir = tmp_path / "runs" / "r-big-report"
    run_dir.mkdir(parents=True)
    report = {"experiment": "big", "conditions": {
        "baseline": {"generations": 3, "choiceReadouts": 3, "choiceRate": 0.5},
    }, "padding": "x" * 2_097_152}  # > 2 MiB of valid JSON
    body = _json.dumps(report).encode()
    assert len(body) > 1_048_576
    (run_dir / "report.json").write_bytes(body)

    # The Swift client's report request: head = 4 MiB.
    resp = client.get("/api/runs/r-big-report/file",
                      params={"name": "report.json", "head": 4_194_304})
    assert resp.status_code == 200
    assert resp.content == body  # complete, byte-identical
    assert resp.headers["x-steerlab-file-size"] == str(len(body))
    assert resp.headers["x-steerlab-truncated"] == "false"
    # And the bytes still parse as the original report.
    parsed = _json.loads(resp.content)
    assert parsed["conditions"]["baseline"]["choiceReadouts"] == 3


def test_runs_listing_carries_stamps_and_file_entries(tmp_path, monkeypatch):
    """GET /api/runs serves the config.json stamps + name/size fileEntries the
    remote Results browser decodes, keeping ``files`` as the legacy string
    list."""
    import json as _json
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    run_dir = tmp_path / "runs" / "20260707T000000000-exp-demo-run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(_json.dumps({
        "schemaVersion": 1, "runType": "run", "createdAt": "2026-07-07T00:00:00Z",
        "substrate": "python-hf-transformers", "appVersion": "steerlab-server 1.0",
        "modelID": "org/m", "revision": "r9", "experiment": "demo",
        "experimentHash": None}), encoding="utf-8")
    (run_dir / "metrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    (run,) = client.get("/api/runs").json()["runs"]
    assert run["runType"] == "run" and run["experiment"] == "demo"
    assert run["modelID"] == "org/m" and run["substrate"] == "python-hf-transformers"
    assert run["files"] == ["config.json", "metrics.csv"]
    sizes = {e["name"]: e["size"] for e in run["fileEntries"]}
    assert sizes["metrics.csv"] == len("a,b\n1,2\n")

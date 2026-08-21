import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.api import variant_chat
from steerlab_server.experiment import model_variant


def _variant_file(tmp_path, *, model_id="org/model"):
    path = tmp_path / "variant.json"
    payload = model_variant.ModelVariant(
        name="v", base_model_id=model_id, base_revision="abc",
        temperature=0.2, prompt_mode="rawCompletion",
        system_prompt="variant system").to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_render_chat_text_from_messages():
    text = variant_chat.render_chat_text(None, [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "continue"},
    ])
    assert text == "user: hello\nassistant: hi\nuser: continue"


def test_load_variant_hash_check(tmp_path):
    path, _payload = _variant_file(tmp_path)
    good = hashlib.sha256(path.read_bytes()).hexdigest()
    variant, resolved = variant_chat.load_variant(str(path), good)
    assert variant.name == "v" and resolved == str(path)
    with pytest.raises(ValueError):
        variant_chat.load_variant(str(path), "deadbeef")


def test_variant_generate_endpoint_uses_variant_model(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    path, _payload = _variant_file(tmp_path, model_id="org/model")
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    state = ServiceState()
    calls = []

    @contextmanager
    def fake_acquire(model_id, revision=None):
        calls.append((model_id, revision))
        yield SimpleNamespace(model_id=model_id, revision=revision, device="cpu")

    state.acquire_model = fake_acquire

    def fake_generate(model, variant, request, on_chunk=None):
        return {
            "output": "server reply",
            "modelID": model.model_id,
            "modelRevision": model.revision,
            "variant": variant.to_dict(),
            "stripInterventions": request.strip_interventions,
            "prompt": variant_chat.render_chat_text(request.text, request.messages),
        }

    monkeypatch.setattr(variant_chat, "generate_with_variant", fake_generate)
    app = FastAPI()
    app.include_router(build_router(state))
    resp = TestClient(app).post("/api/variant/generate", json={
        "variantPath": str(path),
        "messages": [{"role": "user", "content": "hello"}],
        "stripInterventions": True,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "server reply"
    assert body["prompt"] == "user: hello"
    assert body["variantMetadata"]["baseModelID"] == "org/model"
    assert calls == [("org/model", "abc")]


def _inline_app(tmp_path, monkeypatch):
    """App + fake registry acquire for inline-variant tests (same harness the
    stored-variant generate tests use)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    state = ServiceState()
    calls = []

    @contextmanager
    def fake_acquire(model_id, revision=None):
        calls.append((model_id, revision))
        yield SimpleNamespace(model_id=model_id, revision=revision, device="cpu")

    state.acquire_model = fake_acquire
    app = FastAPI()
    app.include_router(build_router(state))
    return TestClient(app), calls


def _fake_generate(model, variant, request, on_chunk=None):
    return {
        "output": "server reply",
        "modelID": model.model_id,
        "modelRevision": model.revision,
        "variant": variant.to_dict(),
        "stripInterventions": request.strip_interventions,
        "prompt": variant_chat.render_chat_text(request.text, request.messages),
    }


def test_variant_generate_inline_spec(tmp_path, monkeypatch):
    client, calls = _inline_app(tmp_path, monkeypatch)
    monkeypatch.setattr(variant_chat, "generate_with_variant", _fake_generate)
    spec = model_variant.ModelVariant(
        name="explore", base_model_id="org/model", base_revision="abc",
        temperature=0.3, prompt_mode="rawCompletion").to_dict()
    resp = client.post("/api/variant/generate", json={
        "variant": spec,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "server reply"
    assert body["prompt"] == "user: hello"
    assert calls == [("org/model", "abc")]
    # Honest provenance: inline stamps source=inline and NEVER a path/hash —
    # an ephemeral spec must not masquerade as a stored hash-pinned variant.
    md = body["variantMetadata"]
    assert md["source"] == "inline"
    assert md["baseModelID"] == "org/model"
    assert md["name"] == "explore"
    assert "path" not in md and "resolvedPath" not in md and "hash" not in md


def test_variant_generate_requires_exactly_one_form(tmp_path, monkeypatch):
    client, _calls = _inline_app(tmp_path, monkeypatch)
    path, payload = _variant_file(tmp_path)
    # Neither form.
    resp = client.post("/api/variant/generate", json={"text": "x"})
    assert resp.status_code == 400
    assert "exactly one" in resp.json()["detail"]
    # Both forms.
    resp = client.post("/api/variant/generate", json={
        "variantPath": str(path), "variant": payload, "text": "x"})
    assert resp.status_code == 400
    assert "exactly one" in resp.json()["detail"]
    # Inline spec must be an object.
    resp = client.post("/api/variant/generate", json={"variant": "nope", "text": "x"})
    assert resp.status_code == 400
    # Same contract on the stream route.
    resp = client.post("/api/variant/generate/stream", json={"text": "x"})
    assert resp.status_code == 400
    resp = client.post("/api/variant/generate/stream", json={
        "variantPath": str(path), "variant": payload, "text": "x"})
    assert resp.status_code == 400


def test_variant_generate_inline_validation(tmp_path, monkeypatch):
    client, _calls = _inline_app(tmp_path, monkeypatch)
    # baseModelID is required (same schema as /api/variants/upload).
    resp = client.post("/api/variant/generate", json={
        "variant": {"name": "x"}, "text": "hi"})
    assert resp.status_code == 400
    assert "invalid inline variant spec" in resp.json()["detail"]
    # Referenced artifacts must resolve; a missing vector is a clear 400.
    spec = model_variant.ModelVariant(
        name="x", base_model_id="org/model",
        injections=[{"concept": "c", "vectorArtifactID": "runs/nope/vec",
                     "layer": 0, "alpha": 1.0}]).to_dict()
    resp = client.post("/api/variant/generate", json={"variant": spec, "text": "hi"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "missing artifacts" in detail and "runs/nope/vec" in detail


def test_variant_stream_inline_stamps_source(tmp_path, monkeypatch):
    client, calls = _inline_app(tmp_path, monkeypatch)
    monkeypatch.setattr(variant_chat, "stream_with_variant",
                        lambda model, variant, request, should_stop=None: iter(["hel", "lo"]))
    spec = model_variant.ModelVariant(
        name="explore", base_model_id="org/model", base_revision="abc").to_dict()
    resp = client.post("/api/variant/generate/stream", json={
        "variant": spec, "text": "hello"})
    assert resp.status_code == 200
    assert '"chunk": "hel"' in resp.text
    assert '"done": true' in resp.text
    assert '"source": "inline"' in resp.text
    assert '"resolvedPath"' not in resp.text
    assert calls == [("org/model", "abc")]


def test_variant_detail_returns_spec_and_hash(tmp_path, monkeypatch):
    client, _calls = _inline_app(tmp_path, monkeypatch)
    path, payload = _variant_file(tmp_path)
    resp = client.get("/api/variant/detail", params={"path": str(path)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["variant"] == json.loads(path.read_text(encoding="utf-8"))
    assert body["variant"]["baseModelID"] == "org/model"
    assert body["hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # The returned hash is exactly what variantHash pins on generate.
    variant_chat.load_variant(str(path), body["hash"])


def test_variant_detail_refuses_bad_paths_and_non_variants(tmp_path, monkeypatch):
    client, _calls = _inline_app(tmp_path, monkeypatch)
    # Escaping relative path → 400 (containment, same as other variant reads).
    resp = client.get("/api/variant/detail",
                      params={"path": "../../../../etc/passwd"})
    assert resp.status_code == 400
    # Missing file → 404.
    resp = client.get("/api/variant/detail",
                      params={"path": str(tmp_path / "nope.json")})
    assert resp.status_code == 404
    # A JSON file that is not a variant artifact → 400.
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    resp = client.get("/api/variant/detail", params={"path": str(other)})
    assert resp.status_code == 400
    assert "not a variant artifact" in resp.json()["detail"]


def test_variant_stream_announces_a_cold_load_before_the_silence(
        tmp_path, monkeypatch):
    # Live 2026-07-17: a chat against a not-yet-resident model performed a
    # multi-minute cold load INSIDE the stream with zero bytes sent — the
    # client had nothing to show but a spinner. The route must emit a status
    # preamble before the first chunk whenever the base model is not already
    # resident, and stay silent when it is.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    state = ServiceState()

    @contextmanager
    def fake_acquire(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision, device="cpu")

    state.acquire_model = fake_acquire
    monkeypatch.setattr(variant_chat, "stream_with_variant",
                        lambda model, variant, request, should_stop=None: iter(["ok"]))
    app = FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    spec = model_variant.ModelVariant(
        name="explore", base_model_id="org/model", base_revision="abc").to_dict()

    # Not resident → the status event precedes the first chunk and names the load.
    resp = client.post("/api/variant/generate/stream",
                       json={"variant": spec, "text": "hi"})
    assert resp.status_code == 200
    assert '"status"' in resp.text
    assert resp.text.index('"status"') < resp.text.index('"chunk"')
    assert "not loaded yet" in resp.text and "org/model" in resp.text

    # Mid-load elsewhere → says "still loading" (queued behind it), not a hang.
    state.registry = SimpleNamespace(
        residency=lambda model_id, revision=None, dtype="auto", device=None:
        "loading")
    resp = client.post("/api/variant/generate/stream",
                       json={"variant": spec, "text": "hi"})
    assert "is still loading" in resp.text

    # Resident → no status noise at all.
    state.registry = SimpleNamespace(
        residency=lambda model_id, revision=None, dtype="auto", device=None:
        "ready")
    resp = client.post("/api/variant/generate/stream",
                       json={"variant": spec, "text": "hi"})
    assert resp.status_code == 200
    assert '"status"' not in resp.text
    assert '"chunk": "ok"' in resp.text


def test_variant_stream_heartbeats_and_marks_the_generating_transition(
        tmp_path, monkeypatch):
    # Second live round 2026-07-17: the one preamble line froze on screen for
    # minutes while the load's real byte movement happened silently. The
    # stream must (a) heartbeat the status with elapsed time while nothing
    # substantive has arrived, and (b) emit a "ready — generating" transition
    # once the model is acquired, so a stall is attributable to a phase.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import time as _time
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("STEERLAB_SSE_HEARTBEAT_SECONDS", "0.05")
    state = ServiceState()

    @contextmanager
    def slow_acquire(model_id, revision=None):
        _time.sleep(0.3)  # the "cold load": nothing substantive arrives yet
        yield SimpleNamespace(model_id=model_id, revision=revision, device="cpu")

    state.acquire_model = slow_acquire
    monkeypatch.setattr(variant_chat, "stream_with_variant",
                        lambda model, variant, request, should_stop=None: iter(["ok"]))
    app = FastAPI()
    app.include_router(build_router(state))
    spec = model_variant.ModelVariant(
        name="explore", base_model_id="org/model", base_revision="abc").to_dict()
    resp = TestClient(app).post("/api/variant/generate/stream",
                                json={"variant": spec, "text": "hi"})
    assert resp.status_code == 200
    assert "elapsed" in resp.text                       # ≥1 heartbeat fired
    # json.dumps escapes the em-dash, so match the phase transition on its
    # distinctive tail word.
    assert "generating" in resp.text                    # phase transition
    # Ordering: preamble → heartbeat(s) → generating → first chunk.
    assert resp.text.index("elapsed") < resp.text.index("generating")
    assert resp.text.index("generating") < resp.text.index('"chunk"')
    assert '"done": true' in resp.text


def test_variant_stream_endpoint_returns_chunks(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    path, _payload = _variant_file(tmp_path)
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    state = ServiceState()

    @contextmanager
    def fake_acquire(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision, device="cpu")

    state.acquire_model = fake_acquire
    monkeypatch.setattr(variant_chat, "stream_with_variant",
                        lambda model, variant, request, should_stop=None: iter(["hel", "lo"]))

    app = FastAPI()
    app.include_router(build_router(state))
    resp = TestClient(app).post("/api/variant/generate/stream", json={
        "variantPath": str(path), "text": "hello"})
    assert resp.status_code == 200
    assert '"chunk": "hel"' in resp.text
    assert '"chunk": "lo"' in resp.text
    assert '"done": true' in resp.text

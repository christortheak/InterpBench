"""Residual-norm backfill (norm-unit alpha denominators for legacy /
SAE-import / reader-derived vector artifacts): core refusals, reuse of the
extraction-time measurement path at the sidecar's reading position, immutable
new-artifact semantics with the pinned ``normBackfill`` provenance shape, and
the /api/vectors/backfill-norms durable-job route. Model-free — activations
are monkeypatched exactly as in test_repe_reader.py."""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.steering import extractor, norm_backfill, vector_store
from steerlab_server.steering.reading_position import mean_from_token

FAKE_MODEL = SimpleNamespace(model_id="org/m", revision="abc")

# Everything backfill is allowed to change; every other sidecar key must come
# through byte-identical. ``residualNormConvention`` joined on 2026-08-20: the
# backfill re-MEASURES the denominator, so it stamps the convention it measured
# under — that is the researcher's opt-in migration path onto the whole-corpus
# rule, and the one place a legacy artifact's denominator legitimately moves.
UPDATED_KEYS = ("residualNormPerLayer", "residualNormSource",
                "residualNormConvention", "neutralCorpusHash", "normBackfill")


class _FakeActs:
    def __init__(self, norms):
        self.values = [[[0.0, 0.0]]]
        self.residual_norm_per_layer = list(norms)


def _fake_activations(norms, seen=None):
    def fake(model, texts, position):
        if seen is not None:
            seen["texts"] = list(texts)
            seen["position"] = position
        return _FakeActs(norms)
    return fake


def _write_corpus(path, texts=("a", "b", "c", "d")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}) + "\n")
    return path


def _save_artifact(directory, name="fear", *, residual=None,
                   reading=mean_from_token(5), model_id="org/m"):
    vectors = vector_store.ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 2.0]])
    sidecar = vector_store.SteeringVectorSidecar.make(
        model_id=model_id, concept=name, stimulus_set_hash="stim-hash",
        vectors=vectors, revision="abc", extraction_method="saeFeature",
        reading_position=reading, residual_norm_per_layer=residual,
        residual_norm_source=("neutral-corpus" if residual else None))
    vector_store.save(vectors, sidecar, directory, name)
    return directory


def _enrich_sidecar(directory, name="fear"):
    """Stamp reader-derived provenance plus a field this engine does not model
    — byte-preservation must survive both."""
    path = os.path.join(directory, f"{name}.json")
    with open(path, encoding="utf-8") as handle:
        d = json.load(handle)
    d.update({"source": "repe-reader-lat", "readerID": "runs/x/reader.json",
              "readerHash": "rh", "controlMode": "reading-vector activation addition",
              "futureUnknownField": "keep-me"})
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(d, handle, sort_keys=True, indent=2)


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


# --- core refusals -------------------------------------------------------------

def test_backfill_refuses_when_norms_already_present(tmp_path):
    orig = _save_artifact(str(tmp_path / "orig"), residual=[3.0, 4.0])
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    with pytest.raises(ValueError, match="backfill never overwrites"):
        norm_backfill.backfill_norms(FAKE_MODEL, orig, "fear", corpus,
                                     str(tmp_path / "out"))


def test_backfill_rejects_model_mismatch(tmp_path):
    orig = _save_artifact(str(tmp_path / "orig"), model_id="org/other")
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    with pytest.raises(ValueError, match="per-model measurement"):
        norm_backfill.backfill_norms(FAKE_MODEL, orig, "fear", corpus,
                                     str(tmp_path / "out"))
    assert not os.path.exists(str(tmp_path / "out"))


def test_backfill_rejects_fewer_measured_layers(tmp_path, monkeypatch):
    # 1 measured layer vs the artifact's 2 → wrong model family.
    monkeypatch.setattr(extractor, "activations",
                        _fake_activations([1.0]))
    orig = _save_artifact(str(tmp_path / "orig"))
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    with pytest.raises(ValueError, match="wrong model family"):
        norm_backfill.backfill_norms(FAKE_MODEL, orig, "fear", corpus,
                                     str(tmp_path / "out"))


def test_backfill_rejects_nonfinite_measurement(tmp_path, monkeypatch):
    # An overflow anywhere in the measurement (even past the artifact's
    # layers) poisons the denominator — reject the whole measurement.
    monkeypatch.setattr(extractor, "activations",
                        _fake_activations([1.0, 2.0, float("nan")]))
    orig = _save_artifact(str(tmp_path / "orig"))
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    with pytest.raises(ValueError, match="non-finite"):
        norm_backfill.backfill_norms(FAKE_MODEL, orig, "fear", corpus,
                                     str(tmp_path / "out"))


def test_backfill_takes_prefix_of_longer_measurement(tmp_path, monkeypatch):
    # Reader-/SAE-derived artifacts carry vectors only up to their injection
    # layer (zeros-below convention), so the measurement legitimately covers
    # MORE layers than the artifact — the block-index prefix is the correct
    # denominator (mirrors Swift NormBackfill.alignedNorms).
    monkeypatch.setattr(extractor, "activations",
                        _fake_activations([1.0, 2.0, 3.0]))
    orig = _save_artifact(str(tmp_path / "orig"))
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    result = norm_backfill.backfill_norms(FAKE_MODEL, orig, "fear", corpus,
                                          str(tmp_path / "out"))
    assert result.residual_norm_per_layer == [1.0, 2.0]


def test_backfill_rejects_tiny_corpus(tmp_path, monkeypatch):
    # Extraction's ≥4-text neutral-corpus gate applies to the denominator here
    # too — a 2-text "corpus" must not silently redefine α.
    monkeypatch.setattr(extractor, "activations", _fake_activations([1.0, 2.0]))
    orig = _save_artifact(str(tmp_path / "orig"))
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"), texts=("a", "b"))
    with pytest.raises(ValueError, match="at least 4"):
        norm_backfill.backfill_norms(FAKE_MODEL, orig, "fear", corpus,
                                     str(tmp_path / "out"))


# --- happy path ------------------------------------------------------------------

def test_backfill_happy_path(tmp_path, monkeypatch):
    orig = _save_artifact(str(tmp_path / "orig"))
    _enrich_sidecar(orig)
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    corpus_bytes = _read_bytes(corpus)
    orig_vec_bytes = _read_bytes(os.path.join(orig, "fear.safetensors"))
    orig_sidecar_bytes = _read_bytes(os.path.join(orig, "fear.json"))

    seen = {}
    monkeypatch.setattr(extractor, "activations",
                        _fake_activations([7.5, 8.25], seen))
    out = str(tmp_path / "backfill-run")
    result = norm_backfill.backfill_norms(FAKE_MODEL, orig, "fear", corpus, out)

    # The extraction measurement path ran over the corpus texts at the
    # artifact's own stamped reading position.
    assert seen["texts"] == ["a", "b", "c", "d"]
    assert seen["position"].label == "mean from token 5"

    # Measured norms land under the extraction-time conventions.
    with open(result.sidecar_path, encoding="utf-8") as handle:
        new = json.load(handle)
    assert new["residualNormPerLayer"] == [7.5, 8.25]
    assert new["residualNormSource"] == "neutral-corpus"
    assert new["neutralCorpusHash"] == hashlib.sha256(corpus_bytes).hexdigest()

    # Pinned provenance contract: exactly this shape (Swift decodes it).
    nb = new["normBackfill"]
    assert set(nb) == {"sourceArtifact", "sourceVectorsHash", "date"}
    assert nb["sourceArtifact"] == os.path.join(orig, "fear")
    assert nb["sourceVectorsHash"] == hashlib.sha256(orig_vec_bytes).hexdigest()
    assert nb["date"].endswith("Z") and "T" in nb["date"]

    # Same vectors bytes; every other sidecar field preserved — including
    # reader-derived provenance and fields this engine does not model.
    assert _read_bytes(result.vectors_path) == orig_vec_bytes
    old = json.loads(orig_sidecar_bytes.decode("utf-8"))
    assert not any(k in old for k in UPDATED_KEYS)
    assert {k: v for k, v in new.items() if k not in UPDATED_KEYS} == old
    assert new["futureUnknownField"] == "keep-me"
    assert new["controlMode"] == "reading-vector activation addition"
    # The convention the norms were just measured under — stamped, not guessed.
    assert new["residualNormConvention"] == "wholeCorpusMean-v1"

    # The original artifact is untouched on disk (runs are immutable).
    assert _read_bytes(os.path.join(orig, "fear.safetensors")) == orig_vec_bytes
    assert _read_bytes(os.path.join(orig, "fear.json")) == orig_sidecar_bytes

    # Result surface + round-trip through the store keeps the provenance.
    assert result.run_directory == out
    assert result.artifact_id == os.path.join(out, "fear")
    assert result.layer_count == 2
    assert result.residual_norm_source == "neutral-corpus"
    vectors, sidecar = vector_store.load(out, "fear")
    assert vectors.per_layer == [[1.0, 0.0], [0.0, 2.0]]
    assert sidecar.residualNormPerLayer == [7.5, 8.25]
    assert sidecar.normBackfill == nb
    # A second backfill of the NEW artifact refuses — norms are now present.
    with pytest.raises(ValueError, match="backfill never overwrites"):
        norm_backfill.backfill_norms(FAKE_MODEL, out, "fear", corpus,
                                     str(tmp_path / "out2"))


def test_backfill_output_name_override(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor, "activations", _fake_activations([1.0, 2.0]))
    orig = _save_artifact(str(tmp_path / "orig"))
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    result = norm_backfill.backfill_norms(
        FAKE_MODEL, orig, "fear", corpus, str(tmp_path / "out"),
        output_name="fear-normed")
    assert os.path.basename(result.vectors_path) == "fear-normed.safetensors"
    _, sidecar = vector_store.load(str(tmp_path / "out"), "fear-normed")
    assert sidecar.residualNormPerLayer == [1.0, 2.0]


# --- POST /api/vectors/backfill-norms (durable job) ----------------------------

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from steerlab_server.api import app as app_module  # noqa: E402

client = TestClient(app_module.app)


def _wait_for_job(job_id, tries=200):
    import time
    job = None
    for _ in range(tries):  # daemon-thread job; poll briefly
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    return job


def _route_harness(tmp_path, monkeypatch, norms=(7.5, 8.25)):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    monkeypatch.setattr(extractor, "activations", _fake_activations(list(norms)))
    fake = SimpleNamespace(model_id="org/m", revision="abc")
    monkeypatch.setattr(app_module.state, "model", fake)
    acquired = []

    @contextmanager
    def fake_acquire_model(model_id, revision=None):
        acquired.append((model_id, revision))
        yield SimpleNamespace(model_id=model_id, revision=revision)

    monkeypatch.setattr(app_module.state, "acquire_model", fake_acquire_model)
    return acquired


def test_api_backfill_body_validation(tmp_path, monkeypatch):
    # All of these fail synchronously, before any job runs.
    _route_harness(tmp_path, monkeypatch)
    _save_artifact(os.path.join(str(tmp_path), "runs", "legacy"))
    _save_artifact(os.path.join(str(tmp_path), "runs", "normed"),
                   residual=[3.0, 4.0])
    _write_corpus(os.path.join(str(tmp_path), "prompts", "neutral.jsonl"))
    ok = {"vectorID": "runs/legacy/fear",
          "neutralCorpusPath": "prompts/neutral.jsonl"}

    assert client.post("/api/vectors/backfill-norms",
                       json={**ok, "vectorID": ""}).status_code == 400
    assert client.post("/api/vectors/backfill-norms",
                       json={**ok, "neutralCorpusPath": ""}).status_code == 400
    # A bare name has no containable run directory.
    assert client.post("/api/vectors/backfill-norms",
                       json={**ok, "vectorID": "fear"}).status_code == 400
    # Missing artifact halves → 400; a run directory that is not there → 404.
    assert client.post("/api/vectors/backfill-norms",
                       json={**ok, "vectorID": "runs/legacy/nope"}).status_code == 400
    assert client.post("/api/vectors/backfill-norms",
                       json={**ok, "vectorID": "runs/absent/fear"}).status_code == 404
    # Missing corpus file → 404 (same containment as other artifact reads).
    assert client.post(
        "/api/vectors/backfill-norms",
        json={**ok, "neutralCorpusPath": "prompts/absent.jsonl"}).status_code == 404
    # Already has residual norms — backfill never overwrites.
    resp = client.post("/api/vectors/backfill-norms",
                       json={**ok, "vectorID": "runs/normed/fear"})
    assert resp.status_code == 400
    assert "backfill never overwrites" in resp.json()["detail"]
    # outputName is a path component.
    assert client.post("/api/vectors/backfill-norms",
                       json={**ok, "outputName": "a/b"}).status_code == 400
    # Wrong-model submission is rejected synchronously.
    resp = client.post("/api/vectors/backfill-norms",
                       json={**ok, "modelID": "org/other"})
    assert resp.status_code == 400
    assert "per-model measurement" in resp.json()["detail"]
    # A valid body without a loaded model (and no explicit modelID) → 409.
    monkeypatch.setattr(app_module.state, "model", None)
    assert client.post("/api/vectors/backfill-norms", json=ok).status_code == 409


def test_api_backfill_job_happy_path(tmp_path, monkeypatch):
    acquired = _route_harness(tmp_path, monkeypatch)
    orig = _save_artifact(os.path.join(str(tmp_path), "runs", "legacy"))
    _enrich_sidecar(orig)
    orig_sidecar_bytes = _read_bytes(os.path.join(orig, "fear.json"))
    _write_corpus(os.path.join(str(tmp_path), "prompts", "neutral.jsonl"))

    resp = client.post("/api/vectors/backfill-norms", json={
        "vectorID": "runs/legacy/fear",
        "neutralCorpusPath": "prompts/neutral.jsonl"})
    assert resp.status_code == 200
    job = _wait_for_job(resp.json()["jobId"])
    assert job is not None and job["status"] == "succeeded", job

    result = job["result"]
    run_dir = result["runDirectory"]
    assert os.path.basename(run_dir).endswith("-api-backfill-norms-fear")
    assert os.path.realpath(run_dir).startswith(os.path.realpath(str(tmp_path)))
    assert result["artifact"] == os.path.join(run_dir, "fear")
    assert result["residualNormSource"] == "neutral-corpus"
    assert result["layerCount"] == 2
    # The job acquired exactly the model snapshotted at submission.
    assert acquired == [("org/m", "abc")]

    # New artifact carries the norms; the original is untouched on disk.
    _, sidecar = vector_store.load(run_dir, "fear")
    assert sidecar.residualNormPerLayer == [7.5, 8.25]
    assert sidecar.normBackfill["sourceArtifact"] == os.path.join(orig, "fear")
    assert _read_bytes(os.path.join(orig, "fear.json")) == orig_sidecar_bytes

    # The catalog now lists both: the original without norms, the new with.
    listed = client.get("/api/vectors").json()["vectors"]
    by_id = {v["id"]: v for v in listed}
    assert by_id[os.path.join(orig, "fear")]["hasResidualNorms"] is False
    assert by_id[os.path.join(run_dir, "fear")]["hasResidualNorms"] is True
    assert by_id[os.path.join(run_dir, "fear")]["residualNormSource"] == "neutral-corpus"


def test_backfill_job_type_advertised():
    body = client.get("/api/capabilities").json()
    assert "vector-backfill-norms" in body["availableJobTypes"]


# --- redenomination (loud, never silent — the freeze --force pattern) -----------

def _stimulus_denominated(tmp_path):
    """An artifact whose EXISTING norms were measured on extraction stimuli —
    the legacy concept-dependent denominator redenomination exists for."""
    orig = _save_artifact(str(tmp_path / "orig"), residual=[3.0, 4.0])
    path = os.path.join(orig, "fear.json")
    with open(path, encoding="utf-8") as handle:
        d = json.load(handle)
    d["residualNormSource"] = "extraction-stimuli"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(d, handle, sort_keys=True, indent=2)
    return orig


def test_redenominate_rewrites_stimulus_denominated_norms(tmp_path, monkeypatch):
    orig = _stimulus_denominated(tmp_path)
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    monkeypatch.setattr(extractor, "activations", _fake_activations([7.0, 8.0]))

    result = norm_backfill.backfill_norms(
        FAKE_MODEL, orig, "fear", corpus, str(tmp_path / "out"),
        redenominate=True)

    with open(result.sidecar_path, encoding="utf-8") as handle:
        sidecar = json.load(handle)
    assert sidecar["residualNormPerLayer"] == [7.0, 8.0]
    assert sidecar["residualNormSource"].startswith("neutral-corpus")
    # Provenance records what the new denominator replaced.
    assert sidecar["normBackfill"]["replacedNormSource"] == "extraction-stimuli"
    # The original is untouched (immutability).
    with open(os.path.join(orig, "fear.json"), encoding="utf-8") as handle:
        original = json.load(handle)
    assert original["residualNormPerLayer"] == [3.0, 4.0]
    assert original["residualNormSource"] == "extraction-stimuli"


def test_redenominate_refuses_already_neutral_artifacts(tmp_path):
    orig = _save_artifact(str(tmp_path / "orig"), residual=[3.0, 4.0])
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    with pytest.raises(ValueError, match="nothing to redenominate"):
        norm_backfill.backfill_norms(
            FAKE_MODEL, orig, "fear", corpus, str(tmp_path / "out"),
            redenominate=True)


def test_classic_backfill_records_no_replaced_source(tmp_path, monkeypatch):
    orig = _save_artifact(str(tmp_path / "orig"))
    corpus = _write_corpus(str(tmp_path / "neutral.jsonl"))
    monkeypatch.setattr(extractor, "activations", _fake_activations([7.0, 8.0]))
    result = norm_backfill.backfill_norms(
        FAKE_MODEL, orig, "fear", corpus, str(tmp_path / "out"))
    with open(result.sidecar_path, encoding="utf-8") as handle:
        sidecar = json.load(handle)
    assert "replacedNormSource" not in sidecar["normBackfill"]

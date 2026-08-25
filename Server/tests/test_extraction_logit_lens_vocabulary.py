"""The per-cell logit-lens VOCABULARY check, written at extract time.

A rendering × reading-position grid produces one direction per cell, and the
question asked of every cell is what vocabulary that direction promotes: the
doctrine-vs-affect confound is invisible in a probe accuracy and obvious in a
top-10 readout. So the extraction itself records it, beside the vectors, for
every extracted direction at the study's own declared depth.

Never a gate, and never in the sidecar (which is a cross-engine artifact
contract). A projection that cannot run is a recorded skip string, exactly as
``validate``'s logit-lens block records one — a diagnostic may not sink a run.

ENGINE ASYMMETRY, deliberate: swift-mlx writes no extract-time equivalent (it
runs the same lens inside ``validate``); the grid this serves runs here.
"""

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering.vector_store import ConceptVectors

VOCABULARY_FILE = "logit-lens-vocabulary.json"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _workspace(root, name="lensstudy", concept="steadiness"):
    directory = os.path.join(root, "prompts", "concepts", concept)
    _write(os.path.join(directory, "positive.jsonl"),
           '{"text": "a steady voice"}\n{"text": "an even keel"}\n')
    _write(os.path.join(directory, "negative.jsonl"),
           '{"text": "a rushed reply"}\n{"text": "a sharp turn"}\n')
    _write(os.path.join(directory, "markers.json"),
           json.dumps({"words": ["steady"]}))
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, [concept], root=root)
    return Manifest.load(name, root).concepts[0].stimulus_set_hash


class _LensTokenizer:
    def decode(self, ids):
        return f"piece{ids[0]}"


class _LensModel:
    """Just enough model for the lens: a residual→vocabulary projection and a
    tokenizer that can name an id. Mirrors ``SteeredModel``'s two members."""

    revision = "abc"
    tokenizer = _LensTokenizer()

    def logits_for_residual_vector(self, vector):
        # A deterministic ramp, so the top-k is a known set of ids.
        return [float(i) * (1.0 + sum(vector)) for i in range(64)]


@contextmanager
def _lens_model(model_id, revision):
    yield _LensModel()


@contextmanager
def _blind_model(model_id, revision):
    """A model with no unembedding reachable — the skip path."""
    yield SimpleNamespace(revision=revision)


def _bundle(stimulus_hash):
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="extraction-stimuli", stimulus_hash=stimulus_hash)


def _extract(tmp_path, monkeypatch, provider, **kwargs):
    root = str(tmp_path)
    stimulus_hash = _workspace(root, **kwargs)
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"steadiness": _bundle(stimulus_hash)})
    logs = []
    run_dir = tasks.extract("lensstudy", root, model_provider=provider,
                            log=logs.append)
    return run_dir, logs


def _read(run_dir):
    with open(os.path.join(run_dir, VOCABULARY_FILE), encoding="utf-8") as h:
        return json.load(h)


def test_every_extracted_direction_gets_a_top_ten_vocabulary_readout(
        tmp_path, monkeypatch):
    run_dir, logs = _extract(tmp_path, monkeypatch, _lens_model)
    report = _read(run_dir)
    depths = report["steadiness"]
    assert len(depths) == 1                    # one declared validation depth
    entry = depths[0]
    assert len(entry["topPositive"]) == tasks.LOGIT_LENS_VOCABULARY_TOP_K == 10
    assert len(entry["topNegative"]) == 10
    # Highest-logit first, and the pieces are named — a reader sees the
    # vocabulary, not a row of ids.
    assert entry["topPositive"][0]["logit"] >= entry["topPositive"][1]["logit"]
    assert entry["topPositive"][0]["token"].startswith("piece")
    # It says WHICH layer it read and why that layer — the same both-halves
    # discipline every other resolution stamp follows.
    assert entry["layer"] == entry["layerResolution"]["layer"]
    assert entry["layerResolution"]["source"]
    assert any("logit-lens vocabulary" in line for line in logs)


def test_the_readout_lands_beside_the_vectors_and_not_in_the_sidecar(
        tmp_path, monkeypatch):
    """The sidecar is a cross-engine contract; a Python-only diagnostic may
    not add keys to it."""
    run_dir, _ = _extract(tmp_path, monkeypatch, _lens_model)
    assert os.path.exists(os.path.join(run_dir, VOCABULARY_FILE))
    with open(os.path.join(run_dir, "steadiness.json"), encoding="utf-8") as h:
        sidecar = json.load(h)
    assert "logitLens" not in sidecar
    assert "logitLensVocabulary" not in sidecar


def test_a_projection_that_cannot_run_is_recorded_not_raised(
        tmp_path, monkeypatch):
    """A diagnostic never sinks an extraction: the run completes, and the file
    says what did not happen."""
    run_dir, _ = _extract(tmp_path, monkeypatch, _blind_model)
    assert os.path.isdir(run_dir)
    depths = _read(run_dir)["steadiness"]
    assert isinstance(depths[0], str)
    assert depths[0].startswith("logit-lens skipped:")


def test_the_readout_is_deterministic_for_one_direction(tmp_path, monkeypatch):
    first, _ = _extract(tmp_path, monkeypatch, _lens_model)
    second, _ = _extract(tmp_path / "again", monkeypatch, _lens_model)
    assert _read(first) == _read(second)

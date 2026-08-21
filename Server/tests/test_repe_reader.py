"""Faithful RepE reader LAT (docs/REPE-IMPLEMENTATION-BRIEF.md): templates,
paired reader datasets, fit math, artifact round-trip, exact inference, the
manifest substrate gate, endpoint stamping, derive-steering provenance, and the
/api/readers + /api/reader/score routes. Model-free throughout — activations
are monkeypatched exactly as in test_grand_mean_lifecycle.py."""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import catalog, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.experiment.prompt_render import (
    READER_RENDERING_CONVENTION, render_reader)
from steerlab_server.steering import extractor, repe_reader
from steerlab_server.steering.vector_math import ScalarProbe

REPO_TEMPLATES = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "prompts", "templates"))

FAKE_MODEL = SimpleNamespace(model_id="org/m", revision="abc")

AMOUNT_TEMPLATE_TEXT = (
    "Consider the amount of {{concept}} in the following scenario:\n"
    "Scenario: {{stimulus}}\n"
    "The amount of {{concept}} in the scenario is")


def _write_template(directory, template_id="amount-in-scenario-v1", **overrides):
    obj = {"id": template_id, "conceptSlot": True, "text": AMOUNT_TEMPLATE_TEXT,
           "latToken": "final"}
    obj.update(overrides)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{template_id}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle)
    return path


def _write_pairs(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _pair_row(i, pos, neg, split="train", concept="fear",
              template_id="amount-in-scenario-v1"):
    return {"id": f"{concept}-pair-{i}", "concept": concept,
            "positiveStimulus": pos, "negativeStimulus": neg,
            "topic": "t", "split": split, "templateID": template_id}


# --- templates ---------------------------------------------------------------

def test_starter_templates_are_committed_and_load():
    named = repe_reader.load_template(
        os.path.join(REPO_TEMPLATES, "amount-in-scenario-v1.json"))
    assert named.concept_slot and named.lat_token == "final"
    assert named.divergence is None
    rendered = named.render(stimulus="It rains.", concept="fear")
    assert "amount of fear" in rendered and "Scenario: It rains." in rendered
    assert "{{" not in rendered

    unnamed = repe_reader.load_template(
        os.path.join(REPO_TEMPLATES, "unnamed-scenario-v1.json"))
    assert not unnamed.concept_slot
    assert unnamed.divergence == "unnamed-clean-room"  # stamped divergence
    rendered = unnamed.render(stimulus="It rains.")  # no concept named
    assert "Scenario: It rains." in rendered and "fear" not in rendered


def test_template_hash_is_raw_bytes_and_stable(tmp_path):
    path = _write_template(str(tmp_path))
    with open(path, "rb") as handle:
        expected = hashlib.sha256(handle.read()).hexdigest()
    first = repe_reader.load_template(path)
    second = repe_reader.load_template(path)
    assert first.hash == second.hash == expected


def test_template_validation(tmp_path):
    # id must match filename (one file per id).
    path = _write_template(str(tmp_path), template_id="a-v1")
    os.rename(path, os.path.join(str(tmp_path), "b-v1.json"))
    with pytest.raises(repe_reader.RepeReaderError, match="does not match filename"):
        repe_reader.load_template(os.path.join(str(tmp_path), "b-v1.json"))
    # named template requires a concept at render time
    named = repe_reader.load_template(_write_template(str(tmp_path)))
    with pytest.raises(repe_reader.RepeReaderError, match="names the concept"):
        named.render(stimulus="x")
    # a {{stimulus}} slot is mandatory
    bad = _write_template(str(tmp_path), template_id="no-slot-v1",
                          text="no slot here", conceptSlot=False)
    with pytest.raises(repe_reader.RepeReaderError, match="stimulus"):
        repe_reader.load_template(bad).render(stimulus="x")
    # only latToken "final" is implemented (honesty: refuse, don't guess)
    other = repe_reader.load_template(_write_template(
        str(tmp_path), template_id="mid-v1", latToken="penultimate"))
    with pytest.raises(repe_reader.RepeReaderError, match="latToken"):
        _ = other.reading_position


def test_reader_scaffold_guard_rejects_embedded_chat_markers():
    # The #1 tokenizer risk: a hand-tokenized scaffold smuggling template
    # markers (double-BOS hazard). Both families' markers are refused.
    for marker in ("<bos>", "<start_of_turn>", "<|im_start|>"):
        with pytest.raises(ValueError, match="special/chat-template"):
            render_reader(f"{marker} scenario text", model_id="org/gemma-3-4b-it")
    with pytest.raises(ValueError, match="BOS"):
        render_reader("<s> scenario", model_id="org/m")
    assert render_reader("plain scaffold", model_id="Qwen/Qwen3-4B") == "plain scaffold"


# --- dataset -----------------------------------------------------------------

def test_pairs_loader_hash_split_and_errors(tmp_path):
    path = _write_pairs(str(tmp_path / "pairs.jsonl"), [
        _pair_row(0, "p0", "n0"),
        {**_pair_row(1, "p1", "n1"), "split": "TEST"},
        {k: v for k, v in _pair_row(2, "p2", "n2").items() if k != "split"},
    ])
    with open(path, "rb") as handle:
        expected = hashlib.sha256(handle.read()).hexdigest()
    dataset = repe_reader.load_pairs(path)
    assert dataset.hash == expected
    assert dataset.concept == "fear"
    assert [p.split for p in dataset.pairs] == ["train", "test", "train"]
    assert len(dataset.train) == 2 and len(dataset.held_out) == 1

    mixed = _write_pairs(str(tmp_path / "mixed.jsonl"), [
        _pair_row(0, "p0", "n0"), _pair_row(1, "p1", "n1", concept="calm")])
    with pytest.raises(repe_reader.RepeReaderError, match="mixed concepts"):
        repe_reader.load_pairs(mixed)

    missing = str(tmp_path / "missing.jsonl")
    with open(missing, "w", encoding="utf-8") as handle:
        handle.write('{"positiveStimulus": "p", "concept": "fear"}\n')
    with pytest.raises(repe_reader.RepeReaderError, match="negativeStimulus"):
        repe_reader.load_pairs(missing)


# --- fit math on synthetic activations ----------------------------------------

# One layer, two dims; the concept lives on the x axis. Held-out pair has one
# deliberately misclassified positive (x below the train center).
ACTS = {
    "p0": [[2.0, 0.5]], "n0": [[0.0, 0.5]],
    "p1": [[2.0, -0.5]], "n1": [[0.0, -0.5]],
    "hp": [[0.5, 0.0]], "hn": [[0.0, 0.0]],
}


class _FakeActs:
    def __init__(self, values):
        self.values = values
        self.residual_norm_per_layer = [1.0]


def _fake_activations(model, texts, position):
    values = []
    for text in texts:
        key = next(k for k in ACTS if f"Scenario: {k}\n" in text)
        values.append(ACTS[key])
    return _FakeActs(values)


def _fit_dataset(tmp_path, swap_labels=False):
    template = repe_reader.load_template(_write_template(str(tmp_path)))
    rows = [
        _pair_row(0, "n0" if swap_labels else "p0", "p0" if swap_labels else "n0"),
        _pair_row(1, "n1" if swap_labels else "p1", "p1" if swap_labels else "n1"),
        _pair_row(2, "hn" if swap_labels else "hp", "hp" if swap_labels else "hn",
                  split="test"),
    ]
    dataset = repe_reader.load_pairs(
        _write_pairs(str(tmp_path / "pairs.jsonl"), rows))
    return dataset, template


def test_fit_math_orientation_and_accuracy(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor, "activations", _fake_activations)
    dataset, template = _fit_dataset(tmp_path)
    readers = repe_reader.fit(FAKE_MODEL, dataset, template)
    assert len(readers) == 1
    reader = readers[0]
    # PC1 of the normalized pair differences is the +x concept axis, oriented
    # so the positive class scores positive.
    assert reader.probe.direction == pytest.approx([1.0, 0.0], abs=1e-5)
    assert reader.pc1_explained_variance == pytest.approx(1.0, abs=1e-5)
    # Training normalization: centered at the train activation mean.
    assert reader.probe.activation_center == pytest.approx([1.0, 0.0], abs=1e-5)
    assert reader.train_accuracy == 1.0
    # Held-out pair: positive at x=0.5 falls below the center → misclassified;
    # negative at x=0 is correct → 0.5.
    assert reader.held_out_accuracy == 0.5
    assert reader.train_pair_count == 2 and reader.held_out_pair_count == 1
    assert reader.substrate == "python-hf-transformers"
    assert reader.rendering_convention == READER_RENDERING_CONVENTION
    assert reader.model_id == "org/m" and reader.revision == "abc"
    # Positive train stimulus scores positive through the fitted probe.
    assert repe_reader.score_activation(reader, ACTS["p0"][0]) > 0
    assert repe_reader.score_activation(reader, ACTS["n0"][0]) < 0


def test_fit_orients_by_paired_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor, "activations", _fake_activations)
    dataset, template = _fit_dataset(tmp_path, swap_labels=True)
    reader = repe_reader.fit(FAKE_MODEL, dataset, template)[0]
    # With labels swapped the reading direction flips: −x is now "positive".
    assert reader.probe.direction == pytest.approx([-1.0, 0.0], abs=1e-5)
    assert repe_reader.score_activation(reader, ACTS["n0"][0]) > 0


def test_fit_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor, "activations", _fake_activations)
    dataset, template = _fit_dataset(tmp_path)
    other = repe_reader.load_template(_write_template(
        str(tmp_path / "other"), template_id="other-v1"))
    with pytest.raises(repe_reader.RepeReaderError, match="pins template"):
        repe_reader.fit(FAKE_MODEL, dataset, other)
    one_pair = repe_reader.load_pairs(_write_pairs(
        str(tmp_path / "one.jsonl"), [_pair_row(0, "p0", "n0")]))
    with pytest.raises(repe_reader.RepeReaderError, match="at least 2 train"):
        repe_reader.fit(FAKE_MODEL, one_pair, template)


# --- artifact round-trip -------------------------------------------------------

def test_artifact_roundtrip_and_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor, "activations", _fake_activations)
    dataset, template = _fit_dataset(tmp_path)
    reader = repe_reader.fit(FAKE_MODEL, dataset, template)[0]
    path = repe_reader.save_reader(reader, str(tmp_path / "run"))
    assert path.endswith("reader-fear-layer0.json")
    loaded = repe_reader.load_reader(path)
    assert loaded.to_dict() == reader.to_dict()
    probe_input = [0.3, 9.9]
    assert repe_reader.score_activation(loaded, probe_input) == \
        repe_reader.score_activation(reader, probe_input)
    # Brief §4 artifact contract.
    d = reader.to_dict()
    assert d["artifactType"] == "repe-reader-lat"
    assert d["substrate"] == "python-hf-transformers"
    assert d["templateID"] == "amount-in-scenario-v1"
    assert d["templateHash"] == template.hash
    assert d["datasetHash"] == dataset.hash
    assert d["latTokenPosition"] == "final"
    assert d["readingPosition"] == "last token"
    assert set(d["probe"]) >= {"direction", "projectionCenter", "projectionScale",
                               "orientation", "activationCenter"}
    assert d["renderingConvention"] == READER_RENDERING_CONVENTION
    assert "trainAccuracy" in d and "heldOutAccuracy" in d
    assert "pc1ExplainedVariance" in d

    with pytest.raises(repe_reader.RepeReaderError, match="not a repe-reader-lat"):
        repe_reader.ReaderArtifact.from_dict({"artifactType": "other"})


# --- exact inference -----------------------------------------------------------

def _manual_reader(layer=0, substrate=repe_reader.SUBSTRATE):
    template = repe_reader.TaskTemplate(
        id="unnamed-scenario-v1", text="S: {{stimulus}} q", concept_slot=False,
        lat_token="final", hash="th", divergence="unnamed-clean-room")
    probe = ScalarProbe(direction=[1.0, 0.0], projection_center=0.5,
                        projection_scale=2.0, orientation=1.0,
                        positive_mean=1.0, negative_mean=-1.0,
                        activation_center=[1.0, 0.0])
    return repe_reader.ReaderArtifact(
        model_id="org/m", revision="abc", concept="fear", layer=layer,
        template=template, dataset_hash="dh", probe=probe,
        pc1_explained_variance=0.9, train_accuracy=1.0, held_out_accuracy=0.8,
        train_pair_count=4, held_out_pair_count=2, substrate=substrate)


def test_score_text_exact_inference(monkeypatch):
    reader = _manual_reader()
    seen = {}

    def fake_activations(model, texts, position):
        seen["texts"] = texts
        seen["position"] = position
        return _FakeActs([[[3.0, 7.0]] for _ in texts])

    monkeypatch.setattr(extractor, "activations", fake_activations)
    score = repe_reader.score_text(FAKE_MODEL, reader, "hello")
    # Same template, same token position…
    assert seen["texts"] == ["S: hello q"]
    assert seen["position"].label == "last token"
    # …and the training normalization: (([3,7]−[1,0])·[1,0] − 0.5) / 2 = 0.75.
    assert score == pytest.approx(0.75)
    assert repe_reader.score_activation(reader, [3.0, 7.0]) == pytest.approx(0.75)


def test_score_rejects_foreign_substrate_and_bad_layer(monkeypatch):
    monkeypatch.setattr(
        extractor, "activations",
        lambda model, texts, position: _FakeActs([[[1.0, 0.0]] for _ in texts]))
    foreign = _manual_reader(substrate="swift-mlx")
    with pytest.raises(repe_reader.RepeReaderError, match="substrate-specific"):
        repe_reader.score_texts(FAKE_MODEL, foreign, ["x"])
    deep = _manual_reader(layer=3)
    with pytest.raises(repe_reader.RepeReaderError, match="out of range"):
        repe_reader.score_texts(FAKE_MODEL, deep, ["x"])


def test_score_rejects_wrong_model(monkeypatch):
    # A reader is a per-model measurement instrument: scoring through a model
    # other than the one it was fitted on must refuse, not silently return
    # meaningless numbers.
    monkeypatch.setattr(
        extractor, "activations",
        lambda model, texts, position: _FakeActs([[[1.0, 0.0]] for _ in texts]))
    reader = _manual_reader()  # fitted on org/m
    other = SimpleNamespace(model_id="org/other", revision=None)
    with pytest.raises(repe_reader.RepeReaderError,
                       match="per-model measurement instrument"):
        repe_reader.score_texts(other, reader, ["x"])


# --- manifest pinning + substrate gate ------------------------------------------

def _reader_tree(tmp_path, substrate=repe_reader.SUBSTRATE, model_id="org/m"):
    """Temp root with one concept (so verify has a pinned study) + one saved
    reader artifact; returns (root, relative reader path, reader file hash)."""
    from steerlab_server.steering.stimulus_set import StimulusSet
    root = str(tmp_path)
    cdir = os.path.join(root, "prompts", "concepts", "joy")
    os.makedirs(cdir)
    with open(os.path.join(cdir, "positive.jsonl"), "w") as handle:
        handle.write('{"text": "yay"}\n')
    with open(os.path.join(cdir, "negative.jsonl"), "w") as handle:
        handle.write('{"text": "meh"}\n')
    stimulus_hash = StimulusSet.from_directory(cdir).hash

    reader = _manual_reader(substrate=substrate)
    reader.model_id = model_id
    run_dir = os.path.join(root, "runs", "20260703T000000000-reader-fit")
    path = repe_reader.save_reader(reader, run_dir)
    with open(path, "rb") as handle:
        reader_hash = hashlib.sha256(handle.read()).hexdigest()
    rel = os.path.relpath(path, root)
    manifest = {
        "name": "r", "modelID": "org/m",
        "concepts": [{"name": "joy", "stimulusSetHash": stimulus_hash,
                      "options": {"method": "meanDifference"}}],
        "outcomeInstruments": ["repeReaderScore", "sampledText"],
        "readerRefs": [{"path": rel, "hash": reader_hash, "concept": "fear"}],
    }
    return root, rel, reader_hash, manifest


def test_manifest_verify_accepts_clean_reader_pin(tmp_path):
    root, _, _, d = _reader_tree(tmp_path)
    assert Manifest.from_dict(d).verify(root) == []


def test_manifest_verify_rejects_foreign_substrate(tmp_path):
    root, _, _, d = _reader_tree(tmp_path, substrate="swift-mlx")
    violations = Manifest.from_dict(d).verify(root)
    assert any("swift-mlx" in v and "substrate-specific" in v for v in violations)


def test_manifest_verify_flags_drift_missing_refs_and_model(tmp_path):
    root, rel, _, d = _reader_tree(tmp_path)
    # Hash drift after pinning.
    with open(os.path.join(root, rel), "a", encoding="utf-8") as handle:
        handle.write("\n")
    violations = Manifest.from_dict(d).verify(root)
    assert any("reader 'fear'" in v and "changed since pinning" in v
               for v in violations)
    # Instrument requested but nothing pinned.
    d2 = dict(d, readerRefs=[])
    violations = Manifest.from_dict(d2).verify(root)
    assert any("no readerRefs" in v for v in violations)
    # Reader fitted on a different model than the study's.
    root3, _, _, d3 = _reader_tree(tmp_path / "m", model_id="org/other")
    violations = Manifest.from_dict(d3).verify(root3)
    assert any("org/other" in v and "study model" in v for v in violations)


# --- study integration: scorers + endpoint stamping ------------------------------

def test_reader_scorers_load_and_reject_foreign(tmp_path, monkeypatch):
    root, _, _, d = _reader_tree(tmp_path)
    scorers = tasks._reader_scorers(Manifest.from_dict(d), root)
    assert [concept for concept, _ in scorers] == ["fear"]
    assert scorers[0][1].layer == 0
    monkeypatch.setattr(repe_reader, "score_text",
                        lambda model, reader, text: 0.25)
    assert tasks._reader_scores(None, scorers, "any text") == {"fear": 0.25}

    root2, _, _, d2 = _reader_tree(tmp_path / "f", substrate="swift-mlx")
    with pytest.raises(RuntimeError, match="swift-mlx"):
        tasks._reader_scorers(Manifest.from_dict(d2), root2)


def test_endpoint_values_pick_up_reader_scores():
    records = [
        {"condition": "baseline", "promptID": "p1", "output": "a",
         "readerScores": {"fear": 1.0, "calm": -1.0}},
        {"condition": "baseline", "promptID": "p1", "output": "b",
         "readerScores": {"fear": 3.0}},
        {"condition": "steered", "promptID": "p1", "output": "c",
         "readerScores": {"fear": 5.0}},
        {"condition": "steered", "promptID": "p1", "error": "boom"},
    ]
    endpoints = tasks._endpoint_values(records)
    # Endpoint per concept, mean over the sample axis, paired by promptID.
    assert endpoints["readerScore:fear"]["baseline"]["p1"] == pytest.approx(2.0)
    assert endpoints["readerScore:fear"]["steered"]["p1"] == pytest.approx(5.0)
    assert endpoints["readerScore:calm"]["baseline"]["p1"] == pytest.approx(-1.0)


# --- derive-steering conversion ---------------------------------------------------

def test_derive_steering_vector_provenance(tmp_path):
    from steerlab_server.steering import vector_store
    reader = _manual_reader(layer=1)
    reader_path = repe_reader.save_reader(reader, str(tmp_path / "fit"))
    with open(reader_path, "rb") as handle:
        reader_hash = hashlib.sha256(handle.read()).hexdigest()

    out_dir = str(tmp_path / "derived")
    base = repe_reader.derive_steering_vector(reader_path, out_dir)
    name = os.path.basename(base)
    assert name == "fear-repe-reader"
    vectors, sidecar = vector_store.load(out_dir, name)
    # Unit reading direction at the reader's layer, zeros below (single-layer
    # import convention) — and honest provenance on the sidecar.
    assert vectors.per_layer[1] == pytest.approx([1.0, 0.0])
    assert vectors.per_layer[0] == [0.0, 0.0]
    assert sidecar.source == "repe-reader-lat"
    assert sidecar.readerID == os.path.basename(reader_path)
    assert sidecar.readerHash == reader_hash
    assert sidecar.controlMode == "reading-vector activation addition"
    assert sidecar.extractionMethod == "repeReaderLAT"
    assert sidecar.stimulusSetHash == reader.dataset_hash
    assert sidecar.modelID == "org/m" and sidecar.concept == "fear"


# --- catalog + API routes -----------------------------------------------------------

def test_catalog_list_readers(tmp_path):
    reader = _manual_reader()
    run_dir = os.path.join(str(tmp_path), "runs", "20260703T000000000-reader-fit")
    path = repe_reader.save_reader(reader, run_dir)
    # A steering-vector sidecar must NOT appear in the reader list.
    from steerlab_server.steering.vector_store import (
        ConceptVectors, SteeringVectorSidecar, save)
    v = ConceptVectors(per_layer=[[1.0, 0.0]])
    save(v, SteeringVectorSidecar.make(model_id="org/m", concept="x",
         stimulus_set_hash="h", vectors=v), run_dir, "x")
    readers = catalog.list_readers(str(tmp_path))
    assert len(readers) == 1
    summary = readers[0]
    assert summary.concept == "fear" and summary.layer == 0
    assert summary.substrate == "python-hf-transformers"
    assert summary.templateID == "unnamed-scenario-v1"
    assert summary.templateDivergence == "unnamed-clean-room"
    assert summary.heldOutAccuracy == pytest.approx(0.8)
    assert summary.id == path


pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from steerlab_server.api import app as app_module  # noqa: E402

client = TestClient(app_module.app)


def test_api_readers_lists_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    reader = _manual_reader()
    run_dir = os.path.join(str(tmp_path), "runs", "20260703T000000000-reader-fit")
    path = repe_reader.save_reader(reader, run_dir)
    body = client.get("/api/readers").json()
    assert len(body["readers"]) == 1
    assert body["readers"][0]["concept"] == "fear"
    assert body["readers"][0]["id"] == path
    assert body["readers"][0]["templateID"] == "unnamed-scenario-v1"


def test_api_reader_score_requires_model_and_valid_body(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    reader_path = repe_reader.save_reader(_manual_reader(), str(tmp_path / "runs" / "r"))
    monkeypatch.setattr(app_module.state, "model", None)
    resp = client.post("/api/reader/score",
                       json={"readerID": reader_path, "texts": ["x"]})
    assert resp.status_code == 409  # no model loaded

    fake = SimpleNamespace(model_id="org/m", revision=None)
    monkeypatch.setattr(app_module.state, "model", fake)
    assert client.post("/api/reader/score", json={"texts": ["x"]}).status_code == 400
    assert client.post("/api/reader/score",
                       json={"readerID": reader_path, "texts": []}).status_code == 400
    assert client.post("/api/reader/score",
                       json={"readerID": reader_path}).status_code == 400


def test_api_reader_score_scores_texts(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    reader_path = repe_reader.save_reader(_manual_reader(), str(tmp_path / "runs" / "r"))
    fake = SimpleNamespace(model_id="org/m", revision=None)
    monkeypatch.setattr(app_module.state, "model", fake)

    @contextmanager
    def fake_acquire():
        yield fake

    monkeypatch.setattr(app_module.state, "acquire_active", fake_acquire)
    # Monkeypatch the score internals (not the HTTP layer): pretend the LAT
    # capture returned a fixed activation at the reader's layer.
    monkeypatch.setattr(
        extractor, "activations",
        lambda model, texts, position: _FakeActs([[[3.0, 7.0]] for _ in texts]))
    resp = client.post("/api/reader/score",
                       json={"readerID": reader_path, "texts": ["a", "b"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["concept"] == "fear" and body["layer"] == 0
    assert body["templateID"] == "unnamed-scenario-v1"
    assert body["scores"] == pytest.approx([0.75, 0.75])

    # A foreign-substrate reader is refused with a clear 400, not scored.
    foreign_path = repe_reader.save_reader(
        _manual_reader(substrate="swift-mlx"), str(tmp_path / "runs" / "f"))
    resp = client.post("/api/reader/score",
                       json={"readerID": foreign_path, "texts": ["a"]})
    assert resp.status_code == 400
    assert "substrate-specific" in resp.json()["detail"]


def test_api_reader_score_rejects_wrong_model_reader(tmp_path, monkeypatch):
    # The loaded model is not the one the reader was fitted on → clear 400.
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    reader_path = repe_reader.save_reader(_manual_reader(),
                                          str(tmp_path / "runs" / "r"))
    fake = SimpleNamespace(model_id="org/other", revision=None)
    monkeypatch.setattr(app_module.state, "model", fake)

    @contextmanager
    def fake_acquire():
        yield fake

    monkeypatch.setattr(app_module.state, "acquire_active", fake_acquire)
    monkeypatch.setattr(
        extractor, "activations",
        lambda model, texts, position: _FakeActs([[[1.0, 0.0]] for _ in texts]))
    resp = client.post("/api/reader/score",
                       json={"readerID": reader_path, "texts": ["a"]})
    assert resp.status_code == 400
    assert "per-model measurement instrument" in resp.json()["detail"]


# --- POST /api/reader/fit (durable job) ---------------------------------------

VALID_TEMPLATE_JSON = json.dumps({
    "id": "custom-amount-v1", "conceptSlot": True, "latToken": "final",
    "text": AMOUNT_TEMPLATE_TEXT})


def _fit_rows(template_id="amount-in-scenario-v1"):
    return [_pair_row(0, "p0", "n0", template_id=template_id),
            _pair_row(1, "p1", "n1", template_id=template_id),
            _pair_row(2, "hp", "hn", split="test", template_id=template_id)]


def _fit_jsonl(template_id="amount-in-scenario-v1"):
    return "".join(json.dumps(r) + "\n" for r in _fit_rows(template_id))


def _wait_for_job(job_id, tries=200):
    import time
    job = None
    for _ in range(tries):  # daemon-thread job; poll briefly
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    return job


def test_api_reader_fit_body_validation(tmp_path, monkeypatch):
    # All of these fail synchronously, before any job (or model) is involved.
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    ok = {"concept": "fear", "templateJSON": VALID_TEMPLATE_JSON,
          "pairsJSONL": _fit_jsonl("custom-amount-v1")}
    # concept: required + safe-named (no traversal into prompts/readers/…).
    assert client.post("/api/reader/fit",
                       json={**ok, "concept": ""}).status_code == 400
    assert client.post("/api/reader/fit",
                       json={**ok, "concept": "../evil"}).status_code == 400
    # Exactly one template source.
    missing_template = {k: v for k, v in ok.items() if k != "templateJSON"}
    assert client.post("/api/reader/fit", json=missing_template).status_code == 400
    both = {**ok, "templateID": "amount-in-scenario-v1"}
    assert client.post("/api/reader/fit", json=both).status_code == 400
    # templateJSON must parse and carry a safe id.
    assert client.post("/api/reader/fit",
                       json={**ok, "templateJSON": "{nope"}).status_code == 400
    assert client.post("/api/reader/fit",
                       json={**ok, "templateJSON": '{"text": "x"}'}).status_code == 400
    assert client.post(
        "/api/reader/fit",
        json={**ok, "templateJSON": '{"id": "../t", "text": "x"}'}).status_code == 400
    # Exactly one pairs source.
    missing_pairs = {k: v for k, v in ok.items() if k != "pairsJSONL"}
    assert client.post("/api/reader/fit", json=missing_pairs).status_code == 400
    assert client.post("/api/reader/fit",
                       json={**ok, "pairsPath": "x.jsonl"}).status_code == 400
    # Layer list shape.
    assert client.post("/api/reader/fit",
                       json={**ok, "layers": "all"}).status_code == 400
    assert client.post("/api/reader/fit",
                       json={**ok, "layers": []}).status_code == 400
    assert client.post("/api/reader/fit",
                       json={**ok, "layers": [0.5]}).status_code == 400
    # outputName is a path component.
    assert client.post("/api/reader/fit",
                       json={**ok, "outputName": "a/b"}).status_code == 400
    # A valid body without a loaded model is the standard 409 (synchronous),
    # and nothing was written to the pinned pairs location by the validation.
    monkeypatch.setattr(app_module.state, "model", None)
    assert client.post("/api/reader/fit", json=ok).status_code == 409
    assert not os.path.exists(os.path.join(str(tmp_path), "prompts", "readers"))


def test_api_reader_fit_rejects_concept_mismatch_and_missing_pairs(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    fake = SimpleNamespace(model_id="org/m", revision="abc")
    monkeypatch.setattr(app_module.state, "model", fake)
    # pairs pinned to a different concept than the request names → 400.
    resp = client.post("/api/reader/fit", json={
        "concept": "calm", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": _fit_jsonl("custom-amount-v1")})
    assert resp.status_code == 400
    assert "'fear'" in resp.json()["detail"]
    # pairsPath must resolve to an existing contained file.
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsPath": "prompts/readers/fear/pairs.jsonl"})
    assert resp.status_code == 404
    # unknown registry template id → 400 (missing template file).
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateID": "no-such-template-v9",
        "pairsJSONL": _fit_jsonl()})
    assert resp.status_code == 400


def _fit_route_harness(tmp_path, monkeypatch):
    """Fake loaded model + fake activation capture (same fixtures as the fit
    math tests) so the durable job runs end-to-end without a GPU. The fit route
    pins its model at submission and acquires it by id via the registry path,
    so the harness fakes ``acquire_model`` and records the (id, revision)
    acquire calls for assertions."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    monkeypatch.setattr(extractor, "activations", _fake_activations)
    fake = SimpleNamespace(model_id="org/m", revision="abc")
    monkeypatch.setattr(app_module.state, "model", fake)
    acquired = []

    @contextmanager
    def fake_acquire_model(model_id, revision=None):
        acquired.append((model_id, revision))
        yield SimpleNamespace(model_id=model_id, revision=revision)

    monkeypatch.setattr(app_module.state, "acquire_model", fake_acquire_model)
    return acquired


def test_api_reader_fit_job_with_registry_template(tmp_path, monkeypatch):
    _fit_route_harness(tmp_path, monkeypatch)
    _write_template(os.path.join(str(tmp_path), "prompts", "templates"))
    pairs_jsonl = _fit_jsonl()
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateID": "amount-in-scenario-v1",
        "pairsJSONL": pairs_jsonl})
    assert resp.status_code == 200
    job = _wait_for_job(resp.json()["jobId"])
    assert job is not None and job["status"] == "succeeded", job

    # Uploaded pairs land at the canonical pinned location (contained under
    # prompts/readers/<concept>/), byte-identical to the upload.
    pinned = os.path.join(str(tmp_path), "prompts", "readers", "fear", "pairs.jsonl")
    assert os.path.isfile(pinned)
    with open(pinned, encoding="utf-8") as handle:
        assert handle.read() == pairs_jsonl

    result = job["result"]
    run_dir = result["runDirectory"]
    assert os.path.basename(run_dir).endswith("-api-reader-fit-reader-fear")
    assert os.path.realpath(run_dir).startswith(os.path.realpath(str(tmp_path)))
    # One artifact per layer (fake capture has one layer), fit on the request's
    # model with the fit-math fixtures' known quality numbers.
    assert len(result["artifacts"]) == 1
    loaded = repe_reader.load_reader(result["artifacts"][0])
    assert loaded.concept == "fear" and loaded.layer == 0
    assert loaded.model_id == "org/m" and loaded.revision == "abc"
    assert loaded.dataset_hash == hashlib.sha256(
        pairs_jsonl.encode("utf-8")).hexdigest()
    assert result["layerScores"] == [{
        "layer": 0, "trainAccuracy": 1.0, "heldOutAccuracy": 0.5,
        "pc1ExplainedVariance": pytest.approx(1.0)}]
    # The run-dir naming is discovered by the catalog (pattern-agnostic scan).
    listed = client.get("/api/readers").json()["readers"]
    assert any(r["runDirectory"] == run_dir for r in listed)


def test_api_reader_fit_rejected_upload_leaves_no_pairs_file(tmp_path, monkeypatch):
    # Validation happens fully in memory BEFORE the canonical write: a
    # malformed or wrong-concept upload must leave nothing behind under the
    # pinned prompts/readers/<concept>/ location.
    _fit_route_harness(tmp_path, monkeypatch)
    readers_root = os.path.join(str(tmp_path), "prompts", "readers")

    # Malformed JSONL (broken JSON line).
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": '{"nope'})
    assert resp.status_code == 400
    assert not os.path.exists(readers_root)

    # Structurally valid rows missing required keys.
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": '{"positiveStimulus": "p", "concept": "fear"}\n'})
    assert resp.status_code == 400
    assert not os.path.exists(readers_root)

    # Wrong concept: rows are for 'fear' but the request names 'calm'.
    resp = client.post("/api/reader/fit", json={
        "concept": "calm", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": _fit_jsonl("custom-amount-v1")})
    assert resp.status_code == 400
    assert not os.path.exists(readers_root)

    # Rows pinning a different template than the fit uses → synchronous 400,
    # nothing written.
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": _fit_jsonl("other-template-v1")})
    assert resp.status_code == 400
    assert "pin template" in resp.json()["detail"]
    assert not os.path.exists(readers_root)


def test_api_reader_fit_rejected_upload_does_not_clobber_good_pairs(tmp_path, monkeypatch):
    _fit_route_harness(tmp_path, monkeypatch)
    # A good corpus already pinned for 'calm'.
    good_rows = [_pair_row(0, "p0", "n0", concept="calm", template_id="custom-amount-v1"),
                 _pair_row(1, "p1", "n1", concept="calm", template_id="custom-amount-v1")]
    good = "".join(json.dumps(r) + "\n" for r in good_rows)
    pinned = os.path.join(str(tmp_path), "prompts", "readers", "calm", "pairs.jsonl")
    os.makedirs(os.path.dirname(pinned))
    with open(pinned, "w", encoding="utf-8") as handle:
        handle.write(good)

    # A wrong-concept upload naming 'calm' (rows are 'fear') must not clobber it.
    resp = client.post("/api/reader/fit", json={
        "concept": "calm", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": _fit_jsonl("custom-amount-v1")})
    assert resp.status_code == 400
    with open(pinned, encoding="utf-8") as handle:
        assert handle.read() == good

    # Neither must a malformed one.
    resp = client.post("/api/reader/fit", json={
        "concept": "calm", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": "not json"})
    assert resp.status_code == 400
    with open(pinned, encoding="utf-8") as handle:
        assert handle.read() == good
    # And no stray temp files linger next to the pinned corpus.
    assert os.listdir(os.path.dirname(pinned)) == ["pairs.jsonl"]


def test_api_reader_fit_snapshots_active_model_at_submission(tmp_path, monkeypatch):
    # No explicit modelID: the ACTIVE model's id/revision are snapshotted
    # synchronously at submission, and the job acquires exactly that snapshot —
    # never "whatever is active later".
    acquired = _fit_route_harness(tmp_path, monkeypatch)
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": _fit_jsonl("custom-amount-v1")})
    assert resp.status_code == 200
    # Unloading / swapping the UI model after submission cannot change what the
    # queued job fits on: the closure captured the snapshot, not state.model.
    monkeypatch.setattr(app_module.state, "model", None)
    job = _wait_for_job(resp.json()["jobId"])
    assert job is not None and job["status"] == "succeeded", job
    assert acquired == [("org/m", "abc")]
    loaded = repe_reader.load_reader(job["result"]["artifacts"][0])
    assert loaded.model_id == "org/m" and loaded.revision == "abc"


def test_api_reader_fit_with_explicit_model_id(tmp_path, monkeypatch):
    # An explicit modelID (+ revision) pins the fit regardless of the loaded
    # model — and works with NO model loaded at all (no 409): the job acquires
    # the named model via the registry path.
    acquired = _fit_route_harness(tmp_path, monkeypatch)
    monkeypatch.setattr(app_module.state, "model", None)
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": _fit_jsonl("custom-amount-v1"),
        "modelID": "org/pinned", "revision": "r9"})
    assert resp.status_code == 200
    job = _wait_for_job(resp.json()["jobId"])
    assert job is not None and job["status"] == "succeeded", job
    assert acquired == [("org/pinned", "r9")]
    loaded = repe_reader.load_reader(job["result"]["artifacts"][0])
    assert loaded.model_id == "org/pinned" and loaded.revision == "r9"


def test_api_reader_fit_job_with_custom_template_json(tmp_path, monkeypatch):
    _fit_route_harness(tmp_path, monkeypatch)
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateJSON": VALID_TEMPLATE_JSON,
        "pairsJSONL": _fit_jsonl("custom-amount-v1"),
        "layers": [0], "outputName": "fear-custom"})
    assert resp.status_code == 200
    job = _wait_for_job(resp.json()["jobId"])
    assert job is not None and job["status"] == "succeeded", job
    run_dir = job["result"]["runDirectory"]
    assert "api-reader-fit-fear-custom" in os.path.basename(run_dir)
    # The one-off template is persisted as raw bytes in the run directory and
    # the artifact's templateHash pins those bytes (hash-over-raw-bytes).
    custom_path = os.path.join(run_dir, "custom-amount-v1.json")
    with open(custom_path, "rb") as handle:
        raw = handle.read()
    assert raw == VALID_TEMPLATE_JSON.encode("utf-8")
    loaded = repe_reader.load_reader(job["result"]["artifacts"][0])
    assert loaded.template.id == "custom-amount-v1"
    assert loaded.template.hash == hashlib.sha256(raw).hexdigest()

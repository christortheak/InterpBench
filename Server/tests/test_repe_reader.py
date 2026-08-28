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
    READER_CHAT_TEMPLATE_RENDERING_CONVENTION, READER_RENDERING_CONVENTION,
    render_reader)
from steerlab_server.steering import extractor, repe_reader, vector_math
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


def _fake_activations(model, texts, position, rendering=None):
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
    # The two train differences are identical after normalization, so the
    # DIFFERENCE CLOUD has no variance to apportion — absent, not 0 (which
    # would read as "PC1 explains nothing"). The pre-2026-08-27 number here
    # was 1.0, measured over the ± alternated copies PCA is fitted on.
    assert reader.difference_cloud_explained_variance is None
    assert reader.explained_variance_basis == "degenerateDifferenceCloud"
    # The held-out pair (hp − hn = [0.5, 0]) projects positive on +x, so the
    # HELD-OUT split fixes the sign — the paper's step 4, not get_signs.
    # …but ONE held-out pair is below the minimum that may decide a sign, so
    # the fit falls back to the reference implementation's train-label majority
    # and says so out loud. See test_held_out_split_fixes_the_sign for the
    # branch where the held-out split does decide.
    assert reader.sign_convention == repe_reader.TRAIN_MAJORITY
    assert reader.sign_held_out_accuracy is None
    assert "below the minimum 2" in reader.sign_fallback_reason
    assert reader.contrast_mode == repe_reader.SUPERVISED_CONTENT
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
    # The legacy `pc1ExplainedVariance` key is gone: its basis changed, and
    # writing the old key with the new semantics would make every pre-existing
    # consumer silently wrong instead of visibly out of date.
    assert "pc1ExplainedVariance" not in d
    assert d["pc1ExplainedVarianceBasis"] == "degenerateDifferenceCloud"
    assert d["contrastMode"] == "supervisedContent"
    assert d["signConvention"] == "trainMajority"

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
        difference_cloud_explained_variance=0.9, train_accuracy=1.0,
        held_out_accuracy=0.8,
        train_pair_count=4, held_out_pair_count=2, substrate=substrate)


def test_score_text_exact_inference(monkeypatch):
    reader = _manual_reader()
    seen = {}

    def fake_activations(model, texts, position, rendering=None):
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
        lambda model, texts, position, rendering=None: _FakeActs([[[1.0, 0.0]] for _ in texts]))
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
        lambda model, texts, position, rendering=None: _FakeActs([[[1.0, 0.0]] for _ in texts]))
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
        lambda model, texts, position, rendering=None: _FakeActs([[[3.0, 7.0]] for _ in texts]))
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
        lambda model, texts, position, rendering=None: _FakeActs([[[1.0, 0.0]] for _ in texts]))
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
        "signConvention": "trainMajority", "signHeldOutAccuracy": None,
        "pc1ExplainedVarianceOfDifferences": None,
        "pc1ExplainedVarianceBasis": "degenerateDifferenceCloud"}]
    assert result["contrastMode"] == "supervisedContent"
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


def test_api_reader_fit_refuses_an_unrenderable_custom_template_synchronously(
        tmp_path, monkeypatch):
    """Review round 6, finding 4. The reviewer's minimal template — valid JSON
    with an id and a text, and nothing a fit can do with it — used to pass the
    synchronous checks, replace the canonical corpus, and fail from a job."""
    _fit_route_harness(tmp_path, monkeypatch)
    good_rows = [_pair_row(0, "p0", "n0", concept="calm", template_id="x"),
                 _pair_row(1, "p1", "n1", concept="calm", template_id="x")]
    good = "".join(json.dumps(r) + "\n" for r in good_rows)
    pinned = os.path.join(str(tmp_path), "prompts", "readers", "calm", "pairs.jsonl")
    os.makedirs(os.path.dirname(pinned))
    with open(pinned, "wb") as handle:
        handle.write(good.encode("utf-8"))
    before = open(pinned, "rb").read()

    resp = client.post("/api/reader/fit", json={
        "concept": "calm", "templateJSON": json.dumps({"id": "x", "text": "x"}),
        "pairsJSONL": good})
    assert resp.status_code == 400
    assert "{{stimulus}}" in resp.json()["detail"]
    assert open(pinned, "rb").read() == before
    assert os.listdir(os.path.dirname(pinned)) == ["pairs.jsonl"]


@pytest.mark.parametrize("template,expected", [
    ({"id": "x", "text": "S: {{stimulus}}", "latToken": "penultimate"},
     "unsupported latToken"),
    ({"id": "x", "text": "S: {{stimulus}} {{instruction}}"},
     "declares no instructionPair"),
    ({"id": "x", "text": "S: {{stimulus}}",
      "instructionPair": {"experimental": "a", "reference": "b"}},
     "has no {{instruction}} slot"),
    ({"id": "x", "text": "S: {{stimulus}} {{concept}}"},
     "conceptSlot=false"),
    ({"id": "x", "text": "<bos>S: {{stimulus}}"},
     "special/chat-template marker"),
])
def test_api_reader_fit_refuses_each_unrenderable_template_shape(
        tmp_path, monkeypatch, template, expected):
    """Every rule the queued job would have applied, applied synchronously and
    typed — parse, slot coherence, LAT vocabulary, scaffold marker hygiene."""
    _fit_route_harness(tmp_path, monkeypatch)
    rows = [_pair_row(0, "p0", "n0", template_id="x"),
            _pair_row(1, "p1", "n1", template_id="x")]
    resp = client.post("/api/reader/fit", json={
        "concept": "fear", "templateJSON": json.dumps(template),
        "pairsJSONL": "".join(json.dumps(r) + "\n" for r in rows)})
    assert resp.status_code == 400
    assert expected in resp.json()["detail"]
    assert not os.path.exists(os.path.join(str(tmp_path), "prompts", "readers"))


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


# --- cross-engine twin literals for the new refusals -------------------------
#
# The twin-literal idiom: this table and its Swift counterpart
# (`RepEReaderTests.newRefusalsAreCrossEngineTwinLiterals`) are two INDEPENDENT
# copies of the same words. Neither engine can reword a refusal — or quietly
# drop the repair off the end of one — without the other engine's suite going
# red. A refusal a researcher meets on the Mac and on the cluster has to be the
# same instruction, or "follow the repair" is advice about one machine.

REFUSAL_TWINS = {
    "chatTemplateMarker":
        "reader scaffold embeds turn marker '<start_of_turn>' inside the user "
        "turn's content — under a chatTemplate rendering the template supplies "
        "the markers, so an embedded one forges a turn boundary and moves the "
        "LAT token off the generation prompt. Repair: write the scaffold as "
        "plain content and let the template do the framing",
    "mixedShapes":
        "S: mixes content-pair rows (positiveStimulus/negativeStimulus) with "
        "template-pair rows (stimulus) — the two produce different "
        "differences, so one file cannot mean both. Repair: split them into "
        "two datasets",
    "bothShapes":
        "S: row declares both 'stimulus' and a positive/negative pair — a "
        "template-pair row holds ONE stimulus (the T+/T− instructions carry "
        "the contrast). Repair: drop 'stimulus' for a content pair, or drop "
        "'positiveStimulus'/'negativeStimulus' for a template pair",
    "contentPairsUnderTemplatePair":
        "template 'tp' declares a T+/T− instructionPair but the dataset holds "
        "content pairs (positiveStimulus/negativeStimulus) — under a template "
        "pair the contrast is the INSTRUCTION and a second stimulus would be a "
        "confound. Repair: fit these pairs through a single-template reader "
        "template, or rewrite the dataset as one-stimulus ('stimulus') rows",
    "singleStimulusUnderPlainTemplate":
        "the dataset holds one-stimulus ('stimulus') rows but template 'pl' "
        "declares no instructionPair — there is nothing to contrast the "
        "stimulus against. Repair: choose a template-pair template (one with "
        "'instructionPair'), or rewrite the dataset as "
        "positiveStimulus/negativeStimulus content pairs",
    "instructionPairWithoutSlot":
        "template 'x' declares an instructionPair but its text has no "
        "{{instruction}} slot — the T+/T− instructions would never reach the "
        "model. Repair: add {{instruction}} to the text, or drop "
        "instructionPair to make this a single-template reader",
    "slotWithoutInstructionPair":
        "template 'x' has a {{instruction}} slot but declares no "
        "instructionPair — nothing would fill it. Repair: add an "
        "instructionPair with 'experimental' and 'reference', or remove the "
        "slot",
    "emptyInstruction":
        "template 'x': instructionPair needs both 'experimental' (T+) and "
        "'reference' (T−) — one empty instruction makes the contrast a "
        "rendering artifact. Repair: write both instructions",
    "identicalInstructions":
        "template 'x': instructionPair's experimental and reference "
        "instructions are identical — every difference would be exactly zero. "
        "Repair: write two instructions that differ in the quality under study",
    "signFallbackNoHeldOut":
        "no held-out pairs (every row is split 'train'): the sign follows "
        "train-label majority, the reference implementation's get_signs. "
        "Repair: mark some rows with a non-'train' split so the paper's "
        "held-out sign selection can run",
    "signFallbackTooFew":
        "1 held-out pair(s) projected off zero, below the minimum 2: a "
        "one-pair vote is a coin flip wearing a validation split's authority, "
        "so the sign follows train-label majority instead",
    "signFallbackTied":
        "held-out pairs split evenly (2 for, 2 against): the held-out set does "
        "not discriminate at this layer, so the sign follows train-label "
        "majority. Read this layer's heldOutAccuracy before trusting its "
        "direction",
}


def _produced_refusals() -> dict[str, str]:
    """Every new refusal, produced by the engine rather than transcribed."""
    from steerlab_server.steering.extraction_rendering import ExtractionRendering

    out: dict[str, str] = {}

    def capture(label, fn):
        try:
            fn()
        except (ValueError, repe_reader.RepeReaderError) as exc:
            out[label] = str(exc)
        else:
            pytest.fail(f"{label}: expected a refusal")

    capture("chatTemplateMarker", lambda: render_reader(
        "a <start_of_turn> b", model_id="m",
        rendering=ExtractionRendering(mode="chatTemplate")))
    capture("mixedShapes", lambda: repe_reader.parse_pairs(
        '{"concept":"c","positiveStimulus":"p","negativeStimulus":"n",'
        '"templateID":"t"}\n{"concept":"c","stimulus":"s","templateID":"t"}\n',
        source="S"))
    capture("bothShapes", lambda: repe_reader.parse_pairs(
        '{"concept":"c","stimulus":"s","positiveStimulus":"p",'
        '"templateID":"t"}\n', source="S"))

    pair_template = repe_reader.TaskTemplate(
        id="tp", text="{{instruction}} S: {{stimulus}}", concept_slot=False,
        lat_token="final", hash="h",
        instruction_pair=repe_reader.InstructionPair("a", "b"))
    plain_template = repe_reader.TaskTemplate(
        id="pl", text="S: {{stimulus}}", concept_slot=False, lat_token="final",
        hash="h")
    content_pairs = repe_reader.ReaderDataset(
        concept="c", pairs=(repe_reader.ReaderPair("p", "n", "c", "tp"),), hash="h")
    single = repe_reader.ReaderDataset(
        concept="c",
        pairs=(repe_reader.ReaderPair("", "", "c", "pl", stimulus="s"),), hash="h")
    capture("contentPairsUnderTemplatePair",
            lambda: repe_reader.resolve_contrast_mode(content_pairs, pair_template))
    capture("singleStimulusUnderPlainTemplate",
            lambda: repe_reader.resolve_contrast_mode(single, plain_template))

    def template(text, pair):
        return repe_reader.TaskTemplate(
            id="x", text=text, concept_slot=False, lat_token="final", hash="h",
            instruction_pair=pair)

    capture("instructionPairWithoutSlot", lambda: template(
        "S: {{stimulus}}",
        repe_reader.InstructionPair("a", "b")).validate_instruction_slot())
    capture("slotWithoutInstructionPair", lambda: template(
        "{{instruction}} S: {{stimulus}}", None).validate_instruction_slot())
    capture("emptyInstruction", lambda: template(
        "{{instruction}} S: {{stimulus}}",
        repe_reader.InstructionPair("a", "")).validate_instruction_slot())
    capture("identicalInstructions", lambda: template(
        "{{instruction}} S: {{stimulus}}",
        repe_reader.InstructionPair("same", "same")).validate_instruction_slot())

    out["signFallbackNoHeldOut"] = repe_reader.held_out_sign_fallback_reason(
        held_out_pair_count=0, decided=0, agree=0, disagree=0)
    out["signFallbackTooFew"] = repe_reader.held_out_sign_fallback_reason(
        held_out_pair_count=1, decided=1, agree=1, disagree=0)
    out["signFallbackTied"] = repe_reader.held_out_sign_fallback_reason(
        held_out_pair_count=4, decided=4, agree=2, disagree=2)
    return out


def test_new_refusals_are_cross_engine_twin_literals():
    produced = _produced_refusals()
    assert set(produced) == set(REFUSAL_TWINS), \
        "a new refusal must be added to BOTH engines' twin-literal tables"
    for label, expected in REFUSAL_TWINS.items():
        assert produced[label] == expected, label


# --- paper step 4: held-out sign and layer selection --------------------------

def _template(tmp_path):
    return repe_reader.load_template(_write_template(str(tmp_path)))


def _fit_from_rows(tmp_path, rows, acts, layers=None, **kwargs):
    """Fit purely over supplied activations — no monkeypatching, no model."""
    template = _template(tmp_path)
    dataset = repe_reader.load_pairs(
        _write_pairs(str(tmp_path / "rows.jsonl"), rows))
    captured = []
    for pair in dataset.train + dataset.held_out:
        captured.append(acts[pair.positive_stimulus])
        captured.append(acts[pair.negative_stimulus])
    return repe_reader.fit_activations(
        dataset, template, captured, model_id="org/m", revision="abc",
        layers=layers, **kwargs)


def test_held_out_split_fixes_the_sign(tmp_path):
    """Two held-out pairs pointing the OPPOSITE way from the train majority:
    the held-out split wins, which is the paper's rule and not get_signs'."""
    acts = {
        "p0": [[2.0, 0.5]], "n0": [[0.0, 0.5]],
        "p1": [[2.0, -0.5]], "n1": [[0.0, -0.5]],
        "hp0": [[0.0, 0.2]], "hn0": [[1.0, 0.2]],
        "hp1": [[0.1, -0.2]], "hn1": [[1.1, -0.2]],
    }
    rows = [_pair_row(0, "p0", "n0"), _pair_row(1, "p1", "n1"),
            _pair_row(2, "hp0", "hn0", split="test"),
            _pair_row(3, "hp1", "hn1", split="test")]
    reader = _fit_from_rows(tmp_path, rows, acts)[0]
    assert reader.sign_convention == repe_reader.HELD_OUT_PAIR_AGREEMENT
    assert reader.sign_held_out_accuracy == pytest.approx(1.0)
    assert reader.sign_fallback_reason is None
    # Train majority alone would have chosen +x; the held-out split flips it.
    assert reader.probe.direction == pytest.approx([-1.0, 0.0], abs=1e-5)


def test_evenly_split_held_out_falls_back_loudly(tmp_path):
    acts = {
        "p0": [[2.0, 0.5]], "n0": [[0.0, 0.5]],
        "p1": [[2.0, -0.5]], "n1": [[0.0, -0.5]],
        "ha": [[1.0, 0.0]], "hb": [[0.0, 0.0]],
        "hc": [[0.0, 0.1]], "hd": [[1.0, 0.1]],
    }
    rows = [_pair_row(0, "p0", "n0"), _pair_row(1, "p1", "n1"),
            _pair_row(2, "ha", "hb", split="test"),
            _pair_row(3, "hc", "hd", split="test")]
    reader = _fit_from_rows(tmp_path, rows, acts)[0]
    assert reader.sign_convention == repe_reader.TRAIN_MAJORITY
    assert reader.sign_held_out_accuracy is None
    assert "split evenly" in reader.sign_fallback_reason
    assert reader.probe.direction == pytest.approx([1.0, 0.0], abs=1e-5)


def test_layer_recommendation_is_stamped_on_the_whole_set(tmp_path):
    def row(x0, x1):
        return [[x0, 0.5], [x1, 0.5]]

    acts = {"p0": row(2, 1), "n0": row(0, 0),
            "p1": row(3, 2), "n1": row(0.5, 0.5),
            "hp": row(2, 0), "hn": row(0, 3)}
    rows = [_pair_row(0, "p0", "n0"), _pair_row(1, "p1", "n1"),
            _pair_row(2, "hp", "hn", split="test")]
    readers = _fit_from_rows(tmp_path, rows, acts)
    assert len(readers) == 2
    # Every artifact of the set carries the SAME recommendation — a reader
    # opened alone must not have to re-derive it from its siblings.
    assert len({r.recommended_layer for r in readers}) == 1
    assert all(r.layer_recommendation_basis == "heldOutAccuracy" for r in readers)
    best = readers[0].recommended_layer
    by_layer = {r.layer: r for r in readers}
    assert by_layer[best].held_out_accuracy == max(
        r.held_out_accuracy for r in readers)
    d = readers[0].to_dict()
    assert d["recommendedLayer"] == best
    assert "never selected automatically" in d["layerRecommendationNote"]


# --- paper step 1b: T+/T- template pairs --------------------------------------

STANCE_PAIR = {
    "id": "instructed-stance-pair-v1", "conceptSlot": False,
    "text": "{{instruction}}\nScenario: {{stimulus}}\nThe described state is",
    "latToken": "final",
    "instructionPair": {"experimental": "T-plus.", "reference": "T-minus."},
}


def test_committed_template_pair_template_loads_and_renders_both(tmp_path):
    path = os.path.join(REPO_TEMPLATES, "instructed-stance-pair-v1.json")
    template = repe_reader.load_template(path)
    assert template.is_template_pair
    pair = template.instruction_pair
    assert pair.experimental != pair.reference
    # Hygiene: the shipped example never names a concept, so a reader fitted
    # through it cannot become a concept-word detector.
    assert not template.concept_slot
    assert "synthetic-neutral" in (template.divergence or "")
    plus = template.render(stimulus="the room went quiet",
                           instruction=pair.experimental)
    minus = template.render(stimulus="the room went quiet",
                            instruction=pair.reference)
    assert plus != minus
    assert plus.endswith("The described state is")
    assert "{{" not in plus


def test_unsupervised_template_pair_fit_is_seeded_and_stamped(tmp_path):
    path = os.path.join(str(tmp_path), "instructed-stance-pair-v1.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(STANCE_PAIR, handle)
    template = repe_reader.load_template(path)
    rows = []
    captured = []
    for index in range(5):
        rows.append({"id": f"fear-row-{index}", "concept": "fear",
                     "stimulus": f"s{index}", "templateID": template.id,
                     "split": "train" if index < 3 else "test"})
        magnitude = float(index + 1)
        captured.append([[magnitude, 0.1 * magnitude]])   # T+
        captured.append([[0.0, 0.1 * magnitude]])         # T-
    dataset = repe_reader.load_pairs(
        _write_pairs(str(tmp_path / "single.jsonl"), rows))
    assert dataset.shape == "singleStimulus"
    reader = repe_reader.fit_activations(
        dataset, template, captured, model_id="org/m", revision="abc")[0]
    assert reader.contrast_mode == repe_reader.UNSUPERVISED_TEMPLATE_PAIR
    assert reader.orientation_seed == repe_reader.DEFAULT_ORIENTATION_SEED
    assert reader.sign_convention == repe_reader.HELD_OUT_PAIR_AGREEMENT
    assert abs(reader.probe.direction[0]) > 0.99
    assert repe_reader.score_activation(reader, [5.0, 0.0]) > 0
    # Explained variance now MEANS the difference cloud's, and this cloud has
    # variance (unlike the supervised fixture's identical rows).
    assert 0 < reader.difference_cloud_explained_variance <= 1 + 1e-5
    assert reader.explained_variance_basis == "differenceCloud"

    # Both engines render the SAME two texts, in the same order.
    texts = repe_reader.fit_texts(dataset, template, model_id="org/m")
    assert texts[0].startswith("T-plus.")
    assert texts[1].startswith("T-minus.")
    assert len(texts) == 10

    again = repe_reader.fit_activations(
        dataset, template, captured, model_id="org/m", revision="abc",
        orientation_seed=repe_reader.DEFAULT_ORIENTATION_SEED)[0]
    assert again.probe.direction == pytest.approx(reader.probe.direction)
    other = repe_reader.fit_activations(
        dataset, template, captured, model_id="org/m", revision="abc",
        orientation_seed=7)[0]
    assert other.orientation_seed == 7


def test_template_pair_scores_carry_the_documented_positive_offset(tmp_path):
    """F4 (2026-08-28 audit): a template-pair reader's probe center is the
    MIDPOINT of the T+ and T− train projections while inference renders T+
    only, so a T+-rendered score is offset upward by exactly
    ``|pos_mean − neg_mean| / (2·projectionScale)``.

    The docstring of ``score_texts`` used to claim the opposite — that T+ is
    "the rendering the probe's center and scale were calibrated on". This test
    pins the real arithmetic, so the corrected prose cannot drift back: the
    number is a RELATIVE endpoint, and ``score > 0`` is not concept presence.
    """
    path = os.path.join(str(tmp_path), "instructed-stance-pair-v1.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(STANCE_PAIR, handle)
    template = repe_reader.load_template(path)
    rows, captured = [], []
    for index in range(5):
        rows.append({"id": f"fear-row-{index}", "concept": "fear",
                     "stimulus": f"s{index}", "templateID": template.id,
                     "split": "train" if index < 3 else "test"})
        magnitude = float(index + 1)
        captured.append([[magnitude, 0.1 * magnitude]])   # T+
        captured.append([[0.0, 0.1 * magnitude]])         # T-
    dataset = repe_reader.load_pairs(
        _write_pairs(str(tmp_path / "single.jsonl"), rows))
    reader = repe_reader.fit_activations(
        dataset, template, captured, model_id="org/m", revision="abc")[0]
    assert reader.contrast_mode == repe_reader.UNSUPERVISED_TEMPLATE_PAIR

    probe = reader.probe
    # The center really is the midpoint of the two RENDERINGS' train means.
    assert probe.projection_center == pytest.approx(
        (probe.positive_mean + probe.negative_mean) / 2, abs=1e-6)

    # An activation projecting exactly at the T+ class mean — the middle of the
    # distribution inference actually draws from — scores at the offset, not 0.
    offset = abs(probe.positive_mean - probe.negative_mean) / (
        2 * probe.projection_scale)
    assert offset > 0.4
    center = probe.activation_center
    direction = probe.direction
    scale = sum(d * d for d in direction)
    at_pos_mean = [c + direction[i] * (probe.positive_mean / scale)
                   for i, c in enumerate(center)]
    assert repe_reader.score_activation(reader, at_pos_mean) == pytest.approx(
        probe.orientation * offset, abs=1e-4)

    # ...and the T− rendering of the SAME stimuli — the neutral pole of the
    # contrast — is the thing that lands at −offset, not at "negative because
    # the concept is absent". Zero is nobody's boundary at inference time.
    at_neg_mean = [c + direction[i] * (probe.negative_mean / scale)
                   for i, c in enumerate(center)]
    assert repe_reader.score_activation(reader, at_neg_mean) == pytest.approx(
        -probe.orientation * offset, abs=1e-4)


def test_orientation_signs_are_deterministic():
    a = repe_reader.orientation_signs(16, 231_001_405)
    assert a == repe_reader.orientation_signs(16, 231_001_405)
    assert set(a) == {1.0, -1.0}
    assert repe_reader.orientation_signs(16, 1) != a


# --- declarable rendering ------------------------------------------------------

def test_raw_rendering_remains_the_default_and_the_stamp_is_absent(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(extractor, "activations", _fake_activations)
    dataset, template = _fit_dataset(tmp_path)
    reader = repe_reader.fit(FAKE_MODEL, dataset, template)[0]
    assert reader.extraction_rendering is None
    assert reader.rendering_convention == READER_RENDERING_CONVENTION
    assert reader.resolved_extraction_rendering.is_raw
    # Absent, not {"mode": "raw"} — a raw fit's bytes stay what they were.
    assert "extractionRendering" not in reader.to_dict()


def test_chat_template_rendering_is_stamped_and_threaded(tmp_path, monkeypatch):
    from steerlab_server.steering.extraction_rendering import ExtractionRendering

    seen = {}

    def capture(model, texts, position, rendering=None):
        seen["rendering"] = rendering
        return _fake_activations(model, texts, position, rendering)

    monkeypatch.setattr(extractor, "activations", capture)
    dataset, template = _fit_dataset(tmp_path)
    rendering = ExtractionRendering(mode="chatTemplate")
    reader = repe_reader.fit(FAKE_MODEL, dataset, template,
                             extraction_rendering=rendering)[0]
    # The fit passes the declaration THROUGH to the extraction path…
    assert seen["rendering"] is rendering
    # …stamps it…
    assert reader.extraction_rendering == rendering.to_dict()
    assert reader.rendering_convention == \
        READER_CHAT_TEMPLATE_RENDERING_CONVENTION
    # …and scoring resolves the SAME rendering off the artifact.
    assert not reader.resolved_extraction_rendering.is_raw
    round_tripped = repe_reader.ReaderArtifact.from_dict(reader.to_dict())
    assert not round_tripped.resolved_extraction_rendering.is_raw


def test_marker_guard_reason_follows_the_rendering():
    from steerlab_server.steering.extraction_rendering import ExtractionRendering

    with pytest.raises(ValueError, match="double-BOS"):
        render_reader("a <bos> b", model_id="org/m")
    with pytest.raises(ValueError, match="forges a turn boundary"):
        render_reader("a <start_of_turn> b", model_id="org/m",
                      rendering=ExtractionRendering(mode="chatTemplate"))
    with pytest.raises(ValueError, match="manual '<s>' BOS"):
        render_reader("<s> hello", model_id="org/m")
    # Under a chat template a leading "<s>" in CONTENT is ordinary text.
    assert render_reader("<s> hello", model_id="org/m",
                         rendering=ExtractionRendering(mode="chatTemplate")) \
        == "<s> hello"


# --- legacy artifacts stay decodable -------------------------------------------

def test_legacy_artifact_decodes_with_stamped_legacy_semantics():
    legacy = {
        "artifactType": "repe-reader-lat", "schemaVersion": 1,
        "modelID": "org/m", "revision": "abc",
        "substrate": "python-hf-transformers", "concept": "fear", "layer": 3,
        "templateID": "unnamed-scenario-v1", "templateHash": "th",
        "template": {"conceptSlot": False, "hash": "th",
                     "id": "unnamed-scenario-v1", "latToken": "final",
                     "text": "S: {{stimulus}} q"},
        "datasetHash": "dh", "latTokenPosition": "final",
        "readingPosition": "last token",
        "probe": {"direction": [1.0, 0.0], "projectionCenter": 0.5,
                  "projectionScale": 2.0, "orientation": 1.0,
                  "positiveMean": 1.0, "negativeMean": -1.0},
        "pc1ExplainedVariance": 0.87, "trainAccuracy": 1.0,
        "heldOutAccuracy": 0.8, "trainPairCount": 4, "heldOutPairCount": 2,
        "renderingConvention": "rawCompletion scaffold",
        "extractionDate": "2026-07-03T00:00:00Z",
    }
    reader = repe_reader.ReaderArtifact.from_dict(legacy)
    assert reader.difference_cloud_explained_variance == pytest.approx(0.87)
    assert reader.explained_variance_basis == "alternatedRows"
    assert reader.contrast_mode == repe_reader.SUPERVISED_CONTENT
    assert reader.sign_convention == repe_reader.TRAIN_MAJORITY
    assert reader.sign_held_out_accuracy is None
    assert reader.orientation_seed is None
    assert reader.recommended_layer is None
    assert reader.extraction_rendering is None
    assert reader.resolved_extraction_rendering.is_raw
    assert repe_reader.score_activation(reader, [3.0, 7.0]) == pytest.approx(1.25)


# --- derive-steering conversion: whose sign the bytes carry -------------------

def test_derived_vector_applies_the_probe_orientation():
    """Round-5 audit finding 1, and still the rule under ``trainMajority``:
    ``ScalarProbe.score`` is ``orientation · (a·d − c)``, so a reader whose PC1
    came out anti-aligned with the positive class stores a direction pointing
    AWAY from the concept."""
    reader = _manual_reader(layer=1)
    reader.probe = ScalarProbe(
        direction=[1.0, 0.0], projection_center=0.5, projection_scale=2.0,
        orientation=-1.0, positive_mean=-1.0, negative_mean=1.0,
        activation_center=[1.0, 0.0])
    vectors, sidecar = repe_reader.derive_steering_sidecar(
        reader, reader_file_name="reader-fear-layer1.json", reader_bytes=b"{}")
    assert vectors.per_layer[1] == pytest.approx([-1.0, 0.0])
    assert sidecar.readerProbeOrientation == -1.0
    # A concept-positive activation scores positive through the probe AND
    # projects positive onto the derived vector — the two agreeing is the whole
    # point of the fix.
    concept_positive = [-4.0, 0.0]
    assert repe_reader.score_activation(reader, concept_positive) > 0
    assert sum(a * b for a, b in zip(concept_positive, vectors.per_layer[1])) > 0

    forward = _manual_reader(layer=1)
    forward_vectors, forward_sidecar = repe_reader.derive_steering_sidecar(
        forward, reader_file_name="r.json", reader_bytes=b"{}")
    assert forward_vectors.per_layer[1] == pytest.approx([1.0, 0.0])
    assert forward_sidecar.readerProbeOrientation == 1.0
    # Held-out never voted on a train-majority fit, so there is nothing to
    # stamp — the field is absent, not False.
    assert sidecar.trainHeldOutSignDisagreement is None
    assert forward_sidecar.trainHeldOutSignDisagreement is None


def test_derived_vector_keeps_the_held_out_sign_when_train_disagrees():
    """Review round 6. Under ``heldOutPairAgreement`` the fitted direction is
    ALREADY the held-out-chosen sign; ``probe.orientation`` still comes from
    the train class means. Applying it when the splits disagree re-flipped the
    vector to the direction held-out rejected — the reviewer's repro exactly:
    direction [−1, 0], orientation −1, convention heldOutPairAgreement used to
    derive [1, 0]."""
    reader = _manual_reader(layer=1)
    reader.sign_convention = repe_reader.HELD_OUT_PAIR_AGREEMENT
    reader.probe = ScalarProbe(
        direction=[-1.0, 0.0], projection_center=0.5, projection_scale=2.0,
        orientation=-1.0, positive_mean=-1.0, negative_mean=1.0,
        activation_center=[1.0, 0.0])
    vectors, sidecar = repe_reader.derive_steering_sidecar(
        reader, reader_file_name="reader-fear-layer1.json", reader_bytes=b"{}")
    assert vectors.per_layer[1] == pytest.approx([-1.0, 0.0])
    assert sidecar.readerSignConvention == "heldOutPairAgreement"
    assert sidecar.signConvention == "heldOutPairAgreement"
    assert sidecar.readerProbeOrientation == -1.0
    # The disagreement is diagnostic information, so it is stamped rather than
    # silently discarded.
    assert sidecar.trainHeldOutSignDisagreement is True


def test_derived_vector_stamps_agreement_when_both_splits_chose_the_same_sign():
    reader = _manual_reader(layer=1)
    reader.sign_convention = repe_reader.HELD_OUT_PAIR_AGREEMENT
    vectors, sidecar = repe_reader.derive_steering_sidecar(
        reader, reader_file_name="r.json", reader_bytes=b"{}")
    assert vectors.per_layer[1] == pytest.approx([1.0, 0.0])
    assert sidecar.readerProbeOrientation == 1.0
    assert sidecar.trainHeldOutSignDisagreement is False


def test_a_legacy_schema_1_reader_derives_under_train_majority():
    """A schema-1 artifact carries no ``signConvention``; ``from_dict`` reads
    that as train-majority, so the round-5 behaviour is what it gets."""
    legacy = {
        "artifactType": repe_reader.ARTIFACT_TYPE,
        "schemaVersion": 1, "modelID": "org/m", "revision": "abc",
        "concept": "fear", "layer": 1, "datasetHash": "dh",
        "template": {"id": "unnamed-scenario-v1", "text": "S: {{stimulus}} q",
                     "conceptSlot": False, "latToken": "final", "hash": "th",
                     "divergence": "unnamed-clean-room"},
        "probe": {"direction": [1.0, 0.0], "projectionCenter": 0.5,
                  "projectionScale": 2.0, "orientation": -1.0,
                  "positiveMean": -1.0, "negativeMean": 1.0,
                  "activationCenter": [1.0, 0.0]},
        "pc1ExplainedVariance": 0.87, "trainAccuracy": 1.0,
        "heldOutAccuracy": 0.8, "trainPairCount": 4, "heldOutPairCount": 2,
        "renderingConvention": "rawCompletion scaffold",
        "extractionDate": "2026-07-03T00:00:00Z",
    }
    reader = repe_reader.ReaderArtifact.from_dict(legacy)
    assert reader.sign_convention == repe_reader.TRAIN_MAJORITY
    vectors, sidecar = repe_reader.derive_steering_sidecar(
        reader, reader_file_name="r.json", reader_bytes=b"{}")
    assert vectors.per_layer[1] == pytest.approx([-1.0, 0.0])
    assert sidecar.trainHeldOutSignDisagreement is None


def test_held_out_accuracy_does_not_move_with_the_direction_s_sign():
    """The reviewer's second claim, audited: it does NOT hold. ``scalar_probe``
    derives ``orientation``, ``projectionCenter`` and ``projectionScale`` from
    the same direction it is handed, so flipping the direction flips the
    orientation and the centre together and every score is identical. Held-out
    accuracy — and therefore the recommended layer built on it — is exactly
    sign-invariant."""
    positive = [[2.0, 0.5], [3.0, -0.5], [2.5, 0.0]]
    negative = [[-2.0, 0.5], [-3.0, -0.5], [-2.5, 0.0]]
    held_positive = [[1.5, 0.2], [2.2, -0.3]]
    held_negative = [[-1.8, 0.1], [-2.4, 0.4]]
    center = vector_math.mean(positive + negative)
    forward = vector_math.scalar_probe([1.0, 0.0], positive, negative,
                                       activation_center=center)
    flipped = vector_math.scalar_probe([-1.0, 0.0], positive, negative,
                                       activation_center=center)
    assert forward.orientation == 1.0 and flipped.orientation == -1.0
    for row in held_positive + held_negative:
        assert forward.score(row) == pytest.approx(flipped.score(row))
    assert (repe_reader._pair_accuracy(forward, held_positive, held_negative)
            == repe_reader._pair_accuracy(flipped, held_positive, held_negative))


def test_derived_sidecar_carries_reader_method_and_instrument_pins():
    from steerlab_server.steering.vector_math import ExtractionMethod

    reader = _manual_reader(layer=1)
    _, sidecar = repe_reader.derive_steering_sidecar(
        reader, reader_file_name="reader-fear-layer1.json", reader_bytes=b"{}")
    assert sidecar.extractionMethod == "repeReaderLAT"
    assert sidecar.recipeMethod == "repeReaderLAT"
    assert sidecar.readerLayer == 1
    assert sidecar.readerTemplateID == "unnamed-scenario-v1"
    assert sidecar.readerTemplateHash == "th"
    assert sidecar.readerContrastMode == "supervisedContent"
    assert sidecar.readerSignConvention == "trainMajority"
    assert sidecar.signConvention == "trainMajority"
    # The method resolves in the ExtractionMethod vocabulary — without that,
    # attaching the artifact was refused as an unknown method.
    method = ExtractionMethod(sidecar.extractionMethod)
    assert method.is_repe_reader_lat and not method.has_source_concept
    assert "RepE reader" in method.source_concept_absence[0]

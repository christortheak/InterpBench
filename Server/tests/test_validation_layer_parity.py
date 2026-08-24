"""One declared layer must reach EVERY consumer, on both engines.

The gap this closes: D4 made the validation layer declarable and wired it to
convergent accuracy — while the cosine matrix stayed hardcoded to mid-network
on the server and per-concept on Swift. A study declaring 41 got its accuracy
at 41 and its discriminant matrix at 31, in the same report, unlabelled.

Every parity test written before this one covered PURE FUNCTIONS, and every
divergence found since has been in the wiring around them. This asserts over
the wiring.
"""

import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import pipeline_spec, tasks
from steerlab_server.experiment.manifest import Manifest

LAYER_COUNT = 62
DECLARED = 41


def _workspace(tmp_path, *, declared_layer=DECLARED, conditions=None):
    root = str(tmp_path)
    directory = os.path.join(root, "prompts", "concepts", "fear")
    os.makedirs(directory, exist_ok=True)
    for side, text in (("positive", "afraid"), ("negative", "calm")):
        with open(os.path.join(directory, f"{side}.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps({"text": text}) + "\n")
    es.create("pl", model_id="org/m", root=root)
    es.attach("pl", ["fear"], root=root)
    d = es.load_raw("pl", root)
    if declared_layer is not None:
        d["validationLayer"] = declared_layer
    if conditions:
        d["conditions"] = conditions
    es.save_raw(d, root)
    return root


def _fake_bundles(concepts=("fear",)):
    out = {}
    for index, name in enumerate(concepts):
        per_layer = [[1.0, float(index)] for _ in range(LAYER_COUNT)]
        out[name] = SimpleNamespace(
            vectors=SimpleNamespace(layer_count=LAYER_COUNT,
                                    per_layer=per_layer),
            residual_norm_per_layer=[1.0] * LAYER_COUNT,
            residual_norm_source="extraction-stimuli",
            stimulus_hash="h")
    return out


def test_a_declared_layer_reaches_accuracy_lens_and_every_matrix_row(
        tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    manifest = Manifest.load("pl", root)
    bundles = _fake_bundles()

    monkeypatch.setattr(tasks, "_extract_all", lambda m, mf, r: bundles)
    monkeypatch.setattr(tasks, "_persist_vectors", lambda *a, **k: None)

    seen_lens_layers = []
    from steerlab_server.steering import extractor

    def fake_lens(model, vectors, layer, top_k=12):
        seen_lens_layers.append(layer)
        return extractor.LogitLensReport(layer=layer, top_positive=[],
                                         top_negative=[])

    monkeypatch.setattr(extractor, "logit_lens", fake_lens)
    run_dir = tasks._validate_impl("pl", manifest, object(), root,
                                   lambda *a: None)

    report = json.load(open(os.path.join(run_dir, "validation-report.json")))
    # The report states the matrix layer rather than leaving it implicit.
    assert report["cosineMatrixLayer"] == DECLARED
    # The logit lens reads the same depth.
    assert seen_lens_layers == [DECLARED]

    # EVERY matrix row records the declared layer — not mid-network, and not
    # a per-concept layer.
    with open(os.path.join(run_dir, "cosine-matrix.csv"), encoding="utf-8") as h:
        rows = [line.strip().split(",") for line in h if line.strip()]
    header = rows[0]
    assert header[0] == "concept" and header[1] == "layer"
    for row in rows[1:]:
        assert row[1] == str(DECLARED), row


def test_the_matrix_layer_ignores_per_concept_conditions(tmp_path, monkeypatch):
    """The legacy per-concept rule still governs ACCURACY (existing manifests
    keep their numbers) but must never govern the matrix, or (A,B) and (B,A)
    land at different depths."""
    root = _workspace(
        tmp_path, declared_layer=None,
        conditions=[{"name": "c", "slots": [
            {"concept": "fear", "layer": 7, "alpha": 0.1}]}])
    manifest = Manifest.load("pl", root)
    bundles = _fake_bundles()

    # Accuracy still follows the condition (legacy behaviour preserved).
    resolutions = tasks._validation_layer_resolutions(
        manifest, "fear", LAYER_COUNT)
    assert [r.layer for r in resolutions] == [7]
    # The matrix does not.
    assert tasks._matrix_layers(manifest, bundles) == [LAYER_COUNT // 2]


def test_an_out_of_range_declaration_refuses_at_resolve(tmp_path):
    root = _workspace(tmp_path, declared_layer=100)
    manifest = Manifest.load("pl", root)
    with pytest.raises(RuntimeError, match="not silently clamped"):
        tasks._validation_layer_resolutions(manifest, "fear", LAYER_COUNT)


# --- the gate parser ---------------------------------------------------------

def _gate_results(rows, cap=0.8):
    gate = pipeline_spec.ValidateGate(max_cross_concept_cosine=cap)
    return pipeline_spec.evaluate_validate_gate(
        gate, ["a", "b"], {"concepts": {}}, rows)


def test_the_gate_parser_reads_both_matrix_formats():
    """Swift has always written `concept,layer,<names>` while Python wrote
    `concept,<names>`. A positional read of the Swift form treated "layer" as
    a concept, shifted every column, and compared the row's LAYER INTEGER
    against the cosine cap."""
    legacy = [["concept", "a", "b"],
              ["a", "1.0000", "0.5000"],
              ["b", "0.5000", "1.0000"]]
    with_layer = [["concept", "layer", "a", "b"],
                  ["a", "41", "1.0000", "0.5000"],
                  ["b", "41", "0.5000", "1.0000"]]

    for rows in (legacy, with_layer):
        results = _gate_results(rows)
        cosine = [r for r in results if r.gate == "maxCrossConceptCosine"]
        assert len(cosine) == 1, rows
        # 0.5 off-diagonal, not 41 read as a cosine.
        assert cosine[0].measured == pytest.approx(0.5), rows
        assert cosine[0].passed, rows

    # The layer travels into the detail when the matrix records it.
    detail = [r for r in _gate_results(with_layer)
              if r.gate == "maxCrossConceptCosine"][0].detail
    assert "layer 41" in detail


# --- the one-layer invariant must not leak through per-artifact clamping ----

def test_mismatched_bundle_depths_refuse_rather_than_clamp():
    """Clamping each row to its own depth is how the matrix became asymmetric
    again through the back door: two rows at different layers give (A,B) and
    (B,A) measured at different depths. All vectors in one study belong to one
    model revision, so differing depths are a corrupt artifact."""
    bundles = _fake_bundles(("fear",))
    shallow = _fake_bundles(("anger",))["anger"]
    shallow.vectors.layer_count = 30
    shallow.vectors.per_layer = shallow.vectors.per_layer[:30]
    bundles["anger"] = shallow

    with pytest.raises(RuntimeError, match="disagree about model depth"):
        tasks._require_uniform_depth(bundles)


def test_uniform_depths_pass():
    assert tasks._require_uniform_depth(_fake_bundles(("a", "b"))) == LAYER_COUNT


def test_the_gate_refuses_a_matrix_recording_more_than_one_layer():
    """An asymmetric matrix has no defined reading for a similarity cap, so
    the gate fails closed rather than taking the max of two incomparable
    numbers."""
    mixed = [["concept", "layer", "a", "b"],
             ["a", "41", "1.0000", "0.5000"],
             ["b", "31", "0.5000", "1.0000"]]
    results = [r for r in _gate_results(mixed)
               if r.gate == "maxCrossConceptCosine"]
    assert len(results) == 1
    assert not results[0].passed
    assert "more than one layer" in results[0].detail


def test_a_declared_depth_list_yields_per_depth_entries_and_matrices(
        tmp_path, monkeypatch):
    """validate-at-the-sweep-layers (2026-08-01): a declared fraction LIST
    produces one report entry per depth under ``depths`` (no flat mirror —
    nothing may read depth[0] as "the" accuracy), a lens read per depth, and
    one complete single-layer cosine matrix per depth."""
    root = _workspace(tmp_path, declared_layer=None)
    with open(os.path.join(root, "prompts", "concepts", "fear",
                           "validation.jsonl"), "w", encoding="utf-8") as h:
        h.write(json.dumps({"text": "s1", "expresses": True}) + "\n")
        h.write(json.dumps({"text": "s2", "expresses": False}) + "\n")
    d = es.load_raw("pl", root)
    d["validationLayerFractions"] = [0.25, 0.75]
    es.save_raw(d, root)
    manifest = Manifest.load("pl", root)
    bundles = _fake_bundles()

    monkeypatch.setattr(tasks, "_extract_all", lambda m, mf, r: bundles)
    monkeypatch.setattr(tasks, "_persist_vectors", lambda *a, **k: None)
    from steerlab_server.steering import extractor

    def fake_activations(model, texts, reading, rendering=None):
        return SimpleNamespace(
            values=[[[1.0, 0.0]] * LAYER_COUNT for _ in texts])

    monkeypatch.setattr(extractor, "activations", fake_activations)
    seen_lens_layers = []

    def fake_lens(model, vectors, layer, top_k=12):
        seen_lens_layers.append(layer)
        return extractor.LogitLensReport(layer=layer, top_positive=[],
                                         top_negative=[])

    monkeypatch.setattr(extractor, "logit_lens", fake_lens)
    run_dir = tasks._validate_impl("pl", manifest, object(), root,
                                   lambda *a: None)

    report = json.load(open(os.path.join(run_dir, "validation-report.json")))
    entry = report["concepts"]["fear"]
    layers = [int(0.25 * LAYER_COUNT), int(0.75 * LAYER_COUNT)]
    assert [sub["layer"] for sub in entry["depths"]] == layers
    for sub in entry["depths"]:
        assert "scenarioAccuracy" in sub
        assert sub["layerResolution"]["source"] == "declaredFraction"
    # No flat mirror with several depths.
    assert "scenarioAccuracy" not in entry
    assert "layer" not in entry
    # The lens read every declared depth; the report carries a list.
    assert seen_lens_layers == layers
    assert [block["layer"] for block in report["logitLens"]["fear"]] == layers
    # One complete matrix per depth, first under the historical name.
    assert report["cosineMatrixLayers"] == layers
    assert os.path.exists(os.path.join(run_dir, "cosine-matrix.csv"))
    assert os.path.exists(
        os.path.join(run_dir, f"cosine-matrix-L{layers[1]}.csv"))


def test_a_single_depth_report_keeps_the_flat_shape_exactly(
        tmp_path, monkeypatch):
    """Every pre-list consumer reads a single-depth report unchanged: the
    flat fields stay, and `depths` is the same entry once."""
    root = _workspace(tmp_path)  # declared_layer=41
    with open(os.path.join(root, "prompts", "concepts", "fear",
                           "validation.jsonl"), "w", encoding="utf-8") as h:
        h.write(json.dumps({"text": "s1", "expresses": True}) + "\n")
    manifest = Manifest.load("pl", root)
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda m, mf, r: _fake_bundles())
    monkeypatch.setattr(tasks, "_persist_vectors", lambda *a, **k: None)
    from steerlab_server.steering import extractor
    monkeypatch.setattr(
        extractor, "activations",
        lambda model, texts, reading, rendering=None: SimpleNamespace(
            values=[[[1.0, 0.0]] * LAYER_COUNT for _ in texts]))
    monkeypatch.setattr(
        extractor, "logit_lens",
        lambda model, vectors, layer, top_k=12: extractor.LogitLensReport(
            layer=layer, top_positive=[], top_negative=[]))
    run_dir = tasks._validate_impl("pl", manifest, object(), root,
                                   lambda *a: None)
    report = json.load(open(os.path.join(run_dir, "validation-report.json")))
    entry = report["concepts"]["fear"]
    assert entry["layer"] == DECLARED
    assert "scenarioAccuracy" in entry
    assert entry["layerResolution"]["layer"] == DECLARED
    assert len(entry["depths"]) == 1
    assert entry["depths"][0]["layer"] == DECLARED
    # The lens keeps its historical flat shape.
    assert report["logitLens"]["fear"]["layer"] == DECLARED


def test_a_colliding_depth_list_leaves_no_validate_directory(
        tmp_path, monkeypatch):
    root = _workspace(tmp_path, declared_layer=None)
    d = es.load_raw("pl", root)
    d["validationLayerFractions"] = [0.6, 0.61]
    es.save_raw(d, root)
    manifest = Manifest.load("pl", root)
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda m, mf, r: _fake_bundles())
    monkeypatch.setattr(tasks, "_persist_vectors", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="both resolve to layer"):
        tasks._validate_impl("pl", manifest, object(), root, lambda *a: None)
    runs = os.path.join(root, "runs")
    validate_dirs = [
        d for d in (os.listdir(runs) if os.path.isdir(runs) else [])
        if d.endswith("-validate")]
    assert validate_dirs == []


def test_an_out_of_range_declaration_leaves_no_validate_directory_at_all(
        tmp_path, monkeypatch):
    """This used to fire inside the per-concept loop — AFTER cosine-matrix.csv
    had been persisted at a clamped layer the researcher never declared. The
    first fix moved it ahead of the matrix but still behind config.json,
    validation-evidence.json and the persisted vectors, so a refusal left a
    half-populated directory that LOOKS like a validation run. Extraction is
    what reveals depth, so the check now runs before anything is created."""
    root = _workspace(tmp_path, declared_layer=999)
    manifest = Manifest.load("pl", root)
    monkeypatch.setattr(tasks, "_extract_all", lambda m, mf, r: _fake_bundles())
    monkeypatch.setattr(tasks, "_persist_vectors", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="not silently clamped"):
        tasks._validate_impl("pl", manifest, object(), root, lambda *a: None)

    runs = os.path.join(root, "runs")
    validate_dirs = [
        d for d in (os.listdir(runs) if os.path.isdir(runs) else [])
        if d.endswith("-validate")]
    assert validate_dirs == [], (
        "a refused validation left artifacts behind: " + repr(validate_dirs))

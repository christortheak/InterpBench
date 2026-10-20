"""An agent's ablation covers exactly the layers it declares.

An ablating injection may carry ``layers``, a list of blocks; nil or empty
means every layer. Release review 2026-10-19: the Mac engine built its cells
from the first declared layer to the last, so ``[3, 7, 10]`` also ablated
4–6 and 8–9, an unsorted list such as ``[10, 3]`` crashed, and a list naming
only layers the vector does not have ablated every layer there and none on
this engine. Both engines now ablate the listed layers and nothing else,
ascending and once each, which is what the preregistration's ``L3+L7+L10``
describes, and both refuse a listed layer the vector does not have with the
same sentence (``ablation_layers.covered``).

``Tests/Fixtures/cross-engine/ablation-layers.json`` holds the layers this
engine's cell builder covers for each declared list, or its refusal; the
Swift twin (``AblationLayerSubsetTests.swift``) builds the same agents and
must cover the same layers or refuse with the same sentence.
"""

import importlib.util
import json
import os
import sys
import warnings

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "Tests", "Fixtures", "cross-engine",
                       "ablation-layers.json")
GENERATOR = os.path.join(REPO, "scripts", "regenerate-cross-engine-fixtures.py")
REGENERATE = ("stale fixture — re-run ablation_layers() in "
              "scripts/regenerate-cross-engine-fixtures.py and commit")


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))  # it prepends Server/
    spec = importlib.util.spec_from_file_location(
        "regenerate_cross_engine_fixtures", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def test_the_cross_engine_fixture_is_current(generator, tmp_path):
    fixture = _fixture()
    assert fixture["layerCount"] == generator.ABLATION_LAYER_COUNT, REGENERATE
    assert [(case["label"], case["injection"]) for case in fixture["cases"]] \
        == generator.ablation_layer_cases(), REGENERATE
    for case in fixture["cases"]:
        expected = {key: case[key] for key in ("layers", "refusal")
                    if key in case}
        assert generator.ablation_layer_cells(case["injection"], str(tmp_path)) \
            == expected, f"{case['label']}: {REGENERATE}"


def test_the_fixture_covers_the_shapes_the_engines_once_read_differently():
    cases = {case["label"]: case.get("layers", case.get("refusal"))
             for case in _fixture()["cases"]}
    every = list(range(_fixture()["layerCount"]))
    assert cases["absent-means-every-layer"] == every
    assert cases["empty-means-every-layer"] == every
    # The gaps stay gaps: never 3 through 10.
    assert cases["non-contiguous"] == [3, 7, 10]
    assert cases["unsorted"] == [3, 10]
    assert cases["repeated"] == [3, 7]
    assert cases["every-layer-listed"] == every
    # Steering ignores the list.
    assert cases["steering-ignores-layers"] == [4, 5, 6]
    # A layer the vector does not have refuses, whether some or all are.
    assert "lists layer 12, which its vector does not have" \
        in cases["partly-out-of-range"]
    assert "lists layers 12, 40, which" in cases["all-out-of-range"]
    assert "lists layer -1, which" in cases["negative"]


def _save_vector(root, layer_count):
    from steerlab_server.steering import vector_store

    run_dir = os.path.join(str(root), "runs", "extract")
    sidecar = vector_store.SteeringVectorSidecar(
        modelID="test/model", concept="fear", stimulusSetHash="h" * 64,
        layerCount=layer_count, hiddenSize=3,
        normsPerLayer=[1.0] * layer_count,
        extractionDate="2026-10-19T00:00:00Z",
        extractionMethod="meanDifference", substrate=vector_store.SUBSTRATE)
    vector_store.save(
        vector_store.ConceptVectors(per_layer=[
            [float(layer + 1), 1.0, 0.0] for layer in range(layer_count)]),
        sidecar, run_dir, "fear")
    return os.path.join(run_dir, "fear")


def _ablated(root, injection):
    from steerlab_server.experiment import model_variant

    variant = model_variant.ModelVariant(
        name="fear-ablation", base_model_id="test/model",
        injections=[injection], band_width=1, alpha_in_norm_units=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cells = model_variant.variant_injections(variant, root=str(root))
    assert all(cell.mode == "ablate" for cell in cells)
    return sorted({cell.layer for cell in cells})


def test_a_layer_list_with_gaps_ablates_exactly_those_layers(tmp_path):
    vector = _save_vector(tmp_path, 12)
    ablated = _ablated(tmp_path, {
        "concept": "fear", "vectorArtifactID": vector, "layer": 3,
        "alpha": 1.0, "mode": "ablate", "layers": [10, 3, 7]})
    assert ablated == [3, 7, 10]
    assert ablated != list(range(3, 11)), "the span between first and last"


def test_the_preregistration_names_each_ablated_layer_once_in_order(tmp_path):
    """The line for an agent's ablation names the set the run ablates, read
    off this engine's own cell builder, however the list was written."""
    from steerlab_server.experiment import experiment_store as es

    with open(os.path.join(REPO, "Tests", "Fixtures", "cross-engine",
                           "preregistration-arms.json"),
              encoding="utf-8") as handle:
        case = next(entry for entry in json.load(handle)
                    if entry["label"] == "agent-ablation-layer-subset")
    manifest = json.loads(json.dumps(case["manifest"]))
    injection = manifest["variantConditions"][0]["artifact"]["injections"][0]
    assert injection["layers"] == [10, 3, 7, 3]
    directory = os.path.join(str(tmp_path), "experiments", manifest["name"])
    os.makedirs(directory)
    es._write_preregistration(manifest, str(tmp_path))
    with open(os.path.join(directory, es.PREREG_FILENAME),
              encoding="utf-8") as handle:
        text = handle.read()
    assert "- **fear-ablate-subset** (agent): fear ablated λ=0.5 at " \
        "L3+L7+L10; base model" in text

    vector = _save_vector(tmp_path, 16)
    ablated = _ablated(tmp_path, {**injection, "vectorArtifactID": vector})
    assert "+".join(f"L{layer}" for layer in ablated) == "L3+L7+L10"


def test_a_listed_layer_the_vector_does_not_have_refuses(tmp_path):
    """Never skipped, and never read as every layer: some or all of the list
    outside the vector refuses with the repair."""
    vector = _save_vector(tmp_path, 12)
    for layers, named in (([3, 12], "layer 12"), ([12, 40], "layers 12, 40"),
                          ([-1, 2], "layer -1")):
        with pytest.raises(ValueError) as refused:
            _ablated(tmp_path, {
                "concept": "fear", "vectorArtifactID": vector, "layer": 3,
                "alpha": 1.0, "mode": "ablate", "layers": layers})
        assert str(refused.value) == (
            f"variant 'fear-ablation': the ablation of 'fear' lists {named}, "
            "which its vector does not have (its layers are 0 to 11); remove "
            f"{'it' if named.startswith('layer ') else 'them'} from the "
            "injection's layers, or delete the list to ablate every layer")


def test_a_list_entry_that_is_not_a_layer_index_refuses(tmp_path):
    from steerlab_server.experiment import ablation_layers

    for declared in ([3, 7.5], [True], ["3"], 3):
        with pytest.raises(ValueError, match="not all whole layer indices"):
            ablation_layers.covered("v", "fear", declared, 12)
    assert ablation_layers.covered("v", "fear", [10, 3, 10], 12) == [3, 10]
    assert ablation_layers.covered("v", "fear", None, 3) == [0, 1, 2]
    assert ablation_layers.covered("v", "fear", [], 3) == [0, 1, 2]

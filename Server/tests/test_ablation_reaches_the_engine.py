"""The far end of the seam: a spec the app composed must ABLATE here.

Reported live from a 27B cluster session (2026-07-27): ablating at λ behaved
"almost exactly like steering with the same α" because it WAS steering at α.
The app dropped the mode composing its inline variant spec, and this side never
read it — so an ablation arrived as an ordinary injection.

It was silent because the wire spells the default by omission (chosen so
steering artifacts keep their bytes and hashes, which is right). A dropped mode
is therefore indistinguishable from a steering one, and unit tests at both ends
passed the whole time. Mirror of Swift
``Tests/ExperimentKitTests/AblationReachesTheServerTests.swift``.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.api import dto  # noqa: E402
from steerlab_server.experiment.generate import CellInjection, _injectors  # noqa: E402
from steerlab_server.steering.ablator import SubspaceAblator  # noqa: E402
from steerlab_server.steering.injector import VectorInjector  # noqa: E402


def test_an_ablation_cell_builds_an_ablator_not_an_injector():
    chain = _injectors(
        [CellInjection(layer=0, vector=[1.0, 0.0, 0.0], alpha=1.0,
                       mode="ablate", concept="anger")],
        prompt_token_count=8)
    assert any(isinstance(item, SubspaceAblator) for item in chain), (
        "the ablation degraded to steering — this is the live 27B bug")
    assert not any(isinstance(item, VectorInjector) for item in chain)


def test_a_steering_cell_still_builds_an_injector():
    chain = _injectors(
        [CellInjection(layer=0, vector=[1.0, 0.0, 0.0], alpha=2.0)],
        prompt_token_count=8)
    assert all(isinstance(item, VectorInjector) for item in chain)


def test_the_two_modes_do_different_things_at_the_same_value():
    """The exact confusion reported: at the same number, the two must diverge."""
    h = torch.tensor([[[4.0, 1.0, 0.0]]])

    def run(mode):
        current = h
        for item in _injectors(
                [CellInjection(layer=0, vector=[1.0, 0.0, 0.0], alpha=1.0,
                               mode=mode, concept="anger")],
                prompt_token_count=1):
            current = item.apply(current, 0, 0)
        return current[0][0].tolist()

    steered, ablated = run("add"), run("ablate")
    assert steered[0] == pytest.approx(5.0)   # 4 + 1·1
    assert ablated[0] == pytest.approx(0.0)   # 4 − 1·4
    assert steered != ablated


def test_the_injection_dto_carries_the_mode_and_defaults_to_add():
    assert dto.InjectionDTO(layer=0, alpha=1.0).mode == "add"
    assert dto.InjectionDTO(layer=0, alpha=1.0, mode="ablate").mode == "ablate"


def test_a_variant_artifact_spec_with_mode_ablates_every_layer(tmp_path):
    """An ablating AGENT is the worse case: it would have written results
    labelled 'ablation' that were actually steering."""
    from steerlab_server.experiment import model_variant
    from steerlab_server.steering import vector_store

    run_dir = tmp_path / "runs" / "20260727T0-extract"
    run_dir.mkdir(parents=True)
    layer_count = 4
    sidecar = vector_store.SteeringVectorSidecar(
        modelID="org/m", concept="anger", stimulusSetHash="h" * 64,
        layerCount=layer_count, hiddenSize=3,
        normsPerLayer=[1.0] * layer_count,
        extractionDate="2026-07-27T00:00:00Z",
        residualNormPerLayer=[1.0] * layer_count,
        extractionMethod="meanDifference", substrate=vector_store.SUBSTRATE)
    vector_store.save(
        vector_store.ConceptVectors(per_layer=[[1.0, 0.0, 0.0]] * layer_count),
        sidecar, str(run_dir), "anger")

    variant = model_variant.ModelVariant(
        name="no-anger", base_model_id="org/m",
        injections=[{
            "concept": "anger",
            "vectorArtifactID": str(run_dir / "anger"),
            "layer": 2, "alpha": 1.0, "mode": "ablate"}],
        band_width=1, alpha_in_norm_units=True)

    cells = model_variant.variant_injections(variant)
    # Every layer, not the declared one widened by the band.
    assert sorted(c.layer for c in cells) == list(range(layer_count))
    assert all(c.mode == "ablate" for c in cells)
    # λ passed through raw: no residual-norm conversion.
    assert all(c.alpha == pytest.approx(1.0) for c in cells)

"""J-lens support readout: decompose a stored concept vector into token atoms.

Hermetic. A StubBackend lens, a synthetic vector artifact, and a monkeypatched
gain-scaled head stand in for the model, so the numerics are checked against
tiny constructed cases rather than live output — house rule, and it means the
solver's guarantees are pinned without a 27B node.

The load-bearing tests here are the honesty ones. The energy fraction cannot
carry the claim (a norm-matched random direction scores comparably in a ~100x
overcomplete non-negative dictionary), so this suite pins that the schema
cannot express an energy figure without its control, and that a diverged solve
raises instead of returning a number.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.jlens import backend, decompose, importer, lens_store
from steerlab_server.jlens.schemas import JLensError
from steerlab_server.steering import vector_store
from steerlab_server.steering.vector_store import (ConceptVectors,
                                                   SteeringVectorSidecar)

D_MODEL = 8
VOCAB = 64
LAYERS = (0, 1, 2)


class FakeTokenizer:
    def convert_ids_to_tokens(self, ids):
        return [f"tok{int(i)}" for i in ids]


def _snapshot(tmp_path, model_id="google/gemma-3-4b-it"):
    entry = importer.SUPPORTED[model_id]
    folder = tmp_path / "snap" / entry["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    backend.StubBackend(d_model=D_MODEL, source_layers=list(LAYERS)).save_checkpoint(
        str(folder / entry["tensor"]))
    (folder / entry["config"]).write_text(
        f"hf_model_name: {model_id}\n"
        "dataset:\n  name: Salesforce/wikitext\n  config: wikitext-103-raw-v1\n"
        "fit:\n  dtype: bfloat16\n  max_seq_len: 128\n"
        "results:\n  prompts_fitted: 546\n", encoding="utf-8")
    return str(tmp_path / "snap")


@pytest.fixture()
def lens(tmp_path):
    root = str(tmp_path / "ws")
    record = importer.import_lens("google/gemma-3-4b-it", root=root,
                                  snapshot=_snapshot(tmp_path))
    return root, record


@pytest.fixture()
def head(monkeypatch):
    """A deterministic, well-spread gain-scaled head, injected for every test."""
    generator = torch.Generator().manual_seed(7)
    table = torch.randn(VOCAB, D_MODEL, generator=generator)
    monkeypatch.setattr(decompose, "gain_scaled_head", lambda *a, **k: table)
    return table


def _write_vector(directory, name, per_layer, *, model_id="google/gemma-3-4b-it",
                  substrate="python-hf-transformers", concept=None):
    os.makedirs(directory, exist_ok=True)
    vectors = ConceptVectors(per_layer=[list(map(float, row)) for row in per_layer])
    sidecar = SteeringVectorSidecar(
        modelID=model_id, concept=concept or name, stimulusSetHash="deadbeef",
        layerCount=vectors.layer_count, hiddenSize=vectors.hidden_size,
        normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)],
        extractionDate="2026-07-29T00:00:00Z", revision="rev1",
        substrate=substrate, extractionMethod="meanDifference")
    vector_store.save(vectors, sidecar, directory, name)
    return directory, name


def _atom(record, root, layer, token, table):
    """The reference atom, computed the way ``derive`` computes a direction."""
    j = lens_store.load_layer(record, layer, root=root).to(torch.float32)
    return j.T @ table[token]


# --------------------------------------------------------------------------
# the atom convention
# --------------------------------------------------------------------------

def test_atom_columns_match_the_derive_convention(lens, head):
    """``J^T (g . u_t)`` — the same arithmetic ``derive.derive_direction`` writes.

    Regression: an earlier version computed ``J w_t`` for the norms and columns
    while correlating with ``J^T w_t``. Selection and reconstruction then
    disagreed, the refinement zeroed every atom it could not reconcile, and the
    readout came back as two tokens at 92%/8% instead of a spread support.
    """
    root, record = lens
    atoms = decompose._LayerAtoms(lens_store.load_layer(record, 1, root=root),
                                  head, torch.device("cpu"))
    for token in (0, 5, 63):
        reference = _atom(record, root, 1, token, head)
        got = atoms.columns([token])[:, 0] * atoms._norms[token]
        assert torch.allclose(got, reference, atol=1e-5)


def test_correlation_agrees_with_explicit_inner_products(lens, head):
    root, record = lens
    atoms = decompose._LayerAtoms(lens_store.load_layer(record, 2, root=root),
                                  head, torch.device("cpu"))
    residual = torch.randn(D_MODEL, generator=torch.Generator().manual_seed(3))
    correlation = atoms.correlate(residual)
    for token in (1, 9, 40):
        reference = _atom(record, root, 2, token, head)
        expected = float(reference @ residual / reference.norm())
        assert correlation[token].item() == pytest.approx(expected, abs=1e-4)


# --------------------------------------------------------------------------
# solver guarantees
# --------------------------------------------------------------------------

def test_a_single_atom_is_recovered_exactly_at_k_one(lens, head, tmp_path):
    """A vector that IS a dictionary column must come back as that column alone.

    This is the positive control on the whole pipeline: it fails if the atom
    convention, the normalization, or the refinement is wrong.
    """
    root, record = lens
    token = 11
    per_layer = [list(_atom(record, root, min(i, 2), token, head)) for i in range(4)]
    directory, name = _write_vector(str(tmp_path / "vec"), "atomlike", per_layer)

    readout = decompose.decompose(
        lens_id=record.lensID, vector_directory=directory, vector_name=name,
        layers=[1], budget=1, root=root, device="cpu", tokenizer=FakeTokenizer())
    layer = readout["layers"][0]
    assert layer["support"][0]["tokenID"] == token
    assert layer["support"][0]["share"] == pytest.approx(1.0)
    assert layer["energyFraction"] == pytest.approx(1.0, abs=1e-4)


def test_a_nonnegative_mix_recovers_both_atoms_and_their_weights(lens, head, tmp_path):
    root, record = lens
    first, second = 3, 20
    a, b = 2.0, 0.5
    mixed = a * _atom(record, root, 1, first, head) + b * _atom(record, root, 1, second, head)
    per_layer = [list(mixed) for _ in range(4)]
    directory, name = _write_vector(str(tmp_path / "vec"), "mix", per_layer)

    readout = decompose.decompose(
        lens_id=record.lensID, vector_directory=directory, vector_name=name,
        layers=[1], budget=2, root=root, device="cpu", tokenizer=FakeTokenizer())
    layer = readout["layers"][0]
    assert {row["tokenID"] for row in layer["support"]} == {first, second}
    assert layer["energyFraction"] == pytest.approx(1.0, abs=1e-3)
    # Coefficients are lengths along unit atoms, so they scale with the atom norm.
    by_token = {row["tokenID"]: row["coefficient"] for row in layer["support"]}
    expected_first = a * float(_atom(record, root, 1, first, head).norm())
    assert by_token[first] == pytest.approx(expected_first, rel=1e-3)


def test_coefficients_are_never_negative(lens, head, tmp_path):
    root, record = lens
    generator = torch.Generator().manual_seed(19)
    per_layer = [list(torch.randn(D_MODEL, generator=generator)) for _ in range(4)]
    directory, name = _write_vector(str(tmp_path / "vec"), "noise", per_layer)
    readout = decompose.decompose(
        lens_id=record.lensID, vector_directory=directory, vector_name=name,
        layers=[0, 1, 2], budget=6, root=root, device="cpu", tokenizer=FakeTokenizer())
    for layer in readout["layers"]:
        assert all(row["coefficient"] >= 0 for row in layer["support"])


def test_energy_fraction_never_decreases_with_budget(lens, head, tmp_path):
    """A larger support cannot fit worse. This held only after the solver stopped
    gradient-stepping the whole support with a step from the Gram diagonal."""
    root, record = lens
    generator = torch.Generator().manual_seed(23)
    per_layer = [list(torch.randn(D_MODEL, generator=generator)) for _ in range(4)]
    directory, name = _write_vector(str(tmp_path / "vec"), "grow", per_layer)
    previous = -1.0
    for budget in (1, 2, 4, 6):
        readout = decompose.decompose(
            lens_id=record.lensID, vector_directory=directory, vector_name=name,
            layers=[1], budget=budget, root=root, device="cpu",
            tokenizer=FakeTokenizer())
        fraction = readout["layers"][0]["energyFraction"]
        assert fraction >= previous - 1e-6
        assert 0.0 <= fraction <= 1.0
        previous = fraction


def test_a_diverged_solve_raises_rather_than_returning_a_number(lens, head, tmp_path):
    """The bound is what makes the readout trustworthy: during development a bad
    step size produced "energy fractions" of -3.3e23, and a milder divergence
    would have looked exactly like data."""
    root, record = lens
    directory, name = _write_vector(str(tmp_path / "vec"), "boom",
                                    [[1.0] * D_MODEL for _ in range(4)])

    def diverge(gram, rhs, target_sq, coeffs, iterations=600):
        return coeffs * 1e12

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(decompose, "_refine", diverge)
        with pytest.raises(JLensError, match="outside"):
            decompose.decompose(
                lens_id=record.lensID, vector_directory=directory,
                vector_name=name, layers=[1], budget=3, root=root,
                device="cpu", tokenizer=FakeTokenizer())


def test_an_all_zero_layer_is_refused(lens, head, tmp_path):
    root, record = lens
    directory, name = _write_vector(str(tmp_path / "vec"), "zeros",
                                    [[0.0] * D_MODEL for _ in range(4)])
    with pytest.raises(JLensError, match="all zeros"):
        decompose.decompose(lens_id=record.lensID, vector_directory=directory,
                            vector_name=name, layers=[1], budget=2, root=root,
                            device="cpu", tokenizer=FakeTokenizer())


# --------------------------------------------------------------------------
# the honesty contract
# --------------------------------------------------------------------------

def test_a_layer_report_cannot_be_built_without_its_null():
    """Structural, not conventional. A norm-matched random direction scores
    comparably to a real concept vector in this dictionary, so an energy figure
    without its control is uninterpretable — the schema must not be able to
    express one."""
    with pytest.raises(TypeError):
        decompose._layer_report(1, pieces=[], indices=[], coefficients=[],
                                fraction=0.07, exhausted_at=None,
                                null_exhausted_at=None)


def test_every_layer_reports_a_null_and_the_margin(lens, head, tmp_path):
    root, record = lens
    generator = torch.Generator().manual_seed(31)
    per_layer = [list(torch.randn(D_MODEL, generator=generator)) for _ in range(4)]
    directory, name = _write_vector(str(tmp_path / "vec"), "withnull", per_layer)
    readout = decompose.decompose(
        lens_id=record.lensID, vector_directory=directory, vector_name=name,
        layers=[0, 1, 2], budget=4, root=root, device="cpu",
        tokenizer=FakeTokenizer())
    assert readout["nullSeed"] == decompose.NULL_SEED
    for layer in readout["layers"]:
        assert 0.0 <= layer["nullEnergyFraction"] <= 1.0
        assert layer["energyOverNull"] == pytest.approx(
            layer["energyFraction"] - layer["nullEnergyFraction"])


def test_the_null_is_reproducible_from_the_record(lens, head, tmp_path):
    root, record = lens
    generator = torch.Generator().manual_seed(37)
    per_layer = [list(torch.randn(D_MODEL, generator=generator)) for _ in range(4)]
    directory, name = _write_vector(str(tmp_path / "vec"), "repro", per_layer)
    kwargs = dict(lens_id=record.lensID, vector_directory=directory,
                  vector_name=name, layers=[1], budget=4, root=root,
                  device="cpu", tokenizer=FakeTokenizer())
    first = decompose.decompose(**kwargs)["layers"][0]["nullEnergyFraction"]
    second = decompose.decompose(**kwargs)["layers"][0]["nullEnergyFraction"]
    assert first == pytest.approx(second)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_a_foreign_substrate_vector_is_refused(lens, head, tmp_path):
    """Activations do not transfer across engines and the lens is Python-only, so
    an MLX-extracted direction cannot be compared with these atoms."""
    root, record = lens
    directory, name = _write_vector(str(tmp_path / "vec"), "mlxvec",
                                    [[0.5] * D_MODEL for _ in range(4)],
                                    substrate="swift-mlx")
    with pytest.raises(JLensError, match="swift-mlx"):
        decompose.decompose(lens_id=record.lensID, vector_directory=directory,
                            vector_name=name, layers=[1], budget=2, root=root,
                            device="cpu", tokenizer=FakeTokenizer())


def test_a_vector_from_another_model_is_refused(lens, head, tmp_path):
    root, record = lens
    directory, name = _write_vector(str(tmp_path / "vec"), "wrongmodel",
                                    [[0.5] * D_MODEL for _ in range(4)],
                                    model_id="google/gemma-3-27b-it")
    with pytest.raises(JLensError, match="fitted on"):
        decompose.decompose(lens_id=record.lensID, vector_directory=directory,
                            vector_name=name, layers=[1], budget=2, root=root,
                            device="cpu", tokenizer=FakeTokenizer())


def test_the_target_layer_has_no_jacobian_and_is_refused(lens, head, tmp_path):
    """Transport AT the target is the identity by construction, so there is no
    ``J`` there to decompose over — asking is a mistake, not a default."""
    root, record = lens
    directory, name = _write_vector(str(tmp_path / "vec"), "target",
                                    [[0.5] * D_MODEL for _ in range(4)])
    with pytest.raises(JLensError, match="not fitted source layers"):
        decompose.decompose(lens_id=record.lensID, vector_directory=directory,
                            vector_name=name, layers=[record.targetLayer],
                            budget=2, root=root, device="cpu",
                            tokenizer=FakeTokenizer())


def test_layers_default_to_every_fitted_source_layer(lens, head, tmp_path):
    """Readability varies by layer, so picking one for the caller would hide the
    thing they most need to see."""
    root, record = lens
    directory, name = _write_vector(str(tmp_path / "vec"), "alllayers",
                                    [[0.3, 0.1, -0.2, 0.4, 0.5, -0.1, 0.2, 0.05]
                                     for _ in range(4)])
    readout = decompose.decompose(
        lens_id=record.lensID, vector_directory=directory, vector_name=name,
        budget=2, root=root, device="cpu", tokenizer=FakeTokenizer())
    assert [layer["layer"] for layer in readout["layers"]] == list(LAYERS)


def test_a_missing_vector_is_named_in_the_error(lens, head, tmp_path):
    root, record = lens
    with pytest.raises(JLensError, match="no vector artifact"):
        decompose.decompose(lens_id=record.lensID,
                            vector_directory=str(tmp_path / "nope"),
                            vector_name="ghost", layers=[1], budget=2,
                            root=root, device="cpu", tokenizer=FakeTokenizer())


# --------------------------------------------------------------------------
# provenance and persistence
# --------------------------------------------------------------------------

def test_identity_hash_moves_with_everything_that_moves_the_numbers():
    base = dict(lens_id="L", lens_source_hash="abc", model_id="m",
                revision="r", vector_sha256="v", layers=[1, 2], budget=25)
    reference = decompose.support_identity_hash(**base)
    for field, value in [("budget", 50), ("layers", [1, 3]), ("vector_sha256", "w"),
                         ("revision", "r2"), ("lens_source_hash", "def")]:
        assert decompose.support_identity_hash(**{**base, field: value}) != reference
    # Layer ORDER is not a difference; the set is.
    assert decompose.support_identity_hash(**{**base, "layers": [2, 1]}) == reference


def test_write_readout_persists_json_csv_and_a_canonical_config(lens, head, tmp_path):
    root, record = lens
    directory, name = _write_vector(str(tmp_path / "vec"), "persist",
                                    [[0.3, 0.1, -0.2, 0.4, 0.5, -0.1, 0.2, 0.05]
                                     for _ in range(4)])
    readout = decompose.decompose(
        lens_id=record.lensID, vector_directory=directory, vector_name=name,
        layers=[0, 1], budget=3, root=root, device="cpu", tokenizer=FakeTokenizer())
    run_dir = decompose.write_readout(readout, root=root)

    with open(os.path.join(run_dir, "support.json"), encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["supportIdentityHash"] == readout["supportIdentityHash"]
    assert stored["artifactType"] == decompose.ARTIFACT_TYPE

    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["runType"] == "jlens-support"
    assert config["substrate"] == "python-hf-transformers"

    with open(os.path.join(run_dir, "support.csv"), encoding="utf-8") as handle:
        rows = handle.read().strip().splitlines()
    expected = 1 + sum(len(layer["support"]) for layer in readout["layers"])
    assert len(rows) == expected
    assert rows[0].startswith("layer,rank,tokenID,piece")
    # The null travels with every row, so a CSV pasted into a note cannot show
    # an energy figure stripped of its control.
    assert "nullEnergyFraction" in rows[0]

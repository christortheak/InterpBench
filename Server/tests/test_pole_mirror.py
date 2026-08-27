"""POLE MIRRORING — ``steerlab_server.steering.pole_mirror`` and its CLI verb
``steerlab-server vectors mirror-poles``.

What has to be true for a mirrored artifact to be citable:

* the tensors are the parent's, negated at EVERY layer, BIT-EXACTLY — asserted
  at the byte level via the double-negation involution, not merely by comparing
  decoded floats, because "close enough after a round trip" is exactly the
  claim a provenance stamp must not launder;
* the residual-mean tensors are NOT negated (they are an absolute activation
  statistic, not a direction);
* everything sign-invariant survives, ``negatedFrom`` +
  ``polesSwappedFromSource`` are stamped, and ``recipeIdentityHash`` — the one
  field that would be a false identity claim about the new bytes — is dropped;
* the result is an ordinary catalog artifact;
* and the refusals are typed, with the ``--concept`` one explaining WHY.

Model-free: nothing here loads or measures anything.
"""

import json
import os

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from steerlab_server import cli
from steerlab_server.experiment import catalog
from steerlab_server.steering import pole_mirror

#: Neutral fixture concepts — a made-up contrast with no study attached.
SOURCE_CONCEPT = "brightness"
MIRROR_CONCEPT = "dimness"

#: The float payload deliberately carries the two values a lossy negation path
#: would betray: a signed zero (which must come back as the OTHER signed zero,
#: and back again) and a subnormal-adjacent tiny.
LAYER_0 = [0.0, -0.0, 1.5, -2.25, 1e-8]
LAYER_1 = [3.0, -4.0, 0.0, 0.5, -0.5]


def _write_artifact(directory, name=SOURCE_CONCEPT, *, extras=None,
                    neutral_mean=True):
    """A two-layer artifact pair written the way the store writes one, plus
    whatever sidecar extras the test needs."""
    os.makedirs(directory, exist_ok=True)
    tensors = {
        "layer_0": np.array(LAYER_0, dtype=np.float32),
        "layer_1": np.array(LAYER_1, dtype=np.float32),
    }
    if neutral_mean:
        tensors["neutral_mean_layer_0"] = np.array([9.0] * 5, dtype=np.float32)
        tensors["neutral_mean_layer_1"] = np.array([-8.0] * 5, dtype=np.float32)
    save_file(tensors, os.path.join(directory, f"{name}.safetensors"))
    sidecar = {
        "schemaVersion": 2,
        "modelID": "org/m",
        "revision": "abc",
        "concept": SOURCE_CONCEPT,
        "stimulusSetHash": "stim-hash",
        "layerCount": 2,
        "hiddenSize": 5,
        "normsPerLayer": [2.0, 5.0],
        "extractionDate": "2026-01-02T03:04:05Z",
        "extractionMethod": "meanDifference",
        "recipeMethod": "caaMeanDifference",
        "readingPosition": "last token",
        "residualNormPerLayer": [11.0, 12.0],
        "residualNormSource": "neutral-corpus",
        "residualNormConvention": "wholeCorpusMean-v1",
        "coversModelDepth": True,
        "substrate": "python-hf-transformers",
        "neutralMeanSource": "neutral-corpus",
        "recipeIdentityHash": "d" * 64,
        "futureUnknownField": "keep-me",
    }
    sidecar.update(extras or {})
    with open(os.path.join(directory, f"{name}.json"), "w",
              encoding="utf-8") as handle:
        json.dump(sidecar, handle, sort_keys=True, indent=2)
    return directory


def _read(path):
    with open(path, "rb") as handle:
        return handle.read()


def _mirror(tmp_path, source_dir, name=SOURCE_CONCEPT,
            concept=MIRROR_CONCEPT, out="mirrored", **kwargs):
    return pole_mirror.mirror_poles(
        source_dir, name, concept=concept,
        run_directory=os.path.join(str(tmp_path), out), **kwargs)


# --- the bytes ------------------------------------------------------------------

def test_every_layer_is_the_parents_negation(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    result = _mirror(tmp_path, source)

    parent = load_file(os.path.join(source, f"{SOURCE_CONCEPT}.safetensors"))
    mirrored = load_file(result.vectors_path)
    for layer in ("layer_0", "layer_1"):
        assert np.array_equal(mirrored[layer], -parent[layer])
        # Signed zero is a SIGN, and it flips like every other one.
        assert list(np.signbit(mirrored[layer])) == [
            not bit for bit in np.signbit(parent[layer])]


def test_neutral_mean_tensors_are_not_negated(tmp_path):
    # The residual mean is the stream's own centre at that layer — an absolute
    # activation statistic that has nothing to do with which pole the concept
    # vector points at. Negating it would corrupt ablation mean-centring.
    source = _write_artifact(str(tmp_path / "src"))
    result = _mirror(tmp_path, source)

    parent = load_file(os.path.join(source, f"{SOURCE_CONCEPT}.safetensors"))
    mirrored = load_file(result.vectors_path)
    for key in ("neutral_mean_layer_0", "neutral_mean_layer_1"):
        assert np.array_equal(mirrored[key], parent[key])


def test_double_negation_returns_the_parents_tensor_bytes(tmp_path):
    # THE bit-exactness assertion, and it is at the BYTE level on purpose: a
    # sign-bit flip is an involution, so anything that decoded and re-encoded
    # the floats (or normalised a NaN payload, or coerced -0.0 to 0.0) shows up
    # here as a byte diff even when every decoded value compares equal.
    source = _write_artifact(str(tmp_path / "src"))
    first = _mirror(tmp_path, source, out="one")
    second = pole_mirror.mirror_poles(
        os.path.dirname(first.vectors_path), MIRROR_CONCEPT,
        concept="brightness-again",
        run_directory=os.path.join(str(tmp_path), "two"))

    original = _read(os.path.join(source, f"{SOURCE_CONCEPT}.safetensors"))
    assert _read(second.vectors_path) == original
    # …and the single negation genuinely changed them.
    assert _read(first.vectors_path) != original


def test_the_header_bytes_are_untouched(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    result = _mirror(tmp_path, source)
    original = _read(os.path.join(source, f"{SOURCE_CONCEPT}.safetensors"))
    mirrored = _read(result.vectors_path)
    assert len(mirrored) == len(original)
    header_end = 8 + int.from_bytes(original[:8], "little")
    assert mirrored[:header_end] == original[:header_end]


def test_a_non_float_tensor_is_refused_rather_than_mangled(tmp_path):
    # Two's-complement negation is not a sign-bit flip, so an integer tensor
    # must refuse instead of coming back silently wrong.
    directory = str(tmp_path / "src")
    os.makedirs(directory, exist_ok=True)
    save_file({"layer_0": np.array([1, -2, 3], dtype=np.int32)},
              os.path.join(directory, "ints.safetensors"))
    with open(os.path.join(directory, "ints.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"modelID": "org/m", "concept": SOURCE_CONCEPT,
                   "layerCount": 1}, handle)
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        _mirror(tmp_path, directory, name="ints")
    assert caught.value.kind == "unreadableArtifact"
    assert "float tensors" in caught.value.reason


# --- the sidecar ----------------------------------------------------------------

def test_sidecar_preserves_every_sign_invariant_field(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    result = _mirror(tmp_path, source)

    with open(os.path.join(source, f"{SOURCE_CONCEPT}.json"),
              encoding="utf-8") as handle:
        parent = json.load(handle)
    with open(result.sidecar_path, encoding="utf-8") as handle:
        mirrored = json.load(handle)

    # Norms are SIGN-INVARIANT: ‖−v‖ = ‖v‖, so a mirrored artifact's α in norm
    # units means exactly the dose the parent's did, and the whole
    # residualNorm* denominator family travels verbatim.
    for key in ("normsPerLayer", "residualNormPerLayer", "residualNormSource",
                "residualNormConvention", "readingPosition",
                "coversModelDepth", "modelID", "revision", "substrate",
                "extractionMethod", "recipeMethod", "extractionDate",
                "hiddenSize", "layerCount", "neutralMeanSource",
                "stimulusSetHash", "futureUnknownField"):
        assert mirrored[key] == parent[key], key
    # Every key the parent had, minus the one identity claim that would now be
    # false, plus the two stamps.
    assert set(mirrored) - set(parent) == {"negatedFrom",
                                           "polesSwappedFromSource"}
    assert set(parent) - set(mirrored) == {"recipeIdentityHash"}


def test_the_derivation_block_names_the_parent_bytes(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    result = _mirror(tmp_path, source)
    with open(result.sidecar_path, encoding="utf-8") as handle:
        mirrored = json.load(handle)

    assert mirrored["concept"] == MIRROR_CONCEPT
    stamp = mirrored["negatedFrom"]
    assert set(stamp) == {"path", "sha256TensorHash", "sha256SidecarHash",
                          "concept", "date"}
    assert stamp["concept"] == SOURCE_CONCEPT
    assert stamp["path"] == os.path.join(source, SOURCE_CONCEPT)
    # The hashes are the PARENT's bytes, which is what makes the stamp
    # checkable: negate the named bytes and you get these bytes back.
    import hashlib
    assert stamp["sha256TensorHash"] == hashlib.sha256(
        _read(os.path.join(source, f"{SOURCE_CONCEPT}.safetensors"))).hexdigest()
    assert stamp["sha256SidecarHash"] == hashlib.sha256(
        _read(os.path.join(source, f"{SOURCE_CONCEPT}.json"))).hexdigest()
    assert stamp["date"].endswith("Z")


def test_the_stimulus_hash_travels_and_is_qualified(tmp_path):
    # The mirrored pole's stimuli ARE the parent's two files with the roles
    # swapped: a fresh hash would claim different bytes were read, and the
    # parent's hash carried silently would claim the same recipe.
    source = _write_artifact(str(tmp_path / "src"))
    result = _mirror(tmp_path, source)
    with open(result.sidecar_path, encoding="utf-8") as handle:
        mirrored = json.load(handle)
    assert mirrored["stimulusSetHash"] == "stim-hash"
    assert mirrored["polesSwappedFromSource"] is True


def test_recipe_identity_hash_is_dropped(tmp_path):
    # It is an identity claim about THESE bytes ("this recipe produces this
    # artifact"), its canonical form includes the concept name, and promotion
    # matches candidates on it — so carrying the parent's onto a renamed,
    # negated artifact is a wrong answer rather than a missing one.
    source = _write_artifact(str(tmp_path / "src"))
    result = _mirror(tmp_path, source)
    with open(result.sidecar_path, encoding="utf-8") as handle:
        assert "recipeIdentityHash" not in json.load(handle)


def test_the_sidecar_decodes_through_the_store_dataclass(tmp_path):
    from steerlab_server.steering import vector_store
    source = _write_artifact(str(tmp_path / "src"))
    result = _mirror(tmp_path, source)
    with open(result.sidecar_path, encoding="utf-8") as handle:
        sidecar = vector_store.SteeringVectorSidecar.from_dict(json.load(handle))
    assert sidecar.polesSwappedFromSource is True
    assert sidecar.negatedFrom["concept"] == SOURCE_CONCEPT


# --- the catalog ----------------------------------------------------------------

def test_the_catalog_lists_the_mirrored_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    source = _write_artifact(os.path.join(str(tmp_path), "runs", "src"))
    pole_mirror.mirror_poles(
        source, SOURCE_CONCEPT, concept=MIRROR_CONCEPT,
        run_directory=os.path.join(str(tmp_path), "runs", "mirror"))

    listed = {(a.concept, a.name) for a in catalog.list_vectors(str(tmp_path))}
    assert (SOURCE_CONCEPT, SOURCE_CONCEPT) in listed
    assert (MIRROR_CONCEPT, MIRROR_CONCEPT) in listed


# --- refusals -------------------------------------------------------------------

def test_a_missing_source_refuses_with_the_artifact_shape(tmp_path):
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        _mirror(tmp_path, str(tmp_path / "nowhere"))
    assert caught.value.kind == "sourceNotFound"
    assert ".safetensors PLUS its" in caught.value.reason


def test_the_source_concept_is_not_an_acceptable_new_name(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        _mirror(tmp_path, source, concept=SOURCE_CONCEPT)
    assert caught.value.kind == "conceptRequired"
    assert "pointing opposite ways" in caught.value.reason


def test_a_blank_concept_refuses_with_the_reason_it_is_required(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        _mirror(tmp_path, source, concept="   ")
    assert caught.value.kind == "conceptRequired"


def test_mirroring_never_replaces_an_artifact(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    destination = os.path.join(str(tmp_path), "out")
    _mirror(tmp_path, source, out="out")
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        _mirror(tmp_path, source, out="out")
    assert caught.value.kind == "destinationOccupied"
    assert "never replaces an artifact" in caught.value.reason
    assert os.path.isfile(os.path.join(destination,
                                       f"{MIRROR_CONCEPT}.safetensors"))


def test_a_double_mirror_names_the_original(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    first = _mirror(tmp_path, source, out="one")
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        pole_mirror.mirror_poles(
            os.path.dirname(first.vectors_path), MIRROR_CONCEPT,
            concept=SOURCE_CONCEPT,
            run_directory=os.path.join(str(tmp_path), "two"))
    assert caught.value.kind == "doubleMirror"
    assert os.path.join(source, SOURCE_CONCEPT) in caught.value.repair_action


def test_an_output_name_must_be_a_file_name_component(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    with pytest.raises(pole_mirror.PoleMirrorError):
        _mirror(tmp_path, source, output_name="a/b")


def test_the_source_is_never_modified(tmp_path):
    source = _write_artifact(str(tmp_path / "src"))
    before = (_read(os.path.join(source, f"{SOURCE_CONCEPT}.safetensors")),
              _read(os.path.join(source, f"{SOURCE_CONCEPT}.json")))
    _mirror(tmp_path, source)
    assert before == (
        _read(os.path.join(source, f"{SOURCE_CONCEPT}.safetensors")),
        _read(os.path.join(source, f"{SOURCE_CONCEPT}.json")))


# --- the CLI verb ---------------------------------------------------------------

def _root(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    return _write_artifact(os.path.join(str(tmp_path), "runs", "src"))


def _run_dir(tmp_path, slug):
    runs = os.path.join(str(tmp_path), "runs")
    hits = [d for d in os.listdir(runs) if d.endswith(slug)]
    assert len(hits) == 1, (slug, os.listdir(runs))
    return os.path.join(runs, hits[0])


def test_cli_mints_into_a_fresh_run_directory(tmp_path, monkeypatch, capsys):
    _root(tmp_path, monkeypatch)
    code = cli.main(["vectors", "mirror-poles", "runs/src/brightness",
                     "--concept", MIRROR_CONCEPT, "--json"])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    result = document["result"]
    assert result["concept"] == MIRROR_CONCEPT
    assert result["sourceConcept"] == SOURCE_CONCEPT
    assert result["polesSwappedFromSource"] is True
    assert result["layerCount"] == 2
    # The success message names the file a researcher must author, and this
    # verb writes NOTHING into prompts/concepts/ itself.
    assert result["validationAuthoring"] == (
        f"to validate the mirrored pole, author prompts/concepts/"
        f"{MIRROR_CONCEPT}/validation.jsonl — the source concept's rows with "
        "every expresses label inverted are the natural starting point")
    assert not os.path.exists(os.path.join(str(tmp_path), "prompts",
                                           "concepts", MIRROR_CONCEPT))
    run_dir = _run_dir(tmp_path, f"mirror-{MIRROR_CONCEPT}")
    assert os.path.isfile(os.path.join(run_dir, f"{MIRROR_CONCEPT}.json"))
    # A run directory this engine wrote carries its config.json.
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as handle:
        assert json.load(handle)["runType"] == "pole-mirror"


def test_cli_refuses_a_missing_concept_with_the_reason(tmp_path, monkeypatch,
                                                       capsys):
    _root(tmp_path, monkeypatch)
    code = cli.main(["vectors", "mirror-poles", "runs/src/brightness",
                     "--json"])
    assert code == 64
    document = json.loads(capsys.readouterr().out)
    assert document["error"]["code"] == "usage"
    assert "pointing opposite ways" in document["error"]["reason"]
    assert "--concept" in document["error"]["repairAction"]
    # A refused mint leaves no empty run directory behind.
    assert not [d for d in os.listdir(os.path.join(str(tmp_path), "runs"))
                if d.endswith(f"mirror-{MIRROR_CONCEPT}")]


def test_cli_refuses_a_missing_source(tmp_path, monkeypatch, capsys):
    _root(tmp_path, monkeypatch)
    code = cli.main(["vectors", "mirror-poles", "runs/src/absent",
                     "--concept", MIRROR_CONCEPT, "--json"])
    assert code == 66
    document = json.loads(capsys.readouterr().out)
    assert document["error"]["code"] == "notFound"
    assert not [d for d in os.listdir(os.path.join(str(tmp_path), "runs"))
                if "mirror-" in d]


def test_cli_refuses_a_double_mirror(tmp_path, monkeypatch, capsys):
    _root(tmp_path, monkeypatch)
    assert cli.main(["vectors", "mirror-poles", "runs/src/brightness",
                     "--concept", MIRROR_CONCEPT]) == 0
    run_dir = os.path.basename(_run_dir(tmp_path, f"mirror-{MIRROR_CONCEPT}"))
    capsys.readouterr()
    code = cli.main(["vectors", "mirror-poles",
                     f"runs/{run_dir}/{MIRROR_CONCEPT}",
                     "--concept", SOURCE_CONCEPT, "--json"])
    assert code == 65
    document = json.loads(capsys.readouterr().out)
    assert document["error"]["code"] == "doubleMirror"
    assert "already exists on disk" in document["error"]["reason"]


def test_cli_declares_an_undeclared_flag_as_usage(tmp_path, monkeypatch,
                                                  capsys):
    _root(tmp_path, monkeypatch)
    code = cli.main(["vectors", "mirror-poles", "runs/src/brightness",
                     "--concept", MIRROR_CONCEPT, "--nope", "--json"])
    assert code == 64

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
                   "extractionMethod": "meanDifference",
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


# --- malformed safetensors headers (round 8, finding 5) -------------------------
#
# The header is untrusted input and this transform WRITES at the offsets it
# names, so every malformed shape must become the typed cross-engine refusal
# with the twin text — never a bare KeyError/TypeError, and never a silent
# no-op. Swift twin: `PoleMirrorTests` (`negatedTensorBytes`'s guards).

def _payload(header: dict, body: bytes = b"") -> bytes:
    import struct as _struct
    blob = json.dumps(header).encode("utf-8")
    return _struct.pack("<Q", len(blob)) + blob + body


@pytest.mark.parametrize("entry,fragment", [
    # A reversed range: `-8 % 4 == 0` in Python, so the modulo test passed, the
    # byte range was empty, and the tensor counted as flipped while nothing
    # flipped at all.
    ({"dtype": "F32", "data_offsets": [8, 0]}, "do not describe whole F32"),
    # A negative start addresses backwards, out of the payload and into the
    # header this transform is supposed to copy unchanged.
    ({"dtype": "F32", "data_offsets": [-16, 0]}, "do not describe whole F32"),
    ({"dtype": "F32", "data_offsets": [-16, 16]}, "do not describe whole F32"),
    # Shapes that used to escape as a raw KeyError/TypeError.
    ({"dtype": "F32"}, "no readable dtype/data_offsets"),
    ({"dtype": "F32", "data_offsets": None}, "no readable dtype/data_offsets"),
    ({"dtype": "F32", "data_offsets": 16}, "no readable dtype/data_offsets"),
    ({"dtype": "F32", "data_offsets": [0]}, "no readable dtype/data_offsets"),
    ({"dtype": "F32", "data_offsets": [0, 4, 8]},
     "no readable dtype/data_offsets"),
    ({"dtype": "F32", "data_offsets": ["0", "4"]},
     "no readable dtype/data_offsets"),
    ({"dtype": "F32", "data_offsets": [True, False]},
     "no readable dtype/data_offsets"),
    ({"data_offsets": [0, 4]}, "no readable dtype/data_offsets"),
    ({"dtype": 32, "data_offsets": [0, 4]}, "no readable dtype/data_offsets"),
])
def test_a_malformed_header_entry_is_a_typed_refusal(entry, fragment):
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        pole_mirror.negate_layer_tensors(
            _payload({"layer_0": entry}, b"\x00" * 16))
    assert caught.value.kind == "unreadableArtifact"
    assert fragment in caught.value.reason
    assert caught.value.repair_action == (
        "re-extract the source artifact; its .safetensors is corrupt")


def test_a_reversed_range_never_counts_as_a_flipped_tensor():
    """The specific silence: `[8, 0]` flips nothing, so a payload whose only
    layer tensor carries it must not be reported as mirrored."""
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        pole_mirror.negate_layer_tensors(
            _payload({"layer_0": {"dtype": "F32", "data_offsets": [8, 0]}},
                     b"\x00" * 16))
    assert "no layer_<i> tensors" not in caught.value.reason


# --- the artifact PAIR is written atomically (round 8, finding 6) ---------------

def test_a_failure_between_the_two_writes_leaves_no_debris(tmp_path,
                                                           monkeypatch):
    """A tensor with no sidecar is not half an artifact, it is an unreadable
    one — and the `destinationOccupied` rule then refuses to replace it. Both
    files land under temporary names and are promoted together."""
    source = _write_artifact(str(tmp_path / "src"))
    out = os.path.join(str(tmp_path), "mirrored")
    real_open = open
    seen: list[str] = []

    def failing_open(file, mode="r", *args, **kwargs):
        if isinstance(file, str) and file.endswith(".partial"):
            seen.append(file)
            if file.endswith(".json.partial") or ".json." in file:
                raise OSError("no space left on device")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)
    with pytest.raises(OSError):
        pole_mirror.mirror_poles(source, SOURCE_CONCEPT,
                                 concept=MIRROR_CONCEPT, run_directory=out)
    monkeypatch.undo()
    # Neither the promoted pair nor the temporaries survive.
    assert os.listdir(out) == [], os.listdir(out)
    # …and the cleanup ran because the WRITE failed, not because a typed
    # refusal was raised: an OSError is not a PoleMirrorError.
    assert seen


# --- the preflight-to-promotion window (round 9, finding 7) --------------------

def test_a_tensor_that_appears_mid_promote_is_refused_not_overwritten(
        tmp_path, monkeypatch):
    """The window the `destinationOccupied` preflight cannot cover.

    The check runs BEFORE the tensors are negated and both temporaries are
    written; `os.replace` then clobbered whatever arrived in between — in the
    name of a rule that had just refused exactly that. The promotion cannot
    overwrite now, so the appearance is the same refusal it would have been a
    moment earlier, and nothing this call created is left behind."""
    source = _write_artifact(str(tmp_path / "src"))
    out = os.path.join(str(tmp_path), "mirrored")
    intruder = b"another writer's tensors"
    real_commit = pole_mirror._commit_no_replace

    def _appear_then_commit(staged, dest):
        if dest.endswith(".safetensors"):
            # Exactly the window: preflight passed, bytes not yet promoted.
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as handle:
                handle.write(intruder)
        return real_commit(staged, dest)

    monkeypatch.setattr(pole_mirror, "_commit_no_replace", _appear_then_commit)
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        pole_mirror.mirror_poles(source, SOURCE_CONCEPT,
                                 concept=MIRROR_CONCEPT, run_directory=out)
    monkeypatch.undo()
    assert caught.value.kind == "destinationOccupied"
    assert "never replaces an artifact" in caught.value.reason
    # The other writer's bytes are untouched…
    with open(os.path.join(out, f"{MIRROR_CONCEPT}.safetensors"), "rb") as handle:
        assert handle.read() == intruder
    # …and the loser left nothing: no sidecar, no temporaries.
    assert sorted(os.listdir(out)) == [f"{MIRROR_CONCEPT}.safetensors"], \
        os.listdir(out)


def test_a_sidecar_that_appears_mid_promote_takes_the_tensor_back_out(
        tmp_path, monkeypatch):
    """The same window, one file later. The tensor is already promoted when
    the sidecar collides, so the refusal has to unwind a name this call
    created — while leaving the sidecar that beat it exactly where it is. A
    tensor with no sidecar is an unreadable artifact that the occupancy rule
    would then refuse to replace, i.e. a poisoned name."""
    source = _write_artifact(str(tmp_path / "src"))
    out = os.path.join(str(tmp_path), "mirrored")
    intruder = b'{"concept": "another writer\'s sidecar"}'
    real_commit = pole_mirror._commit_no_replace

    def _appear_then_commit(staged, dest):
        if dest.endswith(".json"):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as handle:
                handle.write(intruder)
        return real_commit(staged, dest)

    monkeypatch.setattr(pole_mirror, "_commit_no_replace", _appear_then_commit)
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        pole_mirror.mirror_poles(source, SOURCE_CONCEPT,
                                 concept=MIRROR_CONCEPT, run_directory=out)
    monkeypatch.undo()
    assert caught.value.kind == "destinationOccupied"
    with open(os.path.join(out, f"{MIRROR_CONCEPT}.json"), "rb") as handle:
        assert handle.read() == intruder
    # The tensor this call promoted came back out; only the winner remains.
    assert sorted(os.listdir(out)) == [f"{MIRROR_CONCEPT}.json"], \
        os.listdir(out)


def test_the_no_replace_primitive_cannot_overwrite(tmp_path):
    """The primitive itself, on its own: link-or-reserve, never replace, and
    the staged file survives a collision for the caller to clean up. Twin of
    `bundles._commit_no_replace` / `client.runner._commit_no_replace`."""
    staged = str(tmp_path / "staged")
    dest = str(tmp_path / "dest")
    with open(staged, "wb") as handle:
        handle.write(b"mine")
    pole_mirror._commit_no_replace(staged, dest)
    with open(dest, "rb") as handle:
        assert handle.read() == b"mine"
    # The staged name is consumed by a successful promotion.
    assert not os.path.exists(staged)

    with open(staged, "wb") as handle:
        handle.write(b"second")
    with pytest.raises(FileExistsError):
        pole_mirror._commit_no_replace(staged, dest)
    with open(dest, "rb") as handle:
        assert handle.read() == b"mine"
    assert os.path.exists(staged), \
        "a refused promotion leaves the staged file for the caller's cleanup"


def test_a_non_numeric_layer_count_refuses_before_anything_is_written(tmp_path):
    """`layerCount` used to be converted AFTER both files had landed, so a
    sidecar carrying it as a string stranded a complete artifact pair and then
    raised a bare ValueError past every typed refusal."""
    source = _write_artifact(str(tmp_path / "src"), extras={"layerCount": "2"})
    out = os.path.join(str(tmp_path), "mirrored")
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        pole_mirror.mirror_poles(source, SOURCE_CONCEPT,
                                 concept=MIRROR_CONCEPT, run_directory=out)
    assert caught.value.kind == "unreadableArtifact"
    assert "is not a steering-vector artifact" in caught.value.reason
    assert not os.path.isdir(out) or os.listdir(out) == []


def test_layer_count_must_be_a_whole_positive_number(tmp_path):
    """Review round 10, finding 9: a NUMBER is not yet a layer count.

    ``2.5`` truncated and stamped a mirror claiming a depth its source never
    had; ``0`` and ``-3`` stamped an impossible one; ``nan``/``inf`` reached
    ``int()`` and raised a bare ValueError/OverflowError past every typed
    refusal (the Swift twin TRAPS there). Both engines now check finiteness
    and integrality BEFORE converting, and refuse with the value named. Swift
    twin: ``PoleMirrorTests.layerCountMustBeAWholePositiveNumber``."""
    for index, (value, spelled) in enumerate([
            (2.5, "2.5"), (0, "0"), (-3, "-3"), (0.0, "0"),
            (float("nan"), "NaN"), (float("inf"), "Infinity"),
            (float("-inf"), "-Infinity")]):
        source = _write_artifact(str(tmp_path / f"src{index}"),
                                 extras={"layerCount": value})
        out = os.path.join(str(tmp_path), f"mirrored{index}")
        with pytest.raises(pole_mirror.PoleMirrorError) as caught:
            pole_mirror.mirror_poles(source, SOURCE_CONCEPT,
                                     concept=MIRROR_CONCEPT,
                                     run_directory=out)
        assert caught.value.kind == "unreadableArtifact"
        assert f"records layerCount {spelled}" in caught.value.reason, value
        assert ("a steering-vector artifact's layer count is a whole number "
                "of layers, 1 or more") in caught.value.reason
        assert caught.value.repair_action
        # A refusal writes nothing.
        assert not os.path.isdir(out) or os.listdir(out) == []

    # A VALID artifact still mirrors — the gate is not refusing everything.
    source = _write_artifact(str(tmp_path / "good"))
    result = pole_mirror.mirror_poles(
        source, SOURCE_CONCEPT, concept=MIRROR_CONCEPT,
        run_directory=os.path.join(str(tmp_path), "good-out"))
    assert result.layer_count == 2
    # …and an integral FLOAT is a whole number of layers, so it passes.
    source = _write_artifact(str(tmp_path / "float2"),
                             extras={"layerCount": 2.0})
    result = pole_mirror.mirror_poles(
        source, SOURCE_CONCEPT, concept=MIRROR_CONCEPT,
        run_directory=os.path.join(str(tmp_path), "float2-out"))
    assert result.layer_count == 2


def test_a_commit_that_cannot_drop_its_staging_name_takes_the_destination_back(
        tmp_path, monkeypatch):
    """Review round 10, finding 8, on all three mirrors of this primitive.

    ``_commit_no_replace`` lands ``dest`` and THEN drops the staging name. A
    failure in that last step propagated with ``dest`` in place — and the
    caller's cleanup owns the TEMPORARIES, not the destination, so a
    half-final artifact was left wearing the final name: exactly the state
    ``destinationOccupied`` then refuses to repair. ``dest`` is this call's
    reservation; a commit that cannot finish must not leave it."""
    staged = str(tmp_path / "staged")
    dest = str(tmp_path / "dest")
    with open(staged, "wb") as handle:
        handle.write(b"bytes")

    real_remove = os.remove

    def failing_remove(path, *args, **kwargs):
        if path == staged:
            raise OSError("staging directory went read-only")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(pole_mirror.os, "remove", failing_remove)
    with pytest.raises(OSError):
        pole_mirror._commit_no_replace(staged, dest)
    assert not os.path.exists(dest), \
        "a commit that could not finish left its destination standing"

    # BOTH-OR-NEITHER at the CALLER too: the vectors file lands, the SIDECAR
    # commit then fails, and the vectors file it already landed comes back
    # out. A tensor with no sidecar is what the catalog reads as an
    # unreadable artifact and what `destinationOccupied` then refuses to
    # replace, so half a promotion is worse than none.
    monkeypatch.undo()
    source = _write_artifact(str(tmp_path / "src"))
    out = os.path.join(str(tmp_path), "mirrored")
    calls = {"n": 0}
    real_commit = pole_mirror._commit_no_replace

    def sidecar_commit_fails(staged_path, dest_path):
        calls["n"] += 1
        if calls["n"] == 2:  # the sidecar, after the vectors landed
            assert os.path.exists(os.path.join(out, f"{MIRROR_CONCEPT}"
                                               ".safetensors"))
            raise OSError("the disk filled between the two commits")
        return real_commit(staged_path, dest_path)

    monkeypatch.setattr(pole_mirror, "_commit_no_replace", sidecar_commit_fails)
    with pytest.raises(OSError):
        pole_mirror.mirror_poles(source, SOURCE_CONCEPT,
                                 concept=MIRROR_CONCEPT, run_directory=out)
    monkeypatch.undo()
    # Neither final name, and neither temporary: nothing of this promotion
    # survives it.
    assert not os.path.isdir(out) or os.listdir(out) == [], \
        f"a half-landed promotion left {os.listdir(out)}"

    # …and the two fixes COMPOSE: the sidecar's own staged-removal failing is
    # the primitive's problem, and the caller still ends up with neither name.
    source = _write_artifact(str(tmp_path / "src2"))
    out2 = os.path.join(str(tmp_path), "mirrored2")
    sidecar_temp_failed = {"seen": False}

    def remove_failing_on_sidecar_temp(path, *args, **kwargs):
        if path.endswith(".partial") and ".json." in os.path.basename(path):
            sidecar_temp_failed["seen"] = True
            raise OSError("staging directory went read-only")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(pole_mirror.os, "remove",
                        remove_failing_on_sidecar_temp)
    with pytest.raises(OSError):
        pole_mirror.mirror_poles(source, SOURCE_CONCEPT,
                                 concept=MIRROR_CONCEPT, run_directory=out2)
    monkeypatch.undo()
    assert sidecar_temp_failed["seen"]
    assert not os.path.exists(os.path.join(out2, f"{MIRROR_CONCEPT}.json")), \
        "the sidecar the primitive could not finish committing was left behind"


# --- only the paired, source-concept-bearing family (round 8, finding 2) -------

def test_only_the_paired_source_concept_bearing_family_is_mirrorable():
    from steerlab_server.steering.vector_math import ExtractionMethod

    assert pole_mirror.mirrorable_method_list() == "lat, meanDifference"
    for method in ExtractionMethod:
        if method not in pole_mirror.mirrorable_methods():
            assert not (method.is_paired and method.has_source_concept)
    # designatedReference is source-concept-BEARING and still excluded: it is
    # unpaired, so its negation is "the reference corpus minus the concept" —
    # a different comparison, not the opposite pole.
    assert ExtractionMethod.DESIGNATED_REFERENCE.has_source_concept
    assert not ExtractionMethod.DESIGNATED_REFERENCE.is_paired
    assert ExtractionMethod.DESIGNATED_REFERENCE \
        not in pole_mirror.mirrorable_methods()


@pytest.mark.parametrize("method,label", [
    ("emotionGrandMean", "Grand mean (multi-concept)"),
    ("designatedReference",
     "Designated reference (stories − reference stories)"),
    ("optvec", "Optimized injection vector (OptVec)"),
    ("gemmaScopeSAE", "Gemma Scope SAE feature (decoder row)"),
    ("repeReaderLAT", "RepE reader LAT (derived from a fitted reader)"),
])
def test_an_unmirrorable_method_is_refused_with_its_own_name(tmp_path, method,
                                                             label):
    source = _write_artifact(str(tmp_path / f"src-{method}"),
                             extras={"extractionMethod": method})
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        _mirror(tmp_path, source, out=f"out-{method}")
    assert caught.value.kind == "unmirrorableMethod"
    assert method in caught.value.reason
    assert label in caught.value.reason
    assert "PAIRED, source-concept-bearing" in caught.value.reason
    assert "lat, meanDifference" in caught.value.reason
    assert caught.value.repair_action == \
        pole_mirror.UNMIRRORABLE_METHOD_REPAIR
    assert "NEGATIVE α" in caught.value.repair_action


@pytest.mark.parametrize("recorded", ["", "notAMethod"])
def test_an_unknown_or_absent_method_is_refused_too(tmp_path, recorded):
    source = _write_artifact(str(tmp_path / f"src-{recorded or 'none'}"),
                             extras={"extractionMethod": recorded})
    with pytest.raises(pole_mirror.PoleMirrorError) as caught:
        _mirror(tmp_path, source, out=f"out-{recorded or 'none'}")
    assert caught.value.kind == "unmirrorableMethod"
    assert ("records no extractionMethod" if not recorded
            else "which this engine does not know") in caught.value.reason


def test_cli_refuses_an_unmirrorable_method(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("STEERLAB_RUN_ROOT", raising=False)
    _write_artifact(os.path.join(str(tmp_path), "runs", "src"),
                    extras={"extractionMethod": "repeReaderLAT"})
    code = cli.main(["vectors", "mirror-poles", "runs/src/brightness",
                     "--concept", MIRROR_CONCEPT, "--json"])
    assert code == 65
    document = json.loads(capsys.readouterr().out)
    assert document["error"]["code"] == "unmirrorableMethod"
    assert "PAIRED, source-concept-bearing" in document["error"]["reason"]
    assert "NEGATIVE α" in document["error"]["repairAction"]
    # Nothing minted, and no run directory left behind.
    assert not [d for d in os.listdir(os.path.join(str(tmp_path), "runs"))
                if "mirror-" in d]


# --- the mirrored-pole LIFECYCLE, end to end (round 8, finding 1) --------------
#
# mint → author the swapped stimulus files and the inverted validation →
# attach the MINTED artifact under the mirrored concept → verify clean.
#
# The bug this pins: ``stimulusSetHash`` is ``sha256(positive ‖ negative)`` and
# therefore ORDER-SENSITIVE, the mirrored sidecar carries the PARENT's hash
# (qualified ``polesSwappedFromSource``), and the mirrored concept's own
# directory holds those same two files role-swapped. Attach recomputed the
# concept directory's ordinary hash, compared it to the inherited one, and
# refused every mirror ever minted — the verb produced an artifact no study
# could cite. The fix compares the right CLAIM (these files in the source's
# order) rather than weakening any hash, so a directory holding the wrong
# bytes, or the right bytes in the WRONG order, still refuses.
#
# Swift twin: ``Tests/ExperimentKitTests/PoleMirrorTests.swift``
# (``MirroredPoleAttachTests``).

POSITIVE_ROWS = '{"text": "the lamp is on"}\n{"text": "the room is lit"}\n'
NEGATIVE_ROWS = '{"text": "the lamp is off"}\n{"text": "the room is dark"}\n'


def _write_concept(root, name, positive, negative, validation=None):
    from steerlab_server.experiment import paths
    directory = paths.concept_directory(name, root)
    os.makedirs(directory, exist_ok=True)
    for filename, body in (("positive.jsonl", positive),
                           ("negative.jsonl", negative)):
        with open(os.path.join(directory, filename), "w",
                  encoding="utf-8") as handle:
            handle.write(body)
    if validation is not None:
        with open(os.path.join(directory, "validation.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(validation)
    return directory


def _mirror_workspace(tmp_path, *, mirror_positive=NEGATIVE_ROWS,
                      mirror_negative=POSITIVE_ROWS, mirror_validation=None,
                      author_mirror=True, revision=None,
                      neutral_corpus_hash=None):
    """A source concept, a CAA artifact extracted from it, its minted mirror,
    and (optionally) the mirrored concept's own stimulus directory."""
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.steering.stimulus_set import StimulusSet

    root = str(tmp_path)
    source_dir = _write_concept(
        root, SOURCE_CONCEPT, POSITIVE_ROWS, NEGATIVE_ROWS,
        validation='{"text": "she squinted", "expresses": true}\n')
    source_set = StimulusSet.from_directory(source_dir)
    extras = {"stimulusSetHash": source_set.hash, "substrate": None}
    if neutral_corpus_hash:
        extras["neutralCorpusHash"] = neutral_corpus_hash
    _write_artifact(os.path.join(root, "runs", "src"), extras=extras)
    es.create("mirror-study", model_id="org/m", revision=revision, root=root)
    pole_mirror.mirror_poles(
        os.path.join(root, "runs", "src"), SOURCE_CONCEPT,
        concept=MIRROR_CONCEPT,
        run_directory=os.path.join(root, "runs", "mirrored"))
    if author_mirror:
        _write_concept(root, MIRROR_CONCEPT, mirror_positive, mirror_negative,
                       validation=mirror_validation)
    return root, source_set


def test_the_minted_mirror_attaches_under_its_own_concept_and_verifies(tmp_path):
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment import paths
    from steerlab_server.experiment.manifest import Manifest
    from steerlab_server.steering.stimulus_set import StimulusSet

    root, source_set = _mirror_workspace(
        tmp_path,
        mirror_validation='{"text": "she squinted", "expresses": false}\n')
    mirror_set = StimulusSet.from_directory(
        paths.concept_directory(MIRROR_CONCEPT, root))
    # The heart of the finding, stated as an assertion: the mirrored concept's
    # OWN hash is NOT the parent's, and its swapped-order hash IS.
    assert mirror_set.hash != source_set.hash
    assert mirror_set.poles_swapped_hash == source_set.hash

    document = es.attach_artifact(
        "mirror-study", MIRROR_CONCEPT, f"runs/mirrored/{MIRROR_CONCEPT}",
        root=root)
    entry = next(c for c in document["concepts"] if c["name"] == MIRROR_CONCEPT)

    # The LIVE pin is the mirrored concept's own hash — the value every later
    # verify recomputes from prompts/concepts/.
    assert entry["stimulusSetHash"] == mirror_set.hash
    pin = entry["vectorArtifact"]
    # …and the sidecar linkage travels beside it, so the manifest states both
    # halves of the claim.
    assert pin["polesSwappedFromSource"] is True
    assert pin["sourceStimulusSetHash"] == source_set.hash
    assert pin["sourceConcept"] == MIRROR_CONCEPT
    assert pin["sourceMethod"] == "meanDifference"

    # Validation pins the MIRRORED concept's own file, by the normal
    # source-concept rules — not the source's, and not absent.
    def _hash(concept):
        path = os.path.join(paths.concept_directory(concept, root),
                            "validation.jsonl")
        import hashlib
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    assert entry["validationHash"] == _hash(MIRROR_CONCEPT)
    assert entry["validationHash"] != _hash(SOURCE_CONCEPT)

    # Verify passes, and it re-proves BOTH hashes.
    assert Manifest.load("mirror-study", root=root).verify(root=root) == []


@pytest.mark.parametrize("label,positive,negative", [
    ("same order as the source", POSITIVE_ROWS, NEGATIVE_ROWS),
    ("different bytes", NEGATIVE_ROWS, '{"text": "elsewhere"}\n'),
])
def test_a_mirrored_concept_whose_files_are_not_swapped_is_refused(
        tmp_path, label, positive, negative):
    """Nothing was loosened. The swapped claim is CHECKED."""
    from steerlab_server.experiment import experiment_store as es

    root, _ = _mirror_workspace(tmp_path, mirror_positive=positive,
                                mirror_negative=negative)
    with pytest.raises(es.ExperimentStoreError) as caught:
        es.attach_artifact("mirror-study", MIRROR_CONCEPT,
                           f"runs/mirrored/{MIRROR_CONCEPT}", root=root)
    assert "is a MIRRORED pole" in str(caught.value), label
    assert "roles SWAPPED" in str(caught.value)


def test_verify_re_proves_the_swap_after_the_files_are_reordered(tmp_path):
    """Verify checks the LINKAGE, not only the concept's own hash: putting the
    mirrored concept's files back into the source's order breaks both, and the
    mirror-specific violation is the one that says why."""
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest

    root, _ = _mirror_workspace(tmp_path)
    es.attach_artifact("mirror-study", MIRROR_CONCEPT,
                       f"runs/mirrored/{MIRROR_CONCEPT}", root=root)
    assert Manifest.load("mirror-study", root=root).verify(root=root) == []
    _write_concept(root, MIRROR_CONCEPT, POSITIVE_ROWS, NEGATIVE_ROWS)
    violations = Manifest.load("mirror-study", root=root).verify(root=root)
    assert any("is a MIRRORED pole" in v for v in violations), violations


# --- the mirrored-pole PIPELINE: materialize → promote (2026-08-28) -------------
#
# The second half of the lifecycle above. Attach passes, but promotion could
# still never succeed: the run's materialized copy stamps the artifact's OWN
# stimulus identity — the PARENT's hash, qualified ``polesSwappedFromSource``
# — while ``required_identity`` demanded the ref's own (role-swapped
# directory) hash, so the copy's ``recipeIdentityHash`` could never equal the
# required identity and ``promote._matching_vector_artifact`` refused every
# mirrored pinned concept with a stimulusSetHash diff. The fix: a mirrored
# pin's identity demands the pin's ``sourceStimulusSetHash`` (what every
# faithful materialization stamps), and the copy carries the mirror stamps
# (``polesSwappedFromSource`` + ``negatedFrom``) through materialization so
# it never under-claims what it knows — the same defect class as the Gemma
# Scope convention stamps (open-issues #14).
#
# Swift note: materialization is server-only (Swift ``extractAll`` refuses
# pinned concepts), but ``RecipeIdentity.required`` mirrors the same pin
# branches so the two engines agree on what identity a mirrored pin demands.

NEUTRAL_CORPUS_HASH = "c" * 64


def _promotable_workspace(tmp_path):
    """The attach e2e's workspace, made promotable: a pinned revision (the
    identity covers it) and a norm-corpus hash on the source artifact (a
    neutral-corpus denominator without its corpus hash is an unprovable
    recipe, refused at the writer)."""
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest

    root, source_set = _mirror_workspace(
        tmp_path,
        mirror_validation='{"text": "she squinted", "expresses": false}\n',
        revision="abc", neutral_corpus_hash=NEUTRAL_CORPUS_HASH)
    es.attach_artifact("mirror-study", MIRROR_CONCEPT,
                       f"runs/mirrored/{MIRROR_CONCEPT}", root=root)
    return root, source_set, Manifest.load("mirror-study", root=root)


def test_materialization_carries_the_mirror_stamps_and_a_matching_identity(
        tmp_path):
    from types import SimpleNamespace

    from steerlab_server.experiment import promote, recipe_identity, tasks
    from steerlab_server.steering import vector_store

    root, source_set, manifest = _promotable_workspace(tmp_path)
    model = SimpleNamespace(revision="abc")
    bundles = tasks._extract_all(model, manifest, root)
    bundle = bundles[MIRROR_CONCEPT]
    # The bundle claims exactly what the artifact claims: the PARENT's hash,
    # qualified by the swap stamp, plus the negation linkage.
    assert bundle.stimulus_hash == source_set.hash
    assert bundle.poles_swapped_from_source is True
    assert bundle.negated_from["concept"] == SOURCE_CONCEPT

    run_dir = os.path.join(root, "runs", "20260828T000001000-exp-extract")
    os.makedirs(run_dir)
    tasks._persist_vectors(bundles, manifest, model, run_dir)
    _, sidecar = vector_store.load(run_dir, MIRROR_CONCEPT)
    # The copy carries both mirror stamps — dropping them would claim the
    # mirrored concept's own files were read in their own order.
    assert sidecar.polesSwappedFromSource is True
    assert sidecar.negatedFrom["concept"] == SOURCE_CONCEPT
    assert sidecar.stimulusSetHash == source_set.hash

    # The required identity demands the pin's inherited hash — the value the
    # copy stamps — so the copy's own recipeIdentityHash satisfies it.
    ref = manifest.concepts[0]
    required = recipe_identity.required_identity(manifest, ref)
    assert required["stimulusSetHash"] == source_set.hash
    assert required["stimulusSetHash"] != ref.stimulus_set_hash
    assert sidecar.recipeIdentityHash == recipe_identity.identity_hash(required)

    # …and the production promotion matcher finds it.
    artifact, identity_hash = promote._matching_vector_artifact(
        manifest, ref, root)
    assert artifact.runDirectory == run_dir
    assert identity_hash == sidecar.recipeIdentityHash


def test_the_minted_mirror_promotes_end_to_end_through_a_sweep(tmp_path,
                                                               monkeypatch):
    """attach minted mirror → sweep (which materializes and persists the
    pinned bytes into its own run directory) → promote succeeds, and the
    agent's pinned artifact carries the pole provenance."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment import promote, recipe_identity, tasks
    from steerlab_server.experiment.manifest import Manifest

    root, source_set, _ = _promotable_workspace(tmp_path)

    def _write(path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    _write(os.path.join(root, "prompts", "concepts", MIRROR_CONCEPT,
                        "markers.json"), json.dumps({"words": ["dusk"]}))
    _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
           '{"text": "Write about the town."}\n')
    _write(os.path.join(root, "prompts", "batteries", "b.jsonl"),
           '{"prompt": "What is 1+1?", "answer": "2"}\n')
    d = es.load_raw("mirror-study", root)
    d["sweep"] = {"layerFractions": [0.5], "alphas": [0.4],
                  "devPromptsFile": "prompts/dev/dev.jsonl",
                  "batteryFile": "prompts/batteries/b.jsonl", "maxTokens": 16}
    es.save_raw(d, root)

    @contextmanager
    def fake_model(model_id, revision):
        yield SimpleNamespace(revision=revision)

    def fake_generate(model, prompt, *, model_id=None, max_tokens=0,
                      temperature=0.0, injections=None, prompt_mode=None,
                      system_prompt=None, qwen_thinking_enabled=False):
        return ("dusk settled over the dim town before night fell 2"
                if injections else
                "the town woke slowly to a bright morning 2")

    monkeypatch.setattr(tasks, "generate", fake_generate)
    run_dir = tasks.sweep("mirror-study", root, model_provider=fake_model,
                          log=lambda *_: None)

    out = promote.promote("mirror-study", MIRROR_CONCEPT, root=root,
                          log=lambda *_: None)
    promotion = out["variant"]["promotion"]
    assert promotion["promotedBy"] == "criterion"
    # The certificate's identity is the mirrored pin's required identity —
    # inherited hash and all.
    manifest = Manifest.load("mirror-study", root)
    required = recipe_identity.required_identity(manifest, manifest.concepts[0])
    assert promotion["recipeIdentityHash"] == \
        recipe_identity.identity_hash(required)
    # The agent injects the sweep run's materialized copy…
    injection = out["variant"]["injections"][0]
    assert injection["vectorArtifactID"] == os.path.join(
        "runs", os.path.basename(run_dir), MIRROR_CONCEPT)
    # …which carries the pole provenance the certificate's pin resolves to:
    # the swap stamp, the negation linkage, and the parent's stimulus hash.
    with open(os.path.join(root, injection["vectorArtifactID"] + ".json"),
              encoding="utf-8") as handle:
        pinned_sidecar = json.load(handle)
    assert pinned_sidecar["polesSwappedFromSource"] is True
    assert pinned_sidecar["negatedFrom"]["concept"] == SOURCE_CONCEPT
    assert pinned_sidecar["stimulusSetHash"] == source_set.hash


def test_extracting_from_the_swapped_files_reproduces_the_minted_direction():
    """The SEMANTICS' proof, and the reason the swapped files are the right
    evidence rather than a bookkeeping convention: CAA's mean difference is
    ANTISYMMETRIC under a file swap, so extracting freshly from the mirrored
    concept's directory reproduces the minted bytes exactly. The workaround
    this finding came in through — "just re-extract from the swapped files" —
    and the mirror are the same vector."""
    from steerlab_server.steering import vector_math as vm

    positive = [[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]]
    negative = [[-0.25, 4.0, 1.0], [0.5, 0.5, 0.5]]
    forward = vm.mean_difference(positive, negative)
    swapped = vm.mean_difference(negative, positive)
    assert [round(v, 12) for v in swapped] == \
        [round(-v, 12) for v in forward]

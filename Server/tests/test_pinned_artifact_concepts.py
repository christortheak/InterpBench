"""Artifact-pinned concepts: a concept whose VECTOR BYTES are the pin.

Motivation (2026-08-10): family-grand-mean-centred directions were derived
post-hoc from existing extraction artifacts — no stimulus recipe reproduces
them, so the recipe-pinning firewall had nothing to bite on and the vectors
could not enter the validate → sweep → promote lifecycle at all.

Contracts asserted here:
- attach reads the artifact's sidecar (the artifact IS the recipe) and pins
  the extension-less locator plus BOTH file hashes, refusing anything it
  cannot honestly pin (wrong model, foreign substrate, drifted source data);
- extraction MATERIALIZES the pinned bytes into the run directory as an
  ordinary vector artifact stamped ``pinnedFrom``, after re-checking both
  hashes — drift refuses loudly, naming the file and both hashes;
- validate runs the unchanged held-out probe on the materialized vector, at
  the artifact's own reading position, over the SOURCE concept's data;
- both hashes join the verify()/freeze pin surface;
- "pinnedArtifact" surfaces in the vector catalog and the experiment detail,
  and the materialized artifact matches through the production promotion
  matcher.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import catalog, experiment_store as es, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import vector_math as vm
from steerlab_server.steering import vector_store
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.reading_position import mean_from_token

MODEL = "google/gemma-3-27b-it"
REVISION = "005ad3404e59d6023443cb575daa05336842228a"
LONG_A = "the panel weighed the doctrine against the record for a long while " * 4
LONG_B = "an unhurried afternoon of plain and ordinary errands in the town " * 4


def _stories(root, concept, texts):
    path = os.path.join(root, "prompts", "emotions", concept, "stories.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for t in texts:
            handle.write(json.dumps({"concept": concept, "text": t}) + "\n")
    return path


def _validation(root, concept):
    path = os.path.join(root, "prompts", "emotions", concept, "validation.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": LONG_A + "held out", "expresses": True}) + "\n")
        handle.write(json.dumps({"text": LONG_B + "held out", "expresses": False}) + "\n")
    return path


def _sha256(path):
    import hashlib
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _derived_artifact(root, *, concept="crit-gm", layers=2, hidden=32,
                      model_id=MODEL, revision=REVISION, substrate=None,
                      reading="mean from token 50", family_base="crit",
                      rendering=None):
    """A stand-in for the real family-grand-mean-centred artifacts: the same
    sidecar shape (designatedReference source, pooled reading, per-layer
    residual norms, a familyGrandMeanCentring provenance block)."""
    from steerlab_server.experiment import multiconcept
    run_dir = os.path.join(root, "runs", "20260810T045146213-derived-gm")
    vectors = vector_store.ConceptVectors(
        per_layer=[[float(i + 1) * 0.5] * hidden for i in range(layers)])
    sidecar = vector_store.SteeringVectorSidecar.make(
        model_id=model_id, concept=concept, revision=revision,
        stimulus_set_hash=multiconcept.stories_hash(family_base, root),
        vectors=vectors, extraction_method="designatedReference",
        reading_position=mean_from_token(50),
        residual_norm_per_layer=[100.0] * layers,
        residual_norm_source="neutral-corpus",
        neutral_corpus_hash="c" * 64)
    sidecar.readingPosition = reading
    sidecar.designatedReference = {
        "name": "plain-exposition",
        "hash": multiconcept.stories_hash("plain-exposition", root)}
    sidecar.source = "family-grand-mean-centred"
    sidecar.controlMode = "family-grand-mean-centred activation addition"
    sidecar.__dict__["familyGrandMeanCentring"] = {
        "baseConcept": family_base, "family": "stances", "k": 5}
    if substrate is not None:
        sidecar.substrate = substrate
    if rendering is not None:
        sidecar.extractionRendering = rendering
    vector_store.save(vectors, sidecar, run_dir, concept)
    return os.path.join("runs", "20260810T045146213-derived-gm", concept)


def _workspace(tmp_path, **kwargs):
    root = str(tmp_path)
    _stories(root, "crit", [LONG_A, LONG_A + " again"])
    _stories(root, "plain-exposition", [LONG_B, LONG_B + " again"])
    _validation(root, "crit")
    es.create("gm-study", model_id=MODEL, revision=REVISION, root=root)
    return root, _derived_artifact(root, **kwargs)


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        torch.manual_seed(sum(ord(c) for c in text[:12]))
        return SimpleNamespace(input_ids=torch.randint(1, 127, (1, 80)))


def _tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128,
        max_position_embeddings=256)
    lm = LlamaForCausalLM(config).eval()
    return SimpleNamespace(model=lm, hooked=HookedModel(lm),
                           device=torch.device("cpu"), num_layers=2,
                           tokenizer=_FakeTokenizer(),
                           model_id=MODEL, revision=REVISION)


# --- attach ---------------------------------------------------------------

def test_attach_pins_the_locator_and_both_hashes(tmp_path):
    root, artifact = _workspace(tmp_path)
    d = es.attach_artifact("gm-study", "crit-gm", artifact,
                           source_concept="crit", root=root)
    ref = d["concepts"][0]
    assert ref["options"]["method"] == "pinnedArtifact"
    # The reading position is COPIED from the artifact, never re-declared.
    assert ref["options"]["readingPosition"] == {"meanFromToken": {"_0": 50}}
    block = ref["vectorArtifact"]
    assert block["path"] == artifact
    assert block["sha256TensorHash"] == _sha256(
        os.path.join(root, artifact + ".safetensors"))
    assert block["sha256SidecarHash"] == _sha256(
        os.path.join(root, artifact + ".json"))
    assert block["sourceMethod"] == "designatedReference"
    assert block["sourceConcept"] == "crit"
    assert block["residualNormSource"] == "neutral-corpus"
    assert block["normCorpusHash"] == "c" * 64
    # The source recipe's reference travels, so validation knows both classes.
    assert ref["designatedReference"]["name"] == "plain-exposition"
    # The held-out set is the SOURCE concept's.
    assert ref["validationHash"] == _sha256(
        os.path.join(root, "prompts", "emotions", "crit", "validation.jsonl"))
    assert Manifest.load("gm-study", root).verify(root) == []


def test_manifest_round_trip_resolves_effective_method_and_data_concept(tmp_path):
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    concept = Manifest.load("gm-study", root).concepts[0]
    assert concept.is_pinned_artifact
    assert concept.options.method is vm.ExtractionMethod.PINNED_ARTIFACT
    # The DATA questions resolve to the source recipe, not to "pinnedArtifact".
    assert concept.effective_method is vm.ExtractionMethod.DESIGNATED_REFERENCE
    assert concept.data_concept == "crit"
    assert concept.options.reading_position.label == "mean from token 50"


def test_attach_through_the_generic_verb_and_its_refusals(tmp_path):
    root, artifact = _workspace(tmp_path)
    d = es.attach("gm-study", ["crit-gm"], method="pinnedArtifact",
                  vector_artifact=artifact, source_concept="crit", root=root)
    assert d["concepts"][0]["vectorArtifact"]["path"] == artifact
    with pytest.raises(es.ExperimentStoreError, match="exactly one concept"):
        es.attach("gm-study", ["a", "b"], method="pinnedArtifact",
                  vector_artifact=artifact, root=root)
    with pytest.raises(es.ExperimentStoreError, match="needs the artifact path"):
        es.attach("gm-study", ["crit-gm"], method="pinnedArtifact", root=root)


def test_attach_refuses_a_missing_artifact_and_a_foreign_model(tmp_path):
    root, artifact = _workspace(tmp_path)
    with pytest.raises(es.ExperimentStoreError, match="no vector artifact at"):
        es.attach_artifact("gm-study", "crit-gm", "runs/nope/absent", root=root)
    other = _derived_artifact(tmp_path, concept="wrong-model-gm",
                              model_id="org/other")
    with pytest.raises(es.ExperimentStoreError, match="does not transfer"):
        es.attach_artifact("gm-study", "crit-gm", other, source_concept="crit",
                           root=root)


def test_attach_refuses_a_foreign_substrate_artifact(tmp_path):
    root, _ = _workspace(tmp_path)
    foreign = _derived_artifact(tmp_path, concept="swift-gm",
                                substrate="swift-mlx")
    with pytest.raises(es.ExperimentStoreError, match="do not transfer across engines"):
        es.attach_artifact("gm-study", "swift-gm", foreign,
                           source_concept="crit", root=root)


def test_attach_containment_resolves_symlinks(tmp_path):
    """Review finding 2026-08-10: the lexical relative-path check passes a
    symlink INSIDE the workspace whose target is outside it, so the pin would
    record a workspace-relative locator whose bytes a workspace copy does not
    carry. Containment is real-path (safe_paths discipline)."""
    root, _ = _workspace(tmp_path)
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    _stories(str(outside), "crit", [LONG_A])
    _stories(str(outside), "plain-exposition", [LONG_B])
    escaped = _derived_artifact(str(outside), concept="esc-gm")
    os.symlink(os.path.join(str(outside), "runs"),
               os.path.join(root, "runs", "linked"))
    linked_ref = "runs/linked/" + "/".join(escaped.split("/")[1:])
    with pytest.raises(es.ExperimentStoreError,
                       match="resolves outside the workspace"):
        es.attach_artifact("gm-study", "esc-gm", linked_ref,
                           source_concept="crit", root=root)


def test_attach_names_the_base_concept_when_the_data_is_elsewhere(tmp_path):
    """The renamed-derived-direction trap: attaching 'crit-gm' with no
    --source-concept finds no data under that name. The refusal must SAY
    which concept the artifact was derived from, not merely that something
    is missing."""
    root, artifact = _workspace(tmp_path)
    with pytest.raises(es.ExperimentStoreError,
                       match="try --source-concept crit"):
        es.attach_artifact("gm-study", "crit-gm", artifact, root=root)


def test_attach_refuses_when_the_source_stimuli_have_drifted(tmp_path):
    root, artifact = _workspace(tmp_path)
    _stories(root, "crit", ["something else entirely " * 8])
    with pytest.raises(es.ExperimentStoreError, match="restore the bytes"):
        es.attach_artifact("gm-study", "crit-gm", artifact,
                           source_concept="crit", root=root)


def test_attach_refuses_an_artifact_with_no_norm_provenance(tmp_path):
    """Without residualNormSource the artifact's recipe identity is
    unprovable, so promotion could never match it — refuse at authoring
    rather than at the end of a sweep."""
    root, _ = _workspace(tmp_path)
    from steerlab_server.experiment import multiconcept
    run_dir = os.path.join(root, "runs", "bare")
    vectors = vector_store.ConceptVectors(per_layer=[[1.0] * 32] * 2)
    sidecar = vector_store.SteeringVectorSidecar(
        modelID=MODEL, concept="bare-gm",
        stimulusSetHash=multiconcept.stories_hash("crit", root),
        layerCount=2, hiddenSize=32, normsPerLayer=[1.0, 1.0],
        extractionDate="2026-08-01T00:00:00Z", revision=REVISION,
        extractionMethod="designatedReference",
        readingPosition="mean from token 50")
    vector_store.save(vectors, sidecar, run_dir, "bare-gm")
    with pytest.raises(es.ExperimentStoreError, match="residualNormSource"):
        es.attach_artifact("gm-study", "bare-gm", "runs/bare/bare-gm",
                           source_concept="crit", root=root)


# --- verify / freeze pin surface -----------------------------------------

def test_artifact_drift_is_a_verify_violation(tmp_path):
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    tensor = os.path.join(root, artifact + ".safetensors")
    with open(tensor, "ab") as handle:
        handle.write(b"\x00")
    violations = Manifest.load("gm-study", root).verify(root)
    assert any("vector artifact changed since pinning" in v for v in violations)


def test_sidecar_drift_is_a_verify_violation(tmp_path):
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    sidecar_path = os.path.join(root, artifact + ".json")
    sidecar = json.load(open(sidecar_path))
    sidecar["readingPosition"] = "last token"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, sort_keys=True, indent=2)
    violations = Manifest.load("gm-study", root).verify(root)
    assert any("sidecar changed since pinning" in v for v in violations)


def test_a_half_pin_and_a_stray_block_are_violations(tmp_path):
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    d = es.load_raw("gm-study", root)
    del d["concepts"][0]["vectorArtifact"]["sha256TensorHash"]
    assert any("pin is incomplete" in v
               for v in Manifest.from_dict(d).verify(root))
    d["concepts"][0]["options"]["method"] = "designatedReference"
    assert any("only read under method 'pinnedArtifact'" in v
               for v in Manifest.from_dict(d).verify(root))


def test_reading_position_disagreement_is_a_violation(tmp_path):
    """The manifest declares where held-out activations are read; the
    artifact records where the vector was read. Two answers is no answer."""
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    d = es.load_raw("gm-study", root)
    d["concepts"][0]["options"]["readingPosition"] = {"lastToken": {}}
    assert any("held-out activations must be read where the vector was read"
               in v for v in Manifest.from_dict(d).verify(root))


def test_extraction_rendering_disagreement_is_a_violation(tmp_path):
    """The rendering is recipe identity exactly as the position is, so two
    answers is no answer here either. Compared CANONICALLY: this artifact
    carries no rendering stamp — every legacy artifact's shape — so an absent
    AND an explicitly raw declaration both stay clean, and only a real
    disagreement fires."""
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    d = es.load_raw("gm-study", root)
    assert "extractionRendering" not in d["concepts"][0]["options"]
    marker = "held-out activations must be read as the vector was rendered"
    assert not any(marker in v for v in Manifest.from_dict(d).verify(root))
    d["concepts"][0]["options"]["extractionRendering"] = {"mode": "raw"}
    assert not any(marker in v for v in Manifest.from_dict(d).verify(root))
    d["concepts"][0]["options"]["extractionRendering"] = {"mode": "chatTemplate"}
    violations = Manifest.from_dict(d).verify(root)
    assert any(marker in v for v in violations)
    assert any("contradicts the pinned artifact's 'raw'" in v
               for v in violations)


def test_attach_copies_the_artifacts_rendering_so_verify_passes(tmp_path):
    """Attach must never write a pin the very next verify rejects: the
    rendering travels from the sidecar exactly as the reading position does,
    and the assistant voice travels with it."""
    root, artifact = _workspace(
        tmp_path, rendering={"mode": "chatTemplate", "qwenThinkingEnabled": False,
                             "voice": "assistant"})
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    d = es.load_raw("gm-study", root)
    assert d["concepts"][0]["options"]["extractionRendering"] == {
        "mode": "chatTemplate", "qwenThinkingEnabled": False,
        "voice": "assistant"}
    marker = "held-out activations must be read as the vector was rendered"
    assert not any(marker in v for v in Manifest.from_dict(d).verify(root))
    # …and dropping the declaration afterwards IS the contradiction.
    del d["concepts"][0]["options"]["extractionRendering"]
    assert any(marker in v for v in Manifest.from_dict(d).verify(root))


@pytest.mark.parametrize("label,codable", [
    # The two that always survived…
    ("last token", {"lastToken": {}}),
    ("mean from token 50", {"meanFromToken": {"_0": 50}}),
    # …and the six that used to collapse into `lastToken`.
    ("offset from end 3", {"offsetFromEnd": {"_0": 3}}),
    ("last content token", {"lastContentToken": {}}),
    ("turn close token", {"turnCloseToken": {}}),
    ("post-instruction 2", {"postInstruction": {"_0": 2}}),
    ("content offset 2", {"contentOffset": {"_0": 2}}),
    ("mean content from token 4", {"meanContentFromToken": {"_0": 4}}),
])
def test_attach_carries_every_reading_position_faithfully(
        tmp_path, label, codable):
    """The position travels VERBATIM, all eight of them (external review round
    5).

    The pin used to be rebuilt from the parsed position's
    ``requested_start_index`` — an integer only ``meanFromToken`` has — so six
    of the eight positions were silently rewritten as ``lastToken``. The
    manifest then contradicted the very sidecar it had just read, and the
    verify contradiction check refused the pin attach had written one line
    earlier. Swift's ``ExperimentStore.attachArtifactPin`` copies the position
    itself, and this is that behavior, in the identical Codable shape.
    """
    root, artifact = _workspace(tmp_path, reading=label)
    d = es.attach_artifact("gm-study", "crit-gm", artifact,
                           source_concept="crit", root=root)
    assert d["concepts"][0]["options"]["readingPosition"] == codable
    # …and the pin verify() sees is one it passes: attach never writes a pin
    # the very next verify rejects.
    marker = "held-out activations must be read where the vector was read"
    assert not any(marker in v
                   for v in Manifest.load("gm-study", root).verify(root))
    # The manifest's own reader gets the same position back out of the block.
    assert Manifest.load("gm-study", root).concepts[0] \
        .options.reading_position.label == label


def test_an_unreadable_reading_position_refuses_the_attach_by_name(tmp_path):
    """The position's twin of the rendering refusal below, and strict for the
    same reason: a label this engine cannot parse can only be a NEWER engine's
    position or a typo, and last-token is not a safe guess — it would launder
    a wrong reading into the recipe identity a promotion matches on. Swift
    twin: the ``ReadingPosition(label:)`` guard in ``attachArtifactPin``."""
    root, artifact = _workspace(tmp_path, reading="penultimate token")
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.attach_artifact("gm-study", "crit-gm", artifact,
                           source_concept="crit", root=root)
    message = str(exc.value)
    assert "records reading position 'penultimate token'" in message
    assert "which this engine cannot parse" in message
    assert "re-attach on the engine that wrote it" in message


def test_a_sidecar_with_no_reading_position_attaches_as_legacy_last_token(
        tmp_path):
    """ABSENT is the legacy artifact, and stays the default — the Swift twin
    leaves ``ExtractionOptions``' own default in place when the sidecar records
    none, so a legacy attach still writes byte-identical manifest JSON."""
    root, artifact = _workspace(tmp_path)
    sidecar_path = os.path.join(root, artifact + ".json")
    sidecar = json.load(open(sidecar_path))
    del sidecar["readingPosition"]
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, sort_keys=True, indent=2)
    d = es.attach_artifact("gm-study", "crit-gm", artifact,
                           source_concept="crit", root=root)
    assert d["concepts"][0]["options"]["readingPosition"] == {"lastToken": {}}


def test_a_stranger_in_the_artifacts_rendering_refuses_the_attach_by_name(
        tmp_path):
    """READING is as strict as declaring (review 2026-08-26). A key this
    engine does not understand in a recorded stamp can only be a NEWER engine's
    parameter or a typo, and copying the block minus that key would pin a
    rendering the artifact was not extracted under. The refusal is the store's
    own, naming the artifact and the stranger — never a traceback out of the
    sidecar reader."""
    root, artifact = _workspace(
        tmp_path, rendering={"mode": "chatTemplate",
                             "addGenerationPromt": False})
    with pytest.raises(es.ExperimentStoreError) as exc:
        es.attach_artifact("gm-study", "crit-gm", artifact,
                           source_concept="crit", root=root)
    message = str(exc.value)
    assert "records an extractionRendering this engine cannot read" in message
    assert "addGenerationPromt" in message
    assert "re-attach on the engine that wrote it" in message


def test_a_stranger_in_a_pinned_sidecar_is_a_named_verify_violation(tmp_path):
    """The same strictness at the OTHER read: a study that already pins such an
    artifact reports a violation naming the concept and the sidecar, so verify
    stays a list of findings rather than an exception."""
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    d = es.load_raw("gm-study", root)
    sidecar_path = os.path.join(root, artifact + ".json")
    with open(sidecar_path, encoding="utf-8") as handle:
        sidecar = json.load(handle)
    sidecar["extractionRendering"] = {"mode": "chatTemplate",
                                      "addGenerationPromt": False}
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle)
    # Re-pin the hash, so it is the RENDERING that is under test and not drift.
    d["concepts"][0]["vectorArtifact"]["sha256SidecarHash"] = _sha256(sidecar_path)
    violations = Manifest.from_dict(d).verify(root)
    assert any("declares an extractionRendering this engine cannot read" in v
               for v in violations)
    assert any("addGenerationPromt" in v for v in violations)


def test_pin_surface_carries_both_artifact_files_and_the_source_data(tmp_path):
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    entries = es.pinned_input_entries(es.load_raw("gm-study", root), root)
    labels = {e.label: (e.path, e.required) for e in entries}
    assert labels["concept 'crit-gm' pinned vector artifact"][1] is True
    assert labels["concept 'crit-gm' pinned vector artifact sidecar"][1] is True
    # …and the SOURCE concept's data, because validate reads it live.
    assert labels["concept 'crit-gm' stories.jsonl"][0].endswith(
        os.path.join("emotions", "crit", "stories.jsonl"))
    assert labels["concept 'crit-gm' validation.jsonl"][0].endswith(
        os.path.join("emotions", "crit", "validation.jsonl"))


def test_validation_scope_hash_moves_with_the_artifact_bytes(tmp_path):
    """Evidence minted against one set of bytes must not certify another."""
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    before = Manifest.load("gm-study", root).validation_scope_hash()
    d = es.load_raw("gm-study", root)
    d["concepts"][0]["vectorArtifact"]["sha256TensorHash"] = "0" * 64
    assert Manifest.from_dict(d).validation_scope_hash() != before


def test_freeze_moves_the_artifact_out_of_gitignored_runs(tmp_path):
    """A frozen manifest must not depend on a file the repository does not
    version: both artifact files move into experiments/<name>/pinned/ and the
    locator is repointed — byte-identically, so both pins still verify."""
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    frozen = es.freeze("gm-study", force=True, root=root)
    block = frozen["concepts"][0]["vectorArtifact"]
    assert block["path"].startswith("experiments/gm-study/pinned/")
    for suffix, key in ((".safetensors", "sha256TensorHash"),
                        (".json", "sha256SidecarHash")):
        moved = os.path.join(root, block["path"] + suffix)
        assert os.path.isfile(moved)
        assert _sha256(moved) == block[key]
    assert Manifest.from_dict(frozen).verify(root) == []


# --- materialization ------------------------------------------------------

def _attached(tmp_path):
    root, artifact = _workspace(tmp_path)
    es.attach_artifact("gm-study", "crit-gm", artifact, source_concept="crit",
                       root=root)
    return root, artifact, Manifest.load("gm-study", root)


def test_extract_materializes_the_pinned_vector(tmp_path):
    root, artifact, manifest = _attached(tmp_path)
    model = _tiny_model()
    bundles = tasks._extract_all(model, manifest, root)
    bundle = bundles["crit-gm"]
    source, _ = vector_store.load(
        os.path.dirname(os.path.join(root, artifact)), "crit-gm")
    assert bundle.vectors.per_layer == source.per_layer
    assert bundle.residual_norm_per_layer == [100.0, 100.0]
    assert bundle.residual_norm_source == "neutral-corpus"
    assert bundle.designated_reference["name"] == "plain-exposition"

    run_dir = os.path.join(root, "runs", "20260810T060000-exp-extract")
    os.makedirs(run_dir)
    tasks._persist_vectors(bundles, manifest, model, run_dir)
    # It lands exactly like an extracted concept would: <concept>.safetensors
    # + sidecar, so everything downstream sees a normal artifact.
    written, sidecar = vector_store.load(run_dir, "crit-gm")
    assert written.per_layer == source.per_layer
    assert sidecar.extractionMethod == "pinnedArtifact"
    assert sidecar.readingPosition == "mean from token 50"
    # …and it names the bytes it came from.
    assert sidecar.pinnedFrom["path"] == artifact
    assert sidecar.pinnedFrom["sha256TensorHash"] == _sha256(
        os.path.join(root, artifact + ".safetensors"))
    assert sidecar.pinnedFrom["sourceMethod"] == "designatedReference"
    assert sidecar.pinnedFrom["sourceConcept"] == "crit"
    # The norm denominator provenance travels with the norms it describes.
    assert sidecar.neutralCorpusHash == "c" * 64
    assert sidecar.recipeIdentityHash


def test_drifted_bytes_refuse_materialization_naming_both_hashes(tmp_path):
    root, artifact, manifest = _attached(tmp_path)
    tensor = os.path.join(root, artifact + ".safetensors")
    with open(tensor, "ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(RuntimeError) as err:
        tasks._extract_all(_tiny_model(), manifest, root)
    message = str(err.value)
    assert f"{artifact}.safetensors" in message
    assert manifest.concepts[0].vector_artifact["sha256TensorHash"] in message
    assert _sha256(tensor) in message


def test_a_missing_artifact_refuses_materialization(tmp_path):
    root, artifact, manifest = _attached(tmp_path)
    os.remove(os.path.join(root, artifact + ".safetensors"))
    with pytest.raises(RuntimeError, match="is missing"):
        tasks._extract_all(_tiny_model(), manifest, root)


def test_materialization_refuses_a_model_mismatch(tmp_path):
    root, artifact, manifest = _attached(tmp_path)
    manifest.model_id = "org/other"
    with pytest.raises(RuntimeError, match="does not transfer between models"):
        tasks._extract_all(_tiny_model(), manifest, root)


# --- validate -------------------------------------------------------------

def test_validate_runs_the_held_out_probe_on_a_pinned_concept(tmp_path):
    root, _artifact, manifest = _attached(tmp_path)
    run = tasks._validate_impl("gm-study", manifest, _tiny_model(), root,
                               lambda *a: None)
    report = json.load(open(os.path.join(run, "validation-report.json")))
    entry = report["concepts"]["crit-gm"]
    # The SOURCE concept's held-out rows, scored contrastively against the
    # source recipe's two classes — the unchanged probe machinery.
    assert entry["scenarioCount"] == 2
    assert entry["labeled"] is True
    assert "scenarioAccuracy" in entry
    assert entry["diagnostics"]["classMeans"].keys() == {"positive", "negative"}
    # The evidence is scoped to the manifest that produced it.
    evidence = json.load(open(os.path.join(run, "validation-evidence.json")))
    assert evidence["validationScopeHash"] == manifest.validation_scope_hash()
    # …and the vectors persisted beside it are the pinned bytes.
    written, sidecar = vector_store.load(run, "crit-gm")
    assert sidecar.pinnedFrom is not None
    assert written.layer_count == 2


def test_validate_reads_at_the_artifacts_reading_position(tmp_path):
    """The probe reads held-out activations where the VECTOR was read: the
    sidecar's position is copied into the manifest at attach, it is what the
    probe uses, and it is load-bearing — the same rows read at last-token
    give different activations, so a contradicting declaration is refused
    outright rather than quietly measured at the wrong depth."""
    from steerlab_server.steering.extractor import activations
    from steerlab_server.steering.reading_position import LAST_TOKEN

    root, _artifact, manifest = _attached(tmp_path)
    reading = manifest.concepts[0].options.reading_position
    assert reading.label == "mean from token 50"
    model = _tiny_model()
    assert activations(model, [LONG_A], reading).values[0][0] != \
        activations(model, [LONG_A], LAST_TOKEN).values[0][0]

    manifest.concepts[0].options.reading_position = LAST_TOKEN
    with pytest.raises(RuntimeError,
                       match="read where the vector was read"):
        tasks._validate_impl("gm-study", manifest, _tiny_model(), root,
                             lambda *a: None)


# --- downstream surfaces --------------------------------------------------

def test_the_method_token_surfaces_in_catalogs_and_reports(tmp_path):
    root, _artifact, manifest = _attached(tmp_path)
    model = _tiny_model()
    bundles = tasks._extract_all(model, manifest, root)
    run_dir = os.path.join(root, "runs", "20260810T070000-exp-extract")
    os.makedirs(run_dir)
    tasks._persist_vectors(bundles, manifest, model, run_dir)

    materialized = [v for v in catalog.list_vectors(root)
                    if v.runDirectory == run_dir]
    assert [v.method for v in materialized] == ["pinnedArtifact"]
    assert materialized[0].workspaceRelativeID.endswith("crit-gm")
    detail = catalog.experiment_detail(manifest)
    assert detail["concepts"][0]["method"] == "pinnedArtifact"
    assert detail["concepts"][0]["reading"] == "mean from token 50"


def test_the_materialized_artifact_matches_the_promotion_matcher(tmp_path):
    """The lifecycle gate: sweep→promote finds the materialized vector by
    FULL recipe identity, so a pinned concept can actually be promoted."""
    from steerlab_server.experiment import promote, recipe_identity

    root, _artifact, manifest = _attached(tmp_path)
    model = _tiny_model()
    bundles = tasks._extract_all(model, manifest, root)
    run_dir = os.path.join(root, "runs", "20260810T080000-exp-extract")
    os.makedirs(run_dir)
    tasks._persist_vectors(bundles, manifest, model, run_dir)

    required = recipe_identity.required_identity(manifest, manifest.concepts[0])
    assert required["extractionMethod"] == "pinnedArtifact"
    # The denominator provenance is the ARTIFACT's, not the study's corpus.
    assert required["normCorpusHash"] == "c" * 64
    artifact, identity_hash = promote._matching_vector_artifact(
        manifest, manifest.concepts[0], root)
    assert artifact.runDirectory == run_dir
    sidecar = json.load(open(os.path.join(run_dir, "crit-gm.json")))
    assert sidecar["recipeIdentityHash"] == identity_hash


def test_the_cli_verb_attaches_and_reports_verification(tmp_path):
    from steerlab_server import cli

    root, artifact = _workspace(tmp_path)
    assert cli.main(["experiment", "attach-artifact", "gm-study", "crit-gm",
                     "--artifact", artifact, "--source-concept", "crit",
                     "--root", root]) == 0
    assert Manifest.load("gm-study", root).concepts[0].is_pinned_artifact
    # Usage error without the artifact; a refusal exits 1, never 0.
    assert cli.main(["experiment", "attach-artifact", "gm-study", "crit-gm",
                     "--root", root]) == 64
    assert cli.main(["experiment", "attach-artifact", "gm-study", "nope",
                     "--artifact", "runs/absent/nope", "--root", root]) == 1


def test_the_authoring_api_accepts_the_pinned_form(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from steerlab_server.api.app import app

    root, artifact = _workspace(tmp_path)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    client = TestClient(app)
    body = client.post("/api/authoring/gm-study/attach",
                       json={"method": "pinnedArtifact",
                             "concepts": ["crit-gm"],
                             "vectorArtifact": artifact,
                             "sourceConcept": "crit"})
    assert body.status_code == 200
    assert body.json()["concepts"][0]["vectorArtifact"]["path"] == artifact
    # An unpinnable artifact is a 400, not a half-written manifest.
    bad = client.post("/api/authoring/gm-study/attach",
                      json={"method": "pinnedArtifact", "concepts": ["x"],
                            "vectorArtifact": "runs/absent/x"})
    assert bad.status_code == 400


def test_norm_unit_alpha_uses_the_artifacts_residual_norms(tmp_path):
    """What the sweep and run loops actually consume: the materialized bundle
    resolves conditions exactly like an extracted one, including the
    norm-unit denominator carried in from the artifact."""
    from steerlab_server.experiment.manifest import Condition, Slot

    root, _artifact, manifest = _attached(tmp_path)
    bundles = tasks._extract_all(_tiny_model(), manifest, root)
    condition = Condition(name="c", slots=[Slot(concept="crit-gm", layer=1,
                                                alpha=0.25)],
                          alpha_in_norm_units=True)
    injections = tasks._condition_injections(condition, bundles)
    assert len(injections) == 1
    vector = bundles["crit-gm"].vectors.per_layer[1]
    expected = vm.norm_unit_scale(0.25, 100.0, vm.l2_norm(vector))
    assert injections[0].alpha == pytest.approx(expected)

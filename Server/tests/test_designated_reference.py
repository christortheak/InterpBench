"""designatedReference: mean(concept stories) − mean(reference stories),
first-class (METHODS amendment ii).

Contracts:
- attach pins the concept's stories hash AND the reference {name, hash},
  and the pooled reading (token 50) is the method's POLICY — set by
  attach, not remembered by a researcher.
- verify treats reference drift exactly like stimulus drift.
- extraction dispatches through the same core as paired methods (same
  math, same reading-position diagnostic) with stories as the classes.
- the pin surface carries BOTH stories files, so bundles travel whole.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import vector_math as vm
from steerlab_server.steering.hooks import HookedModel


def _stories(root, concept, texts):
    path = os.path.join(root, "prompts", "emotions", concept, "stories.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for t in texts:
            handle.write(json.dumps({"concept": concept, "text": t}) + "\n")
    return path


LONG_A = "a steady voice carried the difficult meeting through real danger " * 3
LONG_B = "an unhurried afternoon of plain and ordinary errands in the town " * 3


def _workspace(tmp_path):
    root = str(tmp_path)
    _stories(root, "courage", [LONG_A, LONG_A + "again"])
    _stories(root, "neutral", [LONG_B, LONG_B + "again"])
    es.create("dr-study", model_id="org/m", root=root)
    return root


def test_attach_pins_reference_and_pooled_reading_policy(tmp_path):
    root = _workspace(tmp_path)
    d = es.attach("dr-study", ["courage"], method="designatedReference",
                  reference="neutral", root=root)
    ref = d["concepts"][0]
    assert ref["options"]["method"] == "designatedReference"
    # The POLICY: pooled from 50 without anyone remembering to set it.
    assert ref["options"]["readingPosition"] == {"meanFromToken": {"_0": 50}}
    assert ref["stimulusSetHash"]
    assert ref["designatedReference"]["name"] == "neutral"
    assert ref["designatedReference"]["hash"]
    assert "grandMeanCorpus" not in d or not d.get("grandMeanCorpus")
    # Verifies clean as attached.
    assert Manifest.load("dr-study", root).verify(root) == []


def test_attach_refuses_without_or_with_unknown_reference(tmp_path):
    root = _workspace(tmp_path)
    with pytest.raises(es.ExperimentStoreError, match="--reference"):
        es.attach("dr-study", ["courage"], method="designatedReference", root=root)
    with pytest.raises(es.ExperimentStoreError, match="no stories.jsonl for reference"):
        es.attach("dr-study", ["courage"], method="designatedReference",
                  reference="nonexistent", root=root)


def test_reference_drift_is_a_verify_violation(tmp_path):
    root = _workspace(tmp_path)
    es.attach("dr-study", ["courage"], method="designatedReference",
              reference="neutral", root=root)
    _stories(root, "neutral", [LONG_B, "a different reference story entirely " * 4])
    violations = Manifest.load("dr-study", root).verify(root)
    assert any("reference 'neutral' stories changed" in v for v in violations)


def test_direction_math_is_mean_difference(tmp_path):
    pos, neg = [[2.0, 0.0], [4.0, 0.0]], [[0.0, 2.0], [0.0, 4.0]]
    assert vm.direction(pos, neg, vm.ExtractionMethod.DESIGNATED_REFERENCE) == \
        vm.direction(pos, neg, vm.ExtractionMethod.MEAN_DIFFERENCE)


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        torch.manual_seed(sum(ord(c) for c in text[:12]))
        return SimpleNamespace(input_ids=torch.randint(1, 127, (1, 60)))


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
                           model_id="org/m", revision="abc123")


def test_extraction_dispatch_end_to_end(tmp_path):
    root = _workspace(tmp_path)
    es.attach("dr-study", ["courage"], method="designatedReference",
              reference="neutral", root=root)
    manifest = Manifest.load("dr-study", root)
    bundles = tasks._extract_all(_tiny_model(), manifest, root)
    bundle = bundles["courage"]
    assert bundle.vectors.layer_count == 2
    assert bundle.designated_reference["name"] == "neutral"
    assert bundle.designated_reference["hash"]
    # The pooled recipe carries its standing diagnostic like any other.
    assert bundle.reading_position_diagnostic is not None
    assert bundle.reading_position_diagnostic["primaryReadingPosition"] == \
        "mean from token 50"


def test_pin_surface_carries_both_stories_files(tmp_path):
    root = _workspace(tmp_path)
    es.attach("dr-study", ["courage"], method="designatedReference",
              reference="neutral", root=root)
    entries = es.pinned_input_entries(es.load_raw("dr-study", root), root)
    labels = {e.label: e.required for e in entries}
    assert labels.get("concept 'courage' stories.jsonl") is True
    assert labels.get(
        "concept 'courage' designated reference 'neutral' stories.jsonl") is True
    # And the paired concepts directory is NOT a required pin for this method.
    dir_entry = [e for e in entries
                 if e.label == "concept 'courage' stimulus directory"]
    assert dir_entry and dir_entry[0].required is False


def test_lifecycle_validate_identity_and_promotion_match(tmp_path):
    """The review's gate: attach → extract → validate → recipe match, the
    exact boundaries the first landing stopped short of. Identity must carry
    the reference (two vectors against different references must never
    match), validation must run CONTRASTIVELY over the concept's own two
    classes, and the sidecar-derived candidate identity must reproduce the
    manifest-derived required identity byte for byte."""
    from steerlab_server.experiment import recipe_identity

    root = _workspace(tmp_path)
    val = os.path.join(root, "prompts", "emotions", "courage", "validation.jsonl")
    with open(val, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": LONG_A + "held-out", "expresses": True}) + "\n")
        handle.write(json.dumps({"text": LONG_B + "held-out", "expresses": False}) + "\n")
    es.attach("dr-study", ["courage"], method="designatedReference",
              reference="neutral", root=root)
    manifest = Manifest.load("dr-study", root)
    manifest.model_revision = "abc123"

    # 1. Promotion's demand no longer explodes and carries the reference.
    required = recipe_identity.required_identity(manifest, manifest.concepts[0])
    ref_pin = manifest.concepts[0].designated_reference
    assert required["methodParameters"] == {
        "referenceHash": ref_pin["hash"], "referenceName": ref_pin["name"]}
    assert required["grandMeanPopulation"] is None

    # 2. A sidecar stamped by extraction proves the SAME identity.
    sidecar = {
        "concept": "courage", "modelID": "org/m", "revision": "abc123",
        "extractionMethod": "designatedReference",
        "stimulusSetHash": manifest.concepts[0].stimulus_set_hash,
        "readingPosition": "mean from token 50",
        "neutralProjection": "none",
        "residualNormSource": "extraction-stimuli",
        "designatedReference": dict(ref_pin),
    }
    candidate, missing = recipe_identity.candidate_identity(sidecar)
    assert missing == []
    assert recipe_identity.canonical_json(candidate) == \
        recipe_identity.canonical_json(required)
    # A DIFFERENT reference must be a different identity.
    other = dict(sidecar, designatedReference={"name": "other", "hash": "f" * 64})
    other_candidate, _ = recipe_identity.candidate_identity(other)
    assert recipe_identity.canonical_json(other_candidate) != \
        recipe_identity.canonical_json(required)
    # A sidecar MISSING the reference is unprovable, never matchable.
    bare, missing = recipe_identity.candidate_identity(
        {k: v for k, v in sidecar.items() if k != "designatedReference"})
    assert bare is None and "designatedReference" in missing

    # 3. Validate runs end to end, contrastively, on the tiny model.
    run = tasks._validate_impl("dr-study", manifest, _tiny_model(), root,
                               lambda *a: None)
    report = json.load(open(os.path.join(run, "validation-report.json")))
    entry = report["concepts"]["courage"]
    assert "scenarioAccuracy" in entry
    assert entry["diagnostics"]["classMeans"].keys() == {"positive", "negative"}


def test_persisted_sidecar_matches_through_the_production_matcher(tmp_path):
    """Review round 2, finding 1: the gate must exercise the PRODUCTION
    promotion matcher over a PERSISTED sidecar — not a hand-built dict. The
    real extraction writer persists; promote's _matching_vector_artifact
    must find it by full recipe identity."""
    from steerlab_server.experiment import promote

    root = _workspace(tmp_path)
    es.attach("dr-study", ["courage"], method="designatedReference",
              reference="neutral", root=root)
    manifest = Manifest.load("dr-study", root)
    manifest.model_revision = "abc123"
    model = _tiny_model()
    bundles = tasks._extract_all(model, manifest, root)
    run_dir = os.path.join(root, "runs", "20260731T000000-dr-extract")
    os.makedirs(run_dir)
    tasks._persist_vectors(bundles, manifest, model, run_dir)
    artifact, identity_hash = promote._matching_vector_artifact(
        manifest, manifest.concepts[0], root)
    assert artifact is not None and identity_hash
    sidecar = json.load(open(
        os.path.join(artifact.runDirectory, "courage.json")))
    assert sidecar["designatedReference"]["name"] == "neutral"
    assert sidecar["recipeIdentityHash"] == identity_hash


def test_extraction_refuses_drifted_stories(tmp_path):
    """Finding 3: a draft extraction must never stamp pinned hashes over
    drifted bytes — it refuses, exactly like the Swift twin."""
    root = _workspace(tmp_path)
    es.attach("dr-study", ["courage"], method="designatedReference",
              reference="neutral", root=root)
    _stories(root, "neutral", ["a silently different reference " * 6])
    manifest = Manifest.load("dr-study", root)
    with pytest.raises(RuntimeError, match="reference 'neutral' stories drifted"):
        tasks._extract_all(_tiny_model(), manifest, root)


def test_canonical_identity_bytes_are_the_cross_engine_fixture():
    """Byte-pinned canonical form, asserted verbatim in BOTH suites (Swift
    twin: canonicalIdentityBytesAreTheCrossEngineFixture) — the identity
    hash is only cross-engine if these bytes are."""
    from steerlab_server.experiment import recipe_identity
    components = {
        "concept": "c", "modelID": "m", "revision": "v",
        "extractionMethod": "designatedReference", "stimulusSetHash": "aa",
        "readingPositionMode": "meanFromToken", "readingPositionParameter": 50,
        "projectionMode": "none", "projectionCount": None,
        "projectionExplainedVariance": None, "projectionBasisHash": None,
        "residualNormSource": "extraction-stimuli", "normCorpusHash": None,
        "grandMeanPopulation": None,
        "methodParameters": {"referenceHash": "bb", "referenceName": "r"},
    }
    assert recipe_identity.canonical_json(components) == '{"concept":"c","extractionMethod":"designatedReference","grandMeanPopulation":null,"methodParameters":{"referenceHash":"bb","referenceName":"r"},"modelID":"m","neutralProjection":{"basisHash":null,"count":null,"explainedVariance":null,"mode":"none"},"normCorpusHash":null,"readingPosition":{"mode":"meanFromToken","parameter":50},"residualNormSource":"extraction-stimuli","revision":"v","schema":1,"stimulusSetHash":"aa"}'

"""How validate resolves ONE pinned concept before it measures anything.

The reference half of open-issues §21 (2026-08-20). Three choices this engine
has always made in ``tasks._validate_impl`` and Swift's validate loop did not:

1. **Method** — the DATA method is the artifact's recorded SOURCE method
   (``ConceptRef.effective_method``), not the declared ``pinnedArtifact``.
2. **Data identity** — stimuli and the held-out ``validation.jsonl`` live
   under ``ConceptRef.data_concept`` ("tidiness-gm" reads "tidiness"'s), which
   is what ``verify()`` pins on BOTH engines.
3. **Skip rule** — a direction with no source concept (OptVec, an imported
   Gemma Scope SAE decoder row) has no stimuli, no class means and no held-out
   set; the loop ``continue``s and the vacuity ledger never contained it.

Nothing here changes this engine's behaviour: it PINS it, over a committed
fixture manifest that is byte-identical to
``Tests/ExperimentKitTests/Fixtures/validate-resolution/experiment.json``, so
drift in either engine trips a test. Swift twin:
``Tests/ExperimentKitTests/ValidateConceptResolutionTests.swift``.
"""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import (
    Manifest, held_out_probe_relpath, owes_held_out_probe,
    resolve_validation_file, validation_lookup_advisory)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "validate-resolution", "experiment.json")
SWIFT_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "Tests", "ExperimentKitTests",
    "Fixtures", "validate-resolution", "experiment.json")

SCENARIO = '{"text": "a held-out scene", "expresses": true}\n'


def _fixture_manifest() -> Manifest:
    with open(FIXTURE, encoding="utf-8") as handle:
        return Manifest.from_dict(json.load(handle))


def _concept(manifest: Manifest, name: str):
    return next(c for c in manifest.concepts if c.name == name)


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# --- the fixture is ONE artifact, shared ------------------------------------

def test_both_engines_read_the_same_fixture_bytes():
    """The parity claim is over identical bytes, not two hand-kept copies
    that drifted apart."""
    with open(FIXTURE, "rb") as handle:
        mine = hashlib.sha256(handle.read()).hexdigest()
    with open(SWIFT_FIXTURE, "rb") as handle:
        theirs = hashlib.sha256(handle.read()).hexdigest()
    assert mine == theirs


# --- (1) method resolves through the ARTIFACT's source method ---------------

def test_an_ordinary_recipe_answers_with_its_declared_method():
    concept = _concept(_fixture_manifest(), "tidiness")
    assert concept.effective_method.value == "emotionGrandMean"
    assert concept.effective_method.is_grand_mean
    assert concept.data_concept == "tidiness"


def test_an_artifact_pinned_concept_answers_with_its_source_method():
    manifest = _fixture_manifest()
    contrastive = _concept(manifest, "stillness-pair")
    assert contrastive.options.method.is_pinned_artifact
    assert contrastive.effective_method.value == "meanDifference"
    assert contrastive.effective_method.uses_contrastive_validation
    assert not contrastive.effective_method.is_grand_mean

    population = _concept(manifest, "tidiness-gm")
    assert population.effective_method.value == "emotionGrandMean"
    assert population.effective_method.is_grand_mean


# --- (2) data identity is the DATA concept, as verify() pins it -------------

def test_the_probe_reads_the_data_concept_not_the_study_side_name():
    manifest = _fixture_manifest()
    assert _concept(manifest, "tidiness-gm").data_concept == "tidiness"
    assert _concept(manifest, "stillness-pair").data_concept == "stillness"
    assert _concept(manifest, "tidiness").data_concept == "tidiness"


def test_the_method_decides_which_recipe_root_is_canonical():
    manifest = _fixture_manifest()
    # `paired=not method.uses_story_corpus` — resolved from the EFFECTIVE
    # method, which is why (1) and (2) cannot land separately.
    story = _concept(manifest, "tidiness-gm").effective_method
    assert story.uses_story_corpus  # ⇒ paired=False: prompts/emotions/
    paired = _concept(manifest, "stillness-pair").effective_method
    assert not paired.uses_story_corpus  # ⇒ paired=True: prompts/concepts/


def test_the_dual_root_lookup_runs_under_the_data_concept(tmp_path):
    """Found under the DATA concept; the advisory is labelled with the
    STUDY-side name (the loop's `validation_lookup_advisory(concept_name,
    location)` over `resolve_validation_file(data_concept, …)`)."""
    root = str(tmp_path)
    concept = _concept(_fixture_manifest(), "tidiness-gm")
    method = concept.effective_method
    paired = not method.uses_story_corpus
    digest = _write(os.path.join(root, "prompts", "emotions", "tidiness",
                                 "validation.jsonl"), SCENARIO)

    # Under the STUDY-side name: nothing — the divergence Swift carried.
    assert resolve_validation_file("tidiness-gm", paired=paired, root=root) is None

    location = resolve_validation_file(concept.data_concept, paired=paired,
                                       root=root)
    assert location is not None
    assert location.relpath == "prompts/emotions/tidiness/validation.jsonl"
    assert not location.used_fallback
    with open(location.path, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == digest


def test_the_fallback_root_still_applies_under_the_data_concept(tmp_path):
    root = str(tmp_path)
    concept = _concept(_fixture_manifest(), "tidiness-gm")
    paired = not concept.effective_method.uses_story_corpus
    _write(os.path.join(root, "prompts", "concepts", "tidiness",
                        "validation.jsonl"), SCENARIO)

    location = resolve_validation_file(concept.data_concept, paired=paired,
                                       root=root)
    assert location is not None and location.used_fallback
    assert location.relpath == "prompts/concepts/tidiness/validation.jsonl"
    assert location.canonical_relpath == "prompts/emotions/tidiness/validation.jsonl"

    advisory = validation_lookup_advisory(concept.name, location)
    assert advisory is not None
    assert advisory.startswith("concept 'tidiness-gm':")
    assert "prompts/emotions/tidiness/validation.jsonl" in advisory


# --- (3) no source concept ⇒ skipped, and never recorded vacuous ------------

@pytest.mark.parametrize("name", ["drifting-optvec", "dictionary-row"])
def test_a_direction_with_no_source_concept_is_skipped(name):
    concept = _concept(_fixture_manifest(), name)
    assert not concept.effective_method.has_source_concept
    # The vacuity ledger is SEEDED from `owes_held_out_probe`, so a skipped
    # concept was never on it — the skip cannot be mistaken for "measured
    # nothing", on either engine.
    assert not owes_held_out_probe(concept)
    assert held_out_probe_relpath(concept) is None


@pytest.mark.parametrize("name,relpath", [
    ("tidiness", "prompts/emotions/tidiness/validation.jsonl"),
    ("tidiness-gm", "prompts/emotions/tidiness/validation.jsonl"),
    ("stillness-pair", "prompts/concepts/stillness/validation.jsonl")])
def test_a_concept_that_owes_a_probe_stays_on_the_hook(name, relpath):
    concept = _concept(_fixture_manifest(), name)
    assert owes_held_out_probe(concept)
    # The path a vacuity refusal NAMES follows both resolutions at once.
    assert held_out_probe_relpath(concept) == relpath


# --- exhaustive semantics: refuse, never mis-branch -------------------------

def test_a_method_with_no_validation_semantics_is_neither_branch():
    """Reachable only by hand-editing a manifest (attach refuses an artifact
    whose own source method is `pinnedArtifact`) — which is exactly why the
    loop refuses instead of falling into whichever branch is last."""
    method = _concept(_fixture_manifest(), "undeclared-semantics").effective_method
    assert method.has_source_concept
    assert not (method.uses_contrastive_validation or method.is_grand_mean)


# --- the skip, asserted through the real loop -------------------------------

def _skip_only_workspace(tmp_path):
    """A workspace whose ONLY concept is a source-concept-less pinned
    artifact: the probe loop must skip it without a report entry and without
    a vacuity record, and everything after the loop must still run."""
    root = str(tmp_path)
    es.create("skip-only", model_id="org/m", root=root)
    raw = es.load_raw("skip-only", root)
    raw["concepts"] = [
        c for c in _fixture_raw()["concepts"] if c["name"] == "drifting-optvec"]
    es.save_raw(raw, root)
    return root


def _fixture_raw() -> dict:
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _fake_bundles(*names, layer_count=8):
    return {name: SimpleNamespace(
        vectors=SimpleNamespace(
            layer_count=layer_count,
            per_layer=[[1.0, float(index)] for _ in range(layer_count)]),
        residual_norm_per_layer=[1.0] * layer_count,
        residual_norm_source="extraction-stimuli",
        stimulus_hash="h")
        for index, name in enumerate(names)}


def test_the_loop_skips_a_source_concept_less_direction(tmp_path, monkeypatch):
    root = _skip_only_workspace(tmp_path)
    manifest = Manifest.load("skip-only", root)
    bundles = _fake_bundles("drifting-optvec")

    monkeypatch.setattr(tasks, "_extract_all", lambda m, mf, r: bundles)
    monkeypatch.setattr(tasks, "_persist_vectors", lambda *a, **k: None)
    from steerlab_server.steering import extractor
    monkeypatch.setattr(
        extractor, "logit_lens",
        lambda model, vectors, layer, top_k=12: extractor.LogitLensReport(
            layer=layer, top_positive=[], top_negative=[]))
    # Nothing may reach a forward pass for a skipped concept.
    monkeypatch.setattr(
        extractor, "activations",
        lambda *a, **k: pytest.fail("a skipped concept must not be measured"))

    run_dir = tasks._validate_impl("skip-only", manifest, object(), root,
                                   lambda *a: None)
    with open(os.path.join(run_dir, "validation-report.json"),
              encoding="utf-8") as handle:
        report = json.load(handle)
    # No report entry — the skip is not a "no validation.jsonl" finding.
    assert report["concepts"] == {}
    # And not vacuous: it owes no probe, so it cannot be unmeasured.
    assert report["vacuousConcepts"] == []
    with open(os.path.join(run_dir, "validation-evidence.json"),
              encoding="utf-8") as handle:
        assert json.load(handle)["vacuousConcepts"] == []
    # The rest of the run is unaffected — the lens still reads the direction.
    assert "drifting-optvec" in report["logitLens"]

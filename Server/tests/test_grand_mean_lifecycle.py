"""Grand-mean extraction as a manifest-pinnable, freeze-compatible method
(WORK-PLAN Phase D′): attach → verify → extract dispatch → validation scoring."""

import json
import os

import pytest

from steerlab_server.experiment import concept_stats, experiment_store as es, multiconcept, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import extractor, vector_math as vm


def _stories(root, name, texts):
    d = os.path.join(root, "prompts", "emotions", name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "stories.jsonl"), "w") as handle:
        for i, text in enumerate(texts):
            handle.write(json.dumps({"concept": name, "text": text,
                                     "id": f"{name}-{i}"}) + "\n")


def _grand_mean_experiment(root, targets=("fear",), corpus=("fear", "calm")):
    _stories(root, "fear", ["Her heart pounded in the dark.",
                            "The floor creaked behind her."])
    _stories(root, "calm", ["Waves lapped gently at the shore.",
                            "The tea steamed in the quiet kitchen."])
    es.create("gm", model_id="org/m", revision="abc", root=root)
    return es.attach("gm", list(targets), method="emotionGrandMean",
                     corpus_concepts=list(corpus), root=root)


def test_enum_and_paired_guard():
    method = vm.ExtractionMethod("emotionGrandMean")
    assert method is vm.ExtractionMethod.GRAND_MEAN
    assert not method.is_paired
    assert vm.ExtractionMethod.MEAN_DIFFERENCE.is_paired
    with pytest.raises(vm.SteeringVectorError, match="not a paired method"):
        vm.direction([[1.0, 0.0]], [[0.0, 1.0]], method)


def test_attach_pins_corpus_and_stories(tmp_path):
    root = str(tmp_path)
    d = _grand_mean_experiment(root)
    ref = d["concepts"][0]
    assert ref["name"] == "fear"
    assert ref["options"]["method"] == "emotionGrandMean"
    assert ref["stimulusSetHash"] == multiconcept.stories_hash("fear", root)
    corpus = d["grandMeanCorpus"]
    assert corpus["concepts"] == ["fear", "calm"]
    assert set(corpus["hashes"]) == {"fear", "calm"}
    # Default reading position is the emotion paper's mean-from-token-50.
    reading = ref["options"]["readingPosition"]
    assert reading == {"meanFromToken": {"_0": 50}}


def test_attach_rejects_missing_corpus_member(tmp_path):
    root = str(tmp_path)
    _stories(root, "fear", ["text one", "text two"])
    es.create("gm2", model_id="org/m", root=root)
    with pytest.raises(es.ExperimentStoreError, match="ghost"):
        es.attach("gm2", ["fear"], method="emotionGrandMean",
                  corpus_concepts=["fear", "ghost"], root=root)


def test_verify_clean_and_corpus_drift(tmp_path):
    root = str(tmp_path)
    _grand_mean_experiment(root)
    manifest = Manifest.load("gm", root=root)
    assert manifest.verify(root) == []
    # Drift in a NON-target corpus member still invalidates: the population is
    # part of every grand-mean vector.
    _stories(root, "calm", ["completely different calm story"])
    violations = Manifest.load("gm", root=root).verify(root)
    assert any("calm" in v and "changed" in v for v in violations)


def test_verify_flags_missing_corpus_and_nonmember_target(tmp_path):
    root = str(tmp_path)
    d = _grand_mean_experiment(root)
    del d["grandMeanCorpus"]
    violations = Manifest.from_dict(d).verify(root)
    assert any("no grandMeanCorpus pinned" in v for v in violations)

    d2 = _grand_mean_experiment(str(tmp_path / "second"))
    d2["grandMeanCorpus"]["concepts"] = ["calm"]  # target dropped from corpus
    violations = Manifest.from_dict(d2).verify(str(tmp_path / "second"))
    assert any("not a member" in v for v in violations)


def test_validation_scope_covers_corpus(tmp_path):
    root = str(tmp_path)
    d = _grand_mean_experiment(root)
    narrow = Manifest.from_dict(d).validation_scope_hash()
    widened = json.loads(json.dumps(d))
    widened["grandMeanCorpus"]["concepts"].append("extra")
    widened["grandMeanCorpus"]["hashes"]["extra"] = "00" * 32
    assert Manifest.from_dict(widened).validation_scope_hash() != narrow


class _FakeActivations:
    def __init__(self, values, norms):
        self.values = values
        self.residual_norm_per_layer = norms
        # Where each stimulus was actually read (added 2026-08-24 with the
        # reading-position resolution stamp); empty is honest for a stub
        # that never resolved anything, and omits the stamp.
        self.resolutions = []


def test_extract_grand_mean_math_and_projection(monkeypatch):
    # 2 fear rows + 2 calm rows, 1 layer, 2 dims. Fear mean (1,1); grand mean
    # (0.5, 0.5) → fear vector (0.5, 0.5).
    rows = [("fear", "f1"), ("fear", "f2"), ("calm", "c1"), ("calm", "c2")]
    acts = {
        ("f1", "f2", "c1", "c2"): [[[2.0, 0.0]], [[0.0, 2.0]],
                                   [[0.0, 0.0]], [[0.0, 0.0]]],
        ("n1", "n2", "n3", "n4"): [[[1.0, 0.0]], [[3.0, 0.0]],
                                   [[5.0, 0.0]], [[7.0, 0.0]]],
    }

    def fake_activations(model, texts, position, rendering=None):
        return _FakeActivations(acts[tuple(texts)], [1.0])

    monkeypatch.setattr(extractor, "activations", fake_activations)
    result = extractor.extract_grand_mean(
        model=None, corpus=rows, target_concepts={"fear"})
    vector = result.per_concept["fear"].per_layer[0]
    assert vector == pytest.approx([0.5, 0.5])
    assert result.residual_norm_source == "extraction-stimuli"

    # Neutral-PC projection: neutral rows vary only along x, so PC1 = x-axis
    # and the projected fear vector keeps only its y component.
    projected = extractor.extract_grand_mean(
        model=None, corpus=rows, target_concepts={"fear"},
        neutral_texts=["n1", "n2", "n3", "n4"], neutral_pc_count=1)
    vector = projected.per_concept["fear"].per_layer[0]
    assert vector[0] == pytest.approx(0.0, abs=1e-5)
    assert vector[1] == pytest.approx(0.5)
    assert projected.residual_norm_source == "neutral-corpus"

    with pytest.raises(extractor.ConceptExtractorError, match="neutral corpus required"):
        extractor.extract_grand_mean(model=None, corpus=rows, neutral_pc_count=2)


def test_scenario_accuracy_grand_mean():
    direction = [1.0, 0.0]
    concept = [[2.0, 0.0], [4.0, 0.0]]      # mean projection 3
    population = [[1.0, 0.0], [1.0, 0.0]]   # mean projection 1 → midpoint 2
    scenarios = [[3.0, 0.0], [0.5, 0.0], [2.5, 0.0]]
    labels = [True, False, False]  # last is wrong on purpose
    accuracy = concept_stats.scenario_accuracy_grand_mean(
        direction=direction, concept=concept, population=population,
        scenarios=scenarios, labels=labels)
    assert accuracy == pytest.approx(2 / 3)
    assert concept_stats.scenario_accuracy_grand_mean(
        direction=direction, concept=[], population=population,
        scenarios=scenarios, labels=labels) is None


def test_extract_all_dispatches_grand_mean(tmp_path, monkeypatch):
    root = str(tmp_path)
    _grand_mean_experiment(root)
    manifest = Manifest.load("gm", root=root)

    captured = {}

    def fake_extract_grand_mean(model, rows, *, target_concepts, reading_position,
                                neutral_texts, neutral_pc_count,
                                extraction_rendering=None):
        captured["rows"] = rows
        captured["targets"] = target_concepts
        captured["pc"] = neutral_pc_count
        from steerlab_server.steering.vector_store import ConceptVectors
        return extractor.MultiConceptExtractionResult(
            per_concept={name: ConceptVectors(per_layer=[[1.0, 2.0]])
                         for name in target_concepts},
            residual_norm_per_layer=[3.0], residual_norm_source="extraction-stimuli",
            included=len(rows))

    monkeypatch.setattr(extractor, "extract_grand_mean", fake_extract_grand_mean)
    bundles = tasks._extract_all(model=None, manifest=manifest, root=root)
    assert set(bundles) == {"fear"}
    assert captured["targets"] == {"fear"}
    assert len(captured["rows"]) == 4  # both corpus members' stories
    assert bundles["fear"].stimulus_hash == multiconcept.stories_hash("fear", root)


def test_extract_all_requires_pinned_corpus(tmp_path):
    root = str(tmp_path)
    d = _grand_mean_experiment(root)
    del d["grandMeanCorpus"]
    with pytest.raises(RuntimeError, match="grandMeanCorpus"):
        tasks._extract_all(model=None, manifest=Manifest.from_dict(d), root=root)

"""Concept validation stats — pure math over synthetic activations."""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import concept_stats
from steerlab_server.steering import vector_math as vm

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine", "concept-stats-splits.json")


def _layered(rows):
    # turn [stimulus][hidden] into [stimulus][layer=1][hidden]
    return [[r] for r in rows]


def _texts(prefix, n):
    return [f"{prefix} stimulus {i}" for i in range(n)]


def test_separable_classes_score_well():
    pos = _layered([[3.0, 0.0], [3.2, 0.1], [2.8, -0.1], [3.1, 0.0],
                    [2.9, 0.0], [3.0, 0.2], [3.3, -0.1]])
    neg = _layered([[-3.0, 0.0], [-3.2, 0.1], [-2.8, -0.1], [-3.1, 0.0],
                    [-2.9, 0.0], [-3.0, 0.2], [-3.3, -0.1]])
    s = concept_stats.compute(positive_by_layer=pos, negative_by_layer=neg,
                              method=vm.ExtractionMethod.MEAN_DIFFERENCE,
                              positive_texts=_texts("pos", 7),
                              negative_texts=_texts("neg", 7))
    assert s["heldOut"]["accuracy"] == 1.0
    assert s["splitHalf"] > 0.99
    assert s["statsLayer"] == 0
    assert len(s["normByLayer"]) == 1


def test_held_out_none_when_too_few():
    pos = _layered([[1.0], [1.0]])
    neg = _layered([[-1.0], [-1.0]])
    s = concept_stats.compute(positive_by_layer=pos, negative_by_layer=neg,
                              method=vm.ExtractionMethod.MEAN_DIFFERENCE,
                              positive_texts=_texts("pos", 2),
                              negative_texts=_texts("neg", 2))
    assert s["heldOut"] is None


# --- F3: the split is content-derived, not order-derived --------------------

def test_content_hash_order_is_the_sha256_sort():
    """The rule in one assertion, so a refactor cannot quietly reinterpret it:
    indices sorted ascending by the lowercase SHA-256 hex of the row text."""
    texts = ["delta", "alpha", "charlie", "bravo"]
    expected = sorted(
        range(len(texts)),
        key=lambda i: hashlib.sha256(texts[i].encode("utf-8")).hexdigest())
    assert concept_stats.content_hash_order(texts) == expected
    assert concept_stats.held_out_indices(texts) == {
        i for position, i in enumerate(expected) if position % 5 == 4}
    assert concept_stats.split_half_indices(texts) == {
        i for position, i in enumerate(expected) if position % 2 == 1}


def test_scrambling_the_rows_does_not_move_either_statistic():
    """The whole point of F3. The same stimuli in a different FILE ORDER must
    produce the identical held-out accuracy and split-half cosine — the old
    positional rules failed this by construction."""
    texts_p = _texts("positive", 12)
    texts_n = _texts("negative", 12)
    rng = [(0.9 + 0.01 * i, 0.02 * (i % 3)) for i in range(12)]
    pos = [[list(v)] for v in rng]
    neg = [[[-v[0], v[1]]] for v in rng]

    order = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4, 6]
    kwargs = dict(method=vm.ExtractionMethod.MEAN_DIFFERENCE)
    straight = concept_stats.compute(
        positive_by_layer=pos, negative_by_layer=neg,
        positive_texts=texts_p, negative_texts=texts_n, **kwargs)
    shuffled = concept_stats.compute(
        positive_by_layer=[pos[i] for i in order],
        negative_by_layer=[neg[i] for i in order],
        positive_texts=[texts_p[i] for i in order],
        negative_texts=[texts_n[i] for i in order], **kwargs)
    assert straight["heldOut"] == shuffled["heldOut"]
    assert straight["splitHalf"] == pytest.approx(shuffled["splitHalf"], abs=1e-6)


def test_topic_blocked_authoring_no_longer_decides_the_held_out_set():
    """A file authored in topic runs used to hand the LAST topic block over
    whole as the test set. Under the content-hash rule the held-out rows come
    from more than one block."""
    texts = [f"{topic} sentence {i}"
             for topic in ("harbour", "kitchen", "ledger") for i in range(5)]
    held = {texts[i].split()[0]
            for i in concept_stats.held_out_indices(texts)}
    assert len(held) > 1, held


def test_split_rule_matches_the_committed_cross_engine_fixture():
    """The bytes the Swift twin asserts (``ConceptStatsSplitCrossEngineTests``).
    Reading them here too means a change to the rule cannot land green on one
    engine while the fixture still describes the other."""
    with open(FIXTURE, encoding="utf-8") as handle:
        root = json.load(handle)
    assert root["minimumRowsPerClass"] == concept_stats.MINIMUM_ROWS_PER_CLASS
    assert (root["minimumRowsPerClassSplitHalf"]
            == concept_stats.MINIMUM_ROWS_PER_CLASS_SPLIT_HALF)
    by_label = {}
    for case in root["cases"]:
        texts = case["texts"]
        assert concept_stats.content_hash_order(texts) == case["contentHashOrder"]
        assert sorted(texts[i] for i in concept_stats.held_out_indices(texts)) \
            == case["heldOutTexts"]
        assert sorted(texts[i] for i in concept_stats.split_half_indices(texts)) \
            == case["splitHalfSecondTexts"]
        by_label[case["label"]] = case
    # The order-independence claim, stated in the fixture itself: the same
    # texts in a different order select the same rows.
    assert (by_label["topic-blocked"]["heldOutTexts"]
            == by_label["topic-blocked-scrambled"]["heldOutTexts"])
    assert (by_label["topic-blocked"]["splitHalfSecondTexts"]
            == by_label["topic-blocked-scrambled"]["splitHalfSecondTexts"])


def test_texts_must_be_row_aligned():
    pos = _layered([[1.0, 0.0]] * 8)
    neg = _layered([[-1.0, 0.0]] * 8)
    with pytest.raises(ValueError, match="one text per activation row"):
        concept_stats.held_out_accuracy(
            [r[0] for r in pos], [r[0] for r in neg],
            vm.ExtractionMethod.MEAN_DIFFERENCE,
            positive_texts=_texts("p", 3), negative_texts=_texts("n", 8))
    with pytest.raises(ValueError, match="one text per activation row"):
        concept_stats.split_half_cosine(
            [r[0] for r in pos], [r[0] for r in neg],
            vm.ExtractionMethod.MEAN_DIFFERENCE,
            positive_texts=_texts("p", 8), negative_texts=_texts("n", 3))


def test_scenario_accuracy_labeled():
    # midpoint = (1*2 + 1*-2)/2 = 0; expresses → projection > 0
    acc = concept_stats.scenario_accuracy(
        direction=[1.0, 0.0], positive=[[2.0, 0.0]], negative=[[-2.0, 0.0]],
        scenarios=[[3.0, 0.0], [-3.0, 0.0]], labels=[True, False])
    assert acc == 1.0
    # one mislabeled scenario → 0.5
    acc2 = concept_stats.scenario_accuracy(
        direction=[1.0, 0.0], positive=[[2.0, 0.0]], negative=[[-2.0, 0.0]],
        scenarios=[[3.0, 0.0], [3.0, 0.0]], labels=[True, False])
    assert acc2 == 0.5


def test_load_validation_labeled_and_unlabeled(tmp_path):
    from steerlab_server.steering.stimulus_set import load_validation
    p = tmp_path / "validation.jsonl"
    p.write_text('{"text":"a","expresses":true}\n{"text":"b","expresses":false}\n', encoding="utf-8")
    rows = load_validation(str(p))
    assert rows == [{"text": "a", "expresses": True}, {"text": "b", "expresses": False}]
    p.write_text('{"text":"a"}\n', encoding="utf-8")  # legacy unlabeled
    rows = load_validation(str(p))
    assert rows == [{"text": "a"}] and "expresses" not in rows[0]


def test_the_coauthoring_prompts_documented_example_parses(tmp_path):
    """Review 2026-08-02 (P1): the app's co-authoring prompt taught
    {"text","label"} while both engines require {"text","expresses":bool} —
    strict loading turned that doc bug into rejected data. The documented
    example must parse through THIS loader (Swift twin:
    documentedValidationExampleParsesThroughTheRealLoader)."""
    from steerlab_server.steering.stimulus_set import load_validation
    p = tmp_path / "validation.jsonl"
    p.write_text(
        '{"text": "The clerk hesitated at the counter.", "expresses": true}\n'
        '{"text": "The bus arrived at seven.", "expresses": false}\n',
        encoding="utf-8")
    rows = load_validation(str(p))
    assert [r["expresses"] for r in rows] == [True, False]


def test_load_validation_is_strict(tmp_path):
    """Review 2026-08-01: this loader silently skipped malformed rows (the
    Swift twin has always thrown — the same file validated over different
    scenario sets per engine) and coerced labels (bool("no") is True: a
    string label silently INVERTED its scenario)."""
    import pytest
    from steerlab_server.steering.stimulus_set import (
        StimulusSetError, load_validation)
    p = tmp_path / "validation.jsonl"
    p.write_text('{"text":"a","expresses":true}\nnot json\n', encoding="utf-8")
    with pytest.raises(StimulusSetError, match="line 2"):
        load_validation(str(p))
    p.write_text('{"expresses":true}\n', encoding="utf-8")
    with pytest.raises(StimulusSetError, match="non-empty 'text'"):
        load_validation(str(p))
    p.write_text('{"text":"a","expresses":"no"}\n', encoding="utf-8")
    with pytest.raises(StimulusSetError, match="JSON boolean"):
        load_validation(str(p))


def test_control_cosine_warn():
    pos = _layered([[1.0, 0.0]] * 6)
    neg = _layered([[-1.0, 0.0]] * 6)
    # a control pointing the same way → high cosine → warned
    s = concept_stats.compute(positive_by_layer=pos, negative_by_layer=neg,
                              method=vm.ExtractionMethod.MEAN_DIFFERENCE,
                              positive_texts=_texts("pos", 6),
                              negative_texts=_texts("neg", 6),
                              control_vectors={"twin": [1.0, 0.0], "ortho": [0.0, 1.0]})
    assert "twin" in s["controlWarn"]
    assert "ortho" not in s["controlWarn"]

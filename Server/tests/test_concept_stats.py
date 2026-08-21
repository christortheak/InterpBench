"""Concept validation stats — pure math over synthetic activations."""

import pytest

from steerlab_server.experiment import concept_stats
from steerlab_server.steering import vector_math as vm


def _layered(rows):
    # turn [stimulus][hidden] into [stimulus][layer=1][hidden]
    return [[r] for r in rows]


def test_separable_classes_score_well():
    pos = _layered([[3.0, 0.0], [3.2, 0.1], [2.8, -0.1], [3.1, 0.0],
                    [2.9, 0.0], [3.0, 0.2], [3.3, -0.1]])
    neg = _layered([[-3.0, 0.0], [-3.2, 0.1], [-2.8, -0.1], [-3.1, 0.0],
                    [-2.9, 0.0], [-3.0, 0.2], [-3.3, -0.1]])
    s = concept_stats.compute(positive_by_layer=pos, negative_by_layer=neg,
                              method=vm.ExtractionMethod.MEAN_DIFFERENCE)
    assert s["heldOut"]["accuracy"] == 1.0
    assert s["splitHalf"] > 0.99
    assert s["statsLayer"] == 0
    assert len(s["normByLayer"]) == 1


def test_held_out_none_when_too_few():
    pos = _layered([[1.0], [1.0]])
    neg = _layered([[-1.0], [-1.0]])
    s = concept_stats.compute(positive_by_layer=pos, negative_by_layer=neg,
                              method=vm.ExtractionMethod.MEAN_DIFFERENCE)
    assert s["heldOut"] is None


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
                              control_vectors={"twin": [1.0, 0.0], "ortho": [0.0, 1.0]})
    assert "twin" in s["controlWarn"]
    assert "ortho" not in s["controlWarn"]

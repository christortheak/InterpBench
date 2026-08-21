"""D4 — the validation read layer as a DECLARED measurement decision.

The historical rule made it a side effect of the injection conditions ("the
layer a condition steers this concept at, else mid-network"), so moving the
validation read meant editing steering conditions — a different decision
entirely. On 2026-07-26 a researcher testing a readout-layer hypothesis had to
hand-write nine conditions at layer 41 because the manifest offered no way to
say the thing they meant.

Mirror of Swift ``ValidationLayerRuleTests``.
"""

import json
import os

import pytest

from steerlab_server.experiment import validation_layer as vl
from steerlab_server.experiment.manifest import Manifest


def _resolve(layer=None, fraction=None, condition=None, layer_count=62):
    return vl.resolve(declared_layer=layer, declared_fraction=fraction,
                      condition_layer=condition, layer_count=layer_count)


# --- precedence --------------------------------------------------------------

def test_a_declared_index_beats_everything():
    r = _resolve(layer=41, condition=7)
    assert r.layer == 41 and r.source == vl.DECLARED_INDEX


def test_a_declared_fraction_beats_the_legacy_rule():
    r = _resolve(fraction=0.66, condition=7)
    assert r.layer == 40 and r.source == vl.DECLARED_FRACTION


def test_the_legacy_rule_is_unchanged_when_nothing_is_declared():
    """Preserved EXACTLY, or existing manifests change their numbers — and
    their content hashes — on upgrade."""
    assert _resolve(condition=7).layer == 7
    assert _resolve(condition=7).source == vl.STEERING_CONDITION
    assert _resolve().layer == 31
    assert _resolve().source == vl.MID_NETWORK


# --- clamping ----------------------------------------------------------------

def test_an_out_of_range_declaration_clamps_rather_than_crashing():
    assert _resolve(layer=999).layer == 61
    assert _resolve(layer=0).layer == 0
    assert _resolve(fraction=1.0).layer == 61
    assert _resolve(fraction=0.0).layer == 0
    assert _resolve(fraction=0.5, layer_count=1).layer == 0


# --- the report says both numbers -------------------------------------------

def test_the_summary_carries_index_depth_and_reason():
    # An index means nothing without the depth it represents, and a depth
    # means nothing without the model it resolved against.
    summary = _resolve(layer=41).summary
    assert "layer 41 of 62" in summary
    assert "0.67 depth" in summary
    assert "declared validationLayer" in summary


def test_the_legacy_summary_says_how_to_say_it_directly():
    assert "declare validationLayer" in _resolve(condition=7).summary
    assert "declare validationLayer" in _resolve().summary


# --- the ambiguity refusal ---------------------------------------------------

def test_declaring_both_an_index_and_a_fraction_is_a_violation():
    problem = vl.violation(41, 0.66)
    assert problem and "both" in problem
    assert vl.violation(41, None) is None
    assert vl.violation(None, 0.66) is None
    assert vl.violation(None, None) is None


def test_nonsense_declarations_are_refused():
    assert vl.violation(-1, None) is not None
    assert vl.violation(None, 1.5) is not None
    assert vl.violation(None, float("nan")) is not None


def test_the_violation_surfaces_through_manifest_verify():
    m = Manifest.from_dict({"name": "x", "modelID": "org/m",
                            "validationLayer": 3,
                            "validationLayerFraction": 0.5})
    assert any("validationLayer" in v for v in m.verify())
    clean = Manifest.from_dict({"name": "x", "modelID": "org/m",
                                "validationLayer": 3})
    assert not any("validationLayer and" in v for v in clean.verify())


# --- depth lists (validate-at-the-sweep-layers, 2026-08-01) ------------------

def test_a_depth_list_resolves_every_entry_in_declared_order():
    resolutions = vl.resolve_all(
        declared_fractions=[0.5, 0.6, 0.7, 0.8], layer_count=62)
    assert [r.layer for r in resolutions] == [31, 37, 43, 49]
    assert all(r.source == vl.DECLARED_FRACTION for r in resolutions)


def test_the_scalar_path_is_a_one_element_list():
    """Every existing manifest keeps its single resolution — same layer,
    same source, through the same function the loop consumes."""
    resolutions = vl.resolve_all(condition_layer=7, layer_count=62)
    assert len(resolutions) == 1
    assert resolutions[0].layer == 7
    assert resolutions[0].source == vl.STEERING_CONDITION


def test_two_entries_resolving_to_one_layer_refuse():
    """0.60 and 0.61 of 62 layers are both layer 37: silently collapsing
    them would misreport what was declared."""
    with pytest.raises(RuntimeError, match="both resolve to layer 37"):
        vl.resolve_all(declared_fractions=[0.6, 0.61], layer_count=62)


def test_an_out_of_range_index_in_a_list_refuses():
    with pytest.raises(RuntimeError, match="not silently clamped"):
        vl.resolve_all(declared_layers=[100], layer_count=62)


def test_plural_declarations_are_exclusive_with_everything_else():
    assert "exactly one" in vl.violation(
        41, None, declared_layers=[1, 2])
    assert "exactly one" in vl.violation(
        None, None, declared_layers=[1], declared_fractions=[0.5])
    assert vl.violation(None, None, declared_fractions=[0.5, 0.6]) is None


def test_plural_declarations_refuse_empty_duplicates_and_nonsense():
    assert "non-empty" in vl.violation(None, None, declared_layers=[])
    assert "duplicate" in vl.violation(None, None, declared_layers=[3, 3])
    assert "non-negative" in vl.violation(None, None, declared_layers=[-1])
    assert "duplicate" in vl.violation(
        None, None, declared_fractions=[0.5, 0.5])
    assert "[0, 1]" in vl.violation(None, None, declared_fractions=[1.5])
    # bool is an int subtype: a layer index of `true` is a typo.
    assert "non-negative" in vl.violation(None, None, declared_layers=[True])


def test_the_plural_violation_surfaces_through_manifest_verify():
    m = Manifest.from_dict({"name": "x", "modelID": "org/m",
                            "validationLayerFraction": 0.5,
                            "validationLayerFractions": [0.5, 0.6]})
    assert any("exactly one" in v for v in m.verify())
    clean = Manifest.from_dict({"name": "x", "modelID": "org/m",
                                "validationLayerFractions": [0.5, 0.6]})
    assert not any("validationLayer" in v for v in clean.verify())


def test_the_depth_list_fixture_matches_this_implementation():
    """Cross-engine: Swift `ValidationLayerRuleTests` consumes the same
    fixture, so both engines resolve a declared list identically."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Tests", "Fixtures", "cross-engine", "validation-depth-lists.json")
    with open(path, encoding="utf-8") as handle:
        cases = json.load(handle)
    assert len(cases) >= 6
    for case in cases:
        if "refusal" in case:
            with pytest.raises(RuntimeError, match=case["refusal"]):
                vl.resolve_all(**case["input"])
            continue
        resolutions = vl.resolve_all(**case["input"])
        assert [(r.layer, r.source) for r in resolutions] \
            == [(e["layer"], e["source"]) for e in case["expect"]], case["label"]


# --- the fixture the Swift suite consumes ------------------------------------

def test_the_committed_fixture_matches_this_implementation():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Tests", "Fixtures", "cross-engine", "validation-layers.json")
    with open(path, encoding="utf-8") as handle:
        cases = json.load(handle)
    assert len(cases) >= 8
    for case in cases:
        r = vl.resolve(**case["input"])
        assert r.layer == case["layer"], case["label"]
        assert r.source == case["source"], case["label"]
        assert abs(r.depth_fraction - case["depthFraction"]) < 1e-9, case["label"]

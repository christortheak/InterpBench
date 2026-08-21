"""Scalar reading probe: items IO, parsing, and layer-selection training."""

import pytest

from steerlab_server.experiment import probes


def test_save_read_roundtrip(tmp_path):
    info = probes.save_items("fear", [
        {"text": "I'm terrified", "expresses": True, "split": "build"},
        {"text": "calm afternoon", "expresses": False, "split": "build"},
        {"text": "  ", "expresses": True}], root=str(tmp_path))
    assert info["items"] == 2 and info["positive"] == 1
    rows = probes.read_items("fear", root=str(tmp_path))
    assert rows[0]["expresses"] is True


def test_parse_items_jsonl_and_array():
    j = probes.parse_items('{"text":"a","expresses":true}\n{"text":"b","expresses":false}')
    assert len(j) == 2
    a = probes.parse_items('[{"text":"a","expresses":true}]')
    assert a[0]["text"] == "a"


def test_train_picks_separating_layer():
    # layer 0 is noise, layer 1 separates expresses by sign.
    items = [
        {"text": "p1", "expresses": True, "split": "build"},
        {"text": "p2", "expresses": True, "split": "build"},
        {"text": "n1", "expresses": False, "split": "build"},
        {"text": "n2", "expresses": False, "split": "build"},
        {"text": "pv", "expresses": True, "split": "validation"},
        {"text": "nv", "expresses": False, "split": "validation"},
    ]
    acts = [
        [[0.0, 0.0], [2.0, 0.0]],   # p1
        [[0.1, 0.0], [2.2, 0.0]],   # p2
        [[0.0, 0.0], [-2.0, 0.0]],  # n1
        [[0.1, 0.0], [-2.2, 0.0]],  # n2
        [[0.0, 0.0], [2.1, 0.0]],   # pv
        [[0.1, 0.0], [-2.1, 0.0]],  # nv
    ]
    result = probes.train(items=items, activations_by_layer=acts)
    assert result["layer"] == 1
    assert result["accuracy"] == 1.0
    assert "direction" in result["probe"]


def test_train_requires_both_classes():
    with pytest.raises(ValueError):
        probes.train(items=[{"text": "a", "expresses": True, "split": "build"}],
                     activations_by_layer=[[[1.0]]])


# --- deterministic content-hash split (cross-engine contract, 2026-07-13) -----

# Fixture texts with their PRECOMPUTED sha256-sorted order. Sorted ascending
# by sha256(text) hex the order is: 'night market', 'burning bridge',
# 'the court affirms', 'storm at sea', 'frozen river', 'green meadow',
# 'quiet library' — so sorted index 4 ('frozen river') is the ONLY validation
# member (index % 5 == 4). Swift must produce the identical membership.
SPLIT_FIXTURE_TEXTS = [
    "the court affirms", "storm at sea", "quiet library", "burning bridge",
    "green meadow", "night market", "frozen river",
]
EXPECTED_VALIDATION_TEXTS = {"frozen river"}


def test_content_hash_split_fixture_membership():
    items = [{"text": t, "expresses": i % 2 == 0}
             for i, t in enumerate(SPLIT_FIXTURE_TEXTS)]
    val_idx, build_idx = probes._content_hash_split(items, list(range(len(items))))
    assert {items[i]["text"] for i in val_idx} == EXPECTED_VALIDATION_TEXTS
    assert {items[i]["text"] for i in build_idx} == \
        set(SPLIT_FIXTURE_TEXTS) - EXPECTED_VALIDATION_TEXTS
    # Membership depends on CONTENT, not file order: shuffling the items
    # holds out the same texts.
    reordered = list(reversed(items))
    val2, _build2 = probes._content_hash_split(reordered,
                                               list(range(len(reordered))))
    assert {reordered[i]["text"] for i in val2} == EXPECTED_VALIDATION_TEXTS


def test_train_uses_content_hash_split_for_untagged_pools():
    # No item carries split: 'frozen river' must be the held-out example.
    # Layer 0 separates by sign, so a correctly-classified holdout gives 1.0.
    items = [
        {"text": "the court affirms", "expresses": True},
        {"text": "storm at sea", "expresses": True},
        {"text": "quiet library", "expresses": False},
        {"text": "burning bridge", "expresses": False},
        {"text": "green meadow", "expresses": False},
        {"text": "night market", "expresses": False},
        {"text": "frozen river", "expresses": True},
    ]
    acts = [[[2.0 if item["expresses"] else -2.0, 0.0]] for item in items]
    result = probes.train(items=items, activations_by_layer=acts)
    assert result["valCount"] == 1
    assert result["buildCount"] == 6
    assert result["accuracy"] == 1.0


def test_content_hash_split_small_pool_has_no_holdout():
    items = [{"text": f"t{i}", "expresses": True} for i in range(4)]
    val_idx, build_idx = probes._content_hash_split(items, list(range(4)))
    assert val_idx == [] and sorted(build_idx) == [0, 1, 2, 3]

"""The neutral token-bank draw — self-contained, and pinned by literals.

Cross-engine twin: ``Tests/SteeringKitTests/TokenBankDownsamplerTests.swift``.
The LITERAL blocks below appear byte-identically in both suites: the same seed
derivation, the same SplitMix64 stream, the same selected indices. Two things
depend on that.

1. **A runtime upgrade cannot move the draw undetected.** The old server draw
   was ``random.sample`` seeded from a corpus hash, and CPython promises
   reproducibility across versions only for ``Random.random()`` — a Python
   upgrade could have silently rebuilt the neutral-PC basis from different
   token positions while every stamp said the recipe was identical
   (2026-08-28 audit, convention note 9). Literals, not properties, are what
   make that impossible.
2. **The two engines now draw the same rows.** Seed derivation, RNG and
   selection rule converged; the default row CAP deliberately did not (see the
   module docstring), so a bit-identical bank needs the same cap passed on
   both sides.
"""

from __future__ import annotations

from steerlab_server.steering import token_bank_downsampling as downsampling

# ------------------------------------------------------------------- literals
#
# Twin values, asserted verbatim on both engines.
LITERAL_SEED_ABC123 = 7_827_605_053_139_634_307
LITERAL_SPLITMIX64_FROM_7 = [
    7_191_089_600_892_374_487,
    309_689_372_594_955_804,
    16_616_101_746_815_609_346,
    10_753_165_928_301_472_203,
]
LITERAL_SELECTION_1000_8_42 = [26, 182, 338, 396, 413, 552, 855, 942]
LITERAL_SELECTION_10000_12_ABC123 = [
    1174, 1287, 2726, 2780, 3665, 4125, 5279, 5775, 6565, 6917, 7573, 8481,
]
LITERAL_CORPUS_HASH_AB_C = \
    "601d5476e2ccfe2c87a2bba7a322659734a05749d5b5aa781f513e4912db0d5f"


def test_splitmix64_stream_is_the_pinned_sequence():
    rng = downsampling.SplitMix64(7)
    assert [rng.next() for _ in range(4)] == LITERAL_SPLITMIX64_FROM_7
    assert all(0 <= value < 2 ** 64 for value in LITERAL_SPLITMIX64_FROM_7)


def test_seed_derivation_is_the_pinned_cross_engine_rule():
    """SHA-256 over the hash STRING's UTF-8, first 8 bytes big-endian — the
    Swift derivation, adopted here. The superseded server rule was
    ``int(hash[:16], 16) % 2**63``, which is a different number."""
    assert downsampling.seed_from_corpus_hash("abc123") == LITERAL_SEED_ABC123
    assert downsampling.seed_from_corpus_hash("abc123") == \
        downsampling.seed_from_corpus_hash("abc123")
    assert downsampling.seed_from_corpus_hash("abc124") != LITERAL_SEED_ABC123
    superseded = int("abc123".ljust(16, "0")[:16], 16) % (2 ** 63)
    assert superseded != LITERAL_SEED_ABC123


def test_selection_is_the_pinned_draw():
    assert downsampling.selected_indices(1000, 8, 42) == LITERAL_SELECTION_1000_8_42
    assert downsampling.selected_indices(
        10_000, 12, LITERAL_SEED_ABC123) == LITERAL_SELECTION_10000_12_ABC123


def test_selection_is_bounded_sorted_and_in_range():
    picked = downsampling.selected_indices(5_000, 128, 7)
    assert len(picked) == 128
    assert picked == sorted(picked)
    assert len(set(picked)) == 128
    assert all(0 <= i < 5_000 for i in picked)


def test_under_the_cap_everything_is_kept():
    assert downsampling.selected_indices(100, 4096, 1) == list(range(100))
    assert downsampling.selected_indices(4096, 4096, 1) == list(range(4096))
    assert downsampling.selected_indices(0, 16, 1) == []
    assert downsampling.selected_indices(16, 0, 1) == []


def test_index_set_mirrors_the_index_list():
    seed = downsampling.seed_from_corpus_hash("abc123")
    assert downsampling.selected_index_set(10_000, 512, seed) == \
        set(downsampling.selected_indices(10_000, 512, seed))
    # None = "keep every row", so callers skip the membership lookup.
    assert downsampling.selected_index_set(100, 4096, 1) is None
    assert downsampling.selected_index_set(4096, 4096, 1) is None
    # The server-only sentinel: no cap at all.
    assert downsampling.selected_index_set(10_000, None, 1) is None
    # A cap of zero selects NOTHING, matching the twin. It used to mean
    # "uncapped" here, which turned a plausible typo into an unbounded bank.
    assert downsampling.selected_index_set(16, 0, 1) == set()


def test_corpus_hash_is_length_prefixed():
    assert downsampling.corpus_hash(["ab", "c"]) == LITERAL_CORPUS_HASH_AB_C
    assert downsampling.corpus_hash(["a", "bc"]) != LITERAL_CORPUS_HASH_AB_C


def test_the_extractor_helper_delegates_to_this_algorithm():
    """``deterministic_row_selection`` is the name the bank driver calls; the
    rule behind it is this module, so the pin above covers the live path."""
    from steerlab_server.steering.extractor import deterministic_row_selection

    assert deterministic_row_selection(1000, 8, seed=42) == \
        set(LITERAL_SELECTION_1000_8_42)

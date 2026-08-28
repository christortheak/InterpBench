"""Deterministic row downsampling for neutral token banks — the line-for-line
twin of ``Sources/SteeringKit/Extraction/TokenBankDownsampling.swift``.

The all-layer per-token float32 bank feeding the neutral-PC basis is the
extraction path's memory hazard at scale, so rows must be bounded before large
corpora are usable. The bound is a per-layer row cap, and the rows kept under
it are chosen by a **seeded partial Fisher–Yates draw over SplitMix64** —
pure, self-contained, and identical on both engines for a given
``(count, cap, seed)``.

**Why this module exists (2026-08-28 audit, convention note 9).** The draw
used to be ``random.sample(range(total), cap)`` seeded from a corpus hash.
CPython guarantees reproducibility across versions only for
``Random.random()``; ``sample()``'s algorithm is explicitly allowed to change,
so a Python upgrade could have silently moved which token positions the
neutral-PC basis was built from while every stamp on the artifact said the
recipe was identical. A basis is not the kind of thing whose bytes you notice
drifting. The algorithm here depends on nothing but 64-bit integer
arithmetic, and a literal fixture (``test_token_bank_downsampling.py`` /
``TokenBankDownsamplerTests``) pins its output so a future runtime CANNOT move
it undetected.

**Cross-engine contract, and the one part of it that is engine-local.** Seed
derivation (:func:`seed_from_corpus_hash`), the RNG (:class:`SplitMix64`), and
the selection rule (:func:`selected_indices`) are now IDENTICAL on both
engines: the same corpus hash and the same cap select byte-identical rows on
the Mac and on the server. The DEFAULT cap is deliberately not shared — this
engine caps at ``extractor.DEFAULT_MAX_TOKEN_ROWS`` (2048) and Swift at
``TokenBankDownsampler.defaultRowCapPerLayer`` (4096) — because a cap is a
memory bound rather than a rule, and the two engines' banks are not the same
size in memory: Swift refuses up front through ``NeutralBankBudget.preflight``,
while this engine materializes the kept rows as Python floats for every
captured layer with no preflight at all. Importing the larger default here
would double an unguarded transient on the engine that runs on shared cluster
GPUs, for no gain in what the draw MEANS. Pass the same ``maxTokenRows`` on
both engines when a bit-identical bank is what you want.

Nothing pinned depends on the draw: a neutral-PC basis travels by
``neutralPCBasisHash``, the SHA-256 of the basis file's own bytes, so an
existing pinned basis keeps verifying unchanged — the draw only decides what a
NEW basis is built from.
"""

from __future__ import annotations

import hashlib

_U64 = (1 << 64) - 1

#: SplitMix64's golden-ratio increment and two mixing multipliers, the
#: published constants. Twin: ``SplitMix64`` in
#: ``Sources/SteeringKit/Extraction/SeededRandom.swift``.
_GAMMA = 0x9E37_79B9_7F4A_7C15
_MIX_1 = 0xBF58_476D_1CE4_E5B9
_MIX_2 = 0x94D0_49BB_1331_11EB


class SplitMix64:
    """The shared cross-engine PRNG, in the same shape the Swift struct has.

    Self-contained by design: every operation is 64-bit integer arithmetic on
    a single state word, so the sequence is a property of THIS FILE and not of
    the interpreter's standard library. The sibling RepE orientation draw uses
    the same generator for the same reason.
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = int(seed) & _U64

    def next(self) -> int:
        self._state = (self._state + _GAMMA) & _U64
        z = self._state
        z = ((z ^ (z >> 30)) * _MIX_1) & _U64
        z = ((z ^ (z >> 27)) * _MIX_2) & _U64
        return (z ^ (z >> 31)) & _U64


def seed_from_corpus_hash(corpus_hash: str) -> int:
    """Seed derived from a corpus hash (hex, or any string): the first 8 bytes
    of SHA-256 over the hash's UTF-8, big-endian.

    Twin: ``TokenBankDownsampler.seed(fromCorpusHash:)``. The old server-side
    derivation was ``int(hash[:16], 16) % 2**63`` — a different number from a
    different half of a different digest, so the two engines drew different
    rows from the same corpus even where everything else agreed.
    """
    digest = hashlib.sha256(corpus_hash.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def corpus_hash(texts: list[str]) -> str:
    """Deterministic hash of raw corpus texts, for callers holding texts but no
    file hash. Length-prefixed (big-endian ``uint64`` per text) so ``["ab",
    "c"]`` and ``["a", "bc"]`` differ. Twin:
    ``TokenBankDownsampler.corpusHash(texts:)``."""
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def selected_indices(count: int, cap: int, seed: int) -> list[int]:
    """The row indices to KEEP out of ``count``, at most ``cap``, chosen by a
    seeded partial Fisher–Yates draw and returned sorted ascending.

    Deterministic for a ``(count, cap, seed)`` triple, bounded by ``cap``, and
    order-stable (kept rows preserve their original relative order). Twin:
    ``TokenBankDownsampler.selectedIndices(count:cap:seed:)`` — same loop, same
    modulo reduction, same sort.
    """
    count = int(count)
    cap = int(cap)
    if count <= 0 or cap <= 0:
        return []
    if count <= cap:
        return list(range(count))
    rng = SplitMix64(seed)
    pool = list(range(count))
    for position in range(cap):
        remaining = count - position
        offset = rng.next() % remaining
        other = position + offset
        pool[position], pool[other] = pool[other], pool[position]
    return sorted(pool[:cap])


def selected_index_set(count: int, cap: int | None, seed: int) -> set[int] | None:
    """:func:`selected_indices` as a membership set, so a bank can DROP
    over-cap rows AS THEY ARRIVE instead of materializing every row and
    downsampling afterwards.

    ``None`` means "keep every row", which lets callers skip the lookup
    entirely. Twin: ``TokenBankDownsampler.selectedIndexSet(count:cap:seed:)``,
    with one server-only case the Swift signature cannot express: ``cap is
    None`` is this engine's explicit "no cap at all" sentinel and keeps
    everything. A cap of ``0`` or less selects NOTHING on both engines — it
    used to mean "uncapped" here, which turned a plausible typo into an
    unbounded bank.
    """
    if cap is None:
        return None
    if int(cap) <= 0:
        return set()
    if int(count) <= int(cap):
        return None
    return set(selected_indices(count, int(cap), seed))

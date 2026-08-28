"""Concept validation diagnostics (parallel to Swift ``ConceptStats`` +
``ConceptBuilder.rebuild``).

Headline checks that a direction is real and discriminative:
- **held-out accuracy** — train the direction on ~80% of stimuli, classify the
  held-out ~20% by projection sign (out-of-sample signal);
- **split-half cosine** — directions from disjoint halves; high = agreement;
- **norm by layer** — where the contrast lives;
- **outliers** — stimuli least aligned with the direction (possible mislabels);
- **control cosines** — cosine against other saved concepts (warn if |·| > 0.6,
  i.e. directions collapsing into one).

The first two splits are drawn by the CONTENT-HASH rule
(:func:`content_hash_order`), never by row order, and both engines draw them
identically. That is why the split-taking functions require the stimulus TEXTS
alongside their activations.

All pure math over captured activations, so it is unit-testable without a model.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Sequence

from ..steering import vector_math as vm

CONTROL_COSINE_WARN = 0.6

#: Minimum stimuli per class before either split-based diagnostic is computed.
#: One rule on both engines (Swift ``ConceptStats``): below this a ~20% holdout
#: is one or two rows and the "accuracy" it reports is noise wearing a
#: percentage sign.
MINIMUM_ROWS_PER_CLASS = 6
MINIMUM_ROWS_PER_CLASS_SPLIT_HALF = 4


def _mean_proj(acts: list[list[float]], direction: list[float]) -> float:
    return statistics.fmean(vm.dot(a, direction) for a in acts) if acts else 0.0


def content_hash_order(texts: Sequence[str]) -> list[int]:
    """Row indices sorted ascending by the lowercase SHA-256 hex of the row's
    UTF-8 text (ties broken by the text itself).

    CROSS-ENGINE SPLIT CONTRACT (2026-08-28 audit, F3). Swift twin:
    ``ConceptStats.contentHashOrder``. The same rule already governs the
    reading-probe validation split (``probes._content_hash_split`` /
    ``ConceptBuilder.splitExamples``, 2026-07-13); this brings the concept
    screening diagnostics under it.

    **Why content-derived, not index-derived.** The old rules were positional:
    Python held out the LAST ~20% of each class in file order and split even/odd
    for split-half; Swift held out ``index % 5 == 4``. Both make the statistic a
    function of how the stimulus file happens to be ORDERED. Stimulus files are
    routinely authored in topic blocks, so the file-order tail is a single topic
    cluster (pessimistic held-out accuracy) while a parity split puts adjacent
    near-duplicates on both sides of the halves (optimistic split-half cosine) —
    and neither shows any variance, because the split is deterministic. Sorting
    by a hash of the row's own CONTENT makes membership independent of row
    order: shuffle the file and every number is unchanged, and the two engines
    agree byte-for-byte on identical data.

    **Why a hash ORDER rather than a hash MODULUS.** Taking ``hash % 5 == 4``
    would also be content-derived, but the test-set SIZE would then be binomial
    rather than exactly ~20%, so the reported ``testCount`` would wander with
    the concept. Sorting and taking every fifth of sorted order keeps the exact
    proportions the positional rules had while throwing away the order.

    **Why no RNG.** A seeded shuffle would need the two engines to share a
    generator; sibling paths in this repo either share SplitMix64 or explicitly
    document the divergence. A content hash needs neither — SHA-256 of UTF-8 is
    the same on every platform, so there is no RNG left to diverge.
    """
    return sorted(
        range(len(texts)),
        key=lambda i: (hashlib.sha256(texts[i].encode("utf-8")).hexdigest(),
                       texts[i]))


def held_out_indices(texts: Sequence[str]) -> set[int]:
    """The ~20% test membership: every 5th row of :func:`content_hash_order`
    (0-based sorted position % 5 == 4). Swift twin ``ConceptStats.heldOutIndices``."""
    return {index for position, index in enumerate(content_hash_order(texts))
            if position % 5 == 4}


def split_half_indices(texts: Sequence[str]) -> set[int]:
    """The SECOND half's membership: odd positions of :func:`content_hash_order`.
    Swift twin ``ConceptStats.splitHalfSecondIndices``."""
    return {index for position, index in enumerate(content_hash_order(texts))
            if position % 2 == 1}


def _partition(rows: list[list[float]], members: set[int]
               ) -> tuple[list[list[float]], list[list[float]]]:
    """``(outside, inside)`` — rows not in ``members`` first."""
    return ([r for i, r in enumerate(rows) if i not in members],
            [r for i, r in enumerate(rows) if i in members])


def held_out_accuracy(pos: list[list[float]], neg: list[list[float]],
                      method: vm.ExtractionMethod, *,
                      positive_texts: Sequence[str],
                      negative_texts: Sequence[str]) -> dict | None:
    """Out-of-sample accuracy over the content-hash held-out split.

    The split is :func:`held_out_indices` per class (stratified), so it depends
    on the stimulus TEXTS and never on their order in the file.
    """
    if len(pos) < MINIMUM_ROWS_PER_CLASS or len(neg) < MINIMUM_ROWS_PER_CLASS:
        return None
    if len(positive_texts) != len(pos) or len(negative_texts) != len(neg):
        raise ValueError(
            "held_out_accuracy needs one text per activation row "
            f"({len(positive_texts)}/{len(pos)} positive, "
            f"{len(negative_texts)}/{len(neg)} negative) — the split is "
            "derived from the stimulus content, not from row order")
    train_p, test_p = _partition(pos, held_out_indices(positive_texts))
    train_n, test_n = _partition(neg, held_out_indices(negative_texts))
    total = len(test_p) + len(test_n)
    if len(train_p) < 2 or len(train_n) < 2 or total < 2:
        return None
    try:
        d = vm.direction(train_p, train_n, method)
    except vm.SteeringVectorError:
        return None
    mp, mn = _mean_proj(train_p, d), _mean_proj(train_n, d)
    thr = (mp + mn) / 2
    orient = 1.0 if mp >= mn else -1.0
    correct = sum(1 for a in test_p if orient * (vm.dot(a, d) - thr) > 0)
    correct += sum(1 for a in test_n if orient * (vm.dot(a, d) - thr) < 0)
    return {"accuracy": correct / total, "testCount": total}


def split_half_cosine(pos: list[list[float]], neg: list[list[float]],
                      method: vm.ExtractionMethod, *,
                      positive_texts: Sequence[str],
                      negative_texts: Sequence[str]) -> float | None:
    """Cosine between directions built from the two content-hash halves.

    Halves are the even/odd positions of :func:`content_hash_order` per class —
    content-derived, so authoring the file in topic blocks no longer splits
    adjacent near-duplicates across the halves and flatters the number.
    """
    if (len(pos) < MINIMUM_ROWS_PER_CLASS_SPLIT_HALF
            or len(neg) < MINIMUM_ROWS_PER_CLASS_SPLIT_HALF):
        return None
    if len(positive_texts) != len(pos) or len(negative_texts) != len(neg):
        raise ValueError(
            "split_half_cosine needs one text per activation row "
            f"({len(positive_texts)}/{len(pos)} positive, "
            f"{len(negative_texts)}/{len(neg)} negative) — the split is "
            "derived from the stimulus content, not from row order")
    pos0, pos1 = _partition(pos, split_half_indices(positive_texts))
    neg0, neg1 = _partition(neg, split_half_indices(negative_texts))
    if min(len(pos0), len(pos1), len(neg0), len(neg1)) < 2:
        return None
    try:
        d0 = vm.direction(pos0, neg0, method)
        d1 = vm.direction(pos1, neg1, method)
        return vm.cosine_similarity(d0, d1)
    except vm.SteeringVectorError:
        return None


def scenario_accuracy(*, direction: list[float], positive: list[list[float]],
                      negative: list[list[float]], scenarios: list[list[float]],
                      labels: list[bool]) -> float | None:
    """True held-out accuracy on labeled scenarios (parallel to Swift
    ``ExperimentTasks.scenarioAccuracy``).

    Threshold is the midpoint of the training-class mean projections; a scenario
    is correct when ``(projection > midpoint) == expresses``. This is the
    convergent half of the circularity firewall — the held-out scenarios played
    no role in extraction.
    """
    if not scenarios or not positive or not negative:
        return None
    midpoint = (vm.dot(direction, vm.mean(positive))
                + vm.dot(direction, vm.mean(negative))) / 2
    correct = sum(1 for act, lab in zip(scenarios, labels)
                  if (vm.dot(direction, act) > midpoint) == lab)
    return correct / len(scenarios)


def scenario_accuracy_grand_mean(*, direction: list[float],
                                 concept: list[list[float]],
                                 population: list[list[float]],
                                 scenarios: list[list[float]],
                                 labels: list[bool]) -> float | None:
    """Held-out accuracy for a grand-mean concept, which has no negative
    class: the population (every row of the multi-concept corpus) plays the
    reference role, and the threshold is the midpoint between the
    concept-mean and population-mean projections — the exact analogue of the
    paired midpoint rule."""
    if not scenarios or not concept or not population:
        return None
    midpoint = (vm.dot(direction, vm.mean(concept))
                + vm.dot(direction, vm.mean(population))) / 2
    correct = sum(1 for act, lab in zip(scenarios, labels)
                  if (vm.dot(direction, act) > midpoint) == lab)
    return correct / len(scenarios)


def compute(*, positive_by_layer: list[list[list[float]]],
            negative_by_layer: list[list[list[float]]],
            method: vm.ExtractionMethod,
            positive_texts: Sequence[str],
            negative_texts: Sequence[str],
            control_vectors: dict[str, list[float]] | None = None) -> dict:
    """``*_by_layer`` is ``[stimulus][layer][hidden]`` (the extractor's shape).

    ``positive_texts``/``negative_texts`` are the stimuli those rows came from,
    row-aligned. They are required, not optional: the held-out and split-half
    splits are derived from the stimulus CONTENT (:func:`content_hash_order`),
    which is the whole point of the 2026-08-28 change — an activation matrix
    alone can only be split by row order, and row order is exactly what the two
    engines used to disagree about.
    """
    layer_count = len(positive_by_layer[0]) if positive_by_layer else 0
    stats_layer = layer_count // 2

    def at(rows, layer):
        return [r[layer] for r in rows]

    pos = at(positive_by_layer, stats_layer)
    neg = at(negative_by_layer, stats_layer)
    direction = vm.direction(pos, neg, method)

    # norm-by-layer uses the mean-difference magnitude (LAT is norm-matched to it).
    norm_by_layer = [vm.l2_norm(vm.mean_difference(at(positive_by_layer, l),
                                                   at(negative_by_layer, l)))
                     for l in range(layer_count)]

    mp, mn = _mean_proj(pos, direction), _mean_proj(neg, direction)
    thr = (mp + mn) / 2
    orient = 1.0 if mp >= mn else -1.0
    margins = ([("positive", i, orient * (vm.dot(pos[i], direction) - thr))
                for i in range(len(pos))]
               + [("negative", i, orient * (thr - vm.dot(neg[i], direction)))
                  for i in range(len(neg))])
    outliers = [{"side": s, "index": i, "margin": round(m, 4)}
                for s, i, m in sorted(margins, key=lambda x: x[2])[:5]]

    controls = {}
    for concept, vec in (control_vectors or {}).items():
        if len(vec) == len(direction):
            try:
                controls[concept] = round(vm.cosine_similarity(direction, vec), 4)
            except vm.SteeringVectorError:
                pass

    return {
        "statsLayer": stats_layer,
        "heldOut": held_out_accuracy(pos, neg, method,
                                     positive_texts=positive_texts,
                                     negative_texts=negative_texts),
        "splitHalf": split_half_cosine(pos, neg, method,
                                       positive_texts=positive_texts,
                                       negative_texts=negative_texts),
        "normByLayer": [round(n, 4) for n in norm_by_layer],
        "outliers": outliers,
        "controlCosines": dict(sorted(controls.items(),
                                      key=lambda kv: abs(kv[1]), reverse=True)),
        "controlWarn": [c for c, v in controls.items() if abs(v) > CONTROL_COSINE_WARN],
    }

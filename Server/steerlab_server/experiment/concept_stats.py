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

All pure math over captured activations, so it is unit-testable without a model.
"""

from __future__ import annotations

import statistics

from ..steering import vector_math as vm

CONTROL_COSINE_WARN = 0.6


def _mean_proj(acts: list[list[float]], direction: list[float]) -> float:
    return statistics.fmean(vm.dot(a, direction) for a in acts) if acts else 0.0


def held_out_accuracy(pos: list[list[float]], neg: list[list[float]],
                      method: vm.ExtractionMethod) -> dict | None:
    if len(pos) < 6 or len(neg) < 6:
        return None
    hp, hn = max(1, len(pos) // 5), max(1, len(neg) // 5)
    train_p, test_p = pos[:-hp], pos[-hp:]
    train_n, test_n = neg[:-hn], neg[-hn:]
    d = vm.direction(train_p, train_n, method)
    mp, mn = _mean_proj(train_p, d), _mean_proj(train_n, d)
    thr = (mp + mn) / 2
    orient = 1.0 if mp >= mn else -1.0
    correct = sum(1 for a in test_p if orient * (vm.dot(a, d) - thr) > 0)
    correct += sum(1 for a in test_n if orient * (vm.dot(a, d) - thr) < 0)
    total = len(test_p) + len(test_n)
    return {"accuracy": correct / total, "testCount": total}


def split_half_cosine(pos: list[list[float]], neg: list[list[float]],
                      method: vm.ExtractionMethod) -> float | None:
    if len(pos) < 4 or len(neg) < 4:
        return None
    try:
        d0 = vm.direction(pos[0::2], neg[0::2], method)
        d1 = vm.direction(pos[1::2], neg[1::2], method)
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
            control_vectors: dict[str, list[float]] | None = None) -> dict:
    """``*_by_layer`` is ``[stimulus][layer][hidden]`` (the extractor's shape)."""
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
        "heldOut": held_out_accuracy(pos, neg, method),
        "splitHalf": split_half_cosine(pos, neg, method),
        "normByLayer": [round(n, 4) for n in norm_by_layer],
        "outliers": outliers,
        "controlCosines": dict(sorted(controls.items(),
                                      key=lambda kv: abs(kv[1]), reverse=True)),
        "controlWarn": [c for c, v in controls.items() if abs(v) > CONTROL_COSINE_WARN],
    }

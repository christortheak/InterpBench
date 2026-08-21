"""Pure CPU-side vector arithmetic for contrastive activation extraction.

A 1:1 NumPy port of Swift ``SteeringVectorMath.swift``. Kept off the GPU
deliberately: these run over small captured activations, and pure functions are
unit-testable against committed fixtures.

**Numerics:** the Swift side uses 32-bit ``Float`` throughout, so this module
computes in ``float32`` to stay numerically comparable. The LAT/PCA path keeps
the **deterministic Gram-matrix power-iteration** (no RNG) with the same two
deterministic starts, so results reproduce rather than merely approximate the
Swift output. Do not silently swap in ``np.linalg.svd`` — that requires a
re-validation, not an assumed equivalence (see the plan's risks).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

Vector = Sequence[float]
Rows = Sequence[Sequence[float]]

_F32 = np.float32


class SteeringVectorError(Exception):
    pass


class ExtractionMethod(str, Enum):
    """Direction-finding method. ``meanDifference``/``lat`` consume a paired
    positive/negative stimulus set; ``emotionGrandMean`` consumes a
    multi-concept story corpus (concept mean − corpus grand mean) and is
    dispatched through ``extractor.extract_grand_mean``, never
    :func:`direction`. The value string matches the Swift
    ``VectorExtractionRecipe.Method`` raw value and existing sidecars."""

    MEAN_DIFFERENCE = "meanDifference"
    LAT = "lat"
    GRAND_MEAN = "emotionGrandMean"
    #: mean(concept stories) − mean(DESIGNATED reference stories), both
    #: pooled (METHODS amendment ii). Same arithmetic as meanDifference over
    #: unpaired classes; a first-class method so the reference is PINNED
    #: recipe data and the pooled-reading policy has a native home, instead
    #: of hand-derived class directories (a live derivation script wrote a
    #: corpus NAME as one-character rows — the workaround this retires).
    DESIGNATED_REFERENCE = "designatedReference"
    #: NOT a direction-finding method: the direction already exists as a
    #: hash-pinned artifact and the "extraction" is a verified materialization
    #: of those bytes (post-hoc derived vectors — e.g. family-grand-mean
    #: centring — which no re-derivable recipe can reproduce from stimuli).
    #: Never reaches :func:`direction`; the lifecycle branches that ask about
    #: DATA (stimulus location, validation semantics) ask the artifact's
    #: SOURCE method instead — see ``ConceptRef.effective_method``.
    PINNED_ARTIFACT = "pinnedArtifact"
    #: NOT a direction-finding method either: an OPTIMIZED injection vector,
    #: found by backprop through the frozen model against a hashed
    #: target/anchor/capability dataset (experiment/optvec_train.py). It never
    #: reaches :func:`direction` — an optvec vector enters studies only as a
    #: ``pinnedArtifact`` concept whose sidecar records this as its SOURCE
    #: method. It is listed here so the lifecycle's DATA questions get an
    #: honest answer instead of an unknown-method fallback: there are no
    #: stimuli (the stimulusSetHash is the ``optvec:<composite>`` dataset
    #: hash), no source concept, and no held-out validation.jsonl — the
    #: evidence is the eval run's eval.json (plan §6).
    OPTVEC = "optvec"
    #: NOT a direction-finding method either: a Gemma Scope SAE DECODER ROW,
    #: written by ``experiment/gemma_scope.py`` and entering studies only as a
    #: ``pinnedArtifact`` concept whose sidecar records this as its SOURCE
    #: method. Listed here for the same reason as ``optvec``: the lifecycle's
    #: DATA questions need an honest answer instead of an unknown-method
    #: fallback. A decoder row has no stimuli (its stimulusSetHash is the
    #: ``gemmascope:<release>:<saeID>:<feature>`` identity), no source concept,
    #: and no held-out validation.jsonl — a feature is a coordinate in an SAE's
    #: dictionary, not a contrast between two authored classes. Its evidence is
    #: the discovery snapshot + qualification artifact recorded in the pinned
    #: SAE candidate roster (proposal r2 §4/§6).
    GEMMA_SCOPE_SAE = "gemmaScopeSAE"

    @property
    def label(self) -> str:
        return {"meanDifference": "Mean difference", "lat": "LAT paired direction (RepE-inspired)",
                "emotionGrandMean": "Grand mean (multi-concept)",
                "designatedReference": "Designated reference (stories − reference stories)",
                "pinnedArtifact": "Pinned artifact (hash-pinned derived vectors)",
                "optvec": "Optimized injection vector (OptVec)",
                "gemmaScopeSAE": "Gemma Scope SAE feature (decoder row)"}[self.value]

    @property
    def is_paired(self) -> bool:
        return self in (ExtractionMethod.MEAN_DIFFERENCE, ExtractionMethod.LAT)

    @property
    def is_designated_reference(self) -> bool:
        return self is ExtractionMethod.DESIGNATED_REFERENCE

    @property
    def is_pinned_artifact(self) -> bool:
        """Materialized from pinned bytes rather than derived from stimuli."""
        return self is ExtractionMethod.PINNED_ARTIFACT

    @property
    def is_optvec(self) -> bool:
        """Optimized against a hashed dataset, not read off stimuli. Every
        DATA-side lifecycle branch (stimulus location, validation semantics,
        pin surface) must ask this before assuming a concept has stimuli: an
        optvec direction has none, by construction."""
        return self is ExtractionMethod.OPTVEC

    @property
    def is_gemma_scope_sae(self) -> bool:
        """An imported Gemma Scope SAE decoder row (see the member's note)."""
        return self is ExtractionMethod.GEMMA_SCOPE_SAE

    @property
    def has_source_concept(self) -> bool:
        """False for directions that were never READ OFF a concept's stimuli:
        an optvec vector (optimized against hashed datasets) and a Gemma Scope
        SAE decoder row (a dictionary coordinate). Ask this — not
        ``is_optvec`` — at every DATA-side lifecycle branch: "where do this
        concept's stimuli live", "what does its held-out validation MEAN",
        "which position was it read at". Answering those for a direction with
        no source concept means inventing an obligation it can never meet, and
        the lifecycle used to answer them by falling back to meanDifference.
        """
        return not (self.is_optvec or self.is_gemma_scope_sae)

    # Explicit lifecycle semantics (external review 2026-07-31, finding 1):
    # "not is_paired" used to MEAN "grand mean", until a third method made
    # that two-valued shortcut wrong at every unswept site — recipe identity
    # demanded a nonexistent grandMeanCorpus and validation dereferenced it.
    # Every lifecycle branch now asks the question it actually means.

    @property
    def is_grand_mean(self) -> bool:
        """Extracts against the pinned multi-concept corpus population."""
        return self is ExtractionMethod.GRAND_MEAN

    @property
    def uses_story_corpus(self) -> bool:
        """Stimuli (and validation.jsonl) live under prompts/emotions/."""
        return self in (ExtractionMethod.GRAND_MEAN,
                        ExtractionMethod.DESIGNATED_REFERENCE)

    @property
    def uses_contrastive_validation(self) -> bool:
        """Validation scores against TWO class means (positive/negative or
        concept/reference), not a corpus population."""
        return self.is_paired or self.is_designated_reference


def _arr(rows: Rows) -> np.ndarray:
    a = np.asarray(rows, dtype=_F32)
    if a.ndim != 2:
        raise SteeringVectorError("expected a 2-D array of rows")
    return a


# --- basic arithmetic ------------------------------------------------------

def mean(rows: Rows) -> list[float]:
    a = _arr(rows)
    if a.shape[0] == 0:
        raise SteeringVectorError("emptyInput")
    return a.mean(axis=0).astype(_F32).tolist()


def mean_difference(positive: Rows, negative: Rows) -> list[float]:
    """CAA direction: mean(positive) − mean(negative)."""
    p = np.asarray(mean(positive), dtype=_F32)
    n = np.asarray(mean(negative), dtype=_F32)
    if p.shape != n.shape:
        raise SteeringVectorError(
            f"dimensionMismatch expected {p.shape} found {n.shape}")
    return (p - n).astype(_F32).tolist()


def grand_mean_difference(concept: Rows, population: Rows) -> list[float]:
    """Emotion-paper concept direction: mean(concept) − mean(whole corpus)."""
    c = np.asarray(mean(concept), dtype=_F32)
    g = np.asarray(mean(population), dtype=_F32)
    if c.shape != g.shape:
        raise SteeringVectorError("dimensionMismatch")
    return (c - g).astype(_F32).tolist()


def l2_norm(v: Vector) -> float:
    a = np.asarray(v, dtype=_F32)
    return float(np.sqrt(np.square(a).sum(dtype=_F32)))


def dot(a: Vector, b: Vector) -> float:
    return float(np.dot(np.asarray(a, dtype=_F32), np.asarray(b, dtype=_F32)))


def cosine_similarity(a: Vector, b: Vector) -> float:
    av = np.asarray(a, dtype=_F32)
    bv = np.asarray(b, dtype=_F32)
    if av.shape != bv.shape:
        raise SteeringVectorError("dimensionMismatch")
    na, nb = l2_norm(av), l2_norm(bv)
    if na <= 0 or nb <= 0:
        raise SteeringVectorError("emptyInput")
    return float(np.dot(av, bv)) / (na * nb)


#: Ablation-direction preflight: warn when an UNCENTERED ablation direction's
#: |cos| with the neutral residual mean exceeds this at any ablated layer.
#: Calibrated on the 2026-08-06 collapse study (both families, 4B tier):
#: Gemma-3-4b concept vectors at |cos| ≥ 0.45 (per-layer values up to 0.98)
#: collapse to single-token repetition under λ=1 ablation at ANY layer, while
#: Qwen3-0.6B vectors at |cos| ≤ 0.28 stay coherent under a full band.
#: Identical constant in Swift ``SteeringVectorMath.ablationMeanAlignmentWarnThreshold``.
ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD = 0.35


def mean_centered(direction: Vector, neutral_mean: Vector) -> list[float]:
    """The direction with its neutral-residual-mean component removed:
    ``v − (v·m̂)m̂``.

    The mean of the residual stream over a neutral corpus is a load-bearing
    "carrier" direction the model needs at every position (projecting it out
    at λ=1 collapses generation into single-token repetition), and extracted
    concept vectors routinely carry a large component along it — differences
    of means do NOT cancel it. Centering removes exactly that shared
    component, leaving the concept-specific part, and rescues coherence under
    full ablation. A zero or degenerate mean returns the direction unchanged
    (there is nothing to center against). Identical math in Swift
    ``SteeringVectorMath.meanCentered``.
    """
    m_norm = l2_norm(neutral_mean)
    if m_norm <= 0:
        return [float(x) for x in np.asarray(direction, dtype=_F32)]
    unit = (np.asarray(neutral_mean, dtype=_F32) / _F32(m_norm)).astype(_F32)
    return projecting_out(direction, [unit.tolist()])


def mean_alignment(direction: Vector, neutral_mean: Vector) -> float:
    """|cos(direction, neutral mean)| — the preflight diagnostic quantity.
    0 for a degenerate input (nothing to align with)."""
    try:
        return abs(cosine_similarity(direction, neutral_mean))
    except SteeringVectorError:
        return 0.0


def rescaled(v: Vector, to_norm: float) -> list[float]:
    """Scale ``v`` to a target norm (matched-norm random-vector control)."""
    n = l2_norm(v)
    if n <= 0:
        raise SteeringVectorError("emptyInput")
    factor = _F32(to_norm) / _F32(n)
    return (np.asarray(v, dtype=_F32) * factor).astype(_F32).tolist()


def centered_rows(rows: Rows) -> list[list[float]]:
    a = _arr(rows)
    center = a.mean(axis=0)
    return (a - center).astype(_F32).tolist()


def variance_trace(centered: Rows) -> float:
    a = _arr(centered)
    return float(np.square(a).sum(dtype=_F32))


def norm_unit_scale(alpha: float, residual_norm: float, vector_norm: float) -> float:
    """Raw injection scalar for a strength expressed in residual-norm units.

    The injector adds ``scale·v``; for that perturbation's L2 norm to equal
    ``alpha × residual_norm`` the vector's own norm must be folded out:
    ``scale = alpha·residual_norm/‖v‖``.
    """
    if vector_norm <= 0 or residual_norm <= 0:
        raise SteeringVectorError("degenerateData")
    return float(_F32(alpha) * _F32(residual_norm) / _F32(vector_norm))


# --- scalar probe ----------------------------------------------------------

@dataclass
class ScalarProbe:
    """RepE-style scalar estimator (parallel to Swift ``ScalarProbe``)."""

    direction: list[float]
    projection_center: float
    projection_scale: float
    orientation: float
    positive_mean: float
    negative_mean: float
    activation_center: list[float] | None = None

    def __post_init__(self) -> None:
        self.orientation = 1.0 if self.orientation >= 0 else -1.0

    def score(self, activation: Vector) -> float:
        a = np.asarray(activation, dtype=_F32)
        d = np.asarray(self.direction, dtype=_F32)
        if a.shape != d.shape:
            raise SteeringVectorError("dimensionMismatch")
        if self.activation_center is not None:
            c = np.asarray(self.activation_center, dtype=_F32)
            if c.shape != a.shape:
                raise SteeringVectorError("dimensionMismatch")
            a = a - c
        return float(self.orientation * (float(np.dot(a, d)) - self.projection_center)
                     / self.projection_scale)

    def classifies_positive(self, activation: Vector) -> bool:
        return self.score(activation) > 0

    def to_dict(self) -> dict:
        """JSON shape matching the Swift ``ScalarProbe`` Codable keys."""
        out = {
            "direction": list(self.direction),
            "projectionCenter": self.projection_center,
            "projectionScale": self.projection_scale,
            "orientation": self.orientation,
            "positiveMean": self.positive_mean,
            "negativeMean": self.negative_mean,
        }
        if self.activation_center is not None:
            out["activationCenter"] = list(self.activation_center)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "ScalarProbe":
        return cls(
            direction=list(d["direction"]),
            projection_center=float(d["projectionCenter"]),
            projection_scale=float(d["projectionScale"]),
            orientation=float(d["orientation"]),
            positive_mean=float(d["positiveMean"]),
            negative_mean=float(d["negativeMean"]),
            activation_center=(list(d["activationCenter"])
                               if d.get("activationCenter") is not None else None),
        )


def scalar_probe(direction: Vector, positive: Rows, negative: Rows,
                 activation_center: Vector | None = None) -> ScalarProbe:
    pos = _arr(positive)
    neg = _arr(negative)
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise SteeringVectorError("emptyInput")
    d = np.asarray(direction, dtype=_F32)
    center_arr = None
    if activation_center is not None:
        center_arr = np.asarray(activation_center, dtype=_F32)
        if center_arr.shape != d.shape:
            raise SteeringVectorError("dimensionMismatch")

    def project(rows: np.ndarray) -> np.ndarray:
        m = rows if center_arr is None else rows - center_arr
        return (m @ d).astype(_F32)

    pos_proj = project(pos)
    neg_proj = project(neg)
    pos_mean = float(pos_proj.mean())
    neg_mean = float(neg_proj.mean())
    orientation = 1.0 if pos_mean >= neg_mean else -1.0
    center = (pos_mean + neg_mean) / 2.0
    allp = np.concatenate([pos_proj, neg_proj])
    m = float(allp.mean())
    variance = float(np.square(allp - m).sum() / max(1, allp.size - 1))
    scale = max(float(np.sqrt(variance)), abs(pos_mean - neg_mean) / 2.0,
                float(np.finfo(_F32).eps))
    return ScalarProbe(direction=list(np.asarray(direction, dtype=_F32).tolist()),
                       activation_center=(None if activation_center is None
                                          else list(center_arr.tolist())),
                       projection_center=center, projection_scale=scale,
                       orientation=orientation, positive_mean=pos_mean,
                       negative_mean=neg_mean)


# --- direction finding (CAA / LAT) -----------------------------------------

def direction(positive: Rows, negative: Rows, method: ExtractionMethod) -> list[float]:
    """Concept direction by the chosen method.

    LAT follows RepE Appendix C.1: each pair difference is L2-normalized
    *before* PCA, differences enter in alternating ± orientation (the Swift
    note explains the determinism), the sign follows the score-directionality
    rule, and the unit PC is rescaled to the mean difference's norm so alpha
    semantics stay comparable across methods.
    """
    if not method.is_paired and not method.is_designated_reference:
        raise SteeringVectorError(
            f"{method.value} is not a paired method — grand-mean concepts are "
            "extracted from a multi-concept corpus via extract_grand_mean")
    mean_diff = mean_difference(positive, negative)
    if method in (ExtractionMethod.MEAN_DIFFERENCE,
                  ExtractionMethod.DESIGNATED_REFERENCE):
        # designatedReference IS the mean difference — the classes are just
        # stories vs a designated reference corpus rather than authored pairs.
        return mean_diff

    pos = _arr(positive)
    neg = _arr(negative)
    if pos.shape[0] != neg.shape[0]:
        raise SteeringVectorError(
            f"unpairedStimuli positive={pos.shape[0]} negative={neg.shape[0]}")

    raw_diffs = (pos - neg).astype(_F32)
    differences: list[np.ndarray] = []
    for d in raw_diffs:
        norm = l2_norm(d)
        if norm > 0:
            differences.append((d / _F32(norm)).astype(_F32))
    if len(differences) < 2:
        raise SteeringVectorError("degenerateData")

    # Alternate orientation before PCA so the shared concept direction is not
    # centered out of normalized labeled pairs (mirrors the Swift comment).
    oriented = [d if i % 2 == 0 else -d for i, d in enumerate(differences)]
    pc = first_principal_component([o.tolist() for o in oriented])

    scores = [dot(d.tolist(), pc) for d in differences]
    positive_scores = sum(1 for s in scores if s > 0)
    n = len(scores)
    if positive_scores * 2 == n:
        flip = dot(pc, mean_diff) < 0  # tie: class-mean criterion
    else:
        flip = positive_scores * 2 < n
    if flip:
        pc = [-x for x in pc]
    return rescaled(pc, l2_norm(mean_diff))


# --- PCA via deterministic Gram power-iteration ----------------------------

def first_principal_component(rows: Rows) -> list[float]:
    a = _arr(rows)
    if a.shape[0] < 2 or a.shape[1] == 0:
        raise SteeringVectorError("emptyInput")
    center = a.mean(axis=0)
    centered = (a - center).astype(_F32)
    return _first_component_of_centered(centered)


@dataclass
class PrincipalComponentsResult:
    components: list[list[float]]
    explained_variance: list[float]

    @property
    def total_explained_variance(self) -> float:
        return float(sum(self.explained_variance))


def _deflate(count: int | None, centered: np.ndarray, *,
             min_variance: float | None, maximum_count: int | None) -> PrincipalComponentsResult:
    total_variance = float(np.square(centered).sum(dtype=_F32))
    if total_variance <= 0:
        raise SteeringVectorError("degenerateData")
    components: list[list[float]] = []
    explained: list[float] = []
    if min_variance is None:
        cap = count or 0
    else:
        cap = maximum_count if maximum_count is not None else max(0, centered.shape[0] - 1)
    while len(components) < cap:
        if min_variance is not None and sum(explained) >= min(min_variance, 1.0):
            break
        try:
            component = _first_component_of_centered(centered)
        except SteeringVectorError:
            break
        comp = np.asarray(component, dtype=_F32)
        components.append(component)
        projections = centered @ comp  # [n]
        captured = float(np.square(projections).sum(dtype=_F32))
        centered = (centered - np.outer(projections, comp)).astype(_F32)
        explained.append(captured / total_variance)
    return PrincipalComponentsResult(components=components, explained_variance=explained)


def principal_components(rows: Rows, count: int) -> list[list[float]]:
    return principal_components_with_variance(rows, count).components


def principal_components_with_variance(rows: Rows, count: int) -> PrincipalComponentsResult:
    if count <= 0:
        return PrincipalComponentsResult(components=[], explained_variance=[])
    centered = _arr(centered_rows(rows))
    return _deflate(count, centered, min_variance=None, maximum_count=None)


def principal_components_by_variance(rows: Rows, minimum_explained_variance: float,
                                     maximum_count: int | None = None) -> PrincipalComponentsResult:
    if minimum_explained_variance <= 0:
        return PrincipalComponentsResult(components=[], explained_variance=[])
    centered = _arr(centered_rows(rows))
    return _deflate(None, centered, min_variance=minimum_explained_variance,
                    maximum_count=maximum_count)


def random_vector(dimension: int, norm: float, seed: int = 0) -> list[float]:
    """A matched-norm random direction for the coherence control (parallel to
    Swift ``randomVector``). Seeded for reproducibility; byte-identity with the
    Swift SplitMix64 RNG is not required — only that the norm matches."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dimension).astype(_F32)
    n = l2_norm(v.tolist())
    if n <= 0:
        raise SteeringVectorError("degenerateData")
    return (v * (_F32(norm) / _F32(n))).astype(_F32).tolist()


def principal_components_by_variance_svd(rows: Rows, minimum_explained_variance: float,
                                         maximum_count: int | None = None) -> PrincipalComponentsResult:
    """Top PCs via economy SVD — O(n·d²), the right choice when the number of
    rows ≫ hidden dim (e.g. a neutral token-position activation bank with tens
    of thousands of rows, where the n×n Gram power-iteration is intractable).

    Sign is arbitrary, which is fine for nuisance projection (``projecting_out``
    is sign-invariant). Used for neutral-PC bases, not for concept directions.
    """
    if minimum_explained_variance <= 0:
        return PrincipalComponentsResult(components=[], explained_variance=[])
    a = _arr(rows)
    if a.shape[0] < 2:
        return PrincipalComponentsResult(components=[], explained_variance=[])
    centered = (a - a.mean(axis=0)).astype(_F32)
    # economy SVD: rows of Vt are the principal directions (unit norm)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = (s.astype(np.float64) ** 2)
    total = float(variance.sum())
    if total <= 0:
        raise SteeringVectorError("degenerateData")
    explained = (variance / total)
    cap = maximum_count if maximum_count is not None else len(s)
    components: list[list[float]] = []
    ev: list[float] = []
    cumulative = 0.0
    for i in range(min(cap, len(s))):
        if cumulative >= min(minimum_explained_variance, 1.0):
            break
        components.append(vt[i].astype(_F32).tolist())
        ev.append(float(explained[i]))
        cumulative += float(explained[i])
    return PrincipalComponentsResult(components=components, explained_variance=ev)


def projecting_out(v: Vector, components: Rows) -> list[float]:
    """Remove the given (unit) components from ``v``: v − Σ (v·cᵢ)cᵢ."""
    result = np.asarray(v, dtype=_F32).copy()
    for component in components:
        c = np.asarray(component, dtype=_F32)
        projection = _F32(np.dot(result, c))
        result = (result - projection * c).astype(_F32)
    return result.tolist()


def _first_component_of_centered(centered: np.ndarray) -> list[float]:
    """First unit eigenvector via the Gram-matrix power-iteration.

    With n rows of dimension d (n ≪ d), power-iterate the n×n Gram matrix and
    map the eigenvector back through the data. Deterministic — two fixed starts
    (uniform, index-ramp), 200 iterations, 1e-7 convergence — exactly mirroring
    the Swift implementation so LAT/PCA reproduce.
    """
    centered = np.asarray(centered, dtype=_F32)
    n = centered.shape[0]
    dimension = centered.shape[1]
    if n < 2 or dimension == 0:
        raise SteeringVectorError("emptyInput")

    gram = (centered @ centered.T).astype(_F32)  # [n, n], symmetric
    trace = float(np.trace(gram))
    if trace <= 0:
        raise SteeringVectorError("degenerateData")

    uniform = np.full(n, 1.0 / np.sqrt(n), dtype=_F32)
    ramp = np.arange(1, n + 1, dtype=_F32)
    ramp = ramp / _F32(l2_norm(ramp.tolist()))
    starts = [uniform, ramp]

    weights: np.ndarray | None = None
    for start in starts:
        candidate = start.astype(_F32)
        diverged = False
        for _ in range(200):
            nxt = (gram @ candidate).astype(_F32)
            norm = _F32(l2_norm(nxt.tolist()))
            if norm <= 0:
                diverged = True  # start orthogonal to the dominant eigenvector
                break
            nxt = (nxt / norm).astype(_F32)
            delta = float(np.abs(nxt - candidate).max())
            candidate = nxt
            if delta < 1e-7:
                break
        if diverged:
            continue
        weights = candidate
        break
    if weights is None:
        raise SteeringVectorError("degenerateData")

    component = (weights @ centered).astype(_F32)  # [d]
    norm = _F32(l2_norm(component.tolist()))
    if norm <= 0:
        raise SteeringVectorError("degenerateData")
    return (component / norm).astype(_F32).tolist()

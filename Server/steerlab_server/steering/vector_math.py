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

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

Vector = Sequence[float]
Rows = Sequence[Sequence[float]]

_F32 = np.float32


class SteeringVectorError(Exception):
    pass


class ExtractionMethod(str, Enum):
    """Direction-finding method. ``meanDifference``/``pairedDifferencePCA``
    consume a paired positive/negative stimulus set; ``emotionGrandMean``
    consumes a multi-concept story corpus (concept mean − corpus grand mean)
    and is dispatched through ``extractor.extract_grand_mean``, never
    :func:`direction`. The value string matches the Swift
    ``VectorExtractionRecipe.Method`` raw value and existing sidecars."""

    MEAN_DIFFERENCE = "meanDifference"
    #: First principal component of the per-pair difference vectors,
    #: sign-aligned and norm-matched to the mean difference.
    #:
    #: **RepE-INSPIRED, not RepE**: the direction math of Zou et al.
    #: (arXiv:2310.01405) App. C.1 and none of the paper's pipeline — no task
    #: template, no template-mediated LAT token, no persisted fit parameters,
    #: no held-out sign/layer selection. :mod:`steerlab_server.steering.
    #: repe_reader` is the faithful one. The member used to be spelled ``LAT``,
    #: which read as "this IS RepE's LAT" (naming honesty ruling 2026-08-27).
    #:
    #: **The value stays ``"lat"``, permanently** — it is written into every
    #: sidecar, every frozen manifest concept block, and every recipe-identity
    #: hash this workspace has produced. Changing it would break decode of
    #: existing artifacts AND move identity hashes. The value is an
    #: artifact-compatibility constant; the member name and the label are where
    #: honesty is expressed. Swift twin:
    #: ``ExtractionMethod.pairedDifferencePCA``.
    PAIRED_DIFFERENCE_PCA = "lat"
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
    #: NOT a direction-finding method either: the reading direction of a
    #: FITTED RepE READER (``repe_reader.ReaderArtifact``), converted to a
    #: steering vector by ``repe_reader.derive_steering_vector`` with the
    #: probe's orientation folded into the bytes.
    #:
    #: Listed here (2026-08-27, audit finding 2) because without it a
    #: reader-derived vector could not be ATTACHED at all: ``attach_artifact``
    #: resolves the sidecar's ``extractionMethod`` to ask where the concept's
    #: held-out data lives, and an unknown method is refused — so the one
    #: artifact the faithful RepE pipeline produces was the one artifact a
    #: study could not cite. Its data questions have honest answers, and they
    #: are not a plain concept's: the stimuli are the READER's dataset
    #: (``prompts/readers/<concept>/pairs.jsonl``, whose SHA-256 is the
    #: stimulusSetHash), there is no ``prompts/concepts/<c>/`` pair set, and
    #: the held-out evidence is the reader artifact's own ``heldOutAccuracy``
    #: — not a ``validation.jsonl``. Hence ``has_source_concept is False``:
    #: every data-side branch must skip rather than invent. The reader itself
    #: is pinned separately as a ``readerRef``, and the derived vector's
    #: sidecar carries ``readerID``/``readerHash`` back to it.
    REPE_READER_LAT = "repeReaderLAT"

    @property
    def label(self) -> str:
        return {"meanDifference": "Mean difference",
                "lat": "Paired-difference PCA (RepE-inspired)",
                "emotionGrandMean": "Grand mean (multi-concept)",
                "designatedReference": "Designated reference (stories − reference stories)",
                "pinnedArtifact": "Pinned artifact (hash-pinned derived vectors)",
                "optvec": "Optimized injection vector (OptVec)",
                "gemmaScopeSAE": "Gemma Scope SAE feature (decoder row)",
                "repeReaderLAT": "RepE reader LAT (derived from a fitted reader)"}[self.value]

    @property
    def is_paired(self) -> bool:
        return self in (ExtractionMethod.MEAN_DIFFERENCE,
                        ExtractionMethod.PAIRED_DIFFERENCE_PCA)

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
    def is_repe_reader_lat(self) -> bool:
        """A direction derived from a fitted RepE reader (see the member)."""
        return self is ExtractionMethod.REPE_READER_LAT

    @property
    def source_concept_absence(self) -> tuple[str, str, str] | None:
        """For a method with NO source concept: ``(kind, evidence, hash
        referent)``.

        One place, so the three families that skip every data-side question
        say WHY in their own words instead of each call site carrying a
        two-way conditional that a third family silently falsified. ``None``
        for every method that does have a source concept. Swift twin:
        ``ExtractionMethod.sourceConceptAbsence``.
        """
        if self is ExtractionMethod.OPTVEC:
            return ("an OptVec vector",
                    "the OptVec eval run (eval.json)",
                    "the OptVec training run's pinned dataset splits")
        if self is ExtractionMethod.GEMMA_SCOPE_SAE:
            return ("an imported Gemma Scope SAE decoder row",
                    "the pinned SAE candidate roster's discovery snapshot and "
                    "qualification artifact",
                    "the published Gemma Scope dictionary the feature was "
                    "imported from")
        if self is ExtractionMethod.REPE_READER_LAT:
            return ("a direction derived from a fitted RepE reader",
                    "the reader artifact's own held-out accuracy (its "
                    "pairs.jsonl test split), pinned as the study's readerRef",
                    "the reader's dataset "
                    "(prompts/readers/<concept>/pairs.jsonl)")
        return None

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

        A third family joined them 2026-08-27: ``repeReaderLAT``, whose
        stimuli are the READER's dataset and whose held-out evidence is the
        reader artifact, not a concept's validation.jsonl.
        """
        return self.source_concept_absence is None

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
        """Sign of :meth:`score` — i.e. "is this activation on the positive side
        of the midpoint the probe was fitted with?".

        VALID ONLY WHERE THE ACTIVATION COMES FROM THE SAME DISTRIBUTION THE
        CENTER WAS FITTED ON (2026-08-28 audit, F4). ``projection_center`` is
        the midpoint of the two TRAIN class means, so this is a real label for
        supervised-content probes (reading probes, concept validation, RepE
        ``_pair_accuracy``, which scores both classes) and NOT a real label for
        a ``unsupervisedTemplatePair`` reader scoring new text, whose center is
        the midpoint of the T+ and T− renderings while inference renders T+
        only: every such activation sits systematically above the midpoint. See
        ``repe_reader.score_texts`` — template-pair scores support relative
        comparisons, never a presence threshold at zero.
        """
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

    ``pairedDifferencePCA`` takes RepE's PC1-of-paired-differences idea and
    adds two steps of our own. **Attribution, corrected 2026-08-27 (audit
    finding 9):** this docstring used to cite "RepE Appendix C.1" for the
    per-pair L2 normalization. It is not the paper's. The reference
    implementation (``repe/rep_readers.py``, ``PCARepReader``) mean-CENTERS the
    difference matrix and fits ``PCA(n_components=1)`` on it — it never
    normalizes a difference. Both departures are OURS, deliberately:

    1. per-pair L2 normalization before PCA, so high-norm pairs cannot
       dominate PC1 (without it PCA is pulled toward the mean difference,
       eroding the method comparison this family exists to make);
    2. rescaling the unit PC to ``‖mean_diff‖`` so alpha semantics stay
       comparable across methods.

    What IS the paper's: the alternating ± orientation (a deterministic stand-in
    for the reference dataset builder's per-pair ``random.shuffle`` followed by
    ``[::2] − [1::2]``) and the TRAIN-label sign rule (``get_signs``). The
    paper's TEXT additionally selects sign and layer on a held-out set; this
    family has no held-out split, so its sidecar stamps
    ``signConvention: "trainMajority"``. The held-out rule lives in
    :mod:`steerlab_server.steering.repe_reader`.
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
    #: One :class:`PowerIterationDiagnostic` per component, in the same order
    #: (2026-08-28 audit, F5). Empty on the SVD path, which has no iteration to
    #: diagnose. Additive: nothing about the components changed when this
    #: appeared, and callers that ignore it behave exactly as before.
    diagnostics: list["PowerIterationDiagnostic"] = field(default_factory=list)

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
    diagnostics: list[PowerIterationDiagnostic] = []
    if min_variance is None:
        # RANK CAP (2026-08-28 audit, F2). n centred rows span at most n−1
        # dimensions, so after n−1 deflations the residual is float32
        # round-off — and round-off has a tiny but POSITIVE Gram trace, which
        # is exactly what the power iteration's RELATIVE degenerate-start
        # floor (1e-6·trace) accepts. The count branch therefore used to
        # normalise rounding noise into unit "components" and hand them back
        # indistinguishable from real directions: 4 rows with count=6 returned
        # 6 components whose explained variances were [0.406, 0.310, 0.284,
        # 7.9e-15, 4.2e-15, 2.1e-15]. `extract()` and `extract_grand_mean`
        # then projected the concept vector's component along those three
        # arbitrary noise directions OUT of the science vector.
        #
        # Clamp rather than refuse: `neutral_pc_count` is a study-level knob
        # applied to whatever neutral corpus each concept has (as few as 4
        # texts is legal), so over-asking is a normal, honest declaration and
        # refusing it would strand existing manifests. The clamp is also what
        # the min_variance branch below and the bank path
        # (`extractor.components_by_layer`, `min(count, cap)`) already do; the
        # advisory says the request was trimmed so the trim is never silent.
        rank_cap = max(0, centered.shape[0] - 1)
        cap = min(count or 0, rank_cap)
        if (count or 0) > rank_cap:
            warnings.warn(
                f"requested {count} principal components from "
                f"{centered.shape[0]} rows, which span at most {rank_cap} "
                f"dimension(s) — returning {rank_cap}; components past the "
                f"data's rank are float round-off, not directions",
                UserWarning, stacklevel=3)
    else:
        cap = maximum_count if maximum_count is not None else max(0, centered.shape[0] - 1)
    while len(components) < cap:
        if min_variance is not None and sum(explained) >= min(min_variance, 1.0):
            break
        try:
            component, diagnostic = _first_component_of_centered_with_diagnostic(
                centered)
        except SteeringVectorError:
            break
        comp = np.asarray(component, dtype=_F32)
        components.append(component)
        diagnostics.append(diagnostic)
        projections = centered @ comp  # [n]
        captured = float(np.square(projections).sum(dtype=_F32))
        centered = (centered - np.outer(projections, comp)).astype(_F32)
        explained.append(captured / total_variance)
    return PrincipalComponentsResult(components=components,
                                     explained_variance=explained,
                                     diagnostics=diagnostics)


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


#: How much of the Gram matrix's trace the first power-iteration product must
#: reach for a start to count as carrying signal. Dimensionless: the trace IS
#: the spectrum's scale, so the ratio is comparable across data of any
#: magnitude. Identical constant in Swift
#: ``SteeringVectorMath.degenerateStartRelativeThreshold``.
DEGENERATE_START_RELATIVE_THRESHOLD = 1e-6

#: Relative Rayleigh-quotient residual ‖Gw − λw‖/λ above which PC1 is reported
#: as ill-determined (2026-08-28 audit, F5). Identical constant in Swift
#: ``SteeringVectorMath.powerIterationResidualWarnThreshold``.
#:
#: CALIBRATION, measured on engineered clouds whose top two sample eigenvalues
#: are separated by a controlled gap (the audit's own repro geometry):
#:
#: ===================  ==================  ==============================
#: sample eigengap      relative residual   |cos| of PC1 with the TRUE PC1
#: ===================  ==================  ==============================
#: isotropic (typical)  ≤ 6e-6              1.000000
#: 5.2 %                2.1e-6              1.000000
#: 4.2 %                1.3e-5              1.000000
#: 3.3 %                8.1e-5              0.999997
#: 2.4 %                4.5e-4              0.999827
#: 1.7 %                2.5e-3              0.988485
#: 1.4 %                6.2e-3              0.872206
#: 1.3 %                6.7e-3              0.665022
#: ===================  ==================  ==============================
#:
#: The audit's reproduced failure (eigenvalue ratio 0.9945, |cos| = 0.148
#: against the true SECOND eigenvector) sits at the bottom of that table; the
#: audit's calibration that "at a 5% eigengap the iteration IS converged" sits
#: at the top. 1e-4 is the order of magnitude between them: about fifty times
#: above the worst healthy residual, about twenty-five times below the mildest
#: genuinely-wrong one. It fires around a 3% gap, while PC1 is still accurate
#: to 1e-5 — early on purpose, because the warning's job is to say the SPECTRUM
#: is near-degenerate, not to wait until the answer is already wrong.
#:
#: Not a refusal: the result is deterministic and mirrored across engines, so a
#: near-tied spectrum is a fact about the DATA that the artifact should record,
#: not a malformed input to reject.
POWER_ITERATION_RESIDUAL_WARN_THRESHOLD = 1e-4

#: Iteration cap and convergence tolerance, named so the diagnostic can report
#: what it was measured against. Twins of the Swift literals.
POWER_ITERATION_MAX_ITERATIONS = 200
POWER_ITERATION_DELTA_TOLERANCE = 1e-7


@dataclass(frozen=True)
class PowerIterationDiagnostic:
    """Convergence health of one Gram power-iteration (2026-08-28 audit, F5).

    Purely additive: it is computed AFTER the component is fixed and never
    changes it, so every fixture that pins a component keeps passing.

    - ``relative_residual`` — ‖Gw − λw‖/λ with λ = wᵀGw, the Rayleigh quotient.
      Dimensionless, so it is comparable across data of any scale. This is the
      number to read: an unconverged iteration on a near-degenerate spectrum
      returns a WRONG unit vector deterministically and silently, and the
      residual is what distinguishes it from a converged one (the explained
      variance does not — a wrong direction in a near-tied 2-plane explains
      almost exactly as much).
    - ``iterations`` / ``converged`` — how many products the winning start used
      and whether the max-abs-delta tolerance was met before the cap. Reported
      for context, NOT thresholded: a healthy 5%-gap cloud routinely uses all
      200 iterations without meeting a 1e-7 delta in float32 and is still
      accurate to six figures, so "hit the cap" on its own is not a defect.
    - ``ill_conditioned`` — ``relative_residual`` above
      :data:`POWER_ITERATION_RESIDUAL_WARN_THRESHOLD`.
    """

    relative_residual: float
    iterations: int
    converged: bool
    max_iterations: int = POWER_ITERATION_MAX_ITERATIONS
    delta_tolerance: float = POWER_ITERATION_DELTA_TOLERANCE

    @property
    def ill_conditioned(self) -> bool:
        return self.relative_residual > POWER_ITERATION_RESIDUAL_WARN_THRESHOLD

    def to_dict(self) -> dict:
        """The artifact stamp shape (twin: Swift ``PowerIterationDiagnostic``)."""
        return {
            "converged": self.converged,
            "illConditioned": self.ill_conditioned,
            "iterations": self.iterations,
            "maxIterations": self.max_iterations,
            "relativeResidual": self.relative_residual,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PowerIterationDiagnostic":
        return cls(relative_residual=float(d["relativeResidual"]),
                   iterations=int(d["iterations"]),
                   converged=bool(d["converged"]),
                   max_iterations=int(d.get("maxIterations",
                                            POWER_ITERATION_MAX_ITERATIONS)))


def power_iteration_warning(diagnostic: PowerIterationDiagnostic) -> str:
    """The warning text, so both engines say the same thing (twin:
    ``SteeringVectorMath.powerIterationWarning``)."""
    return (
        f"PC1 power iteration left a relative Rayleigh residual of "
        f"{diagnostic.relative_residual:.3g} after {diagnostic.iterations} "
        f"iterations (warn above "
        f"{POWER_ITERATION_RESIDUAL_WARN_THRESHOLD:.3g}): "
        "the top two eigenvalues of this cloud are nearly tied, so PC1 is "
        "ill-determined and the returned direction may be an arbitrary vector "
        "in the near-degenerate plane. The result is deterministic and "
        "identical on both engines — it is the DATA that does not single out a "
        "first component. Read the stamped diagnostic before interpreting this "
        "direction.")


def first_principal_component_with_diagnostic(
        rows: Rows) -> tuple[list[float], PowerIterationDiagnostic]:
    """:func:`first_principal_component` plus its convergence health."""
    a = _arr(rows)
    if a.shape[0] < 2 or a.shape[1] == 0:
        raise SteeringVectorError("emptyInput")
    center = a.mean(axis=0)
    centered = (a - center).astype(_F32)
    return _first_component_of_centered_with_diagnostic(centered)


def _first_component_of_centered(centered: np.ndarray) -> list[float]:
    return _first_component_of_centered_with_diagnostic(centered)[0]


def _first_component_of_centered_with_diagnostic(
        centered: np.ndarray) -> tuple[list[float], PowerIterationDiagnostic]:
    """First unit eigenvector via the Gram-matrix power-iteration, plus the
    convergence diagnostic (2026-08-28 audit, F5).

    With n rows of dimension d (n ≪ d), power-iterate the n×n Gram matrix and
    map the eigenvector back through the data. Deterministic — four fixed starts
    (uniform, index-ramp, alternating ±, heaviest row), 200 iterations, 1e-7
    convergence — exactly mirroring the Swift implementation so LAT/PCA
    reproduce.

    The COMPONENT is computed exactly as before; the diagnostic is one extra
    Gram matvec afterwards and cannot move it.
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
    # The THIRD start is the alternating ± pattern itself. The paired-difference
    # constructions feed PCA rows in alternating orientation, so on clean data
    # the dominant eigenvector's weights ARE that pattern — and both earlier
    # starts can be orthogonal to it at once (a uniform start always is when the
    # signs balance; a three-row ramp is too, exactly). Before the
    # degenerate-start guard became real, such data still "worked" because the
    # guard's exact-zero test never fired and the iteration amplified float
    # noise into roughly the right answer. Making the guard honest requires
    # giving it a start that honestly has overlap. Swift twin: the third entry
    # of `starts` in `SteeringVectorMath.firstComponentOfCentered`.
    alternating = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)],
                           dtype=_F32) / _F32(np.sqrt(n))
    # The LAST-RESORT start: the heaviest row's own basis vector. ``gram @
    # e_j`` is column j, whose norm is at least the largest diagonal entry,
    # which is at least trace/n — comfortably above the relative floor for any
    # bank this engine builds. So a Gram matrix with any variance at all always
    # has SOME start the guard accepts, and the guard can only ever refuse a
    # genuinely degenerate matrix. Without it the honest guard stops deflation
    # early on a flat spectrum, where the fixed starts' overlap with the
    # surviving subspace shrinks each round.
    heaviest = np.zeros(n, dtype=_F32)
    heaviest[int(np.argmax(np.diag(gram)))] = _F32(1.0)
    starts = [uniform, ramp, alternating, heaviest]

    weights: np.ndarray | None = None
    used_iterations = 0
    converged = False
    for start in starts:
        candidate = start.astype(_F32)
        diverged = False
        used_iterations = 0
        converged = False
        for iteration in range(POWER_ITERATION_MAX_ITERATIONS):
            nxt = (gram @ candidate).astype(_F32)
            norm = _F32(l2_norm(nxt.tolist()))
            # Degenerate START detection, on the FIRST product only.
            #
            # The old guard was ``norm <= 0`` — exact zero, which in float32
            # essentially never fires: a start orthogonal to the dominant
            # eigenvector still has rounding-level overlap with it and with
            # every other, so ``gram @ start`` comes back as a vector of
            # denormal noise and the iteration then "converges" to whichever
            # direction that noise happened to point. The check has to be
            # RELATIVE to the spectrum's own scale, which is what the trace
            # measures. Only the first product is checked — later iterations
            # legitimately shrink under deflation, and a relative floor there
            # would abandon a converging run. Swift twin:
            # ``SteeringVectorMath.firstComponentOfCentered``.
            floor = (DEGENERATE_START_RELATIVE_THRESHOLD * trace
                     if iteration == 0 else 0.0)
            if norm <= floor:
                diverged = True  # start carries no signal about the spectrum
                break
            nxt = (nxt / norm).astype(_F32)
            delta = float(np.abs(nxt - candidate).max())
            candidate = nxt
            used_iterations = iteration + 1
            if delta < POWER_ITERATION_DELTA_TOLERANCE:
                converged = True
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

    # --- convergence diagnostic (audit F5), strictly after the component ------
    # One extra Gram matvec. λ = wᵀGw is the Rayleigh quotient and
    # ‖Gw − λw‖/λ is how far w is from being an eigenvector of G, measured
    # relative to the spectrum's own scale. A near-tied top pair leaves this
    # residual orders of magnitude above a healthy cloud's (see the constant's
    # calibration table) while every other stamp — the explained variance
    # especially — looks perfectly normal.
    product = (gram @ weights).astype(_F32)
    eigenvalue = float(np.dot(weights, product))
    if eigenvalue > 0:
        residual = float(np.linalg.norm(
            (product - _F32(eigenvalue) * weights).astype(_F32)))
        relative_residual = residual / eigenvalue
    else:
        relative_residual = float("inf")
    diagnostic = PowerIterationDiagnostic(
        relative_residual=relative_residual, iterations=used_iterations,
        converged=converged)
    if diagnostic.ill_conditioned:
        warnings.warn(power_iteration_warning(diagnostic), UserWarning,
                      stacklevel=3)
    return (component / norm).astype(_F32).tolist(), diagnostic

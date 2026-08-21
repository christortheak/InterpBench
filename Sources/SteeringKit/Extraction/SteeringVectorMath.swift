import Foundation

public enum SteeringVectorError: Error, Equatable, CustomStringConvertible {
    case emptyInput
    case dimensionMismatch(expected: Int, found: Int)
    case unpairedStimuli(positive: Int, negative: Int)
    case degenerateData
    case notPairedMethod(String)

    public var description: String {
        switch self {
        case .emptyInput: "empty input"
        case .dimensionMismatch(let expected, let found):
            "dimension mismatch: expected \(expected), found \(found)"
        case .unpairedStimuli(let positive, let negative):
            "unpaired stimuli: positive=\(positive) negative=\(negative)"
        case .degenerateData: "degenerate data"
        case .notPairedMethod(let method):
            "\(method) is not a paired method — grand-mean concepts are extracted "
                + "from a multi-concept corpus via extractGrandMean"
        }
    }
}

/// Direction-finding method (METHODS.md › Method options). `meanDifference`
/// and `lat` consume a paired positive/negative stimulus set;
/// `emotionGrandMean` consumes a multi-concept story corpus (concept mean −
/// corpus grand mean) and is dispatched through
/// `ConceptExtractor.extractGrandMean`, never `direction`. Raw values match
/// the Python server's `ExtractionMethod` and existing sidecars.
public enum ExtractionMethod: String, Codable, Sendable, CaseIterable {
    /// CAA direction: mean(positive) − mean(negative).
    case meanDifference
    /// RepE's LAT: first principal component of the per-pair difference
    /// vectors, sign-aligned and norm-matched to the mean difference so
    /// alpha semantics stay comparable. Requires paired stimuli.
    case lat
    /// Emotion-paper grand mean: concept mean − grand mean over every row
    /// of a multi-concept story corpus (no negative class).
    case emotionGrandMean
    /// METHODS amendment ii: mean(concept stories) − mean(a DESIGNATED
    /// reference corpus's stories), both pooled. Same arithmetic as
    /// meanDifference over unpaired classes; first-class so the reference
    /// is pinned recipe data and the pooled-reading policy has a native
    /// home instead of hand-derived class directories.
    case designatedReference
    /// NOT a direction-finding method: the direction already exists as a
    /// hash-pinned artifact and the "extraction" is a verified
    /// materialization of those bytes (post-hoc derived vectors which no
    /// re-derivable recipe can reproduce from stimuli). Never reaches the
    /// direction math; lifecycle branches that ask about DATA (stimulus
    /// location, validation semantics) must ask the artifact's SOURCE
    /// method instead — `ExperimentManifest.ConceptRef.effectiveMethod`.
    /// Cross-engine contract raw value (server `vector_math.ExtractionMethod
    /// .PINNED_ARTIFACT`).
    case pinnedArtifact
    /// NOT a direction-finding method either: an OPTIMIZED injection vector,
    /// found by backprop through the frozen model against a hashed
    /// target/anchor/capability dataset (server `experiment/optvec_train.py`
    /// — there is no MLX training path, by design). It never reaches the
    /// direction math and is never attachable as a recipe — an optvec
    /// vector enters studies only as a `pinnedArtifact` concept whose
    /// sidecar records this as its SOURCE method. Listed here so the
    /// lifecycle's DATA questions get an honest answer instead of an
    /// unknown-method fallback: there are no stimuli (the stimulusSetHash is
    /// the `optvec:<composite>` dataset hash), no source concept, and no
    /// held-out validation.jsonl — the evidence is the eval run's eval.json
    /// (OptVec plan §6).
    case optvec
    /// NOT a direction-finding method either: a Gemma Scope SAE DECODER ROW,
    /// written by `GemmaScopeReports.deriveFeatureArtifact` here and by the
    /// server's `experiment/gemma_scope.py`, and entering studies only as a
    /// `pinnedArtifact` concept whose sidecar records this as its SOURCE
    /// method. Listed for the same reason as `optvec`: the lifecycle's DATA
    /// questions need an honest answer instead of an unknown-method refusal
    /// (without this case a Gemma Scope import attached on the server and was
    /// refused on the Mac). A decoder row has no stimuli (its stimulusSetHash
    /// is the `gemmascope:<release>:<saeID>:<feature>` identity), no source
    /// concept, and no held-out validation.jsonl — a feature is a coordinate
    /// in an SAE's dictionary, not a contrast between two authored classes.
    /// Its evidence is the discovery snapshot + qualification artifact in the
    /// pinned SAE candidate roster. Cross-engine contract raw value (server
    /// `vector_math.ExtractionMethod.GEMMA_SCOPE_SAE`).
    case gemmaScopeSAE

    public var label: String {
        switch self {
        case .meanDifference: "Mean difference"
        case .lat: "LAT paired direction (RepE-inspired)"
        case .emotionGrandMean: "Grand mean (multi-concept)"
        case .designatedReference: "Designated reference (stories − reference stories)"
        case .pinnedArtifact: "Pinned artifact (hash-pinned derived vectors)"
        case .optvec: "Optimized injection vector (OptVec)"
        case .gemmaScopeSAE: "Gemma Scope SAE feature (decoder row)"
        }
    }

    /// Whether the method consumes a paired positive/negative stimulus set.
    public var isPaired: Bool {
        switch self {
        case .meanDifference, .lat: true
        case .emotionGrandMean, .designatedReference, .pinnedArtifact, .optvec,
            .gemmaScopeSAE:
            false
        }
    }

    /// Materialized from pinned bytes rather than derived from stimuli.
    public var isPinnedArtifact: Bool { self == .pinnedArtifact }

    /// Optimized against a hashed dataset, not read off stimuli. Every
    /// DATA-side lifecycle branch (stimulus location, validation semantics,
    /// pin surface) must ask this before assuming a concept has stimuli:
    /// an optvec direction has none, by construction.
    public var isOptvec: Bool { self == .optvec }

    /// An imported Gemma Scope SAE decoder row (see the case's note).
    public var isGemmaScopeSAE: Bool { self == .gemmaScopeSAE }

    /// False for directions that were never READ OFF a concept's stimuli: an
    /// optvec vector (optimized against hashed datasets) and a Gemma Scope
    /// SAE decoder row (a dictionary coordinate). Ask THIS — not `isOptvec` —
    /// at every DATA-side lifecycle branch: "where do this concept's stimuli
    /// live", "what does its held-out validation MEAN", "which position was
    /// it read at". Answering those for a direction with no source concept
    /// means inventing an obligation it can never meet.
    /// Server twin: `ExtractionMethod.has_source_concept`.
    public var hasSourceConcept: Bool { !(isOptvec || isGemmaScopeSAE) }

    /// A method the researcher can attach as a RECIPE (stimuli by hash,
    /// vector re-derived every run). `pinnedArtifact` attaches through
    /// `attachArtifact` (the bytes are the recipe) and `optvec` is only
    /// ever a pinned artifact's SOURCE method — neither belongs in a
    /// recipe-method picker or an `attach --method` vocabulary. Nor does
    /// `gemmaScopeSAE`, for the same reason: an imported SAE feature enters
    /// as a pinned artifact (server twin: the `_GEMMA_SCOPE_METHOD` refusal
    /// in `experiment_store.attach`).
    public var isRecipeMethod: Bool {
        switch self {
        case .meanDifference, .lat, .emotionGrandMean, .designatedReference:
            true
        case .pinnedArtifact, .optvec, .gemmaScopeSAE: false
        }
    }

    // Explicit lifecycle semantics (external review 2026-07-31, finding 1):
    // "not isPaired" used to MEAN "grand mean", until a third method made
    // that two-valued shortcut wrong at every unswept site. Every lifecycle
    // branch now asks the question it actually means.

    /// Extracts against the pinned multi-concept corpus population.
    public var isGrandMean: Bool { self == .emotionGrandMean }

    /// Stimuli (and validation.jsonl) live under prompts/emotions/.
    /// `pinnedArtifact` answers false — its DATA questions route through
    /// the source method (`ConceptRef.effectiveMethod`), never this case.
    public var usesStoryCorpus: Bool {
        self == .emotionGrandMean || self == .designatedReference
    }

    /// How validation scores this method's vectors. EXHAUSTIVE by
    /// construction (review round 3, finding 1): a new method fails to
    /// compile until it declares its semantics — the fail-open "record a
    /// note and continue" path cannot exist.
    public enum ValidationSemantics: Sendable {
        /// Against two class means (positive/negative or concept/reference).
        case contrastive
        /// Against the concept-vs-population midpoint.
        case population
    }

    public var validationSemantics: ValidationSemantics {
        switch self {
        case .meanDifference, .lat, .designatedReference: .contrastive
        case .emotionGrandMean: .population
        // Neither is ever validated UNDER ITS OWN NAME: a pinnedArtifact
        // concept validates with its SOURCE method's semantics (resolve
        // `ConceptRef.effectiveMethod` before asking), and an optvec
        // direction has nothing to validate at all — no stimuli, no
        // classes, no validation.jsonl; its evidence is the OptVec eval
        // run (plan §6), and the validate gate exempts it. `.population`
        // here is the least-wrong total answer for a caller that failed
        // to resolve first; validate paths refuse these methods before
        // this property can mislead them. `gemmaScopeSAE` is in the same
        // position as optvec: a decoder row has no classes and no held-out
        // set, and validate skips it (`owesHeldOutProbe`).
        case .pinnedArtifact, .optvec, .gemmaScopeSAE: .population
        }
    }

    /// Validation scores against TWO class means (positive/negative or
    /// concept/reference), not a corpus population.
    public var usesContrastiveValidation: Bool {
        isPaired || self == .designatedReference
    }
}

/// Pure CPU-side vector arithmetic for contrastive activation extraction.
/// Kept off the GPU deliberately: these run over small captured activations,
/// and pure functions are unit-testable against committed fixtures
/// (CLAUDE.md › Conventions).
public enum SteeringVectorMath {

    public struct PrincipalComponentsResult: Sendable, Equatable {
        public let components: [[Float]]
        /// Per-component share of the original centered variance.
        public let explainedVariance: [Float]
        public var totalExplainedVariance: Float {
            explainedVariance.reduce(0, +)
        }
    }

    public struct ScalarProbe: Codable, Sendable, Equatable {
        /// Unit or normed reading direction. Projection sign is handled by
        /// `orientation`, so the stored vector can match the steering vector
        /// artifact byte-for-byte when desired.
        public var direction: [Float]
        /// Optional activation-space center subtracted before projection.
        public var activationCenter: [Float]?
        /// Projection value that maps to zero after orientation.
        public var projectionCenter: Float
        /// Positive scale denominator for z-ish scores.
        public var projectionScale: Float
        /// 1 or -1. Scores above zero mean "more concept-positive".
        public var orientation: Float
        public var positiveMean: Float
        public var negativeMean: Float

        public init(
            direction: [Float],
            activationCenter: [Float]? = nil,
            projectionCenter: Float,
            projectionScale: Float,
            orientation: Float,
            positiveMean: Float,
            negativeMean: Float
        ) {
            self.direction = direction
            self.activationCenter = activationCenter
            self.projectionCenter = projectionCenter
            self.projectionScale = projectionScale
            self.orientation = orientation >= 0 ? 1 : -1
            self.positiveMean = positiveMean
            self.negativeMean = negativeMean
        }

        public func score(_ activation: [Float]) throws -> Float {
            guard activation.count == direction.count else {
                throw SteeringVectorError.dimensionMismatch(
                    expected: direction.count, found: activation.count)
            }
            let centered: [Float]
            if let activationCenter {
                guard activationCenter.count == activation.count else {
                    throw SteeringVectorError.dimensionMismatch(
                        expected: activation.count, found: activationCenter.count)
                }
                centered = zip(activation, activationCenter).map(-)
            } else {
                centered = activation
            }
            return orientation
                * (dot(centered, direction) - projectionCenter)
                / projectionScale
        }

        public func classifiesPositive(_ activation: [Float]) throws -> Bool {
            try score(activation) > 0
        }
    }

    /// Element-wise mean of equal-length rows.
    public static func mean(_ rows: [[Float]]) throws -> [Float] {
        guard let first = rows.first else { throw SteeringVectorError.emptyInput }
        var sum = [Float](repeating: 0, count: first.count)
        for row in rows {
            guard row.count == first.count else {
                throw SteeringVectorError.dimensionMismatch(
                    expected: first.count, found: row.count)
            }
            for i in row.indices { sum[i] += row[i] }
        }
        let n = Float(rows.count)
        return sum.map { $0 / n }
    }

    /// CAA direction: mean(positive activations) − mean(negative activations).
    public static func meanDifference(
        positive: [[Float]], negative: [[Float]]
    ) throws -> [Float] {
        let p = try mean(positive)
        let n = try mean(negative)
        guard p.count == n.count else {
            throw SteeringVectorError.dimensionMismatch(expected: p.count, found: n.count)
        }
        return zip(p, n).map(-)
    }

    /// Emotion-paper-style concept direction: mean(activations for this
    /// concept) − mean(activations over the whole multi-concept corpus).
    /// This is intentionally not exposed as `ExtractionMethod.meanDifference`
    /// because its data contract is multi-class, not positive/negative pairs.
    public static func grandMeanDifference(
        concept: [[Float]], population: [[Float]]
    ) throws -> [Float] {
        let conceptMean = try mean(concept)
        let grandMean = try mean(population)
        guard conceptMean.count == grandMean.count else {
            throw SteeringVectorError.dimensionMismatch(
                expected: conceptMean.count, found: grandMean.count)
        }
        return zip(conceptMean, grandMean).map(-)
    }

    public static func l2Norm(_ v: [Float]) -> Float {
        sqrt(v.reduce(0) { $0 + $1 * $1 })
    }

    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) throws -> Float {
        guard a.count == b.count else {
            throw SteeringVectorError.dimensionMismatch(expected: a.count, found: b.count)
        }
        let na = l2Norm(a)
        let nb = l2Norm(b)
        guard na > 0, nb > 0 else { throw SteeringVectorError.emptyInput }
        let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        return dot / (na * nb)
    }

    /// Scales `v` to the given norm — used for the matched-norm random-vector
    /// coherence control (CLAUDE.md › Experiment B specification).
    public static func rescaled(_ v: [Float], toNorm target: Float) throws -> [Float] {
        let n = l2Norm(v)
        guard n > 0 else { throw SteeringVectorError.emptyInput }
        let factor = target / n
        return v.map { $0 * factor }
    }

    public static func dot(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    public static func centeredRows(_ rows: [[Float]]) throws -> [[Float]] {
        let center = try mean(rows)
        return rows.map { row in zip(row, center).map(-) }
    }

    public static func varianceTrace(ofCenteredRows rows: [[Float]]) -> Float {
        rows.reduce(0) { partial, row in
            partial + row.reduce(0) { $0 + $1 * $1 }
        }
    }

    /// Calibrates a RepE-style scalar estimator over held-out/labeled
    /// activations. Projection is centered halfway between class means and
    /// scaled by pooled projection standard deviation; positive scores mean
    /// "more of the labeled concept" regardless of the arbitrary PCA sign.
    public static func scalarProbe(
        direction: [Float],
        positive: [[Float]],
        negative: [[Float]],
        activationCenter: [Float]? = nil
    ) throws -> ScalarProbe {
        guard !positive.isEmpty, !negative.isEmpty else {
            throw SteeringVectorError.emptyInput
        }
        if let activationCenter, activationCenter.count != direction.count {
            throw SteeringVectorError.dimensionMismatch(
                expected: direction.count, found: activationCenter.count)
        }

        func projection(_ row: [Float]) throws -> Float {
            guard row.count == direction.count else {
                throw SteeringVectorError.dimensionMismatch(
                    expected: direction.count, found: row.count)
            }
            if let activationCenter {
                return dot(zip(row, activationCenter).map(-), direction)
            }
            return dot(row, direction)
        }

        let pos = try positive.map(projection)
        let neg = try negative.map(projection)
        let posMean = pos.reduce(0, +) / Float(pos.count)
        let negMean = neg.reduce(0, +) / Float(neg.count)
        let orientation: Float = posMean >= negMean ? 1 : -1
        let center = (posMean + negMean) / 2
        let all = pos + neg
        let mean = all.reduce(0, +) / Float(all.count)
        let variance = all.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) }
            / Float(max(1, all.count - 1))
        let scale = max(sqrt(variance), abs(posMean - negMean) / 2, Float.ulpOfOne)
        return ScalarProbe(
            direction: direction, activationCenter: activationCenter,
            projectionCenter: center, projectionScale: scale,
            orientation: orientation, positiveMean: posMean,
            negativeMean: negMean)
    }

    /// Raw injection scalar for a strength expressed in residual-norm units.
    /// The injector adds `scale·v`; for that perturbation's L2 norm to equal
    /// `alpha × residualNorm` (the emotion paper's "fraction of the typical
    /// residual-stream norm" semantics) the vector's own norm must be folded
    /// out: scale = alpha·residualNorm/‖v‖. Without the division, alpha
    /// units silently depend on ‖v‖ — which varies by layer, method, and
    /// stimulus count — defeating cross-layer/cross-family comparability.
    public static func normUnitScale(
        alpha: Float, residualNorm: Float, vectorNorm: Float
    ) throws -> Float {
        guard vectorNorm > 0, residualNorm > 0 else {
            throw SteeringVectorError.degenerateData
        }
        return alpha * residualNorm / vectorNorm
    }

    /// Concept direction by the chosen method.
    ///
    /// LAT follows RepE Appendix C.1: each pair difference is L2-normalized
    /// BEFORE PCA — D(sᵢ, sᵢ₊₁) = normalize(H(sᵢ) − H(sᵢ₊₁)) — so high-norm
    /// pairs cannot dominate PC1 (without this, PCA is pulled toward the
    /// mean difference, eroding the LAT-vs-CAA method comparison), and the
    /// differences enter PCA in alternating ± orientation, reproducing the
    /// paper's unsupervised random pairing (see the inline note). The sign
    /// follows the paper's score-directionality rule: each (positive −
    /// negative) difference should project positively onto the PC; flip on
    /// a majority of negative scores (ties fall back to the class-mean
    /// criterion). This is stable exactly where dot(pc, meanDiff) ≈ 0 — the
    /// regime where LAT is informative. As a deliberate divergence, the
    /// unit PC is then scaled to the mean difference's norm so injection
    /// alphas mean roughly the same thing under either method.
    public static func direction(
        positive: [[Float]], negative: [[Float]], method: ExtractionMethod
    ) throws -> [Float] {
        guard method.isPaired || method == .designatedReference else {
            throw SteeringVectorError.notPairedMethod(method.rawValue)
        }
        let meanDiff = try meanDifference(positive: positive, negative: negative)
        switch method {
        case .emotionGrandMean, .pinnedArtifact, .optvec, .gemmaScopeSAE:
            // Unreachable past the guard above; listed so a new method must
            // decide its direction semantics here explicitly.
            throw SteeringVectorError.notPairedMethod(method.rawValue)
        case .meanDifference, .designatedReference:
            // designatedReference IS the mean difference — the classes are
            // stories vs a designated reference corpus, not authored pairs.
            return meanDiff
        case .lat:
            guard positive.count == negative.count else {
                throw SteeringVectorError.unpairedStimuli(
                    positive: positive.count, negative: negative.count)
            }
            // Identical pairs carry no direction and cannot be normalized.
            let differences = zip(positive, negative)
                .map { pair in zip(pair.0, pair.1).map(-) }
                .compactMap { d -> [Float]? in
                    let norm = l2Norm(d)
                    return norm > 0 ? d.map { $0 / norm } : nil
                }
            guard differences.count >= 2 else {
                throw SteeringVectorError.degenerateData
            }
            // RepE's PCA sees pair differences in arbitrary orientation (its
            // pairing is unsupervised, so differences point ±concept at
            // random). With labeled pairs, every (positive − negative)
            // difference points the same way — and once magnitudes are
            // normalized away, centering would subtract that shared
            // direction out of the data entirely. Restore the paper's ±
            // symmetry deterministically (no RNG: reproducibility) by
            // alternating orientation before PCA.
            let oriented = differences.enumerated().map { index, d in
                index.isMultiple(of: 2) ? d : d.map { -$0 }
            }
            var pc = try firstPrincipalComponent(of: oriented)
            let scores = differences.map { dot($0, pc) }
            let positiveScores = scores.count { $0 > 0 }
            let flip =
                positiveScores * 2 == scores.count
                ? dot(pc, meanDiff) < 0  // tie: class-mean criterion
                : positiveScores * 2 < scores.count
            if flip {
                pc = pc.map { -$0 }
            }
            return try rescaled(pc, toNorm: l2Norm(meanDiff))
        }
    }

    /// First principal component (unit norm) of the centered rows, via the
    /// Gram-matrix trick: with n rows of dimension d (n ≪ d), power-iterate
    /// the n×n Gram matrix and map the eigenvector back through the data.
    /// Deterministic — no random initialization.
    public static func firstPrincipalComponent(of rows: [[Float]]) throws -> [Float] {
        guard rows.count >= 2, let dimension = rows.first?.count, dimension > 0 else {
            throw SteeringVectorError.emptyInput
        }
        guard rows.allSatisfy({ $0.count == dimension }) else {
            throw SteeringVectorError.dimensionMismatch(
                expected: dimension, found: rows.first(where: { $0.count != dimension })!.count)
        }
        let center = try mean(rows)
        let centered = rows.map { row in zip(row, center).map(-) }
        return try firstComponentOfCentered(centered)
    }

    /// Top-k principal components (unit norm, mutually orthogonal) via
    /// deflation: extract PC1, remove its projection from the data, repeat.
    /// Used for nuisance removal — the emotion paper projects neutral-corpus
    /// PCs out of concept vectors before use.
    public static func principalComponents(
        of rows: [[Float]], count: Int
    ) throws -> [[Float]] {
        try principalComponentsWithVariance(of: rows, count: count).components
    }

    /// Top-k principal components plus the variance share each component
    /// explains. This lets neutral-corpus projection follow the emotion
    /// paper's "enough PCs to explain X% variance" rule instead of only a
    /// fixed top-k rule.
    public static func principalComponentsWithVariance(
        of rows: [[Float]], count: Int
    ) throws -> PrincipalComponentsResult {
        guard count > 0 else {
            return PrincipalComponentsResult(components: [], explainedVariance: [])
        }
        var centered = try centeredRows(rows)
        let totalVariance = varianceTrace(ofCenteredRows: centered)
        guard totalVariance > 0 else { throw SteeringVectorError.degenerateData }
        var components: [[Float]] = []
        var explained: [Float] = []
        for _ in 0 ..< count {
            guard let component = try? firstComponentOfCentered(centered) else { break }
            components.append(component)
            var captured: Float = 0
            for index in centered.indices {
                let projection = dot(centered[index], component)
                captured += projection * projection
                for d in centered[index].indices {
                    centered[index][d] -= projection * component[d]
                }
            }
            explained.append(captured / totalVariance)
        }
        return PrincipalComponentsResult(components: components, explainedVariance: explained)
    }

    public static func principalComponents(
        of rows: [[Float]], minimumExplainedVariance: Float, maximumCount: Int? = nil
    ) throws -> PrincipalComponentsResult {
        guard minimumExplainedVariance > 0 else {
            return PrincipalComponentsResult(components: [], explainedVariance: [])
        }
        var centered = try centeredRows(rows)
        let totalVariance = varianceTrace(ofCenteredRows: centered)
        guard totalVariance > 0 else { throw SteeringVectorError.degenerateData }
        var components: [[Float]] = []
        var explained: [Float] = []
        let cap = maximumCount ?? max(0, rows.count - 1)
        while components.count < cap,
            explained.reduce(0, +) < min(minimumExplainedVariance, 1)
        {
            guard let component = try? firstComponentOfCentered(centered) else { break }
            components.append(component)
            var captured: Float = 0
            for index in centered.indices {
                let projection = dot(centered[index], component)
                captured += projection * projection
                for d in centered[index].indices {
                    centered[index][d] -= projection * component[d]
                }
            }
            explained.append(captured / totalVariance)
        }
        return PrincipalComponentsResult(components: components, explainedVariance: explained)
    }

    /// Removes the given (unit) components from `v`: v − Σ (v·cᵢ)cᵢ.
    /// Ablation-direction preflight: warn when an UNCENTERED ablation
    /// direction's |cos| with the neutral residual mean exceeds this at any
    /// ablated layer. Calibrated on the 2026-08-06 collapse study (both
    /// families, 4B tier): Gemma-3-4b concept vectors at |cos| ≥ 0.45
    /// (per-layer values up to 0.98) collapse to single-token repetition
    /// under λ=1 ablation at ANY layer, while Qwen3-0.6B vectors at
    /// |cos| ≤ 0.28 stay coherent under a full band. Identical constant on
    /// the server (`vector_math.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD`).
    public static let ablationMeanAlignmentWarnThreshold: Float = 0.35

    /// The direction with its neutral-residual-mean component removed:
    /// `v − (v·m̂)m̂`.
    ///
    /// The mean of the residual stream over a neutral corpus is a
    /// load-bearing "carrier" direction the model needs at every position
    /// (projecting it out at λ=1 collapses generation into single-token
    /// repetition), and extracted concept vectors routinely carry a large
    /// component along it — differences of means do NOT cancel it. Centering
    /// removes exactly that shared component, leaving the concept-specific
    /// part, and rescues coherence under full ablation. A zero mean returns
    /// the direction unchanged (nothing to center against). Identical math
    /// on the server (`vector_math.mean_centered`).
    public static func meanCentered(_ direction: [Float], against neutralMean: [Float]) -> [Float] {
        let meanNorm = l2Norm(neutralMean)
        guard meanNorm > 0 else { return direction }
        let unit = neutralMean.map { $0 / meanNorm }
        return projectingOut(direction, components: [unit])
    }

    /// `|cos(direction, neutral mean)|` — the preflight diagnostic quantity.
    /// 0 for a degenerate input (nothing to align with).
    public static func meanAlignment(_ direction: [Float], with neutralMean: [Float]) -> Float {
        guard let cosine = try? cosineSimilarity(direction, neutralMean) else { return 0 }
        return abs(cosine)
    }

    public static func projectingOut(_ v: [Float], components: [[Float]]) -> [Float] {
        var result = v
        for component in components {
            let projection = dot(result, component)
            for d in result.indices {
                result[d] -= projection * component[d]
            }
        }
        return result
    }

    private static func firstComponentOfCentered(_ centered: [[Float]]) throws -> [Float] {
        guard centered.count >= 2, let dimension = centered.first?.count, dimension > 0
        else {
            throw SteeringVectorError.emptyInput
        }
        let n = centered.count

        var gram = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
        for i in 0 ..< n {
            for j in i ..< n {
                let value = dot(centered[i], centered[j])
                gram[i][j] = value
                gram[j][i] = value
            }
        }

        let trace = (0 ..< n).reduce(Float(0)) { $0 + gram[$1][$1] }
        guard trace > 0 else { throw SteeringVectorError.degenerateData }

        // Power iteration with deterministic starts (no RNG — extraction
        // must be reproducible). The uniform start can be exactly
        // orthogonal to the dominant eigenvector (e.g. the alternating ±
        // pattern LAT's orientation symmetry produces on clean data); the
        // index ramp breaks that symmetry.
        let starts: [[Float]] = [
            [Float](repeating: 1 / Float(n).squareRoot(), count: n),
            {
                let ramp = (0 ..< n).map { Float($0 + 1) }
                let norm = l2Norm(ramp)
                return ramp.map { $0 / norm }
            }(),
        ]
        var weights: [Float]?
        outer: for start in starts {
            var candidate = start
            for _ in 0 ..< 200 {
                var next = (0 ..< n).map { i in dot(gram[i], candidate) }
                let norm = l2Norm(next)
                guard norm > 0 else { continue outer }  // start ⊥ eigenvector
                next = next.map { $0 / norm }
                let delta = zip(next, candidate).map { abs($0 - $1) }.max() ?? 0
                candidate = next
                if delta < 1e-7 { break }
            }
            weights = candidate
            break
        }
        guard let weights else { throw SteeringVectorError.degenerateData }

        var component = [Float](repeating: 0, count: dimension)
        for (index, row) in centered.enumerated() {
            let weight = weights[index]
            for d in 0 ..< dimension { component[d] += weight * row[d] }
        }
        let norm = l2Norm(component)
        guard norm > 0 else { throw SteeringVectorError.degenerateData }
        return component.map { $0 / norm }
    }
}

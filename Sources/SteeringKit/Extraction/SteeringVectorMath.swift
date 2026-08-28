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
/// and `pairedDifferencePCA` consume a paired positive/negative stimulus set;
/// `emotionGrandMean` consumes a multi-concept story corpus (concept mean −
/// corpus grand mean) and is dispatched through
/// `ConceptExtractor.extractGrandMean`, never `direction`. Raw values match
/// the Python server's `ExtractionMethod` and existing sidecars.
public enum ExtractionMethod: String, Codable, Sendable, CaseIterable {
    /// CAA direction: mean(positive) − mean(negative).
    case meanDifference
    /// First principal component of the per-pair difference vectors,
    /// sign-aligned and norm-matched to the mean difference so alpha
    /// semantics stay comparable. Requires paired stimuli.
    ///
    /// **RepE-INSPIRED, not RepE.** It borrows the paper's direction math
    /// (Zou et al., arXiv:2310.01405 App. C.1) and none of its pipeline: no
    /// task template, no template-mediated LAT token, no persisted fit
    /// parameters, no held-out sign/layer selection. The faithful pipeline is
    /// `RepEReader` (`repeReaderLAT`). Naming honesty ruling 2026-08-27 —
    /// the symbol used to be `lat`, which read as "this IS RepE's LAT".
    ///
    /// **The raw value stays `"lat"`, permanently.** It is written into every
    /// steering-vector sidecar, every frozen manifest concept block, and every
    /// recipe-identity hash this workspace has ever produced; changing it
    /// would break decode of existing artifacts AND move identity hashes,
    /// which is how a frozen experiment loses its promotions. The raw value
    /// is an ARTIFACT-COMPATIBILITY constant; the symbol and the label are
    /// where honesty is expressed.
    case pairedDifferencePCA = "lat"
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
    /// NOT a direction-finding method either: the reading direction of a
    /// FITTED RepE READER (`RepEReader.Artifact`), converted to a steering
    /// vector by `RepEReader.deriveSteeringArtifact` with the probe's
    /// orientation folded into the bytes.
    ///
    /// Listed here (2026-08-27, audit finding 2) because without it a
    /// reader-derived vector could not be ATTACHED at all: `attachArtifact`
    /// resolves the sidecar's `extractionMethod` to ask where the concept's
    /// held-out data lives, and an unknown method is refused — so the one
    /// artifact the faithful RepE pipeline produces was the one artifact a
    /// study could not cite. Its data questions have honest answers, and they
    /// are not a plain concept's: the stimuli are the READER's dataset
    /// (`prompts/readers/<concept>/pairs.jsonl`, whose SHA-256 is the
    /// stimulusSetHash), there is no `prompts/concepts/<c>/` pair set, and the
    /// held-out evidence is the reader artifact's own `heldOutAccuracy` — not
    /// a `validation.jsonl`. Hence `hasSourceConcept == false`: every
    /// data-side branch must skip rather than invent. The reader itself is
    /// pinned separately as a `readerRef`, and the derived vector's sidecar
    /// carries `readerID`/`readerHash` back to it.
    case repeReaderLAT

    public var label: String {
        switch self {
        case .meanDifference: "Mean difference"
        case .pairedDifferencePCA: "Paired-difference PCA (RepE-inspired)"
        case .emotionGrandMean: "Grand mean (multi-concept)"
        case .designatedReference: "Designated reference (stories − reference stories)"
        case .pinnedArtifact: "Pinned artifact (hash-pinned derived vectors)"
        case .optvec: "Optimized injection vector (OptVec)"
        case .gemmaScopeSAE: "Gemma Scope SAE feature (decoder row)"
        case .repeReaderLAT: "RepE reader LAT (derived from a fitted reader)"
        }
    }

    /// Whether the method consumes a paired positive/negative stimulus set.
    public var isPaired: Bool {
        switch self {
        case .meanDifference, .pairedDifferencePCA: true
        case .emotionGrandMean, .designatedReference, .pinnedArtifact, .optvec,
            .gemmaScopeSAE, .repeReaderLAT:
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

    /// A direction derived from a fitted RepE reader (see the case's note).
    public var isRepeReaderLAT: Bool { self == .repeReaderLAT }

    /// For a method with NO source concept: what it is, where its evidence
    /// lives, and what its `stimulusSetHash` refers to. One place, so the
    /// three families that skip every data-side question say WHY in their own
    /// words instead of each call site carrying a two-way conditional that a
    /// third family silently falsified. nil for every method that does have a
    /// source concept. Server twin: `ExtractionMethod.source_concept_absence`.
    public var sourceConceptAbsence:
        (kind: String, evidence: String, hashReferent: String)?
    {
        switch self {
        case .optvec:
            return (
                "an OptVec vector",
                "the OptVec eval run (eval.json)",
                "the OptVec training run's pinned dataset splits")
        case .gemmaScopeSAE:
            return (
                "an imported Gemma Scope SAE decoder row",
                "the pinned SAE candidate roster's discovery snapshot and "
                    + "qualification artifact",
                "the published Gemma Scope dictionary the feature was imported from")
        case .repeReaderLAT:
            return (
                "a direction derived from a fitted RepE reader",
                "the reader artifact's own held-out accuracy (its pairs.jsonl "
                    + "test split), pinned as the study's readerRef",
                "the reader's dataset (prompts/readers/<concept>/pairs.jsonl)")
        case .meanDifference, .pairedDifferencePCA, .emotionGrandMean,
            .designatedReference, .pinnedArtifact:
            return nil
        }
    }

    /// False for directions that were never READ OFF a concept's stimuli: an
    /// optvec vector (optimized against hashed datasets) and a Gemma Scope
    /// SAE decoder row (a dictionary coordinate). Ask THIS — not `isOptvec` —
    /// at every DATA-side lifecycle branch: "where do this concept's stimuli
    /// live", "what does its held-out validation MEAN", "which position was
    /// it read at". Answering those for a direction with no source concept
    /// means inventing an obligation it can never meet.
    /// Server twin: `ExtractionMethod.has_source_concept`.
    public var hasSourceConcept: Bool { sourceConceptAbsence == nil }

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
        case .meanDifference, .pairedDifferencePCA, .emotionGrandMean, .designatedReference:
            true
        case .pinnedArtifact, .optvec, .gemmaScopeSAE, .repeReaderLAT: false
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
        case .meanDifference, .pairedDifferencePCA, .designatedReference: .contrastive
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
        // `repeReaderLAT` joins them: a reader-derived direction validates
        // through the READER's held-out split (recorded on the reader
        // artifact), never through a concept's validation.jsonl, and
        // `hasSourceConcept == false` routes every validate path around it.
        case .pinnedArtifact, .optvec, .gemmaScopeSAE, .repeReaderLAT: .population
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
        /// One `PowerIterationDiagnostic` per component, in the same order
        /// (2026-08-28 audit, F5). Additive: nothing about the components
        /// changed when this appeared, and callers that ignore it behave
        /// exactly as before.
        public let diagnostics: [PowerIterationDiagnostic]

        public init(
            components: [[Float]], explainedVariance: [Float],
            diagnostics: [PowerIterationDiagnostic] = []
        ) {
            self.components = components
            self.explainedVariance = explainedVariance
            self.diagnostics = diagnostics
        }

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

        /// Sign of `score` — i.e. "is this activation on the positive side of
        /// the midpoint the probe was fitted with?".
        ///
        /// VALID ONLY WHERE THE ACTIVATION COMES FROM THE SAME DISTRIBUTION THE
        /// CENTER WAS FITTED ON (2026-08-28 audit, F4). `projectionCenter` is
        /// the midpoint of the two TRAIN class means, so this is a real label
        /// for supervised-content probes (reading probes, concept validation,
        /// `RepEReader.pairAccuracy`, which scores both classes) and NOT a real
        /// label for an `unsupervisedTemplatePair` reader scoring new text,
        /// whose center is the midpoint of the T+ and T− renderings while
        /// inference renders T+ only: every such activation sits systematically
        /// above the midpoint. See `RepEReader.scoreTexts` — template-pair
        /// scores support relative comparisons, never a presence threshold at
        /// zero.
        public func classifiesPositive(_ activation: [Float]) throws -> Bool {
            try score(activation) > 0
        }
    }

    /// Element-wise mean of equal-length rows.
    ///
    /// **Accumulated in `Double`, returned as `[Float]`** (2026-08-28 audit,
    /// convention note 16). The rows are float32 — the deliberate cross-engine
    /// storage dtype, unchanged — but a SEQUENTIAL float32 sum accrues error
    /// on the order of N·ε, while numpy's `ndarray.mean` sums pairwise at
    /// ~log₂(N)·ε. Over a few hundred stimulus rows that difference was the
    /// ceiling on cross-engine agreement for a re-derived grand mean
    /// (~1e-4..1e-3 relative in unlucky cases), which is a numerics artifact
    /// masquerading as a recipe divergence. Accumulating in `Double` and
    /// casting ONCE at the end costs nothing here (these run over small
    /// captured activations on the CPU) and moves this engine's value TOWARD
    /// the exact mean — and therefore toward numpy's. `ConceptExtractor
    /// .neutralMeanPerLayer` has always accumulated this way; this brings the
    /// mean / meanDifference / grandMeanDifference paths, and the PCA
    /// centring that also calls through here, into line with it.
    public static func mean(_ rows: [[Float]]) throws -> [Float] {
        guard let first = rows.first else { throw SteeringVectorError.emptyInput }
        var sum = [Double](repeating: 0, count: first.count)
        for row in rows {
            guard row.count == first.count else {
                throw SteeringVectorError.dimensionMismatch(
                    expected: first.count, found: row.count)
            }
            for i in row.indices { sum[i] += Double(row[i]) }
        }
        let n = Double(rows.count)
        return sum.map { Float($0 / n) }
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
    /// `pairedDifferencePCA` takes RepE's PC1-of-paired-differences idea and
    /// adds two steps of our own. **Attribution, corrected 2026-08-27 (audit
    /// finding 9):** an earlier comment here cited "RepE Appendix C.1" for the
    /// per-pair L2 normalization. It is not the paper's. The reference
    /// implementation (`repe/rep_readers.py`, `PCARepReader`) mean-CENTERS the
    /// difference matrix and fits `PCA(n_components=1)` on it — it never
    /// normalizes a difference. Both departures below are OURS, deliberately:
    ///
    /// 1. **Per-pair L2 normalization before PCA** — D(sᵢ) = normalize(H(pᵢ) −
    ///    H(nᵢ)) — so high-norm pairs cannot dominate PC1. Without it PCA is
    ///    pulled toward the mean difference, which erodes the very
    ///    method comparison this family exists to make. OUR addition.
    /// 2. **Norm-matching the unit PC to ‖meanDiff‖** so injection α means
    ///    roughly the same thing under either method. OUR addition.
    ///
    /// What IS the paper's: differences enter PCA in alternating ±
    /// orientation, which reproduces deterministically the random per-pair
    /// orientation the reference implementation's dataset builder produces
    /// (`random.shuffle(d)` per pair, then `[::2] − [1::2]`); and the sign is
    /// fixed from the TRAIN labels — each (positive − negative) difference
    /// should project positively onto the PC, so we flip on a majority of
    /// negative scores (ties fall back to the class-mean criterion), which is
    /// `get_signs`' train-label agreement. The paper's TEXT additionally
    /// selects sign and layer on a held-out set; this family has no held-out
    /// split, so it cannot, and its sidecar stamps
    /// `signConvention: "trainMajority"` to say so out loud. The
    /// held-out rule lives in `RepEReader` (`repeReaderLAT`), which does.
    /// This sign rule is stable exactly where dot(pc, meanDiff) ≈ 0 — the
    /// regime where a paired-difference PC is informative at all.
    public static func direction(
        positive: [[Float]], negative: [[Float]], method: ExtractionMethod
    ) throws -> [Float] {
        guard method.isPaired || method == .designatedReference else {
            throw SteeringVectorError.notPairedMethod(method.rawValue)
        }
        let meanDiff = try meanDifference(positive: positive, negative: negative)
        switch method {
        case .emotionGrandMean, .pinnedArtifact, .optvec, .gemmaScopeSAE,
            .repeReaderLAT:
            // Unreachable past the guard above; listed so a new method must
            // decide its direction semantics here explicitly.
            throw SteeringVectorError.notPairedMethod(method.rawValue)
        case .meanDifference, .designatedReference:
            // designatedReference IS the mean difference — the classes are
            // stories vs a designated reference corpus, not authored pairs.
            return meanDiff
        case .pairedDifferencePCA:
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

    /// Safety multiple applied to the per-pass float32 rounding energy when
    /// deciding that a deflation residual is round-off rather than data
    /// (2026-08-28 audit, F2 second pass). Identical constant on the server
    /// (`vector_math.DEFLATION_RESIDUAL_TRACE_FLOOR_EPS_FACTOR`).
    ///
    /// DERIVATION. One deflation pass rewrites every entry of the residual
    /// matrix (`centered − p·cᵀ`), so each entry acquires a rounding of
    /// relative size float32 eps (1.1920929e-07). Squared and summed over the
    /// matrix that is a residual TRACE of order eps² times the original total
    /// variance, per pass, and the passes' errors accumulate. Bounding that
    /// accumulation by the problem's own dimensions — at most `rows − 1`
    /// passes, each touching `columns` entries per row, so `(rows + columns)`
    /// is a generous stand-in for the accumulation length — gives a floor of
    /// `factor · (rows + columns) · eps²` on the residual's share of the total.
    ///
    /// CALIBRATION, measured over 120 random clouds (3–64 rows, 2–2048
    /// columns, constructed rank 1...min(rows−1, columns), row scales spanning
    /// 1e-6...1e6), recording the residual share both at the first pass PAST
    /// the constructed rank (round-off) and at every pass still inside it
    /// (data), each expressed in units of `(rows + columns)·eps²`:
    ///
    /// | residual share, in units  | median  | tail (p99/p1) | extreme (max/min) |
    /// | ------------------------- | ------- | ------------- | ----------------- |
    /// | round-off (past the rank) | 0.0012  | 0.0075 (p99)  | 0.0093 (max)      |
    /// | data (inside the rank)    | 1.5e+10 | 1.8e+07 (p1)  | 2.2e+03 (min)     |
    ///
    /// The two populations are separated by five orders of magnitude, so the
    /// floor only has to land between them. 8 sits about 860× above the
    /// largest round-off residue ever measured and about 270× below the
    /// smallest genuine one; the single tightest genuine case seen in any
    /// sweep (37 rows, 271 columns, constructed rank 36 — the last component
    /// of a saturated-rank cloud, explaining 1.8e-10 of the variance) still
    /// clears it by 5×. The factor is deliberately at the tight end of that
    /// window: cutting a component whose variance share is already at the
    /// 1e-10 level costs nothing, while keeping one costs a normalized noise
    /// direction that extraction then projects out of the science vector.
    public static let deflationResidualTraceFloorEpsFactor: Float = 8

    /// The residual-trace share below which further deflation would return
    /// normalized round-off (server twin: `vector_math._residual_trace_floor`).
    static func residualTraceFloor(rows: Int, columns: Int) -> Float {
        deflationResidualTraceFloorEpsFactor * Float(rows + columns)
            * Float.ulpOfOne * Float.ulpOfOne
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
        // RANK CAP (2026-08-28 audit, F2; server twin `vector_math._deflate`).
        // n centred rows span at most n−1 dimensions, and rows of d columns
        // span at most d — both are real bounds and the smaller one binds, so
        // a 6×2 corpus has rank ≤ 2 no matter how many rows it grows. Past
        // that the residual is float32 round-off — and round-off has a tiny
        // but POSITIVE Gram trace, which is exactly what the power iteration's
        // RELATIVE degenerate-start floor accepts. Without the cap this loop
        // normalises rounding noise into unit "components" that are
        // indistinguishable from real directions, and the neutral-PC
        // projection then removes the concept vector's component along them.
        // Clamping (not refusing) matches the variance branch below, the
        // neutral-bank path, and the fact that `neutralPCCount` is a
        // study-level knob applied to whatever neutral corpus each concept
        // has — over-asking is an honest declaration, not an error.
        var diagnostics: [PowerIterationDiagnostic] = []
        let columns = centered.first?.count ?? 0
        let rankCap = max(0, min(centered.count - 1, columns))
        let residualFloor = residualTraceFloor(rows: centered.count, columns: columns)
        let cap = min(count, rankCap)
        for _ in 0 ..< cap {
            // EFFECTIVE-RANK STOP (2026-08-28 audit, F2 second pass; server
            // twin `vector_math._deflate`). The rank cap above is theoretical —
            // it says how many directions the SHAPE could hold. Rank-deficient
            // data runs out sooner: mathematically rank-one, non-axis-aligned
            // rows with count = 3 used to return three unit components with
            // shares [1.0, 1.8e-15, 3.3e-17], the last two being normalised
            // float32 residue handed back as if they were directions — the very
            // failure the rank cap was written to prevent, arriving through the
            // other door. Once the residual's share of the ORIGINAL total
            // variance is at the round-off floor there is nothing left to
            // decompose, so stop, whichever branch asked. The server also emits
            // a "the trim is never silent" advisory here; this engine has no
            // `warnings` module, so the shorter component list — and the
            // diagnostic per component beside it — is the record.
            if varianceTrace(ofCenteredRows: centered) / totalVariance <= residualFloor {
                break
            }
            guard let fitted = try? firstComponentOfCenteredWithDiagnostic(centered)
            else { break }
            let component = fitted.component
            components.append(component)
            diagnostics.append(fitted.diagnostic)
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
        return PrincipalComponentsResult(
            components: components, explainedVariance: explained,
            diagnostics: diagnostics)
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
        var diagnostics: [PowerIterationDiagnostic] = []
        let columns = centered.first?.count ?? 0
        // Same two caps as the count branch above: the dimensional bound
        // min(rows − 1, columns), and — inside the loop — the effective-rank
        // stop. The variance branch needs the stop for a second reason: a
        // target the data cannot reach (1.0 against a rank-one cloud whose
        // first share is 0.9999999) used to keep deflating past the rank in
        // pursuit of the last 1e-7, returning a normalised-round-off component
        // to close a gap that is itself float32 round-off.
        let rankCap = max(0, min(centered.count - 1, columns))
        let residualFloor = residualTraceFloor(rows: centered.count, columns: columns)
        let cap = min(maximumCount ?? rankCap, rankCap)
        while components.count < cap,
            explained.reduce(0, +) < min(minimumExplainedVariance, 1),
            varianceTrace(ofCenteredRows: centered) / totalVariance > residualFloor
        {
            guard let fitted = try? firstComponentOfCenteredWithDiagnostic(centered)
            else { break }
            let component = fitted.component
            components.append(component)
            diagnostics.append(fitted.diagnostic)
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
        return PrincipalComponentsResult(
            components: components, explainedVariance: explained,
            diagnostics: diagnostics)
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

    /// How much of the Gram matrix's trace the first power-iteration product
    /// must reach for a start to count as carrying signal. Dimensionless: the
    /// trace IS the spectrum's scale, so the ratio is comparable across data
    /// of any magnitude. Identical constant on the server
    /// (`vector_math.DEGENERATE_START_RELATIVE_THRESHOLD`).
    public static let degenerateStartRelativeThreshold: Float = 1e-6

    /// Relative Rayleigh-quotient residual ‖Gw − λw‖/λ above which PC1 is
    /// reported as ill-determined (2026-08-28 audit, F5). Identical constant on
    /// the server (`vector_math.POWER_ITERATION_RESIDUAL_WARN_THRESHOLD`).
    ///
    /// CALIBRATION, measured on engineered clouds whose top two sample
    /// eigenvalues are separated by a controlled gap (the audit's own repro
    /// geometry):
    ///
    /// | sample eigengap | relative residual | \|cos\| of PC1 with the TRUE PC1 |
    /// |---|---|---|
    /// | isotropic (typical) | ≤ 6e-6 | 1.000000 |
    /// | 5.2 % | 2.1e-6 | 1.000000 |
    /// | 4.2 % | 1.3e-5 | 1.000000 |
    /// | 3.3 % | 8.1e-5 | 0.999997 |
    /// | 2.4 % | 4.5e-4 | 0.999827 |
    /// | 1.7 % | 2.5e-3 | 0.988485 |
    /// | 1.4 % | 6.2e-3 | 0.872206 |
    /// | 1.3 % | 6.7e-3 | 0.665022 |
    ///
    /// The audit's reproduced failure (eigenvalue ratio 0.9945, |cos| = 0.148
    /// against the true SECOND eigenvector) sits at the bottom of that table;
    /// the audit's calibration that "at a 5% eigengap the iteration IS
    /// converged" sits at the top. 1e-4 is the order of magnitude between them:
    /// about fifty times above the worst healthy residual, about twenty-five
    /// times below the mildest genuinely-wrong one. It fires around a 3% gap,
    /// while PC1 is still accurate to 1e-5 — early on purpose, because the
    /// warning's job is to say the SPECTRUM is near-degenerate, not to wait
    /// until the answer is already wrong.
    ///
    /// Not a refusal: the result is deterministic and mirrored across engines,
    /// so a near-tied spectrum is a fact about the DATA that the artifact
    /// should record, not a malformed input to reject.
    public static let powerIterationResidualWarnThreshold: Float = 1e-4

    /// Iteration cap and convergence tolerance, named so the diagnostic can
    /// report what it was measured against. Twins of the server literals.
    public static let powerIterationMaxIterations = 200
    public static let powerIterationDeltaTolerance: Float = 1e-7

    /// Convergence health of one Gram power-iteration (2026-08-28 audit, F5).
    ///
    /// Purely additive: it is computed AFTER the component is fixed and never
    /// changes it, so every parity fixture that pins a component keeps passing.
    ///
    /// - `relativeResidual` — ‖Gw − λw‖/λ with λ = wᵀGw, the Rayleigh quotient.
    ///   Dimensionless, so it is comparable across data of any scale. This is
    ///   the number to read: an unconverged iteration on a near-degenerate
    ///   spectrum returns a WRONG unit vector deterministically and silently,
    ///   and the residual is what distinguishes it from a converged one (the
    ///   explained variance does not — a wrong direction in a near-tied 2-plane
    ///   explains almost exactly as much).
    /// - `iterations` / `converged` — how many products the winning start used
    ///   and whether the max-abs-delta tolerance was met before the cap.
    ///   Reported for context, NOT thresholded: a healthy 5%-gap cloud
    ///   routinely uses all 200 iterations without meeting a 1e-7 delta in
    ///   float32 and is still accurate to six figures, so "hit the cap" on its
    ///   own is not a defect.
    /// - `illConditioned` — `relativeResidual` above
    ///   `powerIterationResidualWarnThreshold`.
    ///
    /// Server twin: `vector_math.PowerIterationDiagnostic`.
    public struct PowerIterationDiagnostic: Sendable, Equatable, Codable {
        public let relativeResidual: Float
        public let iterations: Int
        public let converged: Bool
        public let maxIterations: Int

        public init(
            relativeResidual: Float, iterations: Int, converged: Bool,
            maxIterations: Int = SteeringVectorMath.powerIterationMaxIterations
        ) {
            self.relativeResidual = relativeResidual
            self.iterations = iterations
            self.converged = converged
            self.maxIterations = maxIterations
        }

        public var illConditioned: Bool {
            relativeResidual > SteeringVectorMath.powerIterationResidualWarnThreshold
        }

        enum CodingKeys: String, CodingKey {
            case relativeResidual, iterations, converged, maxIterations
            case illConditioned
        }

        /// The stamp carries `illConditioned` as a derived convenience so a
        /// reader of the JSON never has to know the threshold; decoding ignores
        /// it and recomputes from the residual (server twin: `to_dict` writes
        /// it, `from_dict` does not read it).
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(converged, forKey: .converged)
            try container.encode(illConditioned, forKey: .illConditioned)
            try container.encode(iterations, forKey: .iterations)
            try container.encode(maxIterations, forKey: .maxIterations)
            try container.encode(relativeResidual, forKey: .relativeResidual)
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            relativeResidual = try container.decode(Float.self, forKey: .relativeResidual)
            iterations = try container.decode(Int.self, forKey: .iterations)
            converged = try container.decode(Bool.self, forKey: .converged)
            maxIterations = try container.decodeIfPresent(Int.self, forKey: .maxIterations)
                ?? SteeringVectorMath.powerIterationMaxIterations
        }
    }

    /// The warning text, so both engines say the same thing (server twin:
    /// `vector_math.power_iteration_warning`).
    public static func powerIterationWarning(_ diagnostic: PowerIterationDiagnostic)
        -> String
    {
        "PC1 power iteration left a relative Rayleigh residual of "
            + "\(formatResidual(diagnostic.relativeResidual)) after "
            + "\(diagnostic.iterations) iterations (warn above "
            + "\(formatResidual(powerIterationResidualWarnThreshold))): the top two "
            + "eigenvalues of this cloud are nearly tied, so PC1 is ill-determined "
            + "and the returned direction may be an arbitrary vector in the "
            + "near-degenerate plane. The result is deterministic and identical on "
            + "both engines — it is the DATA that does not single out a first "
            + "component. Read the stamped diagnostic before interpreting this "
            + "direction."
    }

    /// `%.3g` — the same C formatting Python's `%.3g` / `{:.3g}` produces,
    /// including its two-digit exponent (`1.23e-07`), so the twin message is
    /// character-identical on both engines. The Float is promoted to Double by
    /// the varargs call, exactly as the server's float32 value widens before
    /// formatting.
    static func formatResidual(_ value: Float) -> String {
        String(format: "%.3g", value)
    }

    /// Where a `firstComponentOfCentered` warning goes. The Swift engine has no
    /// `warnings` module, so an ill-conditioned spectrum is reported through
    /// this hook (the app and the CLI install their own) and, always, through
    /// the DIAGNOSTIC the caller stamps — the warning is the courtesy, the
    /// stamp is the record.
    nonisolated(unsafe) public static var powerIterationWarningSink:
        (@Sendable (String) -> Void)?

    /// `firstPrincipalComponent(of:)` plus its convergence health.
    public static func firstPrincipalComponentWithDiagnostic(
        of rows: [[Float]]
    ) throws -> (component: [Float], diagnostic: PowerIterationDiagnostic) {
        guard rows.count >= 2, let dimension = rows.first?.count, dimension > 0 else {
            throw SteeringVectorError.emptyInput
        }
        guard rows.allSatisfy({ $0.count == dimension }) else {
            throw SteeringVectorError.dimensionMismatch(
                expected: dimension, found: rows.first(where: { $0.count != dimension })!.count)
        }
        let center = try mean(rows)
        let centered = rows.map { row in zip(row, center).map(-) }
        return try firstComponentOfCenteredWithDiagnostic(centered)
    }

    private static func firstComponentOfCentered(_ centered: [[Float]]) throws -> [Float] {
        try firstComponentOfCenteredWithDiagnostic(centered).component
    }

    private static func firstComponentOfCenteredWithDiagnostic(
        _ centered: [[Float]]
    ) throws -> (component: [Float], diagnostic: PowerIterationDiagnostic) {
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
        // pattern the paired-difference orientation symmetry produces on
        // clean data); the index ramp breaks that symmetry.
        // The THIRD start is the alternating ± pattern itself. The
        // paired-difference constructions feed PCA rows in alternating
        // orientation, so on clean data the dominant eigenvector's weights
        // ARE that pattern — and both earlier starts can be orthogonal to it
        // at once (a uniform start always is when the signs balance; a
        // three-row ramp is too, exactly). Before the degenerate-start guard
        // became real, such data still "worked" because the guard's
        // exact-zero test never fired and the iteration amplified float
        // noise into roughly the right answer. Making the guard honest
        // requires giving it a start that honestly has overlap.
        let starts: [[Float]] = [
            [Float](repeating: 1 / Float(n).squareRoot(), count: n),
            {
                let ramp = (0 ..< n).map { Float($0 + 1) }
                let norm = l2Norm(ramp)
                return ramp.map { $0 / norm }
            }(),
            {
                let scale = 1 / Float(n).squareRoot()
                return (0 ..< n).map { $0.isMultiple(of: 2) ? scale : -scale }
            }(),
            // The LAST-RESORT start: the heaviest row's own basis vector.
            // `gram · e_j` is column j, whose norm is at least the largest
            // diagonal entry, which is at least trace/n — comfortably above
            // the relative floor for any bank this engine builds (the token
            // bank caps at 2048 rows). So a Gram matrix with any variance at
            // all always has SOME start the guard accepts, and the guard can
            // only ever refuse a genuinely degenerate matrix. Without it the
            // honest guard stops deflation early on a flat spectrum, where the
            // fixed starts' overlap with the surviving subspace shrinks each
            // round — reporting fewer components than the data has.
            {
                var heaviest = 0
                for i in 1 ..< n where gram[i][i] > gram[heaviest][heaviest] {
                    heaviest = i
                }
                var basis = [Float](repeating: 0, count: n)
                basis[heaviest] = 1
                return basis
            }(),
        ]
        var weights: [Float]?
        var usedIterations = 0
        var converged = false
        outer: for start in starts {
            var candidate = start
            usedIterations = 0
            converged = false
            for iteration in 0 ..< powerIterationMaxIterations {
                var next = (0 ..< n).map { i in dot(gram[i], candidate) }
                let norm = l2Norm(next)
                // Degenerate START detection, on the FIRST product only.
                //
                // The old guard was `norm > 0` — exact zero, which in float32
                // essentially never fires: a start that is orthogonal to the
                // dominant eigenvector still has rounding-level overlap with
                // it and with every other, so `gram·start` comes back as a
                // vector of denormal noise and the iteration then "converges"
                // to whichever direction that noise happened to point. The
                // check has to be RELATIVE to the spectrum's own scale, which
                // is what the trace measures: below this fraction of it, the
                // start carries no signal about the data and the next start
                // must be tried instead. Only the first product is checked —
                // later iterations legitimately shrink under deflation, and a
                // relative floor there would abandon a converging run.
                let floor =
                    iteration == 0 ? degenerateStartRelativeThreshold * trace : 0
                guard norm > floor else { continue outer }  // start ⊥ spectrum
                next = next.map { $0 / norm }
                let delta = zip(next, candidate).map { abs($0 - $1) }.max() ?? 0
                candidate = next
                usedIterations = iteration + 1
                if delta < powerIterationDeltaTolerance {
                    converged = true
                    break
                }
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

        // --- convergence diagnostic (audit F5), strictly after the component --
        // One extra Gram matvec. λ = wᵀGw is the Rayleigh quotient and
        // ‖Gw − λw‖/λ is how far w is from being an eigenvector of G, measured
        // relative to the spectrum's own scale. A near-tied top pair leaves this
        // residual orders of magnitude above a healthy cloud's (see the
        // constant's calibration table) while every other stamp — the explained
        // variance especially — looks perfectly normal.
        let product = (0 ..< n).map { i in dot(gram[i], weights) }
        let eigenvalue = dot(weights, product)
        let relativeResidual: Float
        if eigenvalue > 0 {
            let residual = l2Norm(zip(product, weights).map { $0 - eigenvalue * $1 })
            relativeResidual = residual / eigenvalue
        } else {
            relativeResidual = .infinity
        }
        let diagnostic = PowerIterationDiagnostic(
            relativeResidual: relativeResidual, iterations: usedIterations,
            converged: converged)
        if diagnostic.illConditioned {
            powerIterationWarningSink?(powerIterationWarning(diagnostic))
        }
        return (component.map { $0 / norm }, diagnostic)
    }
}

import CryptoKit
import Foundation

/// Method-level recipe for making a vector or reading direction. This is
/// broader than `ExtractionMethod`, which only covers paired contrastive
/// direction finding inside the existing CAA/LAT extractor.
public struct VectorExtractionRecipe: Codable, Sendable, Equatable {
    public enum Method: String, Codable, Sendable, CaseIterable {
        /// Paired contrastive activation addition:
        /// mean(positive activations) - mean(negative activations).
        case caaMeanDifference
        /// RepE/LAT: PCA over normalized paired differences, plus optional
        /// scalar probe calibration from held-out labeled activations.
        case repeLAT
        /// Faithful RepE reader (template-rendered LAT + PCA training
        /// normalization + held-out scalar accuracy). The reader itself is a
        /// measurement artifact (`RepEReader.Artifact`); this method value
        /// appears on *derived* steering vectors converted from a reader.
        case repeReaderLAT
        /// Emotion-paper-style vectors: concept mean minus grand mean across
        /// a multi-concept story corpus.
        case emotionGrandMean
        /// J-lens token direction: `J_l^T (g . u_t)` for one exact vocabulary
        /// token, from an imported Jacobian lens. DERIVED, not extracted —
        /// there is no stimulus set, and the identity is the token id plus the
        /// source lens, so this method can never map onto a manifest
        /// extraction recipe (see `OptimizationComposer.mapMethod`).
        case jlensTokenDirection

        public var label: String {
            switch self {
            case .caaMeanDifference: "CAA mean difference"
            case .repeLAT: "LAT paired direction (RepE-inspired)"
            case .repeReaderLAT: "RepE reader LAT"
            case .emotionGrandMean: "Emotion grand mean"
            case .jlensTokenDirection: "J-lens token direction"
            }
        }
    }

    public enum DatasetRole: String, Codable, Sendable, CaseIterable {
        case positive
        case negative
        case pairedContrastive
        case multiConceptStories
        case neutralCorpus
        case probeCalibration
        case probeValidation
    }

    public struct DatasetRef: Codable, Sendable, Equatable {
        public var role: DatasetRole
        public var path: String
        /// Optional SHA-256 of the referenced file. Draft recipes may omit
        /// this; frozen experiments should pin it.
        public var sha256: String?
        public var notes: String?

        public init(
            role: DatasetRole, path: String, sha256: String? = nil, notes: String? = nil
        ) {
            self.role = role
            self.path = path
            self.sha256 = sha256
            self.notes = notes
        }
    }

    public enum NeutralPCSelectionKind: String, Codable, Sendable {
        case none
        case fixedCount
        case explainedVariance
    }

    public struct NeutralPCSelection: Codable, Sendable, Equatable {
        /// Hard ceiling on components per layer when none is declared.
        ///
        /// A variance TARGET alone is unbounded: the explained-variance
        /// deflation loop will happily run to `rows − 1` components per layer
        /// on a flat spectrum, which on a thousands-of-rows token bank is both
        /// a compute cliff and a memory one. 64 nuisance directions out of a
        /// 2000–5000-dimensional residual stream is already a generous
        /// projection; anything beyond it is removing signal, not confounds.
        public static let defaultMaximumComponentCount = 64

        public var kind: NeutralPCSelectionKind
        public var count: Int?
        public var minimumExplainedVariance: Double?
        /// Hard ceiling on components per layer. nil resolves to
        /// `defaultMaximumComponentCount`. Encoded only when set, so existing
        /// recipe hashes are unaffected.
        public var maximumCount: Int?

        /// The ceiling actually applied — always a positive number.
        public var resolvedMaximumCount: Int {
            guard let maximumCount, maximumCount > 0 else {
                return Self.defaultMaximumComponentCount
            }
            return maximumCount
        }

        public static var none: NeutralPCSelection {
            NeutralPCSelection(kind: .none)
        }

        public static func fixedCount(_ count: Int) -> NeutralPCSelection {
            NeutralPCSelection(kind: .fixedCount, count: count)
        }

        public static func explainedVariance(
            _ fraction: Double, maximumCount: Int? = nil
        ) -> NeutralPCSelection {
            NeutralPCSelection(
                kind: .explainedVariance,
                minimumExplainedVariance: fraction,
                maximumCount: maximumCount)
        }

        public init(
            kind: NeutralPCSelectionKind,
            count: Int? = nil,
            minimumExplainedVariance: Double? = nil,
            maximumCount: Int? = nil
        ) {
            self.kind = kind
            self.count = count
            self.minimumExplainedVariance = minimumExplainedVariance
            self.maximumCount = maximumCount
        }
    }

    public var name: String
    public var method: Method
    public var targetConcept: String
    public var datasets: [DatasetRef]
    public var readingPosition: ReadingPosition
    public var neutralPCSelection: NeutralPCSelection
    public var promptMode: String?
    public var systemPrompt: String?
    public var notes: String?

    public init(
        name: String,
        method: Method,
        targetConcept: String,
        datasets: [DatasetRef],
        readingPosition: ReadingPosition = .lastToken,
        neutralPCSelection: NeutralPCSelection = .none,
        promptMode: String? = nil,
        systemPrompt: String? = nil,
        notes: String? = nil
    ) {
        self.name = name
        self.method = method
        self.targetConcept = targetConcept
        self.datasets = datasets
        self.readingPosition = readingPosition
        self.neutralPCSelection = neutralPCSelection
        self.promptMode = promptMode
        self.systemPrompt = systemPrompt
        self.notes = notes
    }

    public func canonicalHash() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = (try? encoder.encode(self)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// Persisted diagnostic reader. A reading probe can share a direction with a
/// steering vector, but it has extra calibration state and should be treated
/// as a scoring instrument, not automatically as an injection artifact.
public struct ReadingProbeArtifact: Codable, Sendable, Equatable {
    public var modelID: String
    public var revision: String?
    public var concept: String
    public var layer: Int
    public var recipeName: String
    public var recipeHash: String?
    public var calibrationHash: String?
    public var validationHash: String?
    public var probe: SteeringVectorMath.ScalarProbe
    public var createdAt: String
    public var notes: String?

    public init(
        modelID: String,
        revision: String? = nil,
        concept: String,
        layer: Int,
        recipeName: String,
        recipeHash: String? = nil,
        calibrationHash: String? = nil,
        validationHash: String? = nil,
        probe: SteeringVectorMath.ScalarProbe,
        createdAt: Date = Date(),
        notes: String? = nil
    ) {
        self.modelID = modelID
        self.revision = revision
        self.concept = concept
        self.layer = layer
        self.recipeName = recipeName
        self.recipeHash = recipeHash
        self.calibrationHash = calibrationHash
        self.validationHash = validationHash
        self.probe = probe
        self.createdAt = ISO8601DateFormatter().string(from: createdAt)
        self.notes = notes
    }

    public func score(_ activation: [Float]) throws -> Float {
        try probe.score(activation)
    }
}

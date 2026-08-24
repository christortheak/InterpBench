import Foundation
import MLX

/// Provenance sidecar persisted next to every vector file — a vector without
/// its model id, stimulus hash, and norms is not a usable experimental
/// artifact (CLAUDE.md › Steering method).
public struct SteeringVectorSidecar: Codable, Sendable {
    public var schemaVersion: Int?
    public var modelID: String
    public var revision: String?
    public var concept: String
    public var stimulusSetHash: String
    public var layerCount: Int
    public var hiddenSize: Int
    public var normsPerLayer: [Float]
    public var extractionDate: String
    /// Direction-finding method ("meanDifference" | "lat"). Optional for
    /// artifacts predating method options.
    public var extractionMethod: String?
    /// Reading position description ("last token" | "mean from token k").
    public var readingPosition: String?
    /// Confound removal applied at extraction ("top-k neutral PCs"), if any.
    public var confoundProjection: String?
    /// Typical residual-stream L2 norm per layer — the denominator for
    /// norm-unit steering strengths.
    public var residualNormPerLayer: [Float]?
    /// Where the norms were measured: "neutral-corpus" (with its hash —
    /// the emotion paper's fixed-dataset convention, comparable across
    /// concepts) or "extraction-stimuli" (legacy/no corpus; α units are
    /// concept-dependent). Nil on artifacts predating the distinction
    /// (those are stimulus-based).
    public var residualNormSource: String?
    /// WHICH AVERAGING RULE produced `residualNormPerLayer` — the denominator
    /// convention (`ResidualNormConvention.current`,
    /// `"wholeCorpusMean-v1"`). `residualNormSource` says which corpus was
    /// measured; this says how its positions were averaged, and the two
    /// together are what make an α in norm units mean one dose.
    ///
    /// Pinned cross-engine contract: same JSON key and same value string on
    /// the server's `vector_store.SteeringVectorSidecar`. Stamped only by a
    /// FRESH measurement (extraction, or `vectors backfill-norms`) and
    /// propagated verbatim when an artifact's norms are copied. **Absent is
    /// LEGACY and is never retro-filled** — a pre-stamp artifact's rule
    /// genuinely depends on which engine wrote it and whether its corpus was
    /// downsampled, so guessing would be worse than the honest gap. Legacy
    /// artifacts read exactly as they always have; `backfill-norms` is the
    /// researcher's opt-in path to a stamped denominator.
    public var residualNormConvention: String?
    /// WHICH RENDERING the denominator corpus was tokenized under — the third
    /// member of the `residualNorm*` provenance family (source = which corpus,
    /// convention = how its positions were averaged, rendering = how its texts
    /// reached the model). α in norm units only means one dose if the
    /// denominator was measured on the same distribution the vector was read
    /// from, so the denominator always follows the extraction's rendering and
    /// the artifact says which one that was. Values: "raw" | "chatTemplate".
    /// **Absent is LEGACY RAW** — every pre-2026-08-24 artifact was raw, and
    /// no artifact is retro-stamped.
    public var residualNormRendering: String?
    /// HOW the stimulus strings reached the model during extraction — the
    /// declared rendering block. Pinned cross-engine contract: same JSON key
    /// and inner key names on the server's `vector_store`. Additive and
    /// **absent = legacy raw**, never retro-filled: the same discipline
    /// `residualNormConvention` follows.
    public var extractionRendering: ExtractionRendering?
    /// The requested reading position AND what it RESOLVED to, per sequence
    /// shape. `readingPosition` says what was ASKED for; this says where that
    /// landed, so a reader never has to re-derive a template's internals to
    /// know what was read. Stamped only when the label does not already imply
    /// the index (a template-aware role, an explicit offset, or any non-raw
    /// rendering) — so legacy artifacts keep byte-identical sidecars.
    public var readingPositionResolution: ReadingPositionResolutionReport?
    /// Optional method-family recipe. This lets future artifacts distinguish
    /// a CAA contrastive concept from RepE LAT or emotion-grand-mean vectors
    /// even when older fields are still present for compatibility.
    public var recipeMethod: String?
    public var recipeHash: String?
    public var recipeName: String?
    /// Explicit projection mode for new artifacts. Values are descriptive
    /// strings such as "none", "legacy-pooled top-3 neutral PCs", or
    /// "token-bank explained-variance 0.5".
    public var neutralProjection: String?
    public var neutralCorpusHash: String?
    public var readingMinimumTokenCount: Int?
    public var sourceStimulusCount: Int?
    public var includedStimulusCount: Int?
    public var excludedShortStimulusCount: Int?
    public var neutralSourceStimulusCount: Int?
    public var neutralIncludedStimulusCount: Int?
    public var neutralExcludedShortStimulusCount: Int?
    public var comparisonConcepts: [String]?
    public var selectedTopics: [String]?
    public var selectedSplits: [String]?
    // Derived-artifact provenance (e.g. a vector converted from a RepE reader:
    // source "repe-reader-lat", the reader file id + SHA-256, and the honest
    // control-mode label "reading-vector activation addition"). JSON keys
    // match the server's `vector_store.SteeringVectorSidecar`.
    public var source: String?
    public var readerID: String?
    public var readerHash: String?
    public var controlMode: String?
    /// Which engine extracted this vector — a pinned cross-engine contract
    /// (same JSON key on the server's `vector_store.SteeringVectorSidecar`):
    /// `RepEReader.substrate` ("swift-mlx") here, "python-hf-transformers"
    /// on the Python server. Absent = legacy/unknown artifact predating the
    /// stamp. Steering pickers use it to keep each substrate's vectors on
    /// its own engine — activations do not transfer between engines, so an
    /// MLX vector must never steer a server generation (and vice versa).
    public var substrate: String?
    /// Set when this artifact was produced by the residual-norm BACKFILL job:
    /// same vector bytes as the source artifact, with `residualNormPerLayer`
    /// measured after the fact on a pinned neutral corpus. The JSON shape is
    /// a pinned cross-engine contract (server twin writes the same keys):
    /// `{"sourceArtifact": "...", "sourceVectorsHash": "...", "date": "..."}`.
    public var normBackfill: NormBackfillProvenance?
    /// Gemma Scope import convention (pinned cross-engine contract, WS7.2;
    /// same JSON keys on the server's `vector_store.SteeringVectorSidecar`).
    /// Both engines import an SAE feature the way this app always has —
    /// `"analyzed-vector-norm-match"` (`GemmaScopeReportCatalog
    /// .importConvention`): the decoder row is rescaled AT IMPORT to the
    /// analyzed concept vector's L2 norm at the report layer. Absent on a
    /// Gemma-Scope-sourced sidecar = pre-convention import (unknown/legacy
    /// scaling) — surfaced by `artifacts audit`; re-import before evidence
    /// use. All three are additive: old sidecars decode to nil, nil stays
    /// off the wire.
    public var gemmascopeConvention: String?
    /// The decoder row's PRE-transform L2 norm (with `gemmascopeTargetNorm`,
    /// the transform is fully recoverable).
    public var rawDecoderNorm: Float?
    /// The norm the row was rescaled to — the analyzed vector's L2 norm at
    /// the report layer.
    public var gemmascopeTargetNorm: Float?
    /// Grand-mean artifacts only: the FULL comparison population the grand
    /// mean was computed over — every corpus member's concept name mapped to
    /// the SHA-256 of its stories.jsonl as actually read at extraction.
    /// Pinned cross-engine contract (same JSON key on the server's
    /// `vector_store.SteeringVectorSidecar`); additive — absent on paired
    /// methods and on artifacts predating the stamp.
    public var grandMeanPopulation: [String: String]?
    /// Designated-reference artifacts only: the reference stories corpus
    /// actually subtracted — {"name": conceptName, "hash": storiesSha256}.
    /// Same cross-engine additive contract as `grandMeanPopulation` (server
    /// twin: `vector_store.SteeringVectorSidecar.designatedReference`).
    public var designatedReference: [String: String]?
    /// Set when the artifact's `.safetensors` additionally carries the
    /// per-layer NEUTRAL RESIDUAL MEAN (keys `neutral_mean_layer_<i>`)
    /// measured on the extraction's neutral corpus (`neutralCorpusHash`) at
    /// the reading position. The mean enables neutral-mean CENTERING of
    /// ablation directions and the ablation mean-alignment preflight —
    /// extracted concept vectors routinely share a large component with this
    /// mean, and ablating an uncentered direction at λ=1 collapses
    /// generation (2026-08-06 collapse study). Value is the estimator
    /// description (currently always "neutral-corpus"). Pinned cross-engine
    /// additive contract (same JSON key + tensor keys on the server's
    /// `vector_store.SteeringVectorSidecar`); absent on artifacts predating
    /// the stamp or extracted without a neutral corpus.
    public var neutralMeanSource: String?
    /// Canonical full-recipe identity hash (see `RecipeIdentity` /
    /// `recipe_identity.py` for the pinned canonical form). Stamped by the
    /// experiment extraction writers on both engines so promotion can match
    /// artifacts by complete recipe, not a six-field subset. Additive —
    /// absent on legacy artifacts, which must then prove every recipe field
    /// from their other sidecar fields or be refused for promotion.
    public var recipeIdentityHash: String?
    /// SAE-import provenance, carried VERBATIM (server key
    /// `vector_store.SteeringVectorSidecar.gemmascopeSource`, a free dict).
    ///
    /// The same reason `ExperimentManifest.jlensReadout` is opaque here: this
    /// engine cannot MINT the block — direct-feature-ID Gemma Scope import is
    /// a server verb on the science substrate — but it can decode and
    /// re-encode a sidecar that has one. `NormBackfill.backfillNorms` does
    /// exactly that (decode → measure norms → re-encode into a new artifact),
    /// so without a passthrough a backfilled SAE artifact would silently lose
    /// its feature identity: which release, which SAE id, which decoder row,
    /// at which revision. The server's promotion and qualification chains
    /// READ that block (`sae_qualification.py` matches an artifact by its
    /// `gemmascopeSource` identity), so dropping it destroys the citation
    /// rather than merely losing a label. Stored as opaque JSON so the server
    /// can extend the block without this engine learning each key.
    public var gemmascopeSource: SidecarJSON?
    /// OptVec (trained/optimized vector) provenance, carried VERBATIM (server
    /// key `optvec_train._save_artifact`'s `payload["optvec"]`, a free dict).
    ///
    /// Same reason as `gemmascopeSource`: this engine cannot MINT the block —
    /// OptVec training is a server verb on the science substrate — but it
    /// must not DESTROY it. `NormBackfill.backfillNorms` decodes → measures →
    /// re-encodes through this struct, so before this field existed a
    /// Swift-side backfill of an OptVec artifact silently dropped the whole
    /// block, after which `ExperimentStore.attachArtifact` refused the result
    /// ("carries no 'optvec' provenance block"). The server twin avoids it by
    /// copying raw JSON.
    ///
    /// The block carries the §24 stamps that make an OptVec dose checkable
    /// rather than merely trusted: `vectorPackaging`, `alphaNormFactor` /
    /// `alphaAbsolute`, and the residual-norm denominator's provenance. Read
    /// them through `OptVecPackaging` rather than by poking at keys.
    public var optvec: SidecarJSON?

    public struct NormBackfillProvenance: Codable, Sendable, Equatable {
        /// The original artifact id (`<runDir>/<name>` base path).
        public var sourceArtifact: String
        /// SHA-256 hex of the ORIGINAL `.safetensors` bytes — which are also
        /// this artifact's bytes, byte-for-byte (backfill never re-encodes).
        public var sourceVectorsHash: String
        /// ISO8601 UTC timestamp of the backfill measurement.
        public var date: String
        /// Present only on REDENOMINATION: the residualNormSource the new
        /// artifact replaced (e.g. "extraction-stimuli"). Absent on classic
        /// fills of norm-less artifacts (pinned cross-engine JSON shape).
        public var replacedNormSource: String?

        public init(
            sourceArtifact: String, sourceVectorsHash: String, date: String,
            replacedNormSource: String? = nil
        ) {
            self.sourceArtifact = sourceArtifact
            self.sourceVectorsHash = sourceVectorsHash
            self.date = date
            self.replacedNormSource = replacedNormSource
        }
    }

    public init(
        modelID: String, revision: String? = nil, concept: String,
        stimulusSetHash: String, vectors: ConceptVectors,
        options: ExtractionOptions? = nil, residualNormPerLayer: [Float]? = nil,
        residualNormSource: String? = nil,
        residualNormConvention: String? = nil,
        residualNormRendering: String? = nil,
        extractionRendering: ExtractionRendering? = nil,
        readingPositionResolution: ReadingPositionResolutionReport? = nil,
        recipeMethod: VectorExtractionRecipe.Method? = nil,
        recipeHash: String? = nil,
        recipeName: String? = nil,
        appliedReadingPosition: ReadingPosition? = nil,
        neutralProjection: String? = nil,
        neutralCorpusHash: String? = nil,
        screening: ExtractionScreening? = nil,
        neutralScreening: ExtractionScreening? = nil,
        comparisonConcepts: [String]? = nil,
        selectedTopics: [String]? = nil,
        selectedSplits: [String]? = nil,
        extractionDate: Date = Date()
    ) {
        self.schemaVersion = 2
        self.modelID = modelID
        self.revision = revision
        self.concept = concept
        self.stimulusSetHash = stimulusSetHash
        self.layerCount = vectors.layerCount
        self.hiddenSize = vectors.hiddenSize
        self.normsPerLayer = (0 ..< vectors.layerCount).map { vectors.norm(at: $0) }
        self.extractionDate = ISO8601DateFormatter().string(from: extractionDate)
        self.extractionMethod = options?.method.rawValue
        self.readingPosition = options?.readingPosition.label ?? appliedReadingPosition?.label
        let legacyProjection = (options?.neutralPCCount).flatMap {
            $0 > 0 ? "top-\($0) neutral PCs" : nil
        }
        self.confoundProjection = legacyProjection
        self.residualNormPerLayer = residualNormPerLayer
        self.residualNormSource = residualNormSource
        // Never invented: a caller that measured norms passes the convention
        // it measured under; one that copies or omits norms passes nil, and
        // the artifact stays honestly unstamped (legacy).
        self.residualNormConvention = residualNormConvention
        // Absent-not-null, exactly like the convention stamp: a raw
        // extraction writes NOTHING, so its sidecar bytes stay identical to
        // what this engine has always written.
        self.residualNormRendering =
            residualNormRendering == ExtractionRendering.Mode.raw.rawValue
            ? nil : residualNormRendering
        self.extractionRendering =
            (extractionRendering ?? options?.extractionRendering)?.stamp
        self.readingPositionResolution = readingPositionResolution
        self.recipeMethod = recipeMethod?.rawValue
        self.recipeHash = recipeHash
        self.recipeName = recipeName
        self.neutralProjection = neutralProjection ?? legacyProjection.map { "legacy-pooled \($0)" }
            ?? "none"
        self.neutralCorpusHash = neutralCorpusHash
        self.readingMinimumTokenCount =
            screening?.minimumTokenCount ?? options?.readingPosition.minimumTokenCount
            ?? appliedReadingPosition?.minimumTokenCount
        self.sourceStimulusCount = screening?.sourceCount
        self.includedStimulusCount = screening?.includedCount
        self.excludedShortStimulusCount = screening?.excludedShortCount
        self.neutralSourceStimulusCount = neutralScreening?.sourceCount
        self.neutralIncludedStimulusCount = neutralScreening?.includedCount
        self.neutralExcludedShortStimulusCount = neutralScreening?.excludedShortCount
        self.comparisonConcepts = comparisonConcepts
        self.selectedTopics = selectedTopics
        self.selectedSplits = selectedSplits
        self.source = nil
        self.readerID = nil
        self.readerHash = nil
        self.controlMode = nil
        self.substrate = nil  // stamped at save (SteeringVectorStore.stamped)
        self.normBackfill = nil
        self.gemmascopeConvention = nil  // stamped by the Gemma Scope importer
        self.rawDecoderNorm = nil
        self.gemmascopeTargetNorm = nil
        self.grandMeanPopulation = nil  // stamped by the experiment writers
        self.neutralMeanSource = nil  // stamped by SteeringVectorStore.save
        self.recipeIdentityHash = nil  // stamped by the experiment writers
        self.gemmascopeSource = nil  // stamped by the server's SAE importer
        self.optvec = nil  // stamped by the server's OptVec trainer
    }
}

/// Opaque JSON for sidecar blocks this engine carries but never interprets.
///
/// SteeringKit is concept-agnostic and cannot depend on ExperimentKit (where
/// the equivalent `JSONValue` lives), so the passthrough type is declared
/// here. `Equatable` makes verbatim preservation ASSERTABLE in a test rather
/// than merely intended.
public enum SidecarJSON: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: SidecarJSON])
    case array([SidecarJSON])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([SidecarJSON].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: SidecarJSON].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

public enum SteeringVectorStore {

    public struct NonFiniteArtifactError: Error, CustomStringConvertible {
        public let reason: String
        public var description: String { reason }
    }

    /// Refuses artifacts containing NaN/infinite values — a numerical failure
    /// (e.g. fp16 activation overflow) must become an explicit error, never a
    /// poisoned artifact. Pure, so it is unit-testable without MLX; the
    /// server engine enforces the same invariant in `vector_store.save`.
    public static func validateFinite(
        vectors: ConceptVectors, sidecar: SteeringVectorSidecar
    ) throws {
        for (index, vector) in vectors.perLayer.enumerated()
        where !vector.allSatisfy(\.isFinite) {
            throw NonFiniteArtifactError(
                reason: "vector layer_\(index) contains non-finite values — "
                    + "activation overflow? (check model dtype)")
        }
        for (label, values) in [
            ("normsPerLayer", sidecar.normsPerLayer),
            ("residualNormPerLayer", sidecar.residualNormPerLayer),
        ] {
            if let values, !values.allSatisfy(\.isFinite) {
                throw NonFiniteArtifactError(
                    reason: "\(label) contains non-finite values")
            }
        }
    }

    /// Substrate stamping at save time (pure, so it is unit-testable without
    /// MLX): every artifact this engine writes is stamped
    /// `RepEReader.substrate` ("swift-mlx") unless the sidecar already
    /// carries an explicit stamp — an existing stamp is provenance and is
    /// never overwritten.
    public static func stamped(_ sidecar: SteeringVectorSidecar) -> SteeringVectorSidecar {
        var sidecar = sidecar
        if sidecar.substrate == nil {
            sidecar.substrate = RepEReader.substrate
        }
        return sidecar
    }

    public struct NeutralMeanError: Error, CustomStringConvertible {
        public let reason: String
        public var description: String { reason }
    }

    /// Writes `<name>.safetensors` (keys `layer_<i>`) and `<name>.json`.
    ///
    /// `neutralMeanPerLayer`, when supplied, is stored alongside as
    /// `neutral_mean_layer_<i>` tensors and stamped `neutralMeanSource:
    /// "neutral-corpus"` — additive keys old readers ignore by name (the
    /// same contract the server writes).
    @discardableResult
    public static func save(
        vectors: ConceptVectors, sidecar: SteeringVectorSidecar,
        to directory: URL, name: String,
        neutralMeanPerLayer: [[Float]]? = nil
    ) throws -> (vectors: URL, sidecar: URL) {
        var sidecar = stamped(sidecar)
        try validateFinite(vectors: vectors, sidecar: sidecar)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        var arrays = Dictionary(
            uniqueKeysWithValues: vectors.perLayer.enumerated().map { index, vector in
                ("layer_\(index)", MLXArray(vector))
            })
        if let neutralMeanPerLayer {
            guard neutralMeanPerLayer.count == vectors.layerCount else {
                throw NeutralMeanError(
                    reason: "neutral mean has \(neutralMeanPerLayer.count) layers, "
                        + "vectors have \(vectors.layerCount)")
            }
            for (index, mean) in neutralMeanPerLayer.enumerated() {
                guard mean.allSatisfy(\.isFinite) else {
                    throw NeutralMeanError(
                        reason: "neutral_mean_layer_\(index) contains non-finite values")
                }
                arrays["neutral_mean_layer_\(index)"] = MLXArray(mean)
            }
            sidecar.neutralMeanSource = "neutral-corpus"
        }
        let vectorsURL = directory.appending(component: "\(name).safetensors")
        try MLX.save(arrays: arrays, url: vectorsURL)

        let sidecarURL = directory.appending(component: "\(name).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(sidecar).write(to: sidecarURL)

        return (vectorsURL, sidecarURL)
    }

    public static func load(
        from directory: URL, name: String
    ) throws -> (vectors: ConceptVectors, sidecar: SteeringVectorSidecar) {
        let sidecarURL = directory.appending(component: "\(name).json")
        let sidecar = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(contentsOf: sidecarURL))

        let vectorsURL = directory.appending(component: "\(name).safetensors")
        let arrays = try MLX.loadArrays(url: vectorsURL)
        let perLayer: [[Float]] = try (0 ..< sidecar.layerCount).map { index in
            guard let array = arrays["layer_\(index)"] else {
                throw StimulusSetError.missingFile(vectorsURL)
            }
            return array.asType(.float32).asArray(Float.self)
        }
        return (ConceptVectors(perLayer: perLayer), sidecar)
    }

    /// The per-layer neutral residual mean stored with an artifact, or nil.
    ///
    /// nil means the artifact predates the neutral-mean stamp or was
    /// extracted without a neutral corpus — callers must then treat the
    /// ablation mean-alignment as UNKNOWN, never as zero.
    public static func loadNeutralMean(
        from directory: URL, name: String
    ) throws -> [[Float]]? {
        let sidecarURL = directory.appending(component: "\(name).json")
        let sidecar = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(contentsOf: sidecarURL))
        guard sidecar.neutralMeanSource != nil else { return nil }
        let vectorsURL = directory.appending(component: "\(name).safetensors")
        let arrays = try MLX.loadArrays(url: vectorsURL)
        return try (0 ..< sidecar.layerCount).map { index in
            guard let array = arrays["neutral_mean_layer_\(index)"] else {
                throw NeutralMeanError(
                    reason: "artifact '\(vectorsURL.path)' stamps neutralMeanSource "
                        + "but its safetensors is missing neutral_mean_layer_\(index) "
                        + "— corrupt artifact")
            }
            let values = array.asType(.float32).asArray(Float.self)
            guard values.allSatisfy(\.isFinite) else {
                throw NeutralMeanError(
                    reason: "artifact '\(vectorsURL.path)' neutral_mean_layer_\(index) "
                        + "contains non-finite values")
            }
            return values
        }
    }
}

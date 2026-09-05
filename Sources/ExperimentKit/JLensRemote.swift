import Foundation

/// Remote J-Space surface: the Mac app RENDERS lens artifacts, never produces them.
///
/// J-lens work is server-only by rule (hard requirement): the imported
/// lens artifacts are PyTorch/HF-native and
/// activations do not transfer across substrates, so there is no local/MLX
/// equivalent to fall back to. Every type here is a read/render model over a
/// server response — this file deliberately contains no derivation, no
/// injection, and no readout math.
///
/// Everything is decoded leniently. The server owns the schema and will extend
/// these blocks; a strict decoder here would turn a server-side addition into a
/// broken panel on the Mac.

// MARK: - Records

public struct JLensSupportedModel: Codable, Sendable, Identifiable, Equatable {
    public var modelID: String
    public var tier: String
    public var folder: String?
    public var tensor: String?

    public var id: String { modelID }

    /// Whether this model is the one THIS PROJECT draws evidence from.
    ///
    /// A scope decision, not a capability claim: the study chose 27B, so 27B is
    /// the evidence tier here. Good science could come out of a 12B J-lens run —
    /// it would simply not be this study's evidence, and the tier is what stops a
    /// run being cited as though the decision had been otherwise. Surfaced in the
    /// UI because it must be visible rather than inferable from a model id.
    public var isEvidenceTier: Bool { tier == "evidence" }
}

public struct JLensQualification: Codable, Sendable, Equatable {
    public var qualificationID: String
    public var modelID: String
    public var revision: String
    public var dtype: String
    public var quantization: String?
    public var tier: String?
    public var passed: Bool
    public var qualifiedAt: String?
}

public struct JLensSource: Codable, Sendable, Equatable {
    public var repo: String?
    public var folder: String?
    public var tensorFile: String?
    public var configFile: String?
    public var commit: String?
    public var tensorSHA256: String?
    public var configSHA256: String?
}

public struct JLensFitProvenance: Codable, Sendable, Equatable {
    public var modelID: String?
    /// Nil means UNKNOWN, and unknown is the honest value: the published
    /// artifacts carry no base-model revision. The UI must show it as unknown
    /// rather than substituting the runtime's.
    public var revision: String?
    public var revisionKnown: Bool?
    public var dtype: String?
    public var corpus: String?
    public var promptsFitted: Int?
    public var maxSeqLen: Int?
}

public struct JLensConverted: Codable, Sendable, Equatable {
    public var path: String?
    public var dtype: String?
    public var sha256: String?
    public var layerCount: Int?
}

public struct JLensRecord: Codable, Sendable, Identifiable, Equatable {
    public var lensID: String
    public var source: JLensSource?
    public var fit: JLensFitProvenance?
    public var sourceLayers: [Int]?
    public var dModel: Int?
    public var targetLayer: Int?
    public var nPrompts: Int?
    public var converted: JLensConverted?
    public var referencePackage: String?
    public var referenceCommit: String?
    public var readoutConvention: String?
    public var directionConvention: String?
    public var substrate: String?
    public var importedAt: String?
    public var qualifications: [JLensQualification]?

    public var id: String { lensID }

    public var passingQualifications: [JLensQualification] {
        (qualifications ?? []).filter(\.passed)
    }

    public var layerSpan: String {
        guard let layers = sourceLayers, let first = layers.first,
              let last = layers.last else { return "—" }
        return "\(first)…\(last) → \(targetLayer.map(String.init) ?? "?")"
    }
}

public struct JLensCatalog: Codable, Sendable, Equatable {
    public var lenses: [JLensRecord]
    public var supported: [JLensSupportedModel]
}

public struct JLensTokenCandidate: Codable, Sendable, Identifiable, Equatable {
    public var tokenID: Int
    public var piece: String
    public var decoded: String?
    public var decodedBytes: String
    public var form: String
    public var singleToken: Bool
    public var sequence: [Int]
    public var note: String?

    /// Stable across the two query forms that can surface the same id.
    public var id: String { "\(form)-\(tokenID)" }
}

public struct JLensTokenOptions: Codable, Sendable, Equatable {
    public var query: String
    public var candidates: [JLensTokenCandidate]
    /// Always "explicit": the service never recommends a candidate, because
    /// silently taking the first component of a multi-token word is the exact
    /// failure it exists to prevent.
    public var selection: String?
    public var truncated: Bool?
}

public struct JLensDecodedTokens: Codable, Sendable, Equatable {
    public var modelID: String
    public var pieces: [String: String]
}

// MARK: - Client

extension ClusterClient {

    public func jlensCatalog() async throws -> JLensCatalog {
        try await get("/api/jlens/lenses")
    }

    public func jlensLens(id lensID: String) async throws -> JLensRecord {
        let escaped = lensID.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed) ?? lensID
        return try await get("/api/jlens/lenses/\(escaped)")
    }

    /// Fetch the published lens bytes on the SERVER (privileged; goes online).
    public func jlensAcquire(modelID: String) async throws -> String {
        struct Body: Encodable { let modelID: String }
        struct Reply: Decodable { let jobId: String }
        let reply: Reply = try await post("/api/jlens/lenses/acquire",
                                         body: Body(modelID: modelID))
        return reply.jobId
    }

    /// Convert a cached lens into the server workspace (privileged; offline).
    public func jlensImport(modelID: String) async throws -> String {
        struct Body: Encodable { let modelID: String }
        struct Reply: Decodable { let jobId: String }
        let reply: Reply = try await post("/api/jlens/lenses/import",
                                         body: Body(modelID: modelID))
        return reply.jobId
    }

    public func jlensTokenOptions(
        modelID: String, text: String, includeCaseVariants: Bool = false
    ) async throws -> JLensTokenOptions {
        struct Body: Encodable {
            let modelID: String
            let text: String
            let includeCaseVariants: Bool
        }
        return try await post("/api/jlens/token-options",
                              body: Body(modelID: modelID, text: text,
                                         includeCaseVariants: includeCaseVariants))
    }

    /// Derive a direction from an EXACT token id.
    ///
    /// The id is the parameter, never the word: resolving a word here would
    /// reintroduce the silent mis-selection the token-option service exists to
    /// prevent.
    /// ``concept`` is an OPTIONAL grouping association, separate from ``name``.
    /// Absent, the direction groups only with itself — filing a one-token
    /// direction under a stimulus-defined concept of the same label would show
    /// them as two extractions of one thing in every concept-grouped view, and
    /// doing that silently is what the plan's last non-goal forbids.
    public func jlensDerive(
        lensID: String, modelID: String, tokenID: Int,
        piece: String? = nil, name: String? = nil, concept: String? = nil
    ) async throws -> String {
        struct Body: Encodable {
            let lensID: String
            let modelID: String
            let tokenID: Int
            let piece: String?
            let name: String?
            let concept: String?
        }
        struct Reply: Decodable { let jobId: String }
        let reply: Reply = try await post(
            "/api/jlens/directions/derive",
            body: Body(lensID: lensID, modelID: modelID, tokenID: tokenID,
                       piece: piece, name: name, concept: concept))
        return reply.jobId
    }

    public func jlensDecodeTokens(
        modelID: String, tokenIDs: [Int]
    ) async throws -> JLensDecodedTokens {
        struct Body: Encodable { let modelID: String; let tokenIDs: [Int] }
        return try await post("/api/jlens/decode-tokens",
                              body: Body(modelID: modelID, tokenIDs: tokenIDs))
    }

    /// Read a stored concept vector back as J-lens token atoms (privileged).
    ///
    /// `vectorID` is the catalog's `"<runDirectory>/<name>"`. Layers default to
    /// every fitted source layer of the lens: readability varies by layer, and
    /// choosing one on the caller's behalf would hide exactly that.
    public func jlensSupport(
        lensID: String, vectorID: String, layers: [Int]? = nil,
        budget: Int = 25
    ) async throws -> String {
        struct Body: Encodable {
            let lensID: String
            let vectorID: String
            let layers: [Int]?
            let budget: Int
        }
        struct Reply: Decodable { let jobId: String }
        let reply: Reply = try await post(
            "/api/jlens/support",
            body: Body(lensID: lensID, vectorID: vectorID, layers: layers,
                       budget: budget))
        return reply.jobId
    }
}

// MARK: - Support readout

/// One atom in a vector's support: an exact vocabulary token and its weight.
public struct JLensSupportAtom: Codable, Sendable, Equatable, Identifiable {
    public var tokenID: Int
    public var piece: String
    public var coefficient: Double
    /// Share of the reconstructed direction's length. Atoms are unit-norm, so
    /// coefficients ARE lengths and the shares are comparable across tokens.
    public var share: Double

    public var id: Int { tokenID }
}

/// One layer's readout.
///
/// `energyFraction` is never presented without `nullEnergyFraction`: a
/// matched-norm random direction scores comparably in this dictionary (measured
/// 3–17% for real concept vectors against 11–32% for nulls on gemma-3-4b-it), so
/// the fraction alone says nothing. The server's schema cannot express one
/// without the other, and neither can this view model.
public struct JLensSupportLayer: Codable, Sendable, Equatable, Identifiable {
    public var layer: Int
    public var support: [JLensSupportAtom]
    public var energyFraction: Double
    public var nullEnergyFraction: Double
    public var energyOverNull: Double
    /// Where non-negative pursuit ran out of positively-correlated atoms. The
    /// atom cone is narrow (embedding rows share a strong common component), so
    /// this is a normal outcome and a real ceiling — not a failure.
    public var coneExhaustedAt: Int?
    public var nullConeExhaustedAt: Int?

    public var id: Int { layer }

    /// Whether the reconstruction beat its own control at all. False is the
    /// common case and is not a defect in the vector — read the tokens.
    public var beatsNull: Bool { energyOverNull > 0 }
}

public struct JLensSupportVector: Codable, Sendable, Equatable {
    public var runDirectory: String
    public var name: String
    public var concept: String?
    public var sha256: String?
    public var extractionMethod: String?
    public var recipeMethod: String?
}

/// A full support readout, as returned by the `jlens-support` job and stored in
/// `support.json`.
public struct JLensSupportReadout: Codable, Sendable, Equatable {
    public var lensID: String
    public var modelID: String
    public var revision: String?
    public var vector: JLensSupportVector
    public var budget: Int
    public var layers: [JLensSupportLayer]
    public var supportIdentityHash: String?
    public var runDirectory: String?
    public var lensFitPrompts: Int?
    public var lensFitCorpus: String?
    public var nullSeed: Int?

    /// The layer whose support is most likely to be worth reading: the largest
    /// margin over its own control. Presented as a hint, never as "the answer" —
    /// a negative best margin is normal and the tokens still carry the signal.
    public var strongestLayer: JLensSupportLayer? {
        layers.max { $0.energyOverNull < $1.energyOverNull }
    }
}

// MARK: - Readout traces

/// One armed layer at one recorded position, as stored in `jlens-readout.jsonl`.
public struct JLensObservation: Codable, Sendable, Equatable, Identifiable {
    public var layer: Int
    public var passKind: String?
    public var position: Int
    public var predictedIndex: Int
    public var predictedTokenID: Int?
    public var watched: [Double]?
    public var watchedLogitLens: [Double]?
    public var topKIDs: [Int]?
    public var topKLogits: [Double]?
    public var topKIDsLogitLens: [Int]?
    public var topKLogitsLogitLens: [Double]?

    public var id: String { "\(layer)-\(position)" }

    /// True when the trace could not name the token this activation predicted —
    /// which makes the row uninterpretable, not merely incomplete.
    public var isUnaligned: Bool { predictedTokenID == nil }
}

/// One traced generation: the identity block, plus its nested observations.
public struct JLensTraceRow: Codable, Sendable, Equatable, Identifiable {
    public var run: String?
    public var condition: String?
    public var promptID: String?
    public var promptIndex: Int?
    public var sampleIndex: Int?
    public var modelID: String?
    public var modelRevision: String?
    public var dtype: String?
    public var lensID: String?
    public var lensSHA256: String?
    public var qualificationID: String?
    public var tokenizerHash: String?
    /// Trustworthiness of the READOUT — a property of the model tier and the
    /// lens qualification. It answers "can this lens be believed on this
    /// runtime", never "is this row reportable"; see ``conditionClaim``.
    public var evidenceTier: String?
    /// Reportability of THIS ROW: the weaker of the lens's tier and the
    /// identity of the condition being read. `nil` on traces recorded before
    /// 2026-08-16, when the lens tier was stamped on every condition alike.
    public var conditionClaim: String?
    /// Why ``conditionClaim`` says what it says — the agent's adapter pin
    /// status. Absent for baseline/concept conditions, which pin nothing.
    public var conditionIdentity: JLensConditionIdentity?
    public var substrate: String?
    public var steering: [JLensSteeringCell]?
    public var observations: [JLensObservation]?
    public var observationCount: Int?
    public var generatedTokenCount: Int?
    public var traceComplete: Bool?
    public var traceFailureReason: String?
    public var watchlistTokenIDs: [Int]?
    /// `predictedIndex` (as a string key) → tokenID → already-mentioned.
    public var mentionMask: [String: [String: Bool]]?

    public var id: String {
        "\(condition ?? "?")-\(promptID ?? "?")-\(sampleIndex ?? 0)"
    }

    /// The LENS's tier. Correct for "can the readout be believed", and the
    /// wrong question for "may this row be cited" — a qualified lens over an
    /// agent whose `adapter_config.json` is unpinned is still an exploratory
    /// measurement (external review round 7). Use ``isReportable``.
    public var isEvidenceTier: Bool { evidenceTier == "evidence" }

    /// May this row be described as evidence? Both halves must hold: the lens
    /// believable AND the condition pinned. An unstamped legacy row is not
    /// reportable — its claim was never evaluated, which is not the same as
    /// having been evaluated and passed.
    public var isReportable: Bool { conditionClaim == "qualified" }

    /// Why not, when not — for the UI to say something truer than "no".
    public var unreportableReason: String? {
        if isReportable { return nil }
        if conditionClaim == nil {
            return "recorded before per-condition claims existed; treat as exploratory"
        }
        let unpinned = (conditionIdentity?.adapters ?? [])
            .filter { !($0.configHashPinned ?? false) }
            .map(\.adapterDirectory)
        if !unpinned.isEmpty {
            return "agent configuration unpinned (\(unpinned.joined(separator: ", ")))"
                + " — the lens may be qualified, but the agent being read is not"
        }
        return isEvidenceTier ? "condition is exploratory" : "testing-tier runtime"
    }

    /// Was any watched token primed at this step, in the prompt or the prefix
    /// the model had already written? A primed token sits near ceiling for
    /// reasons unrelated to the model's state.
    public func isMentionPrimed(atStep step: Int) -> Bool {
        (mentionMask?[String(step)] ?? [:]).values.contains(true)
    }

    public var label: String {
        let parts = [condition, promptID].compactMap { $0 }
        let base = parts.isEmpty ? "trace" : parts.joined(separator: " · ")
        if let sample = sampleIndex, sample > 0 { return "\(base) · sample \(sample)" }
        return base
    }
}

/// The pin status of the agent a J-lens condition ran. Rendered, never
/// produced here: J-lens artifacts are server-only (CLAUDE.md hard
/// requirement).
public struct JLensConditionIdentity: Codable, Sendable, Equatable {
    public struct Adapter: Codable, Sendable, Equatable {
        public var adapterDirectory: String
        public var adapterHashPinned: Bool?
        public var configHashPinned: Bool?
        public var adapterHash: String?
        public var configHash: String?
        public var configurationUnpinned: String?
    }
    public var variantName: String?
    public var adapters: [Adapter]?
}

public struct JLensSteeringCell: Codable, Sendable, Equatable {
    public var layer: Int?
    public var alpha: Double?
    public var mode: String?
    public var concept: String?

    public var summary: String {
        let name = concept ?? "vector"
        let where_ = layer.map { "L\($0)" } ?? "L?"
        let amount = alpha.map { String(format: "%.2f", $0) } ?? "?"
        return "\(mode ?? "add") \(name) @ \(where_) α\(amount)"
    }
}

/// Everything the app needs from a trace file, decoded once.
public struct JLensTrace: Sendable, Equatable {
    public var rows: [JLensTraceRow]

    /// Parse `jlens-readout.jsonl`.
    ///
    /// A malformed line is COUNTED, never skipped silently: the whole point of
    /// the completeness machinery is that a partial readout cannot pass as a
    /// whole one, and a viewer that quietly drops rows would undo it.
    public static func parse(_ data: Data) throws -> (trace: JLensTrace, malformed: Int) {
        let decoder = JSONDecoder()
        var rows: [JLensTraceRow] = []
        var malformed = 0
        let text = String(decoding: data, as: UTF8.self)
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            guard let lineData = line.data(using: .utf8) else { malformed += 1; continue }
            if let row = try? decoder.decode(JLensTraceRow.self, from: lineData) {
                rows.append(row)
            } else {
                malformed += 1
            }
        }
        return (JLensTrace(rows: rows), malformed)
    }

    public var isComplete: Bool {
        !rows.isEmpty && rows.allSatisfy { $0.traceComplete == true }
    }

    public var incompleteCount: Int {
        rows.filter { $0.traceComplete != true }.count
    }

    public var totalObservations: Int {
        rows.reduce(0) { $0 + ($1.observationCount ?? $1.observations?.count ?? 0) }
    }
}

/// The layer × prediction-step grid behind the heatmap.
///
/// Pure and separately testable on purpose: the interesting part of a heatmap is
/// the normalization, and a bug there produces a picture that looks plausible
/// and is wrong. Rendering it inside a View body would make that untestable.
public struct JLensHeatmap: Sendable, Equatable {
    public struct Cell: Sendable, Equatable {
        public var layer: Int
        public var step: Int
        public var value: Double?
        /// 0…1 within the grid's own range; nil when there is no value.
        public var intensity: Double?
        public var mentionPrimed: Bool
        public var predictedTokenID: Int?
    }

    public var layers: [Int]
    public var steps: [Int]
    public var cells: [Cell]
    public var minimum: Double?
    public var maximum: Double?
    /// Which watched token the grid is showing. One token, so the colour means
    /// one thing — a blended metric would be unreadable.
    public var watchedTokenID: Int?

    public func cell(layer: Int, step: Int) -> Cell? {
        cells.first { $0.layer == layer && $0.step == step }
    }

    /// Build from one traced generation.
    ///
    /// `watchlistIndex` selects which watched token to colour by. A flat range
    /// (every value equal) yields intensity 0.5 rather than dividing by zero —
    /// a uniform grid is the honest picture of uniform data.
    public static func build(row: JLensTraceRow, watchlistIndex: Int = 0) -> JLensHeatmap {
        let observations = row.observations ?? []
        let layers = Array(Set(observations.map(\.layer))).sorted()
        let steps = Array(Set(observations.map(\.predictedIndex))).sorted()

        func value(_ obs: JLensObservation) -> Double? {
            guard let watched = obs.watched, watchlistIndex < watched.count
            else { return nil }
            return watched[watchlistIndex]
        }

        let values = observations.compactMap(value)
        let low = values.min()
        let high = values.max()
        let span = (low != nil && high != nil) ? (high! - low!) : 0

        var cells: [Cell] = []
        for obs in observations {
            let raw = value(obs)
            let intensity: Double? = raw.flatMap { v in
                guard let low else { return nil }
                return span > 0 ? (v - low) / span : 0.5
            }
            cells.append(Cell(
                layer: obs.layer, step: obs.predictedIndex, value: raw,
                intensity: intensity,
                mentionPrimed: row.isMentionPrimed(atStep: obs.predictedIndex),
                predictedTokenID: obs.predictedTokenID))
        }
        return JLensHeatmap(
            layers: layers, steps: steps, cells: cells,
            minimum: low, maximum: high,
            watchedTokenID: row.watchlistTokenIDs?.indices.contains(watchlistIndex) == true
                ? row.watchlistTokenIDs?[watchlistIndex] : nil)
    }
}

extension ClusterClient {

    /// Read a run's `jlens-readout.jsonl` and decode it.
    ///
    /// Returns the malformed-line count alongside the trace rather than
    /// swallowing it: a viewer that quietly drops unparseable rows would undo
    /// the completeness machinery the whole trace format is built around.
    public func jlensTrace(runID: String) async throws -> (trace: JLensTrace,
                                                           malformed: Int) {
        let data = try await runFile(runID: runID, name: "jlens-readout.jsonl")
        return try JLensTrace.parse(data)
    }
}

import Foundation
import MLX
import MLXLMCommon

public enum ConceptExtractorError: Error, CustomStringConvertible {
    case modelNotHookable(String)
    case missingCaptures(text: String)
    case layerMismatch
    case stimulusTooShort(text: String, tokenCount: Int, minimumTokenCount: Int, reading: String)
    case tooManyShortStimuli(excluded: Int, total: Int, maximumFraction: Double)
    /// A reading position could not be resolved for this stimulus — a named
    /// role under raw rendering, an offset past the sequence start, a
    /// template with no end-of-turn marker. Carries the typed
    /// `ReadingPositionError`'s message (which already names its repair) plus
    /// the offending stimulus.
    case readingPositionUnresolved(reason: String, text: String)
    /// The rendering declared for this extraction could not be applied on
    /// this engine — e.g. the tokenizer has no chat template. Named, never a
    /// silent fallback to raw: a silent fallback is the exact ambiguity the
    /// `extractionRendering` option exists to end.
    case renderingUnsupported(rendering: String, modelID: String, reason: String)
    /// The projected neutral token bank exceeds the memory budget. Thrown
    /// BEFORE any forward pass — an over-budget bank fails inside MLX/native
    /// allocation, which kills the process rather than raising.
    case neutralBankTooLarge(NeutralBankPreflight)
    /// The loaded model does not report its residual-stream shape, so the
    /// bank cannot be sized before capture.
    case residualShapeUnavailable(String)

    public var description: String {
        switch self {
        case .modelNotHookable(let type):
            "model \(type) is not InterventionHookable — wrong factory?"
        case .missingCaptures(let text):
            "no activations recorded for stimulus: \(text.prefix(40))…"
        case .layerMismatch:
            "stimuli produced different layer counts"
        case .stimulusTooShort(let text, let tokenCount, let minimumTokenCount, let reading):
            "stimulus is too short for \(reading): \(tokenCount) tokens, needs "
                + "\(minimumTokenCount) (\(text.prefix(40))…)"
        case .tooManyShortStimuli(let excluded, let total, let maximumFraction):
            "excluded \(excluded)/\(total) stimuli as too short; maximum allowed is "
                + "\(Int((maximumFraction * 100).rounded()))%"
        case .readingPositionUnresolved(let reason, let text):
            "\(reason) (stimulus: \(text.prefix(40))…)"
        case .renderingUnsupported(let rendering, let modelID, let reason):
            "extraction rendering '\(rendering)' cannot be applied on the "
                + "swift-mlx engine for \(modelID): \(reason). Repair: extract "
                + "this concept on an engine that supports the rendering, or "
                + "re-attach it raw — steerlab-cli experiment attach <study> "
                + "<concept> \(ExtractionRendering.declarationFlag) "
                + "'{\"mode\": \"raw\"}' — never let it fall back silently, "
                + "because a direction read from raw activations is not the "
                + "direction read from templated ones."
        case .neutralBankTooLarge(let preflight):
            "neutral token bank is too large to build: \(preflight.summary). "
                + "Restrict the capture to a layer band (the middle third is the "
                + "convention) and/or lower the per-layer row cap (currently "
                + "\(preflight.rowCapPerLayer)); a smaller neutral corpus also works."
        case .residualShapeUnavailable(let type):
            "model \(type) does not report its residual-stream shape "
                + "(ResidualShapeProviding) — the neutral bank cannot be sized "
                + "before capture, so it is refused rather than risked."
        }
    }
}

/// Per-layer concept directions for one concept on one model.
public struct ConceptVectors: Sendable {
    /// `perLayer[l]` is the concept direction at block `l`'s output.
    public let perLayer: [[Float]]

    public var layerCount: Int { perLayer.count }
    public var hiddenSize: Int { perLayer.first?.count ?? 0 }

    public func norm(at layer: Int) -> Float {
        SteeringVectorMath.l2Norm(perLayer[layer])
    }

    public init(perLayer: [[Float]]) {
        self.perLayer = perLayer
    }
}

/// Configuration for one extraction (METHODS.md › Method options).
public struct ExtractionOptions: Codable, Sendable, Equatable {
    public var method: ExtractionMethod
    public var readingPosition: ReadingPosition
    /// Legacy fixed-k neutral projection. This remains available for draft
    /// method exploration, but frozen experiments should not rely on it: it
    /// estimates PCs from one pooled activation per neutral text rather than
    /// from a token-position activation bank, so it can remove genuine
    /// concept signal. Experiment verification blocks this path until the
    /// calibrated neutral bank exists.
    public var neutralPCCount: Int?
    /// HOW the stimulus string reaches the model — raw (legacy, and what an
    /// ABSENT declaration means) or the family chat template. Optional and
    /// nil-defaulted on purpose: Swift's synthesized encoder omits a nil
    /// optional, so a manifest that never declares a rendering encodes to
    /// byte-identical JSON and keeps its recipe-identity hash. See
    /// ``ExtractionRendering``.
    public var extractionRendering: ExtractionRendering?

    /// The rendering actually applied — absent resolves to legacy raw.
    public var resolvedExtractionRendering: ExtractionRendering {
        extractionRendering ?? .raw
    }

    public init(
        method: ExtractionMethod = .meanDifference,
        readingPosition: ReadingPosition = .lastToken,
        neutralPCCount: Int? = nil,
        extractionRendering: ExtractionRendering? = nil
    ) {
        self.method = method
        self.readingPosition = readingPosition
        self.neutralPCCount = neutralPCCount
        self.extractionRendering = extractionRendering
    }
}

public struct ExtractionResult: Sendable {
    public let vectors: ConceptVectors
    /// Mean residual-stream L2 norm per layer — the denominator for
    /// expressing steering strength in norm units. Measured on the neutral
    /// corpus when one is provided (the emotion paper's convention: a fixed
    /// dataset, so α is denominated identically across concepts), else on
    /// the extraction stimuli (legacy; concept-dependent units).
    public let residualNormPerLayer: [Float]
    /// Where `residualNormPerLayer` came from: "neutral-corpus" or
    /// "extraction-stimuli".
    public let residualNormSource: String
    /// HOW those positions were averaged — always the current convention,
    /// because this value describes a measurement THIS code just made.
    /// Sidecar writers stamp it verbatim; see `ResidualNormConvention`.
    public let residualNormConvention: String = ResidualNormConvention.current
    /// WHICH RENDERING produced `residualNormPerLayer`. The denominator
    /// follows the extraction's rendering — α in norm units must divide by a
    /// number from the same distribution the vector was read from — and the
    /// artifact says so rather than leaving a reader to infer it. Same field
    /// family as `residualNormSource`/`residualNormConvention`; "raw" is the
    /// legacy value and stamps nothing.
    public let residualNormRendering: String
    public let options: ExtractionOptions
    /// The requested reading position AND what it resolved to, per sequence
    /// shape. nil for the legacy pair, whose resolved index its label already
    /// implies.
    public let readingPositionResolution: ReadingPositionResolutionReport?
    /// Per-layer mean of the neutral-corpus activations at the reading
    /// position — the residual stream's "carrier" estimate, persisted into
    /// the artifact (`neutral_mean_layer_<i>`) so ablation paths can center
    /// directions against it / run the mean-alignment preflight. nil when no
    /// neutral corpus was supplied.
    public let neutralMeanPerLayer: [[Float]]?

    public init(
        vectors: ConceptVectors,
        residualNormPerLayer: [Float],
        residualNormSource: String,
        options: ExtractionOptions,
        residualNormRendering: String = ExtractionRendering.Mode.raw.rawValue,
        readingPositionResolution: ReadingPositionResolutionReport? = nil,
        neutralMeanPerLayer: [[Float]]? = nil
    ) {
        self.vectors = vectors
        self.residualNormPerLayer = residualNormPerLayer
        self.residualNormSource = residualNormSource
        self.options = options
        self.residualNormRendering = residualNormRendering
        self.readingPositionResolution = readingPositionResolution
        self.neutralMeanPerLayer = neutralMeanPerLayer
    }
}

public struct MultiConceptExtractionResult: Sendable {
    public let vectorsByConcept: [String: ConceptVectors]
    public let residualNormPerLayer: [Float]
    public let residualNormSource: String
    /// See `ExtractionResult.residualNormConvention`.
    public let residualNormConvention: String = ResidualNormConvention.current
    /// See `ExtractionResult.residualNormRendering`.
    public let residualNormRendering: String
    public let readingPosition: ReadingPosition
    /// The rendering every concept in this pass was extracted under (one
    /// corpus, one rendering).
    public let extractionRendering: ExtractionRendering
    /// See `ExtractionResult.readingPositionResolution` — shared by every
    /// concept in the pass.
    public let readingPositionResolution: ReadingPositionResolutionReport?
    public let neutralPCCount: Int?
    public let screening: ExtractionScreening
    public let neutralScreening: ExtractionScreening?
    public let neutralProjectionDescription: String
    public let method: VectorExtractionRecipe.Method
    /// See `ExtractionResult.neutralMeanPerLayer` — shared by every concept
    /// in the pass (one corpus, one carrier estimate). Sourced from the
    /// pooled neutral activations, or from the token bank's rows when the
    /// bank path ran instead. nil when no neutral corpus was supplied.
    public let neutralMeanPerLayer: [[Float]]?
}

public struct ExtractionScreening: Codable, Sendable, Equatable {
    public let readingPosition: String
    public let minimumTokenCount: Int
    public let sourceCount: Int
    public let includedCount: Int
    public let excludedShortCount: Int

    public init(
        readingPosition: ReadingPosition,
        sourceCount: Int,
        includedCount: Int,
        excludedShortCount: Int
    ) {
        self.readingPosition = readingPosition.label
        self.minimumTokenCount = readingPosition.minimumTokenCount
        self.sourceCount = sourceCount
        self.includedCount = includedCount
        self.excludedShortCount = excludedShortCount
    }
}

/// Per-stimulus activations from prefill-only forward passes (no sampling).
///
/// Text reaches the model under the pass's declared ``ExtractionRendering``:
/// raw (the legacy default — no chat template) or the family template.
public struct StimulusActivations: Sendable {
    /// `values[textIndex][layer]` is the pooled hidden vector.
    public let values: [[[Float]]]
    /// Mean residual norm per layer, averaged over stimuli.
    public let residualNormPerLayer: [Float]
    /// What was actually read, per stimulus — the provenance half of the
    /// reading-position stamp.
    public let resolutions: [ResolvedReadingPosition]

    public init(
        values: [[[Float]]], residualNormPerLayer: [Float],
        resolutions: [ResolvedReadingPosition] = []
    ) {
        self.values = values
        self.residualNormPerLayer = residualNormPerLayer
        self.resolutions = resolutions
    }
}

public struct NeutralActivationBank: Sendable {
    public let layers: [Int]
    public let rowsByLayer: [[[Float]]]
    public let residualNormPerLayer: [Float]
    public let screening: ExtractionScreening
    public let tokenRowCount: Int
    /// Token rows per layer BEFORE deterministic downsampling.
    public let sourceRowsPerLayer: Int
    /// Token rows per layer actually kept for PCA (≤ the row cap).
    public let usedRowsPerLayer: Int
    /// Seed of the deterministic row draw (derived from the corpus hash);
    /// nil only on pre-downsampling banks.
    public let downsampleSeed: UInt64?

    /// Per-layer nuisance components. Both selection kinds are clamped by
    /// `selection.resolvedMaximumCount` (default 64): the variance TARGET
    /// still governs below the cap, but it can no longer run away to `rows − 1`
    /// components per layer on a flat spectrum.
    public func componentsByLayer(
        selection: VectorExtractionRecipe.NeutralPCSelection
    ) throws -> [[[Float]]] {
        let maximum = selection.resolvedMaximumCount
        switch selection.kind {
        case .none:
            return layers.map { _ in [] }
        case .fixedCount:
            let count = min(selection.count ?? 0, maximum)
            guard count > 0 else { return layers.map { _ in [] } }
            return try rowsByLayer.map {
                try SteeringVectorMath.principalComponents(of: $0, count: count)
            }
        case .explainedVariance:
            let fraction = Float(selection.minimumExplainedVariance ?? 0)
            guard fraction > 0 else { return layers.map { _ in [] } }
            return try rowsByLayer.map {
                try SteeringVectorMath.principalComponents(
                    of: $0, minimumExplainedVariance: fraction, maximumCount: maximum
                ).components
            }
        }
    }
}

public struct LogitLensToken: Codable, Sendable, Equatable {
    public let tokenID: Int
    public let token: String
    public let logit: Float
}

public struct LogitLensReport: Codable, Sendable, Equatable {
    public let layer: Int
    public let topPositive: [LogitLensToken]
    public let topNegative: [LogitLensToken]
}

private struct ScreenedTexts<T: Sendable>: Sendable {
    let included: [T]
    let report: ExtractionScreening
}

/// Contrastive activation extraction.
public enum ConceptExtractor {
    public static let defaultMaximumShortTextExclusionFraction = 0.10

    /// Activations for each text at every block's output, read at the given
    /// position and under the given rendering.
    ///
    /// `rendering` defaults to `.raw` — the legacy behavior, byte for byte:
    /// the bare stimulus through `tokenizer.encode(text:)`. A chat-template
    /// rendering goes through ``PromptRendering``, the same definition
    /// measured generation uses, so extraction and generation cannot drift.
    ///
    /// Each position is RESOLVED against the (rendered) token ids before the
    /// pass and its window pinned on the recorder, so template-aware roles
    /// land on a concrete index and the resolution is available for stamping.
    public static func activations(
        container: ModelContainer, texts: [String],
        position: ReadingPosition = .lastToken,
        rendering: ExtractionRendering = .raw
    ) async throws -> StimulusActivations {
        try await container.perform { context in
            guard let hookable = context.model as? InterventionHookable else {
                throw ConceptExtractorError.modelNotHookable("\(type(of: context.model))")
            }
            let recorder = ActivationRecorder(layers: 0 ..< 4096, position: position)
            hookable.interventions = [recorder]
            defer { hookable.interventions = [] }

            var results: [[[Float]]] = []
            var normSums: [Float] = []
            var resolutions: [ResolvedReadingPosition] = []
            results.reserveCapacity(texts.count)

            for text in texts {
                recorder.reset()
                let tokens = try tokenIDs(
                    context: context, text: text, rendering: rendering)
                guard tokens.count >= position.minimumTokenCount else {
                    throw ConceptExtractorError.stimulusTooShort(
                        text: text,
                        tokenCount: tokens.count,
                        minimumTokenCount: position.minimumTokenCount,
                        reading: position.label)
                }
                let resolved: ResolvedReadingPosition
                do {
                    resolved = try position.resolve(
                        tokens: tokens, tokenizer: context.tokenizer,
                        renderingIsRaw: rendering.isRaw)
                } catch let error as ReadingPositionError {
                    throw ConceptExtractorError.readingPositionUnresolved(
                        reason: error.reason, text: text)
                }
                resolutions.append(resolved)
                recorder.setWindow(
                    start: resolved.startIndex, end: resolved.endIndex,
                    indices: resolved.pooledIndices)

                let input = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
                let logits = context.model(input, cache: nil)
                eval(logits)

                let captures = recorder.captures.sorted { $0.layer < $1.layer }
                guard !captures.isEmpty else {
                    throw ConceptExtractorError.missingCaptures(text: text)
                }
                results.append(captures.map(\.values))
                if normSums.isEmpty {
                    normSums = captures.map(\.residualNorm)
                } else if normSums.count == captures.count {
                    for (index, capture) in captures.enumerated() {
                        normSums[index] += capture.residualNorm
                    }
                }
            }
            let count = Float(texts.count)
            return StimulusActivations(
                values: results,
                residualNormPerLayer: normSums.map { $0 / count },
                resolutions: resolutions)
        }
    }

    /// Token ids for one stimulus under `rendering`, with the engine-named
    /// refusal when a declared rendering cannot be applied here. Never falls
    /// back to raw: a silent fallback is the exact ambiguity the
    /// `extractionRendering` option exists to end.
    static func tokenIDs(
        context: ModelContext, text: String, rendering: ExtractionRendering
    ) throws -> [Int] {
        guard !rendering.isRaw else { return context.tokenizer.encode(text: text) }
        do {
            return try PromptRendering.tokenIDs(
                tokenizer: context.tokenizer,
                modelID: context.configuration.name,
                text: text, rendering: rendering)
        } catch {
            throw ConceptExtractorError.renderingUnsupported(
                rendering: rendering.label,
                modelID: context.configuration.name,
                reason: "\(error)")
        }
    }

    /// Back-compat convenience: last-token activations only.
    public static func lastTokenActivations(
        container: ModelContainer, texts: [String]
    ) async throws -> [[[Float]]] {
        try await activations(container: container, texts: texts).values
    }

    /// The loaded model's residual-stream shape (block count, hidden size),
    /// read from its decoded config — no forward pass. Callers use it to size
    /// capture work and to compute layer bands.
    public static func residualShape(
        container: ModelContainer
    ) async throws -> (blockCount: Int, hiddenSize: Int) {
        try await container.perform { context in
            guard let shaped = context.model as? ResidualShapeProviding else {
                throw ConceptExtractorError.residualShapeUnavailable("\(type(of: context.model))")
            }
            return (shaped.residualBlockCount, shaped.residualHiddenSize)
        }
    }

    /// Records the neutral token bank for basis PCA.
    ///
    /// **The row cap is enforced during ingestion, not after it** (fixed
    /// 2026-08-18). Token counts are known before any forward pass, so the
    /// deterministic draw — `TokenBankDownsampler.selectedIndices`, the same
    /// seeded partial Fisher–Yates the two-phase code used — is computed up
    /// front and handed to the recorder as a membership set. Rows outside the
    /// draw are never copied off the GPU, so the transient is bounded by
    /// `cap × layers × hidden` instead of `corpusTokens × layers × hidden`.
    /// The selected rows are IDENTICAL to what the old collect-then-downsample
    /// path produced for the same (corpus, cap, seed), so existing bases stay
    /// comparable.
    ///
    /// Capture is restricted to `layers` when given — pass a band (the middle
    /// third is the convention); every banked row is an eager GPU→CPU float32
    /// copy at every captured layer. The projected cost is checked against
    /// `memoryBudgetBytes` (default: a quarter of physical memory) and refused
    /// with `ConceptExtractorError.neutralBankTooLarge` before capture starts,
    /// because an over-budget bank dies inside native allocation where no
    /// Swift `catch` can run.
    ///
    /// Residual norms average over ALL corpus positions, banked or not (norms
    /// are one float per token; the cap protects the O(rows²) PCA and the row
    /// store, not the norm estimate).
    public static func neutralActivationBank(
        container: ModelContainer,
        texts: [String],
        readingPosition: ReadingPosition = .meanFromToken(50),
        layers requestedLayers: Set<Int>? = nil,
        maximumShortTextExclusionFraction: Double = defaultMaximumShortTextExclusionFraction,
        maxRowsPerLayer: Int = TokenBankDownsampler.defaultRowCapPerLayer,
        downsampleSeed: UInt64? = nil,
        memoryBudgetBytes: Int? = nil,
        rendering: ExtractionRendering = .raw,
        log: (@Sendable (String) -> Void)? = nil
    ) async throws -> NeutralActivationBank {
        // Denominator-rendering consistency: the bank IS the norm denominator
        // on the token-bank path, so it is tokenized exactly as the extraction
        // it serves. Absent/raw is the historical behavior, unchanged.
        let screened = try await screenTexts(
            container: container,
            items: texts,
            text: { $0 },
            readingPosition: readingPosition,
            maximumShortTextExclusionFraction: maximumShortTextExclusionFraction,
            rendering: rendering)
        let startIndex = readingPosition.requestedStartIndex ?? 0
        // Seed precedence: caller-supplied (derived from the pinned corpus
        // FILE hash) wins; otherwise derive from the screened texts so the
        // draw is still deterministic for identical content.
        let seed =
            downsampleSeed
            ?? TokenBankDownsampler.seed(
                fromCorpusHash: TokenBankDownsampler.corpusHash(texts: screened.included))

        return try await container.perform { context in
            guard let hookable = context.model as? InterventionHookable else {
                throw ConceptExtractorError.modelNotHookable("\(type(of: context.model))")
            }
            guard let shaped = context.model as? ResidualShapeProviding else {
                throw ConceptExtractorError.residualShapeUnavailable("\(type(of: context.model))")
            }
            let blockCount = shaped.residualBlockCount
            let hiddenSize = shaped.residualHiddenSize
            let captureLayers = (requestedLayers ?? Set(0 ..< blockCount))
                .intersection(Set(0 ..< blockCount))

            // Tokenize once, up front: this both sizes the bank and supplies
            // the tokens the forward passes use (the pre-fix code tokenized
            // inside the loop, so this costs nothing extra).
            let tokenized = try screened.included.map {
                try tokenIDs(context: context, text: $0, rendering: rendering)
            }
            let rowsPerText = tokenized.map { max(0, $0.count - startIndex) }
            let totalRows = rowsPerText.reduce(0, +)

            let preflight = NeutralBankBudget.preflight(
                sourceRowsPerLayer: totalRows,
                rowCapPerLayer: maxRowsPerLayer,
                layerCount: captureLayers.count,
                hiddenSize: hiddenSize,
                budgetBytes: memoryBudgetBytes)
            log?(
                "neutral token bank preflight: \(preflight.summary) · "
                    + "\(LayerBand.description(of: captureLayers, blockCount: blockCount)) · "
                    + "\(totalRows) source rows/layer")
            guard preflight.fits else {
                throw ConceptExtractorError.neutralBankTooLarge(preflight)
            }

            // The draw, computed BEFORE capture so over-cap rows are never
            // materialized. Same rule, same rows as the old post-hoc pass.
            let selected = TokenBankDownsampler.selectedIndexSet(
                count: totalRows, cap: maxRowsPerLayer, seed: seed)

            let recorder = ActivationBankRecorder(
                layers: captureLayers,
                startIndex: startIndex,
                selectedRowIndices: selected)
            hookable.interventions = [recorder]
            defer { hookable.interventions = [] }

            var rowsByLayer: [Int: [[Float]]] = [:]
            // WHOLE-CORPUS denominator convention (researcher ruling
            // 2026-08-20, `ResidualNormConvention`): every measured position
            // feeds the tally, banked or not.
            var tally = ResidualNormConvention.Tally()
            for (tokens, textRows) in zip(tokenized, rowsPerText) where textRows > 0 {
                recorder.reset()
                let input = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
                let logits = context.model(input, cache: nil)
                eval(logits)

                for row in recorder.rows {
                    rowsByLayer[row.layer, default: []].append(row.values)
                    tally.add(layer: row.layer, norm: row.residualNorm)
                }
                // Positions the cap excluded still count toward the residual-
                // norm denominator, which describes the corpus, not the draw.
                for skipped in recorder.skippedNorms {
                    tally.add(layer: skipped.layer, norm: skipped.residualNorm)
                }
            }

            let layers = rowsByLayer.keys.sorted()
            let keptRowsByLayer = layers.map { rowsByLayer[$0] ?? [] }
            let usedRowsPerLayer = keptRowsByLayer.map(\.count).max() ?? 0
            return NeutralActivationBank(
                layers: layers,
                rowsByLayer: keptRowsByLayer,
                residualNormPerLayer: layers.map { tally.mean(at: $0) },
                screening: screened.report,
                tokenRowCount: tally.totalCount,
                sourceRowsPerLayer: totalRows,
                usedRowsPerLayer: usedRowsPerLayer,
                downsampleSeed: seed)
        }
    }

    /// Typical residual-stream norms on a neutral corpus — the norm-unit
    /// denominator measurement `extract`/`extractGrandMean` perform when a
    /// neutral corpus is supplied: screen too-short texts non-fatally
    /// (grand-mean neutral-branch convention), then run the standard
    /// `activations` pass at the reading position and average the per-layer
    /// residual norms. Exposed so the residual-norm backfill job measures
    /// through exactly this code path instead of re-deriving the math.
    public static func neutralCorpusResidualNorms(
        container: ModelContainer,
        texts: [String],
        position: ReadingPosition,
        maximumShortTextExclusionFraction: Double = defaultMaximumShortTextExclusionFraction,
        rendering: ExtractionRendering = .raw
    ) async throws -> (residualNormPerLayer: [Float], screening: ExtractionScreening) {
        let screened = try await screenTexts(
            container: container,
            items: texts,
            text: { $0 },
            readingPosition: position,
            maximumShortTextExclusionFraction: maximumShortTextExclusionFraction,
            enforceMaximumShortTextExclusion: false,
            rendering: rendering)
        // Same minimum `extract` enforces before it accepts a neutral corpus
        // as the denominator source.
        guard screened.included.count >= 4 else {
            throw ConceptExtractorError.missingCaptures(
                text: "neutral corpus needs at least 4 usable texts for "
                    + "\(position.label) norm measurement")
        }
        let measured = try await activations(
            container: container, texts: screened.included, position: position,
            rendering: rendering)
        return (measured.residualNormPerLayer, screened.report)
    }

    public static func logitLens(
        container: ModelContainer,
        vectors: ConceptVectors,
        layer: Int,
        topK: Int = 10
    ) async throws -> LogitLensReport {
        guard vectors.perLayer.indices.contains(layer) else {
            throw ConceptExtractorError.layerMismatch
        }
        return try await container.perform { context in
            guard let readable = context.model as? LogitLensReadable else {
                throw ConceptExtractorError.modelNotHookable("\(type(of: context.model))")
            }
            let logits = readable.logitsForResidualVector(vectors.perLayer[layer])
            func ranked(_ descending: Bool) -> [LogitLensToken] {
                logits.indices.sorted {
                    descending ? logits[$0] > logits[$1] : logits[$0] < logits[$1]
                }
                .prefix(max(0, topK))
                .map { tokenID in
                    LogitLensToken(
                        tokenID: tokenID,
                        token: context.tokenizer.decode(
                            tokenIds: [tokenID], skipSpecialTokens: false),
                        logit: logits[tokenID])
                }
            }
            return LogitLensReport(
                layer: layer,
                topPositive: ranked(true),
                topNegative: ranked(false))
        }
    }

    /// Extracts per-layer concept directions from a stimulus set with the
    /// given method and reading position. When `options.neutralPCCount > 0`,
    /// `neutralTexts` must be provided; their top principal components are
    /// projected out of every layer's direction.
    public static func extract(
        container: ModelContainer, stimuli: StimulusSet,
        options: ExtractionOptions = ExtractionOptions(),
        neutralTexts: [String]? = nil
    ) async throws -> ExtractionResult {
        let rendering = options.resolvedExtractionRendering
        let positive = try await activations(
            container: container, texts: stimuli.positive,
            position: options.readingPosition, rendering: rendering)
        let negative = try await activations(
            container: container, texts: stimuli.negative,
            position: options.readingPosition, rendering: rendering)

        guard let layerCount = positive.values.first?.count,
            positive.values.allSatisfy({ $0.count == layerCount }),
            negative.values.allSatisfy({ $0.count == layerCount })
        else {
            throw ConceptExtractorError.layerMismatch
        }

        // Neutral-corpus activations serve two roles when a corpus is
        // provided: (a) the norm-unit denominator (the emotion paper
        // denominates α against a fixed dataset, not the concept's own
        // stimuli — otherwise α=0.1 for fear and authority are different
        // units), and (b) nuisance PCs to project out when requested.
        var neutralActivations: StimulusActivations?
        let pcCount = options.neutralPCCount ?? 0
        if let neutralTexts, neutralTexts.count >= 4 {
            // DENOMINATOR-RENDERING CONSISTENCY: α is reported in
            // residual-norm units, so the norms must be measured on the same
            // DISTRIBUTION the vector was read from. Measuring a
            // chat-template vector against a raw-tokenized denominator
            // divides by a number from a different distribution, so the
            // neutral corpus follows the extraction's rendering, always.
            neutralActivations = try await activations(
                container: container, texts: neutralTexts,
                position: options.readingPosition, rendering: rendering)
        } else if pcCount > 0 {
            throw ConceptExtractorError.missingCaptures(
                text: "neutral corpus required for confound projection")
        }

        var neutralComponents: [[[Float]]] = []
        if pcCount > 0, let neutralActivations {
            for layer in 0 ..< layerCount {
                neutralComponents.append(
                    try SteeringVectorMath.principalComponents(
                        of: neutralActivations.values.map { $0[layer] }, count: pcCount))
            }
        }

        var perLayer: [[Float]] = []
        perLayer.reserveCapacity(layerCount)
        for layer in 0 ..< layerCount {
            var vector = try SteeringVectorMath.direction(
                positive: positive.values.map { $0[layer] },
                negative: negative.values.map { $0[layer] },
                method: options.method)
            if !neutralComponents.isEmpty {
                vector = SteeringVectorMath.projectingOut(
                    vector, components: neutralComponents[layer])
            }
            perLayer.append(vector)
        }

        let residualNorms: [Float]
        let normSource: String
        if let neutralActivations {
            residualNorms = neutralActivations.residualNormPerLayer
            normSource = "neutral-corpus"
        } else {
            residualNorms = zip(
                positive.residualNormPerLayer, negative.residualNormPerLayer
            ).map { ($0 + $1) / 2 }
            normSource = "extraction-stimuli"
        }

        return ExtractionResult(
            vectors: ConceptVectors(perLayer: perLayer),
            residualNormPerLayer: residualNorms,
            residualNormSource: normSource,
            options: options,
            residualNormRendering: rendering.mode.rawValue,
            readingPositionResolution: ReadingPositionResolutionReport.make(
                position: options.readingPosition, rendering: rendering,
                resolutions: positive.resolutions + negative.resolutions),
            neutralMeanPerLayer: neutralActivations.map {
                neutralMeanPerLayer(of: $0, layerCount: layerCount)
            })
    }

    /// Per-layer mean of the neutral rows already captured for the norm
    /// denominator — no extra forward passes. This is the carrier estimate
    /// the ablation paths center against; the same rows that define the
    /// norm-unit denominator define the mean, so one pinned corpus governs
    /// both (server twin: `extractor._neutral_mean_per_layer`).
    static func neutralMeanPerLayer(
        of neutral: StimulusActivations, layerCount: Int
    ) -> [[Float]] {
        (0 ..< layerCount).map { layer in
            let rows = neutral.values.map { $0[layer] }
            guard let dimension = rows.first?.count, !rows.isEmpty else { return [] }
            var sums = [Double](repeating: 0, count: dimension)
            for row in rows {
                for d in 0 ..< dimension { sums[d] += Double(row[d]) }
            }
            let count = Double(rows.count)
            return sums.map { Float($0 / count) }
        }
    }

    /// Emotion-paper-style multi-concept extraction. Each concept direction
    /// is mean(concept rows) - mean(all rows) at every layer. The neutral
    /// corpus, when supplied, remains the norm denominator and optional PC
    /// nuisance basis; it is not the negative class.
    public static func extractGrandMean(
        container: ModelContainer,
        corpus: [StimulusSet.MultiConceptStimulus],
        targetConcepts: Set<String>? = nil,
        readingPosition: ReadingPosition = .meanFromToken(50),
        neutralPCCount: Int? = nil,
        neutralTexts: [String]? = nil,
        neutralPCSelection: VectorExtractionRecipe.NeutralPCSelection = .none,
        maximumShortTextExclusionFraction: Double = defaultMaximumShortTextExclusionFraction,
        extractionRendering: ExtractionRendering = .raw
    ) async throws -> MultiConceptExtractionResult {
        guard !corpus.isEmpty else {
            throw ConceptExtractorError.missingCaptures(text: "empty multi-concept corpus")
        }
        let rendering = extractionRendering
        let screenedCorpus = try await screenTexts(
            container: container,
            items: corpus,
            text: { $0.text },
            readingPosition: readingPosition,
            maximumShortTextExclusionFraction: maximumShortTextExclusionFraction,
            rendering: rendering)
        let corpusActivations = try await activations(
            container: container,
            texts: screenedCorpus.included.map(\.text),
            position: readingPosition, rendering: rendering)
        guard let layerCount = corpusActivations.values.first?.count,
            corpusActivations.values.allSatisfy({ $0.count == layerCount })
        else {
            throw ConceptExtractorError.layerMismatch
        }

        var neutralActivations: StimulusActivations?
        var neutralBank: NeutralActivationBank?
        var neutralScreening: ExtractionScreening?
        var neutralProjectionDescription = "none"
        let pcCount = neutralPCCount ?? 0
        if let neutralTexts, neutralTexts.count >= 4, neutralPCSelection.kind != .none {
            neutralBank = try await neutralActivationBank(
                container: container,
                texts: neutralTexts,
                readingPosition: readingPosition,
                layers: Set(0 ..< layerCount),
                maximumShortTextExclusionFraction: maximumShortTextExclusionFraction,
                rendering: rendering)
            neutralScreening = neutralBank?.screening
        } else if let neutralTexts, neutralTexts.count >= 4 {
            let screenedNeutral = try await screenTexts(
                container: container,
                items: neutralTexts,
                text: { $0 },
                readingPosition: readingPosition,
                maximumShortTextExclusionFraction: maximumShortTextExclusionFraction,
                enforceMaximumShortTextExclusion: false,
                rendering: rendering)
            neutralScreening = screenedNeutral.report
            if !screenedNeutral.included.isEmpty {
                // Denominator follows the extraction's rendering — see
                // `extract`.
                neutralActivations = try await activations(
                    container: container,
                    texts: screenedNeutral.included,
                    position: readingPosition, rendering: rendering)
            }
        } else if pcCount > 0 {
            throw ConceptExtractorError.missingCaptures(
                text: "neutral corpus required for confound projection")
        }

        var neutralComponents: [[[Float]]] = []
        if let neutralBank, neutralPCSelection.kind != .none {
            let bankComponents = try neutralBank.componentsByLayer(selection: neutralPCSelection)
            var byActualLayer = Dictionary(
                uniqueKeysWithValues: zip(neutralBank.layers, bankComponents))
            neutralComponents = (0 ..< layerCount).map { byActualLayer.removeValue(forKey: $0) ?? [] }
            switch neutralPCSelection.kind {
            case .none:
                neutralProjectionDescription = "none"
            case .fixedCount:
                neutralProjectionDescription =
                    "token-bank fixed-count \(neutralPCSelection.count ?? 0) PCs"
            case .explainedVariance:
                let fraction = neutralPCSelection.minimumExplainedVariance ?? 0
                neutralProjectionDescription =
                    "token-bank explained-variance \(fraction)"
            }
        } else if pcCount > 0, let neutralActivations {
            for layer in 0 ..< layerCount {
                neutralComponents.append(
                    try SteeringVectorMath.principalComponents(
                        of: neutralActivations.values.map { $0[layer] }, count: pcCount))
            }
            neutralProjectionDescription = "legacy-pooled top-\(pcCount) neutral PCs"
        }

        let conceptSet = targetConcepts ?? Set(corpus.map(\.concept))
        var indexesByConcept: [String: [Int]] = [:]
        for (index, row) in screenedCorpus.included.enumerated() where conceptSet.contains(row.concept) {
            indexesByConcept[row.concept, default: []].append(index)
        }

        var vectorsByConcept: [String: ConceptVectors] = [:]
        for concept in indexesByConcept.keys.sorted() {
            guard let indexes = indexesByConcept[concept], !indexes.isEmpty else { continue }
            var perLayer: [[Float]] = []
            perLayer.reserveCapacity(layerCount)
            for layer in 0 ..< layerCount {
                let conceptRows = indexes.map { corpusActivations.values[$0][layer] }
                let populationRows = corpusActivations.values.map { $0[layer] }
                var vector = try SteeringVectorMath.grandMeanDifference(
                    concept: conceptRows, population: populationRows)
                if !neutralComponents.isEmpty {
                    vector = SteeringVectorMath.projectingOut(
                        vector, components: neutralComponents[layer])
                }
                perLayer.append(vector)
            }
            vectorsByConcept[concept] = ConceptVectors(perLayer: perLayer)
        }

        let residualNorms: [Float]
        let normSource: String
        if let neutralActivations {
            residualNorms = neutralActivations.residualNormPerLayer
            normSource = "neutral-corpus"
        } else if let neutralBank {
            residualNorms = neutralBank.residualNormPerLayer
            normSource = "neutral-token-bank"
        } else {
            residualNorms = corpusActivations.residualNormPerLayer
            normSource = "multi-concept-corpus"
        }

        let neutralMean: [[Float]]?
        if let neutralActivations {
            neutralMean = neutralMeanPerLayer(of: neutralActivations, layerCount: layerCount)
        } else if let neutralBank {
            // Token-bank path: mean over the banked token rows, mapped back
            // to dense layer indexing (empty rows for unbanked layers).
            var meansByLayer = [Int: [Float]]()
            for (layer, rows) in zip(neutralBank.layers, neutralBank.rowsByLayer) {
                guard let dimension = rows.first?.count, !rows.isEmpty else { continue }
                var sums = [Double](repeating: 0, count: dimension)
                for row in rows {
                    for d in 0 ..< dimension { sums[d] += Double(row[d]) }
                }
                let count = Double(rows.count)
                meansByLayer[layer] = sums.map { Float($0 / count) }
            }
            neutralMean = (0 ..< layerCount).map { meansByLayer[$0] ?? [] }
        } else {
            neutralMean = nil
        }

        return MultiConceptExtractionResult(
            vectorsByConcept: vectorsByConcept,
            residualNormPerLayer: residualNorms,
            residualNormSource: normSource,
            residualNormRendering: rendering.mode.rawValue,
            readingPosition: readingPosition,
            extractionRendering: rendering,
            readingPositionResolution: ReadingPositionResolutionReport.make(
                position: readingPosition, rendering: rendering,
                resolutions: corpusActivations.resolutions),
            neutralPCCount: neutralPCCount,
            screening: screenedCorpus.report,
            neutralScreening: neutralScreening,
            neutralProjectionDescription: neutralProjectionDescription,
            method: .emotionGrandMean,
            neutralMeanPerLayer: neutralMean)
    }

    /// Screened per-row activations for a multi-concept corpus at one
    /// reading position — the corpus-side inputs for grand-mean validation
    /// scoring (concept-mean vs population-mean projections). Screening
    /// mirrors `extractGrandMean` but never fails on the exclusion fraction:
    /// validation reads whatever rows the position can pool.
    public static func multiConceptActivations(
        container: ModelContainer,
        corpus: [StimulusSet.MultiConceptStimulus],
        readingPosition: ReadingPosition = .meanFromToken(50),
        rendering: ExtractionRendering = .raw
    ) async throws -> (concepts: [String], activations: StimulusActivations) {
        let screened = try await screenTexts(
            container: container,
            items: corpus,
            text: { $0.text },
            readingPosition: readingPosition,
            maximumShortTextExclusionFraction: 1,
            enforceMaximumShortTextExclusion: false,
            rendering: rendering)
        let values = try await activations(
            container: container,
            texts: screened.included.map(\.text),
            position: readingPosition, rendering: rendering)
        return (screened.included.map(\.concept), values)
    }

    private static func screenTexts<T: Sendable>(
        container: ModelContainer,
        items: [T],
        text: @Sendable @escaping (T) -> String,
        readingPosition: ReadingPosition,
        maximumShortTextExclusionFraction: Double,
        enforceMaximumShortTextExclusion: Bool = true,
        rendering: ExtractionRendering = .raw
    ) async throws -> ScreenedTexts<T> {
        let minimum = readingPosition.minimumTokenCount
        // Screening counts tokens under the SAME rendering extraction will
        // use — a templated render adds the template's own tokens, so
        // screening the raw string would answer a question about a sequence
        // the model never sees.
        let included = try await container.perform { context in
            try items.filter { item in
                try tokenIDs(context: context, text: text(item), rendering: rendering)
                    .count >= minimum
            }
        }
        let excluded = items.count - included.count
        if excluded > 0, !items.isEmpty {
            let fraction = Double(excluded) / Double(items.count)
            if enforceMaximumShortTextExclusion,
                fraction > maximumShortTextExclusionFraction
            {
                throw ConceptExtractorError.tooManyShortStimuli(
                    excluded: excluded,
                    total: items.count,
                    maximumFraction: maximumShortTextExclusionFraction)
            }
        }
        return ScreenedTexts(
            included: included,
            report: ExtractionScreening(
                readingPosition: readingPosition,
                sourceCount: items.count,
                includedCount: included.count,
                excludedShortCount: excluded))
    }
}

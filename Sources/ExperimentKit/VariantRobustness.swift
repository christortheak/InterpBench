import Foundation
import MLXLMCommon
import SteeringKit

public struct VariantRobustnessReport: Codable, Sendable, Equatable {
    public struct BatteryItem: Codable, Sendable, Equatable {
        public let index: Int
        public let prompt: String
        public let answer: String
        public let baselineResponse: String
        public let variantResponse: String
        public let baselineCorrect: Bool
        public let variantCorrect: Bool
    }

    public struct CoherenceItem: Codable, Sendable, Equatable {
        public let index: Int
        public let prompt: String
        public let baselineResponse: String
        public let variantResponse: String
        public let baselineWordCount: Int
        public let variantWordCount: Int
        public let baselineDistinct2: Float
        public let variantDistinct2: Float
        public let judge: PairedJudgeResponse?
        public let judgeResult: String?
    }

    public let variantName: String
    public let variantArtifactPath: String?
    public let variantArtifactHash: String?
    public let baseModelID: String
    /// Engine that ran the generations: `RepEReader.substrate` ("swift-mlx")
    /// for the local container path, `WorkspaceScoping.serverSubstrate`
    /// ("python-hf-transformers") for the server path. Optional so reports
    /// written before the stamp still decode.
    public let substrate: String?
    public let presetID: String?
    public let batteryFile: String
    public let coherencePromptsFile: String
    public let judgeModel: String?
    public let generatedAt: String
    public let baselineBatteryAccuracy: Float
    public let variantBatteryAccuracy: Float
    public let meanBaselineDistinct2: Float
    public let meanVariantDistinct2: Float
    public let meanBaselineWords: Float
    public let meanVariantWords: Float
    public let batteryItems: [BatteryItem]
    public let coherenceItems: [CoherenceItem]
    public let warnings: [String]
    /// The battery's declared format and whether its arming came from the
    /// battery itself — the same two stamps a `battery.jsonl` row carries
    /// (server `batteryFormat` / `armingIsolated`), so a robustness report
    /// says on its face whether its two accuracies are instrument-comparable.
    /// Absent — never null — on a legacy (format-1) reading and on reports
    /// written before the stamp, exactly as the record rows are.
    public var batteryFormat: Int?
    public var armingIsolated: Bool?
    /// Which BACKEND judged — `local`, `claude`, or `openrouter`: the
    /// manifest's own kind vocabulary. A report used to name only a model
    /// string, which said nothing about who was actually called; with three
    /// backends reachable from this pane, the kind IS the provenance.
    /// Optional so reports written before the stamp still decode, and absent
    /// — never null — when nothing judged.
    public var judgeKind: String?
    /// The pinned serving provider of an `openrouter` judge. Absent for every
    /// other kind, exactly as `JudgeRef.provider` is.
    public var judgeProvider: String?
}

public enum VariantRobustnessProgress: Sendable {
    case outputStarted(kind: String, index: Int, total: Int, prompt: String, side: String)
    case outputChunk(kind: String, index: Int, side: String, output: String)
    case outputCompleted(kind: String, index: Int, prompt: String, side: String, output: String)
    case judgeStarted(index: Int, total: Int, prompt: String)
    case judgeCompleted(index: Int, prompt: String, result: String, response: PairedJudgeResponse)
}

public typealias VariantRobustnessProgressHandler = @Sendable (VariantRobustnessProgress) async -> Void

public enum VariantRobustness {
    /// Canonical report filename inside a robustness run directory — the
    /// contract `AgentEvidence.scanRobustnessReports` discovers reports by.
    public static let reportFileName = "robustness-report.json"

    public static let defaultBatteryFile = "prompts/batteries/basic.jsonl"
    public static let defaultCoherencePromptsFile = "prompts/dev/dev-prompts.jsonl"
    public static let customPresetID = "custom"

    public struct Preset: Identifiable, Sendable, Equatable {
        public let id: String
        public let label: String
        public let description: String
        public let batteryFile: String
        public let coherencePromptsFile: String
        public let maxCoherencePrompts: Int
        public let maxTokens: Int
    }

    public static let presets: [Preset] = [
        .init(
            id: "quick",
            label: "Quick Guardrail",
            description: "Fast deterministic capability checks plus a few open-ended coherence prompts.",
            batteryFile: defaultBatteryFile,
            coherencePromptsFile: defaultCoherencePromptsFile,
            maxCoherencePrompts: 3,
            maxTokens: 256),
        .init(
            id: "study",
            label: "Study Guardrail",
            description: "Broader arithmetic, factual, instruction-following, and truthfulness checks for reportable study runs.",
            batteryFile: "prompts/batteries/study-guardrail.jsonl",
            coherencePromptsFile: "prompts/dev/robustness-coherence.jsonl",
            maxCoherencePrompts: 6,
            maxTokens: 320),
        .init(
            id: "instruction",
            label: "Instruction Following",
            description: "IFEval-style tiny slice with exact or regex-verifiable formatting constraints.",
            batteryFile: "prompts/batteries/instruction-following.jsonl",
            coherencePromptsFile: "prompts/dev/robustness-coherence.jsonl",
            maxCoherencePrompts: 4,
            maxTokens: 240),
        .init(
            id: "factual",
            label: "Factual Short",
            description: "SimpleQA-style short-answer factual recall with deterministic grading.",
            batteryFile: "prompts/batteries/factual-short.jsonl",
            coherencePromptsFile: "prompts/dev/robustness-coherence.jsonl",
            maxCoherencePrompts: 4,
            maxTokens: 240),
        .init(
            id: "reasoning",
            label: "Reasoning Small",
            description: "Compact arithmetic, symbolic, and commonsense reasoning checks.",
            batteryFile: "prompts/batteries/reasoning-small.jsonl",
            coherencePromptsFile: "prompts/dev/robustness-coherence.jsonl",
            maxCoherencePrompts: 4,
            maxTokens: 240),
        .init(
            id: "truthfulness",
            label: "Truthfulness Small",
            description: "TruthfulQA-inspired misconceptions and false-premise checks.",
            batteryFile: "prompts/batteries/truthfulness-small.jsonl",
            coherencePromptsFile: "prompts/dev/robustness-coherence.jsonl",
            maxCoherencePrompts: 4,
            maxTokens: 240),
    ]

    public static var defaultPreset: Preset { presets[0] }

    public static func preset(id: String) -> Preset? {
        presets.first { $0.id == id }
    }

    /// Deterministic capability answers are short; both substrates cap them
    /// identically so accuracies compare.
    public static let batteryMaxTokens = 24

    public static let defaultJudgeRubric = """
        You are evaluating whether a model variant remains coherent and capable.
        Prefer the response that better follows the prompt, is coherent, avoids repetition,
        and preserves basic reasoning ability. Do not reward a response merely because it
        expresses a target concept more strongly. If both responses are similarly coherent,
        mark a tie.
        """

    /// Returns nil when `shouldCancel` observes a cancellation between units
    /// of work (App gap A1): a partially generated battery/coherence pass is
    /// never scored and NO report or run directory is written — the caller
    /// reports "cancelled by user", never an error.
    @discardableResult
    public static func run(
        record: ModelVariantRecord,
        batteryFile: String = defaultBatteryFile,
        coherencePromptsFile: String = defaultCoherencePromptsFile,
        maxCoherencePrompts: Int = 3,
        maxTokens: Int = 256,
        presetID: String? = nil,
        judgeModel: String? = nil,
        judgeRubric: String = defaultJudgeRubric,
        into runDirectory: URL? = nil,
        shouldCancel: (@Sendable () async -> Bool)? = nil,
        progress: VariantRobustnessProgressHandler? = nil
    ) async throws -> (report: VariantRobustnessReport, directory: URL)? {
        let container = try await SteeredContainerLoader.load(
            modelID: record.artifact.baseModelID,
            revision: record.artifact.baseRevision)
        guard
            let report = try await evaluate(
                variant: record.artifact,
                variantPath: ModelVariantStore.relativePath(for: record),
                variantHash: try? ModelVariantStore.hash(record.url),
                container: container,
                batteryFile: batteryFile,
                coherencePromptsFile: coherencePromptsFile,
                maxCoherencePrompts: maxCoherencePrompts,
                maxTokens: maxTokens,
                presetID: presetID,
                judgeModel: judgeModel,
                judgeRubric: judgeRubric,
                shouldCancel: shouldCancel,
                progress: progress)
        else { return nil }
        let directory =
            if let runDirectory {
                runDirectory
            } else {
                try VectorCatalog.makeUniqueRunDirectory(
                    slug: "variant-robustness-\(FineTuneStore.slugify(record.artifact.name))",
                    under: VectorCatalog.runsDirectory)
            }
        try write(report, to: directory)
        return (report, directory)
    }

    /// nil = cancelled between units of work (see `run`).
    static func evaluate(
        variant: ModelVariantArtifact,
        variantPath: String?,
        variantHash: String?,
        container: ModelContainer,
        batteryFile: String = defaultBatteryFile,
        coherencePromptsFile: String = defaultCoherencePromptsFile,
        maxCoherencePrompts: Int = 3,
        maxTokens: Int = 256,
        presetID: String? = nil,
        judgeModel: String? = nil,
        judgeRubric: String = defaultJudgeRubric,
        shouldCancel: (@Sendable () async -> Bool)? = nil,
        progress: VariantRobustnessProgressHandler? = nil
    ) async throws -> VariantRobustnessReport? {
        let batteryURL = VectorCatalog.projectRoot.appending(path: batteryFile)
        let battery = try CapabilityBattery(url: batteryURL)
        let promptsURL = VectorCatalog.projectRoot.appending(path: coherencePromptsFile)
        let (promptTexts, _) = try StimulusSet.loadTexts(url: promptsURL)
        let coherencePrompts = Array(promptTexts.prefix(max(0, maxCoherencePrompts)))
        // The judge is settled BEFORE a single generation (review round 7,
        // finding 1). A LOCAL judge is loaded further down through
        // `SteeredContainerLoader.load`, which downloads what it cannot find
        // — up to 35 GB for a scale tier, invisibly, outside the named and
        // cancellable Install flow that is supposed to be the only way
        // weights arrive. Refusing here costs nothing; refusing after the
        // battery has run costs the battery.
        if case .local(let model) = JudgeModelSpelling.parse(judgeModel),
            !SteeredContainerLoader.isCached(modelID: model)
        {
            throw ExperimentError(
                reason: JudgeReadiness.notInstalledRefusal(model: model))
        }
        let promptMode = ExperimentManifest.PromptMode(rawValue: variant.promptMode) ?? .chatAssistant
        let systemPrompt = variant.systemPrompt
        let qwenThinking = variant.qwenThinkingEnabled
        let injections = try ExperimentTasks.injections(for: variant)
        // The battery's arming, not the VARIANT's: a format-2 battery is read
        // under its own declared context on BOTH sides of the comparison, so
        // the only difference between the two accuracies is the intervention.
        // A legacy battery keeps the variant's context (and says so).
        let batteryArming = battery.resolveArming(
            promptMode: promptMode, systemPrompt: systemPrompt,
            qwenThinkingEnabled: qwenThinking)
        var batteryWarnings: [String] = []
        if let advisory = battery.contaminationAdvisory(batteryArming) {
            print("WARNING: \(advisory)")
            batteryWarnings.append(advisory)
        }
        // Sweep-pattern cooperative cancellation: polled between generations
        // only (never mid-generation).
        @Sendable func cancellationObserved(_ location: String) async -> Bool {
            guard let shouldCancel, await shouldCancel() else { return false }
            print("cancellation observed at \(location)")
            return true
        }

        var baselineBatteryScores: [BatteryItemScore] = []
        for (index, item) in battery.items.enumerated() {
            if await cancellationObserved("baseline battery \(index + 1)") { return nil }
            await progress?(
                .outputStarted(
                    kind: "Capability", index: index + 1, total: battery.items.count,
                    prompt: item.prompt, side: "Baseline"))
            let backends = ExperimentTasks.batteryBackends(
                container, modelID: variant.baseModelID, injections: []
            ) { output in
                await progress?(
                    .outputChunk(
                        kind: "Capability", index: index + 1, side: "Baseline",
                        output: output))
            }
            let score = try await battery.scoreItem(
                item, arming: batteryArming,
                generate: backends.generate, choice: backends.choice)
            baselineBatteryScores.append(score)
            await progress?(
                .outputCompleted(
                    kind: "Capability", index: index + 1, prompt: item.prompt,
                    side: "Baseline", output: score.output))
        }

        var baselineCoherenceResponses: [String] = []
        for (index, prompt) in coherencePrompts.enumerated() {
            if await cancellationObserved("baseline coherence \(index + 1)") { return nil }
            await progress?(
                .outputStarted(
                    kind: "Coherence", index: index + 1, total: coherencePrompts.count,
                    prompt: prompt, side: "Baseline"))
            baselineCoherenceResponses.append(try await ExperimentTasks.generate(
                container,
                prompt: prompt,
                modelID: variant.baseModelID,
                maxTokens: maxTokens,
                temperature: 0,
                injections: [],
                promptMode: promptMode,
                systemPrompt: systemPrompt,
                qwenThinkingEnabled: qwenThinking
            ) { output in
                await progress?(
                    .outputChunk(
                        kind: "Coherence", index: index + 1, side: "Baseline",
                        output: output))
            })
            await progress?(
                .outputCompleted(
                    kind: "Coherence", index: index + 1, prompt: prompt,
                    side: "Baseline", output: baselineCoherenceResponses[index]))
        }

        let activeAdapter = try await ExperimentTasks.loadAdapter(variant, into: container)
        var variantBatteryScores: [BatteryItemScore] = []
        var variantCoherenceResponses: [String] = []
        var cancelled = false
        do {
            for (index, item) in battery.items.enumerated() {
                if await cancellationObserved("variant battery \(index + 1)") {
                    cancelled = true
                    break
                }
                await progress?(
                    .outputStarted(
                        kind: "Capability", index: index + 1, total: battery.items.count,
                        prompt: item.prompt, side: "Variant"))
                let backends = ExperimentTasks.batteryBackends(
                    container, modelID: variant.baseModelID, injections: injections
                ) { output in
                    await progress?(
                        .outputChunk(
                            kind: "Capability", index: index + 1, side: "Variant",
                            output: output))
                }
                let score = try await battery.scoreItem(
                    item, arming: batteryArming,
                    generate: backends.generate, choice: backends.choice)
                variantBatteryScores.append(score)
                await progress?(
                    .outputCompleted(
                        kind: "Capability", index: index + 1, prompt: item.prompt,
                        side: "Variant", output: score.output))
            }
            for (index, prompt) in coherencePrompts.enumerated() where !cancelled {
                if await cancellationObserved("variant coherence \(index + 1)") {
                    cancelled = true
                    break
                }
                await progress?(
                    .outputStarted(
                        kind: "Coherence", index: index + 1, total: coherencePrompts.count,
                        prompt: prompt, side: "Variant"))
                variantCoherenceResponses.append(try await ExperimentTasks.generate(
                    container,
                    prompt: prompt,
                    modelID: variant.baseModelID,
                    maxTokens: maxTokens,
                    temperature: 0,
                    injections: injections,
                    promptMode: promptMode,
                    systemPrompt: systemPrompt,
                    qwenThinkingEnabled: qwenThinking
                ) { output in
                    await progress?(
                        .outputChunk(
                            kind: "Coherence", index: index + 1, side: "Variant",
                            output: output))
                })
                await progress?(
                    .outputCompleted(
                        kind: "Coherence", index: index + 1, prompt: prompt,
                        side: "Variant", output: variantCoherenceResponses[index]))
            }
        } catch {
            if let activeAdapter {
                await ExperimentTasks.unloadAdapter(activeAdapter, from: container)
            }
            try? await ExperimentTasks.setInterventions(container, [])
            throw error
        }
        if let activeAdapter {
            await ExperimentTasks.unloadAdapter(activeAdapter, from: container)
        }
        try await ExperimentTasks.setInterventions(container, [])
        if cancelled { return nil }

        // Three-way judge dispatch, the SAME kind vocabulary the study paths
        // use (`SweepObjectives.sweepJudgeRoute`, the evaluate loop's
        // `judge.kind == "openrouter"`), reached here through the
        // single-string spelling `JudgeModelSpelling` parses. Before this,
        // the split was local-vs-Claude and the engine's third judge backend
        // was unreachable from this pane.
        let selection = JudgeModelSpelling.parse(judgeModel)
        let localJudgeContainer: ModelContainer?
        if case .local(let model) = selection {
            localJudgeContainer = try await SteeredContainerLoader.load(modelID: model)
        } else {
            localJudgeContainer = nil
        }
        let route = judgeRoute(
            selection, rubric: judgeRubric, localContainer: localJudgeContainer)
        let judge = route.judge

        return try await assembleReport(
            variant: variant,
            variantPath: variantPath,
            variantHash: variantHash,
            substrate: RepEReader.substrate,
            batteryFile: batteryFile,
            coherencePromptsFile: coherencePromptsFile,
            presetID: presetID,
            // Only a judge that actually RAN is stamped: an unpinned
            // OpenRouter selection judged nothing, and a report claiming it
            // did would be the silent-skip bug wearing a stamp.
            judgeModel: judge == nil ? nil : selection?.model,
            judgeKind: judge == nil ? nil : selection?.kind,
            judgeProvider: judge == nil ? nil : selection?.provider,
            battery: battery,
            coherencePrompts: coherencePrompts,
            baselineBatteryScores: baselineBatteryScores,
            variantBatteryScores: variantBatteryScores,
            baselineCoherenceResponses: baselineCoherenceResponses,
            variantCoherenceResponses: variantCoherenceResponses,
            judge: judge,
            extraWarnings: batteryWarnings + route.warnings,
            progress: progress)
    }

    typealias JudgeClosure = @Sendable (
        _ prompt: String, _ baseline: String, _ steered: String
    ) async throws -> PairedJudgeResponse?

    /// The ONE judge dispatch both robustness routes use.
    ///
    /// Three backends, named by the study paths' own kind vocabulary
    /// (`SweepObjectives.sweepJudgeRoute`, the evaluate loop's
    /// `judge.kind == "openrouter"`) and reached here through the
    /// single-string spelling `JudgeModelSpelling` parses. The dispatch used
    /// to be a local-vs-Claude split written twice, which left a fully
    /// implemented judge client unreachable from this pane and let the server
    /// route's skip warning swallow every spelling that was not Claude's.
    ///
    /// `localContainer` is what separates the routes: nil on the server
    /// route, because nothing is loaded on this Mac there, and that is
    /// precisely what turns a LOCAL pick into the skip warning. An OpenRouter
    /// or Claude pick is reachable from either route and is never skipped.
    ///
    /// A missing OpenRouter key is deliberately NOT pre-checked into a
    /// warning here: the client's own typed refusal ("no OpenRouter key —
    /// save one in the Compute section…") is the single message for that, on
    /// every path that calls it.
    static func judgeRoute(
        _ selection: JudgeModelSpelling.Selection?,
        rubric: String,
        localContainer: ModelContainer?,
        openRouterTransport: @escaping OpenRouterPairedJudge.Transport =
            OpenRouterPairedJudge.liveTransport
    ) -> (judge: JudgeClosure?, warnings: [String]) {
        switch selection {
        case nil:
            return (nil, [])
        case .local(let model):
            guard let localContainer else {
                // Written once, in `JudgeModelSpelling`, so the Run gate can
                // refuse up front in the route's own words instead of
                // paraphrasing them (review round 7, finding 2).
                return (
                    nil,
                    [
                        JudgeModelSpelling.localJudgeSkippedInServerWorkspace(
                            model: model)
                    ])
            }
            return (
                { prompt, baseline, steered in
                    try await LocalPairedJudge.judge(
                        container: localContainer,
                        modelID: model,
                        rubric: rubric,
                        structuredPrompt: nil,
                        prompt: prompt,
                        responseA: baseline,
                        responseB: steered)
                }, []
            )
        case .claude(let model):
            return (
                { prompt, baseline, steered in
                    try await ClaudePairedJudge.judge(
                        model: model,
                        rubric: rubric,
                        structuredPrompt: nil,
                        prompt: prompt,
                        responseA: baseline,
                        responseB: steered)
                }, []
            )
        case .openRouter(let model, let provider):
            return (
                { prompt, baseline, steered in
                    try await OpenRouterPairedJudge.judge(
                        model: model,
                        provider: provider,
                        rubric: rubric,
                        structuredPrompt: nil,
                        prompt: prompt,
                        responseA: baseline,
                        responseB: steered,
                        transport: openRouterTransport)
                }, []
            )
        case .openRouterUnpinned(let model):
            return (nil, [JudgeModelSpelling.unpinnedProviderRefusal(model: model)])
        }
    }

    /// The ONE scoring/judging/report assembly, shared by the local-container
    /// and server paths so the pure scoring code and warning rules cannot
    /// drift between substrates. Inputs are already-taken readings,
    /// index-aligned with `battery.items` / `coherencePrompts`. Battery
    /// readings arrive already scored (`BatteryItemScore`) because a
    /// format-2 choice item has no generated text to re-grade — its 0/1 is
    /// the answer-token instrument's selection.
    static func assembleReport(
        variant: ModelVariantArtifact,
        variantPath: String?,
        variantHash: String?,
        substrate: String?,
        batteryFile: String,
        coherencePromptsFile: String,
        presetID: String?,
        judgeModel: String?,
        judgeKind: String? = nil,
        judgeProvider: String? = nil,
        battery: CapabilityBattery,
        coherencePrompts: [String],
        baselineBatteryScores: [BatteryItemScore],
        variantBatteryScores: [BatteryItemScore],
        baselineCoherenceResponses: [String],
        variantCoherenceResponses: [String],
        judge: JudgeClosure?,
        extraWarnings: [String],
        progress: VariantRobustnessProgressHandler?
    ) async throws -> VariantRobustnessReport {
        var batteryItems: [VariantRobustnessReport.BatteryItem] = []
        for (index, item) in battery.items.enumerated() {
            let baseline = baselineBatteryScores[index]
            let steered = variantBatteryScores[index]
            batteryItems.append(
                .init(
                    index: index + 1,
                    prompt: item.prompt,
                    answer: item.answer,
                    baselineResponse: baseline.output,
                    variantResponse: steered.output,
                    baselineCorrect: baseline.correct,
                    variantCorrect: steered.correct))
        }

        var coherenceItems: [VariantRobustnessReport.CoherenceItem] = []
        for (index, prompt) in coherencePrompts.enumerated() {
            let baseline = baselineCoherenceResponses[index]
            let steered = variantCoherenceResponses[index]
            let judgeResponse: PairedJudgeResponse?
            if let judge {
                await progress?(
                    .judgeStarted(index: index + 1, total: coherencePrompts.count, prompt: prompt))
                judgeResponse = try await judge(prompt, baseline, steered)
            } else {
                judgeResponse = nil
            }
            let judgeResult: String? =
                switch judgeResponse?.winner.lowercased() {
                case "a": "baseline"
                case "b": "variant"
                case "tie": "tie"
                default: nil
                }
            if let judgeResult, let judgeResponse {
                await progress?(
                    .judgeCompleted(
                        index: index + 1,
                        prompt: prompt,
                        result: judgeResult,
                        response: judgeResponse))
            }
            coherenceItems.append(
                .init(
                    index: index + 1,
                    prompt: prompt,
                    baselineResponse: baseline,
                    variantResponse: steered,
                    baselineWordCount: wordCount(baseline),
                    variantWordCount: wordCount(steered),
                    baselineDistinct2: distinctBigramRatio(baseline),
                    variantDistinct2: distinctBigramRatio(steered),
                    judge: judgeResponse,
                    judgeResult: judgeResult))
        }

        let baselineAccuracy = meanBool(batteryItems.map(\.baselineCorrect))
        let variantAccuracy = meanBool(batteryItems.map(\.variantCorrect))
        let meanBaselineDistinct = mean(coherenceItems.map(\.baselineDistinct2))
        let meanVariantDistinct = mean(coherenceItems.map(\.variantDistinct2))
        var warnings: [String] = extraWarnings
        if variantAccuracy < baselineAccuracy - 0.15 {
            warnings.append("capability battery dropped by more than 0.15")
        }
        if !coherenceItems.isEmpty, meanVariantDistinct < 0.45 {
            warnings.append("variant mean distinct-2 is below 0.45")
        }
        let baselineWins = coherenceItems.filter { $0.judgeResult == "baseline" }.count
        let variantWins = coherenceItems.filter { $0.judgeResult == "variant" }.count
        if baselineWins > variantWins {
            warnings.append("AI coherence judge preferred baseline more often than variant")
        }

        return VariantRobustnessReport(
            variantName: variant.name,
            variantArtifactPath: variantPath,
            variantArtifactHash: variantHash,
            baseModelID: variant.baseModelID,
            substrate: substrate,
            presetID: presetID,
            batteryFile: batteryFile,
            coherencePromptsFile: coherencePromptsFile,
            judgeModel: judgeModel,
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            baselineBatteryAccuracy: baselineAccuracy,
            variantBatteryAccuracy: variantAccuracy,
            meanBaselineDistinct2: meanBaselineDistinct,
            meanVariantDistinct2: meanVariantDistinct,
            meanBaselineWords: mean(coherenceItems.map { Float($0.baselineWordCount) }),
            meanVariantWords: mean(coherenceItems.map { Float($0.variantWordCount) }),
            batteryItems: batteryItems,
            coherenceItems: coherenceItems,
            warnings: warnings,
            // Stamped from the battery both paths loaded, so the local and
            // server runners say the same thing about the same file. Legacy
            // readings carry neither key, mirroring `battery.jsonl` rows.
            batteryFormat: battery.isolated ? battery.formatVersion : nil,
            armingIsolated: battery.isolated ? true : nil,
            judgeKind: judgeKind,
            judgeProvider: judgeProvider)
    }

    // MARK: - Server path

    /// One generation the server robustness path will request — the pure
    /// "battery items → request bodies" seam. Everything the ClusterClient
    /// call needs besides the variant spec itself: the chat messages, the
    /// token cap, greedy temperature (hard requirement: measured robustness
    /// arms are temperature-0 on both substrates), and the arm flag
    /// (`stripInterventions: true` IS the baseline arm — the same spec with
    /// its injections/adapters stripped server-side, mirroring the local
    /// runner's empty-injections + no-adapter baseline).
    public struct ServerArmRequest: Equatable, Sendable {
        public let kind: String  // "Capability" | "Coherence"
        public let index: Int  // 1-based within kind
        public let total: Int
        public let prompt: String
        public let side: String  // "Baseline" | "Variant"
        public let messages: [[String: String]]
        public let maxTokens: Int
        public let temperature: Double
        public let stripInterventions: Bool
    }

    /// One SIDE's capability-battery evaluation, scored server-side through
    /// `POST /api/variant/battery` (open issues §23). One request per side,
    /// not one per item: a format-2 battery is read under the arming the
    /// FILE declares, which the generate wire cannot express — it renders
    /// under the variant spec's prompt mode and system prompt, and it cannot
    /// read a choice item's answer-token distribution at all.
    ///
    /// The battery rides as a PIN (workspace-relative path + the digest of
    /// this side's bytes), exactly as a manifest pins it, so a server copy
    /// that differs refuses rather than scoring a different file under the
    /// same name. `stripInterventions: true` IS the baseline side, the same
    /// arm flag the generate wire uses.
    public struct ServerBatteryRequest: Equatable, Sendable {
        public let side: String  // "Baseline" | "Variant"
        public let batteryFile: String
        public let batteryHash: String
        public let stripInterventions: Bool
    }

    /// The two battery-side requests, in the local runner's side order.
    /// Pure and fixture-testable, like `serverArmRequests` beside it.
    public static func serverBatteryRequests(
        batteryFile: String, batteryHash: String
    ) -> [ServerBatteryRequest] {
        ["Baseline", "Variant"].map {
            ServerBatteryRequest(
                side: $0, batteryFile: batteryFile, batteryHash: batteryHash,
                stripInterventions: $0 == "Baseline")
        }
    }

    /// Request plan for a server robustness run, in the SAME arm order as the
    /// local runner (baseline battery, baseline coherence, variant battery,
    /// variant coherence). Pure and fixture-testable.
    ///
    /// A format-2 battery contributes NO entries here — its readings are
    /// whole-battery `ServerBatteryRequest`s, taken at the same two points in
    /// the order — so callers pass an empty `batteryPrompts` for one.
    public static func serverArmRequests(
        batteryPrompts: [String],
        coherencePrompts: [String],
        coherenceMaxTokens: Int,
        batteryMaxTokens: Int = VariantRobustness.batteryMaxTokens
    ) -> [ServerArmRequest] {
        var requests: [ServerArmRequest] = []
        func arm(_ kind: String, _ prompts: [String], side: String, maxTokens: Int) {
            for (index, prompt) in prompts.enumerated() {
                requests.append(
                    ServerArmRequest(
                        kind: kind,
                        index: index + 1,
                        total: prompts.count,
                        prompt: prompt,
                        side: side,
                        messages: [["role": "user", "content": prompt]],
                        maxTokens: maxTokens,
                        temperature: 0,
                        stripInterventions: side == "Baseline"))
            }
        }
        arm("Capability", batteryPrompts, side: "Baseline", maxTokens: batteryMaxTokens)
        arm("Coherence", coherencePrompts, side: "Baseline", maxTokens: coherenceMaxTokens)
        arm("Capability", batteryPrompts, side: "Variant", maxTokens: batteryMaxTokens)
        arm("Coherence", coherencePrompts, side: "Variant", maxTokens: coherenceMaxTokens)
        return requests
    }

    /// Server-workspace twin of `run`: batteries and coherence prompts are
    /// local recipe data, scoring policy is the SAME pure code
    /// (`assembleReport`), and only the model work goes through the
    /// caller-provided closures — the panel wires `generate` to
    /// `ClusterClient.variantGenerate` and `scoreBattery` to
    /// `ClusterClient.variantBatteryEvaluate`, with the variant under test as
    /// an INLINE spec. The report lands in the same local run directory
    /// structure, stamped `substrate: WorkspaceScoping.serverSubstrate`.
    /// Judging: Claude judges work from either workspace; a local judge model
    /// would need the local container, so it is skipped with a loud warning
    /// instead of quietly loading MLX in a server workspace.
    @discardableResult
    public static func runViaServer(
        record: ModelVariantRecord,
        batteryFile: String = defaultBatteryFile,
        coherencePromptsFile: String = defaultCoherencePromptsFile,
        maxCoherencePrompts: Int = 3,
        maxTokens: Int = 256,
        presetID: String? = nil,
        judgeModel: String? = nil,
        judgeRubric: String = defaultJudgeRubric,
        into runDirectory: URL? = nil,
        generate: @escaping @Sendable (ServerArmRequest) async throws -> String,
        scoreBattery: @escaping @Sendable (ServerBatteryRequest) async throws
            -> ClusterClient.VariantBatteryEvaluation,
        progress: VariantRobustnessProgressHandler? = nil
    ) async throws -> (report: VariantRobustnessReport, directory: URL) {
        try await runViaServer(
            variant: record.artifact,
            variantPath: ModelVariantStore.relativePath(for: record),
            variantHash: try? ModelVariantStore.hash(record.url),
            batteryFile: batteryFile,
            coherencePromptsFile: coherencePromptsFile,
            maxCoherencePrompts: maxCoherencePrompts,
            maxTokens: maxTokens,
            presetID: presetID,
            judgeModel: judgeModel,
            judgeRubric: judgeRubric,
            into: runDirectory,
            generate: generate,
            scoreBattery: scoreBattery,
            progress: progress)
    }

    /// Spec-level server robustness entry point: the same flow with the
    /// variant spec, its identity path, and its content hash passed directly —
    /// how a SERVER-STORED agent (no local record; spec fetched via
    /// `variantDetail`) is checked. The report still lands in THIS app
    /// workspace's runs/ tree (evidence comes home), stamped with the stored
    /// artifact's server path + hash as its identity.
    @discardableResult
    public static func runViaServer(
        variant: ModelVariantArtifact,
        variantPath: String?,
        variantHash: String?,
        batteryFile: String = defaultBatteryFile,
        coherencePromptsFile: String = defaultCoherencePromptsFile,
        maxCoherencePrompts: Int = 3,
        maxTokens: Int = 256,
        presetID: String? = nil,
        judgeModel: String? = nil,
        judgeRubric: String = defaultJudgeRubric,
        into runDirectory: URL? = nil,
        generate: @escaping @Sendable (ServerArmRequest) async throws -> String,
        scoreBattery: @escaping @Sendable (ServerBatteryRequest) async throws
            -> ClusterClient.VariantBatteryEvaluation,
        progress: VariantRobustnessProgressHandler? = nil
    ) async throws -> (report: VariantRobustnessReport, directory: URL) {
        let batteryURL = VectorCatalog.projectRoot.appending(path: batteryFile)
        let battery = try CapabilityBattery(url: batteryURL)
        // TWO wires, chosen by the battery's FORMAT — not by convenience.
        //
        // A format-2 battery goes through `POST /api/variant/battery`
        // (`scoreBattery`), one request per side. It has to: its choice items
        // are read from the model's answer-token distribution, and its arming
        // must REPLACE the variant's prompt mode and system prompt. Neither is
        // expressible on `/api/variant/generate`, which generates text and
        // whose `systemPrompt: nil` means "use the spec's" — so this path used
        // to refuse v2 by name (open issues §23), which since the seed
        // batteries became v2 meant refusing out of the box. The server wire
        // reuses `battery.py` wholesale, so a reading taken here and a
        // `battery.jsonl` row inside `experiment validate|run` are the same
        // measurement.
        //
        // A format-1 battery keeps the generate wire, byte for byte: legacy
        // arming IS the surrounding instrument's (here, the variant spec's),
        // its items are graded by the historical text matcher, and its pinned
        // hash therefore keeps its historical meaning. The server route
        // refuses format 1 for the mirror-image reason — a bare variant
        // reference is not an instrument to take arming from.
        let batteryArming = battery.resolveArming(
            promptMode: ExperimentManifest.PromptMode(rawValue: variant.promptMode)
                ?? .chatAssistant,
            systemPrompt: variant.systemPrompt,
            qwenThinkingEnabled: variant.qwenThinkingEnabled)
        var extraWarnings: [String] = []
        if let advisory = battery.contaminationAdvisory(batteryArming) {
            print("WARNING: \(advisory)")
            extraWarnings.append(advisory)
        }
        let promptsURL = VectorCatalog.projectRoot.appending(path: coherencePromptsFile)
        let (promptTexts, _) = try StimulusSet.loadTexts(url: promptsURL)
        let coherencePrompts = Array(promptTexts.prefix(max(0, maxCoherencePrompts)))

        // The pin the server verifies against its own copy of the file.
        let batteryHash = ExperimentStore.sha256Hex(try Data(contentsOf: batteryURL))
        let batteryPlan = battery.isolated
            ? serverBatteryRequests(batteryFile: batteryFile, batteryHash: batteryHash)
            : []
        let generationPlan = serverArmRequests(
            // A v2 battery contributes no per-item generations at all.
            batteryPrompts: battery.isolated ? [] : battery.items.map(\.prompt),
            coherencePrompts: coherencePrompts,
            coherenceMaxTokens: maxTokens,
            batteryMaxTokens: batteryArming.maxTokens)

        var responses: [String: [String]] = [:]
        var scoredBattery: [String: [BatteryItemScore]] = [:]
        // The local runner's arm order, preserved across both wires: a side's
        // battery is read before that side's coherence prompts, baseline
        // before variant.
        for side in ["Baseline", "Variant"] {
            if let request = batteryPlan.first(where: { $0.side == side }) {
                let evaluation = try await scoreBattery(request)
                scoredBattery[side] = try batteryScores(battery, evaluation: evaluation)
                // Nothing streamed for a choice reading (nothing is
                // generated), so the live surface is fed from the returned
                // records — one started/completed pair per item, the same
                // events the per-item generate path emitted.
                for (index, item) in evaluation.items.enumerated() {
                    await progress?(
                        .outputStarted(
                            kind: "Capability", index: index + 1,
                            total: evaluation.items.count, prompt: item.prompt,
                            side: side))
                    await progress?(
                        .outputCompleted(
                            kind: "Capability", index: index + 1,
                            prompt: item.prompt, side: side, output: item.output))
                }
                if let advisory = evaluation.advisory {
                    print("WARNING: \(advisory)")
                    extraWarnings.append(advisory)
                }
            }
            for request in generationPlan where request.side == side {
                await progress?(
                    .outputStarted(
                        kind: request.kind, index: request.index, total: request.total,
                        prompt: request.prompt, side: request.side))
                let output = try await generate(request)
                responses["\(request.kind)|\(request.side)", default: []].append(output)
                await progress?(
                    .outputCompleted(
                        kind: request.kind, index: request.index, prompt: request.prompt,
                        side: request.side, output: output))
            }
        }

        // The server route runs GENERATION remotely and judges from this Mac,
        // so an API-backed judge is reachable here and a local one is not.
        // Both API backends now ARE reachable: the skip warning below used to
        // swallow every non-Claude spelling, OpenRouter's included, which is
        // how a fully-implemented judge client became unreachable from this
        // pane.
        // Same dispatch as the local route, with NO local container: that is
        // what keeps the "local judge skipped" warning for a genuinely local
        // pick while letting both API backends through.
        let selection = JudgeModelSpelling.parse(judgeModel)
        let route = judgeRoute(selection, rubric: judgeRubric, localContainer: nil)
        let judge = route.judge
        extraWarnings += route.warnings
        let reportedJudge = judge == nil ? nil : selection?.model
        let reportedKind = judge == nil ? nil : selection?.kind
        let reportedProvider = judge == nil ? nil : selection?.provider

        let report = try await assembleReport(
            variant: variant,
            variantPath: variantPath,
            variantHash: variantHash,
            substrate: WorkspaceScoping.serverSubstrate,
            batteryFile: batteryFile,
            coherencePromptsFile: coherencePromptsFile,
            presetID: presetID,
            judgeModel: reportedJudge,
            judgeKind: reportedKind,
            judgeProvider: reportedProvider,
            battery: battery,
            coherencePrompts: coherencePrompts,
            // Format 2: the server's own scored records. Format 1: generated
            // text graded here by the historical matcher, unchanged.
            baselineBatteryScores: scoredBattery["Baseline"]
                ?? batteryScores(battery, responses["Capability|Baseline"] ?? []),
            variantBatteryScores: scoredBattery["Variant"]
                ?? batteryScores(battery, responses["Capability|Variant"] ?? []),
            baselineCoherenceResponses: responses["Coherence|Baseline"] ?? [],
            variantCoherenceResponses: responses["Coherence|Variant"] ?? [],
            judge: judge,
            extraWarnings: extraWarnings,
            progress: progress)
        let directory =
            if let runDirectory {
                runDirectory
            } else {
                try VectorCatalog.makeUniqueRunDirectory(
                    slug: "variant-robustness-\(FineTuneStore.slugify(variant.name))",
                    under: VectorCatalog.runsDirectory)
            }
        try write(report, to: directory)
        return (report, directory)
    }

    /// Generated battery responses graded by the historical text matcher —
    /// the generatedText reading, in the shape `assembleReport` consumes.
    /// Index-aligned with `battery.items`; a short array (a cancelled or
    /// failed arm) yields as many readings as it has responses.
    static func batteryScores(
        _ battery: CapabilityBattery, _ responses: [String]
    ) -> [BatteryItemScore] {
        zip(battery.items, responses).map { item, response in
            BatteryItemScore(
                scoring: .generatedText,
                output: response,
                correct: CapabilityBattery.isCorrect(
                    response: response, answer: item.answer,
                    grading: item.grading))
        }
    }

    /// The server's per-item battery records, in the shape `assembleReport`
    /// consumes. The readings are the SERVER's — a choice item's 0/1 came
    /// from its answer-token instrument and is never re-graded here (there is
    /// no text to re-grade), so this is a translation, not a second scorer.
    ///
    /// It still checks that the records line up with the local battery,
    /// item for item. The pinned hash makes a mismatch nearly impossible —
    /// the route refuses unless its bytes match ours — so a disagreement here
    /// means the two sides disagree about what the same bytes CONTAIN, which
    /// must never be papered over with a shorter list of readings.
    static func batteryScores(
        _ battery: CapabilityBattery,
        evaluation: ClusterClient.VariantBatteryEvaluation
    ) throws -> [BatteryItemScore] {
        guard evaluation.items.count == battery.items.count else {
            throw ExperimentError(
                reason: "server scored \(evaluation.items.count) capability "
                    + "items for a battery that has \(battery.items.count) "
                    + "(hash \(evaluation.batteryHash.prefix(12))…) — the two "
                    + "engines disagree about the same bytes")
        }
        return try zip(battery.items, evaluation.items).map { item, record in
            guard record.prompt == item.prompt else {
                throw ExperimentError(
                    reason: "server capability record \(record.promptIndex) is "
                        + "for a different item than the local battery's "
                        + "(\(record.prompt.prefix(40))… vs "
                        + "\(item.prompt.prefix(40))…)")
            }
            let scoring =
                CapabilityBattery.Scoring(rawValue: record.scoring)
                ?? .generatedText
            return BatteryItemScore(
                scoring: scoring,
                options: record.options,
                choiceProbability: record.choiceProbability,
                selected: record.selected,
                output: record.output,
                correct: record.correct)
        }
    }

    static func write(_ report: VariantRobustnessReport, to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // Canonical per-run config.json. `substrate` stamps the engine that
        // WROTE the run directory (this app); the report's own `substrate`
        // field records which engine generated the outputs.
        try RunMetadata.write(
            runType: "variant-robustness", to: directory, modelID: report.baseModelID)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(
            to: directory.appending(component: reportFileName),
            options: .atomic)
    }

    private static func meanBool(_ values: [Bool]) -> Float {
        guard !values.isEmpty else { return 0 }
        return Float(values.filter { $0 }.count) / Float(values.count)
    }

    private static func mean(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Float(values.count)
    }

    private static func wordCount(_ text: String) -> Int {
        text.split(whereSeparator: \.isWhitespace).count
    }

    private static func distinctBigramRatio(_ text: String) -> Float {
        let words = text.lowercased().split(whereSeparator: \.isWhitespace).map(String.init)
        guard words.count >= 2 else { return 0 }
        let bigrams = zip(words, words.dropFirst()).map { "\($0.0) \($0.1)" }
        return Float(Set(bigrams).count) / Float(bigrams.count)
    }
}

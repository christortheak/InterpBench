import Foundation
import SteeringKit

/// Draft field decisions over a caller-owned value and supplied capabilities.
/// The store admits the draft, gathers capability facts, and persists on success.
enum ManifestDraftEdits {
    static func setStudyType(
        _ type: StudyIntent, experimentName: String, manifest: inout ExperimentManifest
    ) throws {
        manifest.studyType = type.rawValue
        manifest.studyKind = type.mappedKind
    }

    static func setModelRevision(
        _ revision: String?, experimentName: String, manifest: inout ExperimentManifest
    ) throws {
        let trimmed = revision?.trimmingCharacters(in: .whitespacesAndNewlines)
        manifest.modelRevision = trimmed?.isEmpty == false ? trimmed : nil
    }

    static func setPhase(
        _ phase: String?, experimentName: String, manifest: inout ExperimentManifest
    ) throws {
        let trimmed = phase?.trimmingCharacters(in: .whitespacesAndNewlines)
        manifest.phase = trimmed?.isEmpty == false ? trimmed : nil
    }

    static func setCaseFamily(
        _ caseFamily: String?, experimentName: String, manifest: inout ExperimentManifest
    ) throws {
        let trimmed = caseFamily?.trimmingCharacters(in: .whitespacesAndNewlines)
        manifest.caseFamily = trimmed?.isEmpty == false ? trimmed : nil
    }

    static func setOutcomeInstruments(
        _ instruments: [String]?, experimentName: String, manifest: inout ExperimentManifest
    ) throws {
        let cleaned = (instruments ?? [])
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if let unknown = cleaned.first(where: {
            !knownOutcomeInstruments.contains($0)
        }) {
            throw ExperimentError.malformed(
                "unknown outcome instrument '\(unknown)' — known: "
                    + knownOutcomeInstruments.joined(separator: ", "),
                repair: "steerlab-cli experiment set-instruments "
                    + "\(experimentName) <"
                    + knownOutcomeInstruments.joined(separator: "|") + ">[,…]")
        }
        manifest.outcomeInstruments = cleaned.isEmpty ? nil : cleaned
    }

    static func setOrdinalAggregation(
        _ aggregation: String?, experimentName: String, manifest: inout ExperimentManifest
    ) throws {
        let trimmed = aggregation?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = trimmed, !value.isEmpty else {
            manifest.ordinalAggregation = nil
            return
        }
        guard knownOrdinalAggregations.contains(value) else {
            throw ExperimentError.malformed(
                "unknown ordinalAggregation '\(value)' — known: "
                    + knownOrdinalAggregations.joined(separator: ", "),
                repair: "steerlab-cli experiment set-instruments "
                    + "\(experimentName) ordinalScale --ordinal-aggregation <"
                    + knownOrdinalAggregations.joined(separator: "|") + ">")
        }
        manifest.ordinalAggregation = value
    }

    static func setAcknowledgeUnequalOptionLengths(
        _ acknowledged: Bool, experimentName: String, manifest: inout ExperimentManifest
    ) throws {
        manifest.acknowledgeUnequalOptionLengths = acknowledged ? true : nil
    }

    static func setSamplingPolicy(
        samplesPerItem: Int?, seedPolicy: String?, experimentName: String,
        manifest: inout ExperimentManifest
    ) throws {
        try applySamplesPerItem(
            samplesPerItem ?? 1, to: &manifest,
            experimentName: experimentName)
        try applySeedPolicy(
            seedPolicy ?? "", to: &manifest, experimentName: experimentName)
    }

    static func setSamplingProtocol(
        temperature: Double? = nil, maxTokens: Int? = nil,
        promptMode: String? = nil, samplesPerItem: Int? = nil,
        seedPolicy: String? = nil, reasoningEffort: String? = nil,
        reasoningMaxTokens: Int? = nil, experimentName: String, manifest: inout ExperimentManifest,
        capabilities: ModelCapabilities? = nil
    ) throws {
        if reasoningEffort != nil || reasoningMaxTokens != nil {
            let mergedEffort =
                reasoningEffort ?? manifest.resolvedReasoningEffort.rawValue
            var mergedBudget = reasoningMaxTokens ?? manifest.reasoningMaxTokens
            if mergedEffort == ReasoningEffort.off.rawValue,
                reasoningMaxTokens == nil
            {
                mergedBudget = nil  // the off declaration retires the budget
            }
            // The gate reads the model's CAPABILITY RECORD (the pinned
            // template's probed answers; the id heuristic, saying so,
            // when none exists): a level the template ignores or
            // rejects is refused here, at the declaration, never
            // rendered at the template's default under a manifest that
            // asserts the level. Server twin: `set_protocol`.
            let problems = ReasoningEffort.protocolViolations(
                effort: mergedEffort, reasoningMaxTokens: mergedBudget,
                modelID: manifest.modelID,
                capabilities: capabilities)
            guard problems.isEmpty else {
                throw ExperimentError.malformed(
                    problems.joined(separator: "; "),
                    repair: "steerlab-cli experiment set-sampling "
                        + "\(experimentName) --reasoning-effort <"
                        + ReasoningEffort.vocabulary.joined(separator: "|")
                        + "> --reasoning-max-tokens <n≥1>  (the budget only "
                        + "beside a non-off effort, on a model whose chat "
                        + "template has a thinking switch; a level only when "
                        + "the template accepts it — see model capabilities)")
            }
            manifest.reasoningEffort = mergedEffort
            manifest.reasoningMaxTokens = mergedBudget
            manifest.qwenThinkingEnabled = nil
        }
        if let temperature {
            guard temperature.isFinite, temperature >= 0 else {
                throw ExperimentError.malformed(
                    "temperature must be a non-negative number — got "
                        + "\(temperature)",
                    repair: "steerlab-cli experiment set-sampling "
                        + "\(experimentName) --temperature <t≥0>")
            }
            manifest.temperature = temperature
        }
        if let maxTokens {
            guard maxTokens >= 1 else {
                throw ExperimentError.malformed(
                    "maxTokens must be a positive integer — got \(maxTokens)",
                    repair: "steerlab-cli experiment set-sampling "
                        + "\(experimentName) --max-tokens <n≥1>")
            }
            manifest.maxTokens = maxTokens
        }
        if let promptMode {
            let trimmed =
                promptMode
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                manifest.promptMode = nil
            } else if let mode =
                ExperimentManifest.PromptMode(rawValue: trimmed)
            {
                manifest.promptMode = mode
            } else {
                throw ExperimentError.malformed(
                    "unknown promptMode '\(trimmed)' — known: "
                        + knownPromptModes.joined(separator: ", "),
                    repair: "steerlab-cli experiment set-sampling "
                        + "\(experimentName) --prompt-mode <"
                        + knownPromptModes.joined(separator: "|") + ">")
            }
        }
        if let samplesPerItem {
            try applySamplesPerItem(
                samplesPerItem, to: &manifest, experimentName: experimentName)
        }
        if let seedPolicy {
            try applySeedPolicy(
                seedPolicy, to: &manifest, experimentName: experimentName)
        }
    }

    static func setSeeds(
        _ seeds: [UInt64], experimentName: String, manifest: inout ExperimentManifest
    ) throws {
        guard !seeds.isEmpty else {
            throw ExperimentError(
                reason: "the seed list cannot be empty — the fixed-list "
                    + "seed policy indexes into it")
        }
        guard Set(seeds).count == seeds.count else {
            throw ExperimentError(
                reason: "duplicate seeds — two identical seeds generate "
                    + "two identical records that would masquerade as "
                    + "independent samples")
        }
        manifest.seeds = seeds
    }

    static func setPromotionRule(
        _ rule: ExperimentManifest.PromotionRule?, experimentName: String,
        manifest: inout ExperimentManifest
    ) throws {
        guard let rule else {
            manifest.promotionRule = nil
            return
        }
        if let threshold = rule.fdrThreshold,
            !(threshold.isFinite && threshold > 0 && threshold < 1)
        {
            throw ExperimentError(
                reason: "promotionRule fdrThreshold must be in (0, 1) — "
                    + "got \(threshold)")
        }
        let empty =
            rule.fdrThreshold == nil && rule.doseMonotone == nil
            && rule.exceedsRandomFloor == nil
            && (rule.capabilityGate?.isEmpty ?? true)
        manifest.promotionRule = empty ? nil : rule
    }

    static func setValidationReadDepth(
        layer: Int? = nil, fraction: Double? = nil,
        layers: [Int]? = nil, fractions: [Double]? = nil,
        experimentName: String, manifest: inout ExperimentManifest
    ) throws {
        if let problem = ValidationLayerRule.violation(
            declaredLayer: layer, declaredFraction: fraction,
            declaredLayers: layers, declaredFractions: fractions)
        {
            throw ExperimentError(reason: problem)
        }
        manifest.validationLayer = layers?.count == 1 ? layers?.first : layer
        manifest.validationLayerFraction =
            fractions?.count == 1 ? fractions?.first : fraction
        manifest.validationLayers = (layers?.count ?? 0) > 1 ? layers : nil
        manifest.validationLayerFractions =
            (fractions?.count ?? 0) > 1 ? fractions : nil
    }

    static func setSystemPrompt(
        _ text: String?, experimentName: String, manifest: inout ExperimentManifest,
        capabilities: ModelCapabilities? = nil
    ) throws {
        let trimmed = (text ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // The one prompt-mode-independent refusal: a chat template the
        // capability record marks `systemRole: unsupported` cannot
        // deliver the frame at all (rawCompletion prepends it and is
        // exempt). Server twin: `set_system_prompt`.
        let problems = PromptRendering.systemPromptViolations(
            systemPrompt: trimmed, modelID: manifest.modelID,
            rawCompletion: manifest.promptMode == .rawCompletion,
            capabilities: capabilities)
        guard problems.isEmpty else {
            throw ExperimentError.malformed(
                problems.joined(separator: "; "),
                repair: "drop the system prompt, or pin a model whose chat "
                    + "template can deliver system text")
        }
        manifest.systemPrompt = trimmed.isEmpty ? nil : trimmed
    }

    static func applySamplesPerItem(
        _ samples: Int, to manifest: inout ExperimentManifest,
        experimentName: String
    ) throws {
        guard samples >= 1 else {
            throw ExperimentError.malformed(
                "samplesPerItem must be ≥ 1 — got \(samples)",
                repair: "steerlab-cli experiment set-sampling \(experimentName) "
                    + "--samples-per-item <n≥1>  (1 clears to the "
                    + "deterministic default)")
        }
        manifest.samplesPerItem = samples > 1 ? samples : nil
    }

    static func applySeedPolicy(
        _ policy: String, to manifest: inout ExperimentManifest,
        experimentName: String
    ) throws {
        let trimmed = policy.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty || knownSeedPolicies.contains(trimmed) else {
            throw ExperimentError.malformed(
                "unknown seedPolicy '\(trimmed)' — known: "
                    + knownSeedPolicies.joined(separator: ", "),
                repair: "steerlab-cli experiment set-sampling \(experimentName) "
                    + "--seed-policy <"
                    + knownSeedPolicies.joined(separator: "|") + ">")
        }
        manifest.seedPolicy = trimmed.isEmpty ? nil : trimmed
    }

    static let knownOutcomeInstruments = [
        "sampledText", "answerTokenLogprob", "choiceProbability",
        "repeReaderScore", "ordinalScale",
    ]

    static let knownPromptModes =
        ExperimentManifest.PromptMode.allCases.map(\.rawValue)
    static let knownSeedPolicies = ["manifestSeeds", "derivedSHA256"]

    static let knownOrdinalAggregations = ["expectedValue", "argmax"]
}

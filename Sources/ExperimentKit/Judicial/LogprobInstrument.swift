import Foundation
import MLX
import MLXLMCommon
import SteeringKit

/// Answer-token logprob / choice-probability instrument (Swift twin of the
/// server's `steerlab_server/experiment/logprob.py` — semantics and record
/// shapes must match; changes here must land in both).
///
/// The preferred instrument for any CATEGORICAL outcome: instead of sampling a
/// full response and parsing it, evaluate the prompt once and read the model's
/// distribution over the named answer options. Deterministic, temperature-free,
/// and one forward pass per option — sensitive, and it avoids the confounds of
/// scoring sampled prose. (A judicial-decision study reads a holding this way;
/// the mechanism knows nothing about the domain, only the declared options.)
///
/// Steering parity is load-bearing: options are scored with a stepped forward
/// pass (single-chunk prompt prefill, then one step per option token under the
/// KV cache), so the `VectorInjector` fires at the true prompt end and at
/// every option position — identical semantics to steered generation. A single
/// unbatched full pass over `prompt + option` would inject only at the final
/// option token and silently under-steer.
///
/// Multi-token options are supported: the option score is the joint
/// log-probability (sum of per-token logprobs), with the length-normalized
/// mean also reported.

/// The declared aggregation for the `ordinalScale` outcome instrument: how
/// the renormalized probability distribution over the item's ordered option
/// ladder collapses to one ladder position. An instrument-design choice, so
/// it is DECLARED in the manifest (`ordinalAggregation`) and verify refuses
/// `ordinalScale` without it — never silently defaulted. Cross-engine
/// vocabulary: `expectedValue` | `argmax` (server
/// `logprob.ORDINAL_AGGREGATIONS`).
public enum OrdinalAggregation: String, Sendable, CaseIterable {
    /// Probability-weighted mean of the ladder positions 1..K.
    case expectedValue
    /// 1-based position of the maximum-probability option (first max wins
    /// ties on both engines).
    case argmax
}

/// One answer option's score under the current condition.
public struct OptionScore: Sendable {
    public let option: String
    public let tokenIDs: [Int]
    public let tokenLogprobs: [Float]

    public init(option: String, tokenIDs: [Int], tokenLogprobs: [Float]) {
        self.option = option
        self.tokenIDs = tokenIDs
        self.tokenLogprobs = tokenLogprobs
    }

    /// Joint log-probability of the option continuation.
    public var logprob: Double {
        tokenLogprobs.reduce(0.0) { $0 + Double($1) }
    }

    /// Length-normalized logprob (comparable across option lengths).
    public var meanTokenLogprob: Double {
        logprob / Double(max(1, tokenLogprobs.count))
    }
}

/// The instrument's full readout for one (prompt, option-set) evaluation.
public struct ChoiceResult: Sendable {
    public let options: [OptionScore]
    public let promptTokenCount: Int

    public init(options: [OptionScore], promptTokenCount: Int = 0) {
        self.options = options
        self.promptTokenCount = promptTokenCount
    }

    /// The argmax option under the joint logprob — the deterministic
    /// categorical answer.
    public var selected: String {
        options.max(by: { $0.logprob < $1.logprob })?.option ?? ""
    }

    /// Choice probability normalized over the option set (softmax of the
    /// joint logprobs, numerically stable via max-subtraction).
    public var probability: [String: Double] {
        let totals = options.map(\.logprob)
        guard let peak = totals.max() else { return [:] }
        let weights = totals.map { exp($0 - peak) }
        let z = weights.reduce(0, +)
        return Dictionary(
            zip(options.map(\.option), weights.map { $0 / z }),
            uniquingKeysWith: { _, later in later })
    }

    /// Per-option log-odds against the rest of the option set:
    /// ln(p / (1 − p)), clamped away from ±inf for degenerate probabilities.
    public var logOdds: [String: Double] {
        let eps = 1e-12
        return probability.mapValues { p in
            log(max(p, eps) / max(1.0 - p, eps))
        }
    }

    /// Joint-logprob gap between the selected option and the runner-up — the
    /// deterministic "confidence" of the categorical choice.
    public var margin: Double {
        let totals = options.map(\.logprob).sorted(by: >)
        guard totals.count >= 2 else { return .infinity }
        return totals[0] - totals[1]
    }

    /// max/min option token count (min clamped to 1). Joint logprobs favor
    /// shorter options, so a ratio well above 1 flags an instrument-design
    /// smell: options should be short canonical labels of comparable length,
    /// with the descriptive text outside the scored tokens.
    public var optionLengthRatio: Double {
        let counts = options.map { $0.tokenIDs.count }
        return Double(counts.max() ?? 0) / Double(max(1, counts.min() ?? 0))
    }

    /// Choice probabilities in DECLARED option order — the ladder order the
    /// `ordinalScale` instrument reads. Same softmax as `probability`, kept
    /// as an ordered array so position i always means the i-th declared
    /// option (a dictionary keyed by option text could not).
    public var orderedProbabilities: [Double] {
        let totals = options.map(\.logprob)
        guard let peak = totals.max() else { return [] }
        let weights = totals.map { exp($0 - peak) }
        let z = weights.reduce(0, +)
        return weights.map { $0 / z }
    }

    // Flat readouts for a generations.jsonl choice record. Together these are
    // the Swift twin of the server's `ChoiceResult.as_record_fields` — the
    // record field names they feed must match across engines.

    public var optionNames: [String] { options.map(\.option) }

    public var tokenCountsByOption: [String: Int] {
        byOption { $0.tokenIDs.count }
    }

    public var tokenIDsByOption: [String: [Int]] {
        byOption { $0.tokenIDs }
    }

    public var tokenLogprobsByOption: [String: [Float]] {
        byOption { $0.tokenLogprobs }
    }

    public var logprobByOption: [String: Double] {
        byOption { $0.logprob }
    }

    public var meanTokenLogprobByOption: [String: Double] {
        byOption { $0.meanTokenLogprob }
    }

    private func byOption<T>(_ value: (OptionScore) -> T) -> [String: T] {
        Dictionary(
            options.map { ($0.option, value($0)) },
            uniquingKeysWith: { _, later in later })
    }
}

public enum LogprobInstrument {

    /// Pure scoring math: per-option token logprobs from per-step logits rows.
    ///
    /// `stepLogits[i][k]` is the full next-token logits row *before* option
    /// `i`'s `k`-th token. Split out so the arithmetic is unit-testable
    /// without a model (Swift analogue of the server's
    /// `option_scores_from_step_logits`).
    public static func optionScores(
        options: [String], optionTokenIDs: [[Int]], stepLogits: [[[Float]]]
    ) -> [OptionScore] {
        zip(zip(options, optionTokenIDs), stepLogits).map { pair, rows in
            let (option, ids) = pair
            let logprobs = zip(rows, ids).map { row, tokenID in
                Float(logSoftmax(row, at: tokenID))
            }
            return OptionScore(option: option, tokenIDs: ids, tokenLogprobs: logprobs)
        }
    }

    /// Numerically stable log-softmax of `row` read at `index`. The logits
    /// arrive as float32; the reduction runs in double.
    static func logSoftmax(_ row: [Float], at index: Int) -> Double {
        guard let peak = row.max(), index >= 0, index < row.count else {
            return -.infinity
        }
        let m = Double(peak)
        let sumExp = row.reduce(0.0) { $0 + exp(Double($1) - m) }
        return Double(row[index]) - m - log(sumExp)
    }

    // MARK: - Ordinal-scale aggregation (pure math; server logprob.py twins)

    /// Renormalizes per-option probabilities over the declared ladder so
    /// they sum to 1 (defensive: the softmax already sums to 1, but the
    /// record field is CONTRACTUALLY renormalized, not trusted). Negative
    /// inputs clamp to 0; a degenerate all-zero total returns the uniform
    /// distribution; empty in → empty out. Cross-engine twin of the
    /// server's `logprob.ordinal_distribution` — same fixtures asserted in
    /// both test suites.
    public static func ordinalDistribution(_ probabilities: [Double]) -> [Double] {
        guard !probabilities.isEmpty else { return [] }
        let clamped = probabilities.map { max(0, $0) }
        let total = clamped.reduce(0, +)
        guard total > 0 else {
            return [Double](
                repeating: 1.0 / Double(probabilities.count),
                count: probabilities.count)
        }
        return clamped.map { $0 / total }
    }

    /// The 1-based ladder position under the declared aggregation:
    /// `expectedValue` = Σ position·p (probability-weighted mean of ladder
    /// positions 1..K), `argmax` = position of the maximum-probability
    /// option (first max wins ties — Python `max()` semantics, pinned by
    /// the cross-engine tie fixture). Empty distribution → 0 (degenerate;
    /// the run path guarantees ≥2 options). Server twin:
    /// `logprob.ordinal_position`.
    public static func ordinalPosition(
        distribution: [Double], aggregation: OrdinalAggregation
    ) -> Double {
        guard !distribution.isEmpty else { return 0 }
        switch aggregation {
        case .expectedValue:
            return distribution.enumerated()
                .reduce(0.0) { $0 + Double($1.offset + 1) * $1.element }
        case .argmax:
            var best = 0
            for (index, p) in distribution.enumerated() where p > distribution[best] {
                best = index
            }
            return Double(best + 1)
        }
    }

    /// Scores every answer option as a continuation of the rendered prompt.
    ///
    /// Injections (when given) are armed for the prompt prefill and every
    /// option step, mirroring steered generation (`ExperimentTasks.generate`):
    /// the prompt token count gates the injector, the whole prompt goes
    /// through in one chunk so the injector fires at the true prompt end, and
    /// each option gets a fresh KV cache. Temperature never enters — the
    /// readout is the model's own next-token distribution.
    ///
    /// `transcript` (scripted-transcript study items) renders the item's
    /// whole scripted conversation through the chat template
    /// (`ExperimentTasks.studyUserInput`); the stepped KV-cache scoring then
    /// reads the reply position AFTER the full rendered transcript — the
    /// transcript IS the prompt once rendered, so the injector gate
    /// (`promptTokenCount`) covers every transcript token.
    static func scoreOptions(
        _ container: ModelContainer,
        prompt: String,
        options: [String],
        modelID: String,
        injections: [ExperimentTasks.CellInjection] = [],
        promptMode: ExperimentManifest.PromptMode = .chatAssistant,
        systemPrompt: String? = nil,
        qwenThinkingEnabled: Bool = false,
        transcript: [ExperimentTasks.TranscriptTurn]? = nil
    ) async throws -> ChoiceResult {
        guard options.count >= 2 else {
            throw ExperimentError(
                reason: "answer-token logprob instrument needs at least 2 options")
        }
        let input = try await container.prepare(
            input: ExperimentTasks.studyUserInput(
                text: prompt,
                transcript: transcript,
                modelID: modelID,
                promptMode: promptMode,
                systemPrompt: systemPrompt,
                qwenThinkingEnabled: qwenThinkingEnabled))
        let promptTokenIDs: [Int] = input.text.tokens.asArray(Int.self)
        let promptTokenCount = promptTokenIDs.count

        // Options continue the prompt, so they tokenize WITHOUT special
        // tokens (a BOS inside an option would fracture the continuation).
        let optionTokenIDs: [[Int]] = try await container.perform { context in
            try options.map { option in
                let ids = context.tokenizer.encode(text: option, addSpecialTokens: false)
                guard !ids.isEmpty else {
                    throw ExperimentError(
                        reason: "option '\(option)' tokenizes to zero tokens")
                }
                return ids
            }
        }
        let longestOption = optionTokenIDs.map(\.count).max() ?? 0
        try await ExperimentTasks.validateContextBudget(
            container,
            modelID: modelID,
            promptTokens: promptTokenCount,
            requestedGenerationTokens: longestOption)

        // Through the shared builder: the instrument's whole claim is
        // decode-identical injection semantics, which now includes putting a
        // single ablator at the head of the chain.
        let injectors = try InterventionPlan.interventions(
            injections.map(\.planEdit), promptTokenCount: promptTokenCount)
        let stepLogits: [[[Float]]] = try await container.perform { context in
            guard let hookable = context.model as? InterventionHookable else {
                throw ExperimentError(reason: "loaded model has no intervention hooks")
            }
            hookable.interventions = injectors
            defer { hookable.interventions = [] }

            let promptTokens = MLXArray(promptTokenIDs.map { Int32($0) })
                .reshaped([1, promptTokenIDs.count])
            var rowsPerOption: [[[Float]]] = []
            rowsPerOption.reserveCapacity(optionTokenIDs.count)
            for ids in optionTokenIDs {
                // Fresh cache per option: every option is scored against the
                // identical prompt state, prefilled in a single chunk.
                let cache = context.model.newCache(parameters: nil)
                var rows = [lastPositionRow(context.model(promptTokens, cache: cache))]
                for tokenID in ids.dropLast() {
                    let step = MLXArray([Int32(tokenID)]).reshaped([1, 1])
                    rows.append(lastPositionRow(context.model(step, cache: cache)))
                }
                rowsPerOption.append(rows)
            }
            return rowsPerOption
        }

        return ChoiceResult(
            options: optionScores(
                options: options, optionTokenIDs: optionTokenIDs,
                stepLogits: stepLogits),
            promptTokenCount: promptTokenCount)
    }

    /// Last-position next-token logits as float32, evaluated and copied to
    /// CPU promptly (MLX is lazy — eval before reading; CLAUDE.md › MLX
    /// gotchas).
    private static func lastPositionRow(_ logits: MLXArray) -> [Float] {
        let row = logits[0..., -1, 0...].asType(.float32)
        eval(row)
        return row.asArray(Float.self)
    }
}

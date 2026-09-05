import Foundation
import SteeringKit

/// The two-phase token budget of a thinking-mode generation, as a pure rule
/// fed one sampled token at a time: `reasoningMaxTokens` for everything up to
/// and including the `</think>` token, then `maxTokens` for everything after
/// it. Server twin: `truncation_gate.ReasoningBudget`.
///
/// Exactly the convention a single `maxTokens` has always had, per block: a
/// block may hold exactly its budget, and the generation stops when the
/// budget is full and the block has not ended. So the close token may land on
/// the reasoning block's last budgeted step and still close it, and an answer
/// of exactly `maxTokens` tokens is allowed — the finish reason then decides,
/// from the stop token, whether that was a natural ending.
///
/// `observe(token:)` returns the finish reason a generation should STOP with
/// after this token, or nil to keep going. It is the one definition the MLX
/// token loop (`ExperimentTasks.generateMeasured`) applies, and it is
/// deliberately free of any framework type so a test can drive it with plain
/// integers.
public struct ReasoningBudget: Sendable, Equatable {
    /// The token that closes a reasoning block on every thinking-mode family
    /// this engine renders. Counted by its ID, never by text. Server twin:
    /// `truncation_gate.THINK_CLOSE_TOKEN`.
    public static let closeToken = "</think>"

    public let reasoningMaxTokens: Int
    public let maxTokens: Int
    public let closeID: Int

    public private(set) var closed = false
    public private(set) var reasoningTokens = 0
    public private(set) var answerTokens = 0

    public init(reasoningMaxTokens: Int, maxTokens: Int, closeID: Int) {
        self.reasoningMaxTokens = reasoningMaxTokens
        self.maxTokens = maxTokens
        self.closeID = closeID
    }

    /// The single `maxTokens` the framework is handed: both budgets, so the
    /// framework's own cap can never fire before this rule does.
    public var outerBound: Int { reasoningMaxTokens + maxTokens }

    /// The finish reason to stop with after `token`, or nil to keep going.
    /// Spelled in the record vocabulary (`ExperimentTasks.FinishReason`).
    public mutating func observe(token: Int) -> String? {
        if !closed {
            reasoningTokens += 1
            if token == closeID {
                closed = true
                return nil
            }
            return reasoningTokens >= reasoningMaxTokens ? "lengthInReasoning" : nil
        }
        answerTokens += 1
        return answerTokens >= maxTokens ? "length" : nil
    }

    /// Why a finished generation ended, from its sampled ids alone — the same
    /// reading the server takes from `token_ids_out`. Server twin:
    /// `truncation_gate.finish_reason` under a reasoning budget.
    ///
    /// Never closed: `lengthInReasoning` if the reasoning budget was filled,
    /// else `stop` (the model ended itself mid-reasoning — honestly not a
    /// cap). Closed: the ANSWER ids are judged against `maxTokens`, and at
    /// exactly the cap a stop token as the last id is a natural ending.
    public func finishReason(tokens: [Int], stopIDs: Set<Int>) -> String {
        guard let split = tokens.firstIndex(of: closeID) else {
            if !tokens.isEmpty, tokens.count >= reasoningMaxTokens {
                return "lengthInReasoning"
            }
            return "stop"
        }
        let answer = Array(tokens[(split + 1)...])
        if answer.count < maxTokens { return "stop" }
        if let last = answer.last, stopIDs.contains(last) { return "stop" }
        return "length"
    }
}

extension ExperimentManifest {
    /// The effort this study generates under.
    ///
    /// A NON-OFF `reasoningEffort` wins outright. Otherwise a legacy
    /// `qwenThinkingEnabled: true` still means the template's default effort
    /// (xhigh) — even beside an explicit `off`, which every manifest created
    /// since the effort existed carries by default: the boolean is the older,
    /// more deliberate declaration on such a document (a hand edit, or a
    /// pre-effort toggle written onto a fresh manifest), and reading it as off
    /// would silently drop a thinking mode the study asked for. The writers
    /// never leave both keys on a draft (`setSamplingProtocol` drops the
    /// boolean). An unparseable declared value resolves to off HERE (so a
    /// study can still be listed and read) and is named by `verify`. Server
    /// twin: `prompt_render.read_reasoning_effort`.
    public var resolvedReasoningEffort: ReasoningEffort {
        if let spelled = reasoningEffort,
            let parsed = ReasoningEffort(rawValue: spelled), parsed.isOn
        {
            return parsed
        }
        return ReasoningEffort.legacy(qwenThinkingEnabled: qwenThinkingEnabled ?? false)
    }

    /// Whether the study generates with a thinking block at all — the one
    /// question every pre-effort call site asks. Server twin:
    /// `Manifest.qwen_thinking_enabled` (a derived property there too).
    public var thinkingEnabled: Bool { resolvedReasoningEffort.isOn }

    /// The preregistration's account of the reasoning protocol — the effort
    /// and BOTH budgets — in the exact words the server's Sampling line uses.
    public var reasoningProtocolSummary: String {
        let budget = reasoningMaxTokens.map { String($0) } ?? "none"
        return "reasoningEffort \(resolvedReasoningEffort.rawValue), "
            + "maxTokens \(maxTokens), reasoningMaxTokens \(budget)"
    }

    /// The joint declaration rules, for a manifest that SPELLS
    /// `reasoningEffort` — a legacy boolean manifest is exempt (it was frozen
    /// under one budget and must keep verifying unchanged) — plus the one
    /// rule that needs no spelled effort: a budget beside no effort at all.
    /// Server twin: the reasoning block of `Manifest.verify`.
    public var reasoningProtocolViolations: [String] {
        reasoningProtocolViolations(capabilities: nil)
    }

    /// The same rules against an explicit capability record — the workspace
    /// record for a draft, the id heuristic for a frozen study (see
    /// `ExperimentStore.capabilitiesForGates`).
    public func reasoningProtocolViolations(
        capabilities: ModelCapabilities?
    ) -> [String] {
        if reasoningEffort != nil {
            return ReasoningEffort.protocolViolations(
                effort: reasoningEffort, reasoningMaxTokens: reasoningMaxTokens,
                modelID: modelID, capabilities: capabilities)
        }
        // A budget beside no effort at all: meaningless unless the legacy
        // boolean already says the study reasons (then the run honours it).
        if reasoningMaxTokens != nil, !thinkingEnabled {
            return [ReasoningEffort.budgetWithoutEffortReason]
        }
        return []
    }
}

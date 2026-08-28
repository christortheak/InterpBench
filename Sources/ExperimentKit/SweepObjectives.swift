import Foundation
import MLXLMCommon
import SteeringKit

/// judgeScore / logprobShift sweep objectives (cross-engine contract with the
/// server's `tasks._judge_preference` / `_choice_target_logprobs`).
///
/// The sweep's objective is manifest DATA: markerDensity reads the dev texts'
/// marker rubric; logprobShift reads the answer-token instrument on the
/// criterion's pinned choice rows (armed exactly as the study path arms it);
/// judgeScore reads paired-judge preference of each cell's dev texts against
/// the baseline's (reusing the texts generated for the constraints — never
/// generating twice). Whatever the objective, the capability/coherence
/// constraints stay computed from the generated dev texts and are never
/// bypassed. The pure pieces (shift mean, preference mapping/unblinding) are
/// unit-tested with fakes; only the container-boundary scoring needs a model.
extension ExperimentTasks {

    /// One sweep judge: panel-entry name + a rubric-baked judging closure
    /// `(taskPrompt, responseA, responseB) → verdict`, so `LocalPairedJudge`,
    /// `ClaudePairedJudge`, or a test fake drive scoring interchangeably.
    struct SweepJudge: Sendable {
        let name: String
        let judge:
            @Sendable (
                _ prompt: String, _ responseA: String, _ responseB: String
            ) async throws -> PairedJudgeResponse
    }

    /// The local sweep's one-model-slot rule (cross-engine contract,
    /// 2026-07-08): the sweep holds ONE loaded model for its whole run, so a
    /// LOCAL judge either uses the study model (an empty judge model already
    /// resolved to it) or the sweep refuses AT START — never a second load
    /// mid-sweep. Returns the refusal message, or nil when every local judge
    /// uses the study model. Pure, so the rule is unit-testable.
    ///
    /// The REVISION is part of "uses the study model" (review round 9,
    /// finding 1, audited from the judge-container cache): the held container
    /// is the study's, loaded at the manifest's pin, and a judge that pinned a
    /// different commit of the same repo would have judged with the study's
    /// weights while `resolvedJudges` carried its own revision forward into
    /// the provenance. One slot cannot honour two pins, so this is a refusal
    /// at sweep start, not a silent substitution.
    static func localJudgeSlotProblem(
        _ judges: [ResolvedJudge], studyModelID: String,
        studyRevision: String?
    ) -> String? {
        for judge in judges where judge.kind == "local" {
            if judge.model != studyModelID {
                return "local judge '\(judge.name)' uses model "
                    + "'\(judge.model)', not the study model "
                    + "'\(studyModelID)' — the local sweep holds one loaded "
                    + "model; use the study model as judge or a claude judge"
            }
            if judge.revision != studyRevision {
                return "local judge '\(judge.name)' pins revision "
                    + "\(revisionLabel(judge.revision)) of the study model "
                    + "'\(studyModelID)', which pins "
                    + "\(revisionLabel(studyRevision)) — the local sweep holds "
                    + "one loaded model and it is the study's, so this judge "
                    + "would judge with weights it did not declare; drop the "
                    + "judge's revision to judge with the study's, or use a "
                    + "claude judge"
            }
        }
        return nil
    }

    /// A pinned revision as this engine's refusals name one, or the words for
    /// no pin at all.
    static func revisionLabel(_ revision: String?) -> String {
        guard let revision = revision?.trimmingCharacters(in: .whitespacesAndNewlines),
            !revision.isEmpty
        else { return "no revision" }
        return String(revision.prefix(12)) + "…"
    }

    /// How one resolved judge executes in a judgeScore sweep — the pure
    /// decision `sweepJudgePanel` applies: local judges generate through the
    /// sweep's ALREADY-LOADED study container (never a second load; a
    /// non-study model throws the slot refusal), Claude judges call the
    /// Anthropic API, OpenRouter judges call OpenRouter with their pinned
    /// provider.
    enum SweepJudgeRoute: Equatable {
        case heldStudyContainer
        case claudeAPI(model: String)
        case openRouterAPI(model: String, provider: String)
    }

    static func sweepJudgeRoute(
        _ judge: ResolvedJudge, studyModelID: String, studyRevision: String?
    ) throws -> SweepJudgeRoute {
        if judge.kind == "openrouter" {
            // resolvedJudges refused an openrouter judge without a provider,
            // so this guard is belt-and-braces against a hand-built panel.
            guard let provider = judge.provider, !provider.isEmpty else {
                throw ExperimentError(
                    reason: "openrouter judge '\(judge.name)' has no pinned "
                        + "provider — an unpinned provider is not a pinned "
                        + "judge")
            }
            return .openRouterAPI(model: judge.model, provider: provider)
        }
        guard judge.kind == "local" else { return .claudeAPI(model: judge.model) }
        if let problem = localJudgeSlotProblem(
            [judge], studyModelID: studyModelID, studyRevision: studyRevision)
        {
            throw ExperimentError(reason: problem)
        }
        return .heldStudyContainer
    }

    /// Build the executing judge panel for a judgeScore sweep from the
    /// resolved objective (manifest-pinned rubric + judges). Local judges
    /// generate through the sweep's already-loaded study `container` — the
    /// panel NEVER loads a model (a local judge naming a non-study model was
    /// refused at sweep start; the route check here keeps that invariant).
    /// Claude judges call the API (credential already verified at resolve
    /// time).
    static func sweepJudgePanel(
        objective: SweepSelectionRule.ResolvedObjective,
        studyModelID: String,
        studyRevision: String?,
        container: ModelContainer
    ) throws -> [SweepJudge] {
        let rubric = objective.judgeRubricText ?? ""
        var panel: [SweepJudge] = []
        for judge in objective.judgePanel {
            switch try sweepJudgeRoute(
                judge, studyModelID: studyModelID,
                studyRevision: studyRevision)
            {
            case .heldStudyContainer:
                let model = judge.model
                panel.append(
                    SweepJudge(name: judge.name) { prompt, responseA, responseB in
                        try await LocalPairedJudge.judge(
                            container: container, modelID: model, rubric: rubric,
                            structuredPrompt: nil, prompt: prompt,
                            responseA: responseA, responseB: responseB)
                    })
            case .claudeAPI(let model):
                panel.append(
                    SweepJudge(name: judge.name) { prompt, responseA, responseB in
                        try await ClaudePairedJudge.judge(
                            model: model, rubric: rubric, structuredPrompt: nil,
                            prompt: prompt,
                            responseA: responseA, responseB: responseB)
                    })
            case .openRouterAPI(let model, let provider):
                panel.append(
                    SweepJudge(name: judge.name) { prompt, responseA, responseB in
                        try await OpenRouterPairedJudge.judge(
                            model: model, provider: provider, rubric: rubric,
                            structuredPrompt: nil, prompt: prompt,
                            responseA: responseA, responseB: responseB)
                    })
            }
        }
        return panel
    }

    /// Whether a judge verdict's winner is in the A/B/tie vocabulary
    /// (case-insensitive — the mapping switches below already lowercased).
    static func isValidJudgeWinner(_ winner: String) -> Bool {
        ["a", "b", "tie"].contains(winner.lowercased())
    }

    /// Invalid-verdict closure (2026-07-20, cross-engine with the server's
    /// `paired_judge.valid_verdict`): a verdict whose winner falls outside
    /// A/B/tie used to be silently recorded as a substantive tie (or a 0.5
    /// preference), corrupting the preference mean with invented data.
    /// Judges are LLMs — one malformed response is common — so the judge is
    /// retried ONCE; a second invalid verdict REFUSES the judging phase
    /// with the judge, the item, and both returned winners named. Shared by
    /// the sweep judgeScore objective and the evaluate paired-judge loop.
    /// `onInvalid` receives each malformed ATTEMPT (the server's `on_invalid`
    /// contract, same record keys), so a caller holding a run-status tracker
    /// can count it and keep the raw verdict beside the run. Optional: the
    /// sweep objective path has no run directory to annotate.
    ///
    /// The refusal is `JudgeNoncompliantError`, not a plain
    /// `ExperimentError` (Christian, 2026-08-09): the judge ANSWERED, so
    /// this is a per-item outcome the evaluate loop records as a row and
    /// continues past, while a transport failure from `obtain` propagates
    /// unchanged and still fails the session. Callers that want to survive
    /// noncompliance must catch that exact type — never a generic error, or
    /// a rate-limited judge would be misfiled as a garbage one. The sweep
    /// objective catches nothing and so still refuses, as on the server.
    static func judgmentWithValidWinner(
        judgeName: String,
        item: String,
        onInvalid: (@Sendable ([String: String]) async -> Void)? = nil,
        _ obtain: () async throws -> PairedJudgeResponse
    ) async throws -> PairedJudgeResponse {
        /// One malformed attempt, in the Python recorder's shape.
        func report(attempt: Int, _ verdict: PairedJudgeResponse) async {
            await onInvalid?([
                "attempt": String(attempt),
                "error": "winner '\(verdict.winner)' is not A, B, or tie",
                "verdict": verdict.winner,
                "rawResponse": verdict.briefReason,
                "judge": judgeName,
                "item": item,
            ])
        }
        var verdict = try await obtain()
        if !isValidJudgeWinner(verdict.winner) {
            let firstWinner = verdict.winner
            await report(attempt: 1, verdict)
            verdict = try await obtain()
            guard isValidJudgeWinner(verdict.winner) else {
                await report(attempt: 2, verdict)
                throw JudgeNoncompliantError(
                    reason: "judge '\(judgeName)' returned an invalid "
                        + "verdict twice for \(item) — winner "
                        + "'\(firstWinner)', then '\(verdict.winner)' "
                        + "(expected A, B, or tie); refusing to record "
                        + "invented data — fix the judge or rubric, then "
                        + "re-run this phase")
            }
        }
        return verdict
    }

    /// judgeScore objective for one cell: mean paired-judge preference of the
    /// CELL text vs the same-prompt BASELINE text over judges × dev prompts,
    /// blinded A/B per item with the evaluate path's deterministic flip and
    /// mapped to [0, 1] — cell win 1, baseline win 0, tie 0.5. Pure given the
    /// judge closures, so a test fake drives it without any network. An
    /// out-of-vocabulary winner is retried once and then refuses the sweep
    /// (`judgmentWithValidWinner`) — never scored as an invented 0.5 tie.
    static func sweepJudgePreference(
        experiment: String,
        conditionTag: String,
        prompts: [String],
        cellTexts: [String],
        baselineTexts: [String],
        judges: [SweepJudge]
    ) async throws -> Double {
        var scores: [Double] = []
        for judge in judges {
            for index in prompts.indices {
                // flip == true ⇒ the CELL response is shown as A (same
                // convention as evaluatePairedJudge's condition/baseline).
                let flip = shouldFlip(
                    experiment: experiment, condition: conditionTag,
                    seed: 0, promptID: "dev-\(index + 1)")
                let responseA = flip ? cellTexts[index] : baselineTexts[index]
                let responseB = flip ? baselineTexts[index] : cellTexts[index]
                let judgment = try await judgmentWithValidWinner(
                    judgeName: judge.name, item: "dev item \(index + 1) of "
                        + "condition '\(conditionTag)'"
                ) {
                    try await judge.judge(prompts[index], responseA, responseB)
                }
                switch judgment.winner.lowercased() {
                case "a": scores.append(flip ? 1 : 0)
                case "b": scores.append(flip ? 0 : 1)
                default: scores.append(0.5)  // "tie" — validated above
                }
            }
        }
        guard !scores.isEmpty else { return 0.5 }
        return scores.reduce(0, +) / Double(scores.count)
    }

    /// Per-row joint logprob of the TARGET option under the given injections
    /// (`[row id: logP(target)]`), armed exactly as the study path arms the
    /// choice instrument (same renderer, manifest prompt config, injector
    /// gating). The option-length guard runs per row — the baseline pass
    /// calls this before any grid generation, so a guard failure aborts the
    /// sweep at start, never mid-grid.
    static func sweepChoiceTargetLogprobs(
        _ container: ModelContainer,
        manifest: ExperimentManifest,
        rows: [SweepSelectionRule.ChoiceRow],
        injections: [CellInjection]
    ) async throws -> [String: Double] {
        var logprobs: [String: Double] = [:]
        for row in rows {
            let choice = try await LogprobInstrument.scoreOptions(
                container, prompt: row.prompt, options: row.options,
                modelID: manifest.modelID,
                injections: injections,
                promptMode: manifest.promptMode ?? .chatAssistant,
                systemPrompt: manifest.systemPrompt,
                qwenThinkingEnabled: manifest.qwenThinkingEnabled ?? false)
            try checkOptionLengths(choice, manifest: manifest, promptID: row.id)
            guard let logprob = choice.logprobByOption[row.target] else {
                throw ExperimentError(
                    reason: "choice row '\(row.id)': target '\(row.target)' "
                        + "was not scored")
            }
            logprobs[row.id] = logprob
        }
        return logprobs
    }

    /// logprobShift objective: mean over choice rows of
    /// logP(target | cell) − logP(target | baseline). 0 for the baseline
    /// cell by construction.
    static func meanLogprobShift(
        cell: [String: Double], baseline: [String: Double]
    ) -> Double {
        guard !baseline.isEmpty else { return 0 }
        let total = baseline.reduce(0.0) { sum, entry in
            sum + ((cell[entry.key] ?? 0) - entry.value)
        }
        return total / Double(baseline.count)
    }

    /// The provenance `metrics` block, keyed by the declared objective —
    /// markerDensity keeps its historical keys byte-for-byte; the new
    /// metrics stamp their own value plus the pinned baseline value.
    static func sweepMetricsBlock(
        metric: String,
        best: SweepSelectionRule.Cell,
        baselineMetric: Double,
        baselineDensity: Double,
        baselineAccuracy: Double
    ) -> [String: Double] {
        var metrics: [String: Double]
        switch metric {
        case "judgeScore":
            metrics = ["judgeScore": best.metric, "baselineJudgeScore": baselineMetric]
        case "logprobShift":
            metrics = [
                "logprobShift": best.metric, "baselineLogprobShift": baselineMetric,
            ]
        default:
            metrics = ["markerDensity": best.metric, "baselineDensity": baselineDensity]
        }
        metrics["distinct2"] = best.distinct2
        metrics["batteryAccuracy"] = best.batteryAccuracy
        metrics["baselineBatteryAccuracy"] = baselineAccuracy
        return metrics
    }
}

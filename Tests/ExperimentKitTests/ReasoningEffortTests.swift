import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// The declared reasoning effort and the reasoning token budget (2026-09-03).
///
/// `reasoningEffort` ∈ off | low | medium | xhigh replaced the Qwen-specific
/// boolean `qwenThinkingEnabled` on the study protocol and on a concept's
/// extraction rendering, and `reasoningMaxTokens` gave the reasoning block a
/// cap of its own so it no longer competes with the answer for one budget.
///
/// Server twin: `Server/tests/test_reasoning_effort.py`. The refusal sentences
/// asserted here are the server's, byte for byte.
@Suite(.serialized) struct ReasoningEffortTests {

    static let qwen = "Qwen/Qwen3-0.6B"
    static let gemma = "google/gemma-3-4b-it"
    static let close = 7

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "reasoning-effort-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func draft(_ name: String = "d", modelID: String = qwen) throws -> ExperimentManifest {
        try ExperimentStore.create(name: name, description: "", modelID: modelID)
    }

    func decode(_ json: String) throws -> ExperimentManifest {
        try JSONDecoder().decode(ExperimentManifest.self, from: Data(json.utf8))
    }

    // MARK: - 1. the vocabulary and the legacy reading

    @Test func theVocabularyIsClosedAndOffIsFirst() {
        // `on` (thinking at the template's default, no effort variable) and
        // `high` (a probe candidate) joined 2026-09-05 with the capability
        // record; which LEVELS a model may declare is the pinned template's
        // answer, not the vocabulary's.
        #expect(ReasoningEffort.vocabulary == ["off", "on", "low", "medium", "high", "xhigh"])
        #expect(ReasoningEffort.levels.map(\.rawValue) == ModelCapabilities.effortCandidates)
        #expect(ReasoningEffort.on.isOn && !ReasoningEffort.on.isLevel)
        #expect(ReasoningEffort.legacy(qwenThinkingEnabled: true) == .xhigh)
        #expect(ReasoningEffort.legacy(qwenThinkingEnabled: false) == .off)
        #expect(PromptRendering.hasThinkingMode(Self.qwen))
        #expect(!PromptRendering.hasThinkingMode(Self.gemma))
    }

    @Test func aLegacyBooleanManifestReadsAsOffOrXhighAndIsExempt() throws {
        let off = try decode(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","qwenThinkingEnabled":false}"#)
        let on = try decode(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","qwenThinkingEnabled":true}"#)
        #expect(off.resolvedReasoningEffort == .off && !off.thinkingEnabled)
        #expect(on.resolvedReasoningEffort == .xhigh && on.thinkingEnabled)
        #expect(on.reasoningEffort == nil && on.reasoningMaxTokens == nil)
        // The joint rules are silent for the legacy spelling — even on a
        // family without a thinking mode; the family rule is a declaration
        // rule, and the frozen ladder must keep verifying unchanged.
        let legacyGemma = try decode(
            #"{"name":"s","modelID":"google/gemma-3-4b-it","status":"frozen","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","qwenThinkingEnabled":true}"#)
        #expect(legacyGemma.reasoningProtocolViolations.isEmpty)
        // …and a legacy thinking-on manifest may carry a budget the run honours.
        let legacyWithBudget = try decode(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","qwenThinkingEnabled":true,"reasoningMaxTokens":64}"#)
        #expect(legacyWithBudget.reasoningProtocolViolations.isEmpty)
    }

    @Test func theNewSpellingWinsAndRoundTripsByteForByte() throws {
        let manifest = try decode(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","reasoningEffort":"low","reasoningMaxTokens":300,"qwenThinkingEnabled":false}"#)
        #expect(manifest.resolvedReasoningEffort == .low)
        #expect(manifest.reasoningMaxTokens == 300)
        #expect(manifest.thinkingEnabled)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let text = String(decoding: try encoder.encode(manifest), as: UTF8.self)
        #expect(text.contains(#""reasoningEffort":"low""#))
        #expect(text.contains(#""reasoningMaxTokens":300"#))
        // A legacy manifest re-encodes WITHOUT the new keys: absent stays
        // absent, so its content hash cannot move.
        let legacy = try decode(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","qwenThinkingEnabled":true}"#)
        let legacyText = String(decoding: try encoder.encode(legacy), as: UTF8.self)
        #expect(!legacyText.contains("reasoningEffort"))
        #expect(!legacyText.contains("reasoningMaxTokens"))
        #expect(legacyText.contains(#""qwenThinkingEnabled":true"#))
        // A legacy `true` beside an explicit `off` (a pre-effort toggle set on
        // a fresh manifest, which carries `off` by default) still means the
        // template's default effort; a non-off effort beside it wins outright.
        var mixed = ExperimentManifest(name: "s", description: "", modelID: Self.qwen)
        #expect(mixed.reasoningEffort == "off")
        mixed.qwenThinkingEnabled = true
        #expect(mixed.resolvedReasoningEffort == .xhigh && mixed.thinkingEnabled)
        mixed.reasoningEffort = "low"
        #expect(mixed.resolvedReasoningEffort == .low)
    }

    @Test func aNewManifestSpellsTheEffortAndNeverTheBoolean() async throws {
        try await withTempRoot { _ in
            let created = try draft()
            #expect(created.reasoningEffort == "off")
            #expect(created.qwenThinkingEnabled == nil)
            #expect(created.reasoningMaxTokens == nil)
        }
    }

    // MARK: - 2. the declaration rules

    @Test func aNonOffEffortNeedsItsBudget() async throws {
        try await withTempRoot { _ in
            try draft()
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setSamplingProtocol(
                    reasoningEffort: "low", experimentName: "d")
            }
            do {
                try ExperimentStore.setSamplingProtocol(
                    reasoningEffort: "low", experimentName: "d")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(
                    ReasoningEffort.effortWithoutBudgetReason(.low)))
                #expect(error.malformedInvocation?.repairAction
                    .contains("--reasoning-max-tokens") == true)
            }
            // Nothing was written.
            #expect(try ExperimentStore.load(name: "d").reasoningEffort == "off")
            // Together, they land — and drop nothing else.
            let declared = try ExperimentStore.setSamplingProtocol(
                reasoningEffort: "low", reasoningMaxTokens: 400,
                experimentName: "d")
            #expect(declared.reasoningEffort == "low")
            #expect(declared.reasoningMaxTokens == 400)
            #expect(declared.maxTokens == 2048)
            // A budget alone, on a draft that already reasons, is fine.
            #expect(try ExperimentStore.setSamplingProtocol(
                reasoningMaxTokens: 800, experimentName: "d").reasoningMaxTokens == 800)
        }
    }

    @Test func offTakesNoBudgetAndRetiresOne() async throws {
        try await withTempRoot { _ in
            try draft()
            do {
                try ExperimentStore.setSamplingProtocol(
                    reasoningMaxTokens: 400, experimentName: "d")
                Issue.record("a budget beside off must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(ReasoningEffort.budgetWithoutEffortReason))
            }
            try ExperimentStore.setSamplingProtocol(
                reasoningEffort: "medium", reasoningMaxTokens: 400,
                experimentName: "d")
            // Off in the SAME call as a budget says two things: refused.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setSamplingProtocol(
                    reasoningEffort: "off", reasoningMaxTokens: 400,
                    experimentName: "d")
            }
            // Off alone retires the budget the draft carried.
            let retired = try ExperimentStore.setSamplingProtocol(
                reasoningEffort: "off", experimentName: "d")
            #expect(retired.reasoningEffort == "off")
            #expect(retired.reasoningMaxTokens == nil)
        }
    }

    @Test func theEffortIsClosedVocabularyAndTheBudgetPositive() async throws {
        try await withTempRoot { _ in
            try draft()
            do {
                try ExperimentStore.setSamplingProtocol(
                    reasoningEffort: "hgih", reasoningMaxTokens: 4,
                    experimentName: "d")
                Issue.record("an unknown effort must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(ReasoningEffort.unknownEffortReason("hgih")))
            }
            for bad in [0, -1] {
                do {
                    try ExperimentStore.setSamplingProtocol(
                        reasoningEffort: "low", reasoningMaxTokens: bad,
                        experimentName: "d")
                    Issue.record("a non-positive budget must refuse")
                } catch let error as ExperimentError {
                    #expect(error.reason.contains("positive integer"))
                }
            }
        }
    }

    @Test func aFamilyWithoutAThinkingModeRefusesANonOffEffort() async throws {
        try await withTempRoot { _ in
            try draft(modelID: Self.gemma)
            do {
                try ExperimentStore.setSamplingProtocol(
                    reasoningEffort: "low", reasoningMaxTokens: 100,
                    experimentName: "d")
                Issue.record("a non-off effort on Gemma must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(
                    ReasoningEffort.effortWithoutThinkingModeReason(
                        .low, modelID: Self.gemma)))
            }
            // Off is always declarable.
            #expect(try ExperimentStore.setSamplingProtocol(
                reasoningEffort: "off", experimentName: "d").reasoningEffort == "off")
        }
    }

    @Test func writingTheEffortDropsTheLegacyBooleanFromADraft() async throws {
        try await withTempRoot { _ in
            var manifest = try draft()
            manifest.reasoningEffort = nil
            manifest.qwenThinkingEnabled = true
            try ExperimentStore.save(manifest)
            #expect(try ExperimentStore.load(name: "d").resolvedReasoningEffort == .xhigh)
            let declared = try ExperimentStore.setSamplingProtocol(
                reasoningEffort: "low", reasoningMaxTokens: 64, experimentName: "d")
            #expect(declared.qwenThinkingEnabled == nil)
            #expect(declared.reasoningEffort == "low")
        }
    }

    @Test func theSetSamplingVerbCarriesBothFlagsAndEchoesTheProtocol() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "experiment",
                ["set-sampling", "d", "--reasoning-effort", "medium",
                 "--reasoning-max-tokens", "512"])
            #expect(outcome.envelope.exitCode == 0, "\(outcome)")
            let manifest = try ExperimentStore.load(name: "d")
            #expect(manifest.reasoningEffort == "medium")
            #expect(manifest.reasoningMaxTokens == 512)
            let refused = await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "experiment",
                ["set-sampling", "d", "--reasoning-max-tokens", "x"])
            #expect(refused.envelope.exitCode == 64)
        }
    }

    @Test func verifyAppliesTheJointRulesToTheNewSpellingOnly() throws {
        func violations(_ json: String) throws -> [String] {
            try decode(json).reasoningProtocolViolations
        }
        #expect(try violations(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","reasoningEffort":"low"}"#)
            == [ReasoningEffort.effortWithoutBudgetReason(.low)])
        #expect(try violations(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","reasoningEffort":"off","reasoningMaxTokens":9}"#)
            == [ReasoningEffort.budgetWithoutEffortReason])
        #expect(try violations(
            #"{"name":"s","modelID":"google/gemma-3-4b-it","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","reasoningEffort":"xhigh","reasoningMaxTokens":9}"#)
            == [ReasoningEffort.effortWithoutThinkingModeReason(.xhigh, modelID: Self.gemma)])
        #expect(try violations(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","reasoningEffort":"nope"}"#)
            == [ReasoningEffort.unknownEffortReason("nope")])
        #expect(try violations(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","reasoningEffort":"medium","reasoningMaxTokens":9}"#)
            .isEmpty)
        // A budget beside NO effort at all — and no legacy boolean either.
        #expect(try violations(
            #"{"name":"s","modelID":"Qwen/Qwen3-0.6B","status":"draft","experimentDescription":"","createdAt":"2026-01-01T00:00:00Z","reasoningMaxTokens":9}"#)
            == [ReasoningEffort.budgetWithoutEffortReason])
    }

    @Test func theRefusalSentencesAreTheServersByteForByte() {
        // Pinned as literals, so a rewording on one engine is a test failure
        // here rather than two explanations of one rule.
        #expect(ReasoningEffort.budgetWithoutEffortReason
            == "reasoningMaxTokens is declared but reasoningEffort is off — the "
            + "model generates no reasoning block to cap; declare a non-off "
            + "reasoningEffort or drop the budget")
        #expect(ReasoningEffort.unknownEffortReason("hgih")
            == "unknown reasoningEffort 'hgih' — known: off, on, low, medium, high, xhigh")
        #expect(ReasoningEffort.effortWithoutBudgetReason(.low)
            == "reasoningEffort 'low' needs a reasoningMaxTokens — the reasoning "
            + "block's own token cap, declared and never defaulted; maxTokens "
            + "stays the answer budget, counted from the token after </think>")
        #expect(ReasoningEffort.effortWithoutThinkingModeReason(.low, modelID: Self.gemma)
            == "reasoningEffort 'low' declared for google/gemma-3-4b-it, whose "
            + "chat template has no thinking switch (enable_thinking changes "
            + "nothing it renders) — the template would ignore it and the study "
            + "would look as if it reasoned when it did not; declare "
            + "reasoningEffort off")
        #expect(ReasoningEffort.malformedBudgetReason(0)
            == "reasoningMaxTokens must be a positive integer — got 0")
    }

    // MARK: - 3. the template context

    @Test func theEffortReachesTheTemplateByValueAndOffIsTheOldContext() {
        #expect(PromptRendering.thinkingContext(modelID: Self.qwen, effort: .off)
            .map { $0["enable_thinking"] as? Bool } == false)
        let low = PromptRendering.thinkingContext(modelID: Self.qwen, effort: .low)
        #expect(low?["enable_thinking"] as? Bool == true)
        #expect(low?["reasoning_effort"] as? String == "low")
        #expect(PromptRendering.thinkingContext(modelID: Self.gemma, effort: .low) == nil)
        // The legacy wrapper: true means the template's default effort.
        let legacy = PromptRendering.qwenContext(modelID: Self.qwen, qwenThinkingEnabled: true)
        #expect(legacy?["reasoning_effort"] as? String == "xhigh")
        #expect(PromptRendering.qwenContext(modelID: Self.qwen, qwenThinkingEnabled: false)?
            .keys.contains("reasoning_effort") == false)
        // Raw completion: the family suffix follows the effort.
        #expect(PromptRendering.rawCompletionText(
            prompt: "Decide.", modelID: Self.qwen, systemPrompt: nil,
            qwenThinkingEnabled: false, reasoningEffort: .medium).hasSuffix(" /think"))
        #expect(PromptRendering.rawCompletionText(
            prompt: "Decide.", modelID: Self.qwen, systemPrompt: nil,
            qwenThinkingEnabled: true, reasoningEffort: .off).hasSuffix(" /no_think"))
    }

    // MARK: - 4. the extraction rendering

    @Test func theRenderingReadsBothSpellingsButRefusesThemTogether() throws {
        let legacyOn = try ExtractionRendering.declared(
            object: ["mode": "chatTemplate", "qwenThinkingEnabled": true])
        #expect(legacyOn?.resolvedReasoningEffort == .xhigh)
        let new = try ExtractionRendering.declared(
            object: ["mode": "chatTemplate", "reasoningEffort": "medium"])
        #expect(new?.resolvedReasoningEffort == .medium)
        #expect(new?.resolvedQwenThinkingEnabled == true)
        #expect(new?.label == "chatTemplate (addGenerationPrompt=true, reasoningEffort=medium)")
        do {
            _ = try ExtractionRendering.declared(
                object: ["mode": "chatTemplate", "qwenThinkingEnabled": true,
                         "reasoningEffort": "xhigh"])
            Issue.record("both spellings must refuse")
        } catch let error as ExtractionRendering.DeclarationError {
            #expect(error.message == ExtractionRendering.bothThinkingKeysReason)
        }
        #expect(throws: ExtractionRendering.DeclarationError.self) {
            try ExtractionRendering.declared(
                object: ["mode": "chatTemplate", "reasoningEffort": "hgih"])
        }
        // Reading a recorded block is as strict as declaring.
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(
                ExtractionRendering.self,
                from: Data(#"{"mode":"chatTemplate","qwenThinkingEnabled":true,"reasoningEffort":"low"}"#.utf8))
        }
        // The family gate, asked with a model id.
        let problem = ExtractionRendering.thinkingModeProblem(new, modelID: Self.gemma)
        #expect(problem?.message
            == ExtractionRendering.effortWithoutThinkingModeError(.medium, modelID: Self.gemma).message)
        #expect(ExtractionRendering.thinkingModeProblem(new, modelID: Self.qwen) == nil)
        #expect(ExtractionRendering.thinkingModeProblem(.raw, modelID: Self.gemma) == nil)
    }

    @Test func theIdentityFragmentKeepsTheBooleanSpellingForOffAndXhigh() throws {
        let legacy = try #require(try ExtractionRendering.declared(
            object: ["mode": "chatTemplate", "qwenThinkingEnabled": true]))
        let xhigh = try #require(try ExtractionRendering.declared(
            object: ["mode": "chatTemplate", "reasoningEffort": "xhigh"]))
        let low = try #require(try ExtractionRendering.declared(
            object: ["mode": "chatTemplate", "reasoningEffort": "low"]))
        let legacyJSON = RecipeIdentity.renderingJSON(
            RecipeIdentity.Components.canonicalRendering(legacy))
        #expect(legacyJSON == RecipeIdentity.renderingJSON(
            RecipeIdentity.Components.canonicalRendering(xhigh)))
        #expect(legacyJSON
            == #"{"addGenerationPrompt":true,"mode":"chatTemplate","qwenThinkingEnabled":true,"systemPrompt":null}"#)
        #expect(RecipeIdentity.renderingJSON(RecipeIdentity.Components.canonicalRendering(low))
            == #"{"addGenerationPrompt":true,"mode":"chatTemplate","qwenThinkingEnabled":true,"reasoningEffort":"low","systemPrompt":null}"#)
        // A NEW stamp spells the effort, never the boolean.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let stamp = String(decoding: try encoder.encode(try #require(legacy.stamp)), as: UTF8.self)
        #expect(stamp == #"{"addGenerationPrompt":true,"mode":"chatTemplate","reasoningEffort":"xhigh"}"#)
    }

    // MARK: - 5. the two-phase budget

    @Test func theBudgetRuleCountsReasoningThenAnswer() {
        var budget = ReasoningBudget(reasoningMaxTokens: 3, maxTokens: 2, closeID: Self.close)
        #expect(budget.outerBound == 5)
        #expect(budget.observe(token: 10) == nil)
        #expect(budget.observe(token: Self.close) == nil)
        #expect(budget.closed && budget.reasoningTokens == 2)
        #expect(budget.observe(token: 11) == nil)
        #expect(budget.observe(token: 12) == ExperimentTasks.FinishReason.length)
        // The close token may land on the reasoning block's LAST budgeted step.
        var late = ReasoningBudget(reasoningMaxTokens: 3, maxTokens: 2, closeID: Self.close)
        #expect([10, 11, Self.close].map { late.observe(token: $0) } == [nil, nil, nil])
        #expect(late.closed)
        // …and one step later than that is the reasoning cap.
        var capped = ReasoningBudget(reasoningMaxTokens: 3, maxTokens: 2, closeID: Self.close)
        #expect([10, 11].map { capped.observe(token: $0) } == [nil, nil])
        #expect(capped.observe(token: 12) == ExperimentTasks.FinishReason.lengthInReasoning)
    }

    @Test func theFinishReasonSplitsAtTheFirstCloseToken() {
        let budget = ReasoningBudget(reasoningMaxTokens: 4, maxTokens: 3, closeID: Self.close)
        let eos = 1
        let stops: Set<Int> = [eos]
        #expect(budget.finishReason(tokens: [10, 11, 12, 13], stopIDs: stops) == "lengthInReasoning")
        #expect(budget.finishReason(tokens: [10, eos], stopIDs: stops) == "stop")
        #expect(budget.finishReason(tokens: [10, Self.close, 20], stopIDs: stops) == "stop")
        #expect(budget.finishReason(tokens: [10, Self.close, 20, 21, 22], stopIDs: stops) == "length")
        #expect(budget.finishReason(tokens: [10, Self.close, 20, 21, eos], stopIDs: stops) == "stop")
        #expect(budget.finishReason(tokens: [10, 11, 12, Self.close, 20], stopIDs: stops) == "stop")
        #expect(budget.finishReason(tokens: [], stopIDs: stops) == "stop")
        #expect(ExperimentTasks.FinishReason.vocabulary
            == ["stop", "length", "lengthInReasoning", "cancelled"])
        #expect(ExperimentTasks.FinishReason.isCutOff("lengthInReasoning"))
        #expect(ExperimentTasks.FinishReason.isCutOff("length"))
        #expect(!ExperimentTasks.FinishReason.isCutOff("stop"))
        #expect(!ExperimentTasks.FinishReason.isCutOff("cancelled"))
    }

    // MARK: - 6. the gate and the report

    private func row(_ reason: String, promptID: String = "p") -> ExperimentTasks.MetricRow {
        ExperimentTasks.MetricRow(
            condition: "c", seed: 0, promptIndex: 0, promptID: promptID,
            wordCount: 1, distinct2: 1, finishReason: reason, markerDensity: [:])
    }

    @Test func theReportAndTheTallyDistinguishTheCaps() throws {
        let rows = [row("stop"), row("length"), row("lengthInReasoning"), row("cancelled")]
        let cell = ExperimentTasks.truncationCell(rows: rows, condition: "c", promptID: "p")
        #expect(cell.classified == 4 && cell.lengthStopped == 2 && cell.inReasoning == 1)
        let report = ExperimentTasks.truncationReport(rows: rows, threshold: nil)
        #expect(report.lengthStopped == 2)
        #expect(report.lengthStoppedInReasoning == 1)
        #expect(report.lengthStoppedFraction == 0.5)
        #expect(report.cells.first?.lengthStoppedInReasoning == 1)
        // The wire shape carries the new key beside the old ones.
        let encoded = String(decoding: try JSONEncoder().encode(report), as: UTF8.self)
        #expect(encoded.contains("\"lengthStoppedInReasoning\":1"))
    }

    @Test func theRefusalNamesWhichCapWasHitAndTheRepairNamesBothFlags() throws {
        let plain = try #require(ExperimentTasks.lengthStoppedRefusal(
            classified: 4, lengthStopped: 2, threshold: 0.25,
            condition: "c", promptID: "p", maxTokens: 8))
        #expect(plain.contains("stopped at the 8-token cap instead of finishing (50.0%)"))
        let both = try #require(ExperimentTasks.lengthStoppedRefusal(
            classified: 4, lengthStopped: 2, threshold: 0.25,
            condition: "c", promptID: "p", maxTokens: 8,
            lengthStoppedInReasoning: 1, reasoningMaxTokens: 64))
        #expect(both.contains("1 inside the reasoning block at the 64-token reasoning cap"))
        #expect(both.contains("1 in the answer at the 8-token answer cap"))
        #expect(ExperimentTasks.lengthStoppedRepair(experiment: "s", maxTokens: 8)
            == "steerlab-cli experiment set-sampling s --max-tokens <n>  (n above 8), "
            + "then re-run; a frozen study is iterated by duplicating first: "
            + "steerlab-cli experiment duplicate s s-v2")
        #expect(ExperimentTasks.lengthStoppedRepair(
            experiment: "s", maxTokens: 8, reasoningMaxTokens: 64)
            .contains("--reasoning-max-tokens <m>"))
        // A reasoning-capped generation is truncated for the choice parser too.
        #expect(ExperimentTasks.MeasuredGeneration(text: "x", finishReason: "lengthInReasoning")
            .hitTokenCap)
    }

    // MARK: - 7. the preregistration line and the design summary

    @Test func thePreregistrationPrintsTheEffortAndBothBudgets() throws {
        var manifest = ExperimentManifest(name: "s", description: "", modelID: Self.qwen)
        manifest.maxTokens = 512
        #expect(manifest.reasoningProtocolSummary
            == "reasoningEffort off, maxTokens 512, reasoningMaxTokens none")
        manifest.reasoningEffort = "medium"
        manifest.reasoningMaxTokens = 2048
        #expect(manifest.reasoningProtocolSummary
            == "reasoningEffort medium, maxTokens 512, reasoningMaxTokens 2048")
    }
}

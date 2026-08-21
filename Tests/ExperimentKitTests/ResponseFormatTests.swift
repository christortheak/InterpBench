import Foundation
import Testing

@testable import ExperimentKit

/// A2 — `responseFormat` as a real schema field.
///
/// The field had been in the rule-vs-justice JSONL since it was authored and
/// existed nowhere in executable code: it survived only as a round-tripped
/// unknown key. The visible consequence was wrong advice — a reasons-arm
/// study whose every row is `json` was told to "declare a categorical outcome
/// instrument (answer-token probability)", which would have scored the
/// position holding `{` rather than a choice.
struct ResponseFormatTests {

    private func item(
        _ id: String, options: Bool = true, target: Bool = true,
        format: ResponseFormat? = nil
    ) -> ResponseFormat.Item {
        .init(id: id, hasOptions: options, hasTarget: target, format: format)
    }

    // MARK: the target rule (open-issues #6)

    @Test func aTargetDependentInstrumentRefusesItemsThatDeclareNoTarget()
        throws
    {
        // The run loop no longer synthesizes `target = options[0]`, so a
        // study whose items never declare one produces zero endpoint rows.
        let refusal = try #require(
            ResponseFormat.refusal(
                items: [item("a", target: false, format: .label),
                        item("b", target: false, format: .label)],
                declaredInstruments: ["answerTokenLogprob"],
                declaredScope: nil))
        #expect(refusal.contains("none of the 2 task items declares a 'target'"))
        #expect(refusal.contains("zero endpoint rows"))
        // Every way out is named, including the one that fits a ladder.
        #expect(refusal.contains("declare ordinalScale"))
    }

    @Test func theTargetRuleIsSilentForADeclaredOrdinalLadder() {
        // A rating ladder legitimately declares no target: its endpoint is
        // the ladder position, and target-less is the CORRECT shape there.
        let ladder = [item("a", target: false, format: .label)]
        #expect(
            ResponseFormat.refusal(
                items: ladder, declaredInstruments: ["ordinalScale"],
                declaredScope: nil) == nil)
        // And when the ordinal readout rides an answer-token record (the
        // mixed instrument), the ladder items still may not declare a target.
        #expect(
            ResponseFormat.refusal(
                items: ladder,
                declaredInstruments: ["ordinalScale", "answerTokenLogprob"],
                declaredScope: nil) == nil)
    }

    @Test func partialTargetCoverageIsNotTheInstrumentsBusiness() {
        // Same rule the options gate follows: some items declaring a target
        // is a measurable study, and the untargeted rows simply carry no
        // choice endpoint downstream.
        #expect(
            ResponseFormat.refusal(
                items: [item("a", format: .label),
                        item("b", target: false, format: .label)],
                declaredInstruments: ["choiceProbability"],
                declaredScope: nil) == nil)
    }

    @Test func theTargetRuleReadsThePinnedItemFile() throws {
        // `responseFormatItems` is the only reader of the item bytes, so the
        // rule's input must come from the file's own `target` key.
        let document = try TaskPromptsDocument.load(
            Data(
                """
                {"id": "a", "prompt": "pick", "options": ["A", "B"], "target": "A"}
                {"id": "b", "prompt": "rate", "options": ["1", "2", "3"]}
                {"id": "c", "prompt": "pick", "options": ["A", "B"], "target": ""}

                """.utf8))
        #expect(document.responseFormatItems.map(\.hasTarget)
            == [true, false, false])
    }

    // MARK: the vocabulary

    @Test func onlyLabelSupportsAnswerTokenScoring() {
        #expect(ResponseFormat.label.supportsAnswerTokenScoring)
        #expect(!ResponseFormat.json.supportsAnswerTokenScoring)
        #expect(!ResponseFormat.freeText.supportsAnswerTokenScoring)
    }

    @Test func anAbsentFieldParsesToNilAndAnUnknownOneThrows() throws {
        #expect(try ResponseFormat.parse(nil) == nil)
        #expect(try ResponseFormat.parse("") == nil)
        #expect(try ResponseFormat.parse("label") == .label)
        // A typo must NOT degrade to "unspecified" — that would silently
        // restore the permissive behaviour this type exists to remove.
        #expect(throws: (any Error).self) { try ResponseFormat.parse("lable") }
        #expect(throws: (any Error).self) { try ResponseFormat.parse("JSON") }
    }

    // MARK: the refusal

    @Test func jsonRowsRefuseAnAnswerTokenInstrument() throws {
        let refusal = try #require(
            ResponseFormat.refusal(
                items: [item("a", format: .json), item("b", format: .json)],
                declaredInstruments: ["answerTokenLogprob"],
                declaredScope: nil))
        #expect(refusal.contains("2 items"))
        #expect(refusal.contains("a, b"))
        #expect(refusal.contains("opening brace"))
        // The refusal must name every way out, not merely the problem.
        #expect(refusal.contains("'label'"))
        #expect(refusal.contains("outcomeInstrumentScope"))
    }

    @Test func labelRowsAreFine() {
        #expect(
            ResponseFormat.refusal(
                items: [item("a", format: .label)],
                declaredInstruments: ["answerTokenLogprob"],
                declaredScope: nil) == nil)
    }

    @Test func legacyRowsWithNoDeclaredFormatStayPermissive() {
        // Files predating the field have been measured successfully; treating
        // absence as an objection would refuse studies that are fine.
        #expect(
            ResponseFormat.refusal(
                items: [item("a", format: nil), item("b", format: nil)],
                declaredInstruments: ["answerTokenLogprob"],
                declaredScope: nil) == nil)
    }

    @Test func zeroOptionItemsRefuseTheInstrument() throws {
        // 2026-08-06 field incident: ordinalScale declared while every task
        // item had options: null — the instrument silently produced zero
        // records and the run burned its whole GPU allocation. The old
        // contract read this as "not the instrument's business"; it is now
        // a refusal (mirror of the server's
        // test_zero_option_items_refuse_the_instrument).
        let refusal = try #require(
            ResponseFormat.refusal(
                items: [item("a", options: false, format: .json)],
                declaredInstruments: ["answerTokenLogprob"],
                declaredScope: nil))
        #expect(refusal.contains("none of the 1 task item carries options"))
        #expect(refusal.contains("silently produce zero records"))
        #expect(refusal.contains("Add 'options'"))

        let plural = try #require(
            ResponseFormat.refusal(
                items: [
                    item("a", options: false), item("b", options: false),
                ],
                declaredInstruments: ["ordinalScale"],
                declaredScope: nil))
        #expect(plural.contains("none of the 2 task items carries options"))
    }

    @Test func partialOptionCoverageIsNotTheInstrumentsBusiness() {
        // A mixed file where SOME items carry options stays legal: the
        // choice gate is per-item, and the carrying items produce records.
        #expect(
            ResponseFormat.refusal(
                items: [
                    item("a", format: .label), item("b", options: false),
                ],
                declaredInstruments: ["answerTokenLogprob"],
                declaredScope: nil) == nil)
    }

    @Test func aScopeSelectingZeroItemsRefuses() throws {
        // A pinned scope whose responseFormats select nothing passes the
        // drift check (0 == 0) and used to run the instrument on nothing.
        let items = [item("a", options: true, format: .json)]
        let scope = ResponseFormat.Scope.pin(
            responseFormats: ["label"], items: items)
        #expect(scope.itemCount == 0)
        let refusal = try #require(
            ResponseFormat.refusal(
                items: items, declaredInstruments: ["answerTokenLogprob"],
                declaredScope: scope))
        #expect(refusal.contains("outcomeInstrumentScope selects zero task items"))
        #expect(refusal.contains("silently produce zero records"))
    }

    @Test func zeroInScopeOptionItemsRefuseWithScopeWording() throws {
        // The scope selects rows, but none of THEM carries options.
        let items = [
            item("a", options: false, format: .label),
            item("b", options: true, format: .json),
        ]
        let scope = ResponseFormat.Scope.pin(
            responseFormats: ["label"], items: items)
        #expect(scope.itemCount == 1)
        let refusal = try #require(
            ResponseFormat.refusal(
                items: items, declaredInstruments: ["choiceProbability"],
                declaredScope: scope))
        #expect(
            refusal.contains("none of the 1 in-scope task item carries options"))
    }

    @Test func noChoiceInstrumentMeansNoRefusal() {
        // A json file measured by sampled text is exactly right.
        #expect(
            ResponseFormat.refusal(
                items: [item("a", format: .json)],
                declaredInstruments: ["sampledText"],
                declaredScope: nil) == nil)
        #expect(
            ResponseFormat.refusal(
                items: [item("a", format: .json)],
                declaredInstruments: nil, declaredScope: nil) == nil)
    }

    @Test func aMixedFileExplainsEveryObjection() throws {
        let refusal = try #require(
            ResponseFormat.refusal(
                items: [item("a", format: .json), item("b", format: .freeText)],
                declaredInstruments: ["ordinalScale"],
                declaredScope: nil))
        #expect(refusal.contains("opening brace"))
        #expect(refusal.contains("prose"))
    }

    // MARK: declared scope

    @Test func aDeclaredScopeMakesAMixedFileCoherent() {
        let items = [
            item("label-1", format: .label), item("label-2", format: .label),
            item("json-1", format: .json),
        ]
        let scope = ResponseFormat.Scope.pin(responseFormats: ["label"], items: items)
        #expect(scope.itemCount == 2)
        #expect(
            ResponseFormat.refusal(
                items: items, declaredInstruments: ["answerTokenLogprob"],
                declaredScope: scope) == nil)
        // ...and the scope actually excludes the json row from measurement.
        #expect(!scope.includes(items[2]))
        #expect(scope.includes(items[0]))
    }

    @Test func aScopeThatStillAdmitsUnreadableRowsStillRefuses() {
        let items = [item("json-1", format: .json)]
        let scope = ResponseFormat.Scope.pin(
            responseFormats: ["label", "json"], items: items)
        #expect(
            ResponseFormat.refusal(
                items: items, declaredInstruments: ["answerTokenLogprob"],
                declaredScope: scope) != nil)
    }

    @Test func anUndeclaredRowIsNotProvenInScope() {
        // A scope exists because the file is mixed; a row that declares
        // nothing cannot be shown to belong to the measured subset.
        let scope = ResponseFormat.Scope.pin(
            responseFormats: ["label"], items: [item("x", format: .label)])
        #expect(!scope.includes(item("legacy", format: nil)))
    }

    // MARK: scope drift

    @Test func aScopeDetectsAChangedItemCount() throws {
        let items = [item("a", format: .label), item("b", format: .label)]
        let scope = ResponseFormat.Scope.pin(responseFormats: ["label"], items: items)
        #expect(scope.driftRefusal(items: items) == nil)

        let grown = items + [item("c", format: .label)]
        let drift = try #require(scope.driftRefusal(items: grown))
        #expect(drift.contains("pins 2"))
        #expect(drift.contains("select 3"))
    }

    @Test func aScopeDetectsSameCountDifferentItems() throws {
        // The subtle case a bare count would miss: the file was edited to
        // swap one row for another.
        let items = [item("a", format: .label), item("b", format: .label)]
        let scope = ResponseFormat.Scope.pin(responseFormats: ["label"], items: items)
        let swapped = [item("a", format: .label), item("c", format: .label)]
        let drift = try #require(scope.driftRefusal(items: swapped))
        #expect(drift.contains("same COUNT of different items"))
    }

    // MARK: submit-time surfacing (2026-08-06 field incident)

    /// Duplicate & Adjust + task-file swap left a stale scope pin, and a
    /// 4-shard Slurm submission staged/loaded gemma-3-27b-it before the
    /// run-stage gate refused. The submit-time guard must refuse the same
    /// drift before packaging, with the actionable re-declare instruction.
    @Test func aDriftedScopeRefusesAtSubmitWithTheRedeclareInstruction() throws {
        let pinned = [item("a", format: .label), item("b", format: .label)]
        let scope = ResponseFormat.Scope.pin(
            responseFormats: ["label"], items: pinned)
        let swapped = [item("a", format: .label), item("c", format: .label)]
        let refusal = try #require(
            ExperimentTasks.scopeDriftSubmissionRefusal(
                verb: "run", studyKind: .modelOutput, scope: scope,
                items: swapped, taskPromptsFile: "prompts/kz/tasks.jsonl"))
        // The server's drift message, verbatim…
        #expect(refusal.contains("outcomeInstrumentScope pins"))
        // …plus the way out, naming the file the scope must be re-pinned to.
        #expect(refusal.contains("re-declare the instrument scope"))
        #expect(refusal.contains("prompts/kz/tasks.jsonl"))
    }

    @Test func theSubmitGuardIsARunVerbRuleOnly() {
        let pinned = [item("a", format: .label)]
        let scope = ResponseFormat.Scope.pin(
            responseFormats: ["label"], items: pinned)
        let swapped = [item("b", format: .label)]
        // Non-run verbs never read the task prompts this way.
        for verb in ["extract", "validate", "sweep", "evaluate"] {
            #expect(
                ExperimentTasks.scopeDriftSubmissionRefusal(
                    verb: verb, studyKind: .modelOutput, scope: scope,
                    items: swapped, taskPromptsFile: "f.jsonl") == nil)
        }
        // Multi-agent studies have no task prompts to drift.
        #expect(
            ExperimentTasks.scopeDriftSubmissionRefusal(
                verb: "run", studyKind: .multiAgent, scope: scope,
                items: swapped, taskPromptsFile: "f.jsonl") == nil)
        // No scope, or a scope that still matches, submits silently.
        #expect(
            ExperimentTasks.scopeDriftSubmissionRefusal(
                verb: "run", studyKind: .modelOutput, scope: nil,
                items: swapped, taskPromptsFile: "f.jsonl") == nil)
        #expect(
            ExperimentTasks.scopeDriftSubmissionRefusal(
                verb: "run", studyKind: .modelOutput, scope: scope,
                items: pinned, taskPromptsFile: "f.jsonl") == nil)
    }

    // MARK: the advice surfaces that were wrong

    @Test func theActivationWarningNoLongerRecommendsAnUnreadableInstrument() throws {
        // The exact situation the researcher hit on the reasons arm.
        let warning = try #require(
            InstrumentActivation.activationWarning(
                optionsItemCount: 12, instruments: nil,
                unscorableOptionItemCount: 12))
        #expect(warning.contains("sampled text"))
        #expect(!warning.contains("Declare a categorical outcome instrument"))
    }

    @Test func aPartlyReadableFileIsToldAboutScope() throws {
        let warning = try #require(
            InstrumentActivation.activationWarning(
                optionsItemCount: 12, instruments: nil,
                unscorableOptionItemCount: 4))
        #expect(warning.contains("scope"))
        #expect(warning.contains("8 label items"))
    }

    @Test func aFullyReadableFileKeepsTheOriginalAdvice() throws {
        let warning = try #require(
            InstrumentActivation.activationWarning(
                optionsItemCount: 12, instruments: nil,
                unscorableOptionItemCount: 0))
        #expect(warning.contains("Declare a categorical outcome instrument"))
    }

    // MARK: parsing through the real loader

    @Test func theParserValidatesAndCarriesTheField() throws {
        let jsonl = """
            {"id":"a","text":"pick","options":["A","B"],"responseFormat":"label"}
            {"id":"b","text":"pick","options":["A","B"],"responseFormat":"json"}
            {"id":"c","text":"pick","options":["A","B"]}
            """
        let prompts = try ExperimentTasks.parseTaskPrompts(Data(jsonl.utf8))
        #expect(prompts.map(\.responseFormat) == [.label, .json, nil])

        let items = ExperimentTasks.responseFormatItems(prompts)
        #expect(ResponseFormat.unscorableItems(items).map(\.id) == ["b"])
    }

    @Test func theParserRefusesAnUnknownValueNamingTheItem() throws {
        let jsonl =
            #"{"id":"oops","text":"pick","options":["A"],"responseFormat":"lable"}"#
        #expect(throws: (any Error).self) {
            try ExperimentTasks.parseTaskPrompts(Data(jsonl.utf8))
        }
    }

    /// The editor must still OPEN a file the run loop would refuse —
    /// otherwise there is no way to fix it in the app.
    @Test func theEditorIsLenientWhereTheRunLoopRefuses() throws {
        let jsonl =
            #"{"id":"oops","text":"pick","options":["A"],"responseFormat":"lable"}"#
        let document = try TaskPromptsDocument.load(Data(jsonl.utf8))
        #expect(document.count == 1)
        #expect(document.optionsItemCount == 1)
        // Unrecognised reads as "undeclared" here, so it is not counted as
        // unscorable either — the run loop is where it refuses.
        #expect(document.unscorableOptionItemCount == 0)
    }

    @Test func theEditorCountsUnscorableRows() throws {
        let jsonl = """
            {"id":"a","text":"pick","options":["A","B"],"responseFormat":"label"}
            {"id":"b","text":"pick","options":["A","B"],"responseFormat":"json"}
            """
        let document = try TaskPromptsDocument.load(Data(jsonl.utf8))
        #expect(document.optionsItemCount == 2)
        #expect(document.unscorableOptionItemCount == 1)
    }

    /// Round-tripping must not lose the field — the data-loss bug this
    /// document type was built to prevent.
    @Test func theEditorPreservesTheFieldOnSave() throws {
        let jsonl =
            #"{"id":"a","options":["A","B"],"responseFormat":"json","text":"pick"}"#
        var document = try TaskPromptsDocument.load(Data(jsonl.utf8))
        document = document.applyingEditedTexts(["different prompt"])
        let saved = String(decoding: document.serialized(), as: UTF8.self)
        #expect(saved.contains("\"responseFormat\":\"json\""))
        #expect(saved.contains("different prompt"))
    }
}

/// C4 — what a declared capability tolerance can actually gate on.
///
/// Battery accuracy moves in steps of 1/N, so a tolerance falling between
/// steps is not the tolerance that operates. A sweep reporting "tolerance
/// 0.15" while gating at 0.2 is reporting a number that decided nothing.
struct BatteryResolutionTests {

    @Test func theOperativeGateIsTheFirstStepAboveTheTolerance() throws {
        // The plan's worked example: 10 items, tolerance 0.15. A one-item
        // drop (0.1) passes `>= baseline - 0.15`; the first drop that fails
        // is two items, 0.2.
        let resolution = try #require(
            SweepSelectionRule.batteryResolution(
                itemCount: 10, capabilityTolerance: 0.15))
        #expect(resolution.effectiveTolerance == 0.2)
        #expect(resolution.summary.contains("steps of 0.1"))
        #expect(resolution.summary.contains("first larger step, 0.2"))
    }

    @Test func aToleranceExactlyOnAStepStillGatesOneStepHigher() throws {
        // The constraint is `>=`, so a drop EQUAL to the tolerance passes.
        let resolution = try #require(
            SweepSelectionRule.batteryResolution(
                itemCount: 10, capabilityTolerance: 0.1))
        #expect(resolution.effectiveTolerance == 0.2)
        #expect(resolution.isCoarse)
    }

    @Test func aLargeBatteryResolvesCloseToTheDeclaredTolerance() throws {
        let resolution = try #require(
            SweepSelectionRule.batteryResolution(
                itemCount: 100, capabilityTolerance: 0.15))
        #expect(abs(resolution.effectiveTolerance - 0.16) < 1e-9)
        #expect(!resolution.isCoarse)
    }

    @Test func nothingIsSaidAboutAnEmptyOrNonsenseBattery() {
        #expect(
            SweepSelectionRule.batteryResolution(
                itemCount: 0, capabilityTolerance: 0.15) == nil)
        #expect(
            SweepSelectionRule.batteryResolution(
                itemCount: 10, capabilityTolerance: .nan) == nil)
        #expect(
            SweepSelectionRule.batteryResolution(
                itemCount: 10, capabilityTolerance: -0.1) == nil)
    }

    /// Cross-engine: the two implementations must agree cell for cell, or a
    /// sweep's advisory would depend on which engine ran it.
    @Test func theArithmeticMatchesThePythonTwin() throws {
        // Values verified against sweep_selection.battery_resolution.
        let cases: [(Int, Double, Double)] = [
            (10, 0.15, 0.2), (10, 0.1, 0.2), (20, 0.15, 0.2),
            (100, 0.15, 0.16), (12, 0.15, 1.0 / 6.0), (1, 0.5, 1.0),
        ]
        for (n, tolerance, expected) in cases {
            let resolution = try #require(
                SweepSelectionRule.batteryResolution(
                    itemCount: n, capabilityTolerance: tolerance))
            #expect(abs(resolution.effectiveTolerance - expected) < 1e-9)
        }
    }
}

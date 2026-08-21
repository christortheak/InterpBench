import Foundation
import Testing

@testable import ExperimentKit

/// Team finding P1 — explicit instrument activation: fields preserved is
/// NOT measurement enabled. The warning rule, the two-fact Save & Pin
/// summary, the outcome-mode mapping, and the headless readiness row must
/// all tell the same story.
@Suite(.serialized) struct InstrumentActivationTests {

    // MARK: warning rule

    @Test func warningFiresOnlyWhenOptionsPresentAndNoCategoricalInstrument() {
        // Options present, nothing declared → the exact honest sentence.
        let warning = InstrumentActivation.activationWarning(
            optionsItemCount: 6, instruments: nil)
        #expect(warning?.contains(
            "Options are present, but this study will only generate and parse "
                + "answer text") == true)
        // sampledText alone is NOT a categorical instrument.
        #expect(
            InstrumentActivation.activationWarning(
                optionsItemCount: 6, instruments: ["sampledText"]) != nil)
        // A categorical instrument silences it.
        #expect(
            InstrumentActivation.activationWarning(
                optionsItemCount: 6, instruments: ["answerTokenLogprob"]) == nil)
        #expect(
            InstrumentActivation.activationWarning(
                optionsItemCount: 6,
                instruments: ["sampledText", "choiceProbability"]) == nil)
        // No options → nothing to warn about, whatever is declared.
        #expect(
            InstrumentActivation.activationWarning(
                optionsItemCount: 0, instruments: nil) == nil)
    }

    @Test func detectedCapabilitiesLineStatesTheDataFact() {
        #expect(
            InstrumentActivation.detectedCapabilitiesLine(optionsItemCount: 6)
                == "6 items have categorical options")
        #expect(
            InstrumentActivation.detectedCapabilitiesLine(optionsItemCount: 1)
                == "1 item has categorical options")
        #expect(
            InstrumentActivation.detectedCapabilitiesLine(optionsItemCount: 0) == nil)
    }

    // MARK: Save & Pin reports both facts separately

    @Test func savePinSummarySeparatesPreservationFromActivation() {
        let none = InstrumentActivation.savePinSummary(
            optionsItemCount: 6, itemCount: 6, instruments: nil)
        #expect(none.contains("metadata preserved: 6 of 6 items carry options"))
        #expect(none.contains("instruments enabled: none"))

        let enabled = InstrumentActivation.savePinSummary(
            optionsItemCount: 6, itemCount: 6,
            instruments: ["sampledText", "answerTokenLogprob"])
        #expect(enabled.contains("instruments enabled: answerTokenLogprob"))
        // sampledText is not a categorical instrument — it never appears in
        // the enabled list this summary is about.
        #expect(!enabled.contains("instruments enabled: sampledText"))

        let plain = InstrumentActivation.savePinSummary(
            optionsItemCount: 0, itemCount: 4, instruments: nil)
        #expect(plain.contains("no per-item instrument fields"))
    }

    // MARK: outcome-mode mapping (picker ↔ manifest, one rule)

    @Test func outcomeModeWritesAndReadsBack() {
        typealias Mode = InstrumentActivation.OutcomeMode
        #expect(Mode.generatedChoice.instruments == ["sampledText"])
        #expect(Mode.answerTokenProbability.instruments == ["answerTokenLogprob"])
        #expect(Mode.both.instruments == ["sampledText", "answerTokenLogprob"])
        #expect(Mode.notDeclared.instruments == nil)
        for mode in [Mode.generatedChoice, .answerTokenProbability, .both] {
            #expect(Mode.from(mode.instruments) == mode)
        }
        #expect(Mode.from(nil) == .notDeclared)
        #expect(Mode.from([]) == .notDeclared)
        // Auxiliaries are INVISIBLE to the mode axes (F3): a reader-only
        // list carries no picker-owned declaration, and a reader never
        // drags a categorical-only declaration onto the sampled axis.
        #expect(Mode.from(["repeReaderScore"]) == .notDeclared)
        #expect(
            Mode.from(["repeReaderScore", "choiceProbability"])
                == .answerTokenProbability)
    }

    @Test func changingOutcomeModePreservesAuxiliaryReaderInstruments() {
        typealias Mode = InstrumentActivation.OutcomeMode
        #expect(
            InstrumentActivation.applying(
                .both, to: ["repeReaderScore", "sampledText"])
                == ["repeReaderScore", "sampledText", "answerTokenLogprob"])
        #expect(
            InstrumentActivation.applying(
                .answerTokenProbability,
                to: ["sampledText", "repeReaderScore", "choiceProbability"])
                == ["repeReaderScore", "answerTokenLogprob"])
        #expect(
            InstrumentActivation.applying(.notDeclared, to: ["repeReaderScore"])
                == ["repeReaderScore"])
        // F3: the preserved reader must NOT bend the read-back — the picker
        // holds the mode that was just selected.
        #expect(
            Mode.from(["repeReaderScore", "answerTokenLogprob"])
                == .answerTokenProbability)
    }

    /// F3 — the round-trip invariant the file's "one rule for both
    /// directions" comment promises: `from(applying(m, x)) == m` for every
    /// selectable mode m and ANY auxiliary content x, and re-applying the
    /// same mode never rewrites the manifest (no silent rewrite loop).
    @Test func outcomeModeRoundTripHoldsForAnyAuxiliaryContent() {
        typealias Mode = InstrumentActivation.OutcomeMode
        let existingLists: [[String]?] = [
            nil, [],
            ["sampledText"],
            ["answerTokenLogprob"],
            ["choiceProbability"],
            ["sampledText", "answerTokenLogprob"],
            ["repeReaderScore"],
            ["sampledText", "repeReaderScore"],
            ["repeReaderScore", "answerTokenLogprob"],
            ["sampledText", "repeReaderScore", "choiceProbability"],
            ["someFutureAuxiliary", "sampledText"],
        ]
        for mode in [Mode.generatedChoice, .answerTokenProbability, .both] {
            for existing in existingLists {
                let written = InstrumentActivation.applying(mode, to: existing)
                #expect(Mode.from(written) == mode)
                // Idempotent: reselecting the mode is a byte-level no-op.
                #expect(InstrumentActivation.applying(mode, to: written) == written)
            }
        }
    }

    /// Review F3 acceptance #3, at the rule level: a reader study
    /// (`[sampledText, repeReaderScore]`) switched to Answer-token
    /// probability keeps the reader, reads back the selected mode (no
    /// snap-back to "Both"), and holds on reselect.
    @Test func readerStudyAnswerTokenSelectionHoldsAndIsIdempotent() {
        typealias Mode = InstrumentActivation.OutcomeMode
        let written = InstrumentActivation.applying(
            .answerTokenProbability, to: ["sampledText", "repeReaderScore"])
        #expect(written == ["repeReaderScore", "answerTokenLogprob"])
        #expect(Mode.from(written) == .answerTokenProbability)
        #expect(
            InstrumentActivation.applying(.answerTokenProbability, to: written)
                == written)
    }

    // MARK: auxiliary instruments — presentation + sampling implication (F3)

    @Test func auxiliaryInstrumentsAreExtractedInOrderAndDeduplicated() {
        #expect(
            InstrumentActivation.auxiliaryInstruments(
                of: ["sampledText", "repeReaderScore", "answerTokenLogprob"])
                == ["repeReaderScore"])
        #expect(InstrumentActivation.auxiliaryInstruments(of: nil).isEmpty)
        #expect(
            InstrumentActivation.auxiliaryInstruments(
                of: ["sampledText", "answerTokenLogprob", "choiceProbability"])
                .isEmpty)
        #expect(
            InstrumentActivation.auxiliaryInstruments(
                of: ["repeReaderScore", "repeReaderScore"]) == ["repeReaderScore"])
    }

    @Test func readerAuxiliaryImpliesSampledGenerationAndSaysSo() {
        // The rule mirrors the run loops' wantsSampled dispatch.
        #expect(
            InstrumentActivation.auxiliaryImpliesSampledGeneration("repeReaderScore"))
        #expect(
            !InstrumentActivation.auxiliaryImpliesSampledGeneration("sampledText"))
        #expect(
            !InstrumentActivation.auxiliaryImpliesSampledGeneration(
                "someFutureAuxiliary"))
        let description = InstrumentActivation.auxiliaryDescription("repeReaderScore")
        #expect(description.contains("scores generated responses"))
        #expect(description.contains(
            "sampled generation still runs even in Answer-token mode"))
    }

    @Test func effectiveRecordKindsNoteFiresOnlyWhenModeUnderstatesTheRun() {
        // Answer-token-only mode + reader → both record kinds, stated.
        let note = InstrumentActivation.effectiveRecordKindsNote(
            instruments: ["repeReaderScore", "answerTokenLogprob"])
        #expect(note?.contains(
            "answer-token logprobs AND sampled generations") == true)
        #expect(note?.contains("repeReaderScore") == true)
        // Mode already includes sampled text → nothing understated.
        #expect(
            InstrumentActivation.effectiveRecordKindsNote(
                instruments: ["sampledText", "repeReaderScore", "answerTokenLogprob"])
                == nil)
        // No auxiliary → a logprob-only declaration really is logprob-only.
        #expect(
            InstrumentActivation.effectiveRecordKindsNote(
                instruments: ["answerTokenLogprob"]) == nil)
        #expect(InstrumentActivation.effectiveRecordKindsNote(instruments: nil) == nil)
        // Reader-only lists read back notDeclared (sampled by default) —
        // the mode does not claim answer-token-only, so no note.
        #expect(
            InstrumentActivation.effectiveRecordKindsNote(
                instruments: ["repeReaderScore"]) == nil)
    }

    // MARK: headless readiness row (`data check` surface)

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "p1-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        return try body(temp)
    }

    private func plantPrompts(_ root: URL, withOptions: Bool) throws -> String {
        let file = "prompts/tasks/p1-prompts.jsonl"
        let url = root.appending(path: file)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        // `target` travels with `options`: a target-dependent instrument
        // reads the DECLARED target's log-odds, and a file that names none is
        // a run-start refusal (open-issues #6) which the readiness row now
        // states verbatim.
        let line =
            withOptions
            ? #"{"prompt": "decide", "options": ["affirm", "reverse"], "target": "affirm"}"#
            : #"{"prompt": "decide"}"#
        try (line + "\n").write(to: url, atomically: true, encoding: .utf8)
        return file
    }

    @Test func optionsWithoutDeclaredInstrumentIsAPartialReadinessRow() throws {
        try withTempWorkspace { root in
            var manifest = ExperimentManifest(
                name: "p1", description: "d", modelID: "test/model")
            manifest.taskPromptsFile = try plantPrompts(root, withOptions: true)

            // No instrument declared → partial, with the honest sentence.
            let rows = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: root)
            let prompts = rows.first { $0.id == "taskPrompts" }
            #expect(prompts?.status == .partial)
            #expect(prompts?.detail.contains(
                "no categorical outcome instrument is declared") == true)
            #expect(prompts?.detail.contains(
                "only generate and parse answer text") == true)
            // Partial is never a blocker.
            let summary = StudyDataReadiness.summary(rows)
            #expect(!summary.blockers.contains { $0.id == "taskPrompts" })

            // Declaring the instrument turns the same file present.
            manifest.outcomeInstruments = ["answerTokenLogprob"]
            let declaredRows = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: root)
            #expect(
                declaredRows.first { $0.id == "taskPrompts" }?.status == .present)
        }
    }

    @Test func promptsWithoutOptionsStayPresent() throws {
        try withTempWorkspace { root in
            var manifest = ExperimentManifest(
                name: "p1b", description: "d", modelID: "test/model")
            manifest.taskPromptsFile = try plantPrompts(root, withOptions: false)
            let rows = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: root)
            #expect(rows.first { $0.id == "taskPrompts" }?.status == .present)
        }
    }

    // MARK: F3 acceptance — picker + toggle write through the store

    /// Review F3 acceptance #3, end to end through the panel: a reader
    /// study switched to Answer-token probability keeps the reader, reads
    /// back the selected mode, holds on reselect, shows the auxiliary row
    /// and the effective-record-kinds note, and the remove/add toggle
    /// writes the manifest through the store.
    @MainActor
    @Test func evaluationPickerAndAuxiliaryToggleWriteThroughTheStore() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "f3") { _ in
            var manifest = ExperimentManifest(
                name: "f3-reader", description: "d", modelID: "test/model")
            manifest.outcomeInstruments = ["sampledText", "repeReaderScore"]
            manifest.readerRefs = [
                .init(path: "runs/readers/sympathy.json", hash: "00", concept: "sympathy")
            ]
            try ExperimentStore.save(manifest, allowCreate: true)

            let panel = ExperimentPanel()
            panel.selectedName = "f3-reader"
            #expect(panel.auxiliaryOutcomeInstruments == ["repeReaderScore"])
            #expect(panel.effectiveRecordKindsNote == nil)

            // Select Answer-token probability: manifest becomes
            // [repeReaderScore, answerTokenLogprob]; the picker HOLDS.
            panel.setOutcomeMode(.answerTokenProbability)
            #expect(
                panel.selected?.outcomeInstruments
                    == ["repeReaderScore", "answerTokenLogprob"])
            #expect(panel.outcomeMode == .answerTokenProbability)
            // The mode now understates the run — the note states both
            // record kinds.
            #expect(panel.effectiveRecordKindsNote?.contains(
                "sampled generation still runs") == true)
            // Reselecting rewrites nothing (idempotent — no rewrite loop).
            panel.setOutcomeMode(.answerTokenProbability)
            #expect(
                panel.selected?.outcomeInstruments
                    == ["repeReaderScore", "answerTokenLogprob"])

            // Removing the reader writes through the store and enables a
            // genuinely logprob-only run.
            panel.removeAuxiliaryInstrument("repeReaderScore")
            #expect(panel.selected?.outcomeInstruments == ["answerTokenLogprob"])
            #expect(panel.outcomeMode == .answerTokenProbability)
            #expect(panel.auxiliaryOutcomeInstruments.isEmpty)
            #expect(panel.effectiveRecordKindsNote == nil)

            // Re-adding is offered because readerRefs are pinned — and
            // writes through the store too.
            #expect(panel.canAddReaderInstrument)
            panel.addReaderInstrument()
            #expect(
                panel.selected?.outcomeInstruments
                    == ["answerTokenLogprob", "repeReaderScore"])
            #expect(!panel.canAddReaderInstrument)
            #expect(panel.outcomeMode == .answerTokenProbability)
        }
    }

    /// Without pinned readers the add affordance hides: declaring
    /// `repeReaderScore` with no `readerRefs` is an immediate verify
    /// violation, so the UI never invites one.
    @MainActor
    @Test func addReaderAffordanceRequiresPinnedReaderRefs() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "f3b") { _ in
            var manifest = ExperimentManifest(
                name: "f3-noreader", description: "d", modelID: "test/model")
            manifest.outcomeInstruments = ["answerTokenLogprob"]
            try ExperimentStore.save(manifest, allowCreate: true)
            let panel = ExperimentPanel()
            panel.selectedName = "f3-noreader"
            #expect(!panel.canAddReaderInstrument)
            // The guarded method is a no-op, not an error.
            panel.addReaderInstrument()
            #expect(panel.selected?.outcomeInstruments == ["answerTokenLogprob"])
        }
    }
}

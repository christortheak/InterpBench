import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The APP's writer for the two extraction declarations.
///
/// The engines have had both since 2026-08-25 — `--reading-position` over the
/// whole eight-role vocabulary, `--extraction-rendering` over raw and the chat
/// template, and typed refusals behind each — and the app could declare
/// NEITHER. The Studies attach row offered a `--pool-from` token index (the
/// legacy spelling of exactly one position) and the Concepts builder a pooling
/// toggle, so six of the eight roles and every template rendering were
/// reachable only from a command line. These are the pickers that close that,
/// and the properties worth pinning are the two that fail silently:
///
/// 1. **The default declares NOTHING.** An untouched row attaches the manifest
///    it always attached — no `readingPosition`, no `extractionRendering`, no
///    moved recipe identity. A picker that helpfully declared its own default
///    would rewrite every study's recipe the day it shipped.
/// 2. **The refusals are the ENGINE's, verbatim.** The picker validates
///    nothing: a selection becomes the cross-engine declaration text and
///    `ReadingPosition.declared` / `ExtractionRendering.declared` answer, so
///    the app can never accept what the CLI refuses (the assistant voice on
///    this engine) or refuse what it accepts.
@Suite struct ExtractionDeclarationChoiceTests {

    // MARK: - 1. the vocabulary

    /// Every declarable entry speaks a label the engine's own strict parser
    /// accepts, and displaying an existing position picks that entry back.
    @Test func everyChoiceSpeaksTheCrossEngineVocabularyAndRoundTrips() throws {
        let cases: [(ReadingPositionChoice, Int, ReadingPosition)] = [
            (.lastToken, 0, .lastToken),
            (.meanFromToken, 50, .meanFromToken(50)),
            (.offsetFromEnd, 3, .offsetFromEnd(3)),
            (.lastContentToken, 0, .lastContentToken),
            (.turnCloseToken, 0, .turnCloseToken),
            (.postInstruction, 2, .postInstruction(2)),
            (.contentOffset, 2, .contentOffset(2)),
            (.meanContentFromToken, 4, .meanContentFromToken(4)),
        ]
        #expect(ReadingPositionChoice.declarableCases.count == cases.count)
        for (choice, parameter, expected) in cases {
            #expect(choice.declarationLabel(parameter: parameter) == expected.label)
            #expect(try choice.declaredPosition(parameter: parameter) == expected)
            let (back, backParameter) = ReadingPositionChoice.choice(for: expected)
            #expect(back == choice, "\(expected.label)")
            if choice.takesParameter {
                #expect(backParameter == parameter, "\(expected.label)")
            }
        }
    }

    /// The default entry declares NOTHING — the byte-identity contract, not an
    /// omission.
    @Test func theRecipeDefaultEntryDeclaresNothing() throws {
        #expect(ReadingPositionChoice.recipeDefault.declarationLabel(parameter: 7) == nil)
        #expect(try ReadingPositionChoice.recipeDefault.declaredPosition(parameter: 7) == nil)
        #expect(!ReadingPositionChoice.declarableCases.contains(.recipeDefault))
    }

    /// An out-of-vocabulary parameter is refused HERE, in the engine's words,
    /// rather than clamped into a recipe nobody asked for.
    @Test func anOutOfVocabularyParameterIsRefusedInTheEnginesWords() throws {
        #expect(throws: ReadingPosition.DeclarationError.self) {
            try ReadingPositionChoice.postInstruction.declaredPosition(parameter: 9)
        }
        do {
            _ = try ReadingPositionChoice.postInstruction.declaredPosition(parameter: 9)
            Issue.record("post-instruction 9 was accepted")
        } catch let error as ReadingPosition.DeclarationError {
            #expect(error.reason.contains(ReadingPosition.declarationEngine))
            #expect(error.repair.contains("post-instruction"))
        }
    }

    /// Moving the picker to a kind whose parameter space is different carries
    /// the number into range — a convenience, never a validation: the field
    /// still accepts anything, and the engine still answers.
    @Test func changingKindStepsTheParameterIntoThatKindsRange() {
        #expect(ReadingPositionChoice.postInstruction.steppedParameter(from: 0) == 1)
        #expect(ReadingPositionChoice.postInstruction.steppedParameter(from: 50) == 5)
        #expect(ReadingPositionChoice.meanFromToken.steppedParameter(from: 50) == 50)
        #expect(ReadingPositionChoice.lastToken.steppedParameter(from: 50) == 50)
    }

    // MARK: - 2. the rendering

    /// Raw declares nothing; a chat template comes back as its STAMP, with the
    /// resolved defaults written out.
    @Test func rawDeclaresNothingAndChatTemplateStampsItsDefaults() throws {
        #expect(try ExtractionRenderingChoice().declared() == nil)
        #expect(ExtractionRenderingChoice().isRaw)
        #expect(ExtractionRenderingChoice().declaration == nil)

        let templated = ExtractionRenderingChoice(mode: .chatTemplate)
        let declared = try #require(try templated.declared())
        #expect(declared.mode == .chatTemplate)
        #expect(declared.addGenerationPrompt == true)
        #expect(declared.reasoningEffort == .off)
        #expect(declared.resolvedQwenThinkingEnabled == false)
        // Absent ≡ user: the legacy voice stamps no key at all.
        #expect(declared.voice == nil)
        #expect(declared.resolvedVoice == .user)
    }

    /// THE ENGINE ASYMMETRY. The control offers the assistant voice — it is a
    /// real axis, and the server renders it — and this engine's refusal is
    /// what a person sees, in the engine's own words, naming the engine that
    /// can.
    @Test func theAssistantVoiceIsRefusedByTheEngineNotByThePicker() throws {
        let choice = ExtractionRenderingChoice(mode: .chatTemplate, voice: .assistant)
        // The declaration omits addGenerationPrompt: it reaches nothing under
        // this voice, and declaring it is malformed on BOTH engines.
        let declaration = try #require(choice.declaration)
        #expect(declaration["addGenerationPrompt"] == nil)
        #expect(declaration["voice"] as? String == "assistant")
        do {
            _ = try choice.declared()
            Issue.record("the assistant voice was accepted on this engine")
        } catch let error as ExtractionRendering.DeclarationError {
            #expect(error.reason == PromptRendering.assistantVoiceReason)
            #expect(error.repair.contains("python-hf-transformers"))
        }
    }

    /// The other engine asymmetry, on the same terms.
    @Test func withholdingTheGenerationPromptIsRefusedInTheEnginesWords() throws {
        let choice = ExtractionRenderingChoice(
            mode: .chatTemplate, addGenerationPrompt: false)
        do {
            _ = try choice.declared()
            Issue.record("addGenerationPrompt false was accepted on this engine")
        } catch let error as ExtractionRendering.DeclarationError {
            #expect(error.reason == PromptRendering.addGenerationPromptFalseReason)
        }
    }

    @Test func theRenderingPickerDisplaysAnExistingRendering() {
        #expect(ExtractionRenderingChoice.choice(for: nil) == ExtractionRenderingChoice())
        #expect(
            ExtractionRenderingChoice.choice(for: .raw) == ExtractionRenderingChoice())
        let templated = ExtractionRenderingChoice.choice(
            for: .chatTemplate(addGenerationPrompt: true))
        #expect(templated.mode == .chatTemplate)
        #expect(templated.voice == .user)
        #expect(templated.addGenerationPrompt)
    }
}

// MARK: - The Studies attach row

/// The attach row writes through `ExperimentStore.attachConcept`, so every
/// declaration-time refusal the store already holds must arrive at the panel's
/// notice feed rather than being swallowed by a generic "check your files".
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite. The paired
/// checks reference the committed `french` concept read-only, exactly as the
/// reading-position attach suite does.
@Suite(.serialized) @MainActor struct AttachDeclarationPanelTests {

    static let model = "mlx-community/gemma-3-4b-it-4bit"

    private func withPanel<T>(
        _ body: (ExperimentPanel, URL) throws -> T
    ) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "attach-pickers") { root in
            let panel = ExperimentPanel()
            panel.notices = PanelNotices(fileURL: nil)
            return try body(panel, root)
        }
    }

    @discardableResult
    private func draft(_ panel: ExperimentPanel, named name: String) throws
        -> ExperimentManifest
    {
        let manifest = try ExperimentStore.create(
            name: name, description: "", modelID: Self.model)
        panel.refresh()
        panel.selectedName = name
        panel.attachConceptName = "french"
        panel.attachMethod = .meanDifference
        return manifest
    }

    private func noticeText(_ panel: ExperimentPanel) -> String {
        panel.notices.notices.map(\.message).joined(separator: "\n")
            + "\n" + (panel.status ?? "")
    }

    /// PROPERTY 1. An untouched row declares nothing: no rendering key, the
    /// method's own reading position, and a manifest whose concept entry is
    /// byte-identical to the one a no-declaration store attach writes.
    @Test func defaultSelectionsDeclareNothingAndKeepTheManifestBytes() throws {
        try withPanel { panel, _ in
            try draft(panel, named: "picker-default")
            #expect(panel.attachReadingPositionChoice == .recipeDefault)
            #expect(panel.attachRendering.isRaw)
            panel.attachConceptFromPicker()

            let ref = try #require(
                try ExperimentStore.load(name: "picker-default").concepts.first)
            #expect(ref.options.readingPosition == .lastToken)
            #expect(ref.options.extractionRendering == nil)

            _ = try ExperimentStore.create(
                name: "cli-default", description: "", modelID: Self.model)
            _ = try ExperimentStore.attachConcept(
                "french", method: .meanDifference, experimentName: "cli-default")
            let cliRef = try #require(
                try ExperimentStore.load(name: "cli-default").concepts.first)

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            #expect(try encoder.encode(ref) == encoder.encode(cliRef))
            // The claim behind "byte-identical": the keys are ABSENT, not
            // present-and-defaulted.
            let json = String(
                data: try encoder.encode(ref.options), encoding: .utf8) ?? ""
            #expect(!json.contains("extractionRendering"))
        }
    }

    /// Both declarations reach the store, and the success notice says what was
    /// pinned — a raw rendering staying unmentioned exactly like an absent one.
    @Test func aDeclaredPositionAndRenderingReachTheStore() throws {
        try withPanel { panel, _ in
            try draft(panel, named: "picker-declared")
            panel.attachReadingPositionChoice = .contentOffset
            panel.attachReadingPositionParameter = 2
            panel.attachRendering = ExtractionRenderingChoice(mode: .chatTemplate)
            panel.attachConceptFromPicker()

            let ref = try #require(
                try ExperimentStore.load(name: "picker-declared").concepts.first)
            #expect(ref.options.readingPosition == .contentOffset(2))
            let rendering = try #require(ref.options.extractionRendering)
            #expect(rendering.mode == .chatTemplate)
            #expect(rendering.resolvedAddGenerationPrompt)
            let notice = noticeText(panel)
            #expect(notice.contains("content offset 2"))
            #expect(notice.contains("chatTemplate"))
        }
    }

    /// The store's template-aware refusal — a role that could never resolve
    /// under a raw rendering — arrives VERBATIM, with its repair, and nothing
    /// is written.
    @Test func aTemplateAwarePositionUnderRawRenderingLandsInTheNotices() throws {
        try withPanel { panel, _ in
            try draft(panel, named: "picker-refused-position")
            panel.attachReadingPositionChoice = .lastContentToken
            panel.attachConceptFromPicker()

            #expect(try ExperimentStore.load(name: "picker-refused-position")
                .concepts.isEmpty)
            let refusal = try #require(
                ReadingPosition.templatedRenderingRefusal(.lastContentToken))
            let reason = String(
                refusal[refusal.startIndex ..< (refusal.range(of: " — repair: ")?
                    .lowerBound ?? refusal.endIndex)])
            #expect(noticeText(panel).contains(reason))
        }
    }

    /// THE ENGINE ASYMMETRY, at the row. The control offers the assistant
    /// voice; the engine's own refusal is what lands, naming the engine that
    /// can render it — and nothing is attached.
    @Test func theAssistantVoiceRefusalReachesTheNoticeAndNothingIsAttached() throws {
        try withPanel { panel, _ in
            try draft(panel, named: "picker-refused-voice")
            panel.attachRendering = ExtractionRenderingChoice(
                mode: .chatTemplate, voice: .assistant)
            panel.attachConceptFromPicker()

            #expect(try ExperimentStore.load(name: "picker-refused-voice")
                .concepts.isEmpty)
            #expect(noticeText(panel).contains("python-hf-transformers"))
            #expect(noticeText(panel).contains(ReadingPosition.declarationEngine))
        }
    }

    /// The pin-status line gained the rendering — and a RAW rendering stays
    /// unmentioned, exactly as an absent one does, because they are one recipe.
    @Test func thePinStatusLineNamesANonRawRenderingOnly() throws {
        var options = ExtractionOptions(method: .meanDifference)
        options.readingPosition = .lastContentToken
        var ref = ExperimentManifest.ConceptRef(
            name: "french", stimulusSetHash: String(repeating: "a", count: 64),
            options: options)
        let raw = ExperimentPanel.conceptPinStatusLine(ref)
        #expect(raw.contains("last content token"))
        #expect(!raw.contains("chatTemplate"))
        #expect(!raw.contains("raw"))

        ref.options.extractionRendering = ExtractionRendering.chatTemplate().stamp
        let templated = ExperimentPanel.conceptPinStatusLine(ref)
        #expect(templated.contains("chatTemplate"))
    }
}

// MARK: - The Concepts builder

/// The builder's control replaces a pooling toggle that could reach two of the
/// eight positions and no rendering at all. Its default must still extract
/// what it always extracted.
@Suite(.serialized) @MainActor struct BuilderDeclarationPickerTests {

    private func withBuilder<T>(_ body: (ConceptBuilder) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "builder-pickers") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            return try body(ConceptBuilder())
        }
    }

    /// PROPERTY 1, builder side: untouched, it reads at the last token under
    /// the raw rendering and declares no rendering at all.
    @Test func theBuilderDefaultDeclaresNothing() throws {
        try withBuilder { builder in
            #expect(builder.readingPositionChoice == .lastToken)
            #expect(builder.readingPosition == .lastToken)
            #expect(builder.poolFromToken == nil)
            #expect(builder.extractionRendering == nil)
            #expect(builder.extractionOptions.readingPosition == .lastToken)
            #expect(builder.extractionOptions.extractionRendering == nil)
            #expect(builder.extractionOptions.resolvedExtractionRendering.isRaw)
            #expect(!builder.hasRefusedExtractionDeclaration)
        }
    }

    /// `poolFromToken` is the LEGACY spelling of exactly one position — the
    /// routes that type it (the web client, the server-extract calls) keep
    /// working, and the picker and it are one declaration, never two.
    @Test func poolFromTokenIsTheLegacySpellingOfOnePosition() throws {
        try withBuilder { builder in
            builder.poolFromToken = 50
            #expect(builder.readingPositionChoice == .meanFromToken)
            #expect(builder.readingPositionParameter == 50)
            #expect(builder.readingPosition == .meanFromToken(50))

            builder.poolFromToken = nil
            #expect(builder.readingPositionChoice == .lastToken)
            #expect(builder.readingPosition == .lastToken)

            builder.readingPositionChoice = .lastContentToken
            #expect(builder.poolFromToken == nil)
            #expect(builder.readingPosition == .lastContentToken)
        }
    }

    /// The grand-mean family's token-50 policy still lands through the new
    /// control — that assignment is the one place the pane declares a position
    /// on the researcher's behalf, and it must keep doing it.
    @Test func theGrandMeanFamilyStillPinsItsPooledPolicy() throws {
        try withBuilder { builder in
            builder.recipeFamily = .emotionGrandMean
            #expect(builder.poolFromToken == 50)
            #expect(builder.extractionOptions.readingPosition == .meanFromToken(50))
            builder.recipeFamily = .caaMeanDifference
            #expect(builder.poolFromToken == nil)
            #expect(builder.extractionOptions.readingPosition == .lastToken)
        }
    }

    /// A declared template rendering reaches the options every extraction path
    /// here reads — and the activation cache keys on it, because a raw read
    /// and a templated read are two measurements, not one.
    @Test func aDeclaredRenderingReachesTheOptionsAndTheActivationCacheKey() throws {
        try withBuilder { builder in
            builder.extractionRenderingChoice = ExtractionRenderingChoice(
                mode: .chatTemplate)
            let rendering = try #require(builder.extractionRendering)
            #expect(rendering.mode == .chatTemplate)
            #expect(builder.extractionOptions.extractionRendering != nil)

            var raw = builder.extractionOptions
            raw.extractionRendering = nil
            #expect(
                ConceptBuilder.activationCacheKey(
                    modelID: "m", options: raw, text: "t")
                    != ConceptBuilder.activationCacheKey(
                        modelID: "m", options: builder.extractionOptions, text: "t"))
        }
    }

    /// A standing refusal is the ENGINE's words, and it turns the build off:
    /// extracting under the last valid declaration would write a vector nobody
    /// asked for.
    @Test func aRefusedDeclarationIsHeldAndDisablesTheBuild() throws {
        try withBuilder { builder in
            builder.extractionRenderingChoice = ExtractionRenderingChoice(
                mode: .chatTemplate, voice: .assistant)
            let refusal = try #require(builder.extractionRenderingRefusal)
            #expect(refusal.contains(PromptRendering.assistantVoiceReason))
            #expect(builder.hasRefusedExtractionDeclaration)
            #expect(!builder.canSaveAndExtract)

            builder.extractionRenderingChoice = ExtractionRenderingChoice()
            #expect(builder.extractionRenderingRefusal == nil)
            #expect(!builder.hasRefusedExtractionDeclaration)
            #expect(builder.extractionRendering == nil)
        }
    }

    /// The position control refuses out-of-vocabulary parameters on the same
    /// terms, and holds the last valid position while it does.
    @Test func anOutOfVocabularyPositionParameterIsHeldAsARefusal() throws {
        try withBuilder { builder in
            builder.readingPositionChoice = .postInstruction
            // Moving to a kind carries the number into its range — the one
            // convenience the control allows itself.
            #expect(builder.readingPositionParameter == 1)
            #expect(builder.readingPositionRefusal == nil)
            builder.readingPositionParameter = 9
            #expect(builder.readingPositionRefusal != nil)
            #expect(builder.hasRefusedExtractionDeclaration)
            builder.readingPositionParameter = 2
            #expect(builder.readingPositionRefusal == nil)
            #expect(builder.readingPosition == .postInstruction(2))
        }
    }
}

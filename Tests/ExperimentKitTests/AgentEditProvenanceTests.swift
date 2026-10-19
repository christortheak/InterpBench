import Foundation
import Testing

@testable import ExperimentKit

/// "Save Changes" in the agent editor rebuilt the agent from the editor's
/// controls and wrote it over the file, so a promoted agent lost its
/// promotion record even when nothing was edited. An edit that leaves what
/// the agent does unchanged now keeps the record; one that changes it drops
/// the record and says so, because the record would vouch for settings it
/// never tested.
struct AgentEditProvenanceTests {

    private func stored() -> ModelVariantArtifact {
        var artifact = ModelVariantArtifact(
            name: "french-l28-a15", baseModelID: "test/model",
            injections: [
                .init(concept: "french", vectorArtifactID: "runs/sweep/french",
                      layer: 28, alpha: 0.15)
            ],
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "",
            promotion: ModelVariantArtifact.Promotion(
                experiment: "concept-sweep", experimentHash: "h",
                promotedAt: "2026-10-19T11:34:51Z", promotedBy: "manualOverride",
                substrate: "swift-mlx", appVersion: "test"))
        artifact.createdAt = "2026-10-19T11:34:51Z"
        return artifact
    }

    /// What the editor produces: the same settings, no record of making.
    private func fromEditor(_ base: ModelVariantArtifact) -> ModelVariantArtifact {
        var edited = base
        edited.promotion = nil
        edited.createdAt = "2026-10-20T00:00:00Z"
        return edited
    }

    @Test func savingWithoutEditsKeepsThePromotionRecord() {
        let original = stored()
        let merge = AgentEditProvenance.merge(edited: fromEditor(original), over: original)
        #expect(!merge.droppedPromotion)
        #expect(merge.artifact == original)
    }

    @Test func aRenameKeepsThePromotionRecord() {
        let original = stored()
        var edited = fromEditor(original)
        edited.name = "french-agent"
        // The editor refreshes the concept label from the vector's metadata;
        // a label is not behaviour.
        edited.injections[0].concept = "French"
        let merge = AgentEditProvenance.merge(edited: edited, over: original)
        #expect(!merge.droppedPromotion)
        #expect(merge.artifact.promotion == original.promotion)
        #expect(merge.artifact.name == "french-agent")
        #expect(merge.artifact.createdAt == original.createdAt)
        // The stored label survives: a verify gate compares it with the
        // sweep's concept, so a no-op save must not rewrite it.
        #expect(merge.artifact.injections[0].concept == "french")
        var renamedBack = merge.artifact
        renamedBack.name = original.name
        #expect(renamedBack == original)
    }

    @Test func changingTheStrengthDropsThePromotionRecordAndSaysSo() {
        let original = stored()
        var edited = fromEditor(original)
        edited.injections[0].alpha = 0.2
        let merge = AgentEditProvenance.merge(edited: edited, over: original)
        #expect(merge.droppedPromotion)
        #expect(merge.artifact.promotion == nil)
        #expect(merge.artifact.createdAt == original.createdAt)
    }

    @Test func anAgentWithNoRecordReportsNothingDropped() {
        var original = stored()
        original.promotion = nil
        var edited = fromEditor(original)
        edited.temperature = 0.7
        #expect(!AgentEditProvenance.merge(edited: edited, over: original).droppedPromotion)
    }

    /// An older file stores only the system prompt's hash, and the editor shows
    /// that as an empty prompt. Saving without touching it keeps the hash.
    @Test func anOlderFilesPromptHashSurvivesAnUntouchedPrompt() {
        var original = stored()
        original.systemPrompt = nil
        original.systemPromptHash = "abc"
        var edited = fromEditor(original)
        edited.systemPrompt = ""
        edited.systemPromptHash = nil
        let merge = AgentEditProvenance.merge(edited: edited, over: original)
        #expect(merge.artifact.systemPrompt == nil)
        #expect(merge.artifact.systemPromptHash == "abc")
        #expect(!merge.droppedPromotion)
    }

    /// The Python engine writes a prompt's text without its hash, and freeze
    /// refuses such an agent with "re-save the variant". Re-saving must stamp
    /// the hash, and stamping it is not an edit, so the record stays.
    @Test func reSavingStampsAMissingPromptHashAndKeepsTheRecord() {
        var original = stored()
        original.systemPrompt = "Answer briefly."
        original.systemPromptHash = nil
        var edited = fromEditor(original)
        edited.systemPromptHash = ModelVariantArtifact(
            name: "x", baseModelID: "test/model", promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "Answer briefly."
        ).systemPromptHash
        #expect(edited.systemPromptHash != nil)
        let merge = AgentEditProvenance.merge(edited: edited, over: original)
        #expect(!merge.droppedPromotion)
        #expect(merge.artifact.promotion == original.promotion)
        #expect(merge.artifact.systemPromptHash == edited.systemPromptHash)
        #expect(merge.artifact.systemPrompt == "Answer briefly.")
    }

    /// Spellings that mean the same thing are not an edit: an explicit
    /// steering mode, empty ablation layers (every layer), absent centering
    /// (none), and an adapter's display name.
    @Test func equivalentSpellingsAreNotAnEdit() {
        var original = stored()
        original.injections.append(
            .init(concept: "fear", vectorArtifactID: "runs/sweep/fear", layer: 3,
                  alpha: 1, mode: .ablate, layers: nil, centering: nil))
        var edited = fromEditor(original)
        edited.injections[0].mode = .add
        edited.injections[1].layers = []
        edited.injections[1].centering = "none"
        #expect(!AgentEditProvenance.merge(edited: edited, over: original).droppedPromotion)

        // A real change to the ablation still counts.
        edited.injections[1].centering = "neutralMean"
        #expect(AgentEditProvenance.merge(edited: edited, over: original).droppedPromotion)
    }

    /// The question names the way to keep the original, and the after-save
    /// note says what happened without advice that would come too late.
    @Test func theQuestionOffersSaveAsNewAgent() {
        #expect(AgentEditProvenance.dialogMessage().contains("Save as New Agent"))
        #expect(AgentEditProvenance.dialogMessage()
            .contains("Save Changes and Remove the Record"))
        #expect(AgentEditProvenance.afterSaveNote(agent: "a").hasPrefix("Updated"))
        #expect(!AgentEditProvenance.afterSaveNote(agent: "a").contains("To keep"))
    }

    // MARK: Agents a run saved

    /// A library agent is updated in place, by the same merge as before; an
    /// agent a run saved never is. The library case is the control: the same
    /// edit over the same stored agent differs only in where the file is.
    @Test func onlyALibraryAgentIsUpdatedInPlace() {
        let original = stored()
        var edited = fromEditor(original)
        edited.injections[0].alpha = 0.2
        #expect(AgentEditProvenance.saveChangesPlan(
            edited: edited, over: original, inLibrary: true)
            == .updateInPlace(AgentEditProvenance.merge(edited: edited, over: original)))
        #expect(AgentEditProvenance.saveChangesPlan(
            edited: edited, over: original, inLibrary: false) == .saveAsNewAgent)

        // No promotion record, and an edit that drops nothing: the case that
        // went through the in-place write with no question asked.
        var plain = original
        plain.promotion = nil
        var renamed = fromEditor(plain)
        renamed.name = "french-agent"
        #expect(!AgentEditProvenance.merge(edited: renamed, over: plain).droppedPromotion)
        #expect(AgentEditProvenance.saveChangesPlan(
            edited: renamed, over: plain, inLibrary: false) == .saveAsNewAgent)
        guard case .updateInPlace = AgentEditProvenance.saveChangesPlan(
            edited: renamed, over: plain, inLibrary: true)
        else {
            Issue.record("a library agent's rename is no longer saved in place")
            return
        }
    }

    /// An unedited agent a run saved has nothing to save. A library agent
    /// still goes through the in-place save (which writes its own content
    /// back), and an edit, however small, is something to save.
    @Test func anUneditedRunAgentHasNothingToSave() {
        let original = stored()
        #expect(AgentEditProvenance.saveChangesPlan(
            edited: fromEditor(original), over: original, inLibrary: false) == .nothingToSave)
        #expect(AgentEditProvenance.saveChangesPlan(
            edited: fromEditor(original), over: original, inLibrary: true)
            == .updateInPlace(.init(artifact: original, droppedPromotion: false)))
        var edited = fromEditor(original)
        edited.temperature = 0.1
        #expect(AgentEditProvenance.saveChangesPlan(
            edited: edited, over: original, inLibrary: false) == .saveAsNewAgent)
    }

    /// Freeze refuses a system prompt stored without its hash and asks for a
    /// re-save, and the Python engine writes exactly that. For an agent a run
    /// saved, the re-save is a new agent that carries the hash, never
    /// "nothing to save".
    @Test func aRunAgentMissingItsPromptHashCanStillBeReSaved() {
        var original = stored()
        original.systemPrompt = "Answer briefly."
        original.systemPromptHash = nil
        var edited = fromEditor(original)
        edited.systemPromptHash = ModelVariantArtifact(
            name: "x", baseModelID: "test/model", promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "Answer briefly."
        ).systemPromptHash
        #expect(AgentEditProvenance.saveChangesPlan(
            edited: edited, over: original, inLibrary: false) == .saveAsNewAgent)
        // Control: once the hash is stored, the same save has nothing to do.
        original.systemPromptHash = edited.systemPromptHash
        #expect(AgentEditProvenance.saveChangesPlan(
            edited: edited, over: original, inLibrary: false) == .nothingToSave)
    }

    /// What the button and the editor say before and after a save. The
    /// library agent's words are what they were.
    @Test func theEditorSaysWhatSavingARunAgentDoes() {
        #expect(AgentEditProvenance.saveButtonTitle(editing: false, inLibrary: true)
            == "Save New Agent")
        #expect(AgentEditProvenance.saveButtonTitle(editing: true, inLibrary: true)
            == "Save Changes")
        #expect(AgentEditProvenance.saveButtonTitle(editing: true, inLibrary: false)
            == "Save as New Agent")
        #expect(AgentEditProvenance.savedByRunButtonHelp.contains("evidence"))
        #expect(AgentEditProvenance.savedByRunButtonHelp.contains("new agent"))

        #expect(AgentEditProvenance.editingCaption(agent: "a", savedByRun: nil)
            == "editing the saved agent 'a' — Save Changes updates it in place; "
                + "choose New… above to start a fresh definition instead")
        let runCaption = AgentEditProvenance.editingCaption(
            agent: "a", savedByRun: "20261019T000000000-variant-a")
        #expect(runCaption.contains("20261019T000000000-variant-a"))
        #expect(runCaption.contains("evidence") && runCaption.contains("Save as New Agent"))
        #expect(!runCaption.contains("in place"))

        #expect(AgentEditProvenance.savedAsNewAgentNote(
            agent: "a-edited", original: "a", savedByRun: false, originalHasRecord: true)
            == "Saved a new agent 'a-edited'; 'a' is unchanged.")
        let fromRun = AgentEditProvenance.savedAsNewAgentNote(
            agent: "a-edited", original: "a", savedByRun: true, originalHasRecord: true)
        #expect(fromRun.contains("'a-edited'") && fromRun.contains("evidence"))
        #expect(fromRun.contains("promotion record"))
        #expect(!AgentEditProvenance.savedAsNewAgentNote(
            agent: "a-edited", original: "a", savedByRun: true, originalHasRecord: false)
            .contains("promotion record"))
        #expect(AgentEditProvenance.nothingToSaveNote(agent: "a").hasPrefix("Nothing to save"))
    }

    /// Save as New Agent never reuses a library name, so the original keeps
    /// its name and its record.
    @Test func theCopyGetsAFreeName() {
        #expect(AgentEditProvenance.copyName("fresh", taken: ["french-l28-a15"]) == "fresh")
        #expect(AgentEditProvenance.copyName("french-l28-a15", taken: ["French-L28-A15"])
            == "french-l28-a15-edited")
        #expect(AgentEditProvenance.copyName(
            "french-l28-a15", taken: ["french-l28-a15", "french-l28-a15-edited"])
            == "french-l28-a15-edited-2")
    }
}


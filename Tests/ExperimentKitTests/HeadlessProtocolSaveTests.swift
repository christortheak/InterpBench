import Foundation
import Testing

@testable import ExperimentKit

/// open-issues §8, residual (b) — the model-change reset in
/// `ExperimentPanel.saveProtocol` must be INTENTIONAL, never incidental.
///
/// The hazard: `studyBaseModelID` is a panel FIELD synced from the selected
/// manifest, and `saveProtocol` reads a mismatch against it as "the
/// researcher changed the base model" — which clears the revision pin and
/// every variant condition. The headless route
/// (`POST /api/experiment/protocol`) never sets that field, so a protocol
/// save from a client that names no model could compare the manifest against
/// whatever the panel last synced and silently drop the study's arms.
///
/// This sits deliberately OUTSIDE the `armsCleared` guard: that guard
/// watches the concepts+conditions pair, and clearing only the variant arms
/// is a legitimate edit (see `ExperimentStore.holdsArms`).
@Suite(.serialized) @MainActor struct HeadlessProtocolSaveTests {

    private static let variantJSON =
        #"{"name":"agent-a","fromPromotion":{"concept":"french"}}"#

    /// A draft whose whole measured surface lives in variant conditions —
    /// the agentComparison shape, the one the incidental reset destroys.
    @discardableResult
    private func variantDraft(named name: String) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "d", modelID: "test/model")
        manifest.variantConditions.append(
            try JSONDecoder().decode(
                ExperimentManifest.VariantCondition.self,
                from: Data(Self.variantJSON.utf8)))
        try ExperimentStore.save(manifest)
        return manifest
    }

    /// A panel selecting the draft, configured the way the headless route
    /// drives it (no task-prompts pin, so the save cannot throw on a
    /// missing default prompts file and silently test nothing).
    private func makePanel(selecting name: String) -> ExperimentPanel {
        let panel = ExperimentPanel()
        panel.selectedName = name
        panel.taskPromptsFile = ""
        return panel
    }

    // MARK: - The reproduction

    /// THE regression: a study with variant conditions survives a protocol
    /// save that does not name a new base model. The panel's model field is
    /// stale — as for a headless client that never synced it — and the save
    /// runs the exact sequence the route now performs.
    @Test func headlessProtocolSaveKeepsVariantConditions() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "headless") { _ in
            try variantDraft(named: "kept")
            let panel = makePanel(selecting: "kept")
            // The stale field: whatever the panel last synced (a previous
            // selection, a workspace default) — NOT this study's model.
            panel.studyBaseModelID = "some/other-model"
            panel.protocolDescription = "described headlessly"

            // What POST /api/experiment/protocol does before delegating.
            panel.adoptSelectedManifestBaseModel()
            panel.saveProtocol()

            let saved = try ExperimentStore.load(name: "kept")
            // Canary that the save actually ran — a swallowed error would
            // leave the planted arms intact for the wrong reason.
            #expect(saved.experimentDescription == "described headlessly")
            #expect(saved.modelID == "test/model")
            #expect(saved.variantConditions.count == 1)
        }
    }

    /// An EMPTY model field is "no choice made", never a change to the
    /// empty string — the floor under any caller that skips the resync.
    @Test func emptyModelFieldIsNoChangeRequested() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "headless") { _ in
            try variantDraft(named: "unset")
            let panel = makePanel(selecting: "unset")
            panel.studyBaseModelID = "   "
            panel.saveProtocol()

            let saved = try ExperimentStore.load(name: "unset")
            #expect(saved.modelID == "test/model")
            #expect(saved.variantConditions.count == 1)
        }
    }

    // MARK: - The intentional reset still works

    /// The GUI path is unchanged: a researcher actually picking a new base
    /// model still invalidates the cast agents — an adapter built on one
    /// model is not eligible for an arm running another.
    @Test func aRealModelChangeStillClearsVariantConditions() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "headless") { _ in
            var draft = try variantDraft(named: "rebased")
            draft.modelRevision = "cafe01"
            try ExperimentStore.save(draft)

            let panel = makePanel(selecting: "rebased")
            panel.studyBaseModelID = "new/model"
            panel.saveProtocol()

            let saved = try ExperimentStore.load(name: "rebased")
            #expect(saved.modelID == "new/model")
            #expect(saved.modelRevision == nil)
            #expect(saved.variantConditions.isEmpty)
        }
    }
}

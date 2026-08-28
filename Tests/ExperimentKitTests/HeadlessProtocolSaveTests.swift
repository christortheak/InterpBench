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

/// The other silent loss on this route: a default `JSONDecoder` ignores keys
/// the Body does not declare, so an out-of-vocabulary key wrote nothing while
/// the route answered ok. `unknownProtocolBodyKeys` is the refusal's decision,
/// factored out of the connection handling like `isRequestRefused` — the
/// Python engine's twin gate is `experiment_store.set_protocol` over
/// `PROTOCOL_FIELDS` (the manifest's key spellings; this route speaks the
/// panel's).
@Suite struct ProtocolBodyVocabularyTests {

    @Test func unknownKeysAreNamedAndSorted() {
        let body = Data(
            #"{"temperature":0.7,"notAField":1,"alsoNot":2}"#.utf8)
        #expect(SteerLabWebServer.unknownProtocolBodyKeys(in: body)
            == ["alsoNot", "notAField"])
    }

    @Test func everyDeclaredKeyPasses() {
        let body = Data(#"""
            {"description":"d","task":"t","outcomes":"o","judgeModel":"j",
             "judgePrompt":"p","taskPromptsFile":"f","promptMode":"rawCompletion",
             "systemPrompt":"s","qwenThinkingEnabled":false,
             "temperature":0.1,"maxTokens":16,"samplesPerItem":25,
             "seedPolicy":"derivedSHA256",
             "exclusionRules":[{"rule":"unparseableEndpoint"}]}
            """#.utf8)
        #expect(SteerLabWebServer.unknownProtocolBodyKeys(in: body).isEmpty)
    }

    /// A body that is not a JSON object is the typed decode's refusal
    /// ("bad body"), not this one's — the helper stays out of its way.
    @Test func nonObjectBodiesAreLeftToTheTypedDecode() {
        #expect(SteerLabWebServer.unknownProtocolBodyKeys(in: Data("[1]".utf8)).isEmpty)
        #expect(SteerLabWebServer.unknownProtocolBodyKeys(in: Data("nope".utf8)).isEmpty)
    }

    /// Review round 10, finding 3: the route gated `samplesPerItem` and
    /// `seedPolicy` and assigned `temperature`/`maxTokens` unchecked, so a
    /// negative temperature or a zero maxTokens landed in the panel fields and
    /// the store's note went nowhere. All four are gated, in the STORE
    /// setter's own sentences.
    @Test func temperatureAndMaxTokensAreGatedLikeTheirNeighbours() {
        #expect(
            SteerLabWebServer.protocolBodyValueProblem(temperature: -0.5)
                == "temperature must be a non-negative number — got -0.5")
        #expect(
            SteerLabWebServer.protocolBodyValueProblem(
                temperature: Double.nan) != nil)
        #expect(
            SteerLabWebServer.protocolBodyValueProblem(
                temperature: Double.infinity) != nil)
        #expect(
            SteerLabWebServer.protocolBodyValueProblem(maxTokens: 0)
                == "maxTokens must be a positive integer — got 0")
        #expect(
            SteerLabWebServer.protocolBodyValueProblem(maxTokens: -8)
                == "maxTokens must be a positive integer — got -8")
        // The neighbours still say what they always said.
        #expect(
            SteerLabWebServer.protocolBodyValueProblem(samplesPerItem: 0)
                == "samplesPerItem must be ≥ 1 — got 0")
        #expect(
            SteerLabWebServer.protocolBodyValueProblem(seedPolicy: "diceRoll")?
                .hasPrefix("unknown seedPolicy 'diceRoll' — known: ") == true)

        // Valid values — including the boundaries — still write.
        #expect(
            SteerLabWebServer.protocolBodyValueProblem(
                temperature: 0, maxTokens: 1, samplesPerItem: 1,
                seedPolicy: "derivedSHA256") == nil)
        #expect(
            SteerLabWebServer.protocolBodyValueProblem(
                temperature: 0.7, maxTokens: 512, samplesPerItem: 25) == nil)
        // An absent field is not a bad one: nothing declared, nothing refused.
        #expect(SteerLabWebServer.protocolBodyValueProblem() == nil)
        #expect(SteerLabWebServer.protocolBodyValueProblem(seedPolicy: "") == nil)
    }

}

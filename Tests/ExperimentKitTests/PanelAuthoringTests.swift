import Foundation
import Testing

@testable import ExperimentKit

/// The authoring side of the semantic-panel split: a scenario file is the
/// ENVIRONMENT, and the editor must be unable to write anything else into it.
///
/// The properties worth testing are the ones that fail SILENTLY if they break.
/// A save that leaks a binding produces a file that still loads, still runs,
/// and quietly re-introduces the confound the split exists to remove. A
/// migration that rewrites the source file in place produces a hash-drift
/// refusal weeks later, at run start, on a compute node.
///
/// Serialized because these move the process-global workspace root.
@Suite(.serialized) struct PanelAuthoringTests {

    // MARK: - Fixtures

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "panel-authoring-\(UUID().uuidString)")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try body(temp)
    }

    private func semanticPanel() -> MultiAgentScenario {
        PanelComposition.semanticForm(
            MultiAgentScenario(
                name: "allocation-panel",
                baseModelID: "",
                sharedMaterials: "Two teams share 100 credits.",
                agents: [
                    .init(
                        id: "seat-proposer", name: "Proposer", baseModelID: "",
                        systemPrompt: "You represent Team North."),
                    .init(
                        id: "seat-reviewer", name: "Reviewer", baseModelID: "",
                        systemPrompt: "You represent Team South."),
                ],
                turns: [
                    .init(
                        id: "turn-propose", title: "Proposal",
                        speakerAgentID: "seat-proposer",
                        promptTemplate: "Propose a split.",
                        outputLabel: "proposal"),
                    .init(
                        id: "turn-review", title: "Review",
                        speakerAgentID: "seat-reviewer",
                        promptTemplate: "Accept or decline.",
                        outputLabel: "review"),
                ]))
    }

    private func boundPanel(
        agentPath: String? = nil, agentHash: String? = nil
    ) -> MultiAgentScenario {
        var bound = semanticPanel()
        bound.baseModelID = "test/model"
        bound.temperature = 0.7
        bound.maxTokens = 300
        bound.agents[0].baseModelID = "test/model"
        bound.agents[0].variantArtifactPath = agentPath
        bound.agents[0].variantArtifactHash = agentHash
        bound.agents[1].baseModelID = "test/model"
        return bound
    }

    // MARK: - Mode detection

    @Test func aFileThatDiffersFromItsSemanticFormIsFlaggedAsBound() throws {
        #expect(!PanelAuthoring.carriesBindings(semanticPanel()))
        #expect(PanelAuthoring.mode(for: semanticPanel()) == .semantic)
        #expect(PanelAuthoring.carriesBindings(boundPanel()))
        #expect(PanelAuthoring.mode(for: boundPanel()) == .legacyBound)
    }

    @Test func aFreshEmptyDraftIsNotMistakenForALegacyFile() throws {
        // A brand-new scenario has no seats at all. `isSemantic` says false
        // for it (nothing to be semantic ABOUT), so keying the banner off that
        // would put every new draft behind a migration wall.
        let blank = MultiAgentScenario(name: "new-panel", baseModelID: "")
        #expect(!PanelComposition.isSemantic(blank))
        #expect(!PanelAuthoring.carriesBindings(blank))
    }

    @Test func aRootModelWithNoSeatModelsStillCountsAsBound() throws {
        // The narrow seat-level test would call this semantic, and the next
        // save would silently drop the model it names.
        var rootOnly = semanticPanel()
        rootOnly.baseModelID = "test/model"
        #expect(PanelComposition.isSemantic(rootOnly))
        #expect(PanelAuthoring.carriesBindings(rootOnly))
    }

    // MARK: - Save normalisation

    @Test func aSaveWritesTheSemanticFormEvenWhenEditorStateCarriesBindings() throws {
        // Editor state that a future code path (or a legacy load) could hand
        // in. The save must not depend on the UI having withheld it.
        let dirty = boundPanel(agentPath: "runs/model-variants/x/model-variant.json")
        let saved = PanelAuthoring.scenarioForSave(
            name: "  allocation-panel  ",
            description: dirty.description,
            sharedMaterials: dirty.sharedMaterials,
            agents: dirty.agents,
            turns: dirty.turns)

        #expect(saved.name == "allocation-panel")
        #expect(!PanelAuthoring.carriesBindings(saved))
        #expect(saved.baseModelID.isEmpty)
        #expect(saved.agents.allSatisfy { $0.baseModelID.isEmpty })
        #expect(saved.agents.allSatisfy { $0.variantArtifactPath == nil })
        #expect(saved.agents.allSatisfy { $0.variantArtifactHash == nil })
        // Roles and structure survive — they are the environment.
        #expect(saved.agents.map(\.systemPrompt) == dirty.agents.map(\.systemPrompt))
        #expect(saved.turns == dirty.turns)
        #expect(saved.sharedMaterials == dirty.sharedMaterials)
    }

    // MARK: - Rehearsal

    @Test func aRehearsalCompilesAllSeatsBaselineAtTheGivenSettings() throws {
        let rehearsal = try PanelAuthoring.rehearsalScenario(
            semanticPanel(), modelID: "test/model", temperature: 0.3, maxTokens: 700)

        // Runnable by the engine's own validator — a rehearsal is a real run.
        try MultiAgentRunner.validate(rehearsal)
        #expect(rehearsal.baseModelID == "test/model")
        #expect(rehearsal.temperature == 0.3)
        #expect(rehearsal.maxTokens == 700)
        // No agent is seated: casting is a study decision, and a rehearsal
        // that quietly seated one would read as a measurement.
        #expect(rehearsal.agents.allSatisfy { $0.variantArtifactPath == nil })
        // Stripping it back gives the environment it came from.
        #expect(PanelComposition.semanticForm(rehearsal) == semanticPanel())
    }

    @Test func aRehearsalRefusesWithNoModel() throws {
        #expect(throws: (any Error).self) {
            try PanelAuthoring.rehearsalScenario(
                semanticPanel(), modelID: "  ", temperature: 0, maxTokens: 512)
        }
    }

    // MARK: - Migration

    @Test func migrationWritesANewFileAndLeavesTheSourceBytesUntouched() throws {
        try withTempWorkspace { _ in
            let agentPath = "runs/model-variants/sympathy-agent/model-variant.json"
            let record = try MultiAgentScenarioStore.save(
                boundPanel(agentPath: agentPath, agentHash: "abc123"))
            let sourceBytes = try Data(contentsOf: record.url)

            let migration = try PanelAuthoring.migrateLegacyPanel(at: record.url)

            // The pin holds: a study naming this path by hash still resolves
            // to the same bytes, so it keeps loading and running as before.
            #expect(try Data(contentsOf: record.url) == sourceBytes)
            let newURL = ExperimentStore.resolveProjectPath(migration.semanticPath)
            #expect(newURL != record.url)
            #expect(FileManager.default.fileExists(atPath: newURL.path))

            let written = try JSONDecoder().decode(
                MultiAgentScenario.self, from: try Data(contentsOf: newURL))
            #expect(!PanelAuthoring.carriesBindings(written))
            #expect(written.name == "allocation-panel")
            #expect(written.turns == semanticPanel().turns)

            // What was extracted, reported rather than discarded.
            #expect(migration.modelID == "test/model")
            #expect(migration.temperature == 0.7)
            #expect(migration.maxTokens == 300)
            #expect(migration.seatBindings.map(\.seatID)
                == ["seat-proposer", "seat-reviewer"])
            #expect(migration.seatBindings[0].agentName == "Proposer")
            #expect(migration.seatBindings[0].artifactPath == agentPath)
            #expect(migration.seatBindings[1].agentName == nil)
            #expect(migration.seatBindings[1].summary.contains("baseline"))
        }
    }

    @Test func migrationSurfacesDivergentSeatModelsVerbatim() throws {
        try withTempWorkspace { _ in
            var bound = semanticPanel()
            bound.baseModelID = ""
            bound.agents[0].baseModelID = "test/model"
            bound.agents[1].baseModelID = "other/model"
            let record = try MultiAgentScenarioStore.save(bound)

            let migration = try PanelAuthoring.migrateLegacyPanel(at: record.url)

            let warning = try #require(migration.warnings.first)
            #expect(warning.contains("DIFFERENT models"))
            // Never unified in silence.
            #expect(["test/model", "other/model"].contains(migration.modelID))
        }
    }

    @Test func migrationRefusesAPanelThatCarriesNoBindings() throws {
        try withTempWorkspace { _ in
            let record = try MultiAgentScenarioStore.save(semanticPanel())
            #expect(throws: (any Error).self) {
                try PanelAuthoring.migrateLegacyPanel(at: record.url)
            }
        }
    }

    // MARK: - The editor's read-only latch

    @MainActor
    @Test func openingABoundPanelLocksTheEditorAndRefusesToSaveOverIt() throws {
        try withTempWorkspace { _ in
            let bound = try MultiAgentScenarioStore.save(boundPanel())
            let boundBytes = try Data(contentsOf: bound.url)
            let panel = MultiAgentPanel()
            panel.refresh()

            // Through the library, as the picker does — a record's id is its
            // scanned (symlink-resolved) path, which the writer's URL need not
            // match under a `/var/folders` temp root.
            panel.selectedScenarioID = try #require(panel.scenarios.first).id
            #expect(panel.isLegacyBound)
            #expect(panel.authoringMode == .legacyBound)
            // Loaded STRIPPED: the editor never holds a binding, so it cannot
            // write one back even if the read-only latch were bypassed.
            #expect(panel.agents.allSatisfy { $0.baseModelID.isEmpty })
            #expect(panel.agents.map(\.systemPrompt)
                == semanticPanel().agents.map(\.systemPrompt))

            panel.saveScenario()
            #expect(try Data(contentsOf: bound.url) == boundBytes)
            #expect(panel.status?.contains("migrate it first") == true)

            // Migrating clears the latch and lands on the new environment.
            panel.migrateSelectedScenario()
            #expect(!panel.isLegacyBound)
            #expect(panel.migration != nil)
            #expect(try Data(contentsOf: bound.url) == boundBytes)
            let landed = try #require(panel.selectedScenario)
            #expect(landed.url.lastPathComponent != bound.url.lastPathComponent)
            #expect(!PanelAuthoring.carriesBindings(landed.scenario))
        }
    }

    @MainActor
    @Test func editingASemanticPanelSavesTheEnvironmentAndNothingElse() throws {
        try withTempWorkspace { _ in
            let seed = try MultiAgentScenarioStore.save(semanticPanel())
            let panel = MultiAgentPanel()
            panel.refresh()
            panel.selectedScenarioID = try #require(panel.scenarios.first).id

            #expect(!panel.isLegacyBound)
            #expect(!panel.hasUnsavedChanges)

            panel.sharedMaterials = "Two teams share 200 credits."
            #expect(panel.hasUnsavedChanges)
            // Bindings smuggled into editor state by any route are dropped on
            // the way to disk.
            panel.agents[0].baseModelID = "test/model"
            panel.agents[0].variantArtifactPath = "runs/model-variants/x.json"
            panel.saveScenario()

            let written = try JSONDecoder().decode(
                MultiAgentScenario.self, from: try Data(contentsOf: seed.url))
            #expect(!PanelAuthoring.carriesBindings(written))
            #expect(written.sharedMaterials == "Two teams share 200 credits.")
            #expect(panel.status?.contains("saved scenario") == true)
        }
    }

    @MainActor
    @Test func aNewDraftIsAnEnvironmentWithNoModelAnywhere() throws {
        try withTempWorkspace { _ in
            let panel = MultiAgentPanel()
            panel.newScenario()

            #expect(!panel.isLegacyBound)
            #expect(!panel.agents.isEmpty)
            #expect(panel.agents.allSatisfy { $0.baseModelID.isEmpty })

            // Also the regression on the selection itself: a just-saved panel
            // must come back SELECTED. It did not when the workspace root sat
            // behind a symlink, because the writer's URL and the scanned
            // record's id disagreed on `/var` vs `/private/var`.
            panel.saveScenario()
            let saved = try #require(panel.selectedScenario)
            #expect(!PanelAuthoring.carriesBindings(saved.scenario))
            #expect(!panel.hasUnsavedChanges)
        }
    }

    @MainActor
    @Test func seatsReorderAndTheirRemovalRepairsTheTurnScript() throws {
        try withTempWorkspace { _ in
            let panel = MultiAgentPanel()
            panel.newScenario()
            let original = panel.agents.map(\.id)

            panel.moveSeat(id: original[2], by: -1)
            #expect(panel.agents.map(\.id) == [original[0], original[2], original[1]])
            // Off the ends is a no-op, not a crash or a wrap-around.
            panel.moveSeat(id: original[0], by: -1)
            #expect(panel.agents.map(\.id) == [original[0], original[2], original[1]])

            panel.removeSeat(id: original[0])
            #expect(!panel.agents.contains { $0.id == original[0] })
            let seatIDs = Set(panel.agents.map(\.id))
            #expect(panel.turns.allSatisfy { seatIDs.contains($0.speakerAgentID) })
            #expect(panel.turns.allSatisfy {
                $0.routedAgentIDs.allSatisfy(seatIDs.contains)
            })
        }
    }
}

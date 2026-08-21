import Foundation
import Testing

@testable import ExperimentKit

/// Casting a study's seats from the study editor.
///
/// The property that matters is that a directly-authored panel study can become
/// RUNNABLE. A scenario chosen in Study Setup is semantic — it binds no model to
/// any seat — so before this existed the only way to fill seats was a design's
/// instantiation table, and a study that picked a scenario refused at run start
/// (`MultiAgentRunner.validate`, deliberately). So the tests here follow the
/// whole loop: cast → compile → pin → reopen → recover the same casting, plus
/// the two things that could silently corrupt it — a settings change leaving a
/// stale compiled scenario behind, and a base-model change carrying agents into
/// a panel they cannot run in.
///
/// Serialized because the suite moves the process-global workspace root: the
/// scenario library, the agent library and experiments/ all have to land in the
/// same temporary tree.
@Suite(.serialized) struct SeatCastingTests {

    // MARK: - Fixtures

    func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        // Symlinks RESOLVED: the temp directory is /var → /private/var, and
        // this suite compares scanned record ids (absolute paths) against ids
        // built from the store's own roots. An unresolved root makes the two
        // spellings differ for reasons that have nothing to do with the code
        // under test.
        let temp = FileManager.default.temporaryDirectory
            .resolvingSymlinksInPath()
            .appending(component: "seats-\(UUID().uuidString)")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try body(temp)
    }

    @discardableResult
    private func plantAgent(
        named name: String, baseModel: String = "test/model"
    ) throws -> ModelVariantRecord {
        let directory = ModelVariantStore.directory.appending(component: name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let artifact = ModelVariantArtifact(
            name: name, baseModelID: baseModel,
            injections: [
                .init(
                    concept: "sympathy", vectorArtifactID: "runs/vec-\(name)",
                    layer: 20, alpha: 0.08)
            ],
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "")
        let url = directory.appending(component: "model-variant.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url)
        return ModelVariantRecord(url: url, artifact: artifact)
    }

    /// A two-seat SEMANTIC scenario: roles, turns, materials, no bindings.
    private func semanticScenario() -> MultiAgentScenario {
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

    /// A multi-agent draft that pins a SEMANTIC scenario — exactly what a
    /// researcher gets by picking one in Study Setup and saving.
    @discardableResult
    private func makeUncastStudy(
        named name: String = "allocation-study",
        scenario: MultiAgentScenario? = nil
    ) throws -> (manifest: ExperimentManifest, scenario: MultiAgentScenarioRecord) {
        var manifest = try ExperimentStore.create(
            name: name, description: "Allocation panel",
            modelID: "test/model", modelRevision: "abc123")
        let record = try MultiAgentScenarioStore.save(scenario ?? semanticScenario())
        manifest.studyKind = .multiAgent
        manifest.studyType = StudyIntent.multiAgent.rawValue
        manifest.multiAgentScenarioPath = FineTuneStore.relativePath(for: record.url)
        manifest.multiAgentScenarioHash = try MultiAgentScenarioStore.hash(record.url)
        manifest.maxTokens = 512
        try ExperimentStore.save(manifest)
        return (manifest, record)
    }

    @MainActor
    private func makePanel(root: URL, selecting study: String) -> ExperimentPanel {
        let panel = ExperimentPanel()
        panel.notices = PanelNotices(
            fileURL: root.appending(component: "notices.jsonl"))
        panel.refresh()
        panel.selectedName = study
        return panel
    }

    /// The scenario library's OWN id for a scenario, and the agent library's
    /// own record for an agent.
    ///
    /// Scanned rather than remembered from the plant: a directory listing and a
    /// URL built from the workspace root can spell the same file two ways under
    /// a temporary root (/var vs /private/var), and every store-facing surface
    /// speaks the listing's spelling. Comparing against the planted URL would
    /// test the temp directory, not the code.
    private func scenarioID(named name: String) throws -> String {
        try #require(MultiAgentScenarioStore.scan().first { $0.scenario.name == name }).id
    }

    private func agentRecord(named name: String) throws -> ModelVariantRecord {
        try #require(ModelVariantStore.scan().first { $0.artifact.name == name })
    }

    /// The workspace-relative artifact path a seat pins for a library agent.
    private func agentArtifactPath(_ name: String) throws -> String {
        ModelVariantStore.relativePath(for: try agentRecord(named: name))
    }

    private func decodeScenario(_ path: String) throws -> MultiAgentScenario {
        try JSONDecoder().decode(
            MultiAgentScenario.self,
            from: Data(contentsOf: ExperimentStore.resolveProjectPath(path)))
    }

    // MARK: - Reading what a study is carrying

    @Test func aStudyPinningASemanticScenarioReadsAsUncast() throws {
        try withTempWorkspace { _ in
            let (manifest, _) = try makeUncastStudy()
            let state = try #require(SeatCasting.state(of: manifest))
            #expect(state.form == .uncast)
            #expect(state.isEditable)
            #expect(state.seats.map(\.id) == ["seat-proposer", "seat-reviewer"])
            #expect(state.seats.map(\.name) == ["Proposer", "Reviewer"])
            #expect(state.occupants.values.allSatisfy { $0 == .baseline })
            // The state of the world it describes: not runnable yet.
            #expect(throws: (any Error).self) {
                try MultiAgentRunner.validate(state.semantic)
            }
        }
    }

    /// A scenario that carries its own bindings is READ-ONLY here: its casting
    /// lives in the file, other studies may pin the same file, and re-casting it
    /// from a study editor would rewrite an input under them.
    @Test func aLegacyBoundScenarioIsShownReadOnlyWithTheMigrationPointer() throws
    {
        try withTempWorkspace { _ in
            let agent = try plantAgent(named: "sympathy-agent")
            var bound = semanticScenario()
            bound.baseModelID = "test/model"
            bound.agents[0].baseModelID = "test/model"
            bound.agents[0].variantArtifactPath =
                ModelVariantStore.relativePath(for: agent)
            bound.agents[0].variantArtifactHash = try ModelVariantStore.hash(agent.url)
            bound.agents[1].baseModelID = "test/model"
            let record = try MultiAgentScenarioStore.update(
                bound,
                at: MultiAgentScenarioStore.directory.appending(
                    component: "legacy-bound.json"))

            var manifest = try ExperimentStore.create(
                name: "legacy-study", description: "d", modelID: "test/model")
            manifest.studyKind = .multiAgent
            manifest.multiAgentScenarioPath = FineTuneStore.relativePath(for: record.url)
            manifest.multiAgentScenarioHash = try MultiAgentScenarioStore.hash(record.url)
            try ExperimentStore.save(manifest)

            let state = try #require(SeatCasting.state(of: manifest))
            #expect(state.form == .legacyBound)
            #expect(!state.isEditable)
            // The casting IS shown — read-only means visible, not hidden.
            #expect(state.occupants["seat-proposer"]?.label == "Proposer")
            #expect(state.occupants["seat-reviewer"] == .baseline)
            let advisory = try #require(state.advisories.first)
            #expect(advisory.contains("read-only"))
            #expect(advisory.contains("Panels editor"))
        }
    }

    // MARK: - The round trip

    @Test @MainActor func castingCompilesPinsAndReopensWithTheSameCast() throws {
        try withTempWorkspace { root in
            try makeUncastStudy()
            try plantAgent(named: "sympathy-agent")
            let panel = makePanel(root: root, selecting: "allocation-study")
            let agentID = try agentRecord(named: "sympathy-agent").id

            // The picker points at the semantic scenario the study pins.
            #expect(panel.selectedMultiAgentScenarioID
                == (try scenarioID(named: "allocation-panel")))
            #expect(panel.availableAgentsForSeats.map(\.artifact.name)
                == ["sympathy-agent"])

            panel.setSeatAgent(agentID, seat: "seat-proposer")
            #expect(panel.seatAgentID(for: "seat-proposer") == agentID)
            #expect(panel.seatAgentID(for: "seat-reviewer") == nil)
            panel.saveSeatCasting()

            let saved = try ExperimentStore.load(name: "allocation-study")
            let compiledPath = try #require(saved.multiAgentScenarioPath)
            #expect(compiledPath.hasPrefix("prompts/panels/compiled/"))
            // Pinned bytes, and a scenario the run path accepts.
            let data = try Data(
                contentsOf: ExperimentStore.resolveProjectPath(compiledPath))
            #expect(MultiAgentScenarioStore.hash(data) == saved.multiAgentScenarioHash)
            let compiled = try JSONDecoder().decode(
                MultiAgentScenario.self, from: data)
            try MultiAgentRunner.validate(compiled)
            #expect(compiled.baseModelID == "test/model")
            #expect(compiled.maxTokens == 512)
            #expect(compiled.agents[0].variantArtifactPath
                == (try agentArtifactPath("sympathy-agent")))
            #expect(compiled.agents[1].variantArtifactPath == nil)
            // The source scenario is recorded and untouched.
            let semanticPath = try #require(saved.multiAgentSemanticScenarioPath)
            #expect(semanticPath.hasSuffix("prompts/panels/allocation-panel.json"))
            #expect(saved.multiAgentSemanticScenarioHash
                == (try MultiAgentScenarioStore.hash(
                    ExperimentStore.resolveProjectPath(semanticPath))))
            #expect(PanelComposition.semanticForm(compiled) == semanticScenario())
            // And the pin verifies.
            #expect(ExperimentStore.verify(saved).isEmpty)

            // Reopening recovers the casting from the pinned scenario, and the
            // picker names the SEMANTIC one (a compiled file is deliberately
            // outside the library).
            let reopened = makePanel(root: root, selecting: "allocation-study")
            let state = try #require(reopened.seatCasting)
            #expect(state.form == .cast)
            #expect(state.isEditable)
            #expect(state.seats.map(\.id) == ["seat-proposer", "seat-reviewer"])
            #expect(state.occupants["seat-reviewer"] == .baseline)
            // The seat's occupant NAME comes from the role it fills; the agent
            // is identified by its pinned artifact, exactly as a variant
            // condition is.
            if case .agent(let name, let path, let hash) =
                try #require(state.occupants["seat-proposer"])
            {
                #expect(name == "Proposer")
                #expect(path == (try agentArtifactPath("sympathy-agent")))
                #expect(hash.count == 64)
            } else {
                Issue.record("seat-proposer came back baseline")
            }
            #expect(reopened.selectedMultiAgentScenarioID
                == (try scenarioID(named: "allocation-panel")))
            #expect(reopened.seatAgentID(for: "seat-proposer") == agentID)
        }
    }

    /// The compile inputs are manifest fields, so a settings save RECOMPILES.
    /// The alternative is a study whose manifest says one thing and whose
    /// scenario file — the thing the run actually reads — says another.
    @Test @MainActor func savingTheSetupRecompilesTheCastingAtTheNewSettings()
        throws
    {
        try withTempWorkspace { root in
            try makeUncastStudy()
            try plantAgent(named: "sympathy-agent")
            let panel = makePanel(root: root, selecting: "allocation-study")
            panel.setSeatAgent(
                try agentRecord(named: "sympathy-agent").id, seat: "seat-proposer")
            panel.saveSeatCasting()
            let firstPath = try #require(
                ExperimentStore.load(name: "allocation-study")
                    .multiAgentScenarioPath)

            panel.runMaxTokens = 999
            panel.runTemperature = 0
            panel.saveProtocol()

            let saved = try ExperimentStore.load(name: "allocation-study")
            #expect(saved.maxTokens == 999)
            let recompiledPath = try #require(saved.multiAgentScenarioPath)
            #expect(recompiledPath != firstPath)
            let recompiled = try decodeScenario(recompiledPath)
            #expect(recompiled.maxTokens == 999)
            // The cast survived the recompile — only the settings moved.
            #expect(recompiled.agents[0].variantArtifactPath
                == (try agentArtifactPath("sympathy-agent")))
            #expect(recompiled.agents[1].variantArtifactPath == nil)
            #expect(ExperimentStore.verify(saved).isEmpty)
        }
    }

    /// The one case where a recompile must NOT carry the cast: the base model
    /// changed, so no agent built on the previous model is eligible for any
    /// seat. Same rule the comparison arms already follow.
    @Test @MainActor func changingTheBaseModelResetsEverySeatToBaseline() throws {
        try withTempWorkspace { root in
            try makeUncastStudy()
            try plantAgent(named: "sympathy-agent")
            let panel = makePanel(root: root, selecting: "allocation-study")
            panel.setSeatAgent(
                try agentRecord(named: "sympathy-agent").id, seat: "seat-proposer")
            panel.saveSeatCasting()

            panel.studyBaseModelID = "other/model"
            panel.saveProtocol()

            let saved = try ExperimentStore.load(name: "allocation-study")
            #expect(saved.modelID == "other/model")
            let compiled = try decodeScenario(
                try #require(saved.multiAgentScenarioPath))
            #expect(compiled.baseModelID == "other/model")
            #expect(compiled.agents.allSatisfy { $0.variantArtifactPath == nil })
            #expect(compiled.agents.allSatisfy { $0.baseModelID == "other/model" })
            // Said out loud, not silently — a cast that disappears without a
            // word is the failure mode this guards.
            #expect(
                panel.notices.notices.contains {
                    $0.severity == .warning && $0.message.contains("reset")
                })
        }
    }

    /// Picking a scenario and saving the SETUP (rather than the casting) is the
    /// other order a researcher can do this in, and it must compile too —
    /// otherwise the study pins a semantic scenario and refuses at run start.
    @Test @MainActor func savingTheSetupCompilesAnUncastSelection() throws {
        try withTempWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "fresh-panel", description: "d", modelID: "test/model")
            manifest.studyKind = .multiAgent
            try ExperimentStore.save(manifest)
            try MultiAgentScenarioStore.save(semanticScenario())

            let panel = makePanel(root: root, selecting: "fresh-panel")
            panel.selectedMultiAgentScenarioID = try scenarioID(
                named: "allocation-panel")
            panel.saveProtocol()

            let saved = try ExperimentStore.load(name: "fresh-panel")
            let path = try #require(saved.multiAgentScenarioPath)
            #expect(path.hasPrefix("prompts/panels/compiled/"))
            try MultiAgentRunner.validate(try decodeScenario(path))
            #expect(saved.multiAgentSemanticScenarioPath?
                .hasSuffix("prompts/panels/allocation-panel.json") == true)
        }
    }

    /// A legacy bound scenario is pinned AS IT STANDS by a setup save: its
    /// casting is inside the file, and the save must not rewrite it.
    @Test @MainActor func savingTheSetupPinsALegacyScenarioUnchanged() throws {
        try withTempWorkspace { root in
            var bound = semanticScenario()
            bound.baseModelID = "test/model"
            bound.agents[0].baseModelID = "test/model"
            bound.agents[1].baseModelID = "test/model"
            let record = try MultiAgentScenarioStore.update(
                bound,
                at: MultiAgentScenarioStore.directory.appending(
                    component: "legacy-bound.json"))

            var manifest = try ExperimentStore.create(
                name: "legacy-setup", description: "d", modelID: "test/model")
            manifest.studyKind = .multiAgent
            try ExperimentStore.save(manifest)

            let panel = makePanel(root: root, selecting: "legacy-setup")
            panel.selectedMultiAgentScenarioID = try #require(
                MultiAgentScenarioStore.scan().first {
                    $0.url.lastPathComponent == record.url.lastPathComponent
                }).id
            panel.saveProtocol()

            let saved = try ExperimentStore.load(name: "legacy-setup")
            #expect(saved.multiAgentScenarioPath?
                .hasSuffix("prompts/panels/legacy-bound.json") == true)
            #expect(saved.multiAgentSemanticScenarioPath == nil)
            #expect(ExperimentStore.verify(saved).isEmpty)
        }
    }

    // MARK: - Drift in the source scenario

    /// An edited source scenario is an ADVISORY, never a violation: the study
    /// runs the compiled bytes, which is exactly what it ran last time.
    @Test func editingTheSourceScenarioAdvisesButDoesNotInvalidate() throws {
        try withTempWorkspace { _ in
            let (base, record) = try makeUncastStudy()
            var manifest = base
            try SeatCasting.compile(
                SeatAssignment(
                    seatIDs: ["seat-proposer", "seat-reviewer"],
                    ordered: [.baseline, .baseline]),
                semantic: semanticScenario(),
                semanticPath: FineTuneStore.relativePath(for: record.url),
                into: &manifest)
            try ExperimentStore.save(manifest)

            var edited = semanticScenario()
            edited.sharedMaterials = "Three teams share 100 credits."
            try MultiAgentScenarioStore.update(edited, at: record.url)

            let state = try #require(SeatCasting.state(of: manifest))
            #expect(state.form == .cast)
            let advisory = try #require(
                state.advisories.first { $0.contains("has changed") })
            #expect(advisory.contains("COMPILED"))
            // The pinned scenario — the one that runs — is untouched.
            #expect(ExperimentStore.verify(manifest).isEmpty)
        }
    }

    // MARK: - Permuted siblings

    /// One study runs ONE casting, so re-seating is sibling studies. The
    /// invitation carries every DISTINCT re-seating of the study's own cast.
    @Test @MainActor func permutedSiblingsCarryEveryDistinctReSeating() throws {
        try withTempWorkspace { root in
            try makeUncastStudy()
            try plantAgent(named: "sympathy-agent")
            let panel = makePanel(root: root, selecting: "allocation-study")
            panel.setSeatAgent(
                try agentRecord(named: "sympathy-agent").id, seat: "seat-proposer")
            panel.saveSeatCasting()

            panel.startPermutedSiblings()
            #expect(panel.formErrors[.template] == nil)
            let invitation = try #require(panel.templateInstantiationInvitation)
            #expect(invitation.permuting.count == 2)
            #expect(invitation.permuting.contains(.baseline))
            let agentPath = try agentArtifactPath("sympathy-agent")
            #expect(invitation.permuting.contains {
                if case .agent(_, let path, _) = $0 { return path == agentPath }
                return false
            })

            // The table the invitation opens holds one row per distinct
            // casting — [agent, baseline] over two seats is two, not four.
            let table = TemplateInstantiation(templateName: invitation.design)
            table.preloadPermutations(occupants: invitation.permuting)
            #expect(table.rows.count == 2)
            #expect(table.advisories.isEmpty)
            #expect(Set(table.rows.map { row in
                table.seatIDs.map { row.seating[$0]?.label ?? "?" }.joined(separator: "-")
            }).count == 2)
            #expect(table.rows.allSatisfy { table.refusal(for: $0) == nil })

            // And minting them produces sibling studies, each pinning its own
            // compiled scenario.
            let minted = StudyTemplateStore.mintBatch(
                templateName: invitation.design,
                cells: table.rows.compactMap { table.cell(for: $0) })
            #expect(minted.failures.isEmpty)
            #expect(minted.minted.count == 2)
            let paths = try minted.minted.map {
                try #require(
                    ExperimentStore.load(name: $0).multiAgentScenarioPath)
            }
            #expect(Set(paths).count == 2)
            for path in paths {
                try MultiAgentRunner.validate(try decodeScenario(path))
            }
        }
    }

    /// An uncast study has no cast to permute, and says so rather than opening
    /// a table of all-baseline rows.
    @Test @MainActor func permutingRefusesBeforeTheSeatsAreSaved() throws {
        try withTempWorkspace { root in
            try makeUncastStudy()
            let panel = makePanel(root: root, selecting: "allocation-study")
            panel.startPermutedSiblings()
            #expect(panel.templateInstantiationInvitation == nil)
            let refusal = try #require(panel.formErrors[.template])
            #expect(refusal.contains("Save the seats first"))
        }
    }

    // MARK: - The manifest keys

    /// Omit-when-nil, so every manifest written before seat casting existed
    /// hashes exactly as it did.
    @Test func aManifestWithoutASemanticSourceKeepsItsContentHash() throws {
        let manifest = ExperimentManifest(
            name: "legacy", description: "d", modelID: "test/model")
        let encoded = try JSONEncoder().encode(manifest)
        let object = try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        #expect(object?["multiAgentSemanticScenarioPath"] == nil)
        #expect(object?["multiAgentSemanticScenarioHash"] == nil)
    }
}

import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Study templates: the same replication, run again with a different cast.
///
/// The properties under test are the ones that make a template safe to reuse
/// rather than merely convenient. An instantiated study must be an ORDINARY
/// draft — same arms the panel's own "Add agent" writes, same pins, nothing
/// the run/freeze/epoch machinery has to learn about. Derived pins must be
/// RE-DERIVED at mint time, not copied: a stale `outcomeInstrumentScope`
/// killed four Slurm shards after the model had already loaded. And "load this
/// study as a template" must recognise an unchanged instance, or the template
/// picker fills with indistinguishable near-duplicates within a week.
///
/// Serialized because the whole suite moves the process-global workspace root:
/// templates, experiments, the agent library and the panel library all have to
/// land in the same temporary tree.
@Suite(.serialized) struct StudyTemplateTests {

    // MARK: - Fixtures

    func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "template-\(UUID().uuidString)")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try body(temp)
    }

    /// A MIXED task-prompts file: two `label` rows the answer-token
    /// instruments can read and one `json` row they cannot. The scope pin is
    /// only interesting when the file is mixed.
    @discardableResult
    private func plantTaskPrompts(
        _ relativePath: String = "prompts/tasks/cases.jsonl",
        labelIDs: [String] = ["case-1", "case-2"]
    ) throws -> String {
        let rows =
            labelIDs.map {
                #"{"id":"\#($0)","text":"Decide.","options":["A","B"],"responseFormat":"label"}"#
            } + [
                #"{"id":"essay-1","text":"Explain.","responseFormat":"json"}"#
            ]
        let url = ExperimentStore.resolveProjectPath(relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (rows.joined(separator: "\n") + "\n").write(
            to: url, atomically: true, encoding: .utf8)
        return relativePath
    }

    /// An agent artifact in the workspace's agent library.
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

    /// A compare-agents study with the full measurement surface pinned.
    private func makeComparisonStudy(
        named name: String = "vignette-replication"
    ) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "Vignette replication",
            modelID: "test/model", modelRevision: "abc123")
        let file = try plantTaskPrompts()
        try ExperimentStore.pinTaskPrompts(file, into: &manifest)
        manifest.studyType = StudyIntent.agentComparison.rawValue
        manifest.outcomeInstruments = ["answerTokenLogprob"]
        manifest.maxTokens = 512
        try ExperimentStore.save(manifest)
        return try ExperimentStore.declareOutcomeInstrumentScope(
            responseFormats: ["label"], experimentName: name)
    }

    /// A two-seat SEMANTIC panel: roles, turns, materials, no bindings.
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

    /// A multi-agent study whose scenario is the semantic panel written to the
    /// panel library — the shape a template is minted from.
    private func makePanelStudy(
        named name: String = "allocation-study"
    ) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "Allocation panel",
            modelID: "test/model", modelRevision: "abc123")
        let record = try MultiAgentScenarioStore.save(semanticPanel())
        manifest.studyType = StudyIntent.multiAgent.rawValue
        manifest.studyKind = .multiAgent
        manifest.multiAgentScenarioPath = FineTuneStore.relativePath(for: record.url)
        manifest.multiAgentScenarioHash = try MultiAgentScenarioStore.hash(record.url)
        manifest.multiAgentIncludeBaseline = true
        try ExperimentStore.save(manifest)
        return manifest
    }

    // MARK: - Round trip

    @Test func templateRoundTripsThroughTheWorkspace() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let mint = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication")
            #expect(mint.minted)

            let loaded = try StudyTemplateStore.load(name: mint.template.name)
            #expect(loaded == mint.template)
            #expect(StudyTemplateStore.hash(loaded) == mint.hash)
            #expect(StudyTemplateStore.list().map(\.name) == [mint.template.name])
            #expect(loaded.intent == .agentComparison)
        }
    }

    @Test func mintingStripsTheInstanceAndKeepsTheMeasurementSurface() throws {
        try withTempWorkspace { _ in
            var study = try makeComparisonStudy()
            let agent = try plantAgent(named: "sympathy-agent")
            try ExperimentStore.attachAgent(agent, into: &study)
            study.conditions = [
                .init(name: "sympathy-recommended", slots: [
                    .init(concept: "sympathy", layer: 20, alpha: 0.08)
                ])
            ]
            try ExperimentStore.save(study)

            let body = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication"
            ).template.study

            // Gone: the instance and its arms.
            #expect(body.name.isEmpty)
            #expect(body.createdAt.isEmpty)
            #expect(body.variantConditions.isEmpty)
            #expect(body.conditions.isEmpty)
            #expect(body.status == .draft)
            #expect(body.templateProvenance == nil)
            // Kept: everything that makes two runs comparable.
            #expect(body.modelID == "test/model")
            #expect(body.modelRevision == "abc123")
            #expect(body.taskPromptsFile == "prompts/tasks/cases.jsonl")
            #expect(body.taskPromptsHash == study.taskPromptsHash)
            #expect(body.outcomeInstruments == ["answerTokenLogprob"])
            #expect(body.outcomeInstrumentScope?.responseFormats == ["label"])
            #expect(body.maxTokens == 512)
            #expect(body.studyType == StudyIntent.agentComparison.rawValue)
        }
    }

    @Test func templateNameComesFromTheDisplayLabelWhenSet() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            try ExperimentStore.setDisplayLabel(
                "Vignette — Wave 2", experimentName: "vignette-replication")
            let first = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication")
            #expect(first.template.name == "vignette-wave-2")

            // A second, unrelated study with the same label suffixes rather
            // than colliding.
            try makeComparisonStudy(named: "vignette-again")
            try ExperimentStore.setDisplayLabel(
                "Vignette — Wave 2", experimentName: "vignette-again")
            let second = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-again")
            #expect(second.template.name == "vignette-wave-2-2")
        }
    }

    // MARK: - Mint dedup

    @Test func reloadingAnUndivergedInstanceReturnsTheExistingTemplate() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let original = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication")
            let agent = try plantAgent(named: "sympathy-agent")

            let minted = try StudyTemplateStore.instantiate(
                templateName: original.template.name, cell: .agents([agent]))

            let reloaded = try StudyTemplateStore.templateFromStudy(
                experimentName: minted.name)
            #expect(reloaded.minted == false)
            #expect(reloaded.template.name == original.template.name)
            #expect(reloaded.hash == original.hash)
            // Nothing new landed on disk.
            #expect(StudyTemplateStore.list().count == 1)
        }
    }

    @Test func adivergedInstanceMintsANewTemplateAndRecordsItsParent() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let original = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication")
            let agent = try plantAgent(named: "sympathy-agent")
            var minted = try StudyTemplateStore.instantiate(
                templateName: original.template.name, cell: .agents([agent]))

            // Edit a MEASUREMENT setting — this is now a different study.
            minted.maxTokens = 1024
            try ExperimentStore.save(minted)

            let reloaded = try StudyTemplateStore.templateFromStudy(
                experimentName: minted.name)
            #expect(reloaded.minted)
            #expect(reloaded.template.name != original.template.name)
            #expect(reloaded.divergedFrom == original.template.name)
            #expect(reloaded.template.parentTemplate == original.template.name)
            #expect(reloaded.template.study.maxTokens == 1024)
            #expect(StudyTemplateStore.list().count == 2)
        }
    }

    @Test func changingOnlyTheCastDoesNotDivergeTheTemplate() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let original = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication")
            let one = try plantAgent(named: "sympathy-agent")
            let two = try plantAgent(named: "anger-agent")

            let a = try StudyTemplateStore.instantiate(
                templateName: original.template.name, cell: .agents([one]))
            let b = try StudyTemplateStore.instantiate(
                templateName: original.template.name, cell: .agents([one, two]))

            for study in [a, b] {
                let reloaded = try StudyTemplateStore.templateFromStudy(
                    experimentName: study.name)
                #expect(reloaded.minted == false)
                #expect(reloaded.template.name == original.template.name)
            }
        }
    }

    // MARK: - Instantiation

    @Test func instantiationFillsArmsThroughThePanelsOwnAttachPath() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let template = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication"
            ).template
            let one = try plantAgent(named: "sympathy-agent")
            let two = try plantAgent(named: "anger-agent")

            let minted = try StudyTemplateStore.instantiate(
                templateName: template.name, cell: .agents([one, two]))

            // Byte-for-byte what the Studies panel's Add-agent button writes.
            let expected = [
                try ExperimentStore.agentCondition(for: one),
                try ExperimentStore.agentCondition(for: two),
            ]
            #expect(minted.variantConditions == expected)
            #expect(minted.variantConditions.map(\.name)
                == ["sympathy-agent", "anger-agent"])
            #expect(minted.variantConditions.allSatisfy {
                $0.artifactPath.hasPrefix("runs/model-variants/")
                    && $0.artifactHash.count == 64
            })
            // An ordinary draft in every other respect.
            #expect(minted.status == .draft)
            #expect(minted.frozenAt == nil && minted.freezeHash == nil)
            #expect(!minted.createdAt.isEmpty)
            #expect(try ExperimentStore.load(name: minted.name) == minted)
        }
    }

    @Test func instantiationRepinsTheInstrumentScopeAgainstTheTaskFile() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let template = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication"
            ).template
            let stalePin = template.study.outcomeInstrumentScope
            #expect(stalePin?.itemCount == 2)

            // Simulate the failure this guard exists for: the template's
            // scope pin is stale relative to the file it names. The task
            // hash is re-pinned alongside, because a template whose task
            // bytes drifted must refuse rather than mint (tested below).
            var doctored = template
            doctored.study.outcomeInstrumentScope = ResponseFormat.Scope(
                responseFormats: ["label"], itemCount: 99,
                itemIDsHash: String(repeating: "0", count: 64))
            try StudyTemplateStore.save(doctored)

            let agent = try plantAgent(named: "sympathy-agent")
            let minted = try StudyTemplateStore.instantiate(
                templateName: doctored.name, cell: .agents([agent]))

            let scope = try #require(minted.outcomeInstrumentScope)
            #expect(scope.responseFormats == ["label"])
            #expect(scope.itemCount == 2)  // the two `label` rows, not the json one
            #expect(scope.itemIDsHash == stalePin?.itemIDsHash)
            // The re-pin is what verify would have computed itself.
            let items = try TaskPromptsDocument.load(
                Data(contentsOf: ExperimentStore.resolveProjectPath(
                    minted.taskPromptsFile ?? ""))
            ).responseFormatItems
            #expect(scope.driftRefusal(items: items) == nil)
        }
    }

    @Test func instantiationRefusesWhenTheTaskPromptsDrifted() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let template = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication"
            ).template
            // The file the template pins gains an item.
            try plantTaskPrompts(labelIDs: ["case-1", "case-2", "case-3"])
            let agent = try plantAgent(named: "sympathy-agent")

            #expect(throws: (any Error).self) {
                try StudyTemplateStore.instantiate(
                    templateName: template.name, cell: .agents([agent]))
            }
        }
    }

    @Test func instantiationRefusesAnAgentBuiltOnADifferentModel() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let template = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication"
            ).template
            let foreign = try plantAgent(
                named: "foreign-agent", baseModel: "other/model")

            #expect(throws: (any Error).self) {
                try StudyTemplateStore.instantiate(
                    templateName: template.name, cell: .agents([foreign]))
            }
        }
    }

    @Test func provenanceIsStampedAndEntersTheContentHash() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let template = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication"
            ).template
            let agent = try plantAgent(named: "sympathy-agent")
            let minted = try StudyTemplateStore.instantiate(
                templateName: template.name, cell: .agents([agent]))

            let stamp = try #require(minted.templateProvenance)
            #expect(stamp.template == template.name)
            #expect(stamp.templateHash == StudyTemplateStore.hash(template))
            #expect(stamp.batchGroup == nil)

            // Deliberately hashed: two studies identical but for their
            // lineage are different preregistrations.
            var unstamped = minted
            unstamped.templateProvenance = nil
            #expect(
                ExperimentStore.manifestHash(minted)
                    != ExperimentStore.manifestHash(unstamped))

            // And it survives the manifest round trip on disk.
            #expect(try ExperimentStore.load(name: minted.name)
                .templateProvenance == stamp)
        }
    }

    @Test func aManifestWithoutProvenanceKeepsItsContentHash() throws {
        // The stamp is omit-when-nil, so every study authored before templates
        // existed hashes exactly as it did.
        let manifest = ExperimentManifest(
            name: "legacy", description: "d", modelID: "test/model")
        let encoded = try JSONEncoder().encode(manifest)
        let object = try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        #expect(object?["templateProvenance"] == nil)
    }

    // MARK: - Batch minting

    @Test func batchMintingStampsOneSharedBatchGroup() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let template = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication"
            ).template
            let one = try plantAgent(named: "sympathy-agent")
            let two = try plantAgent(named: "anger-agent")

            let minted = try StudyTemplateStore.instantiateBatch(
                templateName: template.name,
                cells: [.agents([]), .agents([one]), .agents([two]), .agents([one, two])])

            #expect(minted.count == 4)
            let groups = Set(minted.compactMap(\.templateProvenance?.batchGroup))
            #expect(groups.count == 1)
            #expect(groups.first?.hasPrefix("batch-") == true)
            // Distinct studies, auto-named from the casting.
            #expect(Set(minted.map(\.name)).count == 4)
            #expect(minted[0].name.hasSuffix("-baseline"))
            #expect(minted[1].name.hasSuffix("-sympathy-agent"))
            #expect(minted[3].name.hasSuffix("-sympathy-agent-anger-agent"))
            #expect(minted[0].variantConditions.isEmpty)
            #expect(minted[3].variantConditions.count == 2)
        }
    }

    // MARK: - Seat assignment expansion (pure)

    @Test func distinctAssignmentsPermuteThreeDifferentAgents() throws {
        let seats = ["s1", "s2", "s3"]
        let occupants: [SeatOccupant] = [
            .agent(name: "a", artifactPath: "runs/a.json", artifactHash: "1"),
            .agent(name: "b", artifactPath: "runs/b.json", artifactHash: "2"),
            .agent(name: "c", artifactPath: "runs/c.json", artifactHash: "3"),
        ]
        let assignments = try PanelComposition.distinctAssignments(
            seatIDs: seats, occupants: occupants)
        #expect(assignments.count == 6)
        #expect(Set(assignments.map(\.descriptor)).count == 6)
        #expect(assignments.allSatisfy { $0.seatIDs == seats })
    }

    @Test func distinctAssignmentsDedupeARepeatedAgent() throws {
        let seats = ["s1", "s2", "s3"]
        let a = SeatOccupant.agent(
            name: "a", artifactPath: "runs/a.json", artifactHash: "1")
        let b = SeatOccupant.agent(
            name: "b", artifactPath: "runs/b.json", artifactHash: "2")
        // [A, A, B] is three panels, not six: swapping the two A's is the
        // same panel, and running it twice would double-count the condition.
        let assignments = try PanelComposition.distinctAssignments(
            seatIDs: seats, occupants: [a, a, b])
        #expect(assignments.count == 3)
        #expect(assignments.map(\.descriptor) == ["a-a-b", "a-b-a", "b-a-a"])

        // All-identical collapses to one; baseline counts as an occupant.
        #expect(try PanelComposition.distinctAssignments(
            seatIDs: seats, occupants: [a, a, a]).count == 1)
        #expect(try PanelComposition.distinctAssignments(
            seatIDs: seats, occupants: [a, .baseline, .baseline]).count == 3)
    }

    @Test func distinctAssignmentsRefuseACountMismatch() throws {
        #expect(throws: (any Error).self) {
            try PanelComposition.distinctAssignments(
                seatIDs: ["s1", "s2"], occupants: [.baseline])
        }
    }

    @Test func compositionSweepIsAllBaselineSoloEachAndAllTreated() throws {
        let agent = SeatOccupant.agent(
            name: "sym", artifactPath: "runs/s.json", artifactHash: "1")
        let sweep = PanelComposition.compositionSweep(
            seatIDs: ["s1", "s2", "s3"], agent: agent)
        #expect(sweep.map(\.descriptor) == [
            "baseline-baseline-baseline",
            "sym-baseline-baseline",
            "baseline-sym-baseline",
            "baseline-baseline-sym",
            "sym-sym-sym",
        ])
        // A one-seat panel's "all treated" IS its solo casting.
        #expect(PanelComposition.compositionSweep(seatIDs: ["s1"], agent: agent)
            .map(\.descriptor) == ["baseline", "sym"])
    }

    // MARK: - Compile

    @Test func aCompiledPanelRoundTripsThroughTheExistingScenarioLoader() throws {
        try withTempWorkspace { _ in
            let semantic = semanticPanel()
            #expect(PanelComposition.isSemantic(semantic))
            // The semantic form is deliberately NOT runnable.
            #expect(throws: (any Error).self) {
                try MultiAgentRunner.validate(semantic)
            }

            let agent = try plantAgent(named: "sympathy-agent")
            let assignment = SeatAssignment(
                seatIDs: ["seat-proposer", "seat-reviewer"],
                ordered: [
                    .agent(
                        name: agent.artifact.name,
                        artifactPath: ModelVariantStore.relativePath(for: agent),
                        artifactHash: try ModelVariantStore.hash(agent.url)),
                    .baseline,
                ])
            let compiled = try PanelComposition.compileAndPin(
                semantic: semantic, assignment: assignment,
                modelID: "test/model", temperature: 0, maxTokens: 512,
                fileSlug: "allocation-cast-1")

            // Decoded by the SAME loader the run path uses, off the bytes on
            // disk, and accepted by the engine's own validator.
            let url = ExperimentStore.resolveProjectPath(compiled.path)
            let data = try Data(contentsOf: url)
            let reloaded = try JSONDecoder().decode(
                MultiAgentScenario.self, from: data)
            try MultiAgentRunner.validate(reloaded)
            #expect(MultiAgentScenarioStore.hash(data) == compiled.hash)
            #expect(reloaded.baseModelID == "test/model")
            #expect(reloaded.maxTokens == 512)
            #expect(reloaded.agents.allSatisfy { $0.baseModelID == "test/model" })
            #expect(reloaded.agents[0].variantArtifactPath
                == ModelVariantStore.relativePath(for: agent))
            #expect(reloaded.agents[1].variantArtifactPath == nil)
            // Roles survive compilation; only bindings change.
            #expect(reloaded.agents.map(\.systemPrompt)
                == semantic.agents.map(\.systemPrompt))
            #expect(reloaded.turns == semantic.turns)
            // Stripping it back reproduces the semantic panel exactly, which
            // is what makes an instance recognisable as un-diverged.
            #expect(PanelComposition.semanticForm(reloaded) == semantic)

            // Compiled panels stay OUT of the panel picker.
            #expect(compiled.path.hasPrefix("prompts/panels/compiled/"))
            #expect(!MultiAgentScenarioStore.scan().contains { $0.url == url })
        }
    }

    @Test func compileRefusesAnUncastOrUnknownSeat() throws {
        try withTempWorkspace { _ in
            let semantic = semanticPanel()
            #expect(throws: (any Error).self) {
                try PanelComposition.compile(
                    semantic: semantic,
                    assignment: SeatAssignment(
                        seatIDs: ["seat-proposer"],
                        occupants: ["seat-proposer": .baseline]),
                    modelID: "test/model", temperature: 0, maxTokens: 512)
            }
            #expect(throws: (any Error).self) {
                try PanelComposition.compile(
                    semantic: semantic,
                    assignment: SeatAssignment(
                        seatIDs: ["seat-nobody"],
                        occupants: ["seat-nobody": .baseline]),
                    modelID: "test/model", temperature: 0, maxTokens: 512)
            }
        }
    }

    // MARK: - Panel templates end to end

    @Test func aPanelTemplateMintsSiblingStudiesEachWithItsOwnCompiledPanel() throws {
        try withTempWorkspace { _ in
            try makePanelStudy()
            let template = try StudyTemplateStore.templateFromStudy(
                experimentName: "allocation-study"
            ).template
            #expect(template.semanticScenario != nil)
            // The scenario pin is the template's, not the study's copy.
            #expect(template.study.multiAgentScenarioPath == nil)
            #expect(template.study.multiAgentIncludeBaseline)

            let agent = try plantAgent(named: "sympathy-agent")
            let occupant = SeatOccupant.agent(
                name: agent.artifact.name,
                artifactPath: ModelVariantStore.relativePath(for: agent),
                artifactHash: try ModelVariantStore.hash(agent.url))
            let cells = try StudyTemplateStore.compositionSweepCells(
                templateName: template.name, agent: occupant)
            #expect(cells.count == 4)  // all-baseline, 2 solos, all-treated

            let minted = try StudyTemplateStore.instantiateBatch(
                templateName: template.name, cells: cells)
            #expect(minted.count == 4)
            #expect(Set(minted.compactMap(\.templateProvenance?.batchGroup)).count == 1)
            // Each sibling pins its OWN compiled panel.
            let paths = minted.compactMap(\.multiAgentScenarioPath)
            #expect(Set(paths).count == 4)
            for study in minted {
                let url = ExperimentStore.resolveProjectPath(
                    study.multiAgentScenarioPath ?? "")
                let data = try Data(contentsOf: url)
                #expect(MultiAgentScenarioStore.hash(data)
                    == study.multiAgentScenarioHash)
                try MultiAgentRunner.validate(
                    try JSONDecoder().decode(MultiAgentScenario.self, from: data))
            }
            // The all-baseline sibling steers nobody; the all-treated one
            // steers every seat.
            let first = try JSONDecoder().decode(
                MultiAgentScenario.self,
                from: Data(contentsOf: ExperimentStore.resolveProjectPath(paths[0])))
            let last = try JSONDecoder().decode(
                MultiAgentScenario.self,
                from: Data(contentsOf: ExperimentStore.resolveProjectPath(paths[3])))
            #expect(first.agents.allSatisfy { $0.variantArtifactPath == nil })
            #expect(last.agents.allSatisfy { $0.variantArtifactPath != nil })
        }
    }

    @Test func aPanelInstanceReloadsAsItsOwnTemplate() throws {
        try withTempWorkspace { _ in
            try makePanelStudy()
            let original = try StudyTemplateStore.templateFromStudy(
                experimentName: "allocation-study")
            let seats = try StudyTemplateStore.semanticSeatIDs(
                templateName: original.template.name)
            let minted = try StudyTemplateStore.instantiate(
                templateName: original.template.name,
                cell: .seating(
                    SeatAssignment(seatIDs: seats, ordered: [.baseline, .baseline])))

            let reloaded = try StudyTemplateStore.templateFromStudy(
                experimentName: minted.name)
            #expect(reloaded.minted == false)
            #expect(reloaded.template.name == original.template.name)
        }
    }

    // MARK: - Legacy hoist

    @Test func hoistSplitsABoundPanelIntoSemanticFormAndCasting() throws {
        try withTempWorkspace { _ in
            let agent = try plantAgent(named: "sympathy-agent")
            let agentPath = ModelVariantStore.relativePath(for: agent)
            let agentHash = try ModelVariantStore.hash(agent.url)
            var bound = semanticPanel()
            bound.baseModelID = "test/model"
            bound.temperature = 0.7
            bound.maxTokens = 300
            bound.agents[0].baseModelID = "test/model"
            bound.agents[0].variantArtifactPath = agentPath
            bound.agents[0].variantArtifactHash = agentHash
            bound.agents[1].baseModelID = "test/model"
            let record = try MultiAgentScenarioStore.save(bound)

            let hoist = try PanelComposition.hoistLegacyScenario(
                path: FineTuneStore.relativePath(for: record.url))

            #expect(hoist.warnings.isEmpty)
            #expect(hoist.modelID == "test/model")
            #expect(hoist.temperature == 0.7)
            #expect(hoist.maxTokens == 300)
            #expect(PanelComposition.isSemantic(hoist.semantic))
            #expect(hoist.semantic.agents.map(\.systemPrompt)
                == bound.agents.map(\.systemPrompt))
            #expect(
                hoist.assignment["seat-proposer"]
                    == SeatOccupant.agent(
                        name: "Proposer", artifactPath: agentPath,
                        artifactHash: agentHash))
            #expect(hoist.assignment["seat-reviewer"] == SeatOccupant.baseline)

            // Re-compiling the halves reproduces the panel it came from.
            let recompiled = try PanelComposition.compile(
                semantic: hoist.semantic, assignment: hoist.assignment,
                modelID: hoist.modelID, temperature: hoist.temperature,
                maxTokens: hoist.maxTokens)
            #expect(recompiled == bound)
        }
    }

    @Test func hoistWarnsLoudlyAboutDivergentPerSeatModels() throws {
        try withTempWorkspace { _ in
            var bound = semanticPanel()
            bound.baseModelID = ""
            bound.agents[0].baseModelID = "test/model"
            bound.agents[1].baseModelID = "other/model"
            let record = try MultiAgentScenarioStore.save(bound)

            let hoist = try PanelComposition.hoistLegacyScenario(
                path: FineTuneStore.relativePath(for: record.url))

            let warning = try #require(hoist.warnings.first)
            #expect(warning.contains("DIFFERENT models"))
            #expect(warning.contains("Proposer → test/model"))
            #expect(warning.contains("Reviewer → other/model"))
            // Never unified in silence: one model is chosen AND reported.
            #expect(["test/model", "other/model"].contains(hoist.modelID))
        }
    }

    // MARK: - Template store housekeeping

    @Test func renameMovesTheTemplateDirectoryAndDeleteRemovesIt() throws {
        try withTempWorkspace { _ in
            try makeComparisonStudy()
            let mint = try StudyTemplateStore.templateFromStudy(
                experimentName: "vignette-replication")

            let renamed = try StudyTemplateStore.rename(
                templateName: mint.template.name, to: "Vignette Wave 3")
            #expect(renamed == "vignette-wave-3")
            #expect(try StudyTemplateStore.load(name: "vignette-wave-3").name
                == "vignette-wave-3")
            #expect(throws: (any Error).self) {
                try StudyTemplateStore.load(name: mint.template.name)
            }
            // A rename does not change what the template IS.
            #expect(StudyTemplateStore.hash(
                try StudyTemplateStore.load(name: "vignette-wave-3")) == mint.hash)

            try StudyTemplateStore.delete(name: "vignette-wave-3")
            #expect(StudyTemplateStore.list().isEmpty)
        }
    }
}

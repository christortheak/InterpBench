import Foundation
import Testing

@testable import ExperimentKit

/// The panel-level half of study templates: what a batch will COST before it
/// is minted, per-row isolation when a casting refuses, and the sequencing and
/// failure isolation of the submissions the batch produces.
///
/// These are the rules a SwiftUI body cannot be trusted with. The totals line
/// is the only thing standing between one click and a week of cluster time, so
/// its arithmetic is tested rather than eyeballed; and "one failed submission
/// must not stop the rest" is a property, not a hope — a batch that stops at
/// row 2 leaves four castings unrun and nothing saying so.
///
/// Serialized for the usual reason: the whole suite moves the process-global
/// workspace root.
@Suite(.serialized) struct StudyTemplateBatchTests {

    // MARK: - Fixtures

    /// The WORKSPACE override, not `ExperimentStore.rootOverride`: templates,
    /// experiments, the AGENT LIBRARY and the panel library all have to land in
    /// the same temporary tree, and the agent library resolves through
    /// `VectorCatalog.projectRoot`. (Setting only the experiment root leaves
    /// `ModelVariantStore.scan()` reading the real workspace, which shows up as
    /// agents from other tests appearing in the picker.)
    func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "tbatch-\(UUID().uuidString)")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try body(temp)
    }

    /// The async twin: minting a batch is `async` (submission is), so the
    /// override window has to survive an await. Same blocking semaphore the
    /// sync helper uses — legitimate because the whole suite is serialized.
    @MainActor
    func withTempWorkspace<T>(
        _ body: @MainActor (URL) async throws -> T
    ) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "tbatch-\(UUID().uuidString)")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    private func plantTaskPrompts(
        _ relativePath: String = "prompts/tasks/cases.jsonl", rows: Int = 3
    ) throws -> String {
        let lines = (1 ... rows).map {
            #"{"id":"case-\#($0)","text":"Decide.","options":["A","B"],"responseFormat":"label"}"#
        }
        let url = ExperimentStore.resolveProjectPath(relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (lines.joined(separator: "\n") + "\n").write(
            to: url, atomically: true, encoding: .utf8)
        return relativePath
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

    /// The library id of a planted agent AS THE TABLE SEES IT.
    ///
    /// `ModelVariantRecord.id` is the artifact's file path, and a hand-built
    /// record's path is not necessarily the standardized one `scan()` returns
    /// under a symlinked temp root — the UI always picks from the model's own
    /// list, so the tests do too.
    @MainActor
    private func libraryID(
        _ model: TemplateInstantiation, _ name: String
    ) throws -> String {
        try #require(model.agents.first { $0.artifact.name == name }).id
    }

    private func semanticPanel(seats: Int = 2) -> MultiAgentScenario {
        PanelComposition.semanticForm(
            MultiAgentScenario(
                name: "allocation-panel",
                baseModelID: "",
                sharedMaterials: "Two teams share 100 credits.",
                agents: (0 ..< seats).map {
                    .init(
                        id: "seat-\($0)", name: "Seat \($0)", baseModelID: "",
                        systemPrompt: "You represent team \($0).")
                },
                turns: (0 ..< seats).map {
                    .init(
                        id: "turn-\($0)", title: "Turn \($0)",
                        speakerAgentID: "seat-\($0)",
                        promptTemplate: "Speak.",
                        outputLabel: "out-\($0)")
                }))
    }

    /// A compare-agents template with three task items.
    @discardableResult
    private func makeComparisonTemplate(
        named name: String = "vignette-replication", samples: Int? = nil
    ) throws -> StudyTemplate {
        var manifest = try ExperimentStore.create(
            name: name, description: "Vignette replication",
            modelID: "test/model", modelRevision: "abc123")
        let file = try plantTaskPrompts()
        try ExperimentStore.pinTaskPrompts(file, into: &manifest)
        manifest.studyType = StudyIntent.agentComparison.rawValue
        manifest.samplesPerItem = samples
        try ExperimentStore.save(manifest)
        return try StudyTemplateStore.templateFromStudy(experimentName: name)
            .template
    }

    /// A multi-agent template over a semantic panel.
    @discardableResult
    private func makePanelTemplate(
        named name: String = "allocation-study", seats: Int = 2,
        samples: Int? = nil, includeBaseline: Bool = true
    ) throws -> StudyTemplate {
        var manifest = try ExperimentStore.create(
            name: name, description: "Allocation panel",
            modelID: "test/model", modelRevision: "abc123")
        let record = try MultiAgentScenarioStore.save(semanticPanel(seats: seats))
        manifest.studyType = StudyIntent.multiAgent.rawValue
        manifest.studyKind = .multiAgent
        manifest.multiAgentScenarioPath = FineTuneStore.relativePath(for: record.url)
        manifest.multiAgentScenarioHash = try MultiAgentScenarioStore.hash(record.url)
        manifest.multiAgentIncludeBaseline = includeBaseline
        manifest.samplesPerItem = samples
        try ExperimentStore.save(manifest)
        return try StudyTemplateStore.templateFromStudy(experimentName: name)
            .template
    }

    // MARK: - Totals: the number that makes batch size a visible choice

    @Test func panelTotalsMultiplyCastingsArmsAndPlayThroughs() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 3, samples: 20)
            // Six castings, the manifest's baseline+configured pair, twenty
            // play-throughs each: the batch the researcher must SEE before
            // pressing a button.
            let totals = TemplateBatchTotals.totals(
                template: template, armsPerRow: Array(repeating: 2, count: 6),
                taskItemCount: 1, shardsPerStudy: 4, jobNoun: "Slurm jobs")
            #expect(totals.units == 240)
            #expect(totals.jobs == 24)
            #expect(totals.unitNoun == "transcripts")
            #expect(totals.summary
                == "6 castings × 2 arms × 20 play-throughs = 240 transcripts "
                    + "· 24 Slurm jobs")
        }
    }

    @Test func comparisonTotalsCountItemsAndTheBaselineArm() throws {
        try withTempWorkspace { _ in
            let template = try makeComparisonTemplate()
            // Two agents cast → three arms (the run loop always prepends the
            // unsteered baseline), three task items.
            let arms = TemplateBatchTotals.armsInRow(
                template: template, castAgentCount: 2)
            #expect(arms == 3)
            let totals = TemplateBatchTotals.totals(
                template: template, armsPerRow: [arms, arms],
                taskItemCount: TemplateBatchTotals.taskItemCount(template),
                shardsPerStudy: 1)
            #expect(totals.items == 3)
            #expect(totals.units == 18)
            #expect(totals.jobs == 2)
            #expect(totals.summary
                == "2 studies × 3 arms × 3 items = 18 generations · 2 jobs")
        }
    }

    @Test func totalsReportUnevenArmsAsATotalRatherThanAFalseAverage() throws {
        try withTempWorkspace { _ in
            let template = try makeComparisonTemplate()
            // "2 studies × 2.5 arms" would be a lie; the honest form is the
            // sum, because one row runs two arms and the other three.
            let totals = TemplateBatchTotals.totals(
                template: template, armsPerRow: [2, 3], taskItemCount: 1,
                shardsPerStudy: 1)
            #expect(totals.summary.contains("5 arms total"))
            #expect(totals.units == 5)
        }
    }

    @Test func panelWithoutABaselineArmRunsOneArm() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(includeBaseline: false)
            #expect(
                TemplateBatchTotals.armsInRow(template: template, castAgentCount: 0)
                    == 1)
        }
    }

    @Test func anEmptyTableSaysSoRatherThanClaimingZeroWork() throws {
        try withTempWorkspace { _ in
            let template = try makeComparisonTemplate()
            let totals = TemplateBatchTotals.totals(
                template: template, armsPerRow: [], taskItemCount: 3,
                shardsPerStudy: 1)
            #expect(totals.summary == "no rows yet — add a casting below")
        }
    }

    // MARK: - Per-row isolation at mint

    @Test func aRefusedCastingDoesNotStopTheRestOfTheBatch() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 2)
            let seats = try StudyTemplateStore.semanticSeatIDs(
                templateName: template.name)
            let good = StudyTemplateStore.Cell.seating(
                SeatAssignment(seatIDs: seats, ordered: [.baseline, .baseline]))
            // A casting naming a seat this panel does not have: `compile`
            // refuses it, and the two legal castings around it must still land.
            let bad = StudyTemplateStore.Cell.seating(
                SeatAssignment(
                    seatIDs: ["seat-does-not-exist"],
                    occupants: ["seat-does-not-exist": .baseline]))

            let mint = StudyTemplateStore.mintBatch(
                templateName: template.name, cells: [good, bad, good])

            #expect(mint.minted.count == 2)
            #expect(mint.failures.map(\.row) == [1])
            // The refusal is carried VERBATIM — it names the seats.
            #expect(mint.failures[0].failure?.contains("seat") == true)
            // One batch id across everything that landed.
            let studies = ExperimentStore.list()
                .filter { $0.templateProvenance?.batchGroup == mint.batchGroup }
            #expect(studies.count == 2)
        }
    }

    @Test func everyMintedSiblingSharesOneBatchGroup() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 2)
            let agent = try plantAgent(named: "sympathy-agent")
            let occupant = SeatOccupant.agent(
                name: agent.artifact.name,
                artifactPath: ModelVariantStore.relativePath(for: agent),
                artifactHash: try ModelVariantStore.hash(agent.url))
            let cells = try StudyTemplateStore.compositionSweepCells(
                templateName: template.name, agent: occupant)
            #expect(cells.count == 4)  // all-baseline, two solos, all-treated

            let mint = StudyTemplateStore.mintBatch(
                templateName: template.name, cells: cells)
            #expect(mint.failures.isEmpty)
            let groups = Set(
                mint.minted.compactMap {
                    try? ExperimentStore.load(name: $0)
                        .templateProvenance?.batchGroup
                })
            #expect(groups == [mint.batchGroup])
        }
    }

    @Test func aRequestedNameOverridesTheDerivedOne() throws {
        try withTempWorkspace { _ in
            let template = try makeComparisonTemplate()
            let agent = try plantAgent(named: "sympathy-agent")
            let mint = StudyTemplateStore.mintBatch(
                templateName: template.name,
                cells: [.agents([agent]), .agents([agent])],
                names: ["wave-2-sympathy", nil])
            #expect(mint.minted.first == "wave-2-sympathy")
            #expect(mint.minted.last?.hasPrefix(template.name) == true)
        }
    }

    // MARK: - Batch submission: sequencing and failure isolation

    /// A submitter that records the order it was called in and fails the
    /// studies it is told to.
    @MainActor final class FakeSubmitter {
        var calls: [String] = []
        var failing: Set<String> = []

        func submit(_ study: String) -> Result<String, StudyBatchSubmission.Failure> {
            calls.append(study)
            guard !failing.contains(study) else {
                return .failure(.init(reason: "queue refused \(study)"))
            }
            return .success("job-\(calls.count)")
        }
    }

    @Test @MainActor func batchSubmitIsSequentialAndInOrder() async {
        let fake = FakeSubmitter()
        let outcomes = await StudyBatchSubmission.submit(
            studies: ["a", "b", "c"],
            submit: { study in fake.submit(study) })
        #expect(fake.calls == ["a", "b", "c"])
        #expect(outcomes.map(\.jobID) == ["job-1", "job-2", "job-3"])
        #expect(outcomes.filter(\.succeeded).count == 3)
    }

    @Test @MainActor func oneFailedSubmissionDoesNotStopTheRest() async {
        let fake = FakeSubmitter()
        fake.failing = ["b"]
        let outcomes = await StudyBatchSubmission.submit(
            studies: ["a", "b", "c"],
            submit: { study in fake.submit(study) })
        // The failure did not abort the loop.
        #expect(fake.calls == ["a", "b", "c"])
        #expect(outcomes.filter(\.succeeded).map(\.study) == ["a", "c"])
        #expect(outcomes[1].failure == "queue refused b")
    }

    @Test @MainActor func theSummaryNamesEveryStudyThatFailed() async {
        let fake = FakeSubmitter()
        fake.failing = ["b", "d"]
        let outcomes = await StudyBatchSubmission.submit(
            studies: ["a", "b", "c", "d"],
            submit: { study in fake.submit(study) })
        let summary = StudyBatchSubmission.summary(outcomes)
        #expect(summary.hasPrefix("submitted 2 of 4"))
        // Naming them is the point: "2 of 4" alone leaves the researcher
        // diffing job lists by hand.
        #expect(summary.contains("b (queue refused b)"))
        #expect(summary.contains("d (queue refused d)"))
        #expect(!summary.contains("a ("))
    }

    @Test @MainActor func anAllGreenBatchSaysSoWithoutAFailureList() async {
        let fake = FakeSubmitter()
        let outcomes = await StudyBatchSubmission.submit(
            studies: ["a", "b"], submit: { study in fake.submit(study) })
        #expect(StudyBatchSubmission.summary(outcomes) == "submitted 2 of 2")
    }

    // MARK: - The cell table

    @Test @MainActor func aPanelTableStartsFullyCastAtBaseline() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 3)
            let model = TemplateInstantiation(templateName: template.name)
            #expect(model.loadFailure == nil)
            #expect(model.seatIDs.count == 3)
            #expect(model.rows.count == 1)
            // Every seat cast, so the row is READY: an all-baseline casting is
            // the control composition, not an unfinished one.
            #expect(model.refusal(for: model.rows[0]) == nil)
            #expect(model.readyToMint)
        }
    }

    @Test @MainActor func anUncastSeatRefusesWithTheEngineSMessage() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 2)
            let model = TemplateInstantiation(templateName: template.name)
            model.rows[0].seating.removeValue(forKey: "seat-1")
            let refusal = try #require(model.refusal(for: model.rows[0]))
            #expect(refusal.contains("seat-1"))
            #expect(refusal.contains("use baseline for an unsteered seat"))
            #expect(!model.readyToMint)
        }
    }

    /// A zero-agent comparison row MINTS (a baseline-only draft is a legal
    /// study, and the researcher may want the draft before the casting) but
    /// cannot be SUBMITTED — spending cluster time on one arm against nothing
    /// is the case worth stopping at the button.
    @Test @MainActor func aComparisonRowWithNoAgentsAdvisesButStillMints() throws {
        try withTempWorkspace { _ in
            let template = try makeComparisonTemplate()
            try plantAgent(named: "sympathy-agent")
            let model = TemplateInstantiation(templateName: template.name)
            #expect(model.agents.count == 1)
            #expect(model.refusal(for: model.rows[0]) == nil)
            let advisory = try #require(model.advisory(for: model.rows[0]))
            #expect(advisory.contains("no agents cast"))
            #expect(model.readyToMint)
            #expect(!model.readyToSubmit)

            model.rows[0].agentIDs = [model.agents[0].id]
            #expect(model.advisory(for: model.rows[0]) == nil)
            #expect(model.readyToMint)
            #expect(model.readyToSubmit)
        }
    }

    /// A panel seat left uncast is a real refusal on both paths: the casting
    /// cannot compile, so there is nothing to mint.
    @Test @MainActor func anUncastSeatBlocksBothMintAndSubmit() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 2)
            let model = TemplateInstantiation(templateName: template.name)
            model.rows[0].seating.removeValue(forKey: "seat-1")
            #expect(!model.readyToMint)
            #expect(!model.readyToSubmit)
        }
    }

    @Test @MainActor func agentsAreFilteredToTheTemplateSBaseModel() throws {
        try withTempWorkspace { _ in
            let template = try makeComparisonTemplate()
            try plantAgent(named: "same-model")
            try plantAgent(named: "other-model", baseModel: "other/model")
            let model = TemplateInstantiation(templateName: template.name)
            // Filtering here rather than refusing later: `attachAgent` would
            // reject the mismatch anyway, and offering it is a trap.
            #expect(model.agents.map(\.artifact.name) == ["same-model"])
        }
    }

    @Test @MainActor func theCompositionSweepPresetReplacesTheStarterRow() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 3)
            try plantAgent(named: "sympathy-agent")
            let model = TemplateInstantiation(templateName: template.name)
            model.sweepAgentID = try libraryID(model, "sympathy-agent")
            model.addCompositionSweep()
            // all-baseline + three solos + all-treated, and the untouched
            // starter row is discarded rather than minting a study nobody
            // asked for.
            #expect(model.rows.count == 5)
            #expect(model.readyToMint)
        }
    }

    @Test @MainActor func permutationCountShowsTheMultisetDedupe() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 3)
            try plantAgent(named: "agent-a")
            let model = TemplateInstantiation(templateName: template.name)
            // [A, A, baseline] over three seats is 3 distinct castings, not 6
            // — swapping the two A's is the same panel, and running it twice
            // costs GPU hours and double-counts in the analysis.
            let a = try libraryID(model, "agent-a")
            model.permutationAgentIDs = [a, a]
            #expect(model.permutationRefusal == nil)
            #expect(model.permutationCount == 3)
            model.addAllPermutations()
            #expect(model.rows.count == 3)
        }
    }

    @Test @MainActor func tooFewOccupantsForTheSeatsRefusesRatherThanTruncates() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 3)
            let a = try plantAgent(named: "agent-a")
            let model = TemplateInstantiation(templateName: template.name)
            model.permutationPadsWithBaseline = false
            model.permutationAgentIDs = [a.id]
            let refusal = try #require(model.permutationRefusal)
            #expect(refusal.contains("fills every seat exactly once"))
        }
    }

    @Test @MainActor func totalsFollowTheTableLive() throws {
        try withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 2, samples: 10)
            let model = TemplateInstantiation(templateName: template.name)
            model.shardsPerStudy = 2
            model.jobNoun = "Slurm jobs"
            #expect(model.totals.units == 20)  // 1 casting × 2 arms × 10
            model.addRow()
            model.addRow()
            #expect(model.totals.studies == 3)
            #expect(model.totals.units == 60)
            #expect(model.totals.jobs == 6)
        }
    }

    @Test @MainActor func mintWritesOrdinaryDraftsAndReportsTheBatch() async throws {
        try await withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 2)
            let model = TemplateInstantiation(templateName: template.name)
            model.addRow()
            let summary = await model.mint()
            #expect(summary.contains("minted 2 of 2"))
            let batch = try #require(model.lastBatchGroup)
            let minted = ExperimentStore.list().filter {
                $0.templateProvenance?.batchGroup == batch
            }
            #expect(minted.count == 2)
            // ORDINARY drafts: the whole contract of the feature.
            #expect(minted.allSatisfy { $0.status == .draft })
            #expect(minted.allSatisfy { $0.multiAgentScenarioPath != nil })
            for row in model.rows {
                guard case .minted = row.state else {
                    Issue.record("row did not reach .minted")
                    continue
                }
            }
        }
    }

    @Test @MainActor func mintAndSubmitMarksRowsAndIsolatesASubmitFailure() async throws {
        try await withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 2)
            let model = TemplateInstantiation(templateName: template.name)
            model.rows[0].name = "casting-alpha"
            model.addRow()
            model.rows[1].name = "casting-beta"

            let fake = FakeSubmitter()
            fake.failing = ["casting-beta"]
            let summary = await model.mint(submit: { study in fake.submit(study) })

            #expect(fake.calls == ["casting-alpha", "casting-beta"])
            #expect(summary.contains("submitted 1 of 2"))
            #expect(summary.contains("casting-beta"))
            guard case .submitted(let study, let job) = model.rows[0].state else {
                Issue.record("first row should be submitted")
                return
            }
            #expect(study == "casting-alpha")
            #expect(job == "job-1")
            guard case .failed(let message) = model.rows[1].state else {
                Issue.record("second row should carry its failure")
                return
            }
            #expect(message == "queue refused casting-beta")
        }
    }

    @Test @MainActor func mintRefusesWhileTheTableIsNotReady() async throws {
        try await withTempWorkspace { _ in
            let template = try makePanelTemplate(seats: 2)
            let model = TemplateInstantiation(templateName: template.name)
            // A seat with nobody in it: the casting cannot compile, so nothing
            // may be written.
            model.rows[0].seating.removeValue(forKey: "seat-1")
            let summary = await model.mint()
            #expect(summary.isEmpty)
            #expect(ExperimentStore.list().count == 1)  // the source study only
        }
    }

    /// "Load Only" opens the FIRST minted draft in the Studies editor, and
    /// "first" means first in the table — the row the researcher filled in
    /// first, not whatever order the store happened to write.
    @Test @MainActor func mintRecordsTheMintedStudiesInTableOrder() async throws {
        try await withTempWorkspace { _ in
            let template = try makeComparisonTemplate()
            try plantAgent(named: "sympathy-agent")
            let model = TemplateInstantiation(templateName: template.name)
            model.rows[0].name = "wave-a"
            model.addRow()
            model.rows[1].name = "wave-b"
            await model.mint()
            #expect(model.mintedStudies == ["wave-a", "wave-b"])
            #expect(model.firstMintedStudy == "wave-a")
            #expect(model.lastMintWasClean)
        }
    }

    /// A batch with a refused row is NOT clean — the sheet stays up so the
    /// verbatim refusal can be read instead of vanishing with the window.
    @Test @MainActor func aFailedRowLeavesTheBatchUnclean() async throws {
        try await withTempWorkspace { _ in
            let template = try makeComparisonTemplate()
            try plantAgent(named: "sympathy-agent")
            let model = TemplateInstantiation(templateName: template.name)
            try plantTaskPrompts(rows: 5)  // drift the pinned file
            await model.mint()
            #expect(!model.lastMintWasClean)
            #expect(model.firstMintedStudy == nil)
        }
    }

    // MARK: - The panel's own affordances

    // "New from Study" — the one path that fills the design library — is
    // tested in `StudyDesignTests` alongside the divergence check it shares a
    // rule with (the old `loadSelectedStudyAsTemplate` entry point went with
    // the Studies-tab section it lived in, 2026-08-06).

    @Test @MainActor func lineageNamesTheTemplateAndCountsTheBatch() throws {
        try withTempWorkspace { root in
            let template = try makePanelTemplate(seats: 2)
            let seats = try StudyTemplateStore.semanticSeatIDs(
                templateName: template.name)
            let cell = StudyTemplateStore.Cell.seating(
                SeatAssignment(seatIDs: seats, ordered: [.baseline, .baseline]))
            let mint = StudyTemplateStore.mintBatch(
                templateName: template.name, cells: [cell, cell, cell])

            let panel = ExperimentPanel()
            panel.notices = PanelNotices(
                fileURL: root.appending(component: "notices.jsonl"))
            panel.refresh()
            let first = try ExperimentStore.load(name: try #require(mint.minted.first))
            let lineage = try #require(panel.templateLineage(first))
            #expect(lineage.contains("from design '\(template.name)'"))
            #expect(!lineage.contains("diverged"))
            #expect(lineage.contains(mint.batchGroup))
            #expect(lineage.contains("3 studies minted together"))
            #expect(panel.batchSiblings(first).count == 2)

            // A hand-authored study has no lineage line at all.
            let source = try ExperimentStore.load(name: "allocation-study")
            #expect(panel.templateLineage(source) == nil)
        }
    }

    @Test @MainActor func renamingATemplateLeavesMintedLineageAlone() throws {
        try withTempWorkspace { root in
            let template = try makeComparisonTemplate(named: "vignette")
            let agent = try plantAgent(named: "sympathy-agent")
            let minted = try StudyTemplateStore.instantiate(
                templateName: template.name, cell: .agents([agent]))

            let panel = ExperimentPanel()
            panel.notices = PanelNotices(
                fileURL: root.appending(component: "notices.jsonl"))
            panel.refresh()
            panel.renameTemplate(template.name, to: "Vignette Wave 3")
            #expect(panel.templates.map(\.name) == ["vignette-wave-3"])
            #expect(panel.selectedTemplateName == "vignette-wave-3")
            // Provenance records what was true when it was written.
            let reloaded = try ExperimentStore.load(name: minted.name)
            #expect(reloaded.templateProvenance?.template == template.name)
        }
    }

    @Test @MainActor func deletingATemplateLeavesItsStudiesStanding() throws {
        try withTempWorkspace { root in
            let template = try makeComparisonTemplate(named: "vignette")
            let agent = try plantAgent(named: "sympathy-agent")
            let minted = try StudyTemplateStore.instantiate(
                templateName: template.name, cell: .agents([agent]))

            let panel = ExperimentPanel()
            panel.notices = PanelNotices(
                fileURL: root.appending(component: "notices.jsonl"))
            panel.refresh()
            panel.deleteTemplate(template.name)
            #expect(panel.templates.isEmpty)
            #expect(panel.selectedTemplateName == nil)
            let survivor = try ExperimentStore.load(name: minted.name)
            #expect(survivor.templateProvenance?.template == template.name)
        }
    }

    @Test @MainActor func driftedTaskPromptsRefuseEveryRowVerbatim() async throws {
        try await withTempWorkspace { _ in
            let template = try makeComparisonTemplate()
            try plantAgent(named: "sympathy-agent")
            let model = TemplateInstantiation(templateName: template.name)
            model.rows[0].agentIDs = [try libraryID(model, "sympathy-agent")]
            // Edit the pinned file AFTER the template was minted: re-pinning
            // it would launder the drift into a study that verifies cleanly
            // while measuring items the template never described.
            try plantTaskPrompts(rows: 5)
            let summary = await model.mint()
            #expect(summary.contains("minted 0 of 1"))
            #expect(summary.contains("re-mint the template"))
            guard case .failed(let message) = model.rows[0].state else {
                Issue.record("the row should carry the refusal")
                return
            }
            #expect(message.contains("prompts/tasks/cases.jsonl"))
        }
    }
}

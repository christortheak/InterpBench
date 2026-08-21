import Foundation
import Testing

@testable import ExperimentKit

/// The Templates-tab restructure's decision layer: the live divergence check,
/// the read-only design summary, the new-study design picker, and the panel
/// affordances the two tabs render.
///
/// The properties worth testing here are the ones a SwiftUI body would get
/// wrong. Divergence is FILE I/O, so it must be computed once per refresh and
/// read from a cache afterwards — a per-frame recomputation would re-read every
/// minted study's compiled panel on every keystroke. And the honesty of the
/// lineage line is the whole point of the feature: a study that has been edited
/// must stop claiming to be the replication it was minted as, without anything
/// blocking the edit.
///
/// Serialized for the usual reason: the suite moves the process-global
/// workspace root.
@Suite(.serialized) struct StudyDesignTests {

    // MARK: - Fixtures

    func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "tdesign-\(UUID().uuidString)")
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

    @discardableResult
    private func makeDesign(from study: String = "vignette") throws -> StudyTemplate {
        var manifest = try ExperimentStore.create(
            name: study, description: "Vignette replication",
            modelID: "test/model", modelRevision: "abc123def456789")
        let file = try plantTaskPrompts()
        try ExperimentStore.pinTaskPrompts(file, into: &manifest)
        manifest.studyType = StudyIntent.agentComparison.rawValue
        manifest.samplesPerItem = 4
        manifest.maxTokens = 1024
        try ExperimentStore.save(manifest)
        return try StudyTemplateStore.templateFromStudy(experimentName: study).template
    }

    @MainActor
    private func makePanel(root: URL) -> ExperimentPanel {
        let panel = ExperimentPanel()
        panel.notices = PanelNotices(
            fileURL: root.appending(component: "notices.jsonl"))
        panel.refresh()
        return panel
    }

    // MARK: - Does a study still agree with its design?

    @Test func aFreshInstanceAgreesWithItsDesign() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            let agent = try plantAgent(named: "sympathy-agent")
            let minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([agent]))
            // The agents are exactly what a design does NOT hold, so casting
            // them must not read as a change to the design.
            #expect(StudyTemplateStore.agreement(of: minted) == .matches)
        }
    }

    @Test func anEditedInstanceReadsAsDiverged() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            var minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            minted.maxTokens = 4096  // a design-bearing setting
            try ExperimentStore.save(minted)
            #expect(
                StudyTemplateStore.agreement(of: try ExperimentStore.load(name: minted.name))
                    == .diverged)
        }
    }

    @Test func aStudyWithNoLineageIsNotJudgedAgainstAnyDesign() throws {
        try withTempWorkspace { _ in
            try makeDesign()
            let source = try ExperimentStore.load(name: "vignette")
            #expect(StudyTemplateStore.agreement(of: source) == .noLineage)
        }
    }

    /// A deleted or renamed design is ITS OWN state. Calling it divergence
    /// would blame the study for a change made to the library.
    @Test func aMissingDesignIsSaidRatherThanBlamedOnTheStudy() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            let minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            try StudyTemplateStore.delete(name: design.name)
            #expect(StudyTemplateStore.agreement(of: minted) == .designMissing)
        }
    }

    /// The description is excluded from the design's content hash, which is
    /// exactly why the read-only library lets you edit it.
    @Test @MainActor func editingADescriptionCannotMakeAnInstanceDiverge() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            let minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            let panel = makePanel(root: root)
            panel.updateTemplateDescription(design.name, to: "wave 3, re-run for the appendix")
            #expect(
                panel.templates.first?.templateDescription
                    == "wave 3, re-run for the appendix")
            #expect(StudyTemplateStore.agreement(of: minted) == .matches)
        }
    }

    // MARK: - The panel caches divergence, and the line says it

    @Test @MainActor func lineageSaysDivergedWithoutBlockingTheEdit() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            var minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            let panel = makePanel(root: root)
            let clean = try #require(
                panel.templateLineage(try ExperimentStore.load(name: minted.name)))
            #expect(clean.hasPrefix("from design '\(design.name)'"))

            minted.maxTokens = 4096
            try ExperimentStore.save(minted)  // edits are never blocked
            panel.refresh()
            let reloaded = try ExperimentStore.load(name: minted.name)
            let line = try #require(panel.templateLineage(reloaded))
            #expect(line.hasPrefix("diverged from design '\(design.name)'"))
            #expect(line.contains("edits are allowed"))
        }
    }

    /// Divergence is computed ONCE per refresh, never per call: the check
    /// re-reads files, and `templateLineage` is called from a view body.
    @Test @MainActor func divergenceComesFromTheRefreshCacheNotFromDisk() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            var minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            let panel = makePanel(root: root)
            #expect(panel.designLineage[minted.name]?.agreement == .matches)

            // Change the study on disk WITHOUT refreshing. A per-call
            // implementation would notice; a cached one must not.
            minted.maxTokens = 4096
            try ExperimentStore.save(minted)
            let stale = try #require(panel.templateLineage(minted))
            #expect(!stale.contains("diverged"))
            #expect(panel.designLineage[minted.name]?.agreement == .matches)

            panel.refresh()
            #expect(panel.designLineage[minted.name]?.agreement == .diverged)
        }
    }

    @Test @MainActor func onlyMintedStudiesAppearInTheAgreementCache() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            let panel = makePanel(root: root)
            // The hand-authored source study has no lineage, so it is absent
            // rather than stored as ".noLineage" — nothing should pay for a
            // per-study file read it cannot use.
            #expect(panel.designLineage["vignette"] == nil)
            #expect(panel.designLineage.count == 1)
        }
    }

    // MARK: - Two facts, not one

    /// The review finding (2026-08-06). Comparing a study to its MINT-time
    /// stamp is deliberate — revising a design must never re-file older
    /// instances as diverged — but on its own it lets an untouched study keep
    /// reading "matches" after the recipe under that name has moved on. Both
    /// facts are reported, and neither is inferred from the other.
    @Test @MainActor func aRevisedDesignIsSaidWithoutBlamingTheStudy() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            let minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))

            let panel = makePanel(root: root)
            let before = try #require(panel.designLineage[minted.name])
            #expect(before.agreement == .matches)
            #expect(!before.designRevised)
            let cleanLine = try #require(panel.templateLineage(minted))
            #expect(!cleanLine.contains("has since been revised"))

            // Revise the DESIGN, leaving the study untouched.
            var revised = try StudyTemplateStore.load(name: design.name)
            revised.study.maxTokens = 4096
            try StudyTemplateStore.save(revised)

            panel.refresh()
            let after = try #require(panel.designLineage[minted.name])
            // Fact (a) is unchanged — the study did not move, and saying it
            // did would blame it for an edit made to the library.
            #expect(after.agreement == .matches)
            // Fact (b) is new, and independent.
            #expect(after.designRevised)

            let line = try #require(panel.templateLineage(minted))
            #expect(line.hasPrefix("from design '\(design.name)'"))
            #expect(line.contains("matches its design as minted"))
            #expect(line.contains("the design has since been revised"))
            #expect(!line.contains("diverged"))
        }
    }

    /// The two facts are genuinely independent: a study can have been edited
    /// AND have its design revised underneath it, and the line says both.
    @Test @MainActor func anEditedStudyUnderARevisedDesignSaysBoth() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            var minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            minted.maxTokens = 2048
            try ExperimentStore.save(minted)

            var revised = try StudyTemplateStore.load(name: design.name)
            revised.study.samplesPerItem = 9
            try StudyTemplateStore.save(revised)

            let panel = makePanel(root: root)
            let lineage = try #require(panel.designLineage[minted.name])
            #expect(lineage.agreement == .diverged)
            #expect(lineage.designRevised)

            let line = try #require(
                panel.templateLineage(try ExperimentStore.load(name: minted.name)))
            #expect(line.hasPrefix("diverged from design '\(design.name)'"))
            #expect(line.contains("the design has since been revised too"))
            #expect(line.contains("edits are allowed"))
        }
    }

    // MARK: - New from Study

    @Test @MainActor func newFromStudyIsSilentWhenTheStudyIsAlreadyThatDesign()
        throws
    {
        try withTempWorkspace { root in
            let design = try makeDesign()
            let minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            let panel = makePanel(root: root)
            panel.selectedTemplateName = nil
            let before = panel.status

            // An unchanged INSTANCE of a live design says nothing at all: the
            // library is on screen, and selecting the design that already
            // existed IS the answer.
            panel.newDesignFromStudy(named: minted.name)
            #expect(panel.status == before)
            #expect(panel.selectedTemplateName == design.name)
            #expect(StudyTemplateStore.list().count == 1)
        }
    }

    @Test @MainActor func newFromStudyNamesTheParentWhenTheStudyHadDiverged()
        throws
    {
        try withTempWorkspace { root in
            let design = try makeDesign()
            var minted = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            minted.maxTokens = 4096
            try ExperimentStore.save(minted)

            let panel = makePanel(root: root)
            panel.newDesignFromStudy(named: minted.name)
            let status = try #require(panel.status)
            #expect(status.hasPrefix("created design"))
            #expect(status.contains("had diverged from '\(design.name)'"))
            #expect(StudyTemplateStore.list().count == 2)
            #expect(panel.selectedTemplateName != design.name)
        }
    }

    // MARK: - The new-study design picker

    @Test func theDesignPickerOffersFromScratchFirst() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            let choices = StudyDesignChoice.choices(designs: [design])
            #expect(choices.first == .fromScratch)
            #expect(choices.last == .design(design.name))
            // Distinct tags: a design called "from-scratch" must not collide
            // with the blank path.
            #expect(Set(choices.map(\.id)).count == 2)
            #expect(StudyDesignChoice.fromScratch.designName == nil)
            #expect(StudyDesignChoice.design("x").label == "x")
        }
    }

    @Test func aDeletedDesignFallsBackToFromScratch() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            #expect(
                StudyDesignChoice.resolve(.design(design.name), designs: [design])
                    == .design(design.name))
            // A selection naming a design that no longer exists would mint
            // nothing and explain nothing.
            #expect(
                StudyDesignChoice.resolve(.design(design.name), designs: [])
                    == .fromScratch)
        }
    }

    @Test @MainActor func refreshResolvesAStaleDesignChoice() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            let panel = makePanel(root: root)
            panel.newStudyDesign = .design(design.name)
            panel.refresh()
            #expect(panel.newStudyDesign == .design(design.name))

            panel.deleteTemplate(design.name)
            #expect(panel.newStudyDesign == .fromScratch)
        }
    }

    // MARK: - The read-only design summary

    @Test func theDesignSummarySaysWhatTheResultWillMean() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            let rows = StudyDesignSummary.rows(for: design)
            let byLabel = Dictionary(
                uniqueKeysWithValues: rows.map { ($0.label, $0.value) })

            #expect(byLabel["Study type"] == StudyIntent.agentComparison.displayName)
            #expect(byLabel["Model"]?.contains("test/model") == true)
            // The revision is SHOWN, short — an unpinned one says so out loud.
            #expect(byLabel["Model"]?.contains("abc123def456") == true)
            let taskFile = try #require(byLabel["Task file"])
            #expect(taskFile.contains("prompts/tasks/cases.jsonl"))
            #expect(taskFile.contains("@"))  // the pin, not just the path
            #expect(byLabel["Sampling"]?.contains("4 sample(s)") == true)
            #expect(byLabel["Sampling"]?.contains("1024 tokens") == true)
            // Absence is said, never left blank.
            #expect(byLabel["Judging"] == "none")
            #expect(byLabel["Instruments"] == "sampled text (default)")
            #expect(byLabel["Instrument scope"] == "every item")
            #expect(byLabel["Parser"] == "none declared")
            #expect(byLabel["Exclusions"] == "none declared")
            #expect(byLabel["Pipeline"] == "none")
        }
    }

    @Test func anUnpinnedRevisionAndAbsentFilesAreSaidOutLoud() throws {
        try withTempWorkspace { _ in
            let manifest = try ExperimentStore.create(
                name: "bare", description: "", modelID: "test/model")
            let design = try StudyTemplateStore.templateFromStudy(
                experimentName: manifest.name
            ).template
            let byLabel = Dictionary(
                uniqueKeysWithValues: StudyDesignSummary.rows(for: design)
                    .map { ($0.label, $0.value) })
            #expect(byLabel["Model"]?.contains("revision unpinned") == true)
            #expect(byLabel["Task file"] == "none")
            #expect(byLabel["Capability battery"] == "none")
        }
    }

    /// A non-zero temperature is a SUBSTRATE fact, not just a number: the
    /// local MLX runner refuses it outright.
    @Test func aStochasticDesignSaysWhereItCanRun() throws {
        try withTempWorkspace { _ in
            var manifest = try ExperimentStore.create(
                name: "stochastic", description: "", modelID: "test/model")
            manifest.temperature = 0.7
            manifest.samplesPerItem = 8
            try ExperimentStore.save(manifest)
            let design = try StudyTemplateStore.templateFromStudy(
                experimentName: "stochastic"
            ).template
            let sampling = try #require(
                StudyDesignSummary.rows(for: design)
                    .first { $0.label == "Sampling" }?.value)
            #expect(sampling.contains("temperature 0.7"))
            #expect(sampling.contains("server substrate only"))
        }
    }

    // MARK: - Deleting a study

    /// Draft-only, and the REASON is on the control. The store refuses a
    /// frozen study with the immutability line because every run directory's
    /// stamp points at that manifest.
    @Test @MainActor func deleteIsOfferedForDraftsAndExplainedForTheRest() throws {
        try withTempWorkspace { root in
            try makeDesign(from: "vignette")
            let panel = makePanel(root: root)
            #expect(panel.deleteSelectedStudyRefusal == "select a study first")

            panel.selectedName = "vignette"
            #expect(panel.deleteSelectedStudyRefusal == nil)

            var manifest = try ExperimentStore.load(name: "vignette")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            panel.refresh()
            let refusal = try #require(panel.deleteSelectedStudyRefusal)
            #expect(refusal.contains("frozen"))
            #expect(refusal.contains("immutable"))
            #expect(refusal.contains("duplicate as a draft"))
            // And the store agrees — the UI is not inventing a rule.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.moveDraftToTrash(name: "vignette")
            }
        }
    }
}

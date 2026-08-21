import Foundation
import Testing

@testable import ExperimentKit

/// The design round trip: "Edit design…" mints a scratch draft, the ONE manifest
/// editor edits it, "Save back to design" writes it onto the design in place.
///
/// The properties worth pinning here are the ones that decide whether the loop
/// is honest rather than merely convenient. An in-place update must actually
/// move the design's content hash (otherwise the library silently keeps the old
/// recipe under the new name); it must refuse a design that does not exist
/// (otherwise a typo mints a new design that LOOKS like an edit); and — the one
/// that carries the science — it must not disturb what earlier-minted studies
/// say about themselves, because `agreement(of:)` compares against the hash
/// stamped at MINT time and a researcher reads that line as "is this study still
/// the replication it claims to be?".
///
/// Serialized for the usual reason: the suite moves the process-global
/// workspace root.
@Suite(.serialized) struct StudyDesignRoundTripTests {

    // MARK: - Fixtures

    func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "troundtrip-\(UUID().uuidString)")
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

    /// A hosted panel, for the one path that needs one: `ExperimentPanel.create`
    /// seeds the new draft's model from the host and returns silently without
    /// it. The SERVICE is returned, not just its panel — `host` is a weak
    /// reference, so a service that went out of scope would leave the panel
    /// hostless and `create` would do nothing.
    @MainActor
    private func makeHostedService(root: URL, suite: String) throws -> ChatService {
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        let service = ChatService(cluster: ClusterConnectionStore(defaults: defaults))
        service.experiments.notices = PanelNotices(
            fileURL: root.appending(component: "notices.jsonl"))
        service.experiments.refresh()
        // After the refresh: it re-syncs the draft fields from the (absent)
        // selection, so seeding the model first would be undone.
        service.experiments.studyBaseModelID = "test/model"
        return service
    }

    // MARK: - The store's update path

    /// `save` is a create-or-overwrite upsert; `update` is the deliberate
    /// in-place write. The distinction is the whole reason the round trip cannot
    /// mint a lookalike design from a stale name.
    @Test func updateRefusesANameThatIsNotInTheLibrary() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            var invented = design
            invented.name = "never-existed"
            #expect(throws: ExperimentError.self) {
                try StudyTemplateStore.update(invented)
            }
            // And nothing appeared: a refusal must not be a silent create.
            #expect(StudyTemplateStore.list().map(\.name) == [design.name])
        }
    }

    @Test func updateOverwritesAnExistingDesignInPlace() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            var revised = design
            revised.study.maxTokens = 4096
            try StudyTemplateStore.update(revised)

            #expect(StudyTemplateStore.list().count == 1)
            let reloaded = try StudyTemplateStore.load(name: design.name)
            #expect(reloaded.study.maxTokens == 4096)
            // The point of an in-place update: the CONTENT hash moves, the
            // library does not grow.
            #expect(StudyTemplateStore.hash(reloaded) != StudyTemplateStore.hash(design))
        }
    }

    // MARK: - Save a study back onto its design

    @Test func savingBackBumpsTheDesignHashWithoutAddingAnEntry() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            var draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            draft.maxTokens = 4096
            draft.samplesPerItem = 8
            try ExperimentStore.save(draft)

            let update = try StudyTemplateStore.saveStudyBackToDesign(
                experimentName: draft.name)
            #expect(update.design == design.name)
            #expect(update.changed)
            #expect(update.hashBefore == StudyTemplateStore.hash(design))

            let reloaded = try StudyTemplateStore.load(name: design.name)
            #expect(reloaded.study.maxTokens == 4096)
            #expect(reloaded.study.samplesPerItem == 8)
            #expect(StudyTemplateStore.hash(reloaded) == update.hashAfter)
            #expect(StudyTemplateStore.list().count == 1)
        }
    }

    /// Identity and the researcher's own note about a design belong to the
    /// design, not to whichever draft happened to be saved back onto it.
    @Test func savingBackKeepsTheDesignsNameDescriptionAndBirthDate() throws {
        try withTempWorkspace { _ in
            var design = try makeDesign()
            design.templateDescription = "wave 3, appendix table"
            try StudyTemplateStore.update(design)
            let createdAt = design.createdAt

            var draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            draft.experimentDescription = "scratch draft, ignore"
            draft.maxTokens = 4096
            try ExperimentStore.save(draft)
            try StudyTemplateStore.saveStudyBackToDesign(experimentName: draft.name)

            let reloaded = try StudyTemplateStore.load(name: design.name)
            #expect(reloaded.name == design.name)
            #expect(reloaded.templateDescription == "wave 3, appendix table")
            #expect(reloaded.createdAt == createdAt)
        }
    }

    /// An unedited round trip is legal and says so: the write happened, the
    /// recipe did not move.
    @Test func savingBackAnUneditedDraftReportsNoChange() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            let draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            let update = try StudyTemplateStore.saveStudyBackToDesign(
                experimentName: draft.name)
            #expect(!update.changed)
            #expect(update.hashBefore == update.hashAfter)
        }
    }

    @Test func savingBackRefusesAStudyWithNoLineage() throws {
        try withTempWorkspace { _ in
            try makeDesign()
            #expect(throws: ExperimentError.self) {
                try StudyTemplateStore.saveStudyBackToDesign(experimentName: "vignette")
            }
        }
    }

    @Test func savingBackRefusesWhenTheDesignIsGone() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            let draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            try StudyTemplateStore.delete(name: design.name)
            #expect(throws: ExperimentError.self) {
                try StudyTemplateStore.saveStudyBackToDesign(
                    experimentName: draft.name)
            }
        }
    }

    // MARK: - Studies minted EARLIER are untouched

    /// The honesty property. `agreement(of:)` compares a study's stripped form
    /// against `templateProvenance.templateHash` — the stamp written at MINT
    /// time — so revising a design in place cannot silently re-file an existing
    /// instance as matching or diverged. A researcher reading "from design 'x'"
    /// on last month's study is being told what it was minted from, which is
    /// still true.
    @Test func revisingADesignLeavesEarlierInstancesLineageUnchanged() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            let earlier = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            #expect(StudyTemplateStore.agreement(of: earlier) == .matches)
            let stampedAtMint = try #require(earlier.templateProvenance?.templateHash)

            var draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            draft.maxTokens = 4096
            try ExperimentStore.save(draft)
            try StudyTemplateStore.saveStudyBackToDesign(experimentName: draft.name)

            // The design moved…
            #expect(
                StudyTemplateStore.hash(
                    try StudyTemplateStore.load(name: design.name)) != stampedAtMint)
            // …and the earlier instance did not: same stamp, same reading.
            let reloaded = try ExperimentStore.load(name: earlier.name)
            #expect(reloaded.templateProvenance?.templateHash == stampedAtMint)
            #expect(StudyTemplateStore.agreement(of: reloaded) == .matches)
        }
    }

    /// And the draft that WAS saved back now matches the design it wrote — the
    /// round trip's own closing condition, reached without re-minting it.
    @Test func theEditDraftMatchesTheDesignItJustWrote() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            var draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            draft.maxTokens = 4096
            try ExperimentStore.save(draft)
            // Before the write it is an honestly diverged instance.
            #expect(
                StudyTemplateStore.agreement(
                    of: try ExperimentStore.load(name: draft.name)) == .diverged)

            let update = try StudyTemplateStore.saveStudyBackToDesign(
                experimentName: draft.name)
            // Its own stamp still records the PRE-edit hash, so the lineage line
            // keeps saying where it started. Divergence is the truth here: the
            // draft is not what it was minted from any more, and the design's
            // new hash is only reachable by re-minting.
            let saved = try ExperimentStore.load(name: draft.name)
            #expect(saved.templateProvenance?.templateHash == update.hashBefore)
            #expect(StudyTemplateStore.agreement(of: saved) == .diverged)

            // A study minted from the revised design agrees with it.
            let fresh = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            #expect(fresh.templateProvenance?.templateHash == update.hashAfter)
            #expect(StudyTemplateStore.agreement(of: fresh) == .matches)
        }
    }

    // MARK: - The edit draft itself

    @Test func theEditDraftIsNamedForItsDesignAndUniquified() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            let first = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            #expect(first.name == "\(design.name)-edit")
            let second = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            #expect(second.name == "\(design.name)-edit-2")
            #expect(
                StudyTemplateStore.editDraftName(templateName: design.name)
                    == "\(design.name)-edit-3")
        }
    }

    /// Agentless, stamped, and otherwise an ORDINARY draft — the contract that
    /// keeps this out of the measurement path.
    @Test func theEditDraftIsAnOrdinaryAgentlessDraft() throws {
        try withTempWorkspace { _ in
            let design = try makeDesign()
            let draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            #expect(draft.variantConditions.isEmpty)
            #expect(draft.status == .draft)
            #expect(draft.templateProvenance?.template == design.name)
            // Nothing about it is a batch: an edit draft is not a casting.
            #expect(draft.templateProvenance?.batchGroup == nil)
            // The design's settings came through.
            #expect(draft.maxTokens == 1024)
            #expect(draft.samplesPerItem == 4)
            #expect(draft.taskPromptsFile == "prompts/tasks/cases.jsonl")
            #expect(StudyTemplateStore.agreement(of: draft) == .matches)
        }
    }

    // MARK: - Panel logic on the panel, not in a view

    @Test @MainActor func editDesignSelectsTheDraftAndSaysWhatToDoNext() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            let panel = makePanel(root: root)
            let opened = try #require(panel.editDesign(design.name))
            #expect(opened == "\(design.name)-edit")
            #expect(panel.selectedName == opened)
            #expect(panel.experiments.contains { $0.name == opened })
            let status = try #require(panel.status)
            #expect(status.contains("Save back to design"))
            #expect(panel.formErrors[.template] == nil)
        }
    }

    @Test @MainActor func editingAMissingDesignRefusesAtTheControl() throws {
        try withTempWorkspace { root in
            try makeDesign()
            let panel = makePanel(root: root)
            #expect(panel.editDesign("no-such-design") == nil)
            #expect(panel.formErrors[.template] != nil)
        }
    }

    /// The two refusals are both about the DESIGN — it must exist, and this
    /// study must name it. Status is not one of them: a design carries no
    /// lifecycle stamps, so a frozen study's settings write onto a design
    /// exactly as a draft's do, and refusing only forced a duplicate-to-draft
    /// detour to produce the same bytes.
    @Test @MainActor func saveBackIsOfferedForAnyStatusOfALiveDesignsInstance()
        throws
    {
        try withTempWorkspace { root in
            let design = try makeDesign()
            let draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            let panel = makePanel(root: root)

            let minted = try ExperimentStore.load(name: draft.name)
            #expect(panel.saveBackToDesignTarget(for: minted) == design.name)
            #expect(panel.saveBackToDesignRefusal(for: minted) == nil)

            // No lineage: no target, and the refusal points at the sibling path.
            let source = try ExperimentStore.load(name: "vignette")
            #expect(panel.saveBackToDesignTarget(for: source) == nil)
            let noLineage = try #require(panel.saveBackToDesignRefusal(for: source))
            #expect(noLineage.contains("Save as new design"))

            // Frozen and complete are offered too.
            for status in [ExperimentManifest.Status.frozen, .complete] {
                var stamped = minted
                stamped.status = status
                try ExperimentStore.save(stamped)
                panel.refresh()
                let reloaded = try ExperimentStore.load(name: draft.name)
                #expect(reloaded.status == status)
                #expect(panel.saveBackToDesignTarget(for: reloaded) == design.name)
                #expect(panel.saveBackToDesignRefusal(for: reloaded) == nil)
            }
        }
    }

    /// …and the write itself goes through at a non-draft status, landing on the
    /// design and leaving the frozen study's own record alone.
    @Test @MainActor func saveBackFromAFrozenStudyWritesTheDesignAndNotTheStudy()
        throws
    {
        try withTempWorkspace { root in
            let design = try makeDesign()
            var draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            draft.maxTokens = 4096
            draft.status = .frozen
            draft.freezeHash = String(repeating: "a", count: 64)
            try ExperimentStore.save(draft)

            let panel = makePanel(root: root)
            panel.selectedName = draft.name
            panel.saveSelectedStudyBackToDesign()

            #expect(panel.formErrors[.template] == nil)
            #expect(panel.templates.count == 1)
            // The design took the settings…
            #expect(panel.templates.first?.study.maxTokens == 4096)
            // …and none of the lifecycle stamps: a design is never frozen.
            #expect(panel.templates.first?.study.status == .draft)
            #expect(panel.templates.first?.study.freezeHash == nil)
            // The frozen study is byte-for-byte what it was.
            let reloaded = try ExperimentStore.load(name: draft.name)
            #expect(reloaded.status == .frozen)
            #expect(reloaded.freezeHash == draft.freezeHash)
        }
    }

    @Test @MainActor func aDeletedDesignLeavesNoSaveBackTarget() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            let draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            try StudyTemplateStore.delete(name: design.name)
            let panel = makePanel(root: root)
            let minted = try ExperimentStore.load(name: draft.name)
            #expect(panel.saveBackToDesignTarget(for: minted) == nil)
            let refusal = try #require(panel.saveBackToDesignRefusal(for: minted))
            #expect(refusal.contains("no longer in the library"))
        }
    }

    @Test @MainActor func saveBackWritesInPlaceAndSaysWhatItDidToTheLibrary()
        throws
    {
        try withTempWorkspace { root in
            let design = try makeDesign()
            let earlier = try StudyTemplateStore.instantiate(
                templateName: design.name, cell: .agents([]))
            var draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            draft.maxTokens = 4096
            try ExperimentStore.save(draft)

            let panel = makePanel(root: root)
            panel.selectedName = draft.name
            panel.saveSelectedStudyBackToDesign()

            #expect(panel.formErrors[.template] == nil)
            #expect(panel.templates.count == 1)
            #expect(panel.templates.first?.study.maxTokens == 4096)
            #expect(panel.selectedTemplateName == design.name)
            let status = try #require(panel.status)
            #expect(status.hasPrefix("updated design '\(design.name)' in place"))
            #expect(status.contains("keep their original lineage stamps"))
            // And the claim the message makes is true: the earlier instance
            // still reads as matching the design it was minted from.
            let lineage = try #require(panel.designLineage[earlier.name])
            #expect(lineage.agreement == .matches)
            // The second, independent fact — without which "matches" would
            // quietly claim this study replicates the recipe now filed under
            // that name, which it does not.
            #expect(lineage.designRevised)
        }
    }

    @Test @MainActor func saveBackRefusesInsteadOfWritingWhenItCannotApply()
        throws
    {
        try withTempWorkspace { root in
            let design = try makeDesign()
            let panel = makePanel(root: root)
            panel.selectedName = "vignette"  // no lineage
            panel.saveSelectedStudyBackToDesign()
            #expect(panel.formErrors[.template] != nil)
            // Nothing was written: the design is byte-for-byte what it was.
            #expect(
                StudyTemplateStore.hash(
                    try StudyTemplateStore.load(name: design.name))
                    == StudyTemplateStore.hash(design))
        }
    }

    /// "New Template" is the ordinary from-scratch creation path with a
    /// navigation on top — NOT a second creator, and not a draft pre-stamped
    /// with a design that does not exist (which would make the lineage line
    /// report "no longer in the library" about a design nobody ever saved).
    @Test @MainActor func newDesignDraftIsAPlainDraftWithNoInventedLineage()
        throws
    {
        try withTempWorkspace { root in
            let service = try makeHostedService(
                root: root, suite: "steerlab.tests.design-round-trip")
            let panel = service.experiments
            let name = try #require(panel.newDesignDraft())
            let draft = try ExperimentStore.load(name: name)
            #expect(draft.status == .draft)
            #expect(draft.templateProvenance == nil)
            #expect(StudyTemplateStore.agreement(of: draft) == .noLineage)
            #expect(panel.templateLineage(draft) == nil)
            // The rename invitation is what the Studies tab consumes on appear.
            #expect(panel.renameInvitation == name)
        }
    }

    /// The two write-back paths are siblings, and they do different things:
    /// one grows the library, the other does not.
    @Test @MainActor func saveAsNewDesignAddsAnEntryWhereSaveBackDoesNot() throws {
        try withTempWorkspace { root in
            let design = try makeDesign()
            var draft = try StudyTemplateStore.mintEditDraft(
                templateName: design.name)
            draft.maxTokens = 4096
            try ExperimentStore.save(draft)

            let panel = makePanel(root: root)
            panel.newDesignFromStudy(named: draft.name)
            #expect(panel.templates.count == 2)
            // The original design is untouched by the minting path.
            #expect(
                StudyTemplateStore.hash(
                    try StudyTemplateStore.load(name: design.name))
                    == StudyTemplateStore.hash(design))
        }
    }
}

import Foundation
import Observation
import Synchronization
import Testing

@testable import ExperimentKit

@MainActor
struct StudyManagementOwnershipTests {
    private func withWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory.appending(
            component: "management-\(UUID())")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        return try body(root)
    }

    @Test func standaloneControllerCreatesWithSuppliedWorkspaceChoices() throws {
        try withWorkspace { _ in
            let draft = StudyDraftState()
            let owner = StudyManagementController(draft: draft)
            var messages: [String] = []
            owner.presentation.note = { message, _ in messages.append(message) }
            draft.studyBaseModelID = "local/unavailable"
            draft.newName = "explicit"
            draft.newRevision = "  abc123  "
            owner.create(context: nil)
            #expect(owner.experiments.isEmpty)
            #expect(draft.newName == "explicit")

            owner.create(
                context: .init(
                    workspaceDefaultModelID: "server/model", modelOptions: ["server/model"]))
            let created = try ExperimentStore.load(name: "explicit")
            #expect(created.modelID == "server/model")
            #expect(created.modelRevision == "abc123")
            #expect(created.temperature == 0)
            #expect(created.maxTokens == 2048)
            #expect(owner.selectedName == created.name)
            #expect(draft.newName.isEmpty)
            #expect(messages.contains { $0.contains("not available in this workspace") })

            draft.studyBaseModelID = "server/alternate"
            owner.newStudy(
                context: .init(
                    workspaceDefaultModelID: "server/model",
                    modelOptions: ["server/model", "server/alternate"]))
            let invited = try #require(owner.renameInvitation)
            #expect(owner.selectedName == invited)
            #expect(try ExperimentStore.load(name: invited).modelID == "server/alternate")
        }
    }

    @Test func standaloneCommandsPreserveFrozenIdentityAndDuplicateIntoAnEditableDraft() throws {
        try withWorkspace { _ in
            var original = try ExperimentStore.create(
                name: "original", description: "purpose", modelID: "test/model")
            original.status = .frozen
            original.freezeHash = String(repeating: "a", count: 64)
            try ExperimentStore.save(original)
            let owner = StudyManagementController(draft: StudyDraftState())
            owner.refresh()
            owner.selectedName = original.name
            owner.rename(reviewed: try owner.reviewStudy(named: original.name), canonicalName: "forbidden", label: nil)
            #expect(owner.draft.formErrors[.rename] != nil)
            #expect(owner.selectedName == "original")
            owner.rename(reviewed: try owner.reviewStudy(named: original.name), canonicalName: nil, label: "Readable label")
            #expect(owner.displayName(try #require(owner.selected)) == "Readable label")
            #expect(owner.draft.formErrors[.rename] == nil)
            owner.deleteDraft(reviewed: try owner.reviewStudy(named: try #require(owner.selectedName)))
            #expect(try ExperimentStore.load(name: "original").freezeHash == original.freezeHash)
            #expect(owner.deleteSelectedStudyRefusal != nil)

            owner.duplicateSelected()
            let copy = try #require(owner.selected)
            #expect(copy.name == "original-2")
            #expect(copy.status == .draft)
            #expect(copy.freezeHash == nil)
            owner.deleteDraft(reviewed: try owner.reviewStudy(named: try #require(owner.selectedName)))
            #expect(owner.selectedName == nil)
            #expect(owner.experiments.map(\.name) == ["original"])
        }
    }

    @Test func designOperationsWorkWithoutAPanelAndRefreshTheirLibrary() throws {
        try withWorkspace { root in
            _ = try ExperimentStore.create(
                name: "source", description: "purpose", modelID: "test/model")
            let owner = StudyManagementController(draft: StudyDraftState())
            owner.refresh()
            owner.newDesignFromStudy(reviewedSource: try owner.reviewDesignSource(named: "source"))
            let name = try #require(owner.designs.selectedTemplateName)
            // Dedup is lineage-based: an unchanged instance returns its design.
            // Exporting an unlinked source again intentionally mints a new one.
            let unchangedInstance = try #require(owner.editDesign(name))
            owner.newDesignFromStudy(reviewedSource: try owner.reviewDesignSource(named: unchangedInstance))
            #expect(owner.designs.templates.count == 1)
            owner.draft.newName = "unrelated-create-field"
            owner.renameTemplate(name, to: "renamed-design")
            #expect(owner.designs.selectedTemplateName == "renamed-design")
            owner.updateTemplateDescription(reviewed: try StudyDesignSnapshot(workspaceRoot: root, name: "renamed-design"), to: "Updated description")
            #expect(owner.designs.selectedTemplate?.templateDescription == "Updated description")
            let editName = try #require(owner.editDesign("renamed-design"))
            var edit = try ExperimentStore.load(name: editName)
            edit.maxTokens = 333
            try ExperimentStore.save(edit)
            owner.refresh()
            owner.updateDesign(reviewedSource: try owner.reviewDesignSource(named: editName),
                reviewedDesign: try owner.designs.reviewedDesign(named: "renamed-design"))
            #expect(owner.designs.selectedTemplate?.study.maxTokens == 333)
            #expect(owner.designs.templates.count == 1)
            owner.designs.newStudyDesign = .design("renamed-design")
            owner.deleteTemplate("renamed-design")
            #expect(owner.designs.templates.isEmpty)
            #expect(owner.designs.newStudyDesign == .fromScratch)
            #expect(try ExperimentStore.load(name: editName).maxTokens == 333)
        }
    }

    @Test func designDescriptionCannotFollowAWorkspaceSwitch() throws {
        try withWorkspace { root in
            let study = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            try StudyTemplateStore.save(StudyTemplate(name: "design", study: study))
            let reviewed = try StudyDesignSnapshot(workspaceRoot: root, name: "design")
            let owner = StudyManagementController(draft: StudyDraftState())
            WorkspaceRoot.programmaticOverride = root.appending(component: "other")
            #expect(owner.updateTemplateDescription(reviewed: reviewed, to: "Changed") == nil)
            #expect(owner.draft.formErrors[.template]?.contains("another workspace") == true)
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: "design").file.data == reviewed.file.data)
        }
    }

    @Test func selectionOwnerStillSynchronizesPanelWithoutResettingLiveJobs() throws {
        try withWorkspace { root in
            _ = try ExperimentStore.create(name: "one", description: "First", modelID: "test/model")
            _ = try ExperimentStore.create(
                name: "two", description: "Second", modelID: "test/model")
            let panel = ExperimentPanel()
            panel.notices = PanelNotices(fileURL: root.appending(component: "notices.jsonl"))
            panel.management.selectedName = "one"
            panel.draft.protocolDescription = "unsaved"
            panel.refresh()
            #expect(panel.draft.protocolDescription == "unsaved")
            panel.localJobs.isRunning = true
            panel.localJobs.handleStudyProgress(
                .generationStarted(condition: "baseline", promptID: "p", prompt: "question"))
            panel.localJobs.handleStudyProgress(
                .generationChunk(condition: "baseline", promptID: "p", output: "partial"))
            panel.results.selectedResultID = "old-result"
            let changed = Mutex(false)
            withObservationTracking {
                _ = panel.management.selectedName
            } onChange: {
                changed.withLock { $0 = true }
            }
            panel.management.selectedName = "two"
            #expect(changed.withLock { $0 })
            #expect(panel.management.selected?.name == "two")
            #expect(panel.draft.protocolDescription == "Second")
            #expect(panel.results.selectedResultID != "old-result")
            #expect(panel.localJobs.isRunning)
            #expect(panel.localJobs.liveActiveGeneration?.output == "partial")
        }
    }

    @Test func presentationCallbacksDoNotKeepThePanelAlive() {
        withWorkspace { _ in
            var panel: ExperimentPanel? = ExperimentPanel()
            weak var weakPanel = panel
            let owner = panel!.management
            panel = nil
            #expect(weakPanel == nil)
            owner.refresh()
            owner.selectedName = "absent"
        }
    }
}

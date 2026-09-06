import Foundation
import Testing
@testable import ExperimentKit

@MainActor @Suite(.serialized)
struct StudyAssemblySafetyTests {
    @Test func catalogRefreshCannotAuthorizeAStaleConditionCommand() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-stale") { root in
            _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let panel = ExperimentPanel()
            panel.management.selectedName = "study"
            try ExperimentStore.updateDraft(name: "study") { $0.experimentDescription = "external edit" }
            let external = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            panel.refresh()
            panel.draft.conditionConcept = "unattached"
            panel.draft.conditionLayerText = "1"
            panel.draft.conditionAlphaText = "1"
            panel.addVectorCondition()
            #expect(panel.status?.contains("changed") == true)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == external.file.data)
        }
    }

    @Test func successfulFormCommandsAdvanceOnlyTheirOwnReview() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-review") { _ in
            _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let panel = ExperimentPanel()
            panel.management.selectedName = "study"
            panel.setStudyType(.agentComparison)
            panel.setStudyType(.conceptStudy)
            #expect(try ExperimentStore.load(name: "study").studyType == StudyIntent.conceptStudy.rawValue)
            #expect(!panel.management.selectedDraftNeedsReload)
        }
    }

    @Test func staleSweepReviewCannotChangeGridOrSelection() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-sweep") { root in
            _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            try ExperimentStore.updateDraft(name: "study") { $0.maxTokens = 128 }
            let before = try Data(contentsOf: ExperimentStore.manifestURL("study"))
            let panel = ExperimentPanel()
            #expect(!panel.setSweepSpec(.init(), reviewed: reviewed))
            #expect(try Data(contentsOf: ExperimentStore.manifestURL("study")) == before)
        }
    }

    @Test func gridAndCriterionPublishAsOneDraftWrite() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-sweep-atomic") { _ in
            _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let before = try Data(contentsOf: ExperimentStore.manifestURL("study"))
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setSweepGrid(experimentName: "study", alphas: [0.2, 0.4],
                    selectionUpdate: .some(.init(objective: .init(metric: "judgeScore"))))
            }
            #expect(try Data(contentsOf: ExperimentStore.manifestURL("study")) == before)
        }
    }

    @Test func renameAndDeleteUseReviewedTargetAfterSelectionChanges() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-target") { root in
            _ = try ExperimentStore.create(name: "first", description: "", modelID: "test/model")
            _ = try ExperimentStore.create(name: "second", description: "", modelID: "test/model")
            let panel = ExperimentPanel()
            panel.management.selectedName = "first"
            let reviewed = try panel.management.reviewStudy(named: "first")
            panel.management.selectedName = "second"
            #expect(panel.management.rename(reviewed: reviewed, canonicalName: "renamed", label: nil))
            #expect(panel.management.selectedName == "second")
            let renamed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "renamed")
            panel.management.deleteDraft(reviewed: renamed)
            #expect(panel.management.selectedName == "second")
            #expect(try ExperimentStore.load(name: "second").name == "second")
            #expect(!FileManager.default.fileExists(atPath: ExperimentStore.manifestURL("renamed").path))
        }
    }

    @Test func staleRenameDoesNotMoveTheDirectory() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-rename") { root in
            _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            try ExperimentStore.updateDraft(name: "study") { $0.maxTokens = 128 }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.rename(experimentName: "study", to: "other", reviewed: reviewed)
            }
            #expect(try ExperimentStore.load(name: "study").maxTokens == 128)
            #expect(!FileManager.default.fileExists(atPath: ExperimentStore.manifestURL("other").path))
        }
    }

    @Test func recordImportsPreservePreviousInputsAndUnmodeledFields() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-input") { root in
            _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            let first = try TaskPromptsAuthoring.importJSONL(
                #"{"id":"item","text":"first","options":["yes","no"],"target":"yes","extra":{"value":7}}"#, reviewed: reviewed)
            let second = try TaskPromptsAuthoring.importJSONL(#"{"id":"item","text":"second"}"#, reviewed: first.study)
            #expect(first.prompts.path != second.prompts.path)
            #expect(try Data(contentsOf: root.appending(path: first.prompts.path)) == first.prompts.file.data)
            let record = try #require(JSONSerialization.jsonObject(with: first.prompts.file.data) as? [String: Any])
            #expect((record["extra"] as? [String: Int])?["value"] == 7)
            #expect(first.study.manifest.taskPromptsHash == first.prompts.file.sha256)
            #expect(second.study.manifest.taskPromptsHash == second.prompts.file.sha256)
        }
    }

    @Test(arguments: ["", "  \n", "{\"text\":\"valid\"}\ninvalid"])
    func invalidRecordsRefuseBeforePreparingFiles(text: String) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-input-invalid") { root in
            _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            #expect(throws: (any Error).self) { try TaskPromptsAuthoring.importJSONL(text, reviewed: reviewed) }
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "prompts/tasks/versions").path))
        }
    }

    @Test func tableImportCreatesVersionsAndRefusesRetargetedDialog() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-table") { root in
            _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            _ = try ExperimentStore.create(name: "other", description: "", modelID: "test/model")
            let panel = ExperimentPanel()
            panel.management.selectedName = "study"
            let first = try panel.management.reviewStudy(named: "study")
            let table = TabularImport.Table(columns: ["prompt"], rows: [["prompt": .string("First task")]])
            #expect(panel.importTaskPromptsTable(table: table, mapping: ["text": "prompt"], reviewed: first) == nil)
            let saved = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            let path = try #require(saved.manifest.taskPromptsFile)
            let bytes = try Data(contentsOf: root.appending(path: path))
            let changed = TabularImport.Table(columns: ["prompt"], rows: [["prompt": .string("Second task")]])
            #expect(panel.importTaskPromptsTable(table: changed, mapping: ["text": "prompt"], reviewed: saved) == nil)
            #expect(try Data(contentsOf: root.appending(path: path)) == bytes)
            let dialog = try panel.management.reviewStudy(named: "study")
            panel.management.selectedName = "other"
            #expect(panel.importTaskPromptsTable(table: table, mapping: ["text": "prompt"], reviewed: dialog) != nil)
            #expect(try ExperimentStore.load(name: "other").taskPromptsFile == nil)
            #expect(try Data(contentsOf: ExperimentStore.manifestURL("study")) == dialog.file.data)
        }
    }

    @Test func staleTableImportPublishesNoFile() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-table-stale") { root in
            _ = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let panel = ExperimentPanel()
            panel.management.selectedName = "study"
            let reviewed = try panel.management.reviewStudy(named: "study")
            try ExperimentStore.updateDraft(name: "study") { $0.maxTokens = 128 }
            let table = TabularImport.Table(columns: ["prompt"], rows: [["prompt": .string("Task")]])
            #expect(panel.importTaskPromptsTable(table: table, mapping: ["text": "prompt"], reviewed: reviewed) != nil)
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "prompts/tasks/versions").path))
        }
    }

    @Test func staleOrFrozenRecordImportPreparesNoVersion() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "assembly-input-stale") { root in
            var manifest = try ExperimentStore.create(name: "study", description: "", modelID: "test/model")
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            for source in [reviewed, try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")] {
                #expect(throws: ExperimentError.self) {
                    try TaskPromptsAuthoring.importJSONL(#"{"text":"new"}"#, reviewed: source)
                }
            }
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "prompts/tasks/versions").path))
        }
    }
}

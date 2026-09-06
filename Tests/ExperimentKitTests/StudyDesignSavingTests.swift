import Foundation
import Testing

@testable import ExperimentKit

struct StudyDesignSavingTests {
    private func fixture(_ body: (DraftAuthoringSnapshot) throws -> Void) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "design-save") { root in
            let previousWorkspace = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previousWorkspace }
            _ = try ExperimentStore.create(name: "source", description: "Reusable task", modelID: "test/model")
            try body(DraftAuthoringSnapshot(workspaceRoot: root, name: "source"))
        }
    }

    @Test func designPublicationPreservesSourceAndUsesCapturedWorkspace() throws {
        try fixture { study in
            let source = try StudyDesignSourceReview(study: study)
            let other = study.workspaceRoot.appending(component: "other")
            WorkspaceRoot.programmaticOverride = other
            ExperimentStore.rootOverride = other
            #expect(ExperimentStore.workspaceRoot == other)
            let first = try StudyDesignSaving.create(from: source, name: "new-design")
            let second = try StudyDesignSaving.create(from: source, name: "new-design")
            #expect(first.created && second.created)
            #expect(first.snapshot.template.name == "new-design")
            #expect(second.snapshot.template.name == "new-design-2")
            #expect(first.snapshot.template.study == StudyTemplateStore.strippedBody(study.manifest))
            #expect(try DraftAuthoringSnapshot(workspaceRoot: study.workspaceRoot, name: study.manifest.name).file.data == study.file.data)
            #expect(!FileManager.default.fileExists(atPath: other.path))
        }
    }

    @Test func updateRequiresBothReviewsAndLeavesSourceAndPriorStudiesUntouched() throws {
        try fixture { initial in
            let root = initial.workspaceRoot
            let design = try StudyDesignSaving.create(from: StudyDesignSourceReview(study: initial), name: "design").snapshot
            let instance = try StudyDesignInstantiation.instantiate(reviewed: design, casting: .agents([]), studyName: "instance")
            let reusable = try StudyDesignSaving.create(from: StudyDesignSourceReview(study: instance))
            #expect(!reusable.created && !reusable.changed)
            #expect(reusable.snapshot.file.data == design.file.data)
            var replacement = instance.manifest
            replacement.temperature = 0.75
            let edited = try DraftAuthoringTransaction.replace(replacement, reviewed: instance)
            let source = try StudyDesignSourceReview(study: edited)
            let result = try StudyDesignSaving.update(from: source, reviewed: design)
            #expect(result.changed && !result.created)
            #expect(result.snapshot.template.study.temperature == 0.75)
            #expect(result.snapshot.template.createdAt == design.template.createdAt)
            #expect(result.snapshot.template.templateDescription == design.template.templateDescription)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: edited.manifest.name).file.data == edited.file.data)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: initial.manifest.name).file.data == initial.file.data)
            #expect(throws: StudyDesignAuthoringError.self) { try StudyDesignSaving.update(from: source, reviewed: design) }
            replacement.maxTokens += 1
            _ = try DraftAuthoringTransaction.replace(replacement, reviewed: edited)
            #expect(throws: ExperimentError.self) { try StudyDesignSaving.update(from: source, reviewed: result.snapshot) }
            #expect(throws: ExperimentError.self) { try StudyDesignSaving.create(from: source, name: "refused") }
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: "design").file.data == result.snapshot.file.data)
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "templates/refused").path))
        }
    }

    @Test func frozenSourceIsReadOnlyAndLineageMismatchDoesNotGrantUpdateAuthority() throws {
        try fixture { study in
            let root = study.workspaceRoot
            let design = try StudyDesignSaving.create(from: StudyDesignSourceReview(study: study), name: "design").snapshot
            // A disposable frozen-state fixture, never an edit to research evidence.
            var frozen = study.manifest
            frozen.status = .frozen
            let url = ExperimentRepository(workspaceRoot: root).manifestURL(frozen.name)
            try JSONEncoder().encode(frozen).write(to: url)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: frozen.name)
            let source = try StudyDesignSourceReview(study: reviewed)
            let saved = try StudyDesignSaving.create(from: source, name: "from-frozen")
            #expect(saved.snapshot.template.study.status == .draft)
            #expect(try Data(contentsOf: url) == reviewed.file.data)
            #expect(throws: StudyDesignAuthoringError.self) { try StudyDesignSaving.update(from: source, reviewed: design) }
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: design.template.name).file.data == design.file.data)
        }
    }

    @Test func changedPanelInputCannotBeRepinnedIntoANewDesign() throws {
        try fixture { study in
            let root = study.workspaceRoot
            let panel = MultiAgentScenario(name: "panel", baseModelID: "test/model", sharedMaterials: "Generic task",
                agents: [.init(id: "speaker", name: "Speaker", baseModelID: "test/model", systemPrompt: "Speak.")],
                turns: [.init(id: "turn", title: "Turn", speakerAgentID: "speaker", promptTemplate: "Respond.", outputLabel: "answer")])
            let url = root.appending(path: "prompts/panels/source.json")
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(panel)
            try data.write(to: url)
            var manifest = study.manifest
            manifest.studyKind = .multiAgent
            manifest.multiAgentScenarioPath = "prompts/panels/source.json"
            manifest.multiAgentScenarioHash = MultiAgentScenarioStore.hash(data)
            let reviewed = try DraftAuthoringTransaction.replace(manifest, reviewed: study)
            let source = try StudyDesignSourceReview(study: reviewed)
            var changed = panel
            changed.sharedMaterials = "Changed task"
            try JSONEncoder().encode(changed).write(to: url)
            #expect(throws: ExperimentError.self) { try StudyDesignSaving.create(from: source, name: "refused") }
            #expect(throws: ExperimentError.self) { try StudyDesignSourceReview(study: reviewed) }
            #expect(!FileManager.default.fileExists(atPath: root.appending(component: "templates").path))
        }
    }
    @Test func aBoundSourceIsNeverReusedAsTheDesignsSemanticPanel() throws {
        try fixture { study in
            let root = study.workspaceRoot
            let bound = MultiAgentScenario(name: "panel", baseModelID: "test/model", sharedMaterials: "Generic task",
                agents: [.init(id: "speaker", name: "Speaker", baseModelID: "test/model", systemPrompt: "Speak.")],
                turns: [.init(id: "turn", title: "Turn", speakerAgentID: "speaker", promptTemplate: "Respond.", outputLabel: "answer")])
            let url = root.appending(path: "prompts/panels/bound.json")
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(bound)
            try data.write(to: url)
            var manifest = study.manifest
            manifest.studyKind = .multiAgent
            manifest.multiAgentScenarioPath = "prompts/panels/bound.json"
            manifest.multiAgentScenarioHash = MultiAgentScenarioStore.hash(data)
            manifest.multiAgentSemanticScenarioPath = manifest.multiAgentScenarioPath
            let reviewed = try DraftAuthoringTransaction.replace(manifest, reviewed: study)
            let saved = try StudyDesignSaving.create(from: StudyDesignSourceReview(study: reviewed), name: "design")
            let ref = try #require(saved.snapshot.template.semanticScenario)
            let semantic = try StudyTemplateStore.loadSemanticPanel(ref, workspaceRoot: root)
            #expect(semantic == PanelComposition.semanticForm(bound))
            #expect(ref.path != manifest.multiAgentScenarioPath)
            #expect(try Data(contentsOf: url) == data)
        }
    }

    @Test func publicAdaptersSaveAndUpdateWithExplicitSourceAndDestinationReviews() throws {
        try fixture { study in
            let root = study.workspaceRoot
            func cli(_ args: [String]) throws -> ExperimentCLIResult {
                try StudyDesignCLI.run(ExperimentCLIParser.parse(namespace: "design", args + ["--json"]), workspaceRoot: root, sink: .discarding)
            }
            let result = try cli(["save", "source", "--name", "design", "--manifest-sha256", study.file.sha256])
            #expect(result.changed)
            let design = try StudyDesignSnapshot(workspaceRoot: root, name: "design")
            let instance = try StudyDesignInstantiation.instantiate(reviewed: design, casting: .agents([]), studyName: "instance")
            var manifest = instance.manifest
            manifest.temperature = 0.6
            let edited = try DraftAuthoringTransaction.replace(manifest, reviewed: instance)
            var fields: [String: Any] = ["workspaceRoot": root.path, "name": "design", "sourceStudy": "instance"]
            func send() throws -> StudyAuthoringHTTP.Response {
                StudyDesignHTTP.perform(.update, body: try JSONSerialization.data(withJSONObject: fields), workspaceRoot: root)
            }
            #expect(try send().status == "428 Precondition Required")
            fields["manifestFileSHA256"] = edited.file.sha256
            #expect(try send().status == "428 Precondition Required")
            fields["designFileSHA256"] = design.file.sha256
            fields["workspaceRoot"] = root.appending(component: "other").path
            #expect(try send().status == "409 Conflict")
            fields["workspaceRoot"] = root.path
            fields["force"] = true
            #expect(try send().status == "400 Bad Request")
            fields.removeValue(forKey: "force")
            #expect(try send().status == "200 OK")
            let saved = try StudyDesignSnapshot(workspaceRoot: root, name: "design")
            #expect(saved.template.study.temperature == 0.6)
            #expect(try send().status == "412 Precondition Failed")
            let noop = try cli(["update", "design", "--study", "instance", "--manifest-sha256", edited.file.sha256, "--file-sha256", saved.file.sha256])
            #expect(!noop.changed)
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: "design").file.data == saved.file.data)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "instance").file.data == edited.file.data)
            let reused = StudyDesignHTTP.perform(.save, body: try JSONSerialization.data(withJSONObject: [
                "workspaceRoot": root.path, "sourceStudy": "instance", "manifestFileSHA256": edited.file.sha256]), workspaceRoot: root)
            #expect(reused.status == "200 OK")
            let json = try JSONDecoder().decode([String: JSONValue].self, from: reused.body)
            #expect(json["created"] == .bool(false))
            #expect(json["design"] == noop.payload["design"])
        }
    }

    @Test @MainActor func confirmationReviewsSurviveCatalogRefreshAndRefuseWorkspaceChanges() throws {
        try fixture { initial in
            let root = initial.workspaceRoot
            let design = try StudyDesignSaving.create(from: StudyDesignSourceReview(study: initial), name: "design").snapshot
            let instance = try StudyDesignInstantiation.instantiate(reviewed: design, casting: .agents([]), studyName: "instance")
            let owner = StudyManagementController(draft: StudyDraftState())
            owner.refresh()
            owner.selectedName = instance.manifest.name
            owner.beginAuthoringReview(named: instance.manifest.name)
            let source = try owner.reviewEditorDesignSource()
            let destination = try owner.designs.reviewedDesign(named: "design")
            let changedDesign = try StudyDesignAuthoring.updateDescription("Another editor's note", reviewed: destination)
            owner.refresh()
            owner.updateDesign(reviewedSource: source, reviewedDesign: destination)
            #expect(owner.draft.formErrors[.template] != nil)
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: "design").file.data == changedDesign.file.data)
            var changedStudy = instance.manifest
            changedStudy.temperature = 0.8
            _ = try DraftAuthoringTransaction.replace(changedStudy, reviewed: instance)
            owner.refresh()
            #expect(try owner.reviewEditorDesignSource().study.file.sha256 == source.study.file.sha256)
            owner.updateDesign(reviewedSource: source, reviewedDesign: changedDesign)
            #expect(owner.draft.formErrors[.template] != nil)
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: "design").file.data == changedDesign.file.data)
            let librarySource = try owner.reviewDesignSource(named: instance.manifest.name)
            let other = root.appending(component: "other")
            WorkspaceRoot.programmaticOverride = other
            ExperimentStore.rootOverride = other
            #expect(ExperimentStore.workspaceRoot == other)
            owner.updateDesign(reviewedSource: librarySource, reviewedDesign: changedDesign)
            #expect(owner.draft.formErrors[.template]?.contains("workspace changed") == true)
            #expect(!FileManager.default.fileExists(atPath: root.appending(component: "other").path))
        }
    }

}

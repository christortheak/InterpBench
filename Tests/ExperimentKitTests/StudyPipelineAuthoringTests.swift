import Foundation
import Testing

@testable import ExperimentKit

struct StudyPipelineAuthoringTests {
    private func withReview(_ body: (DraftAuthoringSnapshot) throws -> Void) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "pipeline-authoring") { root in
            let manifest = try ExperimentStore.create(name: "study", description: "Original description", modelID: "test/model")
            try body(DraftAuthoringSnapshot(workspaceRoot: root, name: manifest.name))
        }
    }

    @Test func savesAndRemovesOnlyTheReviewedPipelineDeclaration() throws {
        try withReview { reviewed in
            let saved = try StudyPipelineAuthoring.save(PipelineDraft(stages: ["run"]), reviewed: reviewed)
            #expect(saved.manifest.pipeline == PipelineDraft(stages: ["run"]).encoded())
            #expect(saved.manifest.experimentDescription == reviewed.manifest.experimentDescription)
            #expect(saved.file.sha256 != reviewed.file.sha256)
            let removed = try StudyPipelineAuthoring.save(nil, reviewed: saved)
            #expect(removed.manifest.pipeline == nil)
            #expect(removed.manifest.modelID == reviewed.manifest.modelID)
        }
    }

    @Test(arguments: [false, true])
    func concurrentEditsOrFreezeRefuseWithoutReplacingSavedBytes(freeze: Bool) throws {
        try withReview { reviewed in
            var changed = reviewed.manifest
            changed.experimentDescription = "Another author's edit"
            if freeze { changed.status = .frozen }
            let url = ExperimentRepository(workspaceRoot: reviewed.workspaceRoot).manifestURL(changed.name)
            // Disposable fixture models a concurrent writer or completed freeze.
            let bytes = try JSONEncoder().encode(changed)
            try bytes.write(to: url)
            do {
                _ = try StudyPipelineAuthoring.save(PipelineDraft(stages: ["run"]), reviewed: reviewed)
                Issue.record("A stale composer must not replace a later edit or freeze")
            } catch let error as ExperimentError {
                #expect(error.lifecycleRefusal?.gate == .staleManifest)
            }
            #expect(try Data(contentsOf: url) == bytes)
            if freeze {
                let current = try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: changed.name)
                #expect(throws: (any Error).self) { try StudyPipelineAuthoring.save(nil, reviewed: current) }
                #expect(try Data(contentsOf: url) == bytes)
            }
        }
    }

    @Test(arguments: ["missingMinimum", "evaluateWithoutRun"])
    func headlessCallsValidateBeforeWriting(invalid: String) throws {
        try withReview { reviewed in
            var draft = PipelineDraft(stages: invalid == "evaluateWithoutRun" ? ["evaluate"] : ["validate", "run"])
            if invalid == "missingMinimum" { draft.accuracyFloorMetric = "auc"; draft.accuracyFloorMinimum = nil }
            #expect(throws: ExperimentError.self) { try StudyPipelineAuthoring.save(draft, reviewed: reviewed) }
            let current = try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: reviewed.manifest.name)
            #expect(current.file.data == reviewed.file.data)
        }
    }

    @Test func explicitServiceKeepsTheCapturedWorkspace() throws {
        try withReview { reviewed in
            let other = reviewed.workspaceRoot.appending(component: "other")
            ExperimentStore.rootOverride = other
            defer { ExperimentStore.rootOverride = reviewed.workspaceRoot }
            let second = try ExperimentStore.create(name: "study", description: "Other workspace", modelID: "test/model")
            let otherBefore = try DraftAuthoringSnapshot(workspaceRoot: other, name: second.name)
            _ = try StudyPipelineAuthoring.save(PipelineDraft(stages: ["run"]), reviewed: reviewed)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: other, name: second.name).file.data == otherBefore.file.data)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: second.name).manifest.pipeline != nil)
        }
    }
}

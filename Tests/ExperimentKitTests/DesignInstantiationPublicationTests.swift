import Foundation
import Testing

@testable import ExperimentKit

struct DesignInstantiationPublicationTests {
    private func fixture(_ body: (URL, StudyTemplate) throws -> Void) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "design-publication") { root in
            var study = try ExperimentStore.create(name: "source", description: "", modelID: "test/model")
            let relative = "prompts/tasks/items.jsonl"
            let file = root.appending(path: relative)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = Data("{\"id\":\"item-1\",\"text\":\"Choose.\",\"responseFormat\":\"label\",\"options\":[\"A\",\"B\"]}\n".utf8)
            try data.write(to: file)
            study.taskPromptsFile = relative
            study.taskPromptsHash = ExperimentStore.sha256Hex(data)
            study.outcomeInstrumentScope = .init(responseFormats: ["label"], itemCount: 99, itemIDsHash: "old")
            let template = StudyTemplate(name: "design", study: StudyTemplateStore.strippedBody(study))
            try StudyTemplateStore.save(template)
            try body(root, template)
        }
    }

    @Test(arguments: [["not-a-format"], ["freeText"]])
    func failedScopeDerivationPublishesNoDraft(formats: [String]) throws {
        try fixture { root, original in
            var template = original
            template.study.outcomeInstrumentScope?.responseFormats = formats
            try StudyTemplateStore.save(template)
            let before = try StudyDesignSnapshot(workspaceRoot: root, name: template.name)
            let source = try DraftAuthoringSnapshot(workspaceRoot: root, name: "source")
            #expect(throws: (any Error).self) {
                try StudyTemplateStore.instantiate(templateName: template.name, cell: .agents([]), studyName: "new-study")
            }
            #expect(!FileManager.default.fileExists(atPath: root.appending(components: "experiments", "new-study").path))
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "source").file.data == source.file.data)
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: template.name).file.data == before.file.data)
        }
    }

    @Test func successfulMintPublishesTheDerivedScopeAndPreservesSource() throws {
        try fixture { root, template in
            let before = try StudyDesignSnapshot(workspaceRoot: root, name: template.name)
            let minted = try StudyTemplateStore.instantiate(templateName: template.name, cell: .agents([]), studyName: "new-study")
            #expect(minted.outcomeInstrumentScope?.itemCount == 1)
            #expect(minted.outcomeInstrumentScope?.itemIDsHash != "old")
            #expect(try ExperimentStore.load(name: minted.name) == minted)
            #expect(minted.templateProvenance?.templateHash == StudyTemplateStore.hash(template))
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: template.name).file.data == before.file.data)
        }
    }

    @Test func scopeUsesCapturedWorkspaceOrAlreadyReviewedBytes() throws {
        try fixture { root, template in
            let file = root.appending(path: template.study.taskPromptsFile!)
            let reviewed = try Data(contentsOf: file)
            WorkspaceRoot.programmaticOverride = root.appending(component: "another-workspace")
            var explicitRoot = template.study
            try OutcomeInstrumentScopeAuthoring.apply(responseFormats: ["label"], into: &explicitRoot, workspaceRoot: root)
            #expect(explicitRoot.outcomeInstrumentScope?.itemCount == 1)
            try Data("{\"id\":\"different\",\"text\":\"Choose.\",\"responseFormat\":\"label\"}\n".utf8).write(to: file)
            var captured = template.study
            try OutcomeInstrumentScopeAuthoring.apply(responseFormats: ["label"], into: &captured,
                workspaceRoot: root, reviewedPrompts: reviewed)
            #expect(captured.outcomeInstrumentScope == explicitRoot.outcomeInstrumentScope)
            try FileManager.default.removeItem(at: file)
            try OutcomeInstrumentScopeAuthoring.apply(responseFormats: [], into: &captured, workspaceRoot: root)
            #expect(captured.outcomeInstrumentScope == nil)
        }
    }
}

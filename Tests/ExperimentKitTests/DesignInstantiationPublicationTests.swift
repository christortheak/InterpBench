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
    @Test func retainedReviewRefusesChangedOrDeletedDesignWithoutPublishing() throws {
        try fixture { root, template in
            let reviewed = try StudyDesignSnapshot(workspaceRoot: root, name: template.name)
            _ = try StudyDesignAuthoring.updateDescription("Changed", reviewed: reviewed)
            do {
                _ = try StudyDesignInstantiation.instantiate(reviewed: reviewed, casting: .agents([]), studyName: "refused")
                Issue.record("A stale design review must refuse")
            } catch let error as StudyDesignAuthoringError { #expect(error.code == "designChanged") }
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/refused").path))
            try StudyTemplateStore.delete(name: template.name)
            #expect(throws: (any Error).self) {
                try StudyDesignInstantiation.instantiate(reviewed: reviewed, casting: .agents([]), studyName: "refused")
            }
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/refused").path))
        }
    }

    @Test func retainedReviewUsesItsWorkspaceForInputsNamingAndPublication() throws {
        try fixture { root, template in
            let reviewed = try StudyDesignSnapshot(workspaceRoot: root, name: template.name)
            let other = root.appending(component: "other")
            WorkspaceRoot.programmaticOverride = other
            let first = try StudyDesignInstantiation.instantiate(reviewed: reviewed, casting: .agents([]), studyName: "new-study")
            let second = try StudyDesignInstantiation.instantiate(reviewed: reviewed, casting: .agents([]), studyName: "new-study")
            #expect(first.manifest.name == "new-study")
            #expect(second.manifest.name == "new-study-2")
            #expect(first.manifest.outcomeInstrumentScope?.itemCount == 1)
            #expect(!FileManager.default.fileExists(atPath: other.path))
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: first.manifest.name).file.data == first.file.data)
        }
    }

    @Test func bothAdaptersUseTheReviewedCommandAndRequireItsPreconditions() throws {
        try fixture { root, template in
            let reviewed = try StudyDesignSnapshot(workspaceRoot: root, name: template.name)
            let casting = root.appending(component: "casting.json")
            try Data(#"{"agents":[]}"#.utf8).write(to: casting)
            let invocation = try ExperimentCLIParser.parse(namespace: "design", ["instantiate", template.name,
                "--file-sha256", reviewed.file.sha256, "--casting", casting.path, "--study-name", "cli-study", "--json"])
            let result = try StudyDesignCLI.run(invocation, workspaceRoot: root, sink: .discarding)
            #expect(result.changed)
            #expect(result.payload["name"] == .string("cli-study"))
            var fields: [String: Any] = ["workspaceRoot": root.path, "name": template.name,
                "casting": ["agents": []], "studyName": "http-study"]
            func send() throws -> StudyAuthoringHTTP.Response {
                StudyDesignHTTP.perform(.instantiate, body: try JSONSerialization.data(withJSONObject: fields), workspaceRoot: root)
            }
            #expect(try send().status == "428 Precondition Required")
            fields["designFileSHA256"] = reviewed.file.sha256
            fields["workspaceRoot"] = root.appending(component: "other").path
            #expect(try send().status == "409 Conflict")
            fields["workspaceRoot"] = root.path
            fields["casting"] = ["agents": [], "force": true] as [String: Any]
            #expect(try send().status == "400 Bad Request")
            fields["casting"] = ["agents": []]
            #expect(try send().status == "200 OK")
            let cli = try DraftAuthoringSnapshot(workspaceRoot: root, name: "cli-study")
            let http = try DraftAuthoringSnapshot(workspaceRoot: root, name: "http-study")
            #expect(cli.manifest.outcomeInstrumentScope == http.manifest.outcomeInstrumentScope)
            #expect(cli.manifest.templateProvenance == http.manifest.templateProvenance)
            _ = try StudyDesignAuthoring.updateDescription("Changed", reviewed: reviewed)
            fields["studyName"] = "refused"
            #expect(try send().status == "412 Precondition Failed")
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/refused").path))
        }
    }

    @Test func batchRetainsOneReviewAndReportsAlreadyPublishedRows() throws {
        try fixture { root, template in
            let reviewed = try StudyDesignSnapshot(workspaceRoot: root, name: template.name)
            let batch = StudyDesignInstantiation.mintBatch(reviewed: reviewed, castings: [.agents([]), .agents([])],
                names: ["first", "second"], onRow: { index, _ in
                    if index == 0 { _ = try? StudyDesignAuthoring.updateDescription("Changed between rows", reviewed: reviewed) }
                })
            #expect(batch.minted == ["first"])
            #expect(batch.failures.map(\.row) == [1])
            #expect(batch.failures.first?.failure?.contains("design changed") == true)
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/second").path))
        }
    }

    @Test func creationRefusesAStudyDirectoryRedirectIntoEvidence() throws {
        try fixture { root, template in
            let reviewed = try StudyDesignSnapshot(workspaceRoot: root, name: template.name)
            let studies = root.appending(component: "experiments")
            let original = root.appending(component: "original-studies")
            let evidence = root.appending(component: "runs")
            try FileManager.default.moveItem(at: studies, to: original)
            try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(at: studies, withDestinationURL: evidence)
            #expect(throws: ExperimentError.self) {
                try StudyDesignInstantiation.instantiate(reviewed: reviewed, casting: .agents([]), studyName: "refused")
            }
            #expect(try FileManager.default.contentsOfDirectory(atPath: evidence.path).isEmpty)
        }
    }

}

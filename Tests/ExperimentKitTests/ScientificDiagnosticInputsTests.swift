import Foundation
import Testing
@testable import ExperimentKit

@Suite struct ScientificDiagnosticInputsTests {
    private func withWorkspace<T>(_ body: (URL) throws -> T) throws -> T {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "sdi-" + UUID().uuidString)
        try FileManager.default.createDirectory(
            at: root.appending(path: "prompts/batteries"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        return try body(root)
    }

    @Test func batteriesListAsWorkspaceRelativePathsSortedByName() throws {
        try withWorkspace { root in
            let directory = root.appending(path: "prompts/batteries")
            for name in ["reasoning.jsonl", "basic.jsonl", "notes.txt", ".hidden.jsonl"] {
                try Data("{}\n".utf8).write(to: directory.appending(path: name))
            }
            try FileManager.default.createDirectory(
                at: directory.appending(path: "nested.jsonl"), withIntermediateDirectories: true)
            #expect(ScientificDiagnosticInputs.batteryFiles(root: root)
                == ["prompts/batteries/basic.jsonl", "prompts/batteries/reasoning.jsonl"])
        }
    }

    @Test func missingBatteryDirectoryListsNothing() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "sdi-missing-" + UUID().uuidString)
        #expect(ScientificDiagnosticInputs.batteryFiles(root: root).isEmpty)
    }

    @Test func chosenFilesInsideTheWorkspaceBecomeRelativeAndOutsideRefuse() throws {
        try withWorkspace { root in
            let inside = root.appending(path: "prompts/batteries/custom.jsonl")
            #expect(ScientificDiagnosticInputs.workspaceRelativePath(inside, root: root)
                == "prompts/batteries/custom.jsonl")
            // Temporary roots live under a symlinked /var; both spellings match.
            let resolved = root.resolvingSymlinksInPath()
            #expect(ScientificDiagnosticInputs.workspaceRelativePath(inside, root: resolved)
                == "prompts/batteries/custom.jsonl")
            #expect(ScientificDiagnosticInputs.workspaceRelativePath(
                resolved.appending(path: "prompts/batteries/custom.jsonl"), root: root)
                == "prompts/batteries/custom.jsonl")
            let outside = FileManager.default.temporaryDirectory
                .appending(component: "elsewhere-" + UUID().uuidString + ".jsonl")
            #expect(ScientificDiagnosticInputs.workspaceRelativePath(outside, root: root) == nil)
            // The root itself is not a file inside the root.
            #expect(ScientificDiagnosticInputs.workspaceRelativePath(root, root: root) == nil)
            // A sibling whose name merely starts with the root's name is outside.
            let sibling = URL(filePath: root.path + "-other/prompts/x.jsonl")
            #expect(ScientificDiagnosticInputs.workspaceRelativePath(sibling, root: root) == nil)
        }
    }

    @Test func agentLinesTrimAndDropBlanks() {
        let lines = ScientificDiagnosticInputs.agentLines(
            "baseline\n  kindness:17:0.28  \n\n\nruns/model-variants/calm.json\n")
        #expect(lines == ["baseline", "kindness:17:0.28", "runs/model-variants/calm.json"])
        #expect(ScientificDiagnosticInputs.agentLines("\n \n").isEmpty)
    }

    @Test func agentKindsFollowTheEngineGrammar() {
        typealias Inputs = ScientificDiagnosticInputs
        #expect(Inputs.agentKind("baseline") == .baseline)
        #expect(Inputs.agentKind("stock=baseline") == .baseline)
        #expect(Inputs.agentKind("kindness:17:0.28") == .condition)
        #expect(Inputs.agentKind("kind=kindness:17:-0.4") == .condition)
        #expect(Inputs.agentKind("runs/model-variants/calm.json") == .artifact)
        #expect(Inputs.agentKind("calm=runs/model-variants/calm.json") == .artifact)
        // A name split happens at the FIRST '='; the reference keeps later ones.
        #expect(Inputs.splitName("a=b=c").name == "a")
        #expect(Inputs.splitName("a=b=c").reference == "b=c")
        #expect(Inputs.splitName("=baseline").name == nil)
        // A malformed condition is not silently a condition.
        #expect(Inputs.agentKind("kindness:seventeen:0.28") == .artifact)
        #expect(Inputs.agentKind(":17:0.28") == .artifact)
    }

    @Test func modelCaptionNamesTheAgentsThatNeedAPin() {
        typealias Inputs = ScientificDiagnosticInputs
        #expect(Inputs.agentsNeedingModel(["baseline", "kindness:17:0.28", "runs/v.json"])
            == ["baseline", "kindness:17:0.28"])
        #expect(Inputs.modelCaption(agents: []).contains("at least one agent"))
        #expect(Inputs.modelCaption(agents: ["runs/v.json"]).hasPrefix("leave blank"))
        let required = Inputs.modelCaption(agents: ["baseline", "calm=runs/v.json"])
        #expect(required.hasPrefix("required for baseline"))
        #expect(!required.contains("calm="))
    }
}

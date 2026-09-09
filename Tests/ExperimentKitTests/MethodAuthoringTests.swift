import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct MethodAuthoringTests {
    @Test func sharedFormPublishesExactIntegerRequestAndRefusesInputDrift() async throws {
        let repository = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]))
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let input = root.appending(component: "items.jsonl")
        try Data("{\"prompt\":\"Example\"}\n".utf8).write(to: input)
        let fields = ["modelID":"example/model", "revision":String(repeating: "a", count: 40), "layer":"1", "seed":"18446744073709551615", "datasets.targetTrain":"items.jsonl", "datasets.anchorTrain":"items.jsonl", "datasets.capabilityTrain":"items.jsonl"]
        let answers: [String: Any] = ["purpose":"Compare a declared intervention", "claim":"Held-out behavioral change", "controls":"Separate control data before execution", "selection":"Fixed steps chosen before outcomes", "fields":fields, "advanced":["alphaAbsolute":1]]
        let text = String(decoding: try JSONSerialization.data(withJSONObject: answers), as: UTF8.self)
        let payload: [String: JSONValue] = ["workspaceRoot":.string(root.path), "operation":.string("optvec-train"), "answersText":.string(text)]
        let review = try await DiagnosticWorkspace.perform("draft", payload: payload, python: python, source: repository.appending(component: "Server"))
        guard case .object(let object) = review else { Issue.record("Missing review"); return }
        let hash = try #require(object["planSHA256"])
        let publication = payload.merging(["destination":.string("requests/training"), "planSHA256":hash]) { _, new in new }
        _ = try await DiagnosticWorkspace.perform("publish", payload: publication, python: python, source: repository.appending(component: "Server"))
        let request = try String(contentsOf: root.appending(path:"requests/training/request.json"), encoding:.utf8)
        #expect(request.contains("\"seed\":18446744073709551615"))
        #expect(request.contains("\"alphaAbsolute\":1"))
        try Data("changed input".utf8).write(to: input)
        await #expect(throws:(any Error).self) {
            _ = try await DiagnosticWorkspace.perform("publish", payload: payload.merging(["destination":.string("requests/revised"),"planSHA256":hash]) { _, new in new }, python:python, source:repository.appending(component: "Server"))
        }
        #expect(!FileManager.default.fileExists(atPath:root.appending(path:"requests/revised").path))
        let workflows = try ScienceCatalog.workflows()
        #expect(workflows.contains { $0.id == "jspace" })
        #expect(!workflows.contains { $0.id == "optvec-jspace" })
    }
    @Test func lensFittingInterviewReachesPortableValidation() async throws {
        let repository = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]))
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString).resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("{\"id\":\"passage-1\",\"text\":\"A short syntax fixture for authoring.\"}\n".utf8)
            .write(to: root.appending(component: "corpus.jsonl"))
        let fields = ["modelID":"example/model", "revision":String(repeating:"a", count:40), "corpus":"corpus.jsonl"]
        let answers: [String: Any] = ["purpose":"Fit an exploratory instrument", "claim":"Readouts require assessment",
            "controls":"Reserve separate text", "selection":"Begin with a timing pilot", "fields":fields, "advanced":[:]]
        let text = String(decoding: try JSONSerialization.data(withJSONObject:answers), as:UTF8.self)
        let payload: [String: JSONValue] = ["workspaceRoot":.string(root.path), "operation":.string("jlens-fit"), "answersText":.string(text)]
        let review = try await DiagnosticWorkspace.perform("draft", payload:payload, python:python, source:repository.appending(component:"Server"))
        guard case .object(let object) = review, case .object(let request) = object["request"],
              case .object(let parameters) = request["parameters"], case .object(let config) = parameters["config"] else {
            Issue.record("Missing fitting config"); return
        }
        #expect(config["maxPrompts"] == .number(4))
        #expect(config["tier"] == .string("testing"))
        #expect(config["corpus"] != nil)
        let hash = try #require(object["planSHA256"])
        _ = try await DiagnosticWorkspace.perform("publish", payload:payload.merging(
            ["destination":.string("requests/lens"), "planSHA256":hash]) { _, new in new },
            python:python, source:repository.appending(component:"Server"))
        #expect(FileManager.default.fileExists(atPath:root.appending(path:"requests/lens/request.json").path))
        #expect(try ScienceCatalog.workflows().contains { $0.id == "jlens-fit" })
        #expect(try ScienceCatalog.guide("jlens").text.contains("Corpus-authoring prompt"))
    }

}

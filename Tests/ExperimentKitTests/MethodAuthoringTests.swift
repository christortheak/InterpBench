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
        let review = try await DiagnosticWorkspace.perform("draft", payload: payload, python: python, checkout: repository)
        guard case .object(let object) = review else { Issue.record("Missing review"); return }
        let hash = try #require(object["planSHA256"])
        let publication = payload.merging(["destination":.string("requests/training"), "planSHA256":hash]) { _, new in new }
        _ = try await DiagnosticWorkspace.perform("publish", payload: publication, python: python, checkout: repository)
        let request = try String(contentsOf: root.appending(path:"requests/training/request.json"), encoding:.utf8)
        #expect(request.contains("\"seed\":18446744073709551615"))
        #expect(request.contains("\"alphaAbsolute\":1"))
        try Data("changed input".utf8).write(to: input)
        await #expect(throws:(any Error).self) {
            _ = try await DiagnosticWorkspace.perform("publish", payload: payload.merging(["destination":.string("requests/revised"),"planSHA256":hash]) { _, new in new }, python:python, checkout:repository)
        }
        #expect(!FileManager.default.fileExists(atPath:root.appending(path:"requests/revised").path))
        let workflows = try ScienceCatalog.workflows()
        #expect(workflows.contains { $0.id == "jspace" })
        #expect(!workflows.contains { $0.id == "optvec-jspace" })
    }
}

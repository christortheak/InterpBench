import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) @MainActor struct ScienceCatalogTests {
    @Test func shippedCatalogGuidesAndHTTPMatchPythonBytes() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let python = ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"] ?? "python3"
        process.arguments = [python, "-c", """
        import json
        from steerlab_server.experiment import science_catalog as s
        print(json.dumps({'catalog':s.catalog(), 'guides':{m['id']:s.guide(m['id']) for m in s.catalog()['methods']}}))
        """]
        process.currentDirectoryURL = FileManager.default.temporaryDirectory
        process.environment = ProcessInfo.processInfo.environment.merging([
            "PYTHONPATH": repository.appending(path: "Server").path, "HF_HUB_OFFLINE": "1"
        ]) { _, new in new }
        let output = Pipe(), diagnostics = Pipe()
        process.standardOutput = output; process.standardError = diagnostics
        try process.run()
        let bytes = output.fileHandleForReading.readDataToEndOfFile()
        let errors = diagnostics.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0, "\(String(decoding: errors, as: UTF8.self))")
        let result = try #require(JSONSerialization.jsonObject(with: bytes) as? NSDictionary)
        let expected = try #require(result["catalog"] as? NSDictionary)
        let catalog = try ScienceCatalog.catalog()
        let encoded = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(catalog)) as? NSDictionary)
        #expect(encoded == expected)
        let http = ScienceCatalog.http(kind: "catalog", id: nil)
        #expect(http.succeeded)
        #expect(try JSONSerialization.jsonObject(with: http.body) as? NSDictionary == expected)
        let guides = try #require(result["guides"] as? NSDictionary)
        for method in catalog.methods {
            let guide = try ScienceCatalog.guide(method.id)
            #expect(try JSONSerialization.jsonObject(with: JSONEncoder().encode(guide)) as? NSDictionary == guides[method.id] as? NSDictionary)
            #expect(try JSONSerialization.jsonObject(with: ScienceCatalog.http(kind: "guide", id: method.id).body) as? NSDictionary == guides[method.id] as? NSDictionary)
            let shipped = try Data(contentsOf: repository.appending(path: "WorkspaceSeed/prompts/method-guides/\(method.guide)"))
            #expect(Data(guide.text.utf8) == shipped)
        }
    }

    @Test func cliReturnsTheSameGuideAndRefusesInventedVerbs() async throws {
        let result = await ExperimentCLIRunner(sink: .discarding).run(namespace: "science", ["guide", "extraction", "--json"])
        #expect(result.exitCode == 0)
        #expect(result.envelope.result?["text"] == .string(try ScienceCatalog.guide("extraction").text))
        let bad = await ExperimentCLIRunner(sink: .discarding).run(namespace: "science", ["guide", "../secret", "--json"])
        #expect(bad.envelope.exitCode == 64)
        let execute = await ExperimentCLIRunner(sink: .discarding).run(namespace: "science", ["execute", "battery", "--json"])
        #expect(execute.envelope.exitCode == 64)
        #expect(!ScienceCatalog.http(kind: "guide", id: "../secret").succeeded)
        #expect(!ScienceCatalog.http(kind: "execute", id: "battery").succeeded)
    }
}

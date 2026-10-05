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
        print(json.dumps({'catalog':s.catalog(), 'brief':s.brief(), 'guides':{m['id']:s.guide(m['id']) for m in s.catalog()['methods']}}))
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
        // The short index is the same document on both clients.
        let brief = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(ScienceCatalog.brief())) as? NSDictionary)
        #expect(brief == result["brief"] as? NSDictionary)
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

    /// `science list --json` is about 75 KB, and it was the second command a
    /// new assistant was told to run. `--brief` is the index: every method and
    /// operation id, its title, one line of purpose, and the hash of the full
    /// catalog it summarizes. The full form is untouched.
    @Test func briefListIsAShortIndexOfTheSameCatalog() async throws {
        let full = try ScienceCatalog.catalog()
        let outcome = await ExperimentCLIRunner(sink: .discarding).run(namespace: "science", ["list", "--brief", "--json"])
        #expect(outcome.envelope.state == .ready)
        #expect(outcome.envelope.changed == false)
        let result = try #require(outcome.envelope.result)
        #expect(Set(result.keys) == ["schemaVersion", "brief", "catalogSHA256", "methods", "operations"])
        #expect(result["brief"] == .bool(true))
        #expect(result["catalogSHA256"] == .string(try #require(full.catalogSHA256)))

        let brief = try ScienceCatalog.brief()
        #expect(result == (try ScienceCatalog.payload(brief)))
        #expect(brief.methods.map(\.id) == full.methods.map(\.id))
        #expect(brief.operations.map(\.id) == full.operations.map(\.id))
        for (method, source) in zip(brief.methods, full.methods) {
            #expect(method.title == source.title && method.purpose == source.purpose)
        }
        for (operation, source) in zip(brief.operations, full.operations) {
            #expect(operation.method == source.method && operation.title == source.title)
            // One line, one sentence.
            #expect(!operation.purpose.isEmpty && !operation.purpose.contains("\n"))
            #expect(operation.purpose.hasSuffix(".") && !operation.purpose.contains(". "))
        }
        // A guided operation is described by its own workflow; the rest by their method.
        let workflows = Dictionary(uniqueKeysWithValues: try ScienceCatalog.workflows().map { ($0.id, $0.purpose) })
        let purposes = Dictionary(uniqueKeysWithValues: brief.operations.map { ($0.id, $0.purpose) })
        #expect(purposes["optvec-train"] == ScienceCatalog.firstSentence(try #require(workflows["optvec-train"])))
        #expect(workflows["battery"] == nil)
        #expect(purposes["battery"] == full.methods.first { $0.id == "batteries" }?.purpose)

        // The size bound: the whole document an assistant reads. It carries
        // the workspace path, whose length varies by machine, and since wave 3
        // a short `runs` phrase per operation saying where it runs; the
        // relative bound below is the one that keeps it a short index.
        let document = try outcome.envelope.jsonText()
        #expect(document.utf8.count < 14_000, "\(document.utf8.count) bytes")
        let whole = await ExperimentCLIRunner(sink: .discarding).run(namespace: "science", ["list", "--json"])
        #expect(document.utf8.count * 5 < (try whole.envelope.jsonText()).utf8.count)
        // It says where to read next.
        #expect(outcome.envelope.nextAction?.verb == "science guide <method>")

        // The full form stays exactly as it is, and says nothing about a next step.
        #expect(whole.envelope.result == (try ScienceCatalog.payload(full)))
        #expect(whole.envelope.nextAction == nil)
        // --brief belongs to `list` alone.
        let misplaced = await ExperimentCLIRunner(sink: .discarding).run(namespace: "science", ["guide", "extraction", "--brief", "--json"])
        #expect(misplaced.envelope.exitCode == 64)
    }

    /// Under --json the document is on stdout and everything a verb prints
    /// goes to stderr. The same body used to be printed there again, so a
    /// caller reading both streams paid for the catalog twice.
    @Test func jsonReadsSayOneLineInsteadOfEchoingTheBody() async throws {
        for args in [["list"], ["list", "--brief"], ["guide", "extraction"], ["operation", "battery"]] {
            let recorder = ExperimentCLIRecorder()
            let outcome = await ExperimentCLIRunner(sink: recorder.sink).run(namespace: "science", args + ["--json"])
            #expect(outcome.envelope.state == .ready, "\(args)")
            let said = recorder.standardOutput + recorder.standardError
            #expect(recorder.outputLines.count == 1, "\(args): \(said.utf8.count) bytes")
            #expect(said.hasPrefix("science \(args[0])") && said.contains("the document is on stdout"), "\(args)")
            #expect(said.utf8.count < 200, "\(args): \(said.utf8.count) bytes")
        }
        // Without --json the body is the output, as before.
        let recorder = ExperimentCLIRecorder()
        _ = await ExperimentCLIRunner(sink: recorder.sink).run(namespace: "science", ["guide", "extraction"])
        #expect(recorder.standardOutput.contains(try ScienceCatalog.guide("extraction").text))
    }
}

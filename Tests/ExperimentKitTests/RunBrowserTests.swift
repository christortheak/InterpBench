import Foundation
import Testing
@testable import ExperimentKit

/// Results-browser preview loaders — pure-CPU, fixture strings and temp
/// directories only. The contract under test: previews are BOUNDED (a huge
/// generations.jsonl is never slurped), degrade to "unavailable" rather
/// than error, and reduce each file type to what a researcher scans for.
struct RunBrowserTests {

    // MARK: JSON key/value summary

    @Test func jsonSummaryRendersScalarsAndSummarizesContainers() throws {
        let fixture = """
            {
              "experiment": "toy-french",
              "passed": true,
              "conceptCount": 3,
              "alpha": 0.4,
              "note": null,
              "layers": [10, 12, 14],
              "perConcept": {"french": 0.9, "formal": 0.7, "fear": 0.5, "calm": 0.2, "x": 0.1},
              "details": {"a": 1, "b": 2}
            }
            """
        let rows = try #require(RunBrowser.jsonKeyValues(Data(fixture.utf8)))
        let byKey = Dictionary(uniqueKeysWithValues: rows.map { ($0.key, $0.value) })
        #expect(byKey["experiment"] == "toy-french")
        #expect(byKey["passed"] == "true")
        #expect(byKey["conceptCount"] == "3")
        #expect(byKey["note"] == "null")
        #expect(byKey["layers"] == "[10, 12, 14]")
        #expect(byKey["perConcept"] == "{5 keys}")
        #expect(byKey["details"] == "{2 keys}")
        // Deterministic order: sorted keys.
        #expect(rows.map(\.key) == rows.map(\.key).sorted())
    }

    @Test func jsonSummaryTruncatesLongStringsAndCapsEntries() throws {
        let long = String(repeating: "x", count: 500)
        let fixture = "{\"a\": \"\(long)\", \"b\": 1, \"c\": 2}"
        let rows = try #require(
            RunBrowser.jsonKeyValues(Data(fixture.utf8), maxEntries: 2))
        #expect(rows.count == 2)
        #expect(rows[0].key == "a")
        #expect(rows[0].value.count == 201)  // 200 chars + ellipsis
        #expect(rows[0].value.hasSuffix("…"))
    }

    @Test func jsonSummaryRefusesNonObjectTopLevel() {
        #expect(RunBrowser.jsonKeyValues(Data("[1, 2, 3]".utf8)) == nil)
        #expect(RunBrowser.jsonKeyValues(Data("not json".utf8)) == nil)
    }

    // MARK: CSV table

    @Test func csvTableSplitsHeaderAndRows() throws {
        let fixture = """
            concept,layer,alpha,markerDensity
            french,12,0.4,0.31
            french,18,"0.8",0.40
            """
        let table = try #require(RunBrowser.csvTable(fixture))
        #expect(table.header == ["concept", "layer", "alpha", "markerDensity"])
        #expect(table.rows.count == 2)
        #expect(table.rows[0] == ["french", "12", "0.4", "0.31"])
        #expect(table.rows[1][2] == "0.8")  // quotes stripped
        #expect(!table.truncated)
    }

    @Test func csvTableCapsRowsAndFlagsTruncation() throws {
        let body = (1 ... 30).map { "c\($0),\($0)" }.joined(separator: "\n")
        let table = try #require(RunBrowser.csvTable("name,value\n" + body, maxRows: 5))
        #expect(table.rows.count == 5)
        #expect(table.truncated)
    }

    @Test func csvLineSplitIsQuoteAware() {
        #expect(
            RunBrowser.splitCSVLine(#"a,"b, with comma",c"#)
                == ["a", "b, with comma", "c"])
        #expect(RunBrowser.splitCSVLine("plain") == ["plain"])
        #expect(RunBrowser.splitCSVLine("a,,c") == ["a", "", "c"])
    }

    @Test func csvTableRefusesEmptyText() {
        #expect(RunBrowser.csvTable("") == nil)
        #expect(RunBrowser.csvTable("   \n  \n") == nil)
    }

    // MARK: JSONL records

    @Test func jsonlRecordsExtractConditionPromptOutput() {
        let fixture = """
            {"condition": "baseline", "prompt": "Write the holding.", "output": "The court holds…", "wordCount": 12}
            {"condition": "french@L12", "prompt": "Write the holding.", "output": "La cour…"}
            """
        let parsed = RunBrowser.jsonlRecords(fixture)
        #expect(parsed.records.count == 2)
        #expect(parsed.records[0].condition == "baseline")
        #expect(parsed.records[0].prompt == "Write the holding.")
        #expect(parsed.records[0].output == "The court holds…")
        #expect(parsed.records[0].fallback == nil)
        #expect(parsed.records[1].condition == "french@L12")
        #expect(!parsed.truncated)
    }

    @Test func jsonlRecordsExcerptLongOutputs() {
        let long = String(repeating: "w ", count: 400)
        let fixture = "{\"condition\": \"c\", \"output\": \"\(long)\"}"
        let parsed = RunBrowser.jsonlRecords(fixture, excerptLength: 50)
        #expect(parsed.records.count == 1)
        let output = parsed.records[0].output ?? ""
        #expect(output.count == 51)
        #expect(output.hasSuffix("…"))
    }

    @Test func jsonlRecordsFallBackForNonGenerationShapes() {
        let fixture = """
            {"judge": "paired", "winner": "A"}
            not json at all
            """
        let parsed = RunBrowser.jsonlRecords(fixture)
        #expect(parsed.records.count == 2)
        #expect(parsed.records[0].condition == nil)
        #expect(parsed.records[0].fallback?.contains("judge") == true)
        #expect(parsed.records[1].fallback == "not json at all")
    }

    @Test func jsonlRecordsCapCountAndFlagTruncation() {
        let lines = (1 ... 10)
            .map { "{\"condition\": \"c\($0)\", \"output\": \"o\"}" }
            .joined(separator: "\n")
        let parsed = RunBrowser.jsonlRecords(lines, maxRecords: 3)
        #expect(parsed.records.count == 3)
        #expect(parsed.truncated)
    }

    // MARK: Bounded head reads + preview dispatch (temp files)

    private func makeTempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "run-browser-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func readHeadIsBoundedAndDropsPartialTailLine() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appending(component: "big.jsonl")
        let line = "{\"condition\": \"c\", \"output\": \"hello\"}\n"
        let body = String(repeating: line, count: 1000)
        try Data(body.utf8).write(to: url)

        let head = try #require(RunBrowser.readHead(of: url, maxBytes: 100))
        #expect(head.truncated)
        // Only complete lines survive the cap.
        #expect(head.text.split(separator: "\n").allSatisfy { $0.hasSuffix("}") })
        #expect(head.text.utf8.count <= 100)

        let small = try #require(RunBrowser.readHead(of: url, maxBytes: 1_000_000))
        #expect(!small.truncated)
        #expect(small.text == body)
    }

    @Test func previewDispatchesByExtension() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{\"passed\": true}".utf8)
            .write(to: dir.appending(component: "report.json"))
        try Data("a,b\n1,2\n".utf8)
            .write(to: dir.appending(component: "metrics.csv"))
        try Data("{\"condition\": \"x\", \"output\": \"y\"}\n".utf8)
            .write(to: dir.appending(component: "generations.jsonl"))
        try Data([0x00, 0x01, 0x02])
            .write(to: dir.appending(component: "vectors.safetensors"))

        let files = RunBrowser.files(in: dir)
        #expect(files.map(\.name) == [
            "generations.jsonl", "metrics.csv", "report.json", "vectors.safetensors",
        ])

        for file in files {
            switch (file.name, RunBrowser.preview(for: file)) {
            case ("report.json", .keyValues(let rows)):
                #expect(rows == [RunBrowser.KeyValueRow(key: "passed", value: "true")])
            case ("metrics.csv", .table(let header, let rows, let truncated)):
                #expect(header == ["a", "b"])
                #expect(rows == [["1", "2"]])
                #expect(!truncated)
            case ("generations.jsonl", .records(let records, let truncated)):
                #expect(records.count == 1)
                #expect(records[0].condition == "x")
                #expect(!truncated)
            case ("vectors.safetensors", .unavailable):
                break
            default:
                Issue.record("unexpected preview for \(file.name)")
            }
        }
    }

    @Test func previewRefusesOversizedJSONWithoutReading() throws {
        let entry = RunBrowser.FileEntry(
            url: URL(fileURLWithPath: "/nonexistent/huge.json"),
            name: "huge.json",
            size: RunBrowser.jsonPreviewByteLimit + 1,
            isDirectory: false)
        // Size-gated BEFORE any read: a missing path must not matter.
        #expect(RunBrowser.preview(for: entry) == .unavailable(reason: "JSON too large to preview"))
    }

    @Test func previewMarksDirectories() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(
            at: dir.appending(component: "model-variants"),
            withIntermediateDirectories: true)
        let files = RunBrowser.files(in: dir)
        #expect(files.count == 1)
        #expect(files[0].isDirectory)
        #expect(RunBrowser.preview(for: files[0]) == .unavailable(reason: "directory — open in Finder"))
    }

    // MARK: config.json stamp

    @Test func readStampSurfacesRunMetadataFields() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let fixture = """
            {
              "schemaVersion": 1,
              "runType": "sweep",
              "createdAt": "2026-07-07T01:02:03Z",
              "substrate": "swift-mlx",
              "modelID": "Qwen/Qwen3-4B-MLX-4bit",
              "revision": "abc123",
              "experiment": "toy-french",
              "experimentHash": null,
              "appVersion": "swift-app 0.9.0-dev"
            }
            """
        try Data(fixture.utf8).write(to: dir.appending(component: "config.json"))
        let stamp = RunBrowser.readStamp(in: dir)
        #expect(stamp.runType == "sweep")
        #expect(stamp.experiment == "toy-french")
        #expect(stamp.modelID == "Qwen/Qwen3-4B-MLX-4bit")
        #expect(stamp.createdAt == "2026-07-07T01:02:03Z")
        #expect(stamp.revision == "abc123")
        #expect(stamp.substrate == "swift-mlx")
        #expect(stamp.appVersion == "swift-app 0.9.0-dev")
    }

    /// A schema-2 stamp (runId/platform/sampling/jobId/notes added) surfaces
    /// through the same accessor: the browser reads the fields it shows and
    /// ignores the rest, so schema bumps never break listing.
    @Test func readStampReadsSchema2Stamps() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let fixture = """
            {
              "schemaVersion": 2,
              "runId": "20260712T010203000-exp-s1-run",
              "runType": "run",
              "createdAt": "2026-07-12T01:02:03Z",
              "substrate": "python-hf-transformers",
              "appVersion": "steerlab-server 0.4.0",
              "platform": "linux-x86_64",
              "modelID": "Qwen/Qwen3-4B",
              "revision": null,
              "experiment": "s1",
              "experimentHash": null,
              "temperature": 0.7,
              "samplesPerItem": 3,
              "seedPolicy": "derivedSHA256",
              "jobId": "424242",
              "notes": {}
            }
            """
        try Data(fixture.utf8).write(to: dir.appending(component: "config.json"))
        let stamp = RunBrowser.readStamp(in: dir)
        #expect(stamp.runType == "run")
        #expect(stamp.substrate == "python-hf-transformers")
        #expect(stamp.experiment == "s1")
        #expect(stamp.modelID == "Qwen/Qwen3-4B")
        #expect(stamp.revision == nil, "JSON null degrades to absence, never a crash")
    }

    @Test func readStampDegradesToEmptyForMissingOrLegacyConfig() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(RunBrowser.readStamp(in: dir) == RunBrowser.Stamp())
        try Data("[\"legacy\"]".utf8).write(to: dir.appending(component: "config.json"))
        #expect(RunBrowser.readStamp(in: dir) == RunBrowser.Stamp())
        // Outright garbage (not JSON at all) degrades identically — a corrupt
        // stamp must never sink the Results listing.
        try Data([0xFF, 0x00, 0x9C, 0x07]).write(
            to: dir.appending(component: "config.json"))
        #expect(RunBrowser.readStamp(in: dir) == RunBrowser.Stamp())
    }

    // MARK: Memoized item (F10 — no repeated disk I/O per selection)

    @Test func memoizedItemServesRepeatsWithoutRereadingDisk() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"runType":"run","experiment":"s1"}"#.utf8).write(
            to: dir.appending(component: "config.json"))

        var memo = RunBrowser.MemoizedItem()
        let first = memo.item(at: dir)
        #expect(first.runType == "run")
        #expect(first.experiment == "s1")

        // Mutate the stamp on disk: a repeat for the SAME directory must
        // come from the memo (runs are immutable — a re-read is pure waste).
        try Data(#"{"runType":"sweep","experiment":"other"}"#.utf8).write(
            to: dir.appending(component: "config.json"))
        let repeated = memo.item(at: dir)
        #expect(repeated == first)
        #expect(repeated.runType == "run")

        // A DIFFERENT directory replaces the slot and reads fresh.
        let other = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: other) }
        try Data(#"{"runType":"validate"}"#.utf8).write(
            to: other.appending(component: "config.json"))
        #expect(memo.item(at: other).runType == "validate")
    }
}

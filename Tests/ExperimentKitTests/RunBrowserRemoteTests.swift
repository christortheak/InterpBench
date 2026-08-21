import Foundation
import Testing
@testable import ExperimentKit

/// Remote Results assembly — fixture strings and a stub fetcher only, no
/// network. The contract under test: remote previews reuse the SAME pure
/// parsers and byte caps as local browsing; JSON is size-gated from the
/// LISTED size BEFORE any bytes move; CSV/JSONL/text fetch a bounded head;
/// and the enriched `/api/runs` decode tolerates older servers.
struct RunBrowserRemoteTests {

    // MARK: Fetch plan (decided from name+size, before any fetch)

    @Test func fetchPlanSizeGatesJSONBeforeFetching() {
        // F5: even a small (or UNKNOWN-size 0) JSON fetches head-bounded —
        // the unbounded whole-file route is never planned.
        #expect(
            RunBrowser.remoteFetchPlan(name: "report.json", size: 512)
                == .head(parseLimit: RunBrowser.jsonPreviewByteLimit))
        #expect(
            RunBrowser.remoteFetchPlan(name: "report.json", size: 0)
                == .head(parseLimit: RunBrowser.jsonPreviewByteLimit))
        #expect(
            RunBrowser.remoteFetchPlan(
                name: "report.json", size: RunBrowser.jsonPreviewByteLimit + 1)
                == .none(reason: "JSON too large to preview"))
    }

    @Test func fetchPlanUsesLocalParserCapsForHeadReads() {
        // A huge generations.jsonl gets a bounded head — never the whole file.
        let jsonl = RunBrowser.remoteFetchPlan(
            name: "generations.jsonl", size: 500_000_000)
        #expect(jsonl == .head(parseLimit: 262_144))
        // +1 so truncation is detectable from the returned bytes alone.
        #expect(jsonl.requestBytes == 262_145)
        #expect(
            RunBrowser.remoteFetchPlan(name: "metrics.csv", size: 10)
                == .head(parseLimit: 131_072))
        #expect(
            RunBrowser.remoteFetchPlan(name: "notes.md", size: 10)
                == .head(parseLimit: 4_096))
    }

    @Test func fetchPlanRefusesUnknownTypes() {
        #expect(
            RunBrowser.remoteFetchPlan(name: "vectors.safetensors", size: 10)
                == .none(reason: "no preview for this file type"))
    }

    // MARK: Pure parse of fetched bytes (same parsers as local)

    @Test func remotePreviewParsesJSONKeyValues() {
        let preview = RunBrowser.remotePreview(
            name: "report.json", size: 16, data: Data("{\"passed\": true}".utf8))
        #expect(preview == .keyValues([RunBrowser.KeyValueRow(key: "passed", value: "true")]))
    }

    @Test func remotePreviewParsesCSVAndJSONL() {
        let csv = RunBrowser.remotePreview(
            name: "metrics.csv", size: 8, data: Data("a,b\n1,2\n".utf8))
        #expect(csv == .table(header: ["a", "b"], rows: [["1", "2"]], truncated: false))

        let line = "{\"condition\": \"x\", \"output\": \"y\"}\n"
        let jsonl = RunBrowser.remotePreview(
            name: "generations.jsonl", size: line.utf8.count, data: Data(line.utf8))
        guard case .records(let records, let truncated) = jsonl else {
            Issue.record("expected records, got \(jsonl)")
            return
        }
        #expect(records.count == 1)
        #expect(records[0].condition == "x")
        #expect(!truncated)
    }

    @Test func remotePreviewFlagsTruncationFromListedSize() {
        // The head fetch returned complete small bytes, but the LISTING says
        // the file is far bigger than the parser cap — truncated must be true.
        let line = "{\"condition\": \"x\", \"output\": \"y\"}\n"
        let preview = RunBrowser.remotePreview(
            name: "generations.jsonl", size: 500_000_000, data: Data(line.utf8))
        guard case .records(_, let truncated) = preview else {
            Issue.record("expected records, got \(preview)")
            return
        }
        #expect(truncated)
    }

    @Test func remotePreviewDropsPartialTailLineWhenOverCap() {
        // cap+1 bytes of CSV: the tail partial line must be dropped so the
        // parser only sees complete lines (same rule as local readHead).
        let cap = 131_072
        let row = "c,1\n"
        var text = "name,value\n"
        while text.utf8.count <= cap { text += row }
        let data = Data(text.utf8.prefix(cap + 1))
        let preview = RunBrowser.remotePreview(
            name: "sweep.csv", size: cap + 1, data: data)
        guard case .table(let header, _, let truncated) = preview else {
            Issue.record("expected table, got \(preview)")
            return
        }
        #expect(header == ["name", "value"])
        #expect(truncated)
    }

    @Test func remotePreviewRefusesOversizedJSONEvenIfBytesArrive() {
        let big = RunBrowser.jsonPreviewByteLimit + 1
        let preview = RunBrowser.remotePreview(
            name: "huge.json", size: big, data: Data("{}".utf8))
        #expect(preview == .unavailable(reason: "JSON too large to preview"))
    }

    // MARK: Async assembly with a stub fetcher (no network)

    @Test func assemblyNeverFetchesOversizedJSON() async {
        var fetchCount = 0
        let preview = await RunBrowser.remotePreview(
            name: "huge.json", size: RunBrowser.jsonPreviewByteLimit + 1
        ) { _, _ in
            fetchCount += 1
            return Data()
        }
        #expect(preview == .unavailable(reason: "JSON too large to preview"))
        #expect(fetchCount == 0)
    }

    @Test func assemblyRequestsBoundedHeadForJSONL() async {
        var requested: (name: String, maxBytes: Int?)?
        let line = "{\"condition\": \"c\", \"output\": \"o\"}\n"
        let preview = await RunBrowser.remotePreview(
            name: "generations.jsonl", size: 500_000_000
        ) { name, maxBytes in
            requested = (name, maxBytes)
            return Data(line.utf8)
        }
        #expect(requested?.name == "generations.jsonl")
        #expect(requested?.maxBytes == 262_145)
        guard case .records(let records, let truncated) = preview else {
            Issue.record("expected records, got \(preview)")
            return
        }
        #expect(records[0].condition == "c")
        #expect(truncated)  // listed size far exceeds the cap
    }

    @Test func assemblyFetchesJSONHeadBounded() async {
        var requested: (name: String, maxBytes: Int?)?
        let preview = await RunBrowser.remotePreview(
            name: "report.json", size: 16
        ) { name, maxBytes in
            requested = (name, maxBytes)
            return Data("{\"passed\": true}".utf8)
        }
        #expect(requested?.name == "report.json")
        // F5: the unbounded route is never used — a small JSON's bounded
        // head IS the whole file.
        #expect(requested?.maxBytes == RunBrowser.jsonPreviewByteLimit + 1)
        #expect(preview == .keyValues([RunBrowser.KeyValueRow(key: "passed", value: "true")]))
    }

    // MARK: remoteHead (F5: completeness proven, never assumed)

    @Test func remoteHeadProvesCompletenessFromListingOrEOF() {
        let data = Data("a\nb\n".utf8)
        // Listed size accounts for every byte → complete.
        let listed = RunBrowser.remoteHead(
            data: data, listedSize: data.count, requestedBytes: 1_000)
        #expect(!listed.truncated)
        #expect(listed.data == data)
        // Unknown listed size, but the server returned FEWER bytes than
        // requested (EOF before the cap) → complete.
        let eof = RunBrowser.remoteHead(
            data: data, listedSize: 0, requestedBytes: 1_000)
        #expect(!eof.truncated)
        #expect(eof.data == data)
    }

    @Test func remoteHeadTreatsUnknownSizeAtCapAsTruncated() {
        // Unknown (0) listed size and a cap-filling response: completeness
        // is UNPROVEN — must flag truncated and drop the partial tail line.
        let data = Data("aaaa\nbbbb\ncc".utf8)
        let head = RunBrowser.remoteHead(
            data: data, listedSize: 0, requestedBytes: data.count)
        #expect(head.truncated)
        #expect(head.data == Data("aaaa\nbbbb".utf8))
    }

    @Test func remoteHeadFlagsTruncationWhenListingExceedsBytes() {
        let data = Data("aaaa\nbbbb\ncc".utf8)
        let head = RunBrowser.remoteHead(
            data: data, listedSize: 500_000_000, requestedBytes: data.count + 100)
        #expect(head.truncated)
        #expect(head.data == Data("aaaa\nbbbb".utf8))
    }

    @Test func remoteHeadPrefersServerMetadataOverTheListing() {
        let data = Data("aaaa\nbbbb\ncc".utf8)

        // Falsy-zero listing + cap-filling bytes: the heuristic alone says
        // truncated, but the server's truncated=false verdict wins — the
        // full class of "0 means unknown" fallbacks is bypassed.
        let proven = RunBrowser.remoteHead(
            data: data, listedSize: 0, requestedBytes: data.count,
            serverFileSize: data.count, serverTruncated: false)
        #expect(!proven.truncated)
        #expect(proven.data == data)

        // The listing claims completeness, but the server that read the
        // bytes says truncated — the metadata wins, tail line dropped.
        let cut = RunBrowser.remoteHead(
            data: data, listedSize: data.count, requestedBytes: data.count + 100,
            serverFileSize: 500_000_000, serverTruncated: true)
        #expect(cut.truncated)
        #expect(cut.data == Data("aaaa\nbbbb".utf8))

        // Size header alone (no flag): more bytes on disk than returned →
        // truncated; exact match → complete.
        let sizeOnly = RunBrowser.remoteHead(
            data: data, listedSize: 0, requestedBytes: data.count,
            serverFileSize: data.count + 1, serverTruncated: nil)
        #expect(sizeOnly.truncated)
        let exact = RunBrowser.remoteHead(
            data: data, listedSize: 0, requestedBytes: data.count,
            serverFileSize: data.count, serverTruncated: nil)
        #expect(!exact.truncated)
    }

    @Test func assemblyDegradesToUnavailableOnFetchFailure() async {
        struct Boom: Error {}
        let preview = await RunBrowser.remotePreview(
            name: "metrics.csv", size: 10
        ) { _, _ in
            throw Boom()
        }
        guard case .unavailable(let reason) = preview else {
            Issue.record("expected unavailable, got \(preview)")
            return
        }
        #expect(reason.hasPrefix("fetch failed:"))
    }

    // MARK: Enriched /api/runs decode (stamps + fileEntries, older servers)

    @Test func stampedRunRecordDecodesEnrichedListing() throws {
        let fixture = """
            {
              "id": "20260707T000000000-exp-demo-run",
              "path": "/srv/ws/runs/20260707T000000000-exp-demo-run",
              "task": null,
              "hasReport": true,
              "hasGenerations": true,
              "hasCosineMatrix": false,
              "vectorNames": [],
              "files": ["config.json", "generations.jsonl", "report.json"],
              "fileEntries": [
                {"name": "config.json", "size": 312},
                {"name": "generations.jsonl", "size": 52428800},
                {"name": "report.json", "size": 2048}
              ],
              "runType": "run",
              "createdAt": "2026-07-07T00:00:00Z",
              "modelID": "org/m",
              "revision": "r9",
              "experiment": "demo",
              "substrate": "python-hf-transformers",
              "appVersion": "steerlab-server 1.0"
            }
            """
        let record = try JSONDecoder().decode(
            RemoteStampedRunRecord.self, from: Data(fixture.utf8))
        #expect(record.runType == "run")
        #expect(record.experiment == "demo")
        #expect(record.substrate == "python-hf-transformers")
        #expect(record.fileEntries?.count == 3)
        #expect(record.previewFileEntries.first {
            $0.name == "generations.jsonl"
        }?.size == 52_428_800)
    }

    @Test func stampedRunRecordToleratesOlderServers() throws {
        // An older server sends only the lean listing: every enrichment nil,
        // and previewFileEntries degrades to names with size 0 (fetches stay
        // head-bounded regardless, so 0 is safe, not a lie about bytes moved).
        let fixture = """
            {
              "id": "r1", "path": "/srv/ws/runs/r1", "task": null,
              "hasReport": false, "hasGenerations": false,
              "hasCosineMatrix": false, "vectorNames": [],
              "files": ["sweep.csv"]
            }
            """
        let record = try JSONDecoder().decode(
            RemoteStampedRunRecord.self, from: Data(fixture.utf8))
        #expect(record.runType == nil)
        #expect(record.fileEntries == nil)
        #expect(record.previewFileEntries == [RemoteRunFileEntry(name: "sweep.csv", size: 0)])
    }

    // MARK: headText (the shared bounded-head reduction)

    @Test func headTextDropsPartialTailLineOnlyWhenTruncated() {
        let whole = Data("a\nb\nc\n".utf8)
        let intact = RunBrowser.headText(whole, maxBytes: 100)
        #expect(intact.text == "a\nb\nc\n")
        #expect(!intact.truncated)

        let clipped = RunBrowser.headText(Data("aaaa\nbbbb\ncc".utf8), maxBytes: 11)
        #expect(clipped.truncated)
        #expect(clipped.text == "aaaa\nbbbb")
    }

    // MARK: Evidence-bundle discovery in a server run's file list

    @Test func evidenceBundleFileNameMatchesServerNaming() {
        let files = [
            "config.json", "generations.jsonl",
            "20260707T000000000-exp-demo-run.evidence-bundle.tar.gz",
        ]
        #expect(
            ExperimentPanel.evidenceBundleFileName(in: files)
                == "20260707T000000000-exp-demo-run.evidence-bundle.tar.gz")
        #expect(ExperimentPanel.evidenceBundleFileName(in: ["report.json"]) == nil)
    }

    @Test func csvTableAndJSONLSplitCRLFServerFiles() throws {
        // Field incident 2026-08-04 (same trap as SweepRunCatalog): server
        // CSVs are CRLF, "\r\n" is one Swift grapheme, and the previews
        // rendered a lone header row.
        let csv = "endpoint,deltaMean,adjustedP\r\nseverity,7.8,0.012\r\n"
        let table = RunBrowser.csvTable(csv)
        #expect(table?.header == ["endpoint", "deltaMean", "adjustedP"])
        #expect(table?.rows == [["severity", "7.8", "0.012"]])
        let jsonl = "{\"condition\": \"baseline\"}\r\n{\"condition\": \"steered\"}\r\n"
        #expect(RunBrowser.jsonlRecords(jsonl).records.count == 2)
    }
}

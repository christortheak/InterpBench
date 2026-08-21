import Foundation
import Testing

@testable import ExperimentKit

/// The remote run-detail loader (`ExperimentPanel.loadRemoteRunDetail`) —
/// stub fetcher only, no network. The contract under test (F5 + the
/// double-fetch/error-swallowing secondaries): every listed file is fetched
/// exactly ONCE and head-bounded, the same bytes feed previews and the
/// semantic model, unknown listed sizes degrade to possibly-truncated
/// (unless the server's head metadata proves completeness), oversize
/// report.json is still fetched (bounded, larger cap) rather than skipped,
/// fetch failures surface in `remoteResultsStatus`, and a stale failure
/// there never outlives a successful load.
@MainActor
struct RemoteRunDetailTests {

    /// Thread-safe fetch recorder for the @Sendable stub fetcher.
    private actor FetchRecorder {
        private(set) var requests: [(name: String, maxBytes: Int)] = []

        func record(name: String, maxBytes: Int) {
            requests.append((name, maxBytes))
        }

        func count(of name: String) -> Int {
            requests.count(where: { $0.name == name })
        }

        func maxBytes(for name: String) -> Int? {
            requests.first(where: { $0.name == name })?.maxBytes
        }
    }

    private let generationsLine = """
        {"condition":"baseline","promptID":"case-1","prompt":"P?",\
        "output":"guilty","wordCount":1,"distinct2":0.0,\
        "options":["guilty","not guilty"],"target":"guilty",\
        "parsedChoice":"guilty"}
        """

    private let reportJSON = """
        {"experiment":"pilot","conditions":{
           "baseline":{"generations":1,"meanWordCount":1.0,"meanDistinct2":0.0,
                       "meanMarkerDensity":{"fear":0.02}}}}
        """

    private func record(
        files: [RemoteRunFileEntry]
    ) -> RemoteStampedRunRecord {
        RemoteStampedRunRecord(
            id: "r1", path: "/srv/ws/runs/r1",
            files: files.map(\.name), fileEntries: files)
    }

    @Test func fetchesEachFileOnceAndFeedsPreviewsAndModel() async throws {
        let generations = Data(generationsLine.utf8)
        let report = Data(reportJSON.utf8)
        let run = record(files: [
            RemoteRunFileEntry(name: "generations.jsonl", size: generations.count),
            RemoteRunFileEntry(name: "report.json", size: report.count),
            RemoteRunFileEntry(name: "sweep.csv", size: 8),
            RemoteRunFileEntry(name: "vectors.safetensors", size: 4096),
        ])
        let recorder = FetchRecorder()
        let panel = ExperimentPanel()
        let payload = await panel.loadRemoteRunDetail(run: run) { name, maxBytes in
            await recorder.record(name: name, maxBytes: maxBytes)
            switch name {
            case "generations.jsonl": return RemoteRunFileHead(data: generations)
            case "report.json": return RemoteRunFileHead(data: report)
            case "sweep.csv": return RemoteRunFileHead(data: Data("a,b\n1,2\n".utf8))
            default: return RemoteRunFileHead(data: Data())
            }
        }

        // Exactly one fetch per fetchable file; the unpreviewable binary
        // moves no bytes at all.
        #expect(await recorder.count(of: "generations.jsonl") == 1)
        #expect(await recorder.count(of: "report.json") == 1)
        #expect(await recorder.count(of: "sweep.csv") == 1)
        #expect(await recorder.count(of: "vectors.safetensors") == 0)

        // The same fetch feeds BOTH surfaces.
        #expect(payload.previewed.count == 3)
        #expect(payload.other.map(\.name) == ["vectors.safetensors"])
        let model = try #require(payload.model)
        #expect(model.records.count == 1)
        #expect(!model.generationsTruncated)  // listed size fully accounted
        #expect(model.report?.conditions["baseline"]?.meanMarkerDensity?["fear"] == 0.02)
    }

    @Test func unknownListedSizeAtCapReadsAsPossiblyTruncated() async throws {
        // Older server: size 0 (unknown). The fetcher fills the requested
        // cap exactly — completeness is unproven, so the model must flag
        // truncation (the caption's input) instead of posing as complete.
        let line = generationsLine + "\n"
        let run = record(files: [
            RemoteRunFileEntry(name: "generations.jsonl", size: 0)
        ])
        let panel = ExperimentPanel()
        let payload = await panel.loadRemoteRunDetail(run: run) { _, maxBytes in
            var data = Data()
            while data.count < maxBytes { data.append(Data(line.utf8)) }
            return RemoteRunFileHead(data: Data(data.prefix(maxBytes)))
        }
        let model = try #require(payload.model)
        #expect(model.generationsTruncated)
        // The partial tail line was dropped, not miscounted as undecodable.
        #expect(model.skippedRecordLines == 0)
        #expect(!model.records.isEmpty)
    }

    @Test func unknownListedSizeBelowCapReadsAsComplete() async throws {
        // Unknown size but the server returned fewer bytes than requested
        // (EOF before the cap): completeness is proven — no false caption.
        let run = record(files: [
            RemoteRunFileEntry(name: "generations.jsonl", size: 0)
        ])
        let generations = Data(generationsLine.utf8)
        let panel = ExperimentPanel()
        let payload = await panel.loadRemoteRunDetail(run: run) { _, _ in
            RemoteRunFileHead(data: generations)
        }
        let model = try #require(payload.model)
        #expect(!model.generationsTruncated)
        #expect(model.records.count == 1)
    }

    @Test func oversizeReportIsFetchedHeadBoundedNotSkipped() async throws {
        // report.json listed ABOVE the 1 MiB preview gate: the preview is
        // refused, but the semantic fetch still runs — head-bounded at the
        // report cap — so the stamped whole-run numbers survive exactly in
        // the big-run case where head-derived numbers are least trustworthy.
        // (Padded to its listed size: JSON tolerates trailing whitespace.)
        let listedSize = RunBrowser.jsonPreviewByteLimit + 1
        var padded = Data(reportJSON.utf8)
        padded.append(Data(repeating: 0x20, count: listedSize - padded.count))
        let report = padded
        let run = record(files: [
            RemoteRunFileEntry(name: "report.json", size: listedSize)
        ])
        let recorder = FetchRecorder()
        let panel = ExperimentPanel()
        let payload = await panel.loadRemoteRunDetail(run: run) { name, maxBytes in
            await recorder.record(name: name, maxBytes: maxBytes)
            return RemoteRunFileHead(data: report)
        }
        #expect(await recorder.count(of: "report.json") == 1)
        #expect(
            await recorder.maxBytes(for: "report.json")
                == ExperimentPanel.remoteReportByteLimit)
        // Bounded, never unbounded: the request always carries a cap by
        // construction of the fetcher signature (no nil case exists).
        let model = try #require(payload.model)
        #expect(model.report?.conditions.count == 1)
        // The preview side stays size-gated like the local browser.
        #expect(payload.previewed.isEmpty)
        #expect(payload.other.map(\.name) == ["report.json"])
    }

    @Test func fetchFailuresSurfaceInStatusNeverBareNil() async throws {
        let generations = Data(generationsLine.utf8)
        let run = record(files: [
            RemoteRunFileEntry(name: "generations.jsonl", size: generations.count),
            RemoteRunFileEntry(name: "report.json", size: 64),
        ])
        struct Boom: Error {}
        let panel = ExperimentPanel()
        let payload = await panel.loadRemoteRunDetail(run: run) { name, _ in
            if name == "report.json" { throw Boom() }
            return RemoteRunFileHead(data: generations)
        }
        // The failure is attributable in the existing status surface…
        let status = try #require(panel.remoteResultsStatus)
        #expect(status.contains("report.json"))
        #expect(status.contains("could not fetch"))
        // …and the rest of the detail still renders (partial, not nil).
        let model = try #require(payload.model)
        #expect(model.records.count == 1)
        #expect(model.report == nil)
        #expect(payload.other.map(\.name) == ["report.json"])
    }

    @Test func priorFailureStatusIsClearedByASuccessfulLoad() async throws {
        // A failed load leaves an attributable error in remoteResultsStatus;
        // the NEXT successful load must clear it — a stale error caption on
        // fresh, complete results misreports the run.
        let generations = Data(generationsLine.utf8)
        let run = record(files: [
            RemoteRunFileEntry(name: "generations.jsonl", size: generations.count)
        ])
        struct Boom: Error {}
        let panel = ExperimentPanel()
        _ = await panel.loadRemoteRunDetail(run: run) { _, _ in throw Boom() }
        #expect(panel.remoteResultsStatus?.contains("could not fetch") == true)

        let payload = await panel.loadRemoteRunDetail(run: run) { _, _ in
            RemoteRunFileHead(data: generations)
        }
        #expect(payload.model?.records.count == 1)
        #expect(panel.remoteResultsStatus == nil)  // stale error gone
    }

    // MARK: - Server head metadata (the raised-cap contract)

    @Test func twoMiBReportMetricsSurviveThroughRealHeaderMetadata() async throws {
        // Regression for the 1 MiB server clamp: a >1 MiB report.json now
        // arrives COMPLETE (server cap 8 MiB ≥ the client's 4 MiB request)
        // and its stamped run-level metrics survive. The stub mirrors the
        // real route's response exactly: full bytes + X-SteerLab-File-Size
        // + X-SteerLab-Truncated: false.
        var padded = Data(reportJSON.utf8)
        padded.append(Data(repeating: 0x20, count: 2_097_152))  // ~2 MiB JSON
        let report = padded
        let run = record(files: [
            RemoteRunFileEntry(name: "report.json", size: report.count)
        ])
        let panel = ExperimentPanel()
        let payload = await panel.loadRemoteRunDetail(run: run) { _, _ in
            RemoteRunFileHead(
                data: report, fileSize: report.count, truncated: false)
        }
        let model = try #require(payload.model)
        #expect(model.report?.conditions["baseline"]?.generations == 1)
        #expect(
            model.report?.conditions["baseline"]?.meanMarkerDensity?["fear"] == 0.02)
        #expect(panel.remoteResultsStatus == nil)
    }

    @Test func serverMetadataProvesCompletenessDespiteFalsyZeroListing() async throws {
        // Falsy-zero listing (size 0 = unknown) with a response that fills
        // the requested cap EXACTLY: the heuristic must call that "possibly
        // truncated" and cut at the last newline — which destroys a JSON
        // report — but the server's truncated=false metadata proves the file
        // is complete and the stamped metrics survive.
        var padded = Data(reportJSON.utf8)
        padded.append(
            Data(
                repeating: 0x20,
                count: ExperimentPanel.remoteReportByteLimit - padded.count))
        let report = padded  // exactly the requested head size
        let run = record(files: [
            RemoteRunFileEntry(name: "report.json", size: 0)
        ])
        let panel = ExperimentPanel()

        // Older server (no metadata): honest degrade — the cap-filling head
        // is possibly truncated, the newline cut breaks the decode, report
        // metrics are absent rather than silently biased.
        let legacy = await panel.loadRemoteRunDetail(run: run) { _, _ in
            RemoteRunFileHead(data: report)
        }
        #expect(legacy.model?.report == nil)

        // Metadata-stamping server: completeness proven, metrics intact.
        let payload = await panel.loadRemoteRunDetail(run: run) { _, _ in
            RemoteRunFileHead(
                data: report, fileSize: report.count, truncated: false)
        }
        let model = try #require(payload.model)
        #expect(
            model.report?.conditions["baseline"]?.meanMarkerDensity?["fear"] == 0.02)
    }

    @Test func serverTruncatedFlagWinsOverACompleteLookingListing() async throws {
        // The listing claims the returned bytes are the whole file, but the
        // server (which actually read the bytes) says truncated — the
        // metadata wins and the model carries the truncation flag.
        let line = generationsLine + "\n"
        let data = Data((line + line).utf8)
        let run = record(files: [
            RemoteRunFileEntry(name: "generations.jsonl", size: data.count)
        ])
        let panel = ExperimentPanel()
        let payload = await panel.loadRemoteRunDetail(run: run) { _, _ in
            RemoteRunFileHead(
                data: data, fileSize: data.count * 10, truncated: true)
        }
        let model = try #require(payload.model)
        #expect(model.generationsTruncated)
        #expect(!model.records.isEmpty)
    }

    @Test func headMetadataDecodesFromTheRealResponseHeaders() throws {
        // Swift-side decode of the head-route metadata contract: the exact
        // header names + casing the server stamps, plus the degrade-to-nil
        // rule for older servers and garbage values.
        let url = try #require(URL(string: "http://127.0.0.1:8080/api/runs/r1/file"))
        func response(_ headers: [String: String]) throws -> HTTPURLResponse {
            try #require(
                HTTPURLResponse(
                    url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                    headerFields: headers))
        }
        let data = Data("{}".utf8)

        let stamped = RemoteRunFileHead(
            data: data,
            response: try response(
                ["X-SteerLab-File-Size": "2097152", "X-SteerLab-Truncated": "false"]))
        #expect(stamped.fileSize == 2_097_152)
        #expect(stamped.truncated == false)

        let truncated = RemoteRunFileHead(
            data: data,
            response: try response(
                ["X-SteerLab-File-Size": "9000000", "X-SteerLab-Truncated": "true"]))
        #expect(truncated.fileSize == 9_000_000)
        #expect(truncated.truncated == true)

        // Older server: no headers → nil metadata (unknown, never false).
        let legacy = RemoteRunFileHead(data: data, response: try response([:]))
        #expect(legacy.fileSize == nil)
        #expect(legacy.truncated == nil)

        // Garbage degrades to nil rather than a guess.
        let garbage = RemoteRunFileHead(
            data: data,
            response: try response(
                ["X-SteerLab-File-Size": "many", "X-SteerLab-Truncated": "maybe"]))
        #expect(garbage.fileSize == nil)
        #expect(garbage.truncated == nil)
    }
}

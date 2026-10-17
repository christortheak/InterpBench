import Foundation
import Testing

@testable import ExperimentKit

/// The browser companion's response pages
/// (`GET /api/experiment/result/responses`, `StudyResponsePageHTTP`).
///
/// The companion used to show only the state document's preview — at most
/// `StudyResultRepository.previewResponseLimit` responses — and to put that
/// capped number on its button. These pin the route's paging and its true
/// total, over a run whose file holds more records than the preview.
struct StudyResponsePageHTTPTests {

    /// 123 records: 120 responses (the 7th cut off), one answer-option
    /// reading, one failure record, and one line that is not JSON.
    static let recordCount = 123

    static func stagedRun() throws -> (root: URL, detail: StudyRunDetail) {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appending(component: "pages-\(UUID().uuidString)")
        let run = root.appending(components: "runs", "2026-10-17T00-00-00Z-demo-run")
        try fm.createDirectory(at: run, withIntermediateDirectories: true)
        var lines: [String] = []
        for index in 0..<120 {
            let finish = index == 6 ? "length" : "stop"
            lines.append(
                #"{"condition":"baseline","promptID":"p\#(index)","prompt":"question \#(index)","output":"answer \#(index)","wordCount":2,"distinct2":1.0,"finishReason":"\#(finish)"}"#)
        }
        lines.append(#"{"condition":"baseline","promptID":"c1","instrument":"answerTokenLogprob","selected":"B","margin":0.5}"#)
        lines.append(#"{"condition":"agent-a","promptID":"p0","error":"the agent could not be loaded"}"#)
        lines.append("not json at all")
        try (lines.joined(separator: "\n") + "\n").write(
            to: run.appending(component: "generations.jsonl"), atomically: true, encoding: .utf8)
        let item = StudyRunListItem(
            directoryName: run.lastPathComponent, path: run.path, kind: .run,
            createdAt: "2026-10-17T00-00-00Z", generationCount: recordCount, hasReport: false)
        let detail = StudyResultRepository(workspaceRoot: root).detail(for: item)
        return (root, detail)
    }

    static func json(_ response: StudyAuthoringHTTP.Response) throws -> [String: Any] {
        try #require(
            try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
    }

    static func page(
        _ detail: StudyRunDetail?, _ query: String = ""
    ) throws -> (status: String, body: [String: Any]) {
        let response = StudyResponsePageHTTP.page(
            selected: detail, path: "/api/experiment/result/responses" + query)
        return (response.status, try json(response))
    }

    static func records(_ body: [String: Any]) -> [Int] {
        ((body["rows"] as? [[String: Any]]) ?? []).compactMap { $0["record"] as? Int }
    }

    @Test func theTotalIsTheFilesNotThePreviews() throws {
        let (root, detail) = try Self.stagedRun()
        defer { try? FileManager.default.removeItem(at: root) }
        // The state document's preview stops at the cap…
        #expect(detail.generations.count == StudyResultRepository.previewResponseLimit)
        #expect(detail.responseRecordCount == Self.recordCount)
        // …the route counts every record.
        let (status, body) = try Self.page(detail)
        #expect(status == "200 OK")
        #expect(body["total"] as? Int == Self.recordCount)
        #expect(body["offset"] as? Int == 0)
        #expect(body["limit"] as? Int == StudyResponsePageHTTP.defaultLimit)
        #expect(Self.records(body) == Array(1...50))
        #expect(body["caption"] as? String == "Showing 1–50 of 123 records.")
    }

    @Test func pagesCoverEveryRecordExactlyOnce() throws {
        let (root, detail) = try Self.stagedRun()
        defer { try? FileManager.default.removeItem(at: root) }
        var seen: [Int] = []
        var offset = 0
        while true {
            let (_, body) = try Self.page(detail, "?offset=\(offset)&limit=50")
            let records = Self.records(body)
            if records.isEmpty { break }
            seen += records
            offset += 50
        }
        #expect(seen == Array(1...Self.recordCount))

        let (_, last) = try Self.page(detail, "?offset=100&limit=50")
        #expect(Self.records(last) == Array(101...123))
        #expect(last["caption"] as? String == "Showing 101–123 of 123 records.")
        let (_, one) = try Self.page(detail, "?offset=122&limit=50")
        #expect(one["caption"] as? String == "Showing record 123 of 123.")
    }

    @Test func theLimitIsBoundedAndTheReplySaysWhichWasUsed() throws {
        let (root, detail) = try Self.stagedRun()
        defer { try? FileManager.default.removeItem(at: root) }
        let (status, body) = try Self.page(detail, "?limit=100000")
        #expect(status == "200 OK")
        #expect(body["limit"] as? Int == StudyResponsePageHTTP.maximumLimit)
        #expect(Self.records(body).count == Self.recordCount)  // 123 < 200
        #expect(body["caption"] as? String == "Showing all 123 records.")
    }

    @Test func anOffsetPastTheEndIsBroughtIntoRange() throws {
        let (root, detail) = try Self.stagedRun()
        defer { try? FileManager.default.removeItem(at: root) }
        let (status, body) = try Self.page(detail, "?offset=5000")
        #expect(status == "200 OK")
        #expect(body["offset"] as? Int == Self.recordCount)
        #expect(Self.records(body).isEmpty)
        #expect(
            body["caption"] as? String
                == "No records from this position; the file holds 123 records.")
    }

    @Test(arguments: ["?offset=-1", "?offset=abc", "?limit=0", "?limit=-5", "?limit=ten"])
    func aMalformedPageIsRefusedWithARepair(_ query: String) throws {
        let (root, detail) = try Self.stagedRun()
        defer { try? FileManager.default.removeItem(at: root) }
        let (status, body) = try Self.page(detail, query)
        #expect(status == "400 Bad Request")
        #expect(body["code"] as? String == "invalidResponsePage")
        #expect((body["repairAction"] as? String)?.contains("limit from 1 to 200") == true)
    }

    @Test func withNoRunSelectedTheReplySaysWhatToDo() throws {
        let (status, body) = try Self.page(nil)
        #expect(status == "409 Conflict")
        #expect(body["code"] as? String == "noResultSelected")
        #expect((body["repairAction"] as? String)?.contains("Choose a run") == true)
    }

    @Test func everyKindOfLineIsARowAndIsCounted() throws {
        let (root, detail) = try Self.stagedRun()
        defer { try? FileManager.default.removeItem(at: root) }
        let (_, body) = try Self.page(detail, "?offset=0&limit=200")
        let rows = try #require(body["rows"] as? [[String: Any]])
        try #require(rows.count == Self.recordCount)
        #expect(rows[6]["kind"] as? String == "cutOff")
        #expect(rows[6]["statusNote"] as? String != nil)
        #expect(rows[120]["kind"] as? String == "reading")
        #expect(rows[120]["selected"] as? String == "B")
        #expect(rows[121]["kind"] as? String == "failed")
        #expect(rows[121]["failure"] as? String == "the agent could not be loaded")
        #expect(rows[122]["kind"] as? String == "unreadable")
        #expect(rows[122]["rawJSON"] as? String == "not json at all")
        #expect(rows[0]["rawJSON"] == nil)

        let counts = try #require(body["counts"] as? [[String: Any]])
        let byLabel = Dictionary(
            uniqueKeysWithValues: counts.compactMap { entry -> (String, Int)? in
                guard let label = entry["label"] as? String, let count = entry["count"] as? Int
                else { return nil }
                return (label, count)
            })
        #expect(byLabel["Responses"] == 119)
        #expect(byLabel["Cut off"] == 1)
        #expect(byLabel["Answer-option readings"] == 1)
        #expect(byLabel["Failed"] == 1)
        #expect(byLabel["Unreadable"] == 1)
        #expect(byLabel.values.reduce(0, +) == Self.recordCount)
    }

    @Test func aRunWithNoResponsesFileSaysSo() throws {
        let (root, detail) = try Self.stagedRun()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.removeItem(
            at: URL(filePath: detail.item.path).appending(component: "generations.jsonl"))
        let (status, body) = try Self.page(detail)
        #expect(status == "200 OK")
        #expect(body["total"] as? Int == 0)
        #expect((body["caption"] as? String)?.contains("no readable generations.jsonl") == true)
    }

    // MARK: The route and the companion

    private static var root: URL { CodeResources.compiledCheckoutPath }

    @Test func theWebServerServesTheRoute() throws {
        let server = try String(
            contentsOf: Self.root.appending(path: "Sources/ExperimentKit/WebServer.swift"),
            encoding: .utf8)
        #expect(server.contains(#"== "/api/experiment/result/responses""#))
        #expect(server.contains("StudyResponsePageHTTP.page(selected: detail, path: path)"))
    }

    /// The companion pages through the route and labels its button with the
    /// file's own total, never the preview's length.
    @Test(.enabled(
        if: FileManager.default.fileExists(
            atPath: CodeResources.compiledCheckoutPath.appending(path: "web/index.html").path),
        "web/index.html is not in this checkout"))
    func theCompanionPagesAndShowsTheTrueTotal() throws {
        let source = try String(
            contentsOf: Self.root.appending(path: "web/index.html"), encoding: .utf8)
        #expect(source.contains("/api/experiment/result/responses?offset="))
        #expect(source.contains("d.responseRecordCount"))
        #expect(!source.contains("Review Responses (${d.generations.length})"))
    }
}

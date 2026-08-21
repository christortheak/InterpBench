import Foundation
import Testing

@testable import ExperimentKit

/// The server-geometry wire contract from the client side:
///
/// 1. Address form — the panel used to send `RemoteVectorRecord.id`
///    (`<runDirectory>/<name>` in one string) as `vectorPath`; the route
///    `require_dir`s that path, so EVERY vector silently skipped and the
///    panel captioned a "computed 0×0 cosine matrix" success. The refs must
///    send the run DIRECTORY (the form the server's own workbench sends and
///    every server version resolves), with the name separate.
/// 2. Skip surfacing — `skipped: [{name, reason}]` decodes when present
///    (older servers omit it), and the status line names every skipped
///    vector; an empty matrix is reported as a failure, never as a bare
///    "0×0" success.
@Suite struct GeometryServerContractTests {

    private func record(
        runDirectory: String, name: String, concept: String
    ) throws -> RemoteVectorRecord {
        let json = """
            {"id": "\(runDirectory)/\(name)", "runDirectory": "\(runDirectory)",
             "name": "\(name)", "concept": "\(concept)", "modelID": "org/m",
             "layerCount": 4, "hiddenSize": 8}
            """
        return try JSONDecoder().decode(
            RemoteVectorRecord.self, from: Data(json.utf8))
    }

    @Test func refsSendTheRunDirectoryNotTheCatalogID() throws {
        let records = [
            try record(runDirectory: "/srv/runs/r1", name: "arousal", concept: "arousal"),
            try record(runDirectory: "/srv/runs/r2", name: "french", concept: "french"),
        ]
        let refs = ClusterClient.geometryRefs(for: records)
        #expect(refs.map(\.vectorPath) == ["/srv/runs/r1", "/srv/runs/r2"])
        #expect(refs.map(\.name) == ["arousal", "french"])
        #expect(refs[0].label == "arousal · arousal")
        // The broken form: vectorPath must never be the id (dir/name).
        #expect(!refs.contains { $0.vectorPath.hasSuffix("/\($0.name)") })
    }

    @Test func skippedFieldDecodesAndOlderServersDecodeWithoutIt() throws {
        let with = """
            {"labels": ["a"], "matrix": [[1.0]], "layers": [2],
             "skipped": [{"name": "ghost", "reason": "no such directory"}]}
            """
        let decoded = try JSONDecoder().decode(
            RemoteGeometryResult.self, from: Data(with.utf8))
        #expect(decoded.skipped == [.init(name: "ghost", reason: "no such directory")])

        let without = #"{"labels": [], "matrix": [], "layers": []}"#
        let legacy = try JSONDecoder().decode(
            RemoteGeometryResult.self, from: Data(without.utf8))
        #expect(legacy.skipped == nil)
    }

    @Test func statusLineNamesSkippedVectors() {
        let result = RemoteGeometryResult(
            labels: ["a", "b"], matrix: [[1, 0], [0, 1]], layers: [2, 2],
            skipped: [.init(name: "ghost", reason: "no such directory")])
        let line = result.statusLine(requested: 3)
        #expect(line.contains("2×2"))
        #expect(line.contains("skipped 1 of 3"))
        #expect(line.contains("ghost (no such directory)"))
    }

    @Test func emptyMatrixIsNeverASuccessCaption() {
        // Older server without the skip contract: 0×0 comes back as a 200.
        let legacy = RemoteGeometryResult(labels: [], matrix: [], layers: [])
        let line = legacy.statusLine(requested: 2)
        #expect(!line.contains("0×0"))
        #expect(line.contains("none of the 2"))

        let explained = RemoteGeometryResult(
            labels: [], matrix: [], layers: [],
            skipped: [
                .init(name: "a", reason: "no such directory"),
                .init(name: "b", reason: "non-finite values"),
            ])
        let explainedLine = explained.statusLine(requested: 2)
        #expect(explainedLine.contains("none of the 2"))
        #expect(explainedLine.contains("a (no such directory)"))
        #expect(explainedLine.contains("b (non-finite values)"))
    }

    @Test func cleanResultKeepsThePlainCaption() {
        let result = RemoteGeometryResult(
            labels: ["a", "b"], matrix: [[1, 0], [0, 1]], layers: [2, 2],
            skipped: [])
        #expect(
            result.statusLine(requested: 2)
                == "computed 2×2 cosine matrix on the server")
    }
}

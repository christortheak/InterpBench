import Foundation
import Testing

@testable import ExperimentKit

/// Cross-engine vector parity harness (WS7.3) over the committed fixtures.
///
/// `Fixtures/parity/` holds tiny hand-authored artifact pairs (4 layers × 8
/// dims) plus `golden-<pair>.json` files produced by the PYTHON engine's
/// compare verb; this suite asserts the byte-identical fixture copies against
/// the SAME goldens the Python suite (`Server/tests/test_vector_parity.py`)
/// asserts — same numbers to 1e-6, same JSON key set — so engine drift is a
/// test failure, not a paper-review surprise. These integer fixtures are
/// scaffolding for the harness; the real MLX-vs-CUDA french-vector fixture
/// pair is added after the first cluster session. Pure CPU: the parity
/// loader parses safetensors directly, no MLX/Metal.
@Suite struct VectorParityTests {

    private static var fixturesDirectory: URL {
        URL(filePath: #filePath)  // …/Tests/ExperimentKitTests/VectorParityTests.swift
            .deletingLastPathComponent()
            .appending(components: "Fixtures", "parity")
    }

    private func fixture(_ name: String) -> URL {
        Self.fixturesDirectory.appending(component: "\(name).safetensors")
    }

    private func golden(_ pair: String) throws -> [String: Any] {
        let url = Self.fixturesDirectory.appending(component: "golden-\(pair).json")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        return try #require(object as? [String: Any])
    }

    private func compare(_ a: String, _ b: String) throws -> VectorParity.Report {
        try VectorParity.compareArtifacts(pathA: fixture(a), pathB: fixture(b))
    }

    /// Same key sets at every level; numbers equal to 1e-6; exact non-numbers
    /// (the Python twin applies the identical recursion to the same goldens).
    private func assertMatchesGolden(_ got: Any, _ want: Any, path: String = "$") {
        switch (got, want) {
        case (let g as [String: Any], let w as [String: Any]):
            #expect(Set(g.keys) == Set(w.keys), "\(path): key set differs")
            for key in w.keys {
                assertMatchesGolden(g[key] ?? NSNull(), w[key] ?? NSNull(), path: "\(path).\(key)")
            }
        case (let g as [Any], let w as [Any]):
            #expect(g.count == w.count, "\(path): length differs")
            for (index, pair) in zip(g, w).enumerated() {
                assertMatchesGolden(pair.0, pair.1, path: "\(path)[\(index)]")
            }
        case (is NSNull, is NSNull):
            break
        case (let g as NSNumber, let w as NSNumber):
            // NSNumber wraps bools and numbers alike; compare bools exactly.
            if CFGetTypeID(w) == CFBooleanGetTypeID() || CFGetTypeID(g) == CFBooleanGetTypeID() {
                #expect(g.boolValue == w.boolValue, "\(path)")
            } else {
                #expect(abs(g.doubleValue - w.doubleValue) <= 1e-6, "\(path)")
            }
        case (let g as String, let w as String):
            #expect(g == w, "\(path)")
        default:
            Issue.record("\(path): type mismatch (\(got) vs \(want))")
        }
    }

    private func reportJSONObject(_ report: VectorParity.Report) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(
            with: Data(report.jsonText().utf8))
        return try #require(object as? [String: Any])
    }

    // MARK: - The scaffolding pairs against the shared goldens

    @Test func identicalPairMatchesGolden() throws {
        let report = try compare("identical-a", "identical-b")
        assertMatchesGolden(try reportJSONObject(report), try golden("identical"))
        #expect(report.passed)
        let minCosine = try #require(report.minCosine)
        #expect(abs(minCosine - 1.0) <= 1e-6)
        let meanRatio = try #require(report.meanNormRatio)
        #expect(abs(meanRatio - 1.0) <= 1e-6)
    }

    @Test func orthogonalPairMatchesGoldenAndFails() throws {
        let report = try compare("orthogonal-a", "orthogonal-b")
        assertMatchesGolden(try reportJSONObject(report), try golden("orthogonal"))
        #expect(!report.passed)
        let minCosine = try #require(report.minCosine)
        #expect(abs(minCosine) <= 1e-6)
    }

    @Test func scaledPairMatchesGolden() throws {
        let report = try compare("scaled-a", "scaled-b")
        assertMatchesGolden(try reportJSONObject(report), try golden("scaled"))
        #expect(report.passed)
        for row in report.perLayer {
            let ratio = try #require(row.normRatio)
            #expect(abs(ratio - 2.0) <= 1e-6)
        }
    }

    @Test func layerCountMismatchComparesIntersectionAndSaysSo() throws {
        let report = try compare("truncated-a", "identical-b")
        assertMatchesGolden(try reportJSONObject(report), try golden("truncated"))
        #expect(report.layerCountMismatch)
        #expect(report.comparedLayerCount == 3)
        #expect(report.artifactA.layerCount == 3)
        #expect(report.artifactB.layerCount == 4)
    }

    @Test func reportJSONKeySetIsThePinnedContract() throws {
        let object = try reportJSONObject(try compare("identical-a", "identical-b"))
        #expect(
            Set(object.keys) == [
                "artifactA", "artifactB", "comparedLayerCount", "layerCountMismatch",
                "pass", "perLayer", "summary", "threshold",
            ])
        let artifactA = try #require(object["artifactA"] as? [String: Any])
        #expect(Set(artifactA.keys) == ["hiddenSize", "layerCount", "name"])
        let perLayer = try #require(object["perLayer"] as? [[String: Any]])
        #expect(Set(try #require(perLayer.first).keys)
            == ["cosine", "layer", "normA", "normB", "normRatio"])
        let summary = try #require(object["summary"] as? [String: Any])
        #expect(Set(summary.keys)
            == ["meanCosine", "meanNormRatio", "minCosine", "skippedZeroNormLayers"])
    }

    // MARK: - Semantics the fixtures cannot reach (pure, in-memory)

    @Test func zeroNormLayersAreSkippedAndCounted() throws {
        let report = try VectorParity.compare(
            nameA: "a", perLayerA: [[0, 0], [1, 0]],
            nameB: "b", perLayerB: [[1, 0], [1, 0]])
        #expect(report.perLayer[0].cosine == nil)
        #expect(report.perLayer[0].normRatio == nil)  // normA == 0
        let cosine = try #require(report.perLayer[1].cosine)
        #expect(abs(cosine - 1.0) <= 1e-12)
        #expect(report.skippedZeroNormLayers == 1)
    }

    @Test func nothingComparableNeverPasses() throws {
        let report = try VectorParity.compare(
            nameA: "a", perLayerA: [[0, 0]],
            nameB: "b", perLayerB: [[0, 0]],
            threshold: 0.0)
        #expect(report.minCosine == nil)
        #expect(report.meanCosine == nil)
        #expect(!report.passed)
        let object = try reportJSONObject(report)
        let summary = try #require(object["summary"] as? [String: Any])
        #expect(summary["minCosine"] is NSNull)
        #expect(try #require(object["pass"] as? Bool) == false)
    }

    @Test func hiddenSizeMismatchIsAnErrorNotAReport() {
        #expect(throws: VectorParity.ParityError.self) {
            try VectorParity.compare(
                nameA: "a", perLayerA: [[1, 2]],
                nameB: "b", perLayerB: [[1, 2, 3]])
        }
    }

    @Test func normRatioIsBOverA() throws {
        let report = try VectorParity.compare(
            nameA: "a", perLayerA: [[3, 4]],
            nameB: "b", perLayerB: [[6, 8]])
        #expect(abs(report.perLayer[0].normA - 5.0) <= 1e-12)
        #expect(abs(report.perLayer[0].normB - 10.0) <= 1e-12)
        let ratio = try #require(report.perLayer[0].normRatio)
        #expect(abs(ratio - 2.0) <= 1e-12)
    }

    @Test func thresholdGateMirrorsTheCLIExit() throws {
        // The CLI exits nonzero exactly when `passed` is false; pin the flip
        // around the default threshold here (process-level exit codes are
        // covered by the Python twin, same contract).
        let failing = try compare("orthogonal-a", "orthogonal-b")
        #expect(!failing.passed)
        let report = try VectorParity.compareArtifacts(
            pathA: fixture("orthogonal-a"), pathB: fixture("orthogonal-b"),
            threshold: -0.5)
        #expect(report.passed)
        #expect(report.threshold == -0.5)
    }

    // MARK: - The pure safetensors reader

    @Test func safetensorsReaderRoundTripsTheFixtureBytes() throws {
        let tensors = try VectorParity.readSafetensorsF32(
            url: fixture("scaled-a"))
        #expect(Set(tensors.keys) == ["layer_0", "layer_1", "layer_2", "layer_3"])
        // scaled-a layer 2 = 3·[1…8] (generate.py) — exact float32 integers.
        #expect(tensors["layer_2"] == [3, 6, 9, 12, 15, 18, 21, 24])
    }

    @Test func safetensorsReaderRefusesNonF32() throws {
        // Hand-build a tiny F16 safetensors file: reader must refuse loudly.
        var header = Data(
            #"{"t":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#.utf8)
        while header.count % 8 != 0 { header.append(0x20) }  // pad like writers do
        var file = Data()
        var length = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &length) { file.append(contentsOf: $0) }
        file.append(header)
        file.append(Data([0, 0, 0, 0]))
        let url = FileManager.default.temporaryDirectory
            .appending(component: "parity-f16-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: VectorParity.ParityError.self) {
            try VectorParity.readSafetensorsF32(url: url)
        }
    }
}

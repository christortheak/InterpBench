import Foundation
import Testing

@testable import SteeringKit

/// The PCA path's NUMBERS, checked against bytes the PYTHON engine produced
/// (audit finding 6).
///
/// Until 2026-08-27 the two engines' PCA agreed only by construction — same
/// algorithm, same deterministic starts, same float32 — and nothing checked
/// it. The Gram power-iteration is exactly the kind of code where a
/// "harmless" refactor on one side (a different start, an SVD swap, a changed
/// convergence test) silently produces a DIFFERENT unit vector, with the same
/// norm and a plausible cosine, on artifacts nobody re-derives.
///
/// The direction is the point, as with every fixture here: a Swift test that
/// hand-writes what it believes the server computes pins the Swift author's
/// belief, not the server's behavior. Regenerate with
/// `scripts/regenerate-cross-engine-fixtures.py`; a fixture diff means one
/// engine's PCA changed, and the review has to decide deliberately whether the
/// other should follow.
@Suite struct PairedDifferencePCACrossEngineTests {

    /// Tight on purpose: both engines compute in float32 with the identical
    /// iteration, so the only difference that should exist is last-bit
    /// accumulation order. Anything larger is a real divergence.
    static let tolerance: Float = 1e-5

    struct Case {
        let label: String
        let rows: [[Float]]
        let expected: [Float]
    }

    struct DirectionCase {
        let label: String
        let positive: [[Float]]
        let negative: [[Float]]
        let meanDifference: [Float]
        let pairedDifferencePCA: [Float]
    }

    static func load() throws -> (
        threshold: Float, components: [Case], directions: [DirectionCase]
    ) {
        // Repo root from this file's compile-time path — the same seam
        // `RepEReaderTests` uses for committed data files.
        let url = URL(filePath: #filePath)
            .deletingLastPathComponent()  // …/Tests/SteeringKitTests
            .deletingLastPathComponent()  // …/Tests
            .appending(components: "Fixtures", "cross-engine",
                       "paired-difference-pca.json")
        let root = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [String: Any])
        func floats(_ any: Any?) throws -> [Float] {
            try #require(any as? [Double]).map(Float.init)
        }
        func matrix(_ any: Any?) throws -> [[Float]] {
            try (try #require(any as? [[Double]])).map { $0.map(Float.init) }
        }
        let components = try (try #require(root["principalComponents"] as? [[String: Any]]))
            .map { entry in
                Case(
                    label: try #require(entry["label"] as? String),
                    rows: try matrix(entry["rows"]),
                    expected: try floats(entry["firstPrincipalComponent"]))
            }
        let directions = try (try #require(root["directions"] as? [[String: Any]]))
            .map { entry in
                DirectionCase(
                    label: try #require(entry["label"] as? String),
                    positive: try matrix(entry["positive"]),
                    negative: try matrix(entry["negative"]),
                    meanDifference: try floats(entry["meanDifference"]),
                    pairedDifferencePCA: try floats(entry["pairedDifferencePCA"]))
            }
        return (
            Float(try #require(root["degenerateStartRelativeThreshold"] as? Double)),
            components, directions)
    }

    private func expectClose(
        _ actual: [Float], _ expected: [Float], _ label: String
    ) {
        #expect(actual.count == expected.count, "\(label): dimension")
        for (index, pair) in zip(actual, expected).enumerated() {
            #expect(
                abs(pair.0 - pair.1) <= Self.tolerance,
                "\(label)[\(index)]: \(pair.0) vs \(pair.1)")
        }
    }

    @Test func firstPrincipalComponentMatchesTheServer() throws {
        let fixture = try Self.load()
        #expect(!fixture.components.isEmpty)
        for testCase in fixture.components {
            let component = try SteeringVectorMath.firstPrincipalComponent(
                of: testCase.rows)
            expectClose(component, testCase.expected, testCase.label)
            // Whatever else drifts, the answer is a unit vector.
            #expect(abs(SteeringVectorMath.l2Norm(component) - 1) <= Self.tolerance)
        }
    }

    /// `alternating` and `three-row-ramp-degenerate` are the cases the third
    /// (alternating ±) power-iteration start exists for: the uniform start is
    /// exactly orthogonal to their dominant eigenvector, and for the
    /// three-row case the index ramp is too. Before the degenerate-start
    /// guard became relative, the iteration rode float noise from there.
    @Test func degenerateStartsStillReachTheRightComponent() throws {
        let fixture = try Self.load()
        #expect(fixture.threshold == SteeringVectorMath.degenerateStartRelativeThreshold)
        for label in ["alternating", "three-row-ramp-degenerate"] {
            let testCase = try #require(fixture.components.first { $0.label == label })
            let component = try SteeringVectorMath.firstPrincipalComponent(
                of: testCase.rows)
            expectClose(component, testCase.expected, label)
        }
    }

    @Test func pairedDifferencePCADirectionMatchesTheServer() throws {
        let fixture = try Self.load()
        #expect(!fixture.directions.isEmpty)
        for testCase in fixture.directions {
            let meanDiff = try SteeringVectorMath.meanDifference(
                positive: testCase.positive, negative: testCase.negative)
            expectClose(meanDiff, testCase.meanDifference, "\(testCase.label) meanDiff")
            let direction = try SteeringVectorMath.direction(
                positive: testCase.positive, negative: testCase.negative,
                method: .pairedDifferencePCA)
            expectClose(
                direction, testCase.pairedDifferencePCA, "\(testCase.label) direction")
            // The family's own deliberate departure: the unit PC is scaled to
            // the mean difference's norm so α means the same under either
            // method. Asserted here so a fixture regeneration cannot quietly
            // drop it.
            #expect(
                abs(
                    SteeringVectorMath.l2Norm(direction)
                        - SteeringVectorMath.l2Norm(meanDiff))
                    <= Self.tolerance * max(1, SteeringVectorMath.l2Norm(meanDiff)))
        }
    }
}

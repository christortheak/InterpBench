import Foundation
import Testing

@testable import SteeringKit

/// The PC1 power iteration's CONVERGENCE HEALTH — the diagnostic added by the
/// 2026-08-28 audit (F5), checked against the same committed rows the Python
/// engine ran.
///
/// The finding: ≤200 iterations in float32 with a max-abs-delta tolerance, no
/// residual check and no convergence flag, so a near-degenerate spectrum
/// returned a wrong PC1 silently and deterministically — the audit reproduced
/// |cos| = 0.148 against the TRUE second eigenvector at an eigenvalue ratio of
/// 0.9945, with nothing in the artifact able to say so.
///
/// The fix is warn-and-stamp, never refuse: the direction is deterministic and
/// mirrored across engines, so a near-tied spectrum is a fact about the DATA
/// that the artifact should record, not malformed input. The COMPONENT is
/// unchanged — `PairedDifferencePCACrossEngineTests` pins that, unmodified.
@Suite struct PowerIterationDiagnosticTests {

    struct IterationCase {
        let label: String
        let rows: [[Float]]
        let converged: Bool
        let illConditioned: Bool
        let relativeResidualUpperBound: Float
    }

    static func load() throws -> (
        threshold: Float, maxIterations: Int, deltaTolerance: Float,
        warningExample: (residual: Float, iterations: Int, message: String),
        cases: [IterationCase]
    ) {
        let url = URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(components: "Fixtures", "cross-engine",
                       "paired-difference-pca.json")
        let root = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [String: Any])
        let block = try #require(root["powerIteration"] as? [String: Any])
        let example = try #require(block["warningExample"] as? [String: Any])
        let cases = try (try #require(block["cases"] as? [[String: Any]])).map { entry in
            IterationCase(
                label: try #require(entry["label"] as? String),
                rows: try (try #require(entry["rows"] as? [[Double]]))
                    .map { $0.map(Float.init) },
                converged: try #require(entry["converged"] as? Bool),
                illConditioned: try #require(entry["illConditioned"] as? Bool),
                relativeResidualUpperBound: Float(
                    try #require(entry["relativeResidualUpperBound"] as? Double)))
        }
        return (
            Float(try #require(block["residualWarnThreshold"] as? Double)),
            try #require(block["maxIterations"] as? Int),
            Float(try #require(block["deltaTolerance"] as? Double)),
            (
                Float(try #require(example["relativeResidual"] as? Double)),
                try #require(example["iterations"] as? Int),
                try #require(example["message"] as? String)
            ),
            cases)
    }

    @Test func constantsMatchTheServer() throws {
        let loaded = try Self.load()
        #expect(loaded.threshold == SteeringVectorMath.powerIterationResidualWarnThreshold)
        #expect(loaded.maxIterations == SteeringVectorMath.powerIterationMaxIterations)
        #expect(loaded.deltaTolerance == SteeringVectorMath.powerIterationDeltaTolerance)
    }

    /// The message is a cross-engine twin literal: same diagnostic in, same
    /// characters out, so a wording change is a deliberate two-engine edit.
    @Test func theWarningIsATwinLiteral() throws {
        let example = try Self.load().warningExample
        let diagnostic = SteeringVectorMath.PowerIterationDiagnostic(
            relativeResidual: example.residual, iterations: example.iterations,
            converged: false)
        #expect(diagnostic.illConditioned)
        #expect(SteeringVectorMath.powerIterationWarning(diagnostic) == example.message)
    }

    @Test func diagnosisAgreesWithTheServerOnTheSameRows() throws {
        let (threshold, maxIterations, _, _, cases) = try Self.load()
        for fixture in cases {
            let fitted = try SteeringVectorMath.firstPrincipalComponentWithDiagnostic(
                of: fixture.rows)
            let diagnostic = fitted.diagnostic
            #expect(diagnostic.converged == fixture.converged,
                    "convergence flag diverged for \(fixture.label)")
            #expect(diagnostic.illConditioned == fixture.illConditioned,
                    "ill-conditioned verdict diverged for \(fixture.label)")
            #expect(
                diagnostic.relativeResidual < fixture.relativeResidualUpperBound,
                Comment(rawValue: "residual \(diagnostic.relativeResidual) above "
                    + "the server's order of magnitude for \(fixture.label)"))
            #expect(diagnostic.iterations > 0)
            #expect(diagnostic.iterations <= maxIterations)
            #expect(
                diagnostic.illConditioned == (diagnostic.relativeResidual > threshold))
            // The COMPONENT is untouched by the diagnostic.
            let plain = try SteeringVectorMath.firstPrincipalComponent(of: fixture.rows)
            #expect(plain == fitted.component)
        }
    }

    /// The case that rules out thresholding `converged` instead of the
    /// residual: a ~5% sample eigengap uses the whole iteration budget without
    /// meeting a 1e-7 float32 delta, and is still accurate to six figures.
    @Test func hittingTheIterationCapIsNotByItselfADefect() throws {
        let cases = try Self.load().cases
        let healthy = try #require(cases.first { $0.label == "five-percent-eigengap" })
        #expect(!healthy.converged)
        #expect(!healthy.illConditioned)
        let diagnostic = try SteeringVectorMath.firstPrincipalComponentWithDiagnostic(
            of: healthy.rows).diagnostic
        #expect(!diagnostic.converged)
        #expect(!diagnostic.illConditioned)
    }

    @Test func theWarningSinkFiresOnlyForAnIllConditionedSpectrum() throws {
        let cases = try Self.load().cases
        let messages = MessageBox()
        SteeringVectorMath.powerIterationWarningSink = { messages.append($0) }
        defer { SteeringVectorMath.powerIterationWarningSink = nil }

        let healthy = try #require(cases.first { $0.label == "well-separated" })
        _ = try SteeringVectorMath.firstPrincipalComponentWithDiagnostic(of: healthy.rows)
        #expect(messages.all.isEmpty)

        let degenerate = try #require(cases.first { $0.label == "near-degenerate" })
        _ = try SteeringVectorMath.firstPrincipalComponentWithDiagnostic(
            of: degenerate.rows)
        #expect(messages.all.count == 1)
        #expect(messages.all.first?.contains("nearly tied") == true)
    }

    @Test func theStampRoundTripsThroughJSON() throws {
        let diagnostic = SteeringVectorMath.PowerIterationDiagnostic(
            relativeResidual: 0.006171, iterations: 200, converged: false)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(diagnostic)
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        // The server's `to_dict` key set, exactly.
        #expect(Set(object.keys) == [
            "converged", "illConditioned", "iterations", "maxIterations",
            "relativeResidual",
        ])
        #expect(object["illConditioned"] as? Bool == true)
        let decoded = try JSONDecoder().decode(
            SteeringVectorMath.PowerIterationDiagnostic.self, from: data)
        #expect(decoded == diagnostic)
    }
}

/// Collects sink callbacks from the synchronous iteration under test.
private final class MessageBox: @unchecked Sendable {
    private var messages: [String] = []
    private let lock = NSLock()
    func append(_ message: String) {
        lock.lock()
        defer { lock.unlock() }
        messages.append(message)
    }
    var all: [String] {
        lock.lock()
        defer { lock.unlock() }
        return messages
    }
}

import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The web client must CONSUME the engine's α default, never restate it.
///
/// The native Playground resolves a freshly-selected vector's α through
/// `SlotAlphaDefault` (promoted strength → 1.0 norm unit → labelled raw
/// fallback). The browser client is a thin twin over the Swift WebServer, so
/// the same decision reaches it serialized — `POST
/// /api/steering/select-vector` applies it server-side and the state payload
/// carries it as `alphaDefault` — and `web/index.html` renders it. A literal
/// α default in the JS (the historical `alpha: 2`) is exactly the unit-blind
/// bug the decision table replaced, reintroduced on one surface.
@Suite struct WebClientAlphaDefaultTests {

    private static var repoRoot: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// `web/index.html` is a research-tree fixture: the release tree
    /// deliberately ships NO committed web build surface (the CI lane's own
    /// rule — a cold clone must produce it), so these checks predicate on
    /// the file's presence per the ResearchTreeFixtures discipline and skip
    /// with a reason on a foreign tree. Caught by the clean-clone rehearsal
    /// (2026-08-20). The skip is a real one (`.enabled(if:)`), so the run
    /// summary counts it as skipped rather than passed.
    private static var webClientURL: URL { repoRoot.appending(path: "web/index.html") }

    static var hasWebClient: Bool { FileManager.default.fileExists(atPath: webClientURL.path) }

    static let noWebClient: Comment =
        "web/index.html is not in this checkout, so the web client was not checked"

    private func webClientSource() throws -> String {
        try String(contentsOf: Self.webClientURL, encoding: .utf8)
    }

    /// No slot in the web client is born with, or falls back to, a numeric α
    /// literal — `alpha: 2`, `alpha ?? 2`, `last.alpha : 2` and kin.
    @Test(.enabled(if: hasWebClient, noWebClient))
    func webClientCarriesNoHardcodedAlphaDefault() throws {
        let source = try webClientSource()
        let literalDefault = try Regex("alpha\\s*(:|\\?\\?|=)\\s*-?[0-9]")
        #expect(
            source.firstMatch(of: literalDefault) == nil,
            "web/index.html hardcodes an α default; vector selection must go through /api/steering/select-vector so SlotAlphaDefault decides server-side")
    }

    /// The client actually uses the engine paths that replace the literal:
    /// selection and slot-addition are server verbs, and the served decision
    /// is rendered.
    @Test(.enabled(if: hasWebClient, noWebClient))
    func webClientConsumesTheServedDecision() throws {
        let source = try webClientSource()
        #expect(source.contains("/api/steering/select-vector"))
        #expect(source.contains("/api/steering/slots/add"))
        #expect(source.contains("S.alphaDefault"))
    }

    // MARK: Serialization — the DTO carries the decision verbatim

    @Test func alphaDefaultDTOSerializesTheDecisionVerbatim() {
        let decision = SlotAlphaDefault.decide(
            SlotAlphaDefault.ArtifactFacts(
                artifactID: "runs/20260820T000000Z-concept-fear/fear",
                layer: 18))
        let dto = StateDTO.AlphaDefaultDTO(decision)

        // No denominator → the labelled raw fallback, backfill hint included.
        #expect(dto.alpha == SlotAlphaDefault.rawUnitsDefault)
        #expect(dto.units == "raw")
        #expect(dto.label == decision.alphaLabel)
        #expect(dto.rationale == decision.rationale)
        #expect(dto.conventionNote == nil)
        #expect(dto.backfillHint == decision.backfillHint)
        #expect(dto.backfillHint?.contains("vectors backfill-norms") == true)
    }

    @Test func alphaDefaultDTOKeepsNormUnitsAndConvention() {
        let decision = SlotAlphaDefault.decide(
            SlotAlphaDefault.ArtifactFacts(
                artifactID: "runs/20260820T000000Z-concept-fear/fear",
                layer: 18,
                residualNormAtLayer: 42,
                residualNormConvention: ResidualNormConvention.perTextMean))
        let dto = StateDTO.AlphaDefaultDTO(decision)

        #expect(dto.alpha == SlotAlphaDefault.normUnitsDefault)
        #expect(dto.units == "normUnits")
        #expect(dto.conventionNote == decision.conventionNote)
        #expect(dto.backfillHint == nil)
    }
}

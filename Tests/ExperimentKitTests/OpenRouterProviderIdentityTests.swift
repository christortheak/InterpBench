import Foundation
import Testing

@testable import ExperimentKit

/// OpenRouter provider identity from the committed fixture (2026-07-24).
/// Cross-engine twin of `Server/tests/test_provider_identity.py` — both
/// suites assert the SAME property over every row of the SAME file, so
/// parity holds by construction instead of by two hand-listed case sets
/// that drift.
struct OpenRouterProviderIdentityTests {

    private struct Fixture: Decodable {
        struct Provider: Decodable {
            let name: String
            let slug: String
        }
        let schemaVersion: Int
        let providerCount: Int
        let source: String
        let providers: [Provider]
    }

    private func fixture() throws -> Fixture {
        let root = URL(filePath: #filePath)
            .deletingLastPathComponent()  // ExperimentKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
        let url = root.appending(
            path: OpenRouterProviderIdentity.fixtureRelativePath)
        return try JSONDecoder().decode(
            Fixture.self, from: try Data(contentsOf: url))
    }

    @Test func fixtureLoadsAndIsSubstantial() throws {
        // A silently-missing fixture degrades canonicalization to plain
        // lowercasing — which still fails closed, but resurrects the Vertex
        // bug. Assert it actually loaded rather than trusting the fallback.
        let fixture = try fixture()
        #expect(fixture.schemaVersion == 1)
        #expect(fixture.source.hasSuffix("/api/v1/providers"))
        #expect(fixture.providerCount == fixture.providers.count)
        #expect(fixture.providers.count >= 50)
        #expect(OpenRouterProviderIdentity.knownSpellingCount >= fixture.providers.count)
    }

    @Test func everyDisplayNameCanonicalizesToItsSlug() throws {
        for provider in try fixture().providers {
            #expect(
                OpenRouterProviderIdentity.canonical(provider.name)
                    == provider.slug,
                "display name '\(provider.name)'")
            #expect(
                OpenRouterProviderIdentity.canonical(provider.slug)
                    == provider.slug,
                "slug '\(provider.slug)'")
        }
    }

    @Test func vertexDisplayNameMatchesItsRoutingSlug() {
        // THE REGRESSION. OpenRouter's display name for Vertex is "Google",
        // not "Google Vertex". The old hand-written alias list mapped
        // "google vertex" -> "google-vertex" and knew nothing about
        // "Google", so a judgment correctly served by a pinned
        // google-vertex reported "Google", canonicalized to "google", and
        // was refused as off-pin — a correct run dying on a spelling.
        #expect(
            OpenRouterProviderIdentity.equivalent("Google", "google-vertex"))
        // ... and Vertex is still NOT AI Studio: different endpoints that
        // can serve one model at different quantizations.
        #expect(
            OpenRouterProviderIdentity.equivalent(
                "Google AI Studio", "google-ai-studio"))
        #expect(
            !OpenRouterProviderIdentity.equivalent(
                "Google", "Google AI Studio"))
    }

    @Test func namesNoSlugifyRuleWouldProduce() {
        // The reason a fetched table beats a clever transform.
        #expect(OpenRouterProviderIdentity.canonical("Moonshot AI") == "moonshotai")
        #expect(OpenRouterProviderIdentity.canonical("Z.AI") == "z-ai")
        #expect(OpenRouterProviderIdentity.canonical("Sakana AI") == "sakana")
        #expect(OpenRouterProviderIdentity.canonical("InferenceNet") == "inference-net")
    }

    @Test func unknownSpellingsStillFailClosed() {
        #expect(OpenRouterProviderIdentity.canonical("NotAProvider") == "notaprovider")
        #expect(
            !OpenRouterProviderIdentity.equivalent("NotAProvider", "deepinfra"))
        #expect(OpenRouterProviderIdentity.canonical("") == "")
        #expect(OpenRouterProviderIdentity.canonical("   ") == "")
    }

    @Test func caseAndWhitespaceAreNotAMismatch() {
        #expect(OpenRouterProviderIdentity.canonical("  DEEPINFRA  ") == "deepinfra")
        #expect(OpenRouterProviderIdentity.equivalent("deepinfra", "DeepInfra"))
    }

    // MARK: The verified-provider stamp

    @Test func inlineEvaluateStampsTheVerifiedServedByNotThePin() {
        // Fixed 2026-07-24: this path stamped `judge.provider` verbatim, so
        // one judge recorded "Google AI Studio" from a Mac inline evaluate
        // and "google-ai-studio" from the server — and recorded the
        // REQUESTED provider rather than the verified one.
        let judge = ExperimentTasks.ResolvedJudge(
            name: "or-j", kind: "openrouter", model: "google/gemini-3.6-flash",
            provider: "google-ai-studio")
        let verdict = PairedJudgeResponse(
            winner: "A", confidence: 0.9, briefReason: "r",
            provider: "Google AI Studio")
        #expect(
            ExperimentTasks.verifiedJudgeProvider(judge: judge, judgment: verdict)
                == "google-ai-studio")
    }

    @Test func nonOpenRouterJudgesCarryNoProvider() {
        let local = ExperimentTasks.ResolvedJudge(
            name: "local-j", kind: "local", model: "org/m")
        let verdict = PairedJudgeResponse(
            winner: "tie", confidence: 0.5, briefReason: "r")
        #expect(
            ExperimentTasks.verifiedJudgeProvider(judge: local, judgment: verdict)
                == nil)
    }

    @Test func olderClientVerdictFallsBackToTheCanonicalPin() {
        // A verdict from a client that predates the served-by pass-through
        // carries no provider. The pin was still verified at call time, so
        // recording its canonical form is honest — and canonical, so it
        // matches what the server would have written.
        let judge = ExperimentTasks.ResolvedJudge(
            name: "or-j", kind: "openrouter", model: "m", provider: "Google")
        let verdict = PairedJudgeResponse(
            winner: "B", confidence: 0.7, briefReason: "r")
        #expect(
            ExperimentTasks.verifiedJudgeProvider(judge: judge, judgment: verdict)
                == "google-vertex")
    }

    @Test func verdictDecodesWithoutAProviderKey() throws {
        // Verdict payloads written before 2026-07-24 carry no `provider`; a
        // synthesized decoder would reject them outright.
        let json = #"""
            {"winner": "A", "confidence": 0.8, "brief_reason": "clearer"}
            """#
        let verdict = try JSONDecoder().decode(
            PairedJudgeResponse.self, from: Data(json.utf8))
        #expect(verdict.winner == "A")
        #expect(verdict.provider == nil)
    }
}

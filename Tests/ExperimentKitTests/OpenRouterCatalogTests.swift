import Foundation
import Testing

@testable import ExperimentKit

/// OpenRouter catalogue discovery + provider preflight (2026-07-24).
/// Cross-engine twin of `Server/tests/test_provider_preflight.py` — same
/// rules, same asymmetry. Every test injects a fetcher; nothing here touches
/// the network.
struct OpenRouterCatalogTests {

    private func payload(_ endpoints: [(String, String, Int, Int)]) -> Data {
        let rows = endpoints.map { name, quantization, status, ctx in
            """
            {"provider_name": "\(name)", "quantization": "\(quantization)",
             "status": \(status), "context_length": \(ctx),
             "tag": "\(name.lowercased())/\(quantization)"}
            """
        }
        return Data("""
            {"data": {"id": "org/m", "endpoints": [\(rows.joined(separator: ","))]}}
            """.utf8)
    }

    private func fetcher(
        _ data: Data, status: Int = 200
    ) -> OpenRouterCatalog.Fetcher {
        { url in
            (data, HTTPURLResponse(
                url: url, statusCode: status,
                httpVersion: nil, headerFields: nil)!)
        }
    }

    private func failing() -> OpenRouterCatalog.Fetcher {
        { _ in throw URLError(.notConnectedToInternet) }
    }

    // MARK: Discovery

    @Test func listsEndpointsWithCanonicalSlugsAndQuantization() async throws {
        let endpoints = try await OpenRouterCatalog.endpoints(
            forModel: "google/gemma-3-27b-it",
            fetch: fetcher(payload([("DeepInfra", "fp8", 0, 131072),
                                    ("Novita", "bf16", -5, 98304)])))
        #expect(endpoints.map(\.provider) == ["deepinfra", "novita"])
        // The display name is kept: it is what a judgment response reports.
        #expect(endpoints.map(\.providerName) == ["DeepInfra", "Novita"])
        #expect(endpoints.map(\.quantization) == ["fp8", "bf16"])
        #expect(endpoints[0].summary == "deepinfra — fp8, 131072 ctx")
        #expect(endpoints[1].summary.contains("degraded"))
    }

    @Test func vertexDisplayNameBecomesItsRoutingSlug() async throws {
        // The catalogue reports "Google"; the pin has to be "google-vertex".
        let endpoints = try await OpenRouterCatalog.endpoints(
            forModel: "org/m",
            fetch: fetcher(payload([("Google", "bf16", 0, 1_000_000)])))
        #expect(endpoints.map(\.provider) == ["google-vertex"])
    }

    @Test func malformedModelIDRefusesBeforeAnyRequest() async {
        for bad in ["gemma-3-27b-it", "a/b/c", "", "/", "org/"] {
            await #expect(throws: PairedJudgeError.self) {
                // The failing fetcher would throw a DIFFERENT error if the
                // guard let a request through.
                _ = try await OpenRouterCatalog.endpoints(
                    forModel: bad, fetch: failing())
            }
        }
    }

    @Test func unknownModelSaysSoSpecifically() async {
        await #expect(throws: PairedJudgeError.self) {
            _ = try await OpenRouterCatalog.endpoints(
                forModel: "org/nope",
                fetch: fetcher(Data("{}".utf8), status: 404))
        }
    }

    // MARK: Preflight

    @Test func pinThatServesTheModelPasses() async {
        let result = await OpenRouterCatalog.preflight(
            model: "org/m", provider: "DeepInfra",
            fetch: fetcher(payload([("DeepInfra", "fp8", 0, 131072)])))
        #expect(result.problem == nil)
        #expect(result.checked)
        #expect(result.warnings.isEmpty)
    }

    @Test func displayNamePinMatchesTheSlugEndpoint() async {
        let result = await OpenRouterCatalog.preflight(
            model: "org/m", provider: "google-vertex",
            fetch: fetcher(payload([("Google", "bf16", 0, 1000)])))
        #expect(result.problem == nil)
        #expect(result.checked)
    }

    @Test func wrongPinRefusesAndNamesTheAlternatives() async {
        let result = await OpenRouterCatalog.preflight(
            model: "org/m", provider: "google-ai-studio",
            fetch: fetcher(payload([("DeepInfra", "fp8", 0, 1000),
                                    ("Nebius", "fp8", 0, 1000)])))
        #expect(result.checked)
        let problem = result.problem ?? ""
        #expect(problem.contains("does not serve"))
        // Naming what IS available is the difference between a dead end and
        // a fix — the error is the only place to learn the vocabulary.
        #expect(problem.contains("deepinfra, nebius"))
    }

    @Test func unreachableCatalogueWarnsButNeverRefuses() async {
        // THE asymmetry: an offline laptop or an air-gapped compute node
        // must not make a study unrunnable. Call-time still refuses off-pin.
        let result = await OpenRouterCatalog.preflight(
            model: "org/m", provider: "deepinfra", fetch: failing())
        #expect(result.problem == nil)
        #expect(!result.checked)
        #expect(result.warnings.contains { $0.contains("UNVERIFIED, not wrong") })
    }

    @Test func emptyPinRefusesWithoutALookup() async {
        let result = await OpenRouterCatalog.preflight(
            model: "org/m", provider: "  ", fetch: failing())
        #expect(result.problem?.contains("no pinned provider") == true)
        #expect(!result.checked)
    }

    @Test func multipleQuantizationsWarnBecauseThatIsWhatThePinIsFor() async {
        // The pin exists because quantization changes verdicts. A provider
        // serving one model at two quantizations is not fully pinned by
        // provider alone — say so rather than implying more rigour than the
        // pin delivers.
        let result = await OpenRouterCatalog.preflight(
            model: "org/m", provider: "deepinfra",
            fetch: fetcher(payload([("DeepInfra", "fp8", 0, 1000),
                                    ("DeepInfra", "bf16", 0, 1000)])))
        #expect(result.problem == nil)
        #expect(result.warnings.contains { $0.contains("more than one quantization") })
    }

    @Test func allEndpointsDegradedWarns() async {
        let result = await OpenRouterCatalog.preflight(
            model: "org/m", provider: "deepinfra",
            fetch: fetcher(payload([("DeepInfra", "fp8", -5, 1000)])))
        #expect(result.problem == nil)
        #expect(result.warnings.contains { $0.contains("degraded") })
    }

    // MARK: Wired into judging (external review 2026-07-24, finding 5)

    private func judge(
        _ name: String, kind: String, model: String, provider: String? = nil
    ) -> ExperimentTasks.ResolvedJudge {
        .init(name: name, kind: kind, model: model, provider: provider)
    }

    @Test func judgingPreflightRefusesTheNamedJudge() async {
        // The check existed on this engine but was reachable only from the
        // manual Discover control, so Swift-side judging still learned
        // about a wrong pin from an off-pin refusal mid-run — after the
        // paid call — which the server had already stopped doing.
        await #expect(throws: ExperimentError.self) {
            try await ExperimentTasks.preflightOpenRouterJudges(
                [judge("or-j", kind: "openrouter", model: "org/m",
                       provider: "google-ai-studio")],
                fetch: fetcher(payload([("DeepInfra", "fp8", 0, 1000)])))
        }
    }

    @Test func judgingPreflightPassesAPinThatServes() async throws {
        try await ExperimentTasks.preflightOpenRouterJudges(
            [judge("or-j", kind: "openrouter", model: "org/m",
                   provider: "DeepInfra")],
            fetch: fetcher(payload([("DeepInfra", "fp8", 0, 1000)])))
    }

    @Test func judgingPreflightIgnoresNonOpenRouterJudges() async throws {
        // `failing()` would throw if anything issued a request.
        try await ExperimentTasks.preflightOpenRouterJudges(
            [judge("local-j", kind: "local", model: "org/study"),
             judge("claude-j", kind: "claude", model: "claude-opus-4-8")],
            fetch: failing())
    }

    @Test func judgingPreflightProceedsWhenTheCatalogueIsUnreachable() async throws {
        // The asymmetry, at the wiring level: an offline laptop or an
        // air-gapped node must not make a study unrunnable.
        try await ExperimentTasks.preflightOpenRouterJudges(
            [judge("or-j", kind: "openrouter", model: "org/m",
                   provider: "deepinfra")],
            fetch: failing())
    }

    @Test func emptyCatalogueIsUncheckedNotWrong() async {
        let result = await OpenRouterCatalog.preflight(
            model: "org/m", provider: "deepinfra",
            fetch: fetcher(Data(#"{"data": {"endpoints": []}}"#.utf8)))
        #expect(result.problem == nil)
        #expect(!result.checked)
    }
}

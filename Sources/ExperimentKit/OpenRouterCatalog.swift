import Foundation

/// OpenRouter's PUBLIC model catalogue — which providers serve a model, at
/// what quantization (2026-07-24). Python twin:
/// `paired_judge.openrouter_model_endpoints` / `preflight_openrouter_provider`.
///
/// Keyless by design. The catalogue needs no API key, which is what lets a
/// provider pin be *discovered* rather than typed, and *checked* before any
/// judging starts — including at freeze, where holding no judge credential is
/// the default custody posture.
///
/// Why this exists: an `openrouter` judge pins a model slug and a serving
/// provider, and a wrong provider used to surface at the first judge call —
/// mid-run, after generation was already paid for, as an off-pin refusal the
/// researcher had no vocabulary to fix. There is no way to learn the right
/// spelling from the error alone; this is that way.
public enum OpenRouterCatalog {

    public struct Endpoint: Sendable, Equatable, Identifiable {
        /// Canonical routing slug — what goes in a judge's `provider` pin.
        public let provider: String
        /// OpenRouter's display name — what a judgment response reports.
        public let providerName: String
        /// Weight quantization this endpoint serves, when OpenRouter states
        /// it. THE reason the provider pin exists: the same model slug at
        /// fp8 and at bf16 is not the same judge.
        public let quantization: String?
        public let contextLength: Int?
        /// OpenRouter's health code; 0 is healthy, negatives are degraded.
        public let status: Int?

        public var id: String { "\(provider)/\(quantization ?? "?")" }

        /// One row for a picker: "deepinfra — fp8, 131072 ctx".
        public var summary: String {
            var parts: [String] = []
            if let quantization { parts.append(quantization) }
            if let contextLength { parts.append("\(contextLength) ctx") }
            if let status, status != 0 { parts.append("degraded") }
            return parts.isEmpty ? provider : "\(provider) — \(parts.joined(separator: ", "))"
        }
    }

    public struct Preflight: Sendable, Equatable {
        /// Non-nil means the pin is WRONG on positive evidence — refuse.
        public var problem: String?
        public var warnings: [String]
        /// False when the catalogue could not be consulted. An unverified
        /// pin is not a bad pin.
        public var checked: Bool
    }

    /// Injectable fetch seam so tests never touch the network.
    public typealias Fetcher = @Sendable (URL) async throws -> (Data, URLResponse)

    private static let session: Fetcher = { url in
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        return try await URLSession.shared.data(for: request)
    }

    private struct Payload: Decodable {
        struct Data: Decodable {
            struct Endpoint: Decodable {
                let provider_name: String?
                let quantization: String?
                let context_length: Int?
                let status: Int?
            }
            let endpoints: [Endpoint]?
        }
        let data: Data?
    }

    /// The endpoints currently serving `model` (an `author/slug` id).
    public static func endpoints(
        forModel model: String, fetch: Fetcher? = nil
    ) async throws -> [Endpoint] {
        let slug = model.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let parts = slug.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
            throw PairedJudgeError(
                reason: "'\(model)' is not an OpenRouter model id — expected "
                    + "'author/slug' (e.g. 'google/gemma-3-27b-it'). Note "
                    + "that OpenRouter's id is not always the Hugging Face "
                    + "repo id.")
        }
        guard
            let url = URL(
                string: "https://openrouter.ai/api/v1/models/\(slug)/endpoints")
        else {
            throw PairedJudgeError(reason: "could not build a catalogue URL for '\(model)'")
        }
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await (fetch ?? session)(url)
        } catch {
            throw PairedJudgeError(
                reason: "could not reach OpenRouter's model catalogue: "
                    + error.localizedDescription)
        }
        if let http = response as? HTTPURLResponse {
            if http.statusCode == 404 {
                throw PairedJudgeError(
                    reason: "OpenRouter does not list a model '\(slug)' — "
                        + "check the model slug (it is OpenRouter's id, which "
                        + "is not always the same as the Hugging Face repo id)")
            }
            guard http.statusCode == 200 else {
                throw PairedJudgeError(
                    reason: "OpenRouter model catalogue failed: HTTP "
                        + "\(http.statusCode)")
            }
        }
        let payload = try JSONDecoder().decode(Payload.self, from: data)
        return (payload.data?.endpoints ?? []).compactMap { raw in
            let name = (raw.provider_name ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }
            let quantization = raw.quantization?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return Endpoint(
                provider: OpenRouterProviderIdentity.canonical(name),
                providerName: name,
                quantization: (quantization?.isEmpty ?? true) ? nil : quantization,
                contextLength: raw.context_length,
                status: raw.status)
        }
    }

    /// Check a provider pin against the live catalogue.
    ///
    /// Refuses ONLY on positive evidence (the catalogue answered and the pin
    /// is not among the serving endpoints). An unreachable catalogue warns
    /// and passes: a laptop offline on a train, or a compute node with no
    /// outbound network, must not make a study unrunnable. The call-time
    /// off-pin refusal still guarantees correctness either way — this moves
    /// discovery earlier, it does not replace the guarantee.
    public static func preflight(
        model: String, provider: String, fetch: Fetcher? = nil
    ) async -> Preflight {
        var warnings: [String] = []
        let pinned = OpenRouterProviderIdentity.canonical(provider)
        guard !pinned.isEmpty else {
            return Preflight(
                problem: "openrouter judge for '\(model)' has no pinned "
                    + "provider — an unpinned provider is not a pinned judge",
                warnings: warnings, checked: false)
        }
        let endpoints: [Endpoint]
        do {
            endpoints = try await self.endpoints(forModel: model, fetch: fetch)
        } catch {
            warnings.append(
                "could not verify the provider pin for '\(model)' "
                    + "(\(error)) — the pin is UNVERIFIED, not wrong; judging "
                    + "still refuses an off-pin response at call time")
            return Preflight(problem: nil, warnings: warnings, checked: false)
        }
        guard !endpoints.isEmpty else {
            warnings.append(
                "OpenRouter lists no serving endpoints for '\(model)' right "
                    + "now — the pin could not be checked against a live "
                    + "catalogue")
            return Preflight(problem: nil, warnings: warnings, checked: false)
        }
        let matching = endpoints.filter { $0.provider == pinned }
        guard !matching.isEmpty else {
            let available = Set(endpoints.map(\.provider)).sorted()
            return Preflight(
                problem: "provider '\(provider)' does not serve '\(model)' on "
                    + "OpenRouter. Available: \(available.joined(separator: ", ")). "
                    + "Pin one of those (they are routing slugs — Discover "
                    + "fills this in from the same catalogue).",
                warnings: warnings, checked: true)
        }
        let quantizations = Set(matching.compactMap(\.quantization)).sorted()
        if quantizations.count > 1 {
            warnings.append(
                "provider '\(pinned)' serves '\(model)' at more than one "
                    + "quantization (\(quantizations.joined(separator: ", "))) "
                    + "— the provider pin alone does not fix which one judges, "
                    + "and quantization is the reason this pin exists. "
                    + "Verdicts may vary between runs")
        }
        if matching.allSatisfy({ ($0.status ?? 0) != 0 }) {
            warnings.append(
                "every '\(pinned)' endpoint for '\(model)' is currently "
                    + "flagged degraded by OpenRouter — judging may be slow "
                    + "or fail")
        }
        return Preflight(problem: nil, warnings: warnings, checked: true)
    }
}

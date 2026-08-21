import Foundation

public struct StimulusGeneratorError: Error, CustomStringConvertible {
    public let reason: String
    public var description: String { reason }
}

/// Generates candidate contrastive stimulus pairs with Claude (Messages API
/// via URLSession — there is no official Swift SDK). Proposals are never
/// added to a concept directly: the panel shows them for human accept/reject,
/// so the committed stimulus files remain human-curated regardless of source.
public enum ClaudeStimulusGenerator {

    static let endpoint = URL(string: "https://api.anthropic.com/v1/messages")!
    static let model = "claude-opus-4-8"

    /// ANTHROPIC_API_KEY from the environment wins; falls back to the key
    /// stored in the macOS Keychain (`AnthropicKeyStore`, which silently
    /// migrates the pre-2026-07-08 plaintext UserDefaults value on first
    /// read). Every Claude credential consumer — judges, stimulus
    /// generation, sweep credential preflights, the web-client status —
    /// reads through this one accessor.
    public static var apiKey: String? {
        AnthropicKeyStore.resolve()
    }

    /// Persist (or, for empty/whitespace input, delete) the key in the
    /// macOS Keychain. The legacy plaintext UserDefaults slot is cleared
    /// either way.
    public static func saveAPIKey(_ key: String) {
        AnthropicKeyStore.save(key)
    }

    // MARK: - Request/response shapes

    private struct PairsResult: Decodable {
        struct Pair: Decodable {
            let positive: String
            let negative: String
        }
        let pairs: [Pair]
    }

    private struct APIResponse: Decodable {
        struct ContentBlock: Decodable {
            let type: String
            let text: String?
        }
        let content: [ContentBlock]
    }

    private struct APIError: Decodable {
        struct Detail: Decodable {
            let type: String
            let message: String
        }
        let error: Detail
    }

    /// JSON schema for structured output: {"pairs": [{"positive", "negative"}]}.
    private static var pairsSchema: [String: Any] { [
        "type": "object",
        "properties": [
            "pairs": [
                "type": "array",
                "items": [
                    "type": "object",
                    "properties": [
                        "positive": ["type": "string"],
                        "negative": ["type": "string"],
                    ],
                    "required": ["positive", "negative"],
                    "additionalProperties": false,
                ] as [String: Any],
            ]
        ],
        "required": ["pairs"],
        "additionalProperties": false,
    ] }

    // MARK: - Generation

    public static func generatePairs(
        concept: String, guidance: String,
        examplePositives: [String], exampleNegatives: [String], count: Int
    ) async throws -> [(positive: String, negative: String)] {
        guard let apiKey else {
            throw StimulusGeneratorError(
                reason: "no API key — set ANTHROPIC_API_KEY or save a key in "
                    + "the Compute section (stored in the macOS Keychain)")
        }

        let prompt = buildPrompt(
            concept: concept, guidance: guidance,
            examplePositives: examplePositives, exampleNegatives: exampleNegatives,
            count: count)

        let body: [String: Any] = [
            "model": model,
            "max_tokens": 16000,
            "thinking": ["type": "adaptive"],
            "output_config": ["format": ["type": "json_schema", "schema": pairsSchema]],
            "messages": [["role": "user", "content": prompt]],
        ]

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 300

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw StimulusGeneratorError(reason: "no HTTP response")
        }
        guard http.statusCode == 200 else {
            if let apiError = try? JSONDecoder().decode(APIError.self, from: data) {
                throw StimulusGeneratorError(
                    reason: "\(apiError.error.type): \(apiError.error.message)")
            }
            throw StimulusGeneratorError(reason: "HTTP \(http.statusCode)")
        }

        let decoded = try JSONDecoder().decode(APIResponse.self, from: data)
        // Thinking blocks precede the text block; structured output guarantees
        // the text block is valid JSON matching the schema.
        guard let text = decoded.content.first(where: { $0.type == "text" })?.text,
            let result = try? JSONDecoder().decode(PairsResult.self, from: Data(text.utf8))
        else {
            throw StimulusGeneratorError(reason: "response did not contain valid pairs JSON")
        }
        return result.pairs.map { ($0.positive, $0.negative) }
    }

    /// Prompt variant for use with ANY LLM via the clipboard: same
    /// methodology instructions, plus a strict output-format section so the
    /// response can be pasted straight back into the app (pair-JSONL, the
    /// import format).
    static func clipboardPrompt(
        concept: String, guidance: String,
        examplePositives: [String], exampleNegatives: [String], count: Int
    ) -> String {
        buildPrompt(
            concept: concept, guidance: guidance,
            examplePositives: examplePositives, exampleNegatives: exampleNegatives,
            count: count)
            + """


            Output format (strict): respond with ONLY the new pairs as JSON Lines — \
            one object per line, exactly {"positive": "…", "negative": "…"} — no \
            prose, no numbering, no code fences, no trailing commentary. The output \
            is machine-parsed: it will be pasted directly back into the SteerLab \
            app's stimulus box (or saved as a .jsonl file and imported).
            """
    }

    static func buildPrompt(
        concept: String, guidance: String,
        examplePositives: [String], exampleNegatives: [String], count: Int
    ) -> String {
        var examples = ""
        for (index, (positive, negative)) in zip(examplePositives, exampleNegatives).enumerated()
        {
            examples += "\(index + 1). positive: \(positive)\n   negative: \(negative)\n"
        }
        let guidanceLine = guidance.isEmpty ? "" : "\nAdditional guidance: \(guidance)\n"

        // A fresh concept has no pairs yet: the prompt must read coherently
        // with zero examples, so both the examples section and the "do not
        // reuse the examples' scenarios" clause are omitted when empty.
        let varietyRule =
            examples.isEmpty
            ? "- Vary topics, settings, and protagonists widely across pairs."
            : "- Vary topics, settings, and protagonists widely across pairs; do not reuse "
                + "the scenarios of the examples below."
        let closing =
            examples.isEmpty
            ? "Generate exactly \(count) new pairs."
            : """
            Existing pairs in the set (for style; do NOT duplicate their scenarios):
            \(examples)
            Generate exactly \(count) new pairs.
            """

        return """
            You are helping build a contrastive stimulus set for activation-steering \
            research (contrastive activation addition). The concept under study is: \
            "\(concept)".
            \(guidanceLine)
            Each stimulus pair consists of one "positive" sentence that vividly DEPICTS \
            the concept through situation, behavior, and sensory detail, and one \
            "negative" sentence that is content-matched — same topic, same setting, \
            similar length and syntax — but does NOT express the concept. The pair \
            should differ as purely as possible in the concept alone.

            Strict requirements:
            - Depict the concept implicitly; NEVER name the concept or use its obvious \
            synonyms in the text (the direction must come from meaning, not keywords).
            - Keep each sentence a single sentence of roughly 10-22 words.
            \(varietyRule)
            - Avoid confounds: the positive must not be systematically more negative in \
            valence, more arousing, longer, or more emotionally worded than the \
            negative requires — except as inherent to the concept itself.

            \(closing)
            """
    }
}

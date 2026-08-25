import Foundation
import MLXLMCommon
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// The rendering VOICE and the content-coordinate reading positions, checked
/// against bytes the PYTHON engine produced.
///
/// The direction is the point (B1): a Swift test that hand-writes what it
/// believes the server emits pins the Swift author's belief, not the server's
/// behavior. Regenerate with `scripts/regenerate-cross-engine-fixtures.py`; a
/// fixture diff means one engine changed its contract, and the review has to
/// decide deliberately whether the other should follow.
///
/// Three contracts travel in the fixture:
///
/// 1. what a DECLARATION canonicalizes to — including the two spellings that
///    must canonicalize to nothing at all (`{"mode":"raw"}` and an explicit
///    `"voice":"user"`), which is the hash-compatibility rule as data;
/// 2. what each reading position contributes to a RECIPE IDENTITY;
/// 3. where each position RESOLVES on a fixed token sequence, the content
///    mask included — so the two template maps are compared on arithmetic
///    rather than on prose.
@Suite struct ExtractionRenderingCrossEngineTests {

    // MARK: - fixture plumbing

    struct Fixture {
        let tokens: [Int]
        let contentIndices: [Int]
        let renderings: [[String: Any]]
        let positions: [[String: Any]]
        let resolutions: [[String: Any]]
        let refusals: [String: String]
    }

    static func load() throws -> Fixture {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "extraction-rendering-and-positions.json")
        let root = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [String: Any])
        return Fixture(
            tokens: try #require(root["tokens"] as? [Int]),
            contentIndices: try #require(root["contentIndices"] as? [Int]),
            renderings: try #require(root["renderings"] as? [[String: Any]]),
            positions: try #require(root["positions"] as? [[String: Any]]),
            resolutions: try #require(root["resolutions"] as? [[String: Any]]),
            refusals: try #require(root["refusals"] as? [String: String]))
    }

    /// A tokenizer that answers the template map's questions with the ids the
    /// fixture was generated against — markers, role words, and the newline
    /// that ends a role tag. No model, no download.
    struct FixtureTokenizer: MLXLMCommon.Tokenizer {
        static let vocabulary = [
            "<bos>": 2, "<start_of_turn>": 105, "<end_of_turn>": 106,
            "\n": 107, "model": 108, "user": 109, "<unk>": 3,
        ]

        var bosToken: String? { "<bos>" }
        var eosToken: String? { "<end_of_turn>" }
        var unknownToken: String? { "<unk>" }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { convertIdToToken($0) ?? "" }.joined()
        }

        func convertTokenToId(_ token: String) -> Int? {
            Self.vocabulary[token]
        }

        func convertIdToToken(_ id: Int) -> String? {
            Self.vocabulary.first { $0.value == id }?.key ?? "tok\(id)"
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    // MARK: - 1. declarations canonicalize identically on both engines

    @Test func everyDeclarationCanonicalizesTheWayThePythonEngineDoes() throws {
        let fixture = try Self.load()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        for entry in fixture.renderings {
            let label = try #require(entry["label"] as? String)
            let stamp = entry["stamp"] as? [String: Any]

            // The STAMP: decode the server's bytes, re-encode them, and
            // require the round trip to be byte-identical — this is the
            // artifact contract a Mac reads a server-extracted vector with.
            if let stamp {
                let stampData = try JSONSerialization.data(
                    withJSONObject: stamp, options: [.sortedKeys])
                let decoded = try JSONDecoder().decode(
                    ExtractionRendering.self, from: stampData)
                #expect(String(decoding: try encoder.encode(decoded), as: UTF8.self)
                        == String(decoding: stampData, as: UTF8.self),
                        "stamp round trip drifted for \(label)")
                #expect(decoded.label == entry["humanLabel"] as? String, "\(label)")
                // The IDENTITY FRAGMENT this engine would hash for it.
                let fragment = RecipeIdentity.Components.canonicalRendering(decoded)
                let expected = entry["identityFragment"] as? [String: Any]
                #expect(RecipeIdentity.renderingJSON(fragment)
                        == Self.canonicalFragmentJSON(expected), "\(label)")
            } else {
                // Absent and explicit-raw both canonicalize to NOTHING.
                #expect(entry["identityFragment"] == nil
                        || entry["identityFragment"] is NSNull, "\(label)")
            }
        }
    }

    /// The declarations this engine ACCEPTS produce the server's stamp; the
    /// assistant-voice ones are refused naming the engine that can.
    @Test func theDeclarationParserAgreesWithThePythonEngineOrRefusesLoudly()
        throws
    {
        let fixture = try Self.load()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        for entry in fixture.renderings {
            let label = try #require(entry["label"] as? String)
            guard let declaration = entry["declaration"] as? [String: Any] else {
                continue    // "absent" has no declaration to parse
            }
            let isAssistant = (declaration["voice"] as? String) == "assistant"
            do {
                let parsed = try ExtractionRendering.declared(object: declaration)
                #expect(!isAssistant, "swift-mlx accepted the assistant voice")
                if let stamp = entry["stamp"] as? [String: Any] {
                    let parsed = try #require(parsed, "\(label)")
                    let expected = try JSONSerialization.data(
                        withJSONObject: stamp, options: [.sortedKeys])
                    #expect(String(decoding: try encoder.encode(parsed), as: UTF8.self)
                            == String(decoding: expected, as: UTF8.self), "\(label)")
                } else {
                    #expect(parsed == nil, "\(label) should canonicalize away")
                }
            } catch let error as ExtractionRendering.DeclarationError {
                #expect(isAssistant, "unexpected refusal for \(label): \(error)")
                #expect(error.reason == PromptRendering.assistantVoiceReason)
            }
        }
    }

    /// The two MALFORMED-on-both-engines combinations carry twin texts,
    /// byte for byte.
    @Test func theAssistantVoiceRefusalTextsAreTwins() throws {
        let refusals = try Self.load().refusals
        #expect(ExtractionRendering.assistantVoiceGenerationPromptReason
                == refusals["assistantVoiceGenerationPrompt"])
        #expect(ExtractionRendering.assistantVoiceSystemPromptReason
                == refusals["assistantVoiceSystemPrompt"])
    }

    // MARK: - 2. reading positions contribute the same identity

    @Test func everyReadingPositionAgreesWithThePythonEngine() throws {
        for entry in try Self.load().positions {
            let label = try #require(entry["label"] as? String)
            let position = try #require(ReadingPosition(label: label),
                                        "swift cannot parse the label '\(label)'")
            #expect(position.identityMode == entry["identityMode"] as? String, "\(label)")
            #expect(position.identityParameter == entry["identityParameter"] as? Int,
                    "\(label)")
            #expect(position.minimumTokenCount == entry["minimumTokenCount"] as? Int,
                    "\(label)")
            #expect(position.requiresTemplatedRendering
                    == entry["requiresTemplatedRendering"] as? Bool, "\(label)")
            let canonical = RecipeIdentity.canonicalReading(position)
            #expect(canonical.mode == entry["canonicalMode"] as? String, "\(label)")
            #expect(canonical.parameter == entry["canonicalParameter"] as? Int,
                    "\(label)")
        }
    }

    // MARK: - 3. resolutions land on the same indices

    @Test func everyResolutionLandsWhereThePythonEngineLandsIt() throws {
        let fixture = try Self.load()
        let tokenizer = FixtureTokenizer()
        #expect(try ReadingPosition.contentIndices(
            tokens: fixture.tokens, tokenizer: tokenizer, label: "fixture")
            == fixture.contentIndices)
        for entry in fixture.resolutions {
            let label = try #require(entry["label"] as? String)
            let position = try #require(ReadingPosition(label: label))
            let resolved = try position.resolve(
                tokens: fixture.tokens, tokenizer: tokenizer,
                renderingIsRaw: try #require(entry["renderingIsRaw"] as? Bool))
            let where_ = "\(label) (raw: \(entry["renderingIsRaw"] ?? "?"))"
            #expect(resolved.startIndex == entry["startIndex"] as? Int, "\(where_)")
            #expect(resolved.endIndex == entry["endIndex"] as? Int, "\(where_)")
            #expect(resolved.source == entry["source"] as? String, "\(where_)")
            #expect(resolved.pooledIndices == entry["pooledIndices"] as? [Int],
                    "\(where_)")
            #expect(resolved.maskedTokenCount == entry["maskedTokenCount"] as? Int,
                    "\(where_)")
            #expect(resolved.pooledTokenCount == entry["pooledTokenCount"] as? Int,
                    "\(where_)")
        }
    }

    // MARK: - helpers

    /// The fixture's identity fragment, rendered the way
    /// `RecipeIdentity.renderingJSON` renders one — sorted keys, explicit
    /// nulls, `voice` last and only when present.
    static func canonicalFragmentJSON(_ fragment: [String: Any]?) -> String? {
        guard let fragment else { return nil }
        func string(_ value: Any?) -> String {
            guard let text = value as? String else { return "null" }
            return "\"\(text)\""
        }
        var out = "{\"addGenerationPrompt\":\(fragment["addGenerationPrompt"] as? Bool ?? true)"
        out += ",\"mode\":\(string(fragment["mode"]))"
        out += ",\"qwenThinkingEnabled\":\(fragment["qwenThinkingEnabled"] as? Bool ?? false)"
        out += ",\"systemPrompt\":\(string(fragment["systemPrompt"]))"
        if let voice = fragment["voice"] as? String {
            out += ",\"voice\":\"\(voice)\""
        }
        return out + "}"
    }
}

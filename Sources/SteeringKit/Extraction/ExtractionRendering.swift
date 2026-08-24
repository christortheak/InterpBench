import Foundation
import MLXLMCommon

/// HOW a stimulus string reaches the model during extraction — the rendering.
///
/// Until this option existed the answer was an undeclared implementation
/// detail: extraction tokenized the RAW stimulus string, while measured
/// generation rendered the same text through the model family's chat template
/// with a generation prompt appended. Extracting one concept both ways yields
/// two DIFFERENT directions — cosine ≈ 0.18 mid-network, per-layer norms
/// diverging ~10× with the ordering inverting across depth — that both
/// separate the classes near-perfectly on a probe (measured 2026-08-23). Which
/// direction a study got, and what the α denominator equalled, therefore
/// depended on something nobody declared. That is exactly what the freeze
/// discipline exists to forbid, so the choice is now explicit, pinned, and
/// sweepable.
///
/// **Absent is LEGACY RAW, and is never retro-applied.** A recipe with no
/// `extractionRendering` renders exactly as it always has, its manifest bytes
/// are byte-identical to before, and its recipe-identity hash is unchanged —
/// the same discipline `residualNormConvention: wholeCorpusMean-v1` follows.
/// Only an explicitly declared CHAT-TEMPLATE rendering changes a recipe's
/// identity; an explicitly declared `.raw` is the legacy semantics said out
/// loud, and canonicalizes away.
///
/// Cross-engine contract — the server twin is
/// `Server/steerlab_server/steering/extraction_rendering.py`. Same JSON key
/// (`extractionRendering`), same mode spellings, same inner key names, same
/// defaults:
///
/// ```json
/// {"mode": "raw"}
/// {"mode": "chatTemplate", "addGenerationPrompt": true,
///  "qwenThinkingEnabled": false, "systemPrompt": null}
/// ```
///
/// `mode` is the only required key; `addGenerationPrompt` defaults to true and
/// `qwenThinkingEnabled` to false — the parameters the MEASUREMENT renderer
/// actually varies, and no more. Family handling (Gemma has no system role;
/// Qwen's `enable_thinking`) is derived from the pinned model id by
/// ``PromptRendering``, the single definition both extraction and generation
/// call, so it is not re-declared here.
public struct ExtractionRendering: Codable, Sendable, Equatable {

    public enum Mode: String, Codable, Sendable, CaseIterable {
        /// Today's behavior, and what an absent declaration means: the bare
        /// stimulus string through the tokenizer's defaults.
        case raw
        /// The model family's chat template, rendered by ``PromptRendering``
        /// — the same path measured generation takes.
        case chatTemplate
    }

    public var mode: Mode
    /// chatTemplate only. nil resolves to `true` — what generation does.
    public var addGenerationPrompt: Bool?
    /// chatTemplate only. nil resolves to `false`.
    public var qwenThinkingEnabled: Bool?
    /// chatTemplate only. The system text included in the render, if any.
    public var systemPrompt: String?

    /// The legacy rendering, and the value an absent declaration means.
    public static let raw = ExtractionRendering(mode: .raw)

    public static func chatTemplate(
        addGenerationPrompt: Bool = true,
        qwenThinkingEnabled: Bool = false,
        systemPrompt: String? = nil
    ) -> ExtractionRendering {
        ExtractionRendering(
            mode: .chatTemplate,
            addGenerationPrompt: addGenerationPrompt,
            qwenThinkingEnabled: qwenThinkingEnabled,
            systemPrompt: systemPrompt)
    }

    public init(
        mode: Mode = .raw,
        addGenerationPrompt: Bool? = nil,
        qwenThinkingEnabled: Bool? = nil,
        systemPrompt: String? = nil
    ) {
        self.mode = mode
        self.addGenerationPrompt = addGenerationPrompt
        self.qwenThinkingEnabled = qwenThinkingEnabled
        self.systemPrompt = systemPrompt
    }

    public var isRaw: Bool { mode == .raw }

    /// What the renderer actually passes for `add_generation_prompt`.
    public var resolvedAddGenerationPrompt: Bool { addGenerationPrompt ?? true }

    /// What the renderer actually passes for the Qwen thinking switch.
    public var resolvedQwenThinkingEnabled: Bool { qwenThinkingEnabled ?? false }

    /// Short human label for logs and refusal messages (server twin:
    /// `ExtractionRendering.label`).
    public var label: String {
        guard !isRaw else { return "raw" }
        var bits = ["addGenerationPrompt=\(resolvedAddGenerationPrompt)"]
        if resolvedQwenThinkingEnabled { bits.append("qwenThinkingEnabled=true") }
        if systemPrompt?.isEmpty == false { bits.append("systemPrompt=set") }
        return "chatTemplate (" + bits.joined(separator: ", ") + ")"
    }

    /// The value an ARTIFACT stamps: defaults written explicitly, so a reader
    /// never has to know this type's defaults to know what happened. nil for
    /// a raw rendering — absent is the legacy meaning, and a raw artifact's
    /// sidecar bytes must stay identical to what this engine always wrote.
    public var stamp: ExtractionRendering? {
        guard !isRaw else { return nil }
        return ExtractionRendering(
            mode: .chatTemplate,
            addGenerationPrompt: resolvedAddGenerationPrompt,
            qwenThinkingEnabled: resolvedQwenThinkingEnabled,
            systemPrompt: systemPrompt)
    }
}

/// The family-specific rules for turning a prompt into what the model sees.
///
/// **One definition, two callers.** `ExperimentTasks.userInput` (measured
/// generation) and `ConceptExtractor` (extraction under a chat-template
/// rendering) both come here, so the two can never drift into two renderings
/// — which is precisely the drift the `extractionRendering` option exists to
/// make visible. Server twin:
/// `Server/steerlab_server/experiment/prompt_render.py`.
public enum PromptRendering {

    public static func isGemma(_ modelID: String) -> Bool {
        modelID.lowercased().contains("gemma")
    }

    public static func isQwen(_ modelID: String) -> Bool {
        modelID.lowercased().contains("qwen")
    }

    /// The chat turns for a single-prompt render.
    ///
    /// **Gemma 3 has no system role**: system text is prepended to the user
    /// turn (`system + "\n\n" + user`). Every other family gets a real system
    /// turn.
    public static func chatMessages(
        prompt: String, modelID: String, systemPrompt: String?
    ) -> [Chat.Message] {
        let system = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let system, !system.isEmpty else { return [.user(prompt)] }
        if isGemma(modelID) {
            return [.user(system + "\n\n" + prompt)]
        }
        return [.system(system), .user(prompt)]
    }

    /// The template's extra context: Qwen3 studies disable thinking mode
    /// unless it is explicitly enabled. nil for every other family.
    public static func qwenContext(
        modelID: String, qwenThinkingEnabled: Bool
    ) -> [String: any Sendable]? {
        isQwen(modelID) ? ["enable_thinking": qwenThinkingEnabled] : nil
    }

    /// The rawCompletion text: system prepended, family thinking suffix
    /// appended. Tokenization is the caller's (the tokenizer adds its default
    /// special tokens exactly once).
    public static func rawCompletionText(
        prompt: String, modelID: String, systemPrompt: String?,
        qwenThinkingEnabled: Bool
    ) -> String {
        var text = prompt
        let system = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let system, !system.isEmpty {
            text = system + "\n\n" + text
        }
        if isQwen(modelID) {
            text += qwenThinkingEnabled ? " /think" : " /no_think"
        }
        return text
    }

    /// A declared rendering this engine cannot apply. Typed and carrying a
    /// repair, and never a silent fallback to raw — a silent fallback is the
    /// exact ambiguity the `extractionRendering` option exists to end.
    public struct UnsupportedForm: Error, CustomStringConvertible {
        public let reason: String
        public init(reason: String) { self.reason = reason }
        public var description: String { reason }
    }

    /// Token ids for one stimulus under an extraction rendering.
    ///
    /// The raw branch is the historical `tokenizer.encode(text:)` call,
    /// unchanged — an absent or raw declaration tokenizes byte-for-byte as it
    /// always did. The chat-template branch goes through
    /// `applyChatTemplate(messages:tools:additionalContext:)`, which is
    /// EXACTLY what mlx-swift-lm's `LLMUserInputProcessor.prepare` executes
    /// for a measured generation prompt (pinned by the
    /// `prompts/fixtures/render` goldens) — so extraction reaches the model
    /// through the measurement renderer rather than a second copy of it.
    ///
    /// **`addGenerationPrompt: false` is refused on this engine.**
    /// MLXLMCommon's tokenizer bridge — the one a loaded `ModelContext`
    /// hands us — exposes only the generation-prompt form; swift-transformers'
    /// parameterized `applyChatTemplate` is not reachable from a
    /// `ModelContext`. Rather than quietly rendering WITH a generation prompt
    /// and stamping that the recipe asked for none, the form is a named
    /// refusal. The server engine supports it, so a study that needs it
    /// extracts there.
    public static func tokenIDs(
        tokenizer: any MLXLMCommon.Tokenizer, modelID: String, text: String,
        rendering: ExtractionRendering
    ) throws -> [Int] {
        guard !rendering.isRaw else { return tokenizer.encode(text: text) }
        guard rendering.resolvedAddGenerationPrompt else {
            throw UnsupportedForm(
                reason: "addGenerationPrompt false is not available through "
                    + "MLXLMCommon's tokenizer bridge, which exposes only the "
                    + "generation-prompt form of applyChatTemplate. Repair: "
                    + "extract this concept on the python-hf-transformers "
                    + "engine, which supports it, or declare "
                    + "addGenerationPrompt true here — never let it fall back, "
                    + "because the trailing template tokens change both the "
                    + "vector and every named reading position that lands on "
                    + "them.")
        }
        let messages = chatMessages(
            prompt: text, modelID: modelID,
            systemPrompt: rendering.systemPrompt)
        return try tokenizer.applyChatTemplate(
            messages: DefaultMessageGenerator().generate(messages: messages),
            tools: nil,
            additionalContext: qwenContext(
                modelID: modelID,
                qwenThinkingEnabled: rendering.resolvedQwenThinkingEnabled))
    }
}

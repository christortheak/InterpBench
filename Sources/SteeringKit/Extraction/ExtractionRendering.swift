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
///
/// ## The voice (2026-08-25, maintainer ruling)
///
/// `voice` names WHOSE TURN the stimulus is rendered as: `"user"` (the model
/// READS it — the legacy behavior) or `"assistant"` (the model PRODUCED it).
/// **Absent ≡ `"user"` ≡ today's bytes**, and an explicit `"user"`
/// canonicalizes away exactly as an explicit `{"mode": "raw"}` does, so no
/// existing manifest, sidecar, freeze hash or recipe identity moves.
///
/// **This engine REFUSES the assistant voice** (see
/// ``PromptRendering/assistantVoiceReason``) and names the server engine, the
/// `addGenerationPrompt: false` precedent exactly. It still UNDERSTANDS the
/// voice everywhere else — decoding it, hashing it into a recipe identity,
/// diffing it — because a Mac that promotes an artifact the server extracted
/// under the assistant voice must compute the identical identity for it.
public struct ExtractionRendering: Codable, Sendable, Equatable {

    public enum Mode: String, Codable, Sendable, CaseIterable {
        /// Today's behavior, and what an absent declaration means: the bare
        /// stimulus string through the tokenizer's defaults.
        case raw
        /// The model family's chat template, rendered by ``PromptRendering``
        /// — the same path measured generation takes.
        case chatTemplate
    }

    /// WHOSE TURN the stimulus is rendered as. Server twin:
    /// `extraction_rendering.VOICES`.
    public enum Voice: String, Codable, Sendable, CaseIterable {
        /// The stimulus is what the model READS. The legacy voice, and what
        /// an absent `voice` means.
        case user
        /// The stimulus is the model's OWN OUTPUT — the template's
        /// assistant-turn markers around it, no preceding user content.
        case assistant
    }

    public var mode: Mode
    /// chatTemplate only. nil resolves to `true` — what generation does.
    /// Meaningless (and refused) under the assistant voice.
    public var addGenerationPrompt: Bool?
    /// chatTemplate only. The LEGACY spelling of the reasoning effort
    /// (false ≡ off, true ≡ xhigh) — read from a recorded block, never
    /// written by a new declaration. nil with no `reasoningEffort` resolves to
    /// off.
    public var qwenThinkingEnabled: Bool?
    /// chatTemplate only. The declared reasoning effort (2026-09-03); nil
    /// resolves through the legacy boolean, else to `.off`. A block carrying
    /// BOTH spellings is refused as two declarations of one parameter.
    public var reasoningEffort: ReasoningEffort?
    /// chatTemplate only. The system text included in the render, if any.
    /// Meaningless (and refused) under the assistant voice.
    public var systemPrompt: String?
    /// chatTemplate only. nil resolves to `.user` — and STAYS nil there, so a
    /// recipe that never heard of the voice keeps its bytes.
    public var voice: Voice?

    /// The legacy rendering, and the value an absent declaration means.
    public static let raw = ExtractionRendering(mode: .raw)

    public static func chatTemplate(
        addGenerationPrompt: Bool = true,
        qwenThinkingEnabled: Bool = false,
        systemPrompt: String? = nil,
        voice: Voice? = nil,
        reasoningEffort: ReasoningEffort? = nil
    ) -> ExtractionRendering {
        ExtractionRendering(
            mode: .chatTemplate,
            addGenerationPrompt: addGenerationPrompt,
            // One spelling in memory: an effort given explicitly supersedes
            // the boolean, and a boolean alone becomes its effort.
            reasoningEffort: reasoningEffort
                ?? ReasoningEffort.legacy(qwenThinkingEnabled: qwenThinkingEnabled),
            systemPrompt: systemPrompt,
            voice: voice)
    }

    public init(
        mode: Mode = .raw,
        addGenerationPrompt: Bool? = nil,
        qwenThinkingEnabled: Bool? = nil,
        reasoningEffort: ReasoningEffort? = nil,
        systemPrompt: String? = nil,
        voice: Voice? = nil
    ) {
        self.mode = mode
        self.addGenerationPrompt = addGenerationPrompt
        self.qwenThinkingEnabled = qwenThinkingEnabled
        self.reasoningEffort = reasoningEffort
        self.systemPrompt = systemPrompt
        self.voice = voice
    }

    public var isRaw: Bool { mode == .raw }

    /// What the renderer actually speaks in. nil ≡ `.user`.
    public var resolvedVoice: Voice { voice ?? .user }

    public var isAssistantVoice: Bool { resolvedVoice == .assistant }

    /// What the renderer actually passes for `add_generation_prompt`.
    ///
    /// `false` under the ASSISTANT voice, always: that construction renders a
    /// COMPLETED assistant turn, which is the no-generation-prompt form of the
    /// template call. The declaration refuses the key there (it is not a
    /// choice), but the recipe identity still records what the render did —
    /// and it must record the same thing the server does, or a Mac and the
    /// server would compute two identities for one artifact.
    public var resolvedAddGenerationPrompt: Bool {
        isAssistantVoice ? false : (addGenerationPrompt ?? true)
    }

    /// The effort the renderer actually applies: the declared one, else the
    /// legacy boolean's meaning, else off. Server twin:
    /// `ExtractionRendering.reasoning_effort` (resolved at parse).
    public var resolvedReasoningEffort: ReasoningEffort {
        ReasoningEffort.resolve(
            reasoningEffort, qwenThinkingEnabled: qwenThinkingEnabled ?? false)
    }

    /// Whether the render carries a thinking scaffold at all — derived from
    /// the effort, never stored separately.
    public var resolvedQwenThinkingEnabled: Bool { resolvedReasoningEffort.isOn }

    /// Short human label for logs and refusal messages (server twin:
    /// `ExtractionRendering.label`).
    public var label: String {
        guard !isRaw else { return "raw" }
        let effort = resolvedReasoningEffort
        if isAssistantVoice {
            // addGenerationPrompt is not a knob under this voice, so naming
            // it in a human label would invite a reader to look for it.
            var bits = ["voice=assistant"]
            if effort.isOn { bits.append("reasoningEffort=\(effort.rawValue)") }
            return "chatTemplate (" + bits.joined(separator: ", ") + ")"
        }
        var bits = ["addGenerationPrompt=\(resolvedAddGenerationPrompt)"]
        if effort.isOn { bits.append("reasoningEffort=\(effort.rawValue)") }
        if systemPrompt?.isEmpty == false { bits.append("systemPrompt=set") }
        return "chatTemplate (" + bits.joined(separator: ", ") + ")"
    }

    /// The value an ARTIFACT stamps: defaults written explicitly, so a reader
    /// never has to know this type's defaults to know what happened. nil for
    /// a raw rendering — absent is the legacy meaning, and a raw artifact's
    /// sidecar bytes must stay identical to what this engine always wrote.
    ///
    /// The USER voice stamps no `voice` key (absent ≡ user); the ASSISTANT
    /// voice stamps the key and OMITS `addGenerationPrompt`, which is refused
    /// at declaration time as meaningless there — stamping the internal
    /// `false` would be an artifact claiming a choice nobody made. Server
    /// twin: `ExtractionRendering.to_dict`.
    public var stamp: ExtractionRendering? {
        guard !isRaw else { return nil }
        // A NEW stamp spells the effort, never the legacy boolean (the
        // server's `to_dict` writes the same keys). A block READ under the
        // old spelling is never re-serialized into a frozen file.
        if isAssistantVoice {
            return ExtractionRendering(
                mode: .chatTemplate,
                reasoningEffort: resolvedReasoningEffort,
                voice: .assistant)
        }
        return ExtractionRendering(
            mode: .chatTemplate,
            addGenerationPrompt: resolvedAddGenerationPrompt,
            reasoningEffort: resolvedReasoningEffort,
            systemPrompt: systemPrompt)
    }

    // MARK: - Reading is as strict as declaring

    public enum CodingKeys: String, CodingKey {
        case mode, addGenerationPrompt, qwenThinkingEnabled, reasoningEffort
        case systemPrompt, voice
    }

    /// A key of ARBITRARY name, so the decoder can see the strangers a typed
    /// container silently steps over.
    private struct AnyKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    /// Decoding refuses a parameter under `raw`, and an unknown chat-template
    /// key, exactly as ``declared(object:)`` does — and for a sharper reason:
    /// a stranger in a RECORDED block — a manifest's
    /// `options.extractionRendering`, an artifact sidecar's stamp — can only
    /// have been written by a newer engine declaring a field this one does not
    /// understand, or by a hand edit with a typo in it. Reading such a block
    /// as though the key were absent would hand the study a rendering nobody
    /// declared, which is the whole failure this type exists to end. Server
    /// twin: `from_json`, whose strictness the declaration path and the read
    /// path already share.
    ///
    /// The RAW branch closed 2026-08-26 (external review round 5). Declaring
    /// `{"mode": "raw", "addGenerationPrompt": false}` was already refused by
    /// ``declared(object:)`` and by the server's `from_json` on read, but this
    /// engine's DECODE accepted it and silently dropped the parameter — the
    /// documented "reading is as strict as declaring" rule broken in exactly
    /// the direction that lets the two engines disagree about a recorded
    /// block: the server refusing a manifest this one reads happily.
    ///
    /// The encoder stays synthesized: what this engine WRITES is only ever the
    /// five keys (and never a parameter under raw), so no artifact's bytes
    /// move.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // `mode` stays REQUIRED here (the synthesized decoder's rule, kept):
        // a recorded block with no mode at all is malformed, not raw.
        let mode = try container.decode(Mode.self, forKey: .mode)
        if mode == .raw {
            let everyKey = try decoder.container(keyedBy: AnyKey.self)
            var extra: [String] = []
            for key in everyKey.allKeys where key.stringValue != "mode" {
                if try !everyKey.decodeNil(forKey: key) {
                    extra.append(key.stringValue)
                }
            }
            guard extra.isEmpty else {
                throw ExtractionRendering.rawParametersError(extra.sorted())
            }
        }
        if mode == .chatTemplate {
            let everyKey = try decoder.container(keyedBy: AnyKey.self)
            var unknown: [String] = []
            for key in everyKey.allKeys
            where !ExtractionRendering.chatTemplateKeys.contains(key.stringValue) {
                if try !everyKey.decodeNil(forKey: key) {
                    unknown.append(key.stringValue)
                }
            }
            guard unknown.isEmpty else {
                throw ExtractionRendering.unknownChatTemplateKeyError(
                    unknown.sorted())
            }
        }
        self.mode = mode
        self.addGenerationPrompt = try container.decodeIfPresent(
            Bool.self, forKey: .addGenerationPrompt)
        self.qwenThinkingEnabled = try container.decodeIfPresent(
            Bool.self, forKey: .qwenThinkingEnabled)
        // Decoded as the string it is, so a value outside the vocabulary is
        // THIS type's refusal (the server's `unknown_effort_reason`), not a
        // bare decoder error — and both spellings at once is refused as two
        // declarations of one parameter, exactly as `declared(object:)` does.
        if let spelled = try container.decodeIfPresent(
            String.self, forKey: .reasoningEffort)
        {
            guard let parsed = ReasoningEffort(rawValue: spelled) else {
                throw ExtractionRendering.unknownEffortError(spelled)
            }
            if qwenThinkingEnabled != nil {
                throw ExtractionRendering.bothThinkingKeysError
            }
            self.reasoningEffort = parsed
        } else {
            self.reasoningEffort = nil
        }
        self.systemPrompt = try container.decodeIfPresent(
            String.self, forKey: .systemPrompt)
        self.voice = try container.decodeIfPresent(Voice.self, forKey: .voice)
    }
}

// MARK: - The WRITER: a typed declaration off the command line

/// Parsing an `extractionRendering` declaration typed by a person or an agent.
///
/// The option shipped 2026-08-24 with every CONSUMER live — recipe identity,
/// the α denominator, the template-aware reading positions, the sidecar
/// stamps, the refusals that name it — and no writer at all: no flag, no
/// route field, nothing. The engines' own refusals said "declare
/// extractionRendering" and could name no command to declare it with. This is
/// that command's parser.
///
/// **Validation happens HERE, at parse time, not at extraction.** A malformed
/// declaration, an unknown mode, a parameter this engine cannot honor: every
/// one of them is answered before a single byte is written to a manifest, so
/// the study never carries a declaration whose first refusal arrives hours
/// later on a GPU. Server twin: `extraction_rendering.parse_declaration`.
extension ExtractionRendering {

    /// The flag both engines' CLIs spell this with. One constant, so a help
    /// page, a refusal, and a parser cannot name three different flags.
    public static let declarationFlag = "--extraction-rendering"

    /// This engine's name in a refusal about an unsupportable declaration
    /// (server twin: `extraction_rendering.ENGINE`).
    public static let declarationEngine = "swift-mlx"

    /// Refusal text shared VERBATIM with the server twin
    /// (`extraction_rendering.ASSISTANT_VOICE_GENERATION_PROMPT_REASON`).
    /// A MALFORMED declaration on both engines, so it is checked before the
    /// engine-asymmetry refusal below: retyping without the key is the first
    /// repair, and only then does the question of which engine can render it
    /// arise.
    public static let assistantVoiceGenerationPromptReason =
        "extractionRendering declares voice 'assistant' together with "
        + "addGenerationPrompt: under the assistant voice the stimulus IS the "
        + "generation, so there is no generation prompt to add or withhold and "
        + "the key would reach nothing — repair: drop addGenerationPrompt, or "
        + "declare voice 'user' if you meant the model READING the stimulus"

    /// Refusal text shared VERBATIM with the server twin
    /// (`extraction_rendering.ASSISTANT_VOICE_SYSTEM_PROMPT_REASON`).
    public static let assistantVoiceSystemPromptReason =
        "extractionRendering declares voice 'assistant' together with a "
        + "systemPrompt: the assistant-voice construction renders the assistant "
        + "turn ALONE, with no preceding turn for system text to live in "
        + "(injected context would confound the very contrast the voice exists "
        + "to isolate) — repair: drop systemPrompt, or declare voice 'user' if "
        + "the model is meant to read the stimulus under that system prompt"

    /// A declaration this engine will not accept. Typed and carrying a
    /// repair, per house style — the CLI turns it into a malformed-invocation
    /// refusal (exit 64: nothing ran, and retrying cannot help).
    public struct DeclarationError: Error, CustomStringConvertible, Equatable {
        public let reason: String
        public let repair: String
        public init(reason: String, repair: String) {
            self.reason = reason
            self.repair = repair
        }
        public var description: String { reason }

        /// The ONE-LINE form, which is how the server twin's
        /// `ExtractionRenderingError` carries the same refusal: reason, an em
        /// dash, and the repair. Used where a refusal is quoted INSIDE another
        /// engine's message (an attach refusal, a verify violation), so the
        /// two engines' composed texts match as well as their parts do.
        public var message: String { reason + " — repair: " + repair }
    }

    /// EVERY key a `chatTemplate` rendering may carry, sorted — the vocabulary
    /// a refusal about a stranger names. Server twin:
    /// `extraction_rendering.CHAT_TEMPLATE_KEYS`.
    public static let chatTemplateKeys = [
        "addGenerationPrompt", "mode", "qwenThinkingEnabled", "reasoningEffort",
        "systemPrompt", "voice",
    ]

    // MARK: - The reasoning effort (2026-09-03)

    /// Refusal text shared VERBATIM with the server twin
    /// (`extraction_rendering.BOTH_THINKING_KEYS_REASON`), in its one-line
    /// form; `bothThinkingKeysError` carries the same text split into reason
    /// and repair.
    public static let bothThinkingKeysReason =
        "extractionRendering declares both qwenThinkingEnabled and "
        + "reasoningEffort — two spellings of one parameter; qwenThinkingEnabled "
        + "is the legacy boolean (false ≡ off, true ≡ xhigh) and reasoningEffort "
        + "replaces it — repair: keep reasoningEffort and drop qwenThinkingEnabled"

    public static let bothThinkingKeysError = DeclarationError(
        reason: "extractionRendering declares both qwenThinkingEnabled and "
            + "reasoningEffort — two spellings of one parameter; "
            + "qwenThinkingEnabled is the legacy boolean (false ≡ off, true ≡ "
            + "xhigh) and reasoningEffort replaces it",
        repair: "keep reasoningEffort and drop qwenThinkingEnabled")

    /// Server twin: `extraction_rendering.unknown_effort_reason`.
    public static func unknownEffortError(_ value: String) -> DeclarationError {
        DeclarationError(
            reason: "extractionRendering.reasoningEffort '\(value)' is not in "
                + "the closed vocabulary",
            repair: "declare one of "
                + ReasoningEffort.vocabulary.joined(separator: ", ")
                + " (absent means off)")
    }

    /// Server twin: `extraction_rendering.effort_without_thinking_mode_reason`.
    public static func effortWithoutThinkingModeError(
        _ effort: ReasoningEffort, modelID: String
    ) -> DeclarationError {
        DeclarationError(
            reason: "extractionRendering declares reasoningEffort "
                + "'\(effort.rawValue)' for \(modelID), whose chat template has "
                + "no thinking switch (enable_thinking changes nothing it "
                + "renders) — the template would ignore it and the recipe would "
                + "look as if it rendered a reasoning scaffold when it did not",
            repair: "declare reasoningEffort off, or pin a model with a "
                + "thinking mode")
    }

    /// The refusal for a non-off effort on a family without a thinking mode,
    /// or nil — asked wherever a declaration meets the pinned model id
    /// (attach, the build routes), because the parser itself never sees one.
    /// Server twin: `extraction_rendering.thinking_mode_problem`.
    public static func thinkingModeProblem(
        _ rendering: ExtractionRendering?, modelID: String,
        capabilities: ModelCapabilities? = nil
    ) -> DeclarationError? {
        guard let rendering, !rendering.isRaw else { return nil }
        let effort = rendering.resolvedReasoningEffort
        guard effort.isOn else { return nil }
        let record = PromptRendering.capabilities(for: modelID, explicit: capabilities)
        guard record.hasThinkingSwitch else {
            return effortWithoutThinkingModeError(effort, modelID: modelID)
        }
        guard effort.isLevel else { return nil }
        guard let problem = ReasoningEffort.effortLevelViolations(
            effort, modelID: modelID, capabilities: record).first
        else { return nil }
        return DeclarationError(
            reason: effortLevelProblemPrefix + problem,
            repair: "declare \"reasoningEffort\": \"\(ReasoningEffort.on.rawValue)\" "
                + "(thinking at the template's default), a level the template "
                + "accepts, or off")
    }

    /// The prefix the study-protocol level sentences carry when the
    /// declaration is a concept's `extractionRendering`. Server twin:
    /// `extraction_rendering.effort_level_problem_prefix`.
    public static let effortLevelProblemPrefix = "extractionRendering: "

    /// The non-null keys a `raw` block carries beyond `mode`, sorted.
    ///
    /// NON-null, for the same reason ``unknownChatTemplateKeys(in:)`` is: an
    /// explicit `null` says "this parameter is absent", which is what an
    /// absent key already says. Server twin:
    /// `extraction_rendering.raw_parameters`.
    public static func rawParameters(in object: [String: Any]) -> [String] {
        object.keys
            .filter { $0 != "mode" }
            .filter { object[$0].map { !($0 is NSNull) } ?? false }
            .sorted()
    }

    /// Refusal text shared VERBATIM with the server twin
    /// (`extraction_rendering.raw_parameters_reason`), as
    /// `DeclarationError.message`.
    ///
    /// ONE definition, reached by both the declaration path
    /// (``declared(object:)``) and the DECODE path (`init(from:)`), so reading
    /// a recorded block is as strict as declaring one and the two engines
    /// refuse the same input with the same words.
    public static func rawParametersError(
        _ keys: [String]
    ) -> DeclarationError {
        DeclarationError(
            reason: "extractionRendering mode 'raw' takes no parameters but "
                + "declares \(keys.joined(separator: ", "))",
            repair: "drop the parameters, or declare mode 'chatTemplate' if "
                + "you meant the template rendering")
    }

    /// The non-null keys of `object` this engine does not read, sorted.
    ///
    /// NON-null, exactly as the raw branch's own extra-parameter check: an
    /// explicit `null` says "this parameter is absent", which is what an
    /// absent key already says, so it declares nothing.
    public static func unknownChatTemplateKeys(
        in object: [String: Any]
    ) -> [String] {
        object.keys
            .filter { !chatTemplateKeys.contains($0) }
            .filter { object[$0].map { !($0 is NSNull) } ?? false }
            .sorted()
    }

    /// Refusal text shared VERBATIM with the server twin
    /// (`extraction_rendering.unknown_chat_template_key_reason`), as
    /// `DeclarationError.message`.
    ///
    /// The raw branch has always refused strangers; the chat-template branch
    /// used to read its five known keys and IGNORE everything else, so
    /// `"addGenerationPromt": false` — one transposed letter — silently kept
    /// the default `true` and a frozen study measured something other than
    /// what its manifest appears to declare. A key that reaches nothing is the
    /// same failure this option exists to end, so it is answered out loud
    /// (review 2026-08-26).
    public static func unknownChatTemplateKeyError(
        _ keys: [String]
    ) -> DeclarationError {
        DeclarationError(
            reason: "extractionRendering mode 'chatTemplate' does not accept "
                + "\(keys.joined(separator: ", ")): a key this engine does not "
                + "read reaches nothing, so a misspelling (addGenerationPromt) "
                + "silently leaves the default in place and changes what a "
                + "frozen study measured",
            repair: "correct or drop the key; the accepted parameters are "
                + chatTemplateKeys.joined(separator: ", "))
    }

    /// Parse what was typed after `--extraction-rendering`.
    ///
    /// Accepts the JSON object the schema documents (`{"mode":
    /// "chatTemplate", "addGenerationPrompt": true, …}`) and, as the same
    /// convenience the server twin offers, a bare mode word (`raw`,
    /// `chatTemplate`) — a shell quoting a whole JSON object for the common
    /// case is friction with no meaning.
    ///
    /// **Returns nil for a RAW declaration**, and that is the hash contract,
    /// not an omission: an explicit `{"mode": "raw"}` is the legacy semantics
    /// said out loud, so it must write exactly what saying nothing writes —
    /// the same rule `RecipeIdentity` already applies when it canonicalizes
    /// an explicit raw away. A chat-template declaration comes back as its
    /// ``stamp``: every resolved default written out, so the manifest says
    /// what the extraction will do without a reader knowing this type's
    /// defaults.
    public static func declared(_ text: String) throws -> ExtractionRendering? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw DeclarationError(
                reason: "\(declarationFlag) was given no value",
                repair: "declare a rendering, e.g. \(declarationFlag) "
                    + "'{\"mode\": \"chatTemplate\"}' — or omit the flag "
                    + "entirely, which means the legacy raw rendering")
        }
        let object: [String: Any]
        if trimmed.hasPrefix("{") || trimmed.hasPrefix("\"") {
            guard let data = trimmed.data(using: .utf8),
                let parsed = try? JSONSerialization.jsonObject(
                    with: data, options: [.fragmentsAllowed])
            else {
                throw DeclarationError(
                    reason: "\(declarationFlag) is not valid JSON: \(trimmed)",
                    repair: "quote the whole object in the shell, e.g. "
                        + "\(declarationFlag) '{\"mode\": \"chatTemplate\"}'")
            }
            if let dictionary = parsed as? [String: Any] {
                object = dictionary
            } else if let mode = parsed as? String {
                object = ["mode": mode]
            } else {
                throw DeclarationError(
                    reason: "extractionRendering must be an object or a mode "
                        + "string, got \(type(of: parsed))",
                    repair: "declare {\"mode\": \"raw\"} or "
                        + "{\"mode\": \"chatTemplate\"}")
            }
        } else {
            object = ["mode": trimmed]
        }
        return try declared(object: object)
    }

    /// The same parse over an already-decoded object — the form a route body
    /// arrives in. Shares every rule with the text form above.
    public static func declared(object: [String: Any]) throws -> ExtractionRendering? {
        let rawMode = object["mode"] ?? Mode.raw.rawValue
        guard let modeText = rawMode as? String, let mode = Mode(rawValue: modeText)
        else {
            throw DeclarationError(
                reason: "extraction rendering '\(rawMode)' is not supported by "
                    + "the \(declarationEngine) engine",
                repair: "declare one of "
                    + Mode.allCases.map(\.rawValue).joined(separator: ", ")
                    + " (absent means legacy raw)")
        }
        guard mode == .chatTemplate else {
            // A raw rendering takes no parameters: accepting them silently
            // would let a manifest look like it declared something it cannot
            // get. Server twin: the same check in `from_json`. The refusal
            // itself lives in `rawParametersError`, which `init(from:)` also
            // reaches — declaring and READING refuse identically.
            let extra = rawParameters(in: object)
            guard extra.isEmpty else {
                throw rawParametersError(extra)
            }
            // Explicit raw == absent, byte for byte.
            return nil
        }
        // …and neither does a chat-template rendering, beyond the five
        // parameters it actually varies. Checked FIRST, before any of them is
        // read, so a stranger is answered before the meaningless-under-this-
        // voice combinations below — one deterministic order, twinned in the
        // server's `from_json`.
        let unknown = unknownChatTemplateKeys(in: object)
        guard unknown.isEmpty else {
            throw unknownChatTemplateKeyError(unknown)
        }
        let addGenerationPrompt = try strictBool(
            object["addGenerationPrompt"], key: "addGenerationPrompt")
        let qwenThinkingEnabled = try strictBool(
            object["qwenThinkingEnabled"], key: "qwenThinkingEnabled")
        // The effort (2026-09-03): closed vocabulary, and never beside the
        // legacy boolean — one parameter, one spelling per declaration.
        var declaredEffort: ReasoningEffort?
        if let spelled = object["reasoningEffort"], !(spelled is NSNull) {
            guard let text = spelled as? String,
                let parsed = ReasoningEffort(rawValue: text)
            else {
                throw unknownEffortError("\(spelled)")
            }
            if qwenThinkingEnabled != nil {
                throw bothThinkingKeysError
            }
            declaredEffort = parsed
        }
        var systemPrompt: String?
        if let declaredSystem = object["systemPrompt"], !(declaredSystem is NSNull) {
            guard let text = declaredSystem as? String else {
                throw DeclarationError(
                    reason: "extractionRendering.systemPrompt must be a string "
                        + "or absent",
                    repair: "drop the key when the render carries no system "
                        + "prompt")
            }
            systemPrompt = text
        }
        var voice: Voice?
        if let declaredVoice = object["voice"], !(declaredVoice is NSNull) {
            guard let text = declaredVoice as? String,
                let parsed = Voice(rawValue: text)
            else {
                throw DeclarationError(
                    reason: "extraction rendering voice '\(declaredVoice)' is "
                        + "not supported by the \(declarationEngine) engine",
                    repair: "declare one of "
                        + Voice.allCases.map(\.rawValue).joined(separator: ", ")
                        + " (absent means '\(Voice.user.rawValue)', the legacy "
                        + "voice: the stimulus is what the model reads)")
            }
            voice = parsed
        }
        if voice == .assistant {
            // MALFORMED ON BOTH ENGINES, and therefore answered before the
            // engine question below: these parameters reach nothing under this
            // voice, and a declaration that looks like a recipe axis but does
            // nothing is the failure this option exists to end. Twin texts.
            guard addGenerationPrompt == nil else {
                throw DeclarationError(
                    reason: assistantVoiceGenerationPromptReason,
                    repair: "drop addGenerationPrompt, or declare voice 'user'")
            }
            guard systemPrompt == nil else {
                throw DeclarationError(
                    reason: assistantVoiceSystemPromptReason,
                    repair: "drop systemPrompt, or declare voice 'user'")
            }
            // THE ENGINE ASYMMETRY, ANSWERED AT PARSE TIME (the
            // addGenerationPrompt-false precedent, one step further).
            throw DeclarationError(
                reason: PromptRendering.assistantVoiceReason,
                repair: "extract this concept on the python-hf-transformers "
                    + "engine, or declare \"voice\": \"user\" here")
        }
        let rendering = ExtractionRendering(
            mode: .chatTemplate,
            addGenerationPrompt: addGenerationPrompt,
            qwenThinkingEnabled: qwenThinkingEnabled,
            reasoningEffort: declaredEffort,
            systemPrompt: systemPrompt,
            voice: voice)
        // THE ENGINE ASYMMETRY, ANSWERED AT PARSE TIME. This engine cannot
        // render without a generation prompt (see `PromptRendering.tokenIDs`),
        // and the honest moment to say so is while the person is still typing
        // — not after a frozen manifest reaches an extraction run.
        guard rendering.resolvedAddGenerationPrompt else {
            throw DeclarationError(
                reason: PromptRendering.addGenerationPromptFalseReason,
                repair: "extract this concept on the python-hf-transformers "
                    + "engine, or declare \"addGenerationPrompt\": true here")
        }
        return rendering.stamp
    }

    /// Booleans only — JSON's `1` is not `true` here. `nil` when the key is
    /// absent (the type's own default then applies).
    private static func strictBool(_ value: Any?, key: String) throws -> Bool? {
        guard let value, !(value is NSNull) else { return nil }
        if let number = value as? NSNumber,
            CFGetTypeID(number) == CFBooleanGetTypeID()
        {
            return number.boolValue
        }
        throw DeclarationError(
            reason: "extractionRendering.\(key) must be a boolean",
            repair: "use true or false")
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

    /// The capability record a renderer or gate reads, in precedence order:
    /// the one the caller resolved; else the one registered for the loaded
    /// model (`ModelCapabilitiesStore.ensure`, at load); else the id
    /// heuristic, which says so. Server twin:
    /// `prompt_render.capabilities_for`.
    public static func capabilities(
        for modelID: String, explicit: ModelCapabilities? = nil
    ) -> ModelCapabilities {
        explicit ?? ModelCapabilities.registered(for: modelID)
            ?? ModelCapabilities.heuristicView(modelID: modelID)
    }

    /// Whether the pinned chat template has a SYSTEM ROLE — the record's
    /// `systemRole == systemTurn`.
    ///
    /// The capability `chatMessages` branches on, named so authoring
    /// surfaces can ask it without re-deriving the rule: true means a
    /// declared system prompt is inserted as a genuine system turn, false
    /// means the SAME text is prepended to the first user turn
    /// (`system + separator + user`, the separator the probe recorded)
    /// because the template has nowhere else to put it. Either way it
    /// reaches the model — which is the property `experiment
    /// set-system-prompt` reports — unless the record says `unsupported`,
    /// which the declaration gates refuse. Server twin:
    /// `prompt_render.has_system_role`.
    public static func hasSystemRole(
        _ modelID: String, capabilities: ModelCapabilities? = nil
    ) -> Bool {
        self.capabilities(for: modelID, explicit: capabilities).hasSystemRole
    }

    /// How a declared system prompt reaches the model: `promptPrepend`
    /// (rawCompletion renders no template at all), `systemTurn`,
    /// `prependedToFirstUserTurn`, or `undeliverable`. Server twin:
    /// `prompt_render.system_prompt_delivery`.
    public static func systemPromptDelivery(
        modelID: String, rawCompletion: Bool, capabilities: ModelCapabilities? = nil
    ) -> String {
        if rawCompletion { return "promptPrepend" }
        let record = self.capabilities(for: modelID, explicit: capabilities)
        if record.hasSystemRole { return "systemTurn" }
        if record.systemPromptDeliverable { return "prependedToFirstUserTurn" }
        return "undeliverable"
    }

    /// The one rule about a system prompt: under a chat template, the
    /// template must be able to deliver it. Server twin:
    /// `prompt_render.system_prompt_violations`.
    public static func systemPromptViolations(
        systemPrompt: String?, modelID: String, rawCompletion: Bool,
        capabilities: ModelCapabilities? = nil
    ) -> [String] {
        let trimmed = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty else { return [] }
        guard systemPromptDelivery(
            modelID: modelID, rawCompletion: rawCompletion, capabilities: capabilities)
            == "undeliverable"
        else { return [] }
        return [ReasoningEffort.systemPromptUnsupportedReason(modelID: modelID)]
    }

    /// The fold a template without a system role gets: the probe-recorded
    /// separator (`"\n\n"` on Gemma 3 and under the heuristic).
    public static func foldSystemPrompt(
        _ system: String, into user: String, capabilities: ModelCapabilities
    ) -> String {
        system + (capabilities.foldSeparator ?? "\n\n") + user
    }

    /// The chat turns for a single-prompt render.
    ///
    /// A template with a system role gets a real system turn; one without
    /// (Gemma 3) gets the system text prepended to the user turn with the
    /// recorded separator; one that cannot deliver system text at all gets
    /// the prompt alone — the declaration gates refuse that frame first, and
    /// `systemPromptViolations` is the render-time check.
    public static func chatMessages(
        prompt: String, modelID: String, systemPrompt: String?,
        capabilities: ModelCapabilities? = nil
    ) -> [Chat.Message] {
        let system = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let system, !system.isEmpty else { return [.user(prompt)] }
        let record = self.capabilities(for: modelID, explicit: capabilities)
        if record.hasSystemRole { return [.system(system), .user(prompt)] }
        if record.systemPromptDeliverable {
            return [.user(foldSystemPrompt(system, into: prompt, capabilities: record))]
        }
        return [.user(prompt)]
    }

    /// Whether the pinned chat template HAS a thinking switch to declare an
    /// effort for — the record's `thinkingSwitch` (Qwen3/Qwen3.8 do; Gemma 3
    /// does not — a non-off effort on it would reach nothing, so the
    /// declaration gates refuse it by name). Server twin:
    /// `prompt_render.has_thinking_mode`.
    public static func hasThinkingMode(
        _ modelID: String, capabilities: ModelCapabilities? = nil
    ) -> Bool {
        self.capabilities(for: modelID, explicit: capabilities).hasThinkingSwitch
    }

    /// The template's extra context for the declared effort — ONLY the
    /// variables the record says the template reads. nil for a template
    /// without a thinking switch. `off` is exactly the context every study
    /// has always rendered with (`enable_thinking: false`), so an off
    /// study's prompt bytes cannot move; `on` is `enable_thinking: true`
    /// alone; a level adds `reasoning_effort` beside it when the template
    /// reads the variable and accepts the level — a level the template
    /// IGNORES renders as `on` (byte-identical to what such a template
    /// always produced; the gates refuse declaring it anew), a level the
    /// template REJECTS also renders as `on` here, because a render cannot
    /// refuse — the gates already did. A heuristic record keeps the
    /// pre-record behaviour: every assumed level is passed through.
    /// Server twin: `prompt_render.thinking_template_kwargs`.
    public static func thinkingContext(
        modelID: String, effort: ReasoningEffort, capabilities: ModelCapabilities? = nil
    ) -> [String: any Sendable]? {
        let record = self.capabilities(for: modelID, explicit: capabilities)
        guard record.hasThinkingSwitch else { return nil }
        guard effort.isOn else { return ["enable_thinking": false] }
        guard effort.isLevel else { return ["enable_thinking": true] }
        switch record.effortVerdict(effort.rawValue) {
        case .accepted, .assumed:
            return ["enable_thinking": true, "reasoning_effort": effort.rawValue]
        case .ignored, .rejected:
            return ["enable_thinking": true]
        case nil:
            return record.effortVariableRead == false
                ? ["enable_thinking": true]
                : ["enable_thinking": true, "reasoning_effort": effort.rawValue]
        }
    }

    /// The template's extra context, from the LEGACY boolean: Qwen3 studies
    /// disable thinking mode unless it is explicitly enabled (true means the
    /// template's default effort). nil for every family without a switch.
    public static func qwenContext(
        modelID: String, qwenThinkingEnabled: Bool, capabilities: ModelCapabilities? = nil
    ) -> [String: any Sendable]? {
        thinkingContext(
            modelID: modelID,
            effort: ReasoningEffort.legacy(qwenThinkingEnabled: qwenThinkingEnabled),
            capabilities: capabilities)
    }

    /// The rawCompletion text: system prepended, family thinking suffix
    /// appended. Tokenization is the caller's (the tokenizer adds its default
    /// special tokens exactly once).
    public static func rawCompletionText(
        prompt: String, modelID: String, systemPrompt: String?,
        qwenThinkingEnabled: Bool, reasoningEffort: ReasoningEffort? = nil,
        capabilities: ModelCapabilities? = nil
    ) -> String {
        var text = prompt
        let system = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let system, !system.isEmpty {
            text = system + "\n\n" + text
        }
        // The soft switch of a ChatML thinking family, appended to raw text
        // exactly as before the record existed.
        if self.capabilities(for: modelID, explicit: capabilities).hasThinkingSwitch {
            let effort = ReasoningEffort.resolve(
                reasoningEffort, qwenThinkingEnabled: qwenThinkingEnabled)
            text += effort.isOn ? " /think" : " /no_think"
        }
        return text
    }

    /// THE ENGINE ASYMMETRY, said once. Declared here rather than inline in
    /// `tokenIDs` because two callers need the identical sentence: the
    /// extraction path that would otherwise render the wrong thing, and
    /// ``ExtractionRendering/declared(object:)``, which refuses the form the
    /// moment someone types it rather than letting a manifest carry it to a
    /// GPU. One string, so the two can never drift into two explanations.
    /// THE OTHER ENGINE ASYMMETRY, said once, for the same two callers:
    /// ``ExtractionRendering/declared(object:)`` (which refuses the voice the
    /// moment it is typed) and ``tokenIDs(tokenizer:modelID:text:rendering:)``
    /// (the guard that can never be reached through a declaration, and exists
    /// so a hand-built rendering cannot slip past either).
    ///
    /// WHY THIS ENGINE CANNOT. The assistant voice needs the template's own
    /// assistant-turn markers around the stimulus and NOTHING before them,
    /// which the server obtains by rendering the conversation with and
    /// without the turn (`add_generation_prompt=False` both times) and
    /// subtracting the strings. MLXLMCommon's tokenizer bridge — the one a
    /// loaded `ModelContext` hands us — exposes only the generation-prompt
    /// form, and only as token ids: the first render is unavailable, and
    /// reconstructing the turn by id arithmetic holds for Gemma 3 (whose
    /// generation prompt IS the assistant turn's opening) and breaks for
    /// Qwen 3, whose generation prompt injects a thinking scaffold the
    /// assistant turn does not carry. Rather than render the WRONG sequence
    /// on some families and stamp that the recipe asked for the assistant
    /// voice, the form is a named refusal.
    public static let assistantVoiceReason =
        "extractionRendering voice 'assistant' is not available on the "
        + "swift-mlx engine: MLXLMCommon's tokenizer bridge exposes only the "
        + "generation-prompt form of applyChatTemplate, so a COMPLETED "
        + "assistant turn cannot be rendered here, and reconstructing one by "
        + "token arithmetic holds for Gemma 3 but breaks on Qwen 3's thinking "
        + "scaffold. Repair: extract this concept on the "
        + "python-hf-transformers engine, which supports it, or declare voice "
        + "'user' here — never let it fall back, because the two voices are "
        + "different directions, not two spellings of one."

    public static let addGenerationPromptFalseReason =
        "addGenerationPrompt false is not available through MLXLMCommon's "
        + "tokenizer bridge, which exposes only the generation-prompt form of "
        + "applyChatTemplate. Repair: extract this concept on the "
        + "python-hf-transformers engine, which supports it, or declare "
        + "addGenerationPrompt true here — never let it fall back, because the "
        + "trailing template tokens change both the vector and every named "
        + "reading position that lands on them."

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
        guard !rendering.isAssistantVoice else {
            throw UnsupportedForm(reason: assistantVoiceReason)
        }
        guard rendering.resolvedAddGenerationPrompt else {
            throw UnsupportedForm(reason: addGenerationPromptFalseReason)
        }
        let record = capabilities(for: modelID)
        let messages = chatMessages(
            prompt: text, modelID: modelID,
            systemPrompt: rendering.systemPrompt, capabilities: record)
        return try tokenizer.applyChatTemplate(
            messages: DefaultMessageGenerator().generate(messages: messages),
            tools: nil,
            additionalContext: thinkingContext(
                modelID: modelID,
                effort: rendering.resolvedReasoningEffort,
                capabilities: record))
    }
}

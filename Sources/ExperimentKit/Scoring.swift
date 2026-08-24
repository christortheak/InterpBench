import Foundation
import SteeringKit

/// Concept-expression scoring as DATA: each concept may ship a
/// `markers.json` ({"words": […], "characters": "…"}) beside its stimuli.
/// Keyword rubric first, model-graded second, never model-graded only
/// (CLAUDE.md › Experiment B). Adding a concept requires zero code.
public struct MarkerRubric: Sendable {
    public let words: Set<String>
    public let characters: Set<Character>

    public init?(directory: URL) {
        struct File: Decodable {
            var words: [String]?
            var characters: String?
        }
        let url = directory.appending(component: "markers.json")
        guard let data = try? Data(contentsOf: url),
            let file = try? JSONDecoder().decode(File.self, from: data)
        else { return nil }
        self.words = Set((file.words ?? []).map { $0.lowercased() })
        self.characters = Set(file.characters ?? "")
        if words.isEmpty && characters.isEmpty { return nil }
    }

    public init(words: Set<String>, characters: Set<Character> = []) {
        self.words = words
        self.characters = characters
    }

    public func count(in text: String) -> Int {
        let tokens = Self.tokens(of: text)
        let wordHits = tokens.count { words.contains($0) }
        let characterHits = text.count { characters.contains($0) }
        return wordHits + characterHits
    }

    /// Markers per word — comparable across generations of different length.
    public func density(in text: String) -> Float {
        let tokenCount = Self.tokens(of: text).count
        guard tokenCount > 0 else { return 0 }
        return Float(count(in: text)) / Float(tokenCount)
    }

    static func tokens(of text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.letters.inverted)
            .filter { !$0.isEmpty }
    }
}

/// Short reasoning/recall probes unrelated to any concept. If accuracy
/// drops under steering, a "bias" finding is confounded by degradation —
/// report both (CLAUDE.md › Experiment B specification).
///
/// TWO formats load here, and which one a file declares decides both how it
/// is scored and how it is ARMED (Swift twin of
/// `Server/steerlab_server/experiment/battery.py`, the contract authority —
/// changes must land on both engines).
///
/// **Format 1 (legacy — no `batteryFormat` header).** One
/// `{"prompt": …, "answer": …, "grading"?: …}` per line, generated with
/// `VariantRobustness.batteryMaxTokens` and graded by the pure text matcher
/// below (`isCorrect`). Its arming is the SURROUNDING INSTRUMENT's: the study
/// manifest's promptMode/systemPrompt/thinking flag for baseline and steering
/// conditions, the variant artifact's for variant conditions. That arming is
/// the defect diagnosed in `docs/BATTERY-REPAIR-DIAGNOSIS-2026-08-13.md` — a
/// study's "answer in JSON" system prompt makes the untouched model answer
/// "What is the capital of France?" with a JSON memo, so the same pinned
/// battery reads 0.45 on one instrument and 1.00 on another. **Legacy
/// behaviour is preserved bit for bit** so an existing pinned hash keeps its
/// historical meaning; the contaminated case is LOUD
/// (`contaminationAdvisory`) instead of silent.
///
/// **Format 2 (repaired).** The first non-empty line is a HEADER object
/// carrying `{"batteryFormat": 2, …}`; the remaining lines are items. A v2
/// battery declares its own arming (`promptMode`, `systemPrompt`,
/// `qwenThinkingEnabled`, `maxTokens`), applied identically to baseline,
/// steering, and variant conditions, and defaults to `choiceProbability`
/// scoring through the answer-token logprob instrument — nothing is
/// generated, so the score cannot move with response length, verbosity, or
/// format compliance. `generatedText` remains available per item/file for
/// probes that genuinely need free text, but v2 requires an EXPLICIT
/// `grading` there — no inferred normalization.
public struct CapabilityBattery: Sendable {
    /// Format versions. 1 = the legacy headerless file; 2 = the
    /// header-declared repaired format (server `FORMAT_LEGACY` /
    /// `FORMAT_CURRENT`). Additive by construction: a v1 file's bytes — and
    /// therefore its pinned hash — still mean exactly what they meant before.
    public static let legacyFormat = 1
    public static let currentFormat = 2

    /// How an item is turned into a 0/1 (server `SCORING_MODES`).
    /// `choiceProbability` reads the model's distribution over the item's
    /// declared options (no generation, so no length or format sensitivity);
    /// `generatedText` generates and text-matches — the legacy behaviour,
    /// kept for probes that cannot be posed as a choice.
    public enum Scoring: String, Codable, Sendable {
        case choiceProbability
        case generatedText
    }

    public enum GradingMode: String, Codable, Sendable {
        /// Numeric equivalence against standalone numbers in the response.
        case exactNumber = "exact_number"
        /// First standalone yes/no token must match.
        case yesNo = "yes_no"
        /// Expected answer must appear as a standalone normalized token.
        case tokenExact = "token_exact"
        /// Full normalized response must equal the normalized answer.
        case exactNormalized = "exact_normalized"
        /// `answer` is an NSRegularExpression pattern, matched case-insensitively.
        case regex
    }

    public struct Item: Decodable, Sendable {
        public let prompt: String
        public let answer: String
        public let grading: GradingMode?
        /// Format-2 only: stable item identity (records key on it; absent
        /// falls back to the ordinal, server `battery-<index>`).
        public let id: String?
        /// Per-item override of the file's declared scoring mode.
        public let scoring: Scoring?
        /// The choice set a `choiceProbability` item is read over; the
        /// answer is one of them.
        public let options: [String]?

        public init(
            prompt: String, answer: String, grading: GradingMode? = nil,
            id: String? = nil, scoring: Scoring? = nil, options: [String]? = nil
        ) {
            self.prompt = prompt
            self.answer = answer
            self.grading = grading
            self.id = id
            self.scoring = scoring
            self.options = options
        }

        private enum CodingKeys: String, CodingKey {
            case prompt, answer, grading, id, scoring, options
        }

        /// Hand-written so the FORMAT-1 decode is exactly what it always was:
        /// `prompt`/`answer` strict strings, `grading` a strict enum, and the
        /// format-2 keys tolerated rather than validated here. Shape
        /// validation for v2 lives in ONE place (`PinShapeValidation`
        /// .batteryFormat2Problem, the server's `_parse_item` twin) and runs
        /// before this decoder ever sees a v2 line, so a strict decode here
        /// would only add a second, worse-worded refusal — and would newly
        /// refuse legacy files the server still loads.
        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            prompt = try container.decode(String.self, forKey: .prompt)
            answer = try container.decode(String.self, forKey: .answer)
            grading = try container.decodeIfPresent(GradingMode.self, forKey: .grading)
            id = try? container.decodeIfPresent(String.self, forKey: .id)
            scoring = try? container.decodeIfPresent(Scoring.self, forKey: .scoring)
            options = try? container.decodeIfPresent([String].self, forKey: .options)
        }
    }

    public let items: [Item]

    /// The file this battery was loaded from (its last path component, or the
    /// relative path a caller named) — quoted in the contamination advisory.
    public let file: String

    /// 1 = legacy headerless, 2 = header-declared (server `format_version`).
    public let formatVersion: Int

    /// The FILE's default scoring mode; a per-item `scoring` overrides it.
    public let defaultScoring: Scoring

    /// The declared generation cap (format 2) — the shared legacy cap
    /// otherwise.
    public let maxTokens: Int

    /// The declared prompt mode as WRITTEN (server keeps the raw string).
    /// `promptMode` resolves it for this engine's renderer.
    public let promptModeName: String

    /// The battery's own system prompt (format 2; nil = none, the default and
    /// the recommended value).
    public let systemPrompt: String?

    public let qwenThinkingEnabled: Bool

    /// True when the battery's arming comes from the BATTERY, so a reading is
    /// comparable across instruments. Format 2 only (server `isolated`).
    public var isolated: Bool { formatVersion >= Self.currentFormat }

    /// The declared mode resolved for this engine's renderer. An unknown
    /// spelling resolves to `chatAssistant`, matching the server's renderer,
    /// which treats anything but `rawCompletion` as chat.
    public var promptMode: ExperimentManifest.PromptMode {
        ExperimentManifest.PromptMode(rawValue: promptModeName) ?? .chatAssistant
    }

    /// The scoring mode ONE item is read under (server `item_scoring`).
    public func scoring(for item: Item) -> Scoring {
        item.scoring ?? defaultScoring
    }

    public init(url: URL) throws {
        try self.init(
            data: try Data(contentsOf: url), file: url.lastPathComponent)
    }

    /// Parses a battery of EITHER format (server `battery.load_spec` twin).
    ///
    /// A v2 header is identified by the `batteryFormat` KEY exactly as the
    /// server identifies it, and then validated through the ONE format-2
    /// shape checker this engine has (`PinShapeValidation`, which the pin,
    /// the readiness preflight and freeze already run) rather than a second
    /// parser that could drift from it. The legacy path below is byte-for-byte
    /// the loader it always was, error type included.
    public init(data: Data, file: String) throws {
        self.file = file
        let lines = PinShapeValidation.batteryLines(data)
        let decoder = JSONDecoder()
        if let first = lines.first,
            let header = PinShapeValidation.batteryHeaderObject(first)
        {
            if let problem = PinShapeValidation.batteryFormat2Problem(
                lines, file: file)
            {
                throw ExperimentError(reason: problem)
            }
            // Every field below was just validated; the fallbacks are the
            // server's declared defaults, never a repair of bad input.
            formatVersion = Self.currentFormat
            defaultScoring =
                if case .string(let declared)? = header["scoring"],
                    let mode = Scoring(rawValue: declared)
                {
                    mode
                } else {
                    .choiceProbability
                }
            maxTokens =
                if case .number(let value)? = header["maxTokens"] {
                    Int(value)
                } else {
                    VariantRobustness.batteryMaxTokens
                }
            promptModeName =
                if case .string(let value)? = header["promptMode"], !value.isEmpty {
                    value
                } else {
                    ExperimentManifest.PromptMode.chatAssistant.rawValue
                }
            systemPrompt =
                if case .string(let value)? = header["systemPrompt"], !value.isEmpty {
                    value
                } else {
                    nil
                }
            qwenThinkingEnabled =
                if case .bool(let value)? = header["qwenThinkingEnabled"] {
                    value
                } else {
                    false
                }
            items = try lines.dropFirst().enumerated().map { index, line in
                guard
                    let item = try? decoder.decode(Item.self, from: Data(line.utf8))
                else {
                    throw StimulusSetError.malformedLine(
                        file: file, line: index + 2)
                }
                return item
            }
            return
        }
        formatVersion = Self.legacyFormat
        defaultScoring = .generatedText
        maxTokens = VariantRobustness.batteryMaxTokens
        promptModeName = ExperimentManifest.PromptMode.chatAssistant.rawValue
        systemPrompt = nil
        qwenThinkingEnabled = false
        let rawLines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        items = try rawLines.enumerated().map { index, line in
            guard
                let item = try? decoder.decode(
                    Item.self, from: Data(line.trimmingCharacters(in: .whitespaces).utf8))
            else {
                throw StimulusSetError.malformedLine(file: file, line: index + 1)
            }
            return item
        }
    }

    public static func isCorrect(
        response: String, answer: String, grading: GradingMode? = nil
    ) -> Bool {
        let mode = grading ?? inferredGradingMode(for: answer)
        switch mode {
        case .exactNumber:
            guard let expected = parseNumber(answer) else { return false }
            guard let finalNumber = finalAnswerNumber(in: response) else { return false }
            return numericEquals(finalNumber, expected)
        case .yesNo:
            guard let expected = yesNoValue(answer) else { return false }
            guard let first = normalizedTokens(of: response)
                .compactMap(yesNoValue)
                .first
            else { return false }
            return first == expected
        case .tokenExact:
            let expected = normalizedToken(answer)
            guard !expected.isEmpty else { return false }
            return normalizedTokens(of: response).contains(expected)
        case .exactNormalized:
            return normalizedText(response) == normalizedText(answer)
        case .regex:
            guard
                let regex = try? NSRegularExpression(
                    pattern: answer, options: [.caseInsensitive])
            else { return false }
            let range = NSRange(response.startIndex ..< response.endIndex, in: response)
            return regex.firstMatch(in: response, range: range) != nil
        }
    }

    private static func inferredGradingMode(for answer: String) -> GradingMode {
        if parseNumber(answer) != nil { return .exactNumber }
        if yesNoValue(answer) != nil { return .yesNo }
        if normalizedTokens(of: answer).count == 1 { return .tokenExact }
        return .exactNormalized
    }

    private static func normalizedText(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private static func normalizedTokens(of text: String) -> [String] {
        normalizedText(text).split(separator: " ").map(String.init)
    }

    private static func normalizedToken(_ text: String) -> String {
        normalizedTokens(of: text).joined(separator: " ")
    }

    private static func yesNoValue(_ text: String) -> Bool? {
        switch normalizedText(text) {
        case "yes", "y", "true": return true
        case "no", "n", "false": return false
        default: return nil
        }
    }

    private static func parseNumber(_ text: String) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            !trimmed.isEmpty,
            trimmed.range(
                of: #"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$"#,
                options: .regularExpression) != nil
        else { return nil }
        return Double(trimmed)
    }

    private static func numbers(in text: String) -> [Double] {
        let pattern = #"(?<![A-Za-z0-9.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9])"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex ..< text.endIndex, in: text)
        return regex.matches(in: text, range: range).compactMap { match in
            guard let substringRange = Range(match.range, in: text) else { return nil }
            return Double(text[substringRange])
        }
    }

    private static func finalAnswerNumber(in text: String) -> Double? {
        let number = #"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"#
        let labeledPattern =
            #"(?i)(?:final\s+answer|answer|equals?|=|therefore)\s*(?:is|:)?\s*("#
            + number + #")"#
        if let regex = try? NSRegularExpression(pattern: labeledPattern) {
            let range = NSRange(text.startIndex ..< text.endIndex, in: text)
            let matches = regex.matches(in: text, range: range)
            if let match = matches.last,
                let numberRange = Range(match.range(at: 1), in: text)
            {
                return Double(text[numberRange])
            }
        }

        let values = numbers(in: text)
        if values.count == 1 { return values[0] }
        return nil
    }

    private static func numericEquals(_ lhs: Double, _ rhs: Double) -> Bool {
        abs(lhs - rhs) < 1e-9
    }

    // MARK: - Arming (server `battery.resolve_arming` twin)

    /// The arming this battery is scored under.
    ///
    /// Format 2 ignores the caller's *instrument* context entirely and uses
    /// what the battery FILE declares — the same rendering context for
    /// baseline, steering and variant conditions, which is what makes a
    /// reading comparable across instruments and across conditions.
    ///
    /// **Battery isolation under composition (2026-08-24 ruling).** One thing
    /// does reach a format-2 reading from outside the file: the arm's AGENT
    /// persona, composed FIRST, ahead of the battery's own declared arming
    /// text (`SystemPromptComposition.compose`). The agent is the model under
    /// test — an agent whose capability is measured without its identity is
    /// not the thing the study runs — whereas the STUDY FRAME is the
    /// deployment context of the study's own task and has no business shaping
    /// a capability control. So a baseline arm (no agent) reads under the
    /// battery's arming ALONE, and an agent arm reads under persona + battery
    /// arming. `agentSystemPrompt` is the only channel for it; `systemPrompt`
    /// remains the format-1 caller context and is ignored here exactly as it
    /// always was.
    ///
    /// Format 1 keeps the historical behaviour untouched, `agentSystemPrompt`
    /// included (it is ignored): the surrounding instrument's rendering
    /// context leaks in, so its readings stay reproducible and its pinned
    /// hash keeps its meaning. Server twin: `battery.resolve_arming`.
    public func resolveArming(
        promptMode: ExperimentManifest.PromptMode? = nil,
        systemPrompt: String? = nil,
        qwenThinkingEnabled: Bool = false,
        agentSystemPrompt: String? = nil
    ) -> BatteryArming {
        guard isolated else {
            return BatteryArming(
                promptModeName: (promptMode ?? .chatAssistant).rawValue,
                systemPrompt: systemPrompt,
                qwenThinkingEnabled: qwenThinkingEnabled,
                maxTokens: VariantRobustness.batteryMaxTokens,
                isolated: false)
        }
        return BatteryArming(
            promptModeName: promptModeName,
            systemPrompt: SystemPromptComposition.compose(
                agent: agentSystemPrompt, frame: self.systemPrompt),
            qwenThinkingEnabled: self.qwenThinkingEnabled,
            maxTokens: maxTokens,
            isolated: true,
            agentSystemPrompt: agentSystemPrompt,
            declaredSystemPrompt: self.systemPrompt)
    }

    /// The warning a legacy battery earns when the surrounding instrument's
    /// system prompt is applied to it — the mechanism behind the 0.45-vs-1.00
    /// split. nil when the reading is clean. Server
    /// `battery.contamination_advisory`, same wording.
    public func contaminationAdvisory(_ arming: BatteryArming) -> String? {
        guard !arming.isolated, arming.systemPrompt?.isEmpty == false else {
            return nil
        }
        return "capability battery '\(file)' is format \(formatVersion) "
            + "(legacy): it is scored under the STUDY's system prompt, so its "
            + "accuracy is not comparable across instruments and a condition "
            + "that breaks format compliance can SCORE HIGHER. Re-pin a "
            + "format-2 battery (steerlab-server battery lint <path>) before "
            + "citing this number as a capability control."
    }

    // MARK: - Item scoring (server `battery.score_item` twin)

    /// Scores ONE battery item; returns the fields to stamp on its record.
    ///
    /// Both back-ends are injected so the arithmetic is testable without a
    /// model — exactly as on the server. `choiceProbability` is the repaired
    /// path: `correct` is `selected == answer` over the model's own
    /// answer-token distribution, so it is invariant to how long or how
    /// format-compliant the surrounding instrument makes the model, and
    /// NOTHING is generated.
    public func scoreItem(
        _ item: Item,
        arming: BatteryArming,
        generate: (_ prompt: String, _ arming: BatteryArming) async throws -> String,
        choice: (_ prompt: String, _ options: [String], _ arming: BatteryArming)
            async throws -> (selected: String, probability: [String: Double])
    ) async throws -> BatteryItemScore {
        guard scoring(for: item) == .choiceProbability, let options = item.options
        else {
            let text = try await generate(item.prompt, arming)
            return BatteryItemScore(
                scoring: .generatedText,
                output: text,
                correct: Self.isCorrect(
                    response: text, answer: item.answer, grading: item.grading))
        }
        let read = try await choice(item.prompt, options, arming)
        return BatteryItemScore(
            scoring: .choiceProbability,
            options: options,
            choiceProbability: read.probability,
            selected: read.selected,
            output: read.selected,
            correct: read.selected == item.answer)
    }
}

/// The generation/rendering context a battery is actually scored under
/// (server `battery.BatteryArming`).
public struct BatteryArming: Sendable, Equatable {
    /// The prompt mode as WRITTEN — what a record stamps. `promptMode`
    /// resolves it for this engine's renderer.
    public let promptModeName: String
    /// The EFFECTIVE system prompt of the reading — for format 2, the agent's
    /// persona composed with the battery's own declared arming text; for
    /// format 1, the surrounding instrument's context, exactly as before.
    public let systemPrompt: String?
    public let qwenThinkingEnabled: Bool
    public let maxTokens: Int
    /// True when this arming came from the battery file itself.
    public let isolated: Bool
    /// The two LEVELS behind `systemPrompt` on a format-2 reading: the arm's
    /// agent persona and the battery file's own declared text. Both nil on a
    /// format-1 reading, whose arming has no composition to describe.
    public let agentSystemPrompt: String?
    public let declaredSystemPrompt: String?

    public init(
        promptModeName: String, systemPrompt: String?,
        qwenThinkingEnabled: Bool, maxTokens: Int, isolated: Bool,
        agentSystemPrompt: String? = nil, declaredSystemPrompt: String? = nil
    ) {
        self.promptModeName = promptModeName
        self.systemPrompt = systemPrompt
        self.qwenThinkingEnabled = qwenThinkingEnabled
        self.maxTokens = maxTokens
        self.isolated = isolated
        self.agentSystemPrompt = agentSystemPrompt
        self.declaredSystemPrompt = declaredSystemPrompt
    }

    public var promptMode: ExperimentManifest.PromptMode {
        ExperimentManifest.PromptMode(rawValue: promptModeName) ?? .chatAssistant
    }

    /// JSON-safe provenance for a battery.jsonl record (server
    /// `BatteryArming.as_record_fields`, identical KEY NAMES). Only the
    /// system prompt's PRESENCE is stamped as a bool — the text itself is the
    /// instrument's, is already in the manifest snapshot, and would bloat
    /// every row — plus, since the 2026-08-24 composition ruling, the HASHES
    /// that say which levels produced it.
    ///
    /// `composition` spells its second key `battery`, not `study`: a battery
    /// generation's second term is the battery file's declared arming, and
    /// the study frame never enters one. The difference in spelling from a
    /// study record's `systemPromptComposition` is the point.
    public var recordFields:
        (isolated: Bool, promptMode: String, systemPrompt: Bool, maxTokens: Int,
         systemPromptHash: String?, composition: BatteryArmingCompositionStamp)
    {
        (isolated, promptModeName, systemPrompt?.isEmpty == false, maxTokens,
         SystemPromptComposition.hash(systemPrompt),
         BatteryArmingCompositionStamp(
            agentText: agentSystemPrompt, batteryText: declaredSystemPrompt))
    }
}

/// One battery item's reading — the Swift twin of what the server's
/// `battery.score_item` returns, in the shape a `battery.jsonl` record wants.
public struct BatteryItemScore: Sendable, Equatable {
    public let scoring: CapabilityBattery.Scoring
    /// choiceProbability only: the option set that was read, in declared
    /// order, and the model's normalized distribution over it.
    public let options: [String]?
    public let choiceProbability: [String: Double]?
    public let selected: String?
    /// The generated text, or — under choiceProbability — the selected
    /// option (server parity: `output` is populated either way).
    public let output: String
    public let correct: Bool

    public init(
        scoring: CapabilityBattery.Scoring,
        options: [String]? = nil,
        choiceProbability: [String: Double]? = nil,
        selected: String? = nil,
        output: String,
        correct: Bool
    ) {
        self.scoring = scoring
        self.options = options
        self.choiceProbability = choiceProbability
        self.selected = selected
        self.output = output
        self.correct = correct
    }
}

/// Degeneration metric: distinct bigrams / total bigrams over whitespace
/// tokens. Repetition collapse (the high-alpha failure mode) drives this
/// toward 0; healthy prose sits near 1.
public func distinctBigramRatio(_ text: String) -> Float {
    let tokens = text.split(whereSeparator: \.isWhitespace)
    guard tokens.count >= 3 else { return 0 }
    var bigrams = Set<String>()
    var total = 0
    for index in 0 ..< tokens.count - 1 {
        bigrams.insert("\(tokens[index]) \(tokens[index + 1])")
        total += 1
    }
    return Float(bigrams.count) / Float(total)
}

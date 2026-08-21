import Foundation

/// What the runner writes onto a turn record about the turn's VOICE (Wave-2
/// contract, spec §5):
///
/// ```json
/// "voiceLint": {"version": 1, "speaksForOthers": true,
///               "otherSpeakerLines": {"Judge Marsden": 3},
///               "thirdPersonSelf": 2}
/// ```
///
/// `otherSpeakerLines` is omitted when empty — additive-key discipline, the
/// same rule the `endpoint` stamp follows.
public struct VoiceLintStamp: Codable, Sendable, Equatable {
    public let version: Int
    /// At least one line of the output signs, labels, or stage-directs as
    /// another participant.
    public let speaksForOthers: Bool
    /// How many such lines, per participant. nil ⇒ key omitted (none found).
    public let otherSpeakerLines: [String: Int]?
    /// Mentions of the speaker's own name in non-first-person prose.
    public let thirdPersonSelf: Int

    public init(otherSpeakerLines: [String: Int], thirdPersonSelf: Int) {
        self.version = VoiceLint.version
        self.speaksForOthers = !otherSpeakerLines.isEmpty
        self.otherSpeakerLines = otherSpeakerLines.isEmpty ? nil : otherSpeakerLines
        self.thirdPersonSelf = thirdPersonSelf
    }

    private enum CodingKeys: String, CodingKey {
        case version, speaksForOthers, otherSpeakerLines, thirdPersonSelf
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.version = try container.decodeIfPresent(Int.self, forKey: .version)
            ?? VoiceLint.version
        self.speaksForOthers = try container.decodeIfPresent(
            Bool.self, forKey: .speaksForOthers) ?? false
        let lines = try container.decodeIfPresent(
            [String: Int].self, forKey: .otherSpeakerLines)
        self.otherSpeakerLines = (lines?.isEmpty ?? true) ? nil : lines
        self.thirdPersonSelf = try container.decodeIfPresent(
            Int.self, forKey: .thirdPersonSelf) ?? 0
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(version, forKey: .version)
        try container.encode(speaksForOthers, forKey: .speaksForOthers)
        if let otherSpeakerLines, !otherSpeakerLines.isEmpty {
            try container.encode(otherSpeakerLines, forKey: .otherSpeakerLines)
        }
        try container.encode(thirdPersonSelf, forKey: .thirdPersonSelf)
    }
}

/// Voice lint for multi-agent turns — the Swift twin of
/// `Server/steerlab_server/experiment/voice_lint.py`.
///
/// A panel turn is supposed to be ONE participant's document. The measured
/// runs show three ways that fails, and all three are condition-correlated —
/// steered arms lose format compliance where baseline arms keep it — so voice
/// noncompliance is an OUTCOME to record, never noise to hide. Nothing here
/// blocks, retries, or regenerates: silent regeneration would select on the
/// dependent variable.
///
/// **No regex anywhere**, matching the `TurnEndpointParser` house style:
/// scanning is literal and line-oriented, because two engines byte-agree on
/// literal scans trivially while regex dialects diverge. The committed fixture
/// `prompts/fixtures/voice-lint/cases.jsonl` — excerpts from the real failing
/// transcripts — is replayed by BOTH engines' tests.
///
/// The failure shapes, and the rules calibrated against them (corpus: the
/// workspace's four `multiagent-s3-prec-delib-*` runs of 2026-08-11, 600
/// turn records, and `20260808T175038808-…-consciousness-a`, 240 records):
///
/// 1. **Colleague signature blocks** — a judge filing its colleagues' separate
///    opinions inside its own: `**WHITFIELD, Judge,** concurring.`,
///    `**CALLOWAY, J.,** dissenting.`, `MARSDEN, Circuit Judge, concurring.`
/// 2. **Screenplay speaker labels / stage directions** — a whole-panel
///    transcript written by one seat: `**(Judge A, as presiding judge):**`,
///    `**(Judge B):** Certainly. My initial vote is to affirm.`
/// 3. **Third-person self-reference** — the speaker naming itself as a third
///    party ("given Judge A's scale position", written by Judge A).
///
/// Pinned decisions (the Python twin implements each one identically):
///
/// * **Name forms come from the agent's own name, never from a vocabulary.**
///   No domain word appears in this file: the "role" half of a signature is
///   whatever the scenario's agent name happens to carry. `"Judge Whitfield"`
///   yields the FULL form `Judge Whitfield` and the SURNAME form `Whitfield`
///   (last whitespace-delimited token). The surname form is used only when it
///   is ≥3 scalars, so a panel of `Judge A` / `Judge B` does not match every
///   line starting with "A".
/// * **The two forms carry different evidence, and that is the whole trick.**
///   Across 600 records the transcripts use the FULL name to address or
///   discuss a colleague ("Judge Calloway raises a compelling point") and the
///   BARE SURNAME only in signature position ("CALLOWAY, J., concurring"). A
///   surname followed by a comma is therefore a signature block; a full name
///   followed by a comma is legitimate address and is NOT flagged.
/// * **Only the line's leading content is scanned.** A colleague named
///   mid-line is discussion, which the panel is supposed to do.
/// * Leading Markdown/quotation decoration (`* _ # > - + ~ ` ( [ " '` and
///   whitespace) is stripped first; an opening `(` or `[` among it marks the
///   line BRACKETED, which is what distinguishes a stage direction from prose.
/// * **Case-insensitivity is ASCII folding**, positions are Unicode scalars,
///   and "whole word" means not adjacent to a Unicode letter — literally
///   `TurnEndpointParser`'s functions, so the two stamps cannot drift apart.
/// * **A Markdown list item is a roll-call, not a voice.** `*   **Judge B:**
///   Affirm` inside a disposition package reports a colleague's vote; the
///   `NAME:` rule is therefore suppressed on list items. Without this, ten of
///   ten "final disposition package" turns — a legitimate turn type — flagged.
/// * **Third-person self counts the FULL own name only**, and skips positions
///   compliant transcripts genuinely use: line-initial (a heading or the
///   speaker's own signature), immediately after a field label ending in `:`
///   (`From: Judge Whitfield` — 60 occurrences in the corpus), after a
///   first-person lead-in (`I, Judge C` / `I am Judge C` / `I'm …` /
///   `you are …`), and `As <name>,` apposition. Everything else counts. It
///   counts, it does not judge: a byline without a colon (`**Opinion by Judge
///   A**`) is counted, deliberately, because the alternative is a heuristic
///   that guesses at prose.
public enum VoiceLint {
    /// Stamp schema version. Bump only for a shape change, never for a rule
    /// tweak — a rules change is a re-lint of the corpus, which the fixture
    /// pins.
    public static let version = 1

    enum NameKind {
        case full
        case surname
    }

    /// Leading characters removed before a line's content is examined.
    private static let decoration: Set<Unicode.Scalar> = [
        "*", "_", "#", ">", "-", "+", "~", "`", "(", "[", "\"", "'", " ", "\t", "\r",
    ]
    /// Characters skipped when looking for the first significant character
    /// AFTER a matched name (emphasis closes there: `**WHITFIELD, Judge,** …`).
    private static let skip: Set<Unicode.Scalar> = ["*", "_", "`", "~", " ", "\t", "\r"]
    /// Continuations that make a bare SURNAME at the head of a line a signature.
    private static let signatureAfter: Set<Unicode.Scalar> = [",", ".", ";"]
    /// Continuations that make a BRACKETED name a speaker label / stage direction.
    private static let bracketAfter: Set<Unicode.Scalar> = [",", ")", "]"]
    /// First-person lead-ins that make an own-name mention first-person framing.
    private static let firstPerson = ["i,", "i am", "i'm", "you are"]

    // MARK: the stamp

    /// The turn-record stamp. Never throws, never blocks, never rewrites.
    ///
    /// Python twin: `voice_lint.stamp`.
    public static func stamp(
        in text: String, speaker: String, others: [String]
    ) -> VoiceLintStamp {
        VoiceLintStamp(
            otherSpeakerLines: otherSpeakerLines(in: text, others: others),
            thirdPersonSelf: thirdPersonSelf(in: text, speaker: speaker))
    }

    // MARK: name forms

    /// `[(form, kind)]` for one agent name, longest first. Derived from the
    /// name alone: the linter never knows what domain the panel is in.
    static func nameForms(_ name: String) -> [(form: String, kind: NameKind)] {
        let normalized = normalize(name)
        guard !normalized.isEmpty else { return [] }
        var forms: [(form: String, kind: NameKind)] = [(normalized, .full)]
        let tokens = normalized.split(separator: " ").map(String.init)
        if tokens.count > 1, let last = tokens.last, last.unicodeScalars.count >= 3 {
            forms.append((last, .surname))
        }
        return forms.sorted {
            $0.form.unicodeScalars.count == $1.form.unicodeScalars.count
                ? $0.form < $1.form
                : $0.form.unicodeScalars.count > $1.form.unicodeScalars.count
        }
    }

    static func normalize(_ name: String) -> String {
        name.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    // MARK: line scanning

    /// The agent this LINE signs, labels, or stage-directs as, or nil.
    ///
    /// Ambiguity resolves by longest form across all candidate names, so a
    /// panel holding both `Judge Marsden` and `Marsden` cannot flag the short
    /// one for the long one's line.
    static func lineSpeaker(_ line: [Unicode.Scalar], names: [String]) -> String? {
        var start = 0
        var bracketed = false
        while start < line.count, decoration.contains(line[start]) {
            if line[start] == "(" || line[start] == "[" { bracketed = true }
            start += 1
        }
        guard start < line.count else { return nil }
        let listed = isListItem(line)
        var candidates: [(form: [Unicode.Scalar], kind: NameKind, name: String)] = []
        for name in names {
            for form in nameForms(name) {
                candidates.append((Array(form.form.unicodeScalars), form.kind, name))
            }
        }
        candidates.sort {
            if $0.form.count != $1.form.count { return $0.form.count > $1.form.count }
            let left = String(String.UnicodeScalarView($0.form))
            let right = String(String.UnicodeScalarView($1.form))
            return left == right ? $0.name < $1.name : left < right
        }
        for candidate in candidates {
            guard hasPrefix(line, from: start, candidate.form) else { continue }
            let end = start + candidate.form.count
            // A longer word that merely starts with the name.
            if end < line.count, TurnEndpointParser.isLetter(line[end]) { continue }
            let after = firstSignificant(line, from: end)
            // A missing continuation (the name ends the line) reads as the
            // bare form of whichever rule is in play, exactly as in the twin.
            let signs = after.map(signatureAfter.contains) ?? true
            let labels = after.map(bracketAfter.contains) ?? true
            if after == ":", !listed { return candidate.name }
            if candidate.kind == .surname, signs { return candidate.name }
            if bracketed, labels { return candidate.name }
            return nil  // the name leads the line, but as prose
        }
        return nil
    }

    /// How many lines of `text` sign or label as each OTHER participant.
    public static func otherSpeakerLines(
        in text: String, others: [String]
    ) -> [String: Int] {
        let names = others.filter { !normalize($0).isEmpty }
        guard !names.isEmpty else { return [:] }
        var counts: [String: Int] = [:]
        for line in splitLines(text) {
            if let name = lineSpeaker(line, names: names) {
                counts[name, default: 0] += 1
            }
        }
        return counts
    }

    /// A Markdown bullet or ordered-list marker opens this line.
    static func isListItem(_ line: [Unicode.Scalar]) -> Bool {
        var index = 0
        while index < line.count, line[index] == " " || line[index] == "\t" {
            index += 1
        }
        if index < line.count, line[index] == "*" || line[index] == "-" || line[index] == "+",
            index + 1 < line.count, line[index + 1] == " " || line[index + 1] == "\t"
        {
            return true
        }
        var digits = index
        while digits < line.count, isASCIIDigit(line[digits]) { digits += 1 }
        return digits > index && digits + 1 < line.count
            && (line[digits] == "." || line[digits] == ")")
            && (line[digits + 1] == " " || line[digits + 1] == "\t")
    }

    // MARK: third-person self

    /// Occurrences of the speaker's own full name in non-first-person prose.
    public static func thirdPersonSelf(in text: String, speaker: String) -> Int {
        let name = Array(normalize(speaker).unicodeScalars)
        guard !name.isEmpty else { return 0 }
        let body = Array(text.unicodeScalars)
        var count = 0
        for index in wholeWordOccurrences(body, name) {
            // Start of the line holding this occurrence.
            var lineStart = index
            while lineStart > 0, body[lineStart - 1] != "\n" { lineStart -= 1 }
            var content = lineStart
            while content < body.count, decoration.contains(body[content]) { content += 1 }
            if content == index { continue }  // a heading, or an own signature

            var before = index
            while before > 0, isTrailingSkippable(body[before - 1]) { before -= 1 }
            if before > 0, body[before - 1] == ":" { continue }  // "From: <name>"

            if firstPerson.contains(where: { endsWithPhrase(body, before, $0) }) { continue }
            if endsWithPhrase(body, before, "as"),
                firstSignificant(body, from: index + name.count) == ","
            {
                continue  // "As Judge C, I concur …" — first-person framing
            }
            count += 1
        }
        return count
    }

    // MARK: literal scanning

    private static func splitLines(_ text: String) -> [[Unicode.Scalar]] {
        var lines: [[Unicode.Scalar]] = [[]]
        for scalar in text.unicodeScalars {
            if scalar == "\n" {
                lines.append([])
            } else {
                lines[lines.count - 1].append(scalar)
            }
        }
        return lines
    }

    private static func isASCIIDigit(_ scalar: Unicode.Scalar) -> Bool {
        scalar.value >= 48 && scalar.value <= 57
    }

    /// Trailing characters ignored when looking back from a match: whitespace
    /// and emphasis markers, matching the Python twin's rstrip loop.
    private static func isTrailingSkippable(_ scalar: Unicode.Scalar) -> Bool {
        skip.contains(scalar) || scalar == "\n"
    }

    private static func hasPrefix(
        _ scalars: [Unicode.Scalar], from start: Int, _ needle: [Unicode.Scalar]
    ) -> Bool {
        guard start >= 0, start + needle.count <= scalars.count else { return false }
        for offset in 0 ..< needle.count {
            if TurnEndpointParser.fold(scalars[start + offset])
                != TurnEndpointParser.fold(needle[offset]) {
                return false
            }
        }
        return true
    }

    private static func firstSignificant(
        _ scalars: [Unicode.Scalar], from start: Int
    ) -> Unicode.Scalar? {
        var index = start
        while index < scalars.count {
            if !skip.contains(scalars[index]) { return scalars[index] }
            index += 1
        }
        return nil
    }

    /// `scalars[..<end]` ends with `phrase` (ASCII-folded) on a word boundary.
    private static func endsWithPhrase(
        _ scalars: [Unicode.Scalar], _ end: Int, _ phrase: String
    ) -> Bool {
        let needle = Array(phrase.unicodeScalars)
        let start = end - needle.count
        guard start >= 0, hasPrefix(scalars, from: start, needle) else { return false }
        return start == 0 || !TurnEndpointParser.isLetter(scalars[start - 1])
    }

    private static func wholeWordOccurrences(
        _ haystack: [Unicode.Scalar], _ needle: [Unicode.Scalar]
    ) -> [Int] {
        guard !needle.isEmpty, haystack.count >= needle.count else { return [] }
        var out: [Int] = []
        var start = 0
        while start + needle.count <= haystack.count {
            if hasPrefix(haystack, from: start, needle) {
                let end = start + needle.count
                let beforeOK = start == 0 || !TurnEndpointParser.isLetter(haystack[start - 1])
                let afterOK = end >= haystack.count || !TurnEndpointParser.isLetter(haystack[end])
                if beforeOK && afterOK { out.append(start) }
            }
            start += 1
        }
        return out
    }
}

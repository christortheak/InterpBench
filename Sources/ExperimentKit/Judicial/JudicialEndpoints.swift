import Foundation

/// Built-in outcome-endpoint parsers + derived endpoints (Swift twin of the
/// server's `steerlab_server/experiment/judicial.py` — the two share fixture
/// values, so changes here must land in both). Everything is pure
/// text/arithmetic — no model, no GPU.
///
/// These are the engine's two SHIPPED parsers, both concept-agnostic text
/// readers over whatever prose a study's items elicit:
///
/// - **categorical choice**: the option a response settled on, read from a
///   JSON object with a declared key (tolerantly, when the object does not
///   parse), an exact whole-response match, or (over complete text only — a
///   truncated output parses as a failure, never as its first-enumerated
///   option) the last answer marker, an option opening the reply, or the
///   earliest mention, where a single-letter option counts only as its
///   capital letter. Derived endpoints over it are outcome rates and
///   between-item contrasts.
/// - **duration in months**: a numeric quantity stated in years and/or
///   months; endpoints are the mean shift, spread, anchor slope, and
///   proportionality; the parse-failure rate is itself a coherence signal.
///
/// Neither parser knows any domain. A study picks one by declaring a parser in
/// its manifest (see `ParserRegistry` for the declared-grammar path, which
/// generalizes both). The case families of the judicial-decision study that
/// motivated them are one worked example: a choice-of-law answer and a
/// rule-adherence choice use the categorical parser, and a sentence length in
/// months uses the duration parser.
public enum Judicial {

    // MARK: - Categorical option parsing

    /// Extracts a categorical choice from sampled prose.
    ///
    /// Tries, in order:
    ///
    /// 1. a JSON object carrying `jsonKey` (the usual answer-in-JSON schema);
    /// 2. `jsonKey` read TOLERANTLY when the object does not parse — the
    ///    last `"<jsonKey>": "<value>"` whose value names an option;
    /// 3. an exact normalized match of the whole response;
    ///
    /// and then, only for text that finished saying its piece:
    ///
    /// 4. an explicit answer marker — the word "answer" followed by `:` or
    ///    "is", Markdown emphasis ignored — where the LAST marker that names
    ///    an option wins ("Final answer: B", "**Final answer:** B", "my
    ///    answer is B");
    /// 5. an option at the very start of the reply followed by punctuation
    ///    ("B. Because…", "b) …", "**B**");
    /// 6. the option whose earliest mention appears first in the text.
    ///
    /// Returns the matched option verbatim, or nil (a parse failure — count
    /// it). The shared hand-written fixture
    /// `Tests/Fixtures/cross-engine/choice-parsing.json` states the intended
    /// reading of every rule; the server twin reads the same file.
    ///
    /// **Single-letter options and the article "a" (2026-10-17).** Rules 4–6
    /// once matched every option case-insensitively at a word boundary, so
    /// the English article "a" was a mention of option A: "As a doctor, I
    /// would choose B." parsed as A — a bias toward A in every letter-option
    /// study with free-text replies. A single-letter option now counts in
    /// running text only as its CAPITAL letter, standing alone; a lower-case
    /// letter counts only where it cannot be the article: as the whole reply,
    /// or directly followed by punctuation or the end of the reply at the
    /// start of the reply or after an answer marker. A capital "A" that opens
    /// a sentence still cannot be told from option A without understanding
    /// the sentence; the fixture keeps that case as a known limit.
    ///
    /// **Malformed JSON answers (2026-10-17).** A model told to answer in
    /// JSON often writes an object that does not parse (unescaped quotes in
    /// its reasons, raw newlines in a string). Rule 1 then found nothing and
    /// first mention read the article in the reasons as option A — more often
    /// under steering, which makes the JSON messier. Rule 2 reads the declared
    /// key from the text instead, and it is honored on a truncated output
    /// too: a complete key and value written before the cut are a decision.
    ///
    /// The first-mention fallback is legitimate only for a COMPLETE free-form
    /// answer that names its choice in prose. Over an output that was cut off
    /// before it decided, first-mention does not read a decision — it reads
    /// the order the options were enumerated in, since a deliberation
    /// restates every option long before it rules. A 1,200-record run
    /// observed 2026-08-29 made that concrete: 19 outputs hit the token cap
    /// mid-deliberation, every one was coerced to the first-enumerated
    /// option, and the declared unparseableEndpoint exclusion excluded zero
    /// records. (The duration parser's `numberWords` documents the same
    /// non-random-attrition hazard; here it was worse — a WRONG VALUE instead
    /// of a null.) Two truncation signals therefore force nil instead of the
    /// fallback: `truncated: true` (the caller's honest signal that the
    /// decode ended by exhausting its token budget — the run loop reads the
    /// stream's own stop reason), and a JSON object the text opens but never
    /// closes (detectable from the text alone, covering records parsed
    /// without a stop reason). A complete embedded JSON answer, a complete
    /// answer key, or an exact whole-response match is still honored on a
    /// truncated output: answering and then running out of room while
    /// elaborating is a decision, not a coerced one. Markers and the
    /// start-of-reply rule are not read in a truncated output, exactly as
    /// first mention is not: a deliberation that was cut off may restate an
    /// option after a marker or open with a list of the options.
    ///
    /// The new rules (2, 4, 5, and the single-letter scan in 6) are literal
    /// scans with no regex, in the panel turn parser's house style, so the
    /// two engines agree character for character: ASCII folding and the
    /// letter test are `TurnEndpointParser.fold`/`isLetter`, and positions
    /// are Unicode scalars (Python code points). The marker logic itself is
    /// not shared with the panel parser: a panel turn declares its own marker
    /// and reads the FIRST occurrence in an 80-scalar window, while a choice
    /// reply has no declared marker, reads the LAST "answer" marker that
    /// names an option, and reads the option directly after it.
    ///
    /// Server twin: `judicial.parse_choice` — fixture-locked on both engines.
    public static func parseChoice(
        _ text: String, options: [String], jsonKey: String = "answer",
        truncated: Bool = false
    ) -> String? {
        guard !text.isEmpty, !options.isEmpty else { return nil }
        let normalized = Dictionary(
            options.map { (normalize($0), $0) },
            uniquingKeysWith: { _, later in later })

        for payload in jsonObjects(in: text) {
            if let value = payload[jsonKey] as? String,
                let match = normalized[normalize(value)]
            {
                return match
            }
        }

        let flat = normalizeSpaces(text)
        let scalars = Array(flat.unicodeScalars)
        if let keyed = answerKeyValue(scalars, normalized: normalized, jsonKey: jsonKey) {
            return keyed
        }

        if let match = normalized[normalize(text)] { return match }

        if truncated || opensUnclosedJSON(text) { return nil }

        if let marked = markedOption(scalars, options: options) { return marked }

        if let leading = optionAfter(
            scalars, from: 0, options: options, delimiterRequired: true)
        {
            return leading
        }

        let haystackRange = NSRange(flat.startIndex..., in: flat)
        var earliest: (position: Int, option: String)?
        for option in options {
            var position: Int?
            if let letter = singleCharacter(option) {
                position = firstStandalone(scalars, letter)
            } else {
                let pattern =
                    "\\b" + NSRegularExpression.escapedPattern(for: normalize(option)) + "\\b"
                if let regex = try? NSRegularExpression(
                    pattern: pattern, options: [.caseInsensitive]),
                    let hit = regex.firstMatch(in: flat, range: haystackRange),
                    let range = Range(hit.range, in: flat)
                {
                    // Scalar offset, so a regex hit and a single-letter scan
                    // compare in one unit (UTF-16 offsets would not).
                    position = flat.unicodeScalars.distance(
                        from: flat.unicodeScalars.startIndex, to: range.lowerBound)
                }
            }
            if let position, earliest == nil || position < (earliest?.position ?? .max) {
                earliest = (position, option)
            }
        }
        return earliest?.option
    }

    // MARK: Literal-scan pieces of parseChoice (server twins: `_answer_key_value`,
    // `_option_at`, `_option_after`, `_marked_option`, `_first_standalone`,
    // `_single_character` in `judicial.py`)

    /// The answer-marker word, matched whole and ignoring ASCII case.
    private static let markerWord = Array("answer".unicodeScalars).map(\.value)
    /// An optional word between a marker and the option ("the answer is
    /// option B").
    private static let optionWord = Array("option".unicodeScalars).map(\.value)
    /// Skipped between the marker word and its `:` or "is": spaces, Markdown
    /// emphasis, and the quotes and backticks a model wraps a label in.
    private static let markerGap: Set<Unicode.Scalar> = [" ", "*", "_", "\"", "'", "`"]
    /// Skipped between a marker (or the start of the reply) and the option.
    private static let valueGap: Set<Unicode.Scalar> = [
        " ", "*", "_", "\"", "'", "`", "(", "[", ":",
    ]
    /// Punctuation that may follow an option where a delimiter is required.
    private static let delimiters: Set<Unicode.Scalar> = [
        ".", ",", ":", ";", ")", "]", "!", "?", "*", "_", "\"", "'", "`",
    ]

    /// A letter (Unicode L*) or an ASCII digit — what an option may not touch.
    private static func isWord(_ scalar: Unicode.Scalar) -> Bool {
        TurnEndpointParser.isLetter(scalar) || (scalar.value >= 48 && scalar.value <= 57)
    }

    private static func isASCIILetter(_ scalar: Unicode.Scalar) -> Bool {
        (scalar.value >= 65 && scalar.value <= 90) || (scalar.value >= 97 && scalar.value <= 122)
    }

    private static func asciiUpper(_ scalar: Unicode.Scalar) -> Unicode.Scalar {
        (scalar.value >= 97 && scalar.value <= 122)
            ? Unicode.Scalar(scalar.value - 32)! : scalar
    }

    private static func asciiLower(_ scalar: Unicode.Scalar) -> Unicode.Scalar {
        (scalar.value >= 65 && scalar.value <= 90)
            ? Unicode.Scalar(scalar.value + 32)! : scalar
    }

    /// Naive literal search, the twin of Python's `str.find`.
    private static func find(_ haystack: [UInt32], _ needle: [UInt32], from: Int) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        var start = Swift.max(0, from)
        while start + needle.count <= haystack.count {
            var offset = 0
            while offset < needle.count, haystack[start + offset] == needle[offset] {
                offset += 1
            }
            if offset == needle.count { return start }
            start += 1
        }
        return nil
    }

    /// The scalar a single-character option is found by in running text: an
    /// ASCII letter as its CAPITAL (so the article "a" is never option A, and
    /// a declared "a" is found as "A"), anything else as itself. nil for a
    /// longer option.
    private static func singleCharacter(_ option: String) -> Unicode.Scalar? {
        var stripped = normalizeSpaces(option).trimmingCharacters(in: .whitespaces)
        while stripped.hasPrefix(".") { stripped.removeFirst() }
        while stripped.hasSuffix(".") { stripped.removeLast() }
        let scalars = Array(stripped.unicodeScalars)
        guard scalars.count == 1 else { return nil }
        return isASCIILetter(scalars[0]) ? asciiUpper(scalars[0]) : scalars[0]
    }

    /// Position of the first `scalar` that touches no letter or digit.
    private static func firstStandalone(
        _ scalars: [Unicode.Scalar], _ scalar: Unicode.Scalar
    ) -> Int? {
        for index in scalars.indices where scalars[index] == scalar {
            if (index == 0 || !isWord(scalars[index - 1]))
                && (index + 1 >= scalars.count || !isWord(scalars[index + 1]))
            {
                return index
            }
        }
        return nil
    }

    /// The last `"<jsonKey>": "<value>"` whose value names an option, read
    /// from the text without parsing it as JSON (`scalars` has every
    /// whitespace run collapsed to one space, so raw newlines inside the
    /// object are gone). The value runs to the next `"` not preceded by a
    /// backslash; one that never closes is not read.
    private static func answerKeyValue(
        _ scalars: [Unicode.Scalar], normalized: [String: String], jsonKey: String
    ) -> String? {
        let haystack = scalars.map(\.value)
        let needle = Array(("\"" + jsonKey + "\"").unicodeScalars).map(\.value)
        var found: String?
        var index = find(haystack, needle, from: 0)
        while let at = index {
            var cursor = at + needle.count
            if cursor < scalars.count, scalars[cursor] == " " { cursor += 1 }
            if cursor < scalars.count, scalars[cursor] == ":" {
                cursor += 1
                if cursor < scalars.count, scalars[cursor] == " " { cursor += 1 }
                if cursor < scalars.count, scalars[cursor] == "\"" {
                    let start = cursor + 1
                    var end = start
                    while end < scalars.count,
                        !(scalars[end] == "\"" && scalars[end - 1] != "\\")
                    {
                        end += 1
                    }
                    if end < scalars.count {
                        let value = String(String.UnicodeScalarView(scalars[start ..< end]))
                        if let match = normalized[normalize(value)] { found = match }
                    }
                }
            }
            index = find(haystack, needle, from: at + 1)
        }
        return found
    }

    /// The option written at `scalars[start...]`, or nil.
    ///
    /// A single ASCII letter counts as its capital followed by anything but a
    /// letter or digit, or — where it cannot be the article — as its lower
    /// case followed by punctuation or the end of the reply. A longer option
    /// is compared ignoring ASCII case and must not run into a letter or
    /// digit. With `delimiterRequired` (the start of the reply) every option
    /// must be followed by punctuation or the end, so "A hard case" opens
    /// with no option. The longest match wins, then declaration order.
    private static func optionAt(
        _ scalars: [Unicode.Scalar], from start: Int, options: [String],
        delimiterRequired: Bool
    ) -> String? {
        var best: (length: Int, order: Int, option: String)?
        for (order, option) in options.enumerated() {
            let needle = Array(
                normalizeSpaces(option).trimmingCharacters(in: .whitespaces).unicodeScalars)
            guard !needle.isEmpty else { continue }
            let end = start + needle.count
            guard end <= scalars.count else { continue }
            let written = Array(scalars[start ..< end])
            let capital: Bool
            if needle.count == 1, isASCIILetter(needle[0]) {
                if written[0] == asciiUpper(needle[0]) {
                    capital = true
                } else if written[0] == asciiLower(needle[0]) {
                    capital = false
                } else {
                    continue
                }
            } else if written.map(TurnEndpointParser.fold)
                != needle.map(TurnEndpointParser.fold)
            {
                continue
            } else {
                capital = true
            }
            let atEnd = end >= scalars.count
            let delimited = atEnd || delimiters.contains(scalars[end])
            if delimiterRequired || !capital {
                if !delimited { continue }
            } else if !atEnd, isWord(scalars[end]) {
                continue
            }
            if best == nil || needle.count > best!.length
                || (needle.count == best!.length && order < best!.order)
            {
                best = (needle.count, order, option)
            }
        }
        return best?.option
    }

    /// Skip spaces, emphasis, quotes, and opening brackets, then read an
    /// option — directly, or after the word "option" (tried second, so an
    /// option that itself begins with "Option" is still read whole).
    private static func optionAfter(
        _ scalars: [Unicode.Scalar], from start: Int, options: [String],
        delimiterRequired: Bool
    ) -> String? {
        var cursor = start
        while cursor < scalars.count, valueGap.contains(scalars[cursor]) { cursor += 1 }
        if let choice = optionAt(
            scalars, from: cursor, options: options, delimiterRequired: delimiterRequired)
        {
            return choice
        }
        let end = cursor + optionWord.count
        guard end < scalars.count,
            scalars[cursor ..< end].map(TurnEndpointParser.fold) == optionWord,
            scalars[end] == " "
        else { return nil }
        cursor = end
        while cursor < scalars.count, valueGap.contains(scalars[cursor]) { cursor += 1 }
        return optionAt(
            scalars, from: cursor, options: options, delimiterRequired: delimiterRequired)
    }

    /// The option named by the LAST answer marker that names one: the whole
    /// word "answer" (any ASCII case), then — past spaces, emphasis, and
    /// quotes — a `:` or the whole word "is".
    private static func markedOption(
        _ scalars: [Unicode.Scalar], options: [String]
    ) -> String? {
        let folded = scalars.map(TurnEndpointParser.fold)
        var found: String?
        var index = find(folded, markerWord, from: 0)
        while let at = index {
            let end = at + markerWord.count
            if (at == 0 || !TurnEndpointParser.isLetter(scalars[at - 1]))
                && (end >= scalars.count || !TurnEndpointParser.isLetter(scalars[end]))
            {
                var cursor = end
                while cursor < scalars.count, markerGap.contains(scalars[cursor]) {
                    cursor += 1
                }
                var after: Int?
                if cursor < scalars.count, scalars[cursor] == ":" {
                    after = cursor + 1
                } else if cursor + 2 <= scalars.count,
                    folded[cursor] == 105, folded[cursor + 1] == 115,  // "is"
                    cursor + 2 >= scalars.count
                        || !TurnEndpointParser.isLetter(scalars[cursor + 2])
                {
                    after = cursor + 2
                }
                if let after,
                    let choice = optionAfter(
                        scalars, from: after, options: options, delimiterRequired: false)
                {
                    found = choice
                }
            }
            index = find(folded, markerWord, from: at + 1)
        }
        return found
    }

    private static func normalize(_ text: String) -> String {
        var result = normalizeSpaces(text).trimmingCharacters(in: .whitespaces)
        while result.hasSuffix(".") { result.removeLast() }
        while result.hasPrefix(".") { result.removeFirst() }
        return result.lowercased()
    }

    private static func normalizeSpaces(_ text: String) -> String {
        text.replacing(/\s+/, with: " ")
    }

    /// JSON objects embedded in the text (fenced or bare), outermost first —
    /// a model told to answer in JSON often wraps it in prose or a fence.
    private static func jsonObjects(in text: String) -> [[String: Any]] {
        var objects: [[String: Any]] = []
        let fence = /```(?:json)?\s*(\{.*?\})\s*```/.dotMatchesNewlines()
        for match in text.matches(of: fence) {
            if let payload = decodeObject(String(match.1)) {
                objects.append(payload)
            }
        }
        var searchIndex = text.startIndex
        while let start = text[searchIndex...].firstIndex(of: "{") {
            if let end = balancedObjectEnd(in: text, from: start),
                let payload = decodeObject(String(text[start ... end]))
            {
                objects.append(payload)
            }
            searchIndex = text.index(after: start)
        }
        return objects
    }

    /// Does the text open a JSON object it never closes — the shape of an
    /// answer-in-JSON response cut off mid-object? Complete objects are
    /// skipped wholesale (braces inside their strings do not count), so only
    /// a brace depth still open at end-of-text answers true. Server twin:
    /// `judicial._opens_unclosed_json`.
    static func opensUnclosedJSON(_ text: String) -> Bool {
        var searchIndex = text.startIndex
        while let start = text[searchIndex...].firstIndex(of: "{") {
            guard let end = balancedObjectEnd(in: text, from: start) else {
                return true
            }
            searchIndex = text.index(after: end)
        }
        return false
    }

    /// Index of the `}` closing the object opened at `start`, tracking JSON
    /// string/escape state so braces inside strings do not count (approximates
    /// Python's `JSONDecoder.raw_decode`, which tolerates trailing prose).
    private static func balancedObjectEnd(
        in text: String, from start: String.Index
    ) -> String.Index? {
        var depth = 0
        var inString = false
        var escaped = false
        var index = start
        while index < text.endIndex {
            let char = text[index]
            if inString {
                if escaped {
                    escaped = false
                } else if char == "\\" {
                    escaped = true
                } else if char == "\"" {
                    inString = false
                }
            } else {
                switch char {
                case "\"": inString = true
                case "{": depth += 1
                case "}":
                    depth -= 1
                    if depth == 0 { return index }
                default: break
                }
            }
            index = text.index(after: index)
        }
        return nil
    }

    private static func decodeObject(_ candidate: String) -> [String: Any]? {
        guard
            let object = try? JSONSerialization.jsonObject(with: Data(candidate.utf8))
        else { return nil }
        return object as? [String: Any]
    }

    // MARK: - Duration-in-months parsing

    /// Number words for the spelled-out formal register — e.g. a
    /// judicial-decision study's "I sentence the defendant, A, to **ten years
    /// and six months'** imprisonment" (an anchoring-arm run, 2026-08-10:
    /// 58/1320 records unparsed, and dose-dependently, because steering pushes
    /// the formal register — so a digit-only grammar loses records
    /// NON-randomly).
    /// One through twelve, accepted wherever a digit number is in the
    /// COMPOUND and SINGLE terms only — ranges stay digit-only. Server twin:
    /// `judicial.NUMBER_WORDS`; the registry's durationMonths grammar shares
    /// this vocabulary (`ParserRegistry.numberWordFragment`).
    static let numberWords: [String: Double] = [
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12,
    ]

    /// Extracts a duration in months from prose (e.g. a sentence length in a
    /// judicial-decision study).
    ///
    /// Handles "18 months", "2 years", "1.5 years", "18 to 24 months"
    /// (midpoint), compound durations ("8 years 3 months" → 99 — a compound
    /// must never drop a term), the spelled-out formal register ("ten years
    /// and six months" → 126 — number words one through twelve), German units
    /// with identical semantics ("2 Jahre 6 Monate" → 30, "18 Monate" → 18),
    /// and comma decimals. Years normalize ×12. Returns nil on failure —
    /// parse-failure *rate* is a first-class coherence endpoint, so failures
    /// must be counted, never silently dropped or coerced to 0.
    ///
    /// Cross-engine contract: must agree fixture-for-fixture with the
    /// server's `parse_months` (`Server/steerlab_server/experiment/`).
    public static func parseMonths(_ text: String) -> Double? {
        guard !text.isEmpty else { return nil }
        let range =
            /(\d+(?:[.,]\d+)?)\s*(?:to|-|–|—)\s*(\d+(?:[.,]\d+)?)\s*(years?|yrs?|months?|mos?|jahre?n?|monate?n?)\b/
            .ignoresCase().wordBoundaryKind(.simple)
        if let match = text.firstMatch(of: range) {
            guard let low = toDouble(match.1), let high = toDouble(match.2) else {
                return nil
            }
            let months = (low + high) / 2.0
            return isYears(match.3) ? months * 12.0 : months
        }
        // Compound "X years Y months" (optionally joined by ","/"and"/"und"):
        // both terms count — matching only the first silently corrupts the
        // duration DV (96 where the model said 99). Number terms are
        // digits-or-word; the \b pair keeps "brighten years" from reading as
        // "ten years".
        let compound =
            /(\d+(?:[.,]\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)\s*(?:years?|yrs?|jahre?n?)\b,?\s*(?:(?:and|und)\s+)?(\d+(?:[.,]\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)\s*(?:months?|mos?|monate?n?)\b/
            .ignoresCase().wordBoundaryKind(.simple)
        if let match = text.firstMatch(of: compound) {
            guard let years = toDouble(match.1), let months = toDouble(match.2) else {
                return nil
            }
            return years * 12.0 + months
        }
        let single =
            /(\d+(?:[.,]\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)\s*(years?|yrs?|months?|mos?|jahre?n?|monate?n?)\b/
            .ignoresCase().wordBoundaryKind(.simple)
        if let match = text.firstMatch(of: single) {
            guard let value = toDouble(match.1) else { return nil }
            return isYears(match.2) ? value * 12.0 : value
        }
        return nil
    }

    private static func toDouble(_ token: Substring) -> Double? {
        if let word = numberWords[token.lowercased()] { return word }
        return Double(token.replacingOccurrences(of: ",", with: "."))
    }

    private static func isYears(_ unit: Substring) -> Bool {
        let lowered = unit.lowercased()
        return lowered.hasPrefix("year") || lowered.hasPrefix("yr")
            || lowered.hasPrefix("jahr")
    }

    /// Fraction of items whose parse came back nil.
    public static func parseFailureRate(_ parsed: [Double?]) -> Double {
        guard !parsed.isEmpty else { return 0 }
        return Double(parsed.count(where: { $0 == nil })) / Double(parsed.count)
    }

    // MARK: - Derived endpoints

    /// P(parsed choice == target) over successfully parsed items.
    public static func outcomeRate(_ choices: [String?], target: String) -> Double {
        let hits = choices.compactMap { $0 }
        guard !hits.isEmpty else { return .nan }
        return Double(hits.count(where: { $0 == target })) / Double(hits.count)
    }

    /// How much larger the concept's outcome-rate shift is on the
    /// high-discretion arm than on the low-discretion one.
    ///
    /// A generic two-arm contrast of two already-computed deltas. In the
    /// judicial-decision study it compares a doctrine framed as a standard
    /// (high discretion) against the same doctrine framed as a rule (low
    /// discretion).
    public static func sympathyGap(
        standardArmDelta: Double, ruleArmDelta: Double
    ) -> Double {
        standardArmDelta - ruleArmDelta
    }

    /// Per-item difference-in-differences:
    /// (treated − baseline | standard) − (treated − baseline | rule).
    /// Feed the result to `StudyStatistics`.
    public static func ruleVsStandardInteraction(
        ruleBaseline: [Double], ruleTreated: [Double],
        standardBaseline: [Double], standardTreated: [Double]
    ) -> [Double] {
        let n = min(
            min(ruleBaseline.count, ruleTreated.count),
            min(standardBaseline.count, standardTreated.count))
        return (0 ..< n).map { i in
            (standardTreated[i] - standardBaseline[i])
                - (ruleTreated[i] - ruleBaseline[i])
        }
    }

    /// OLS slope of the numeric outcome on the numeric value presented to the
    /// model in the item — the anchoring endpoint (Englich & Mussweiler's
    /// design; in the judicial-decision study, sentence months on anchor
    /// months). NaN when degenerate.
    public static func anchorSlope(anchors: [Double?], sentences: [Double?]) -> Double {
        let pairs = pairedValues(anchors, sentences)
        guard pairs.count >= 2 else { return .nan }
        let meanX = pairs.map(\.0).reduce(0, +) / Double(pairs.count)
        let meanY = pairs.map(\.1).reduce(0, +) / Double(pairs.count)
        let varX = pairs.map { ($0.0 - meanX) * ($0.0 - meanX) }.reduce(0, +)
        guard varX != 0 else { return .nan }
        let cov = pairs.map { ($0.0 - meanX) * ($0.1 - meanY) }.reduce(0, +)
        return cov / varX
    }

    /// Pearson correlation of the numeric outcome with each item's declared
    /// severity — does the model still scale its response to the item under
    /// the intervention? (In the judicial-decision study: sentence length
    /// against offense severity.)
    public static func proportionality(
        severities: [Double?], sentences: [Double?]
    ) -> Double {
        let pairs = pairedValues(severities, sentences)
        guard pairs.count >= 2 else { return .nan }
        let meanX = pairs.map(\.0).reduce(0, +) / Double(pairs.count)
        let meanY = pairs.map(\.1).reduce(0, +) / Double(pairs.count)
        let varX = pairs.map { ($0.0 - meanX) * ($0.0 - meanX) }.reduce(0, +)
        let varY = pairs.map { ($0.1 - meanY) * ($0.1 - meanY) }.reduce(0, +)
        guard varX != 0, varY != 0 else { return .nan }
        let cov = pairs.map { ($0.0 - meanX) * ($0.1 - meanY) }.reduce(0, +)
        return cov / (varX * varY).squareRoot()
    }

    private static func pairedValues(
        _ xs: [Double?], _ ys: [Double?]
    ) -> [(Double, Double)] {
        zip(xs, ys).compactMap { x, y in
            guard let x, let y else { return nil }
            return (x, y)
        }
    }

    /// Distributional readout over sampled numeric outcomes for one item —
    /// the mean AND spread endpoints, computed per (condition, prompt).
    public struct DistributionSummary: Codable, Sendable, Equatable {
        public let count: Int
        public let mean: Double
        /// Sample standard deviation (n − 1 denominator).
        public let stdev: Double
        public let min: Double
        public let q25: Double
        public let median: Double
        public let q75: Double
        public let max: Double

        public init(
            count: Int, mean: Double, stdev: Double, min: Double,
            q25: Double, median: Double, q75: Double, max: Double
        ) {
            self.count = count
            self.mean = mean
            self.stdev = stdev
            self.min = min
            self.q25 = q25
            self.median = median
            self.q75 = q75
            self.max = max
        }
    }

    /// Summary statistics over the successfully parsed samples of one item.
    public static func summarize(_ values: [Double?]) -> DistributionSummary? {
        let clean = values.compactMap { value -> Double? in
            guard let value, !value.isNaN else { return nil }
            return value
        }.sorted()
        guard let first = clean.first, let last = clean.last else { return nil }
        let n = clean.count
        let mean = clean.reduce(0, +) / Double(n)
        let stdev =
            n > 1
            ? (clean.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(n - 1))
                .squareRoot()
            : 0.0
        return DistributionSummary(
            count: n, mean: mean, stdev: stdev, min: first,
            q25: quantile(clean, 0.25), median: quantile(clean, 0.5),
            q75: quantile(clean, 0.75), max: last)
    }

    /// Linear-interpolation quantile over a pre-sorted, non-empty list.
    private static func quantile(_ ordered: [Double], _ q: Double) -> Double {
        guard ordered.count > 1 else { return ordered[0] }
        let position = q * Double(ordered.count - 1)
        let low = Int(position.rounded(.down))
        let high = Swift.min(low + 1, ordered.count - 1)
        let weight = position - Double(low)
        return ordered[low] * (1 - weight) + ordered[high] * weight
    }
}

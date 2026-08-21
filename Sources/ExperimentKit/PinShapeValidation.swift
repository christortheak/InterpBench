import Foundation

/// Pin-time SHAPE validation (Usability Plan Phase 0, item 2): a pin should
/// validate the file's shape at the moment the pin is made, not only its
/// bytes. Pins verify identity (SHA-256), so without this a schema-garbage
/// file pins cleanly and fails much later — at analyze for the human
/// baseline, at run time for the capability battery. Errors follow the
/// plain-language rule: one sentence saying what is wrong and what to do,
/// technical detail second.
///
/// The checks mirror the CONSUMER of each file, never a stricter invented
/// schema:
/// - human-baseline CSV → the analyze loader's required columns
///   (`Server/steerlab_server/experiment/residuals.py`,
///   `HUMAN_BASELINE_FIELDS` — the loader's columns ARE the contract);
/// - capability battery → `CapabilityBattery` (Scoring.swift), the parser
///   every battery run uses on this engine.
///
/// This generalizes: new pinned instrument, new shape check, same mechanism.
public enum PinShapeValidation {

    // MARK: - Human-baseline CSV

    /// The columns the analyze step's loader requires, one row per endpoint
    /// (residuals.py `HUMAN_BASELINE_FIELDS`). Order does not matter and
    /// extra columns (source, n, notes, …) are fine — presence is what the
    /// loader needs.
    public static let humanBaselineRequiredColumns = [
        "endpoint", "deltaHuman", "ciLower", "ciUpper",
    ]

    /// Header columns of a CSV's first non-empty line (BOM-stripped,
    /// whitespace- and quote-trimmed). A naive comma split is correct for
    /// header NAMES — the contract's names contain no commas. Data rows
    /// are checked separately (`humanBaselineRowProblem`, through the real
    /// CSV parser) against the loader's own row tolerances.
    public static func csvHeaderColumns(_ data: Data) -> [String] {
        var text = String(decoding: data, as: UTF8.self)
        if text.hasPrefix("\u{FEFF}") { text.removeFirst() }
        guard
            let line = text.split(whereSeparator: \.isNewline)
                .first(where: {
                    !$0.trimmingCharacters(in: .whitespaces).isEmpty
                })
        else { return [] }
        return line.split(separator: ",", omittingEmptySubsequences: false)
            .map {
                $0.trimmingCharacters(in: .whitespaces)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            }
            .filter { !$0.isEmpty }
    }

    /// nil = the shape is fine. Otherwise a plain-language problem: what
    /// the analyze step reads, what this file has, and the remedy. Header
    /// first, then EVERY data row's required numeric fields — a malformed
    /// number would pin cleanly and die much later at analyze, so it
    /// refuses at the pin instead (feedback at the moment of action).
    public static func humanBaselineShapeProblem(
        _ data: Data, file: String
    ) -> String? {
        let found = csvHeaderColumns(data)
        let missing = humanBaselineRequiredColumns.filter { !found.contains($0) }
        guard missing.isEmpty else {
            let required = humanBaselineRequiredColumns.joined(separator: ", ")
            let have = found.isEmpty
                ? "no header columns at all"
                : "columns " + found.joined(separator: ", ")
            return "the analyze step reads human-baseline columns \(required) — "
                + "'\(file)' has \(have); add the missing column(s) "
                + "(\(missing.joined(separator: ", "))) or start from Create "
                + "from template (extra columns are fine)"
        }
        return humanBaselineRowProblem(data, file: file)
    }

    /// The fields the analyze loader parses as numbers on every data row
    /// (residuals.py `float(row[...])`) — everything but the join key.
    public static let humanBaselineNumericFields =
        humanBaselineRequiredColumns.filter { $0 != "endpoint" }

    /// Row-level check mirroring the LOADER's tolerances (residuals.py
    /// `load_human_baseline`, `csv.DictReader` + `float()`): quoted
    /// numbers, surrounding whitespace, and scientific notation are fine;
    /// blank/whitespace-only lines are skipped; extra columns are ignored.
    /// A required numeric cell that is absent, empty, or non-numeric
    /// refuses, naming the first bad data row and field.
    static func humanBaselineRowProblem(_ data: Data, file: String) -> String? {
        let table: TabularImport.Table
        do {
            table = try TabularImport.parseCSV(data, fileName: file)
        } catch {
            // Header-only (row-less) files load as zero endpoints in the
            // analyze loader — nothing to check; empty/header-garbage
            // files were already refused by the header check above.
            return nil
        }
        for (index, row) in table.rows.enumerated() {
            for field in humanBaselineNumericFields {
                guard let value = row[field] else {
                    return "the analyze step reads \(field) as a number on "
                        + "every data row — '\(file)' data row \(index + 1) "
                        + "has no \(field) value; fill the cell in (blank "
                        + "lines are fine, blank cells are not)"
                }
                let text = value.text
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard parsesAsLoaderNumber(text) else {
                    return "the analyze step reads \(field) as a number on "
                        + "every data row — '\(file)' data row \(index + 1) "
                        + "has \(field) '\(text)', which is not a number; "
                        + "fix the value (quoted numbers and scientific "
                        + "notation are fine)"
                }
            }
        }
        return nil
    }

    /// Whether the analyze loader's `float(...)` would accept this cell —
    /// the LOADER's tolerances, not Swift's: surrounding whitespace,
    /// signs, scientific notation, inf/nan, and Python's digit-grouping
    /// underscores (between digits only) are accepted; hexadecimal float
    /// syntax ("0x1p3"), which Swift's `Double` parses but Python's
    /// `float()` refuses, is refused.
    static func parsesAsLoaderNumber(_ text: String) -> Bool {
        var candidate = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !candidate.isEmpty else { return false }
        if candidate.contains("_") {
            let characters = Array(candidate)
            for (index, character) in characters.enumerated()
            where character == "_" {
                guard index > 0, index + 1 < characters.count,
                    characters[index - 1].isNumber,
                    characters[index + 1].isNumber
                else { return false }
            }
            candidate.removeAll { $0 == "_" }
        }
        if candidate.lowercased().contains("0x") { return false }
        return Double(candidate) != nil
    }

    // MARK: - Capability battery (JSONL)

    /// nil = the shape is fine. An empty battery always refuses: it would
    /// silently score every condition 0-of-0.
    ///
    /// TWO formats pin here, and this layer is the ONE shape checker for both
    /// — the local runner (`CapabilityBattery.init(data:file:)`) calls
    /// `batteryFormat2Problem` itself rather than parsing a header twice, so
    /// what pins is exactly what loads:
    ///
    /// - **format 1** (headerless): one `{"prompt": …, "answer": …}` object
    ///   per non-empty line with an optional `"grading"`, checked through the
    ///   runner's own `Item` decoder.
    /// - **format 2** (a first-line `{"batteryFormat": 2, …}` header) mirrors
    ///   the SERVER's loader (`battery._parse_header` / `_parse_item`), which
    ///   is the cross-engine contract authority: declared arming, declared
    ///   scoring, options that contain the answer, explicit grading on
    ///   generatedText items.
    ///
    /// Since 2026-08-19 both formats also RUN here: the local runner scores
    /// format 2 under the battery's own arming, by choice probability,
    /// through the answer-token logprob instrument. Byte-hashing is
    /// format-agnostic, so a pin means the same thing either way.
    public static func capabilityBatteryShapeProblem(
        _ data: Data, file: String
    ) -> String? {
        let lines = batteryLines(data)
        let remedy = "each line must be a JSON object like "
            + #"{"prompt": "…", "answer": "…"}"#
            + " (optional \"grading\") — start from prompts/batteries/"
            + "basic.jsonl"
        guard let first = lines.first else {
            return "the capability battery '\(file)' has no rows, so every "
                + "condition would score 0 of 0; " + remedy
        }
        if batteryHeaderObject(first) != nil {
            return batteryFormat2Problem(lines, file: file)
        }
        let decoder = JSONDecoder()
        for (index, line) in lines.enumerated() {
            guard
                (try? decoder.decode(
                    CapabilityBattery.Item.self, from: Data(line.utf8)))
                    != nil
            else {
                return "the capability battery '\(file)' is not shaped the "
                    + "way battery runs read it (line \(index + 1) is not a "
                    + "battery item); " + remedy
            }
        }
        return nil
    }

    /// A battery file's candidate lines: trimmed, blank lines dropped. The
    /// ONE line split both the pin check and the runner read a battery
    /// through, so "line 3" means the same row on both.
    static func batteryLines(_ data: Data) -> [String] {
        String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// The declared header of a format-2 battery line, or nil when the line
    /// is not a header (no `batteryFormat` key) or not a JSON object at all.
    ///
    /// A header is identified by the KEY's presence, exactly as the server
    /// does (`battery._header`) — the VALUE is validated separately, so a bad
    /// version number is a loud refusal rather than a silent fallback to the
    /// legacy format (which would score the file under a different regime
    /// than it declares).
    ///
    /// Lines decode through `JSONValue` rather than `JSONSerialization`
    /// because the distinctions here are type distinctions: `true` must not
    /// pass as `maxTokens: 1`, and `2.5` must not pass as format 2.
    /// `NSNumber` bridging blurs exactly those.
    static func batteryHeaderObject(_ line: String) -> [String: JSONValue]? {
        guard
            let value = try? JSONDecoder().decode(
                JSONValue.self, from: Data(line.utf8)),
            case .object(let object) = value,
            object["batteryFormat"] != nil
        else { return nil }
        return object
    }

    /// Scoring modes a format-2 battery may declare (server `SCORING_MODES`).
    public static let batteryScoringModes = ["choiceProbability", "generatedText"]

    /// Header + item validation for a format-2 battery, mirroring the
    /// server's `battery._parse_header` / `_parse_item` — the loader that
    /// will actually score the file. Shared with `CapabilityBattery`, which
    /// runs it before parsing a v2 file rather than duplicating the rules.
    static func batteryFormat2Problem(
        _ lines: [String], file: String
    ) -> String? {
        guard let header = batteryHeaderObject(lines[0]) else { return nil }
        let remedy = "fix the header line, or run "
            + "`steerlab-server battery lint \(file)` which writes a valid one"
        guard case .number(let version)? = header["batteryFormat"], version == 2
        else {
            let raw = header["batteryFormat"]?.displayString ?? "nothing"
            return "the capability battery '\(file)' declares batteryFormat "
                + "\(raw) — the only header format is 2 (a file with NO "
                + "header is the legacy format 1, and declaring 1 would "
                + "change what an existing pinned hash means); " + remedy
        }
        let defaultScoring: String
        switch header["scoring"] {
        case .none, .some(.null): defaultScoring = batteryScoringModes[0]
        case .some(.string(let declared)): defaultScoring = declared
        case .some(let other):
            return "the capability battery '\(file)' header declares scoring "
                + "\(other.displayString), which must be one of "
                + batteryScoringModes.joined(separator: ", ") + "; " + remedy
        }
        guard batteryScoringModes.contains(defaultScoring) else {
            return "the capability battery '\(file)' header declares scoring "
                + "'\(defaultScoring)' — one of "
                + batteryScoringModes.joined(separator: ", ") + "; " + remedy
        }
        if let maxTokens = header["maxTokens"], !isJSONNull(maxTokens) {
            guard case .number(let value) = maxTokens, value > 0,
                value == value.rounded()
            else {
                return "the capability battery '\(file)' header declares "
                    + "maxTokens \(maxTokens.displayString), which must be a "
                    + "positive whole number; " + remedy
            }
        }
        if let promptMode = header["promptMode"], !isJSONNull(promptMode) {
            guard case .string(let value) = promptMode, !value.isEmpty else {
                return "the capability battery '\(file)' header declares a "
                    + "promptMode that is not a non-empty string; " + remedy
            }
        }
        if let systemPrompt = header["systemPrompt"], !isJSONNull(systemPrompt) {
            guard case .string = systemPrompt else {
                return "the capability battery '\(file)' header declares a "
                    + "systemPrompt that is not a string; " + remedy
            }
        }
        if let thinking = header["qwenThinkingEnabled"], !isJSONNull(thinking) {
            guard case .bool = thinking else {
                return "the capability battery '\(file)' header declares "
                    + "qwenThinkingEnabled, which must be true or false; "
                    + remedy
            }
        }

        let items = Array(lines.dropFirst())
        guard !items.isEmpty else {
            return "the capability battery '\(file)' has a header but no "
                + "items, so every condition would score 0 of 0; " + remedy
        }
        for (offset, line) in items.enumerated() {
            if let problem = batteryFormat2ItemProblem(
                line, file: file, lineNumber: offset + 2,
                defaultScoring: defaultScoring)
            {
                return problem
            }
        }
        return nil
    }

    private static func isJSONNull(_ value: JSONValue) -> Bool {
        if case .null = value { return true }
        return false
    }

    /// One format-2 item. The `options`-contains-`answer` rule is the one
    /// that matters scientifically: a choice item whose answer is absent from
    /// its options can NEVER be scored correct, so the battery would read a
    /// depressed accuracy under every condition and the capability CONTROL
    /// would silently become noise.
    private static func batteryFormat2ItemProblem(
        _ line: String, file: String, lineNumber: Int, defaultScoring: String
    ) -> String? {
        let remedy = "fix line \(lineNumber), or run "
            + "`steerlab-server battery lint \(file)`"
        guard
            let decoded = try? JSONDecoder().decode(
                JSONValue.self, from: Data(line.utf8)),
            case .object(let item) = decoded
        else {
            return "the capability battery '\(file)' line \(lineNumber) is "
                + "not a JSON object; " + remedy
        }
        guard case .string(let prompt)? = item["prompt"], !prompt.isEmpty else {
            return "the capability battery '\(file)' line \(lineNumber) has "
                + "no \"prompt\" text; " + remedy
        }
        guard case .string(let answer)? = item["answer"] else {
            return "the capability battery '\(file)' line \(lineNumber) has "
                + "no \"answer\" string; " + remedy
        }
        if let identifier = item["id"], !isJSONNull(identifier) {
            guard case .string(let value) = identifier, !value.isEmpty else {
                return "the capability battery '\(file)' line \(lineNumber) "
                    + "has an \"id\" that is not a non-empty string; " + remedy
            }
        }
        var declaredGrading: String?
        if let grading = item["grading"], !isJSONNull(grading) {
            guard case .string(let value) = grading,
                CapabilityBattery.GradingMode(rawValue: value) != nil
            else {
                return "the capability battery '\(file)' line \(lineNumber) "
                    + "declares an unknown \"grading\" mode; " + remedy
            }
            declaredGrading = value
        }
        let scoring: String
        switch item["scoring"] {
        case .none, .some(.null): scoring = defaultScoring
        case .some(.string(let declared)): scoring = declared
        case .some: scoring = ""
        }
        guard batteryScoringModes.contains(scoring) else {
            return "the capability battery '\(file)' line \(lineNumber) "
                + "declares an unknown scoring mode — one of "
                + batteryScoringModes.joined(separator: ", ") + "; " + remedy
        }
        let options = item["options"].flatMap { value -> [JSONValue]? in
            if case .array(let entries) = value { return entries }
            return nil
        }
        if scoring == "choiceProbability" {
            guard let options, options.count >= 2 else {
                return "the capability battery '\(file)' line \(lineNumber) "
                    + "is scored by choice probability but declares no "
                    + "\"options\" list of at least 2 answers; " + remedy
            }
            var texts: [String] = []
            for option in options {
                guard case .string(let text) = option, !text.isEmpty else {
                    return "the capability battery '\(file)' line "
                        + "\(lineNumber) has an \"options\" entry that is not "
                        + "a non-empty string; " + remedy
                }
                texts.append(text)
            }
            guard Set(texts).count == texts.count else {
                return "the capability battery '\(file)' line \(lineNumber) "
                    + "repeats an option, so the same choice would be scored "
                    + "twice; " + remedy
            }
            guard texts.contains(answer) else {
                return "the capability battery '\(file)' line \(lineNumber) "
                    + "has an \"answer\" that is not one of its \"options\", "
                    + "so the item could never be scored correct; " + remedy
            }
        } else {
            if let raw = item["options"], !isJSONNull(raw) {
                return "the capability battery '\(file)' line \(lineNumber) "
                    + "declares \"options\" on a generatedText item — options "
                    + "belong to choiceProbability items only; " + remedy
            }
            guard declaredGrading != nil else {
                return "the capability battery '\(file)' line \(lineNumber) "
                    + "is a format-2 generatedText item and must DECLARE a "
                    + "\"grading\" mode — inferred normalization is what made "
                    + "legacy readings format-sensitive; " + remedy
            }
        }
        return nil
    }

    /// The outcome vocabulary of a human-validation row — the cross-engine
    /// data contract (`_load_human_validation` on the server reads the same
    /// rows). "variant" is the file's spelling for this engine's
    /// "condition" result label.
    public static let humanValidationOutcomes = ["baseline", "variant", "tie"]

    /// nil = the shape is fine. Delegates to the ONE human-validation
    /// parser evaluation uses (`ExperimentTasks.parseHumanValidation` —
    /// review 2026-08-01, P1: the first validator advertised numeric
    /// `sampleIndex` and never checked it, so a string index pinned
    /// cleanly and failed much later at evaluate; a validator with its own
    /// parser drifts by construction). What the parser refuses, the pin
    /// refuses: malformed rows, non-integer sampleIndex, duplicate keys,
    /// bad outcome vocabulary, empty files.
    public static func humanValidationShapeProblem(
        _ data: Data, file: String
    ) -> String? {
        let remedy = "each line must be a JSON object like "
            + #"{"condition": "…", "promptID": "…", "outcome": "baseline"}"#
            + " with outcome one of "
            + humanValidationOutcomes.joined(separator: "|")
            + " (optional non-negative integer \"sampleIndex\"; absent = "
            + "wildcard across the pair's sample cells; one row per key)"
        do {
            _ = try ExperimentTasks.parseHumanValidation(data)
            return nil
        } catch let error as ExperimentError {
            return "the human-validation file '\(file)' refuses: "
                + error.reason + "; " + remedy
        } catch {
            return "the human-validation file '\(file)' refuses: \(error); "
                + remedy
        }
    }
}

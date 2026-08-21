import Foundation

/// Factorial / counterbalancing generator (Usability Plan Phase 4, items
/// 20–21) — AUTHORING-TIME data generation, the audit's own
/// firewall-friendly recommendation.
///
/// A design spec declares FACTORS with LEVELS (each level carries literal
/// substitution values for `{{PLACEHOLDER}}` variables) and one or more
/// prompt TEMPLATES containing `{{VAR}}` placeholders. Generation fully
/// crosses every factor-level combination with every template and emits
/// ORDINARY, LITERAL task-prompt items — fully substituted text, a
/// deterministic id encoding the cell, and a `factors` metadata object per
/// record. Nothing changes at run time on either engine: the emitted JSONL
/// is byte-deterministic and pins like any hand-authored file, and the
/// design spec is saved NEXT TO the output purely as provenance (it is not
/// a pin — the pin is on the emitted prompts file the run loop actually
/// reads).
///
/// Determinism is a design rule, not an accident: generation has no
/// randomness anywhere (seeded shuffling is a documented non-goal for now —
/// deterministic order is the firewall-friendly default), so re-generating
/// from the same spec re-produces the same bytes and the same pin hash.
public struct FactorialDesign: Codable, Equatable, Sendable {

    // MARK: - Spec

    /// One level of a factor, carrying the literal text substituted for
    /// each `{{VARIABLE}}` it supplies (e.g. level "high" of factor
    /// "anchor" → `{"DEMAND": "9 years"}`).
    public struct Level: Codable, Equatable, Sendable {
        public var name: String
        /// Placeholder variable → literal replacement text. Values must be
        /// literal (no `{{…}}` inside a value — substitution is single-pass
        /// and never recursive).
        public var substitutions: [String: String]

        public init(name: String, substitutions: [String: String]) {
            self.name = name
            self.substitutions = substitutions
        }
    }

    public struct Factor: Codable, Equatable, Sendable {
        public var name: String
        public var levels: [Level]

        public init(name: String, levels: [Level]) {
            self.name = name
            self.levels = levels
        }
    }

    /// One prompt template. `options` and `target` (the categorical
    /// instrument's fields) may also contain placeholders; they are
    /// substituted per cell exactly like the text.
    public struct Template: Codable, Equatable, Sendable {
        public var id: String
        public var text: String
        public var options: [String]?
        public var target: String?

        public init(
            id: String, text: String, options: [String]? = nil,
            target: String? = nil
        ) {
            self.id = id
            self.text = text
            self.options = options
            self.target = target
        }
    }

    public var factors: [Factor]
    public var templates: [Template]
    /// When true, every cell is emitted TWICE — once as declared and once
    /// with `options` reversed — with an `orderFlipped` factor recorded
    /// (`"false"` / `"true"`). The target names its option by value, so it
    /// follows the option through the flip unchanged. Deterministic: the
    /// unflipped copy always precedes the flipped one.
    public var counterbalanceOptionOrder: Bool

    public init(
        factors: [Factor], templates: [Template],
        counterbalanceOptionOrder: Bool = false
    ) {
        self.factors = factors
        self.templates = templates
        self.counterbalanceOptionOrder = counterbalanceOptionOrder
    }

    /// A missing `counterbalanceOptionOrder` key decodes as false, so a
    /// hand- or LLM-authored spec can omit it.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        factors = try container.decodeIfPresent([Factor].self, forKey: .factors) ?? []
        templates = try container.decode([Template].self, forKey: .templates)
        counterbalanceOptionOrder =
            try container.decodeIfPresent(
                Bool.self, forKey: .counterbalanceOptionOrder) ?? false
    }

    // MARK: - Errors

    /// Plain-language refusal: one sentence saying what is wrong and what
    /// to do next (the project's error-copy rule).
    public struct Problem: Error, CustomStringConvertible, Equatable {
        public let message: String
        public var description: String { message }

        public init(_ message: String) {
            self.message = message
        }
    }

    // MARK: - Validation

    /// Factor names, level names, and template ids become id components,
    /// so they are restricted to a filename/id-safe alphabet.
    static func safeName(_ name: String) -> Bool {
        !name.isEmpty
            && name.allSatisfy {
                $0.isLetter && $0.isASCII || $0.isNumber && $0.isASCII
                    || $0 == "-"
            }
    }

    /// Structural validation, run before any generation. Every message is a
    /// hard, plain-language error — a spec that does not validate never
    /// emits a single item.
    public func validate() throws {
        guard !templates.isEmpty else {
            throw Problem("the design has no templates — add at least one prompt template")
        }
        var templateIDs = Set<String>()
        for template in templates {
            guard Self.safeName(template.id) else {
                throw Problem(
                    "template id '\(template.id)' is not usable in item ids — "
                        + "use letters, digits, and dashes only (e.g. 't1')")
            }
            guard templateIDs.insert(template.id).inserted else {
                throw Problem(
                    "two templates share the id '\(template.id)' — template ids "
                        + "must be unique")
            }
        }
        var factorNames = Set<String>()
        var variableOwner: [String: String] = [:]  // variable → factor name
        for factor in factors {
            guard Self.safeName(factor.name) else {
                throw Problem(
                    "factor name '\(factor.name)' is not usable in item ids — "
                        + "use letters, digits, and dashes only (e.g. 'anchor')")
            }
            guard factor.name != Self.orderFactorName else {
                throw Problem(
                    "'\(Self.orderFactorName)' is reserved for the counterbalancing "
                        + "factor — rename the factor")
            }
            guard factorNames.insert(factor.name).inserted else {
                throw Problem(
                    "two factors share the name '\(factor.name)' — factor names "
                        + "must be unique")
            }
            guard !factor.levels.isEmpty else {
                throw Problem(
                    "factor '\(factor.name)' has no levels — every factor needs "
                        + "at least one level")
            }
            var levelNames = Set<String>()
            var levelVariables: Set<String>?
            for level in factor.levels {
                guard Self.safeName(level.name) else {
                    throw Problem(
                        "level '\(level.name)' of factor '\(factor.name)' is not "
                            + "usable in item ids — use letters, digits, and "
                            + "dashes only (e.g. 'high')")
                }
                guard levelNames.insert(level.name).inserted else {
                    throw Problem(
                        "factor '\(factor.name)' has two levels named "
                            + "'\(level.name)' — level names must be unique "
                            + "within a factor")
                }
                for (variable, value) in level.substitutions {
                    guard Self.validVariable(variable) else {
                        throw Problem(
                            "'\(variable)' (level '\(level.name)' of factor "
                                + "'\(factor.name)') is not a valid placeholder "
                                + "variable — use letters, digits, and "
                                + "underscores (e.g. DEMAND)")
                    }
                    if let owner = variableOwner[variable], owner != factor.name {
                        throw Problem(
                            "variable '\(variable)' is supplied by two factors "
                                + "('\(owner)' and '\(factor.name)') — each "
                                + "placeholder must belong to exactly one factor")
                    }
                    variableOwner[variable] = factor.name
                    if value.contains("{{") {
                        throw Problem(
                            "the value for '\(variable)' (level '\(level.name)' "
                                + "of factor '\(factor.name)') contains '{{' — "
                                + "substitution values must be literal text, "
                                + "never another placeholder")
                    }
                }
                // Every level of a factor must fill the SAME variables — a
                // level that forgets one would emit half-filled items.
                let variables = Set(level.substitutions.keys)
                if let expected = levelVariables, expected != variables {
                    throw Problem(
                        "levels of factor '\(factor.name)' fill different "
                            + "variables (\(Self.variableList(expected)) vs "
                            + "\(Self.variableList(variables))) — every level of "
                            + "a factor must supply the same placeholders")
                }
                levelVariables = variables
            }
        }
        if counterbalanceOptionOrder {
            for template in templates
            where (template.options ?? []).count < 2 {
                throw Problem(
                    "counterbalancing flips option order, but template "
                        + "'\(template.id)' has fewer than two options — give "
                        + "every template at least two options or turn "
                        + "counterbalancing off")
            }
        }
    }

    static func validVariable(_ name: String) -> Bool {
        !name.isEmpty
            && name.allSatisfy {
                $0.isLetter && $0.isASCII || $0.isNumber && $0.isASCII
                    || $0 == "_"
            }
    }

    private static func variableList(_ variables: Set<String>) -> String {
        variables.isEmpty ? "none" : variables.sorted().joined(separator: ", ")
    }

    // MARK: - Generation

    /// The synthesized counterbalancing factor's name (reserved).
    public static let orderFactorName = "orderFlipped"

    /// One emitted item: ordinary literal task-prompt data plus the cell's
    /// factor metadata.
    public struct GeneratedItem: Equatable, Sendable {
        public var id: String
        public var text: String
        public var options: [String]?
        public var target: String?
        /// Factor name → level name for this cell (plus
        /// `orderFlipped: "false"/"true"` when counterbalancing is on).
        public var factors: [String: String]
    }

    /// Full crossing: every factor-level combination × every template (×2
    /// orders when counterbalancing), in declared order — templates outermost,
    /// then factors left to right (the LAST factor varies fastest), unflipped
    /// before flipped. No randomness anywhere.
    public func generate() throws -> [GeneratedItem] {
        try validate()
        var items: [GeneratedItem] = []
        for template in templates {
            for cell in Self.crossings(of: factors) {
                var substitutions: [String: String] = [:]
                var cellFactors: [String: String] = [:]
                var idComponents = [template.id]
                for (factor, level) in cell {
                    cellFactors[factor.name] = level.name
                    substitutions.merge(level.substitutions) { _, new in new }
                    idComponents.append("\(factor.name)_\(level.name)")
                }
                let cellName = Self.cellDescription(cell)
                let text = try Self.substitute(
                    template.text, substitutions, field: "text",
                    template: template.id, cell: cellName)
                let options = try template.options.map { options in
                    try options.enumerated().map { index, option in
                        try Self.substitute(
                            option, substitutions, field: "option \(index + 1)",
                            template: template.id, cell: cellName)
                    }
                }
                let target = try template.target.map {
                    try Self.substitute(
                        $0, substitutions, field: "target",
                        template: template.id, cell: cellName)
                }
                if counterbalanceOptionOrder {
                    var original = cellFactors
                    original[Self.orderFactorName] = "false"
                    items.append(
                        GeneratedItem(
                            id: (idComponents + ["\(Self.orderFactorName)_false"])
                                .joined(separator: "-"),
                            text: text, options: options, target: target,
                            factors: original))
                    var flipped = cellFactors
                    flipped[Self.orderFactorName] = "true"
                    // The target names its option BY VALUE, so it follows
                    // the option through the reversal unchanged.
                    items.append(
                        GeneratedItem(
                            id: (idComponents + ["\(Self.orderFactorName)_true"])
                                .joined(separator: "-"),
                            text: text, options: options.map { Array($0.reversed()) },
                            target: target, factors: flipped))
                } else {
                    items.append(
                        GeneratedItem(
                            id: idComponents.joined(separator: "-"),
                            text: text, options: options, target: target,
                            factors: cellFactors))
                }
            }
        }
        return items
    }

    /// All factor-level combinations in deterministic order (last factor
    /// varies fastest). No factors → one empty cell (templates emit as-is).
    private static func crossings(of factors: [Factor]) -> [[(Factor, Level)]] {
        var cells: [[(Factor, Level)]] = [[]]
        for factor in factors {
            cells = cells.flatMap { cell in
                factor.levels.map { cell + [(factor, $0)] }
            }
        }
        return cells
    }

    private static func cellDescription(_ cell: [(Factor, Level)]) -> String {
        cell.isEmpty
            ? "(no factors)"
            : cell.map { "\($0.name)=\($1.name)" }.joined(separator: ", ")
    }

    /// Single-pass literal substitution, then a hard completeness check: an
    /// unsubstituted `{{…}}` ANYWHERE in the output is a plain-language
    /// error naming the variable, template, cell, and field — a half-filled
    /// item is never emitted.
    static func substitute(
        _ text: String, _ substitutions: [String: String],
        field: String, template: String, cell: String
    ) throws -> String {
        var result = text
        for (variable, value) in substitutions.sorted(by: { $0.key < $1.key }) {
            result = result.replacingOccurrences(
                of: "{{\(variable)}}", with: value)
        }
        if let leftover = firstPlaceholder(in: result) {
            throw Problem(
                "template '\(template)', cell \(cell): the \(field) still "
                    + "contains {{\(leftover)}} after substitution — no factor "
                    + "level supplies a value for '\(leftover)'")
        }
        return result
    }

    /// The first `{{…}}` occurrence's inner name (trimmed), or nil.
    static func firstPlaceholder(in text: String) -> String? {
        guard let open = text.range(of: "{{") else { return nil }
        guard let close = text.range(of: "}}", range: open.upperBound..<text.endIndex)
        else {
            // An unclosed '{{' still means an unfilled placeholder-shaped
            // hole — surface what follows it.
            return String(text[open.upperBound...].prefix(30))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return String(text[open.upperBound..<close.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Counts

    /// Total items the design will emit (0 when the spec is structurally
    /// empty).
    public var itemCount: Int {
        let cells = factors.reduce(1) { $0 * $1.levels.count }
        return cells * templates.count * (counterbalanceOptionOrder ? 2 : 1)
    }

    /// The sheet's live count line, e.g.
    /// "2 anchor × 2 frame × 3 templates × 2 orders = 24 items".
    public var cellCountLine: String {
        var parts = factors.map { "\($0.levels.count) \($0.name)" }
        parts.append("\(templates.count) template\(templates.count == 1 ? "" : "s")")
        if counterbalanceOptionOrder { parts.append("2 orders") }
        return parts.joined(separator: " × ")
            + " = \(itemCount) item\(itemCount == 1 ? "" : "s")"
    }

    // MARK: - Serialization (emitted JSONL + design JSON)

    /// The standard task-prompts JSONL shape (`text` + optional
    /// `id`/`options`/`target`, plus the `factors` metadata object — an
    /// extra key both engines' parsers carry along and the
    /// `TaskPromptsDocument` editor round-trips byte-faithfully). Key order
    /// is canonical (sorted), so the same spec always emits the same bytes
    /// and the same pin hash.
    public func generatedJSONL() throws -> Data {
        let items = try generate()
        var out = ""
        for item in items {
            var object: [String: Any] = [
                "id": item.id,
                "text": item.text,
            ]
            if let options = item.options { object["options"] = options }
            if let target = item.target { object["target"] = target }
            if !item.factors.isEmpty { object["factors"] = item.factors }
            let data = try JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys])
            out += String(decoding: data, as: UTF8.self) + "\n"
        }
        return Data(out.utf8)
    }

    /// Canonical (sorted-keys, pretty-printed) design JSON — the sheet's
    /// Save design… format, the study-pack `files` map can carry it, and
    /// the provenance copy written next to the generated output.
    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        return try encoder.encode(self) + Data("\n".utf8)
    }

    public static func decode(_ data: Data) throws -> FactorialDesign {
        do {
            return try JSONDecoder().decode(FactorialDesign.self, from: data)
        } catch {
            throw Problem(
                "not a factorial design JSON file — expected an object with "
                    + "\"factors\" and \"templates\" (details: "
                    + "\(error.localizedDescription))")
        }
    }
}

/// Lands a generated design as ordinary pinned study data through the SAME
/// machinery every importer uses: the study-pack write rule
/// (`TabularImport.writeRefusingDifferingOverwrite` — identical bytes
/// idempotent, differing bytes refuse unless the explicit replace
/// affordance is set), then `ExperimentStore.pinTaskPrompts` (the run
/// loop's own parser re-checks the emitted records), with the caller's
/// manifest save inside the transaction. A failure after the writes rolls
/// BOTH files back exactly as the importers do — a refused generation
/// leaves neither an orphaned nor a clobbered file.
///
/// The design spec lands next to the output as `…-design.json` for
/// PROVENANCE only: it records how the prompts were generated, but the pin
/// — the thing freeze verifies and the run loop re-checks — is on the
/// emitted prompts file itself.
public enum FactorialImport {

    public struct Result: Sendable, Equatable {
        /// Workspace-relative path of the generated prompts JSONL (also set
        /// as the manifest's `taskPromptsFile`).
        public var file: String
        /// Workspace-relative path of the provenance design JSON.
        public var designFile: String
        public var itemCount: Int
        /// The pinned SHA-256 (manifest `taskPromptsHash`).
        public var hash: String
    }

    /// The provenance spec's destination: the prompts destination with
    /// `-design.json` in place of `.jsonl`.
    public static func designDestination(experiment: String) -> String {
        let prompts = DataTemplates.taskPromptsDestination(experiment: experiment)
        guard prompts.hasSuffix(".jsonl") else { return prompts + "-design.json" }
        return String(prompts.dropLast(".jsonl".count)) + "-design.json"
    }

    @discardableResult
    public static func generateIntoStudy(
        design: FactorialDesign, manifest: inout ExperimentManifest,
        replacingExisting: Bool = false,
        persist: (ExperimentManifest) throws -> Void = { _ in }
    ) throws -> Result {
        // Generation validates the WHOLE design before anything is written.
        let jsonl = try design.generatedJSONL()
        let designData = try design.encoded()
        let file = DataTemplates.taskPromptsDestination(experiment: manifest.name)
        let designFile = designDestination(experiment: manifest.name)
        let remedy =
            "check \"Replace the existing file\" to overwrite it "
            + "deliberately, or move it aside and generate again"
        let promptsWrite = try TabularImport.writeRefusingDifferingOverwrite(
            jsonl, relativePath: file,
            replacingExisting: replacingExisting, differingRemedy: remedy)
        let designWrite: TabularImport.ImportWrite
        do {
            designWrite = try TabularImport.writeRefusingDifferingOverwrite(
                designData, relativePath: designFile,
                replacingExisting: replacingExisting, differingRemedy: remedy)
        } catch {
            TabularImport.rollBack(promptsWrite, relativePath: file)
            throw error
        }
        do {
            let hash = try ExperimentStore.pinTaskPrompts(file, into: &manifest)
            try persist(manifest)
            return Result(
                file: file, designFile: designFile,
                itemCount: design.itemCount, hash: hash)
        } catch {
            TabularImport.rollBack(designWrite, relativePath: designFile)
            TabularImport.rollBack(promptsWrite, relativePath: file)
            throw error
        }
    }
}

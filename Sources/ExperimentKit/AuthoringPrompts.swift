import CryptoKit
import Foundation

// =============================================================================
// `authoring prompt <kind>` — the generation-prompt emitter.
//
// The gap: a study is blocked far more often by MISSING DATA than by a missing
// verb, and the answer to missing data is a prompt for an LLM. Those prompts
// were being re-improvised per study, so each one re-learned the same lessons
// the hard way — a corpus whose poles are readable from sentence shape, a
// choice instrument whose longer option is the target, a probe that names the
// concept it is testing. Every re-improvisation also lost the audit numbers,
// which are the only part an acceptor can check.
//
// So the prompts are DATA, in a registry directory, and the emitter's whole job
// is: resolve the template, substitute the study's own seam, stamp a content
// hash, and print. The wording lives in `prompts/authoring-prompts/`, a
// workspace's copy wins over the shipped one, and the hash follows the bytes
// that were actually rendered — so a study's provenance can cite the exact
// prompt a corpus was generated from.
//
// Python twin: `steerlab_server.experiment.authoring_prompts`. The numbers
// below are a cross-engine literal pair, pinned by twin tests, because two
// engines emitting different thresholds for one kind would be two different
// instruments wearing one name.
// =============================================================================

/// The generation-prompt registry: the kinds, their parameters, and the
/// rendering.
public enum AuthoringPrompts {

    // MARK: - Where the templates live

    /// The registry directory, workspace-relative. The DIRECTORY IS THE INDEX
    /// (the `prompts/templates/<id>.json` reader-template rule, one family
    /// over): one file per kind, and the kind is the filename's stem. Files
    /// whose name begins with `_` are shared partials, not kinds.
    public static let registryRelativeDirectory = "prompts/authoring-prompts"

    /// Where one registry file resolves from, workspace copy first. A study
    /// that edits the wording for itself gets its own bytes AND its own
    /// `promptSpecHash`, which is the honest outcome: the emission cites what
    /// was rendered, not what shipped.
    public static func templateURL(
        _ fileName: String, workspaceRoot: URL? = nil, seedRoot: URL? = nil
    ) -> URL {
        DataTemplates.workspaceOrSeedURL(
            relativePath: "\(registryRelativeDirectory)/\(fileName)",
            workspaceRoot: workspaceRoot ?? WorkspaceRoot.current,
            seedRoot: seedRoot)
    }

    // MARK: - The thresholds every prompt carries

    /// The audit numbers the templates interpolate. They are HERE rather than
    /// written into the Markdown because several of them are engine
    /// constants: when a linter threshold moves, every prompt that quotes it
    /// must move with it, or the emitter starts asking for data the engine
    /// will reject.
    ///
    /// Python twin: `authoring_prompts.THRESHOLDS`. Pinned equal by
    /// `AuthoringPromptTests.thresholdsMatchTheServerLiteral` and
    /// `test_thresholds_match_the_swift_literal`.
    public static let thresholds: [String: String] = [
        // Distribution caps — the general form of "a conviction is not a
        // keyword". Both are properties of a whole file, not word lists.
        "stemCapPercent": "40",
        "frameCapPercent": "25",
        // The parity band every matched-rate audit is measured against.
        "parityPercent": "10",
        // Pair length discipline.
        "lengthDeltaWords": "10",
        "minWords": "60",
        "maxWords": "90",
        // Choice-instrument target balance. Twin of the OptVec bundle
        // readiness check's BALANCE_LOW / BALANCE_HIGH, which enforces the
        // same band on the same row shape.
        "balanceLowPercent": "45",
        "balanceHighPercent": "55",
        // Battery lint constants (server `battery_lint`): a battery authored
        // outside these is authored to fail its own linter.
        "optionLengthRatio": "3",
        "minItems": "10",
        "minOptions": "3",
        "maxTokens": "24",
    ]

    // MARK: - The kinds

    /// One declared parameter of one kind.
    public struct Parameter: Sendable, Equatable {
        /// The placeholder it fills, e.g. `concept` for `{{concept}}`.
        public let key: String
        /// The flag that supplies it, e.g. `--concept`.
        public let flag: String
        /// What it is, for `--help` and for the missing-argument refusal.
        public let purpose: String
        /// nil = required. A default is only ever a COUNT or a shape choice;
        /// nothing that describes the study is ever defaulted, because a
        /// plausible default there is a study nobody declared.
        public let defaultValue: String?
        /// Whether the value is a ROW COUNT. Counts are substituted into a
        /// prompt an LLM will obey literally, so `--count bananas` used to
        /// reach the author as an instruction to write "bananas" rows, and
        /// `--count -5` as an instruction nobody can follow. Checked at
        /// emission (review round 6, finding 7). Server twin:
        /// `authoring_prompts.Parameter.is_count`.
        public let isCount: Bool

        public init(
            key: String, flag: String, purpose: String,
            defaultValue: String? = nil, isCount: Bool = false
        ) {
            self.key = key
            self.flag = flag
            self.purpose = purpose
            self.defaultValue = defaultValue
            self.isCount = isCount
        }

        public var isRequired: Bool { defaultValue == nil }
    }

    public struct Kind: Sendable {
        /// As typed, and as the template's filename stem: `contrastive-pairs`.
        public let id: String
        /// One line for `--help`.
        public let purpose: String
        public let parameters: [Parameter]
        /// The workspace-relative file(s) the delivered data lands at, as a
        /// template over the same parameters. Rendered into `{{path}}`.
        public let destination: String

        public var templateFileName: String { "\(id).md" }
    }

    /// Shared parameters every kind takes, declared once.
    static let conceptParameter = Parameter(
        key: "concept", flag: "--concept",
        purpose: "The concept this data is for; also names its destination.")
    static let positiveParameter = Parameter(
        key: "positive", flag: "--positive",
        purpose: "What the positive pole IS, in a sentence or two.")
    static let negativeParameter = Parameter(
        key: "negative", flag: "--negative",
        purpose: "What the negative pole IS — a second considered position, "
            + "never the absence of the first.")

    /// The registry, in `--help` order. Adding a kind here without adding
    /// `prompts/authoring-prompts/<id>.md` is a typed refusal at emission,
    /// never a silently empty prompt.
    public static let kinds: [Kind] = [
        .init(
            id: "contrastive-pairs",
            purpose: "Paired extraction stimuli and their held-out probe.",
            parameters: [
                conceptParameter, positiveParameter, negativeParameter,
                .init(
                    key: "count", flag: "--count",
                    purpose: "Pairs to write.", defaultValue: "48",
                    isCount: true),
                .init(
                    key: "validationCount", flag: "--validation-count",
                    purpose: "Held-out probe rows.", defaultValue: "40",
                    isCount: true),
            ],
            destination: "prompts/concepts/{{concept}}/"),
        .init(
            id: "choice-prompts",
            purpose: "The closed-answer instrument a sweep selects a dose on.",
            parameters: [
                conceptParameter,
                .init(
                    key: "decision", flag: "--decision",
                    purpose: "The decision each row puts to the model, in a "
                        + "sentence or two."),
                .init(
                    key: "count", flag: "--count",
                    purpose: "Rows to write.", defaultValue: "40",
                    isCount: true),
            ],
            destination: "prompts/tasks/{{concept}}-choices.jsonl"),
        .init(
            id: "validation-set",
            purpose: "A held-out, vocabulary-free probe on its own.",
            parameters: [
                conceptParameter, positiveParameter, negativeParameter,
                .init(
                    key: "count", flag: "--count",
                    purpose: "Rows to write.", defaultValue: "40",
                    isCount: true),
            ],
            destination: "prompts/concepts/{{concept}}/validation.jsonl"),
        .init(
            id: "reader-pairs",
            purpose: "The dataset a RepE reader is fitted on.",
            parameters: [
                conceptParameter, positiveParameter, negativeParameter,
                .init(
                    key: "templateID", flag: "--template-id",
                    purpose: "The task template the fit will use; every row "
                        + "declares it."),
                .init(
                    key: "shape", flag: "--shape",
                    purpose: "contentPair (two texts, one template) or "
                        + "singleStimulus (one text, a template pair).",
                    defaultValue: "contentPair"),
                .init(
                    key: "count", flag: "--count",
                    purpose: "Rows to write.", defaultValue: "40",
                    isCount: true),
                .init(
                    key: "heldOut", flag: "--held-out",
                    purpose: "Trailing rows marked split \"test\"; they decide "
                        + "the direction's sign.", defaultValue: "10",
                    isCount: true),
            ],
            destination: "prompts/readers/{{concept}}/pairs.jsonl"),
        .init(
            id: "battery",
            purpose: "A format-2 capability battery — the sweep's brake.",
            parameters: [
                .init(
                    key: "name", flag: "--name",
                    purpose: "Names the battery file.",
                    defaultValue: "capability"),
                .init(
                    key: "count", flag: "--count",
                    purpose: "Items to write.", defaultValue: "20",
                    isCount: true),
            ],
            destination: "prompts/batteries/{{name}}.jsonl"),
    ]

    public static func kind(_ id: String) -> Kind? {
        kinds.first { $0.id == id }
    }

    /// The closed shape vocabulary for `reader-pairs`, and the partial each
    /// one contributes. The two shapes fit DIFFERENT contrasts and may not be
    /// mixed in one file, so the guidance is per-shape data rather than a
    /// paragraph the emitter concatenates.
    public static let readerShapes = ["contentPair", "singleStimulus"]

    /// The largest row count any kind will ask for. A STATED ceiling, not a
    /// technical one: the delivery is one LLM generation, and past a few
    /// hundred rows a model starts repeating itself while the batch stops
    /// being reviewable by the second acceptor this verb's whole design turns
    /// on. A study that genuinely needs more emits twice and reviews twice.
    /// Server twin: `authoring_prompts.MAXIMUM_COUNT`.
    public static let maximumCount = 500

    /// The integer behind a count flag, or a typed usage refusal naming the
    /// value that is not one.
    ///
    /// Counts are the one parameter class with a machine-checkable shape, and
    /// they are substituted straight into a prompt an LLM will obey literally:
    /// `--count bananas` asked an author for "bananas" rows and `--count -5`
    /// asked for something nobody can deliver, and both emitted a perfectly
    /// well-formed prompt with a hash. Review round 6, finding 7. Server twin:
    /// `authoring_prompts.count_value`.
    static func countValue(
        _ parameter: Parameter, _ value: String, kindID: String, program: String
    ) throws -> Int {
        let text = value.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, text.allSatisfy(\.isASCII), text.allSatisfy(\.isNumber),
            let number = Int(text), number > 0
        else {
            throw ExperimentError.malformed(
                "\(parameter.flag) takes a whole number of rows above 0 — got "
                    + "'\(value)'. It is substituted into the prompt verbatim, "
                    + "so an author would be asked for exactly that many",
                repair: "\(program) authoring prompt \(kindID) "
                    + "\(parameter.flag) \(parameter.defaultValue ?? "40")")
        }
        guard number <= maximumCount else {
            throw ExperimentError.malformed(
                "\(parameter.flag) is \(number), above the ceiling of "
                    + "\(maximumCount) — one delivery is one generation, and "
                    + "past a few hundred rows a model repeats itself and a "
                    + "second acceptor cannot review the batch. Emit twice and "
                    + "review twice instead",
                repair: "\(program) authoring prompt \(kindID) "
                    + "\(parameter.flag) \(maximumCount)")
        }
        return number
    }

    /// THE refusal for a held-out split that is not a split. Server twin:
    /// `authoring_prompts.held_out_exceeds_count_message`.
    static func heldOutExceedsCountMessage(heldOut: Int, count: Int) -> String {
        "--held-out is \(heldOut) of \(count) rows, which leaves "
            + "\(count - heldOut) to fit on — the held-out rows are the "
            + "TRAILING rows of the same file, so they have to be fewer than "
            + "the total. They decide the direction's sign; a fit with nothing "
            + "left to fit on has no direction for them to sign"
    }

    static func readerShapePartial(_ shape: String) -> String {
        "_reader-shape-\(shape).md"
    }

    // MARK: - Emission

    public struct Emission: Sendable {
        /// The rendered prompt. This is the whole product.
        public let text: String
        public let kind: String
        /// SHA-256 over the partials and the template, in assembly order —
        /// what a study's PROVENANCE cites as `promptSpecHash`. It identifies
        /// the WORDING, not this emission: change only the concept and it does
        /// not move.
        public let promptSpecHash: String
        /// SHA-256 over the rendered body and the resolved parameters — what
        /// identifies THIS emission (`promptInstanceHash`). Change only the
        /// concept and this moves while `promptSpecHash` does not.
        public let promptInstanceHash: String
        /// The registry files that were read, workspace-relative, in assembly
        /// order.
        public let templateFiles: [String]
        /// True when the workspace's own copy of the FIRST file was used —
        /// reported because it changes the hash and is otherwise invisible.
        public let fromWorkspaceCopy: Bool
        /// Every parameter as resolved, defaults included.
        public let parameters: [String: String]
        /// Where the delivered data is to land.
        public let destination: String
    }

    /// The default program spelling in every repair below — the Mac CLI's,
    /// because that is the binary most callers have. The CROSS-PLATFORM
    /// client passes its own name instead (`steerlab`), which is why these
    /// take a program at all: this verb exists on both surfaces, so a repair
    /// that hard-coded one binary would send half its readers to a command
    /// they do not have. Server twin: `authoring_prompts.DEFAULT_PROGRAM`.
    public static let defaultProgram = "steerlab-cli"

    /// THE repair for a kind nobody declared.
    public static func unknownKindRepair(program: String = defaultProgram)
        -> String
    {
        "\(program) authoring prompt <"
            + kinds.map(\.id).joined(separator: "|") + "> …"
    }

    /// THE repair for a required parameter nobody supplied.
    public static func missingParametersRepair(
        kind: Kind, missing: [Parameter], program: String = defaultProgram
    ) -> String {
        (["\(program) authoring prompt \(kind.id)"]
            + missing.map { "\($0.flag) \"…\"" }).joined(separator: " ")
    }

    /// THE repair for a registry file that is not on disk.
    public static func missingTemplateRepair(
        _ relativePath: String, program: String = defaultProgram
    ) -> String {
        "restore \(relativePath) from the shipped seed tree, or re-create the "
            + "workspace with \(program) workspace init <path>  (the "
            + "emitter reads the workspace's copy first and the shipped copy "
            + "second, and refuses rather than emitting a prompt with a hole "
            + "in it)"
    }

    /// Render one kind with one set of arguments.
    ///
    /// `arguments` are the caller's own, keyed by parameter KEY (not flag) —
    /// so the CLI, an HTTP caller, and a test all reach the same function
    /// through the same vocabulary. Unknown keys are refused rather than
    /// ignored: a misspelled parameter that silently left a `{{placeholder}}`
    /// in the emitted prompt is a prompt an LLM would answer literally.
    public static func emit(
        kind kindID: String, arguments: [String: String],
        program: String = defaultProgram,
        workspaceRoot: URL? = nil, seedRoot: URL? = nil
    ) throws -> Emission {
        guard let kind = kind(kindID) else {
            throw ExperimentError.malformed(
                "no authoring prompt for kind '\(kindID)' — known kinds: "
                    + kinds.map(\.id).joined(separator: ", "),
                repair: unknownKindRepair(program: program))
        }
        let declared = Set(kind.parameters.map(\.key))
        let unknown = arguments.keys.filter { !declared.contains($0) }.sorted()
        if let first = unknown.first {
            throw ExperimentError.malformed(
                "'\(kindID)' takes no parameter '\(first)' — it takes "
                    + kind.parameters.map(\.key).joined(separator: ", "),
                repair: "\(program) authoring prompt \(kindID) --help")
        }
        var values: [String: String] = thresholds
        var resolved: [String: String] = [:]
        var missing: [Parameter] = []
        for parameter in kind.parameters {
            let given = arguments[parameter.key]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let given, !given.isEmpty {
                resolved[parameter.key] = given
            } else if let fallback = parameter.defaultValue {
                resolved[parameter.key] = fallback
            } else {
                missing.append(parameter)
            }
        }
        guard missing.isEmpty else {
            throw ExperimentError.malformed(
                "'\(kindID)' needs "
                    + missing.map { "\($0.flag) (\($0.purpose))" }
                        .joined(separator: ", ")
                    + " — these describe the STUDY, and a plausible default "
                    + "for any of them would be a study nobody declared",
                repair: missingParametersRepair(
                    kind: kind, missing: missing, program: program))
        }
        if kindID == "reader-pairs" {
            let shape = resolved["shape"] ?? ""
            guard readerShapes.contains(shape) else {
                throw ExperimentError.malformed(
                    "unknown reader shape '\(shape)' — known shapes: "
                        + readerShapes.joined(separator: ", "),
                    repair: "\(program) authoring prompt reader-pairs "
                        + "--shape \(readerShapes.joined(separator: "|")) …")
            }
        }
        // Counts are the one parameter class with a checkable shape, and they
        // are substituted into a prompt an LLM obeys literally (review round
        // 6, finding 7). Checked per field first, then across fields.
        var counts: [String: Int] = [:]
        for parameter in kind.parameters where parameter.isCount {
            counts[parameter.key] = try countValue(
                parameter, resolved[parameter.key] ?? "", kindID: kindID,
                program: program)
        }
        if let heldOut = counts["heldOut"], let count = counts["count"],
            heldOut >= count
        {
            throw ExperimentError.malformed(
                heldOutExceedsCountMessage(heldOut: heldOut, count: count),
                repair: "\(program) authoring prompt \(kindID) --count "
                    + "\(count) --held-out \(max(1, count / 4)) …")
        }
        for (key, value) in resolved { values[key] = value }
        values["path"] = substitute(kind.destination, values)

        // Assembly order IS hash order: the shape partial (when the kind has
        // one), then the two universal partials, then the template. A reader
        // reproducing the hash reads this list top to bottom.
        var files: [String] = []
        if kindID == "reader-pairs", let shape = resolved["shape"] {
            files.append(readerShapePartial(shape))
        }
        files.append(contentsOf: ["_discipline.md", "_delivery.md"])
        files.append(kind.templateFileName)

        var bodies: [String: String] = [:]
        var hasher = SHA256()
        var fromWorkspace = false
        for (index, file) in files.enumerated() {
            let url = templateURL(
                file, workspaceRoot: workspaceRoot, seedRoot: seedRoot)
            guard let data = FileManager.default.contents(atPath: url.path)
            else {
                let relative = "\(registryRelativeDirectory)/\(file)"
                throw ExperimentError.refusing(
                    .missingPrerequisite,
                    "the authoring-prompt registry has no \(relative) — "
                        + "'\(kindID)' is assembled from "
                        + files.joined(separator: " + ")
                        + " and cannot be emitted without all of them",
                    repair: missingTemplateRepair(relative, program: program))
            }
            hasher.update(data: data)
            bodies[file] = String(decoding: data, as: UTF8.self)
            if index == 0 {
                let workspaceCopy = (workspaceRoot ?? WorkspaceRoot.current)
                    .appending(path: "\(registryRelativeDirectory)/\(file)")
                fromWorkspace = url.standardizedFileURL
                    == workspaceCopy.standardizedFileURL
            }
        }
        let hash = hasher.finalize()
            .map { String(format: "%02x", $0) }
            .joined()

        // Partials are substituted FIRST, then spliced into the template, so a
        // threshold reaches a partial's own text as well as the template's.
        if let shape = resolved["shape"],
            let block = bodies[readerShapePartial(shape)]
        {
            values["shapeBlock"] = substitute(block, values).trimmedTrailing()
        }
        values["discipline"] = substitute(bodies["_discipline.md"] ?? "", values)
            .trimmedTrailing()
        values["delivery"] = substitute(bodies["_delivery.md"] ?? "", values)
            .trimmedTrailing()
        let body = substitute(bodies[kind.templateFileName] ?? "", values)

        // Over the FINAL body, header excluded: the header carries the hash,
        // so hashing it would be circular.
        let instance = instanceDigest(
            body: body.trimmedTrailing() + "\n", parameters: resolved)
        let text = header(
            kind: kindID, hash: hash, instance: instance, files: files)
            + "\n\n" + body.trimmedTrailing() + "\n"
        return Emission(
            text: text, kind: kindID, promptSpecHash: hash,
            promptInstanceHash: instance,
            templateFiles: files.map { "\(registryRelativeDirectory)/\($0)" },
            fromWorkspaceCopy: fromWorkspace, parameters: resolved,
            destination: values["path"] ?? kind.destination)
    }

    /// SHA-256 over THIS EMISSION: the fully rendered body and the resolved
    /// parameter set that produced it.
    ///
    /// The spec hash identifies the WORDING — the template and partials,
    /// before substitution — so two emissions of the same kind for two
    /// different concepts share it. That is the right identity for "which
    /// prompt text is this study citing", and the wrong one for "which
    /// emission produced this corpus": nothing in the spec hash distinguishes
    /// the run that asked for concept A from the run that asked for B. This is
    /// the second half (review round 6, finding 5), and the two are stamped
    /// side by side because they answer different questions.
    ///
    /// Framing: the body's bytes, then each `key` and `value` of the resolved
    /// parameters in key order, every field preceded by a NUL. The NULs are
    /// not decoration — without them a value ending in a key's name could
    /// forge a field boundary and two different parameter sets could hash the
    /// same. Server twin: `authoring_prompts.instance_digest`.
    static func instanceDigest(body: String, parameters: [String: String])
        -> String
    {
        var hasher = SHA256()
        hasher.update(data: Data(body.utf8))
        for key in parameters.keys.sorted() {
            hasher.update(data: Data([0]))
            hasher.update(data: Data(key.utf8))
            hasher.update(data: Data([0]))
            hasher.update(data: Data((parameters[key] ?? "").utf8))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The stamped first line. An HTML comment so it does not render as prose
    /// when the prompt is pasted into a chat, and so an acceptor reading a
    /// delivery can recover exactly which prompt produced it.
    ///
    /// Both hashes are on it, because they recover different things:
    /// `promptSpecHash` recovers the WORDING (which template and partials, at
    /// which bytes) and `promptInstanceHash` recovers the EMISSION (that
    /// wording, rendered with these parameters).
    static func header(
        kind: String, hash: String, instance: String, files: [String]
    ) -> String {
        "<!-- steerlab authoring prompt — kind: \(kind); promptSpecHash: "
            + "sha256:\(hash); promptInstanceHash: sha256:\(instance); "
            + "assembled from: "
            + files.joined(separator: " + ") + " -->"
    }

    /// `{{key}}` → value, single pass over the KEYS (never re-scanning
    /// substituted text): a substituted value that happened to contain
    /// `{{count}}` must not then be substituted itself — the study's own words
    /// are data, not template.
    static func substitute(_ template: String, _ values: [String: String]) -> String {
        var out = ""
        out.reserveCapacity(template.count)
        var rest = Substring(template)
        while let open = rest.range(of: "{{") {
            guard let close = rest[open.upperBound...].range(of: "}}") else {
                break
            }
            let key = String(rest[open.upperBound ..< close.lowerBound])
            out += rest[rest.startIndex ..< open.lowerBound]
            // An unknown placeholder survives VERBATIM rather than becoming
            // an empty string: a hole an LLM would answer literally has to be
            // visible in the emitted text and in the test that reads it.
            out += values[key] ?? "{{\(key)}}"
            rest = rest[close.upperBound...]
        }
        out += rest
        return out
    }
}

extension String {
    /// Trailing newlines and spaces removed — partials are spliced INTO a
    /// document, and a partial's own trailing blank lines would open a gap
    /// wherever it lands.
    func trimmedTrailing() -> String {
        var out = self
        while let last = out.last, last.isNewline || last == " " {
            out.removeLast()
        }
        return out
    }
}

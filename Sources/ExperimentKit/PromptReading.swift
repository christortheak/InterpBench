import Foundation

/// How a study's prompts go through the model on the PYTHON engine: in one
/// pass, or in chunks (issue #6). Swift twin of
/// `Server/steerlab_server/experiment/prompt_reading.py`, sentence for
/// sentence; the hand-authored contract both check is
/// `Tests/Fixtures/cross-engine/prompt-reading.json`.
///
/// The rule is one JSON object, `{"mode": "singlePass"}` or
/// `{"mode": "chunked", "chunkTokens": N, "aboveTokens": M}`. A prompt of `n`
/// tokens is chunked when the mode is chunked and `n > max(aboveTokens,
/// chunkTokens)`. Chunking changes only the floating-point summation order,
/// so seeded texts can differ at near ties; that is why a study may declare
/// its reading, and why the declaration is frozen with it.
///
/// This engine (MLX) does not read the declaration: it reads every prompt in
/// its own native steps of `mlxPrefillStepSize` tokens. A study that declares
/// one is not refused here; it gets one advisory at run start
/// (`macEngineAdvisory`). What this file owns on the Mac is the manifest
/// key's shape and sentences, the authoring verb, and the refusal to send a
/// declaring study to an engine that would ignore it.
public enum PromptReading {
    public static let declarationKey = "promptReading"
    public static let verb = "set-prompt-reading"
    public static let singlePass = "singlePass"
    public static let chunked = "chunked"
    static let ruleKeys: Set<String> = ["mode", "chunkTokens", "aboveTokens"]
    /// The largest token count either number may hold (Python twin
    /// `MAX_RULE_TOKENS`).
    public static let maxRuleTokens = 100_000_000

    /// The step this engine's prefill reads a prompt in
    /// (`GenerateParameters.prefillStepSize` at every study generation site
    /// in `ExperimentTasks`, and the value `sampling-provenance.json` records).
    public static let mlxPrefillStepSize = 512

    /// `remoteStudy.promptReading` in the capability snapshot.
    public static let capabilityKey = "remoteStudy.promptReading"

    /// The rule in canonical form.
    public struct Rule: Equatable, Sendable {
        public var chunkTokens: Int?
        public var aboveTokens: Int?
        public var isChunked: Bool { chunkTokens != nil }

        public static let single = Rule(chunkTokens: nil, aboveTokens: nil)

        public static func chunked(_ chunk: Int, above: Int) -> Rule {
            Rule(chunkTokens: chunk, aboveTokens: above)
        }

        /// The block as it is written to the manifest.
        public var jsonValue: ExactJSONValue {
            guard let chunkTokens, let aboveTokens else {
                return .object(["mode": .string(PromptReading.singlePass)])
            }
            return .object([
                "mode": .string(PromptReading.chunked),
                "chunkTokens": .integer(Int64(chunkTokens)),
                "aboveTokens": .integer(Int64(aboveTokens)),
            ])
        }

        /// The block in a JSON result payload.
        public var payload: JSONValue {
            guard let chunkTokens, let aboveTokens else {
                return .object(["mode": .string(PromptReading.singlePass)])
            }
            return .object([
                "mode": .string(PromptReading.chunked),
                "chunkTokens": .number(Double(chunkTokens)),
                "aboveTokens": .number(Double(aboveTokens)),
            ])
        }
    }

    // MARK: - The shape (Python twin: violations, parse, chunk_size)

    static func wholeNumber(_ value: ExactJSONValue?) -> Int? {
        switch value {
        case .integer(let number)?: return Int(exactly: number)
        case .unsigned(let number)?: return Int(exactly: number)
        default: return nil
        }
    }

    /// Every way `value` departs from the rule's shape, in plain words.
    public static func violations(_ value: ExactJSONValue) -> [String] {
        guard case .object(let object) = value else {
            return ["promptReading must be an object: {\"mode\": \"singlePass\"} "
                + "or {\"mode\": \"chunked\", \"chunkTokens\": <n>, "
                + "\"aboveTokens\": <m>}"]
        }
        var problems: [String] = []
        let unknown = object.keys.filter { !ruleKeys.contains($0) }
            .sorted { $0.utf8.lexicographicallyPrecedes($1.utf8) }
        if !unknown.isEmpty {
            problems.append("promptReading has unknown key(s) "
                + unknown.joined(separator: ", ")
                + " — it takes only mode, chunkTokens, and aboveTokens")
        }
        guard case .string(let mode)? = object["mode"], mode == singlePass || mode == chunked
        else {
            problems.append("promptReading mode must be singlePass or chunked")
            return problems
        }
        if mode == singlePass {
            if object["chunkTokens"] != nil || object["aboveTokens"] != nil {
                problems.append(
                    "promptReading mode singlePass takes no chunkTokens or aboveTokens")
            }
            return problems
        }
        for (key, minimum) in [("chunkTokens", 1), ("aboveTokens", 0)] {
            guard let raw = object[key] else {
                problems.append("promptReading mode chunked needs \(key)")
                continue
            }
            guard let number = wholeNumber(raw), number >= minimum, number <= maxRuleTokens
            else {
                problems.append("promptReading \(key) must be a whole number from "
                    + "\(minimum) to \(grouped(maxRuleTokens))")
                continue
            }
        }
        return problems
    }

    /// The rule, or nil when `value` is outside the contract.
    public static func rule(_ value: ExactJSONValue?) -> Rule? {
        guard let value, violations(value).isEmpty, case .object(let object) = value
        else { return nil }
        if case .string(singlePass)? = object["mode"] { return .single }
        guard let chunk = wholeNumber(object["chunkTokens"]),
            let above = wholeNumber(object["aboveTokens"])
        else { return nil }
        return .chunked(chunk, above: above)
    }

    /// The chunk a prompt of `promptTokens` is read in, or nil for one pass.
    public static func chunkSize(_ rule: Rule, promptTokens: Int) -> Int? {
        guard let chunk = rule.chunkTokens, let above = rule.aboveTokens else { return nil }
        return promptTokens > max(above, chunk) ? chunk : nil
    }

    /// `8192` → `8,192`, the Python client's `{:,}`.
    static func grouped(_ number: Int) -> String {
        let digits = String(number.magnitude)
        var out = ""
        for (index, character) in digits.enumerated() {
            if index > 0, (digits.count - index) % 3 == 0 { out.append(",") }
            out.append(character)
        }
        return (number < 0 ? "-" : "") + out
    }

    /// The rule in a researcher's words.
    public static func describe(_ rule: Rule) -> String {
        guard let chunk = rule.chunkTokens, let above = rule.aboveTokens else {
            return "every prompt is read in a single pass"
        }
        return "prompts longer than \(grouped(max(above, chunk))) tokens are read in "
            + "chunks of \(grouped(chunk)) tokens; shorter prompts are read in a "
            + "single pass"
    }

    // MARK: - The verb (Python twin: rule_from_flags, echo, result_payload)

    public static func usage(_ name: String) -> String {
        "steerlab-cli experiment \(verb) \(name) --single-pass  |  "
            + "steerlab-cli experiment \(verb) \(name) --chunk-tokens <n> "
            + "--above-tokens <m>  |  steerlab-cli experiment \(verb) \(name) --clear"
    }

    /// The flags → a rule, or nil for `--clear`. Malformed (64) otherwise,
    /// with the usage as the repair.
    public static func rule(
        singlePass: Bool, chunkTokens: String?, aboveTokens: String?, clear: Bool,
        name: String
    ) throws -> Rule? {
        let chunkedFlags = chunkTokens != nil || aboveTokens != nil
        let chosen = [singlePass, chunkedFlags, clear].filter { $0 }.count
        guard chosen == 1 else {
            throw ExperimentError.malformed(
                "choose one reading: --single-pass, or --chunk-tokens with "
                    + "--above-tokens, or --clear", repair: usage(name))
        }
        if clear { return nil }
        if singlePass { return .single }
        guard let chunkTokens, let aboveTokens else {
            throw ExperimentError.malformed(
                "a chunked reading needs both --chunk-tokens and --above-tokens",
                repair: usage(name))
        }
        func number(_ raw: String, flag: String, minimum: Int) throws -> Int {
            let text = raw.trimmingCharacters(in: .whitespaces)
            guard (1...9).contains(text.utf8.count),
                text.utf8.allSatisfy({ (0x30...0x39).contains($0) }),
                let value = Int(text), value >= minimum, value <= maxRuleTokens
            else {
                throw ExperimentError.malformed(
                    "\(flag) must be a whole number from \(minimum) to "
                        + grouped(maxRuleTokens), repair: usage(name))
            }
            return value
        }
        return .chunked(
            try number(chunkTokens, flag: "--chunk-tokens", minimum: 1),
            above: try number(aboveTokens, flag: "--above-tokens", minimum: 0))
    }

    /// The verb's one line, the same on both clients.
    public static func echo(
        name: String, rule: Rule?, previous: ExactJSONValue?, dryRun: Bool
    ) -> String {
        let prefix = dryRun ? "(dry run, nothing written) would " : ""
        guard let rule else {
            if previous == nil {
                return "'\(name)' declares no prompt reading — nothing to clear"
            }
            return prefix + (dryRun ? "clear" : "cleared") + " the prompt reading of "
                + "'\(name)'; the engine's device default applies again"
        }
        if let previous, self.rule(previous) == rule {
            return "'\(name)' already declares this prompt reading: \(describe(rule))"
        }
        return prefix + (dryRun ? "declare" : "declared")
            + " the prompt reading of '\(name)': \(describe(rule))"
    }

    static func payload(_ value: ExactJSONValue?) -> JSONValue {
        guard let value else { return .null }
        if let rule = rule(value) { return rule.payload }
        // A hand-edited block outside the contract, echoed as it was.
        return (try? JSONDecoder().decode(
            JSONValue.self, from: JSONEncoder().encode(value))) ?? .null
    }

    /// `experiment set-prompt-reading`, from the strict parser's `args` (the
    /// verb, its positional, and its flags). Python twin:
    /// `client_cli._set_prompt_reading`, with the same flat result keys.
    static func run(_ args: [String], sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let valueFlags: Set<String> = ["--chunk-tokens", "--above-tokens"]
        var positionals: [String] = []
        var values: [String: String] = [:]
        var switches = Set<String>()
        var index = 1
        while index < args.count {
            let token = args[index]
            if valueFlags.contains(token), index + 1 < args.count {
                values[token] = args[index + 1]
                index += 2
            } else if token.hasPrefix("--") {
                switches.insert(token)
                index += 1
            } else {
                positionals.append(token)
                index += 1
            }
        }
        guard positionals.count == 1 else {
            throw ExperimentError.malformed(
                "usage: experiment \(verb) <name> (--single-pass | --chunk-tokens <n> "
                    + "--above-tokens <m> | --clear) [--dry-run]",
                repair: usage(positionals.first ?? "<name>"))
        }
        let name = positionals[0]
        let dryRun = switches.contains("--dry-run")
        // Shape first, in the Python client's order: a frozen study given a
        // well-formed reading answers with its duplicate repair.
        let rule = try rule(
            singlePass: switches.contains("--single-pass"),
            chunkTokens: values["--chunk-tokens"], aboveTokens: values["--above-tokens"],
            clear: switches.contains("--clear"), name: name)
        let (previous, written) = try declare(name: name, rule: rule, dryRun: dryRun)
        let line = echo(name: name, rule: rule, previous: previous, dryRun: dryRun)
        sink.out(line)
        return ExperimentCLIResult(
            message: line, changed: written,
            payload: [
                "experiment": .string(name),
                "promptReading": rule?.payload ?? .null,
                "previousPromptReading": payload(previous),
                "reading": rule.map { .string(describe($0)) } ?? .null,
                "written": .bool(written),
            ])
    }

    /// Write (or, with `rule` nil, clear) the declaration on a draft.
    /// Returns the declaration it replaced and whether the file changed. A
    /// frozen study refuses with the duplicate-to-iterate repair, on a dry
    /// run too; nothing changing is not a write.
    @discardableResult
    public static func declare(
        name: String, rule: Rule?, dryRun: Bool = false
    ) throws -> (previous: ExactJSONValue?, written: Bool) {
        let current = try ExperimentStore.load(name: name)
        try ManifestMutationPolicy.admitDraftEdit(current)
        let previous = current.promptReading
        let changes = rule.map { previous != $0.jsonValue } ?? (previous != nil)
        guard changes, !dryRun else { return (previous, false) }
        try ExperimentStore.updateDraft(name: name) { manifest in
            manifest.promptReading = rule?.jsonValue
        }
        return (previous, true)
    }

    // MARK: - Engines (Python twin: outdated_engine_reason/repair)

    /// The refusal for sending a declaring study to an engine without
    /// `remoteStudy.promptReading`. Same sentence as the Python client's.
    public static func outdatedEngineReason(server: String) -> String {
        "The engine at \(server) predates declared prompt reading: its "
            + "capabilities do not list remoteStudy.promptReading, so it would "
            + "ignore this study's promptReading declaration and read its "
            + "prompts its own way, and the texts would not be the ones the "
            + "study declared. Nothing was submitted."
    }

    public static let outdatedEngineRepair =
        "Bring that engine up to date (for a cluster site, steerlab-cli "
        + "cluster push --site <id>; for a local runner, restart it from "
        + "this release), then submit again."

    /// The one advisory a study that declares a reading gets when it runs
    /// on this engine. Not a refusal: the declaration says how the Python
    /// engine reads long prompts, and this engine has its own native step.
    public static func macEngineAdvisory(_ manifest: ExperimentManifest) -> String? {
        guard let declared = manifest.promptReading else { return nil }
        let words = rule(declared).map(describe) ?? "a reading outside the contract"
        return "This study declares how the Python engine reads its prompts "
            + "(promptReading: \(words)). The declaration applies to the Python "
            + "engine; this Mac engine reads every prompt in its own steps of "
            + "\(grouped(mlxPrefillStepSize)) tokens, so this run did not use the "
            + "declared reading, and its texts are not comparable token for token "
            + "with runs of this study on the Python engine."
    }
}

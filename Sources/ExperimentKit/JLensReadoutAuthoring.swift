import Foundation

/// `experiment set-jlens-readout` — author a draft study's J-lens run readout
/// (`jlensReadout`). Python twin: `experiment/jlens_readout_authoring.py`.
///
/// The engine has always read, priced, and enforced this block; no verb wrote
/// it, and a manifest is never hand-edited, so no researcher could arm a
/// readout at all.
///
/// **The server-only rule, and why this engine takes the pins as flags.** The
/// Mac engine never resolves or loads a lens. Reading `lens.json` for its
/// pins would not load a tensor, but it would resolve a lens: the record is
/// the Python engine's own schema (its qualifications, tier declarations, and
/// converted cache), it usually lives in the engine's root rather than in
/// this workspace, and the two remaining pins are not in it at all — the
/// tokenizer hash is read from the runtime's model cache and the
/// qualification is matched against the runtime. A second reader here would
/// be a second owner of that schema. So the lens pins (`--lens-sha256`,
/// `--lens-model`, `--fitted-layers`) and the optional `--tokenizer-hash` and
/// `--qualification` are typed, exactly as the Python client's `--dry-run`
/// prints them (`result.pins`, `result.macCommand`), and the Python engine
/// re-checks every one at verify and at run start. The settings rules, the
/// block, the configuration hash, and the budget pricing are mirrored here,
/// and the cross-client parity test holds the two clients to the same bytes.
public enum JLensReadoutAuthoring {
    public static let verb = "set-jlens-readout"

    /// The engine's default ceilings (`jlens.readout.Budget`). `topK` and the
    /// watchlist are capped at their defaults: the cap belongs to the
    /// instrument, not to a budget a study raises here.
    static let defaultBudget: [String: Int] = [
        "maxArmedLayers": 8, "maxWatchlist": 64, "maxTopK": 50,
        "maxObservations": 2_000_000, "maxTraceBytes": 2 << 30,
        "maxFullVocabProjections": 200_000,
    ]
    static let maxTopK = 50
    static let maxWatchlist = 64
    static let maxTokenID = 2_147_483_647
    /// Whole numbers up to 10^15: every value then encodes as an integer on
    /// both clients, so the configuration hash cannot differ by a rounding.
    static let maxBudgetValue = 1_000_000_000_000_000

    /// The ceilings a study may declare, flag → budget field, in the order
    /// the canonical spelling lists them.
    public static let budgetFlags: [(flag: String, field: String)] = [
        ("--max-armed-layers", "maxArmedLayers"),
        ("--max-observations", "maxObservations"),
        ("--max-trace-bytes", "maxTraceBytes"),
        ("--max-full-vocab-projections", "maxFullVocabProjections"),
    ]
    /// The ceilings that scale with the number of responses, with the
    /// per-response quantity each bounds, in the order a binding one is named.
    static let proportionalCeilings: [(ceiling: String, quantity: String)] = [
        ("maxFullVocabProjections", "fullVocabProjections"),
        ("maxTraceBytes", "projectedTraceBytes"),
        ("maxObservations", "observations"),
    ]
    static let ceilingWords = [
        "maxFullVocabProjections": "full-vocabulary projections",
        "maxTraceBytes": "trace bytes", "maxObservations": "observations",
    ]

    /// The conventions the configuration hash covers — the server's
    /// `jlens.recorder` and `jlens.schemas` constants, restated.
    static let observationConvention = "postInterventionBlockOutput"
    static let alignmentConvention =
        "predictionAligned: position p predicts generated token p-promptLen+1"
    static let readoutConvention =
        "softcap(U.RMSNorm(J_l h)); final-norm gain g folded into token row "
        + "(g=1+w offset RMSNorm, g=w direct RMSNorm; observed per model)"
    static let directionConvention = "J_l^T (g . u_t)"

    /// What the researcher declared, already parsed and range-checked.
    public struct Settings: Sendable, Equatable {
        public var layers: [Int]
        public var topK: Int = 0
        public var topKLayers: [Int]?
        public var watchlist: [Int] = []
        public var companion = true
        public var budget: [String: Int] = [:]
        public var qualificationID: String?
        public var tokenizerHash: String?
        public var recordTokenIDs = false

        public init(layers: [Int]) { self.layers = layers }
    }

    /// The lens pins this engine is given rather than reads.
    public struct Pins: Sendable, Equatable {
        public var lensID: String
        public var lensSHA256: String
        /// `<model-id>` or `<model-id>@<revision>` when the lens records the
        /// revision it was fitted on.
        public var lensModel: String
        public var fittedLayers: [Int]

        public init(lensID: String, lensSHA256: String, lensModel: String, fittedLayers: [Int]) {
            self.lensID = lensID
            self.lensSHA256 = lensSHA256
            self.lensModel = lensModel
            self.fittedLayers = fittedLayers
        }
    }

    public struct Report: Sendable {
        public var manifest: ExperimentManifest
        public var block: [String: JSONValue]
        public var estimate: [String: JSONValue]
        public var macCommand: String
        public var notes: [String]
    }

    static func usage(_ name: String) -> String {
        "steerlab-cli experiment \(verb) \(name) <lens-id> --lens-sha256 <sha256> "
            + "--lens-model <model-id> --fitted-layers <l,…> --layers <l,…> "
            + "[--top-k <k>] [--watchlist <token-id,…>]  (\"\" as the lens id "
            + "clears the readout; steerlab experiment \(verb) \(name) <lens-id> "
            + "--layers <l,…> --dry-run --json, where the lens is imported, "
            + "prints the pins as result.macCommand)"
    }

    static func malformed(_ reason: String, _ name: String, repair: String? = nil) -> ExperimentError {
        .malformed(reason, repair: repair ?? usage(name))
    }

    // MARK: - Parsing (Python twin: parse_*)

    static func isDigits(_ text: Substring, max: Int) -> Bool {
        !text.isEmpty && text.count <= max && text.allSatisfy { ("0"..."9").contains($0) }
    }

    /// Comma-separated layer indices or inclusive ranges (`20-24`), sorted
    /// and de-duplicated.
    public static func parseLayers(_ text: String, flag: String, name: String) throws -> [Int] {
        let parts = text.split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        guard !parts.isEmpty else {
            throw malformed("\(flag) is empty — name at least one layer", name)
        }
        var layers = Set<Int>()
        for part in parts {
            let pieces = part.split(separator: "-", omittingEmptySubsequences: false)
            guard (1...2).contains(pieces.count), pieces.allSatisfy({ isDigits($0, max: 4) }),
                let low = Int(pieces[0]), let high = Int(pieces[pieces.count - 1])
            else {
                throw malformed(
                    "\(flag) expects comma-separated layer indices or ranges such as "
                        + "20-24 — got '\(part)'", name)
            }
            guard low <= high else {
                throw malformed(
                    "\(flag) range '\(part)' runs backwards — write the lower layer first",
                    name)
            }
            layers.formUnion(low...high)
        }
        return layers.sorted()
    }

    /// Comma-separated token ids, in the order given. `""` is no watchlist.
    public static func parseWatchlist(_ text: String, name: String) throws -> [Int] {
        let parts = text.split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        var ids: [Int] = []
        for part in parts {
            guard isDigits(Substring(part), max: 10), let id = Int(part), id <= maxTokenID else {
                throw malformed("--watchlist expects comma-separated token ids — got '\(part)'", name)
            }
            guard !ids.contains(id) else {
                throw malformed(
                    "--watchlist names token \(id) twice — each watched token is one "
                        + "column of the trace", name)
            }
            ids.append(id)
        }
        guard ids.count <= maxWatchlist else {
            throw malformed(
                "--watchlist names \(ids.count) tokens; a readout watches at most \(maxWatchlist)",
                name)
        }
        return ids
    }

    public static func parseWhole(
        _ text: String, flag: String, low: Int, high: Int, name: String
    ) throws -> Int {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard isDigits(Substring(trimmed), max: 16), let value = Int(trimmed),
            (low...high).contains(value)
        else {
            throw malformed("\(flag) must be a whole number from \(low) to \(high) — got '\(text)'", name)
        }
        return value
    }

    public static func parseSHA256(_ text: String, flag: String, name: String) throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard trimmed.count == 64, trimmed.allSatisfy({ "0123456789abcdef".contains($0) }) else {
            throw malformed(
                "\(flag) must be a SHA-256 digest: 64 lower-case hexadecimal characters "
                    + "— got '\(text)'", name)
        }
        return trimmed
    }

    /// `[0, 1, 2, 5]` → `"0-2,5"`: the spelling the layer flags accept.
    public static func compactLayers(_ layers: [Int]) -> String {
        let ordered = Array(Set(layers)).sorted()
        var out: [String] = []
        var index = 0
        while index < ordered.count {
            let start = ordered[index]
            var end = start
            while index + 1 < ordered.count, ordered[index + 1] == end + 1 {
                index += 1
                end = ordered[index]
            }
            out.append(start == end ? "\(start)" : "\(start)-\(end)")
            index += 1
        }
        return out.joined(separator: ",")
    }

    /// The rules between settings. Python twin: `check_settings`.
    public static func checkSettings(_ settings: Settings, name: String) throws {
        if let topKLayers = settings.topKLayers {
            guard settings.topK > 0 else {
                throw malformed(
                    "--top-k-layers chooses which layers carry the top-k readout, and no "
                        + "--top-k width is declared", name)
            }
            let outside = Set(topKLayers).subtracting(settings.layers).sorted()
            guard outside.isEmpty else {
                throw malformed(
                    "--top-k-layers names \(compactLayers(outside)), which --layers does "
                        + "not arm — top-k is read only at armed layers", name)
            }
        }
        if settings.watchlist.isEmpty, settings.topK <= 0 {
            throw malformed(
                "the readout declares neither a --watchlist nor a --top-k width, so it "
                    + "would record nothing", name)
        }
    }

    // MARK: - The block

    static func ints(_ values: [Int]) -> JSONValue { .array(values.map { .number(Double($0)) }) }

    /// The `jlensReadout` block exactly as both clients write it. Absent
    /// settings are absent keys, never null (the design identity hashes this
    /// block opaquely), and `configHash` is written only beside a
    /// `tokenizerHash`, because run start hashes the tokenizer it finds.
    public static func buildBlock(
        _ settings: Settings, lensID: String, lensSHA256: String
    ) -> [String: JSONValue] {
        var block: [String: JSONValue] = [
            "lensID": .string(lensID), "lensSHA256": .string(lensSHA256),
            "layers": ints(settings.layers), "topK": .number(Double(settings.topK)),
            "watchlist": ints(settings.watchlist),
            "logitLensCompanion": .bool(settings.companion),
        ]
        if let topKLayers = settings.topKLayers { block["topKLayers"] = ints(topKLayers) }
        if !settings.budget.isEmpty {
            block["budget"] = .object(settings.budget.mapValues { .number(Double($0)) })
        }
        if let qualification = settings.qualificationID { block["qualificationID"] = .string(qualification) }
        if let tokenizer = settings.tokenizerHash {
            block["tokenizerHash"] = .string(tokenizer)
            block["configHash"] = .string(configurationHash(block))
        }
        return block
    }

    static func intList(_ value: JSONValue?) -> [Int] {
        guard case .array(let values)? = value else { return [] }
        return values.compactMap { if case .number(let n) = $0 { return Int(n) }; return nil }
    }

    static func budget(of block: [String: JSONValue]) -> [String: Int] {
        var budget = defaultBudget
        if case .object(let declared)? = block["budget"] {
            for (key, value) in declared where budget[key] != nil {
                if case .number(let n) = value { budget[key] = Int(n) }
            }
        }
        return budget
    }

    /// `ReadoutConfig.armed_topk_layers`.
    static func armedTopKLayers(_ block: [String: JSONValue]) -> [Int] {
        let topK: Int = if case .number(let n)? = block["topK"] { Int(n) } else { 0 }
        guard topK > 0 else { return [] }
        let layers = intList(block["layers"])
        let chosen = intList(block["topKLayers"])
        return Set(chosen.isEmpty ? layers : chosen).intersection(layers).sorted()
    }

    /// The hash run start computes for this block — the engine's
    /// `trace.readout_config_payload`, restated, serialized the way Python's
    /// `json.dumps(sort_keys=True, separators=(",", ":"))` does.
    public static func configurationHash(_ block: [String: JSONValue]) -> String {
        var payload: [String: JSONValue] = [
            "layers": ints(Array(Set(intList(block["layers"]))).sorted()),
            "watchlist": ints(intList(block["watchlist"])),
            "topK": block["topK"] ?? .number(0),
            "topKLayers": ints(armedTopKLayers(block)),
            "logitLensCompanion": block["logitLensCompanion"] ?? .bool(true),
            "observationConvention": .string(observationConvention),
            "alignmentConvention": .string(alignmentConvention),
            "readoutConvention": .string(readoutConvention),
            "directionConvention": .string(directionConvention),
            "lensID": block["lensID"] ?? .null,
            "lensSHA256": block["lensSHA256"] ?? .null,
            "budget": .object(budget(of: block).mapValues { .number(Double($0)) }),
        ]
        if let tokenizer = block["tokenizerHash"] { payload["tokenizerHash"] = tokenizer }
        if let qualification = block["qualificationID"] { payload["qualificationID"] = qualification }
        return ManifestFileTransaction.digest(Data(pythonJSON(.object(payload)).utf8))
    }

    /// Python's `json.dumps(value, sort_keys=True, separators=(",", ":"))`
    /// with its default `ensure_ascii=True`, for the values a payload holds.
    static func pythonJSON(_ value: JSONValue) -> String {
        switch value {
        case .null: return "null"
        case .bool(let flag): return flag ? "true" : "false"
        case .number(let number):
            // Inclusive: 10^15 is the largest budget value both clients
            // accept, and Python writes it as an integer.
            return number.rounded() == number && abs(number) <= 1e15
                ? String(Int64(number)) : "\(number)"
        case .string(let text):
            var out = "\""
            for unit in text.utf16 {
                switch unit {
                case 0x22: out += "\\\""
                case 0x5C: out += "\\\\"
                case 0x0A: out += "\\n"
                case 0x0D: out += "\\r"
                case 0x09: out += "\\t"
                case 0x08: out += "\\b"
                case 0x0C: out += "\\f"
                case 0x20...0x7E: out += String(UnicodeScalar(UInt8(unit)))
                default: out += String(format: "\\u%04x", unit)
                }
            }
            return out + "\""
        case .array(let values):
            return "[" + values.map(pythonJSON).joined(separator: ",") + "]"
        case .object(let object):
            let keys = object.keys.sorted {
                $0.unicodeScalars.lexicographicallyPrecedes($1.unicodeScalars) { $0.value < $1.value }
            }
            return "{" + keys.map { pythonJSON(.string($0)) + ":" + pythonJSON(object[$0]!) }
                .joined(separator: ",") + "}"
        }
    }

    // MARK: - Budget (Python twin: estimate, the engine's `preflight`)

    /// ONE response priced at the token cap, as the engine's `preflight`
    /// prices it, and how many responses the declared budget covers.
    public static func estimate(_ block: [String: JSONValue], maxTokens: Int) -> [String: JSONValue] {
        let ceilings = budget(of: block)
        let steps = max(0, maxTokens)
        let layers = Set(intList(block["layers"])).count
        let watchlist = intList(block["watchlist"]).count
        let topK: Int = if case .number(let n)? = block["topK"] { Int(n) } else { 0 }
        let companion: Bool = if case .bool(let on)? = block["logitLensCompanion"] { on } else { true }
        let armed = armedTopKLayers(block).count
        let observations = steps * layers
        let fields = (watchlist + topK) * (companion ? 2 : 1)
        let projections = steps * armed * (companion ? 2 : 1)
        let bytes = observations * (120 + 24 * fields)
        let perResponse = [
            "observations": observations, "fullVocabProjections": projections,
            "projectedTraceBytes": bytes,
        ]
        func gigabytes(_ value: Int) -> String { String(format: "%.1f", Double(value) / 1e9) }
        var problems: [String] = []
        if layers > ceilings["maxArmedLayers"]! {
            problems.append("\(layers) armed layers exceeds \(ceilings["maxArmedLayers"]!)")
        }
        if watchlist > ceilings["maxWatchlist"]! {
            problems.append("watchlist of \(watchlist) exceeds \(ceilings["maxWatchlist"]!)")
        }
        if topK > ceilings["maxTopK"]! {
            problems.append("topK \(topK) exceeds \(ceilings["maxTopK"]!)")
        }
        if observations > ceilings["maxObservations"]! {
            problems.append(
                "\(observations) projected observations exceeds \(ceilings["maxObservations"]!)")
        }
        if bytes > ceilings["maxTraceBytes"]! {
            problems.append(
                "projected \(gigabytes(bytes)) GB of trace exceeds "
                    + "\(gigabytes(ceilings["maxTraceBytes"]!)) GB")
        }
        if projections > ceilings["maxFullVocabProjections"]! {
            problems.append(
                "\(projections) full-vocabulary projections exceeds "
                    + "\(ceilings["maxFullVocabProjections"]!) — this is the COMPUTE ceiling, "
                    + "and lowering topK does not reduce it: k selects from the result, it "
                    + "does not avoid the matmul. Arm fewer top-k layers, or use the "
                    + "watchlist, which needs no full projection at all")
        }
        var allowed: Int?
        var limitedBy: String?
        for (ceiling, quantity) in proportionalCeilings {
            let need = perResponse[quantity]!
            guard need > 0 else { continue }
            let fits = ceilings[ceiling]! / need
            if allowed == nil || fits < allowed! { allowed = fits; limitedBy = ceiling }
        }
        return [
            "maxTokens": .number(Double(maxTokens)),
            "armedLayers": .number(Double(layers)),
            "topKLayers": .number(Double(armed)),
            "perResponse": .object(perResponse.mapValues { .number(Double($0)) }),
            "budget": .object(ceilings.mapValues { .number(Double($0)) }),
            "responsesAllowed": allowed.map { .number(Double($0)) } ?? .null,
            "limitedBy": limitedBy.map { .string($0) } ?? .null,
            "problems": .array(problems.map { .string($0) }),
        ]
    }

    static func budgetRefusal(_ priced: [String: JSONValue], name: String) -> ExperimentError? {
        guard case .array(let problems)? = priced["problems"], !problems.isEmpty,
            case .number(let cap)? = priced["maxTokens"]
        else { return nil }
        let reasons = problems.compactMap { if case .string(let s) = $0 { return s }; return nil }
        return malformed(
            "the J-lens readout cannot fit its declared budget even for one response at "
                + "the \(Int(cap))-token cap: " + reasons.joined(separator: "; "), name,
            repair: "steerlab-cli experiment \(verb) \(name) <lens-id> … with fewer --layers "
                + "or --top-k-layers, or with a raised ceiling: "
                + budgetFlags.map(\.flag).joined(separator: ", "))
    }

    /// The canonical `steerlab-cli` spelling that writes this exact block.
    public static func macCommand(
        name: String, block: [String: JSONValue], pins: Pins, recordTokenIDs: Bool
    ) -> String {
        func text(_ key: String) -> String? {
            if case .string(let value)? = block[key] { return value }
            return nil
        }
        var parts = [
            "steerlab-cli", "experiment", verb, name, pins.lensID,
            "--lens-sha256", pins.lensSHA256, "--lens-model", pins.lensModel,
            "--fitted-layers", compactLayers(pins.fittedLayers),
            "--layers", compactLayers(intList(block["layers"])),
        ]
        if case .number(let k)? = block["topK"], k > 0 { parts += ["--top-k", "\(Int(k))"] }
        if block["topKLayers"] != nil {
            parts += ["--top-k-layers", compactLayers(intList(block["topKLayers"]))]
        }
        let watched = intList(block["watchlist"])
        if !watched.isEmpty { parts += ["--watchlist", watched.map(String.init).joined(separator: ",")] }
        if case .bool(false)? = block["logitLensCompanion"] { parts.append("--no-companion") }
        if let qualification = text("qualificationID") { parts += ["--qualification", qualification] }
        if let tokenizer = text("tokenizerHash") { parts += ["--tokenizer-hash", tokenizer] }
        if case .object(let declared)? = block["budget"] {
            for (flag, field) in budgetFlags {
                if case .number(let n)? = declared[field] { parts += [flag, "\(Int(n))"] }
            }
        }
        if recordTokenIDs { parts.append("--keep-tokens") }
        return parts.joined(separator: " ")
    }

    // MARK: - The verb

    /// Python twin: `unfitted_reason`.
    static func unfittedReason(_ unfitted: [Int], lensID: String, fitted: [Int]) -> String {
        let one = unfitted.count == 1
        return (one ? "layer " : "layers ") + compactLayers(unfitted)
            + (one ? " is not a fitted source layer" : " are not fitted source layers")
            + " of '\(lensID)' (it fitted \(compactLayers(fitted)))"
    }

    /// The typed lens pins must describe a lens for this study: fitted on its
    /// model (and revision, when the lens records one), at every armed layer.
    static func checkPins(
        _ pins: Pins, settings: Settings, manifest: ExperimentManifest, name: String
    ) throws {
        let model = manifest.modelID
        let split = pins.lensModel.split(separator: "@", maxSplits: 1).map(String.init)
        let fittedOn = split.first ?? ""
        let fittedRevision = split.count == 2 ? split[1] : nil
        guard fittedOn == model else {
            throw ExperimentError.refusing(
                .artifactPin,
                "J-lens '\(pins.lensID)' was fitted on '\(fittedOn)', and this study "
                    + "measures '\(model)' — a lens reads only the model it was fitted on",
                repair: "steerlab-server jlens list  (find a lens fitted on \(model)), then "
                    + "steerlab-cli experiment \(verb) \(name) <that-lens-id> …")
        }
        if let fittedRevision, !fittedRevision.isEmpty,
            let revision = manifest.modelRevision, !revision.isEmpty, fittedRevision != revision
        {
            throw ExperimentError.refusing(
                .artifactPin,
                "J-lens '\(pins.lensID)' was fitted on \(model)@\(fittedRevision), and this "
                    + "study pins revision \(revision)",
                repair: "steerlab-cli experiment pin-revision \(name) \(fittedRevision), or "
                    + "name a lens fitted on revision \(revision)")
        }
        let unfitted = Set(settings.layers).subtracting(pins.fittedLayers).sorted()
        guard unfitted.isEmpty else {
            throw malformed(
                unfittedReason(unfitted, lensID: pins.lensID, fitted: pins.fittedLayers),
                name,
                repair: "steerlab-cli experiment \(verb) \(name) \(pins.lensID) --layers <a "
                    + "subset of \(compactLayers(pins.fittedLayers))> …")
        }
    }

    /// What the researcher should know before running. Never a refusal.
    static func notes(manifest: ExperimentManifest, block: [String: JSONValue]) -> [String] {
        var notes: [String] = []
        if block["tokenizerHash"] == nil {
            notes.append(
                "No --tokenizer-hash was given, so tokenizerHash and configHash are not "
                    + "pinned. The study can run as exploratory, and freeze refuses until "
                    + "both are pinned. Take the hash from steerlab experiment \(verb) … "
                    + "--dry-run on the machine that runs the study, and run this verb "
                    + "again.")
        }
        if block["qualificationID"] == nil {
            notes.append(
                "No qualification is pinned, so a run is exploratory and freeze refuses. "
                    + "This command line never resolves one: name it with --qualification "
                    + "<id>, whose id the Python client's --dry-run of this verb prints "
                    + "(result.pins.qualificationID), or run the Python client's verb, "
                    + "which pins the newest passing one for the study's declared "
                    + "revision and dtype.")
        }
        if (manifest.modelRevision ?? "").isEmpty || (manifest.dtype ?? "").isEmpty {
            notes.append(
                "The study pins no model revision or dtype, and a qualification is keyed "
                    + "by both: pin them before freezing.")
        }
        if !manifest.recordTokenIDs {
            notes.append(
                "recordTokenIDs is off, so the exact sampled tokens are not kept and the "
                    + "run cannot be replayed exactly later. Add --keep-tokens to keep "
                    + "them.")
        }
        return notes
    }

    static func price(
        _ manifest: inout ExperimentManifest, settings: Settings, pins: Pins, name: String
    ) throws -> Report {
        try checkPins(pins, settings: settings, manifest: manifest, name: name)
        let block = buildBlock(settings, lensID: pins.lensID, lensSHA256: pins.lensSHA256)
        let priced = estimate(block, maxTokens: manifest.maxTokens)
        if let refusal = budgetRefusal(priced, name: name) { throw refusal }
        manifest.jlensReadout = .object(block)
        if settings.recordTokenIDs { manifest.recordTokenIDs = true }
        return Report(
            manifest: manifest, block: block, estimate: priced,
            macCommand: macCommand(
                name: name, block: block, pins: pins, recordTokenIDs: settings.recordTokenIDs),
            notes: notes(manifest: manifest, block: block))
    }

    /// Validate, price, and (unless `dryRun`) write the readout on a draft.
    public static func declare(
        name: String, settings: Settings, pins: Pins, dryRun: Bool = false
    ) throws -> Report {
        guard dryRun else {
            var report: Report?
            try ExperimentStore.updateDraft(name: name) { manifest in
                report = try price(&manifest, settings: settings, pins: pins, name: name)
            }
            return report!
        }
        var manifest = try ExperimentStore.load(name: name)
        try ManifestMutationPolicy.admitDraftEdit(manifest)
        return try price(&manifest, settings: settings, pins: pins, name: name)
    }

    // MARK: - The command line

    /// Every setting flag. A clear takes none.
    static let settingFlags = [
        "--lens-sha256", "--lens-model", "--fitted-layers", "--layers", "--top-k",
        "--top-k-layers", "--watchlist", "--qualification", "--tokenizer-hash",
        "--max-armed-layers", "--max-observations", "--max-trace-bytes",
        "--max-full-vocab-projections", "--no-companion", "--keep-tokens",
    ]

    /// `experiment set-jlens-readout`, from the strict parser's `args` (the
    /// verb, its positionals, and its flags in the order typed). Python twin:
    /// `client_cli._set_jlens_readout`, with the same flat result keys.
    static func run(_ args: [String], sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let valueFlags = ExperimentCLIParser.spec(namespace: "experiment", verb: verb)?
            .valueFlags ?? []
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
        guard positionals.count == 2 else {
            throw ExperimentError.malformed(
                "usage: experiment \(verb) <name> <lens-id> [flags]",
                repair: usage(positionals.first ?? "<name>"))
        }
        let name = positionals[0]
        let lensID = positionals[1].trimmingCharacters(in: .whitespaces)
        let dryRun = switches.contains("--dry-run")

        if lensID.isEmpty {
            let typed = settingFlags.filter { values[$0] != nil || switches.contains($0) }
            guard typed.isEmpty else {
                throw malformed(
                    "\"\" clears the J-lens readout, and \(typed.joined(separator: ", ")) "
                        + "declares one — a clear takes no settings", name,
                    repair: "steerlab-cli experiment \(verb) \(name) \"\"  (clears it), or "
                        + usage(name))
            }
            let existed = try clear(name: name, dryRun: dryRun)
            let line =
                existed
                ? (dryRun
                    ? "(dry run, nothing written) would clear the J-lens readout on '\(name)'"
                    : "cleared the J-lens readout on '\(name)'")
                : "'\(name)' declares no J-lens readout — nothing to clear"
            sink.out(line)
            return ExperimentCLIResult(
                message: line, changed: existed && !dryRun,
                payload: [
                    "experiment": .string(name), "lensID": .null, "jlensReadout": .null,
                    "written": .bool(existed && !dryRun), "pinsComplete": .bool(false),
                    "estimate": .null, "pins": .null, "macCommand": .null, "notes": .array([]),
                ])
        }

        // Shape first, in the Python client's order, so the same mistake is
        // refused with the same sentence on either client.
        guard let layersText = values["--layers"] else {
            throw malformed("a J-lens readout needs --layers: the source layers it reads", name)
        }
        var settings = Settings(layers: try parseLayers(layersText, flag: "--layers", name: name))
        if let text = values["--top-k"] {
            settings.topK = try parseWhole(text, flag: "--top-k", low: 0, high: maxTopK, name: name)
        }
        if let text = values["--top-k-layers"] {
            settings.topKLayers = try parseLayers(text, flag: "--top-k-layers", name: name)
        }
        settings.watchlist = try parseWatchlist(values["--watchlist"] ?? "", name: name)
        settings.companion = !switches.contains("--no-companion")
        for (flag, field) in budgetFlags {
            if let text = values[flag] {
                settings.budget[field] = try parseWhole(
                    text, flag: flag, low: 1, high: maxBudgetValue, name: name)
            }
        }
        let qualification = (values["--qualification"] ?? "").trimmingCharacters(in: .whitespaces)
        settings.qualificationID = qualification.isEmpty ? nil : qualification
        if let text = values["--tokenizer-hash"] {
            settings.tokenizerHash = try parseSHA256(text, flag: "--tokenizer-hash", name: name)
        }
        settings.recordTokenIDs = switches.contains("--keep-tokens")
        try checkSettings(settings, name: name)

        // The lens pins: typed here, read from the lens record on the client.
        let missing = ["--lens-sha256", "--lens-model", "--fitted-layers"].filter {
            (values[$0] ?? "").trimmingCharacters(in: .whitespaces).isEmpty
        }
        guard missing.isEmpty else {
            throw malformed(
                "this engine never reads a lens record (J-lens work is server-only), so "
                    + "the lens's pins are flags, and \(missing.joined(separator: ", ")) "
                    + "\(missing.count == 1 ? "is" : "are") missing", name)
        }
        let pins = Pins(
            lensID: lensID,
            lensSHA256: try parseSHA256(values["--lens-sha256"]!, flag: "--lens-sha256", name: name),
            lensModel: values["--lens-model"]!.trimmingCharacters(in: .whitespaces),
            fittedLayers: try parseLayers(values["--fitted-layers"]!, flag: "--fitted-layers", name: name))

        let report = try declare(name: name, settings: settings, pins: pins, dryRun: dryRun)
        let block = report.block
        let priced = report.estimate
        func number(_ value: JSONValue?) -> Int? {
            if case .number(let n)? = value { return Int(n) }
            return nil
        }
        var pieces = ["lens \(lensID)", "layers \(compactLayers(settings.layers))"]
        if settings.topK > 0 {
            pieces.append("top-\(settings.topK) at \(number(priced["topKLayers"]) ?? 0) layer(s)")
        }
        if !settings.watchlist.isEmpty { pieces.append("\(settings.watchlist.count) watched token(s)") }
        pieces.append("logit-lens companion " + (settings.companion ? "on" : "off"))
        let line =
            (dryRun ? "(dry run, nothing written) would declare" : "declared")
            + " a J-lens readout on '\(name)': " + pieces.joined(separator: ", ")
        let perResponse: [String: JSONValue] =
            if case .object(let per)? = priced["perResponse"] { per } else { [:] }
        let grouped: (Int) -> String = { value in
            let formatter = NumberFormatter()
            formatter.numberStyle = .decimal
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.usesGroupingSeparator = true
            formatter.groupingSeparator = ","
            formatter.groupingSize = 3
            return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
        }
        let bytes = Double(number(perResponse["projectedTraceBytes"]) ?? 0)
        var cost =
            "priced at the \(number(priced["maxTokens"]) ?? 0)-token cap, one response costs "
            + "\(grouped(number(perResponse["observations"]) ?? 0)) observations, "
            + "\(grouped(number(perResponse["fullVocabProjections"]) ?? 0)) "
            + "full-vocabulary projections, and about "
            + String(format: "%.1f", bytes / 1e6) + " MB of trace; "
        if let allowed = number(priced["responsesAllowed"]),
            case .string(let ceiling)? = priced["limitedBy"]
        {
            cost +=
                "the declared budget covers \(grouped(allowed)) response(s), limited by "
                + "\(ceilingWords[ceiling] ?? ceiling). Run start refuses a study with more "
                + "responses than that (conditions × items × samples)"
        } else {
            cost += "the declared budget does not bound the number of responses"
        }
        sink.out(line)
        sink.out(cost)
        for note in report.notes { sink.err("note: \(note)") }
        let pinPayload: [String: JSONValue] = [
            "lensID": .string(pins.lensID), "lensSHA256": .string(pins.lensSHA256),
            "lensModel": .string(pins.lensModel),
            "fittedLayers": .string(compactLayers(pins.fittedLayers)),
            "tokenizerHash": block["tokenizerHash"] ?? .null,
            "qualificationID": block["qualificationID"] ?? .null,
        ]
        return ExperimentCLIResult(
            message: line, changed: !dryRun,
            payload: [
                "experiment": .string(name),
                "lensID": .string(lensID),
                // The block AS WRITTEN (or as it would be, on a dry run).
                "jlensReadout": .object(block),
                "written": .bool(!dryRun),
                "pinsComplete": .bool(block["configHash"] != nil),
                "estimate": .object(priced),
                "pins": .object(pinPayload),
                "macCommand": .string(report.macCommand),
                "notes": .array(report.notes.map { .string($0) }),
            ])
    }

    /// Remove the readout. `recordTokenIDs` stays as it is: keeping tokens is
    /// harmless, and it is a separate setting.
    public static func clear(name: String, dryRun: Bool = false) throws -> Bool {
        let current = try ExperimentStore.load(name: name)
        try ManifestMutationPolicy.admitDraftEdit(current)
        // Nothing to remove is not a write: the manifest's bytes stay as they are.
        guard !dryRun, current.jlensReadout != nil else { return current.jlensReadout != nil }
        var existed = false
        try ExperimentStore.updateDraft(name: name) { manifest in
            existed = manifest.jlensReadout != nil
            manifest.jlensReadout = nil
        }
        return existed
    }
}

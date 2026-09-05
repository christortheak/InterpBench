import CryptoKit
import Foundation
import Synchronization

/// Chat-template capabilities DERIVED from the pinned template, recorded per
/// model (2026-09-05). Server twin: `experiment/model_capabilities.py`.
///
/// Until this type, every family rule the renderers branched on was a
/// substring test on the model id: `"qwen"` meant "has a thinking mode and
/// reads `reasoning_effort`", `"gemma"` meant "no system role". Two of those
/// were false in ways a frozen manifest could not see — Qwen/Qwen3-14B's
/// template reads `enable_thinking` but IGNORES `reasoning_effort`, so a
/// study declaring `medium` ran at the template's default while its manifest
/// asserted medium. The capabilities are now PROBED: the engine's own
/// renderer is asked, once per (model id, revision), what the template
/// actually does, and the answer is a hashed JSON record in the workspace
/// (`prompts/models/<owner>--<repo>@<revision>.json`) that every gate,
/// renderer, run stamp and preregistration reads.
///
/// This is the EFFECTIVE view — the probe's `detected` facts with the human
/// `overrides` applied — plus the provenance every consumer stamps or
/// displays. ``Record`` is the on-disk / wire form, which keeps the two
/// blocks apart; ``ModelCapabilitiesProbe`` derives a record from a
/// renderer; ``ModelCapabilitiesStore`` reads, writes and ensures records in
/// a workspace.
///
/// THE FALLBACK. With no tokenizer to probe (a gate before the model was
/// ever installed, a test double) the old id heuristics still answer, as a
/// record whose ``source`` is `heuristic` and whose ``advisories`` say so —
/// never as something a reader could mistake for a probed fact.
public struct ModelCapabilities: Sendable, Equatable {

    public static let schemaVersion = 1
    /// Where a workspace keeps its records — under `prompts/` because a
    /// record is recipe-side data (git-tracked, hashed, shared by both
    /// engines), not a run output.
    public static let directory = "prompts/models"
    /// The reasoning-effort levels the probe tries, in the fixed cross-engine
    /// order. Server twin: `model_capabilities.EFFORT_CANDIDATES`.
    public static let effortCandidates = ["low", "medium", "high", "xhigh"]
    /// The value no template accepts, used to settle whether
    /// `reasoning_effort` is read at all before any real candidate is judged.
    public static let effortProbeValue = "steerlab-probe-effort"
    /// Probe texts: single ASCII words no template transforms.
    public static let probeUserText = "steerlab-probe-user"
    public static let probeSystemText = "steerlab-probe-system"
    public static let thinkOpenToken = "<think>"
    public static let thinkCloseToken = "</think>"

    public enum SystemRole: String, Sendable, CaseIterable {
        case systemTurn, foldedIntoUser, unsupported
    }

    public enum ThinkingSwitch: String, Sendable, CaseIterable {
        case supported, unsupported
    }

    public enum EffortVerdict: String, Sendable, CaseIterable {
        case accepted, rejected, ignored
        /// A heuristic record's verdict on a level it never probed but the
        /// old id rule assumed — allowed by the gates with an advisory.
        case assumed
    }

    public enum Source: String, Sendable {
        case probe, heuristic
    }

    /// One human override: the value, why, and when. Displayed beside the
    /// detected value and stamped into runs — never silent.
    public struct Override: Sendable, Equatable {
        public var value: OverrideValue
        public var reason: String
        public var setAt: String

        public init(value: OverrideValue, reason: String, setAt: String) {
            self.value = value
            self.reason = reason
            self.setAt = setAt
        }
    }

    public enum OverrideValue: Sendable, Equatable {
        case string(String)
        case boolean(Bool)

        public var any: Any {
            switch self {
            case .string(let text): text
            case .boolean(let flag): flag
            }
        }

        public var description: String {
            switch self {
            case .string(let text): text
            case .boolean(let flag): flag ? "True" : "False"
            }
        }
    }

    /// The fields a human may override, each with the closed vocabulary its
    /// value must come from (nil = a boolean).
    public static let overridableFields: [String: [String]?] = [
        "systemRole": SystemRole.allCases.map(\.rawValue),
        "thinkingSwitch": ThinkingSwitch.allCases.map(\.rawValue),
        "thinkOpenInPrompt": nil,
    ]

    public struct ThinkTokens: Sendable, Equatable {
        public var open: Int?
        public var close: Int?
        public init(open: Int?, close: Int?) {
            self.open = open
            self.close = close
        }
    }

    public struct Architecture: Sendable, Equatable {
        public var layerCount: Int?
        public var hiddenSize: Int?
        public var layerTypes: [String]?
        public init(layerCount: Int?, hiddenSize: Int?, layerTypes: [String]?) {
            self.layerCount = layerCount
            self.hiddenSize = hiddenSize
            self.layerTypes = layerTypes
        }
    }

    /// What the probe found. Never hand-edited: a human's disagreement is an
    /// override, displayed beside it.
    public struct Detected: Sendable, Equatable {
        public var systemRole: SystemRole
        public var systemRoleDetail: String?
        public var foldSeparator: String?
        public var thinkingSwitch: ThinkingSwitch
        public var thinkOpenInPrompt: Bool?
        public var effortVariableRead: Bool?
        public var effortLevels: [String: EffortVerdict]?
        public var thinkTokens: ThinkTokens
        public var architecture: Architecture?

        public init(
            systemRole: SystemRole, systemRoleDetail: String? = nil,
            foldSeparator: String? = nil, thinkingSwitch: ThinkingSwitch,
            thinkOpenInPrompt: Bool? = nil, effortVariableRead: Bool? = nil,
            effortLevels: [String: EffortVerdict]? = nil,
            thinkTokens: ThinkTokens = .init(open: nil, close: nil),
            architecture: Architecture? = nil
        ) {
            self.systemRole = systemRole
            self.systemRoleDetail = systemRoleDetail
            self.foldSeparator = foldSeparator
            self.thinkingSwitch = thinkingSwitch
            self.thinkOpenInPrompt = thinkOpenInPrompt
            self.effortVariableRead = effortVariableRead
            self.effortLevels = effortLevels
            self.thinkTokens = thinkTokens
            self.architecture = architecture
        }

        public var jsonObject: [String: Any] {
            var levels: Any = NSNull()
            if let effortLevels {
                levels = effortLevels.mapValues { $0.rawValue }
            }
            var architectureObject: Any = NSNull()
            if let architecture {
                architectureObject = [
                    "layerCount": architecture.layerCount ?? NSNull(),
                    "hiddenSize": architecture.hiddenSize ?? NSNull(),
                    "layerTypes": architecture.layerTypes ?? NSNull(),
                ] as [String: Any]
            }
            return [
                "systemRole": systemRole.rawValue,
                "systemRoleDetail": systemRoleDetail ?? NSNull(),
                "foldSeparator": foldSeparator ?? NSNull(),
                "thinkingSwitch": thinkingSwitch.rawValue,
                "thinkOpenInPrompt": thinkOpenInPrompt ?? NSNull(),
                "effortVariableRead": effortVariableRead ?? NSNull(),
                "effortLevels": levels,
                "thinkTokens": [
                    "open": thinkTokens.open ?? NSNull(),
                    "close": thinkTokens.close ?? NSNull(),
                ] as [String: Any],
                "architecture": architectureObject,
            ]
        }

        public static func parse(_ object: [String: Any]) throws -> Detected {
            guard let roleText = object["systemRole"] as? String,
                let role = SystemRole(rawValue: roleText)
            else {
                throw RecordError("detected.systemRole must be one of "
                    + SystemRole.allCases.map(\.rawValue).joined(separator: ", "))
            }
            guard let switchText = object["thinkingSwitch"] as? String,
                let thinkingSwitch = ThinkingSwitch(rawValue: switchText)
            else {
                throw RecordError("detected.thinkingSwitch must be supported or unsupported")
            }
            var levels: [String: EffortVerdict]?
            if let raw = object["effortLevels"], !(raw is NSNull) {
                guard let map = raw as? [String: String] else {
                    throw RecordError("detected.effortLevels must be an object or null")
                }
                var parsed: [String: EffortVerdict] = [:]
                for (key, value) in map {
                    guard ModelCapabilities.effortCandidates.contains(key),
                        let verdict = EffortVerdict(rawValue: value)
                    else {
                        throw RecordError(
                            "detected.effortLevels.\(key)=\(value) is not a probe "
                                + "candidate with a known verdict")
                    }
                    parsed[key] = verdict
                }
                levels = parsed
            }
            let tokens = object["thinkTokens"] as? [String: Any] ?? [:]
            var architecture: Architecture?
            if let block = object["architecture"] as? [String: Any] {
                architecture = Architecture(
                    layerCount: block["layerCount"] as? Int,
                    hiddenSize: block["hiddenSize"] as? Int,
                    layerTypes: block["layerTypes"] as? [String])
            }
            return Detected(
                systemRole: role,
                systemRoleDetail: object["systemRoleDetail"] as? String,
                foldSeparator: object["foldSeparator"] as? String,
                thinkingSwitch: thinkingSwitch,
                thinkOpenInPrompt: object["thinkOpenInPrompt"] as? Bool,
                effortVariableRead: object["effortVariableRead"] as? Bool,
                effortLevels: levels,
                thinkTokens: ThinkTokens(
                    open: tokens["open"] as? Int, close: tokens["close"] as? Int),
                architecture: architecture)
        }
    }

    public struct ProbedBy: Sendable, Equatable {
        public var engine: String
        public var version: String
        public var at: String
        public init(engine: String, version: String, at: String) {
            self.engine = engine
            self.version = version
            self.at = at
        }
    }

    public struct TemplateHashes: Sendable, Equatable {
        public var sha256: String?
        public var tokenizerConfigSha256: String?
        public init(sha256: String?, tokenizerConfigSha256: String?) {
            self.sha256 = sha256
            self.tokenizerConfigSha256 = tokenizerConfigSha256
        }
    }

    /// A malformed record, or a verb argument the record cannot take.
    public struct RecordError: Error, CustomStringConvertible {
        public let description: String
        public init(_ description: String) { self.description = description }
    }

    // MARK: - The record (on-disk / wire form)

    /// The record as the JSON schema pins it
    /// (`Tests/Fixtures/cross-engine/model-capabilities.schema.json`).
    /// `recordHash` is sha256 over the canonical JSON of every other key.
    public struct Record: Sendable, Equatable {
        public var modelID: String
        public var revision: String?
        public var source: Source
        public var probedBy: ProbedBy?
        public var template: TemplateHashes?
        public var detected: Detected
        public var overrides: [String: Override]
        public var advisories: [String]

        public init(
            modelID: String, revision: String?, source: Source,
            probedBy: ProbedBy?, template: TemplateHashes?, detected: Detected,
            overrides: [String: Override] = [:], advisories: [String] = []
        ) {
            self.modelID = modelID
            self.revision = revision
            self.source = source
            self.probedBy = probedBy
            self.template = template
            self.detected = detected
            self.overrides = overrides
            self.advisories = advisories
        }

        /// Everything but `recordHash`.
        public var canonicalPayload: [String: Any] {
            var probed: Any = NSNull()
            if let probedBy {
                probed = ["engine": probedBy.engine, "version": probedBy.version,
                          "at": probedBy.at] as [String: Any]
            }
            var templateObject: Any = NSNull()
            if let template {
                templateObject = [
                    "sha256": template.sha256 ?? NSNull(),
                    "tokenizerConfigSha256": template.tokenizerConfigSha256 ?? NSNull(),
                ] as [String: Any]
            }
            var overrideObject: [String: Any] = [:]
            for (key, override) in overrides {
                overrideObject[key] = [
                    "value": override.value.any,
                    "reason": override.reason,
                    "setAt": override.setAt,
                ] as [String: Any]
            }
            return [
                "schemaVersion": ModelCapabilities.schemaVersion,
                "modelID": modelID,
                "revision": revision ?? NSNull(),
                "source": source.rawValue,
                "probedBy": probed,
                "template": templateObject,
                "detected": detected.jsonObject,
                "overrides": overrideObject,
                "advisories": advisories,
            ]
        }

        /// sha256 over the canonical JSON of everything but `recordHash`.
        /// Server twin: `model_capabilities.record_hash`.
        public var recordHash: String {
            ModelCapabilities.sha256Hex(CanonicalJSON.encode(canonicalPayload))
        }

        /// The full JSON object, hash included.
        public var jsonObject: [String: Any] {
            var object = canonicalPayload
            object["recordHash"] = recordHash
            return object
        }

        /// Parse a record object, refusing a malformed one by name — the same
        /// checks the pinned schema states. Server twin:
        /// `model_capabilities.validate_record` + `read_record`.
        public static func parse(_ object: [String: Any]) throws -> Record {
            guard (object["schemaVersion"] as? Int) == ModelCapabilities.schemaVersion else {
                throw RecordError("schemaVersion must be \(ModelCapabilities.schemaVersion)")
            }
            guard let modelID = object["modelID"] as? String, !modelID.isEmpty else {
                throw RecordError("modelID must be a non-empty string")
            }
            guard let sourceText = object["source"] as? String,
                let source = Source(rawValue: sourceText)
            else { throw RecordError("source must be probe or heuristic") }
            guard let detectedObject = object["detected"] as? [String: Any] else {
                throw RecordError("detected must be an object")
            }
            let detected = try Detected.parse(detectedObject)
            var probedBy: ProbedBy?
            if let block = object["probedBy"] as? [String: Any] {
                probedBy = ProbedBy(
                    engine: block["engine"] as? String ?? "",
                    version: block["version"] as? String ?? "",
                    at: block["at"] as? String ?? "")
            }
            var template: TemplateHashes?
            if let block = object["template"] as? [String: Any] {
                template = TemplateHashes(
                    sha256: block["sha256"] as? String,
                    tokenizerConfigSha256: block["tokenizerConfigSha256"] as? String)
            }
            var overrides: [String: Override] = [:]
            for (key, raw) in object["overrides"] as? [String: Any] ?? [:] {
                guard let vocabulary = ModelCapabilities.overridableFields[key] else {
                    throw RecordError("overrides.\(key) is not an overridable field")
                }
                guard let entry = raw as? [String: Any],
                    let reason = entry["reason"] as? String, !reason.isEmpty
                else { throw RecordError("overrides.\(key) needs a value and a reason") }
                let value: OverrideValue
                if let allowed = vocabulary {
                    guard let text = entry["value"] as? String, allowed.contains(text) else {
                        throw RecordError(
                            "overrides.\(key).value must be one of "
                                + allowed.joined(separator: ", "))
                    }
                    value = .string(text)
                } else {
                    guard let number = entry["value"] as? NSNumber,
                        CFGetTypeID(number) == CFBooleanGetTypeID()
                    else { throw RecordError("overrides.\(key).value must be a boolean") }
                    value = .boolean(number.boolValue)
                }
                overrides[key] = Override(
                    value: value, reason: reason, setAt: entry["setAt"] as? String ?? "")
            }
            let record = Record(
                modelID: modelID, revision: object["revision"] as? String,
                source: source, probedBy: probedBy, template: template,
                detected: detected, overrides: overrides,
                advisories: object["advisories"] as? [String] ?? [])
            if let stored = object["recordHash"] as? String, stored != record.recordHash {
                throw RecordError("recordHash does not match the record's canonical bytes")
            }
            return record
        }

        /// The effective view.
        public func effective(path: String? = nil) -> ModelCapabilities {
            var detected = self.detected
            for (key, override) in overrides {
                switch (key, override.value) {
                case ("systemRole", .string(let text)):
                    detected.systemRole = SystemRole(rawValue: text) ?? detected.systemRole
                case ("thinkingSwitch", .string(let text)):
                    detected.thinkingSwitch =
                        ThinkingSwitch(rawValue: text) ?? detected.thinkingSwitch
                case ("thinkOpenInPrompt", .boolean(let flag)):
                    detected.thinkOpenInPrompt = flag
                default:
                    break
                }
            }
            return ModelCapabilities(
                modelID: modelID, revision: revision, source: source,
                systemRole: detected.systemRole, foldSeparator: detected.foldSeparator,
                thinkingSwitch: detected.thinkingSwitch,
                thinkOpenInPrompt: detected.thinkOpenInPrompt,
                effortVariableRead: detected.effortVariableRead,
                effortLevels: detected.effortLevels, thinkTokens: detected.thinkTokens,
                architecture: detected.architecture,
                templateSha256: template?.sha256,
                tokenizerConfigSha256: template?.tokenizerConfigSha256,
                overrides: overrides, advisories: advisories,
                recordHash: recordHash, path: path)
        }
    }

    // MARK: - The effective view

    public let modelID: String
    public let revision: String?
    public let source: Source
    public let systemRole: SystemRole
    public let foldSeparator: String?
    public let thinkingSwitch: ThinkingSwitch
    public let thinkOpenInPrompt: Bool?
    public let effortVariableRead: Bool?
    public let effortLevels: [String: EffortVerdict]?
    public let thinkTokens: ThinkTokens
    public let architecture: Architecture?
    public let templateSha256: String?
    public let tokenizerConfigSha256: String?
    public let overrides: [String: Override]
    public let advisories: [String]
    public let recordHash: String?
    /// Workspace-relative record path when the view came from a file.
    public let path: String?

    public init(
        modelID: String, revision: String?, source: Source, systemRole: SystemRole,
        foldSeparator: String?, thinkingSwitch: ThinkingSwitch, thinkOpenInPrompt: Bool?,
        effortVariableRead: Bool?, effortLevels: [String: EffortVerdict]?,
        thinkTokens: ThinkTokens, architecture: Architecture?, templateSha256: String?,
        tokenizerConfigSha256: String?, overrides: [String: Override] = [:],
        advisories: [String] = [], recordHash: String? = nil, path: String? = nil
    ) {
        self.modelID = modelID
        self.revision = revision
        self.source = source
        self.systemRole = systemRole
        self.foldSeparator = foldSeparator
        self.thinkingSwitch = thinkingSwitch
        self.thinkOpenInPrompt = thinkOpenInPrompt
        self.effortVariableRead = effortVariableRead
        self.effortLevels = effortLevels
        self.thinkTokens = thinkTokens
        self.architecture = architecture
        self.templateSha256 = templateSha256
        self.tokenizerConfigSha256 = tokenizerConfigSha256
        self.overrides = overrides
        self.advisories = advisories
        self.recordHash = recordHash
        self.path = path
    }

    public var isProbed: Bool { source == .probe }
    public var hasSystemRole: Bool { systemRole == .systemTurn }
    public var systemPromptDeliverable: Bool { systemRole != .unsupported }
    public var hasThinkingSwitch: Bool { thinkingSwitch == .supported }

    public var acceptedEfforts: [String] {
        ModelCapabilities.effortCandidates.filter { effortLevels?[$0] == .accepted }
    }

    /// The verdict on one level, or nil when this record never judged it.
    public func effortVerdict(_ effort: String) -> EffortVerdict? {
        effortLevels?[effort]
    }

    public func withAdvisory(_ note: String) -> ModelCapabilities {
        ModelCapabilities(
            modelID: modelID, revision: revision, source: source, systemRole: systemRole,
            foldSeparator: foldSeparator, thinkingSwitch: thinkingSwitch,
            thinkOpenInPrompt: thinkOpenInPrompt, effortVariableRead: effortVariableRead,
            effortLevels: effortLevels, thinkTokens: thinkTokens, architecture: architecture,
            templateSha256: templateSha256, tokenizerConfigSha256: tokenizerConfigSha256,
            overrides: overrides, advisories: advisories + [note], recordHash: recordHash,
            path: path)
    }

    /// The compact provenance a run's `config.json` carries under
    /// `notes.modelCapabilities`. Server twin: `Capabilities.stamp`.
    public var stamp: [String: Any] {
        var levels: Any = NSNull()
        if let effortLevels { levels = effortLevels.mapValues { $0.rawValue } }
        return [
            "source": source.rawValue,
            "record": path ?? NSNull(),
            "recordHash": recordHash ?? NSNull(),
            "templateSha256": templateSha256 ?? NSNull(),
            "systemRole": systemRole.rawValue,
            "thinkingSwitch": thinkingSwitch.rawValue,
            "thinkOpenInPrompt": thinkOpenInPrompt ?? NSNull(),
            "effortVariableRead": effortVariableRead ?? NSNull(),
            "effortLevels": levels,
            "overrides": overrides.mapValues { $0.value.any },
        ]
    }

    /// The preregistration's account: one line per fact, in the words both
    /// engines print. Server twin: `Capabilities.summary_lines`.
    public var summaryLines: [String] {
        let levels = effortLevels ?? [:]
        let effortText = ModelCapabilities.effortCandidates
            .compactMap { level in levels[level].map { "\(level) \($0.rawValue)" } }
            .joined(separator: ", ")
        let overrideText = overrides.keys.sorted().map { key in
            let override = overrides[key]!
            return "\(key)=\(override.value.description) (\(override.reason))"
        }.joined(separator: ", ")
        var first = "- **Model capabilities:** source \(source.rawValue)"
        if let path { first += ", record `\(path)`" }
        if let recordHash { first += " (hash `\(recordHash)`)" }
        var role = "- **System role:** \(systemRole.rawValue)"
        if systemRole == .foldedIntoUser {
            role += " (separator \(CanonicalJSON.pythonRepr(foldSeparator ?? "")))"
        }
        var thinking = "- **Thinking switch:** \(thinkingSwitch.rawValue)"
        if thinkingSwitch == .supported, let open = thinkOpenInPrompt {
            thinking += ", opening tag in prompt \(open ? "true" : "false")"
        }
        return [
            first, role, thinking,
            "- **Reasoning effort levels:** " + (effortText.isEmpty ? "none" : effortText),
            "- **Capability overrides:** " + (overrideText.isEmpty ? "none" : overrideText),
        ]
    }

    // MARK: - The heuristic fallback

    public static func heuristicAdvisory(modelID: String) -> String {
        "model capabilities for \(modelID) derive from the model id, not from its "
            + "chat template — no probed record under \(directory)/; the declaration "
            + "was gated on the old family rule (qwen → thinking switch + effort "
            + "levels assumed, gemma → system text folded into the first user "
            + "turn). Probe the pinned template: steerlab-server model capabilities "
            + "<modelID> --probe, or on the Mac steerlab-cli model capabilities "
            + "<modelID> --probe"
    }

    /// Today's id heuristics, as a record that SAYS it is one — the
    /// pre-record rules exactly, so a consumer with no tokenizer behaves as
    /// it always did. Server twin: `model_capabilities.heuristic`.
    public static func heuristic(modelID: String, revision: String? = nil) -> Record {
        let lowered = modelID.lowercased()
        let isQwen = lowered.contains("qwen")
        let isGemma = lowered.contains("gemma")
        return Record(
            modelID: modelID, revision: revision, source: .heuristic,
            probedBy: nil, template: nil,
            detected: Detected(
                systemRole: isGemma ? .foldedIntoUser : .systemTurn,
                systemRoleDetail: "assumed from the model id",
                foldSeparator: isGemma ? "\n\n" : nil,
                thinkingSwitch: isQwen ? .supported : .unsupported,
                thinkOpenInPrompt: nil, effortVariableRead: nil,
                effortLevels: isQwen
                    ? ["low": .assumed, "medium": .assumed, "xhigh": .assumed] : nil,
                thinkTokens: .init(open: nil, close: nil), architecture: nil),
            overrides: [:], advisories: [heuristicAdvisory(modelID: modelID)])
    }

    public static func heuristicView(modelID: String, revision: String? = nil) -> ModelCapabilities {
        heuristic(modelID: modelID, revision: revision).effective()
    }

    // MARK: - Hashing

    public static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The hash of the template SOURCE the engine rendered with. Server twin:
    /// `model_capabilities.template_sha256`.
    public static func templateSha256(_ source: String) -> String { sha256Hex(source) }

    // MARK: - The per-process registry

    /// Capabilities registered for a loaded model, keyed by model id — what
    /// `PromptRendering` reads when a caller passes none, so a render can
    /// never disagree with the record the load ensured. Registered by
    /// `ExperimentTasks.loadContainer` / the Playground load; absent means
    /// the heuristic (which says so).
    private static let registry = Mutex<[String: ModelCapabilities]>([:])

    public static func register(_ capabilities: ModelCapabilities, for modelID: String) {
        registry.withLock { $0[modelID] = capabilities }
    }

    public static func registered(for modelID: String) -> ModelCapabilities? {
        registry.withLock { $0[modelID] }
    }

    public static func forgetRegistered() {
        registry.withLock { $0.removeAll() }
    }
}

// MARK: - Canonical JSON

/// The canonical text both engines hash: keys sorted by UTF-8 bytes (which is
/// code-point order for the record's ASCII keys — NOT Foundation's
/// `.sortedKeys`, whose comparison is not byte order), compact separators,
/// raw UTF-8, Python's minimal escaping. Server twin:
/// `model_capabilities.canonical_json` (`json.dumps(sort_keys=True,
/// separators=(",", ":"), ensure_ascii=False)`).
public enum CanonicalJSON {

    public static func encode(_ value: Any) -> String {
        var out = ""
        append(value, to: &out)
        return out
    }

    private static func append(_ value: Any, to out: inout String) {
        switch value {
        case is NSNull:
            out += "null"
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                out += number.boolValue ? "true" : "false"
            } else if let int = value as? Int {
                out += String(int)
            } else {
                out += "\(number.doubleValue)"
            }
        case let text as String:
            appendString(text, to: &out)
        case let array as [Any]:
            out += "["
            for (index, element) in array.enumerated() {
                if index > 0 { out += "," }
                append(element, to: &out)
            }
            out += "]"
        case let object as [String: Any]:
            out += "{"
            let keys = object.keys.sorted { $0.utf8.lexicographicallyPrecedes($1.utf8) }
            for (index, key) in keys.enumerated() {
                if index > 0 { out += "," }
                appendString(key, to: &out)
                out += ":"
                append(object[key]!, to: &out)
            }
            out += "}"
        default:
            out += "null"
        }
    }

    private static func appendString(_ text: String, to out: inout String) {
        out += "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        out += "\""
    }

    /// Python's `repr` of a short string, for the one place a separator is
    /// displayed (`'\n\n'`): single quotes, the same escapes.
    public static func pythonRepr(_ text: String) -> String {
        var out = "'"
        for scalar in text.unicodeScalars {
            switch scalar {
            case "'": out += "\\'"
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 || scalar.value == 0x7f {
                    out += String(format: "\\x%02x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "'"
    }
}

// MARK: - The probe

/// Derive a record from a renderer. Server twin: `model_capabilities.probe`.
///
/// Each verdict is a render comparison, never a regex over the template
/// source: `systemRole` from `[user]` vs `[system, user]` (folded when the
/// second is the first with the system text prepended inside the user turn,
/// separator recorded; unsupported when the template raises on, or drops,
/// the turn); `thinkingSwitch` from `enable_thinking` true vs false;
/// `effortLevels` from thinking-on renders per candidate, after a bogus
/// value settles whether `reasoning_effort` is read at all (so a template
/// whose default equals a candidate is still `accepted`, not `ignored`).
public enum ModelCapabilitiesProbe {

    /// Render a chat WITH a generation prompt under extra template variables,
    /// throwing on a template refusal.
    public typealias Render = (_ messages: [[String: String]], _ context: [String: Any]) throws -> String

    public static func probe(
        render: Render, modelID: String, revision: String?,
        thinkTokenID: ((String) -> Int?)? = nil,
        architecture: ModelCapabilities.Architecture? = nil,
        templateSha256: String? = nil, tokenizerConfigSha256: String? = nil,
        engine: String = RepEReader.substrate, engineVersion: String,
        probedAt: String = ModelCapabilitiesProbe.now()
    ) throws -> ModelCapabilities.Record {
        let user = [["role": "user", "content": ModelCapabilities.probeUserText]]
        let withSystem = [["role": "system", "content": ModelCapabilities.probeSystemText]] + user

        func attempt(_ messages: [[String: String]], _ context: [String: Any] = [:]) -> Result<String, Error> {
            Result { try render(messages, context) }
        }

        // -- systemRole
        guard case .success(let plain) = attempt(user) else {
            throw ModelCapabilities.RecordError(
                "the chat template of \(modelID) refuses a plain user turn")
        }
        var systemRole: ModelCapabilities.SystemRole
        var detail: String?
        var foldSeparator: String?
        switch attempt(withSystem) {
        case .failure:
            // Engine-neutral on purpose: the two Jinja engines spell the
            // exception differently, and the detail is pinned cross-engine.
            systemRole = .unsupported
            detail = "the template raises on a system turn"
        case .success(let framed):
            guard let systemRange = framed.range(of: ModelCapabilities.probeSystemText) else {
                systemRole = .unsupported
                detail = "the template silently drops a system turn"
                break
            }
            let prefix = plain.range(of: ModelCapabilities.probeUserText)
                .map { String(plain[..<$0.lowerBound]) } ?? ""
            let userRange = framed.range(of: ModelCapabilities.probeUserText)
            let folded = !prefix.isEmpty && framed.hasPrefix(prefix)
                && userRange.map { systemRange.lowerBound < $0.lowerBound } == true
            if folded, let userRange {
                let separator = String(framed[systemRange.upperBound ..< userRange.lowerBound])
                let hand = attempt([[
                    "role": "user",
                    "content": ModelCapabilities.probeSystemText + separator
                        + ModelCapabilities.probeUserText,
                ]])
                if case .success(let handRendered) = hand, handRendered == framed {
                    systemRole = .foldedIntoUser
                    foldSeparator = separator
                } else {
                    systemRole = .systemTurn
                    detail = "system text appears inside the user turn but not as a "
                        + "plain prefix; treated as a system turn"
                }
            } else {
                systemRole = .systemTurn
            }
        }

        // -- thinkingSwitch
        let on = attempt(user, ["enable_thinking": true])
        let off = attempt(user, ["enable_thinking": false])
        var thinkingSwitch = ModelCapabilities.ThinkingSwitch.unsupported
        var thinkOpenInPrompt: Bool?
        var onText: String?
        if case .success(let onRendered) = on, case .success(let offRendered) = off,
            onRendered != offRendered
        {
            thinkingSwitch = .supported
            onText = onRendered
            thinkOpenInPrompt = Self.trailingTrimmed(onRendered)
                .hasSuffix(ModelCapabilities.thinkOpenToken)
        }

        // -- effortLevels
        var effortVariableRead: Bool?
        var levels: [String: ModelCapabilities.EffortVerdict]?
        if thinkingSwitch == .supported, let onText {
            let bogus = attempt(user, [
                "enable_thinking": true, "reasoning_effort": ModelCapabilities.effortProbeValue,
            ])
            let read: Bool
            switch bogus {
            case .failure: read = true
            case .success(let rendered): read = rendered != onText
            }
            var verdicts: [String: ModelCapabilities.EffortVerdict] = [:]
            for candidate in ModelCapabilities.effortCandidates {
                guard read else {
                    verdicts[candidate] = .ignored
                    continue
                }
                switch attempt(user, ["enable_thinking": true, "reasoning_effort": candidate]) {
                case .failure: verdicts[candidate] = .rejected
                case .success: verdicts[candidate] = .accepted
                }
            }
            effortVariableRead = read
            levels = verdicts
        }

        let detected = ModelCapabilities.Detected(
            systemRole: systemRole, systemRoleDetail: detail, foldSeparator: foldSeparator,
            thinkingSwitch: thinkingSwitch, thinkOpenInPrompt: thinkOpenInPrompt,
            effortVariableRead: effortVariableRead, effortLevels: levels,
            thinkTokens: .init(
                open: thinkTokenID?(ModelCapabilities.thinkOpenToken),
                close: thinkTokenID?(ModelCapabilities.thinkCloseToken)),
            architecture: architecture)
        return ModelCapabilities.Record(
            modelID: modelID, revision: revision, source: .probe,
            probedBy: .init(engine: engine, version: engineVersion, at: probedAt),
            template: .init(sha256: templateSha256, tokenizerConfigSha256: tokenizerConfigSha256),
            detected: detected)
    }

    static func trailingTrimmed(_ text: String) -> String {
        var scalars = Substring(text)
        while let last = scalars.last, last.isWhitespace { scalars = scalars.dropLast() }
        return String(scalars)
    }

    static func message(of error: Error) -> String {
        String(describing: error)
    }

    public static func now() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(from: Date())
    }
}

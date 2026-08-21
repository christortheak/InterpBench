import CryptoKit
import Foundation

public struct MultiAgentScenario: Codable, Sendable, Equatable {
    public struct Agent: Codable, Identifiable, Sendable, Equatable {
        public var id: String
        public var name: String
        public var baseModelID: String
        public var variantArtifactPath: String?
        public var variantArtifactHash: String?
        public var systemPrompt: String
        /// A noun phrase describing the SEAT — "a judge of the United States
        /// Court of Appeals" — with no trailing period. Read only by the
        /// contract renderer's identity opener; a template turn never sees it.
        /// Absent or blank ⇒ omitted from the opener and from the file, so a
        /// panel authored before contracts re-encodes byte for byte.
        public var role: String?

        public init(
            id: String = UUID().uuidString,
            name: String,
            baseModelID: String,
            variantArtifactPath: String? = nil,
            variantArtifactHash: String? = nil,
            systemPrompt: String = "",
            role: String? = nil
        ) {
            self.id = id
            self.name = name
            self.baseModelID = baseModelID
            self.variantArtifactPath = variantArtifactPath
            self.variantArtifactHash = variantArtifactHash
            self.systemPrompt = systemPrompt
            self.role = Self.normalizedRole(role)
        }

        /// Spelled out because this type hand-writes both halves of Codable,
        /// which turns off the synthesis that would otherwise supply it.
        private enum CodingKeys: String, CodingKey {
            case id, name, baseModelID, variantArtifactPath, variantArtifactHash
            case systemPrompt, role
        }

        /// Blank is the same as absent, normalized once on the way in and once
        /// on the way out, so `role: ""` never round-trips into a file and
        /// never makes two otherwise-equal seats compare unequal.
        static func normalizedRole(_ role: String?) -> String? {
            guard let role,
                !role.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return nil }
            return role
        }

        /// Mirrors the synthesized decode key for key — same required keys,
        /// same optionality — and adds only the blank-role normalization.
        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.id = try container.decode(String.self, forKey: .id)
            self.name = try container.decode(String.self, forKey: .name)
            self.baseModelID = try container.decode(String.self, forKey: .baseModelID)
            self.variantArtifactPath =
                try container.decodeIfPresent(String.self, forKey: .variantArtifactPath)
            self.variantArtifactHash =
                try container.decodeIfPresent(String.self, forKey: .variantArtifactHash)
            self.systemPrompt = try container.decode(String.self, forKey: .systemPrompt)
            self.role = Self.normalizedRole(
                try container.decodeIfPresent(String.self, forKey: .role))
        }

        /// Hand-written for one reason: `role` must be OMITTED, not written as
        /// `""` or `null`, when the seat has none. A panel file is a
        /// hash-pinned study input, so a key that appears out of nowhere on a
        /// no-op re-save is freeze drift.
        public func encode(to encoder: any Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(id, forKey: .id)
            try container.encode(name, forKey: .name)
            try container.encode(baseModelID, forKey: .baseModelID)
            try container.encodeIfPresent(variantArtifactPath, forKey: .variantArtifactPath)
            try container.encodeIfPresent(variantArtifactHash, forKey: .variantArtifactHash)
            try container.encode(systemPrompt, forKey: .systemPrompt)
            try container.encodeIfPresent(Self.normalizedRole(role), forKey: .role)
        }
    }

    public struct Turn: Codable, Identifiable, Sendable, Equatable {
        public enum Routing: String, Codable, CaseIterable, Sendable {
            case all
            case speakerOnly
            case selected
            case none

            public var label: String {
                switch self {
                case .all: "All agents"
                case .speakerOnly: "Speaker only"
                case .selected: "Selected agents"
                case .none: "No agent context"
                }
            }
        }

        public var id: String
        public var title: String
        public var speakerAgentID: String
        public var promptTemplate: String
        public var outputLabel: String
        public var routing: Routing
        public var routedAgentIDs: [String]
        public var includeScenarioMaterials: Bool
        public var includeSpeakerContext: Bool
        public var maxTokens: Int?
        /// The quantity this turn is supposed to produce (Wave-2 contract),
        /// parsed from the generated text at write time and stamped on the
        /// turn record. Additive and optional: nil ⇒ key omitted, so panels
        /// authored before the contract encode byte for byte as before. A
        /// malformed declaration refuses on DECODE — see `TurnEndpoint`.
        public var endpoint: TurnEndpoint?
        /// The structured prompt declaration (panel turn contracts,
        /// 2026-08-17). Present ⇒ this turn renders through the canonical
        /// sandwich and `promptTemplate` must be empty; absent ⇒ the turn is
        /// exactly the free-text template turn it always was. Optional and
        /// omitted when nil, so contract-free panels encode byte for byte as
        /// before.
        public var contract: TurnContract?
        /// Output labels this turn reads ON PURPOSE across routing
        /// (spec §4.1), which silences the private-read advisory for exactly
        /// those labels and leaves it live for every undeclared one.
        ///
        /// A blind-round design reads its colleagues' private round-1 memos
        /// by name in every later round — precisely what the advisory looks
        /// for — so with no way to SAY SO it fired 18 times per scenario by
        /// design and stopped being read at all.
        ///
        /// ADVISORY-ONLY: it changes no rendered byte, no validate error, no
        /// run behavior, and NOT the derived `schemaVersion` — an older build
        /// that ignores the key loses nothing measurable, it just prints
        /// advisories a newer build suppresses. Nil and empty are the same
        /// thing, normalized to nil (the `role` precedent) so the key is
        /// OMITTED rather than written as `[]`: a key that appears out of
        /// nowhere on a no-op re-save is freeze drift on a pinned panel.
        public var acknowledgedInputs: [String]? {
            didSet {
                acknowledgedInputs = Self.normalizedAcknowledgedInputs(acknowledgedInputs)
            }
        }

        /// Empty is the same as absent. Applied at init and at decode; the
        /// `didSet` above covers later assignment, which is what lets the
        /// ENCODE stay synthesized (an Optional is emitted with
        /// `encodeIfPresent`, so nil is an absent key) rather than this type
        /// hand-writing a thirteen-key encoder that a future property could
        /// silently fall out of.
        static func normalizedAcknowledgedInputs(_ labels: [String]?) -> [String]? {
            guard let labels, !labels.isEmpty else { return nil }
            return labels
        }

        public init(
            id: String = UUID().uuidString,
            title: String,
            speakerAgentID: String,
            promptTemplate: String,
            outputLabel: String = "",
            routing: Routing = .all,
            routedAgentIDs: [String] = [],
            includeScenarioMaterials: Bool = true,
            includeSpeakerContext: Bool = true,
            maxTokens: Int? = nil,
            endpoint: TurnEndpoint? = nil,
            contract: TurnContract? = nil,
            acknowledgedInputs: [String]? = nil
        ) {
            self.id = id
            self.title = title
            self.speakerAgentID = speakerAgentID
            self.promptTemplate = promptTemplate
            self.outputLabel = outputLabel
            self.routing = routing
            self.routedAgentIDs = routedAgentIDs
            self.includeScenarioMaterials = includeScenarioMaterials
            self.includeSpeakerContext = includeSpeakerContext
            self.maxTokens = maxTokens
            self.endpoint = endpoint
            self.contract = contract
            self.acknowledgedInputs = Self.normalizedAcknowledgedInputs(acknowledgedInputs)
        }

        /// Mirrors the synthesized decode key for key, with one relaxation:
        /// `promptTemplate` defaults to empty. Writers emit
        /// `"promptTemplate": ""` on a contract turn for decode compatibility
        /// with older builds, but a file that simply omits the key is a
        /// contract turn too and must not be rejected as malformed.
        ///
        /// Encoding stays synthesized — `contract` and `maxTokens` are
        /// Optionals, which the compiler emits with `encodeIfPresent`, so nil
        /// is an ABSENT key rather than a JSON null.
        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.id = try container.decode(String.self, forKey: .id)
            self.title = try container.decode(String.self, forKey: .title)
            self.speakerAgentID = try container.decode(String.self, forKey: .speakerAgentID)
            self.promptTemplate =
                try container.decodeIfPresent(String.self, forKey: .promptTemplate) ?? ""
            self.outputLabel = try container.decode(String.self, forKey: .outputLabel)
            self.routing = try container.decode(Routing.self, forKey: .routing)
            self.routedAgentIDs = try container.decode([String].self, forKey: .routedAgentIDs)
            self.includeScenarioMaterials =
                try container.decode(Bool.self, forKey: .includeScenarioMaterials)
            self.includeSpeakerContext =
                try container.decode(Bool.self, forKey: .includeSpeakerContext)
            self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
            self.endpoint = try container.decodeIfPresent(TurnEndpoint.self, forKey: .endpoint)
            self.contract = try container.decodeIfPresent(TurnContract.self, forKey: .contract)
            // Same leniency floor as `contract`: absent or null is "none
            // declared", while a value of the wrong shape throws — an
            // acknowledgment list that is not a list of strings is not a
            // half-written declaration, it is a file that does not mean what
            // it says. The Python twin refuses the same shapes.
            self.acknowledgedInputs = Self.normalizedAcknowledgedInputs(
                try container.decodeIfPresent([String].self, forKey: .acknowledgedInputs))
        }
    }

    public var schemaVersion: Int
    public var name: String
    public var description: String
    public var baseModelID: String
    public var sharedMaterials: String
    public var agents: [Agent]
    public var turns: [Turn]
    public var temperature: Double
    public var maxTokens: Int

    public init(
        name: String,
        description: String = "",
        baseModelID: String,
        sharedMaterials: String = "",
        agents: [Agent] = [],
        turns: [Turn] = [],
        temperature: Double = 0,
        maxTokens: Int = 2048
    ) {
        self.schemaVersion = Self.schemaVersion(for: turns)
        self.name = name
        self.description = description
        self.baseModelID = baseModelID
        self.sharedMaterials = sharedMaterials
        self.agents = agents
        self.turns = turns
        self.temperature = temperature
        self.maxTokens = maxTokens
    }

    /// The highest version this build WRITES. Decoders accept 1 and 2.
    public static let currentSchemaVersion = 2

    /// The version a given turn script has to be written at: 2 once any turn
    /// carries a `contract`, 1 otherwise.
    ///
    /// Derived rather than latched, so adding a contract turn bumps the file
    /// and removing the last one puts it back — a panel that never used
    /// contracts keeps saying `1` forever, and its bytes (and therefore its
    /// pinned hash) never move because this build learned a new key.
    public static func schemaVersion(for turns: [Turn]) -> Int {
        turns.contains { $0.contract != nil } ? 2 : 1
    }

    /// This scenario's required version — what the store stamps on save.
    public var requiredSchemaVersion: Int { Self.schemaVersion(for: turns) }

    /// The one key that says "this file is not a panel". See
    /// `ScenarioProtocolTemplate`.
    enum PanelMarkerKeys: String, CodingKey {
        case kind
    }

    /// Tolerant decode (B2): `schemaVersion` defaults when absent, so a
    /// scenario written by the Python engine — which did not emit it — opens
    /// here instead of being silently dropped by `scan`'s `try?`.
    ///
    /// `createdAt`/`updatedAt` were REMOVED from the format rather than made
    /// optional. They are volatile metadata on a content-hashed pinned input:
    /// re-saving with no semantic edit bumped `updatedAt`, changed the bytes,
    /// and produced a hash-drift verify violation on a frozen study out of
    /// nothing. Git holds that history; the file holds the recipe. Any such
    /// keys in an older file are ignored on read and not written back.
    public init(from decoder: any Decoder) throws {
        // A PROTOCOL TEMPLATE is deliberately byte-compatible with a panel —
        // it IS a panel whose `sharedMaterials` is empty by design, plus two
        // extra keys — so the tolerant decode below would accept one happily
        // and hand back a runnable-looking scenario with no case record in it.
        // Every seat would then deliberate about nothing and the transcript
        // would read as a real run rather than as a mistake. Refusing on the
        // marker covers every Swift decode path in one place, and mirrors the
        // same refusal in the Python engine's `Scenario.from_dict`.
        let marker = try decoder.container(keyedBy: PanelMarkerKeys.self)
        if let kind = try marker.decodeIfPresent(String.self, forKey: .kind),
            kind == ScenarioProtocolTemplate.kindMarker
        {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "this file is a scenario PROTOCOL TEMPLATE "
                        + "(\"kind\": \"\(kind)\"), not a runnable panel — it "
                        + "carries no shared materials on purpose; instantiate "
                        + "it into a scenario first"))
        }
        try self.init(bodyFrom: decoder)
    }

    /// The decode itself, with the template-marker check left out.
    ///
    /// Internal rather than private so `ScenarioProtocolTemplate` — whose file
    /// IS a panel body plus the marker — reuses this one decoder instead of
    /// keeping a second, drift-prone copy of the same key handling.
    init(bodyFrom decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try container.decode(String.self, forKey: .name)
        self.description =
            try container.decodeIfPresent(String.self, forKey: .description) ?? ""
        self.baseModelID =
            try container.decodeIfPresent(String.self, forKey: .baseModelID) ?? ""
        self.sharedMaterials =
            try container.decodeIfPresent(String.self, forKey: .sharedMaterials) ?? ""
        self.agents = try container.decodeIfPresent([Agent].self, forKey: .agents) ?? []
        self.turns = try container.decodeIfPresent([Turn].self, forKey: .turns) ?? []
        self.temperature =
            try container.decodeIfPresent(Double.self, forKey: .temperature) ?? 0
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 2048
        // Absent means an engine that did not emit it (the Python dialect once
        // did not), so the version is read off the turn script rather than
        // assumed to be whatever this build happens to write today.
        self.schemaVersion =
            try container.decodeIfPresent(Int.self, forKey: .schemaVersion)
            ?? Self.schemaVersion(for: self.turns)
    }
}

public struct MultiAgentScenarioRecord: Identifiable, Sendable {
    public let url: URL
    public let scenario: MultiAgentScenario

    public init(url: URL, scenario: MultiAgentScenario) {
        self.url = url
        self.scenario = scenario
    }

    public var id: String { url.path }
    public var label: String {
        "\(scenario.name) · \(scenario.agents.count) agents · \(scenario.turns.count) turns"
    }
}

/// A file in the panel library that could not be read as what it claims to be.
///
/// It exists because the alternative was worse: the scanner used to be a
/// `compactMap` over `try?`, so a panel with one bad key — a semantic scenario
/// missing the non-optional `Agent.baseModelID`, say — simply was not in the
/// picker. Not greyed out, not flagged: absent. The researcher's report is
/// "my panel disappeared", the file is fine on disk, and nothing anywhere says
/// which key failed. A blocker the user cannot see is the one bug class this
/// project refuses to ship, so the scan carries its failures out with it and
/// the picker renders them.
public struct PanelFileIssue: Identifiable, Sendable, Equatable {
    public let url: URL
    /// Plain-language decode failure: what was wrong, and where in the file.
    public let reason: String

    public init(url: URL, reason: String) {
        self.url = url
        self.reason = reason
    }

    public var id: String { url.path }
    public var fileName: String { url.lastPathComponent }
    /// One line fit for a picker row.
    public var label: String { "\(fileName) — unreadable" }
    /// The full detail line, file name included.
    public var detail: String { "\(fileName): \(reason)" }
}

public enum MultiAgentScenarioStore {
    /// The canonical location (B1, 2026-07-27): `prompts/panels/<slug>.json`,
    /// one flat file per panel, in the WORKSPACE — the same tree the Python
    /// engine reads and writes.
    ///
    /// A panel script is ex-ante input, hash-pinned like stimuli and task
    /// prompts, so it belongs under `prompts/` with them. The old home,
    /// `runs/multi-agent-scenarios/`, was doubly wrong: `runs/` is meant to be
    /// immutable while these are edited in place, and `runs/` is GITIGNORED —
    /// so the freeze cleanliness gate ("every pinned input must be committed")
    /// structurally could not see a study's most important input, and the
    /// `experiments/<name>/pinned/` snapshot floor was covering for it. Under
    /// `prompts/` the primary gate does its own job.
    ///
    /// Rooted at the workspace, not `VectorCatalog.projectRoot`, so the test
    /// root override applies here as it does everywhere else.
    public static var directory: URL {
        ExperimentStore.workspaceRoot.appending(components: "prompts", "panels")
    }

    /// Pre-B1 home. Read-only: already-frozen manifests pin paths in here and
    /// must keep resolving, but nothing new is ever written to it.
    public static var legacyDirectory: URL {
        ExperimentStore.workspaceRoot.appending(
            components: "runs", "multi-agent-scenarios")
    }

    /// Everything a scan of the panel library found: the panels, and the files
    /// that claim to be panels and are not readable as one.
    public struct Scan: Sendable {
        public let records: [MultiAgentScenarioRecord]
        public let unreadable: [PanelFileIssue]

        public init(records: [MultiAgentScenarioRecord], unreadable: [PanelFileIssue]) {
            self.records = records
            self.unreadable = unreadable
        }
    }

    /// What one `.json` file in the panel library turned out to be.
    enum Classification {
        case scenario(MultiAgentScenario)
        /// A protocol template. NOT an error and NOT a panel: it belongs to
        /// the template picker, so the scenario scan passes over it in
        /// silence rather than listing it as broken.
        case protocolTemplate
        case unreadable(String)
    }

    public static func scan(
        directory root: URL = directory,
        legacyDirectory legacyRoot: URL? = nil
    ) -> [MultiAgentScenarioRecord] {
        scanAll(directory: root, legacyDirectory: legacyRoot).records
    }

    /// The scan the picker uses: panels AND the files it could not read.
    public static func scanAll(
        directory root: URL = directory,
        legacyDirectory legacyRoot: URL? = nil
    ) -> Scan {
        let legacy = legacyRoot ?? (root == directory ? legacyDirectory : nil)
        var (records, unreadable) = sift(jsonFiles(in: root))
        if let legacy {
            let (legacyRecords, legacyUnreadable) = sift(legacyFiles(in: legacy))
            records += legacyRecords
            unreadable += legacyUnreadable
        }
        // Ordered by file mtime — recently edited first. The scenario used to
        // carry its own `updatedAt` and sort on it; that field is gone (B2),
        // because a timestamp inside a content-hashed pinned input turns a
        // no-op save into hash drift. The filesystem already knows.
        records.sort {
            let a = modificationDate($0.url)
            let b = modificationDate($1.url)
            if a == b { return $0.url.path < $1.url.path }
            return a > b
        }
        // Broken rows sort by name: they are a work list, not a history.
        unreadable.sort { $0.url.path < $1.url.path }
        return Scan(records: records, unreadable: unreadable)
    }

    static func classify(_ url: URL) -> Classification {
        guard let data = try? Data(contentsOf: url) else {
            return .unreadable("could not be read from disk")
        }
        if ScenarioProtocolTemplate.isTemplateFile(data) { return .protocolTemplate }
        do {
            return .scenario(try JSONDecoder().decode(MultiAgentScenario.self, from: data))
        } catch {
            return .unreadable(decodeFailureReason(error))
        }
    }

    /// A decode failure rendered as a sentence a researcher can act on.
    ///
    /// `DecodingError`'s own description is a debug dump; the thing that
    /// matters — WHICH key, WHERE — is buried in `codingPath`. This pulls it
    /// out, because the whole point of surfacing a broken file is that the row
    /// tells you what to fix.
    static func decodeFailureReason(_ error: any Error) -> String {
        func location(_ context: DecodingError.Context) -> String {
            let path = context.codingPath
                .map { $0.intValue.map { index in "[\(index)]" } ?? ".\($0.stringValue)" }
                .joined()
                .drop(while: { $0 == "." })
            return path.isEmpty ? "the top level of the file" : String(path)
        }
        switch error {
        case let DecodingError.keyNotFound(key, context):
            return "missing required key '\(key.stringValue)' at \(location(context))"
        case let DecodingError.typeMismatch(type, context):
            return "wrong type at \(location(context)): expected \(type)"
        case let DecodingError.valueNotFound(type, context):
            return "null where a \(type) is required, at \(location(context))"
        case let DecodingError.dataCorrupted(context):
            let where_ = context.codingPath.isEmpty ? "" : " at \(location(context))"
            return "\(context.debugDescription)\(where_)"
        default:
            return "\(error)"
        }
    }

    /// One classification pass over a file list. One read per file: the
    /// picker's rows and its broken rows are two views of the same answer, and
    /// deriving them from two separate scans is how they come to disagree.
    private static func sift(
        _ urls: [URL]
    ) -> (records: [MultiAgentScenarioRecord], unreadable: [PanelFileIssue]) {
        var records: [MultiAgentScenarioRecord] = []
        var unreadable: [PanelFileIssue] = []
        for url in urls {
            // A legacy directory with no `scenario.json` in it is an empty
            // husk, not a broken panel — it was never claiming to be one.
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            switch classify(url) {
            case .scenario(let scenario):
                records.append(MultiAgentScenarioRecord(url: url, scenario: scenario))
            case .protocolTemplate:
                continue
            case .unreadable(let reason):
                unreadable.append(PanelFileIssue(url: url, reason: reason))
            }
        }
        return (records, unreadable)
    }

    private static func jsonFiles(in root: URL) -> [URL] {
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return [] }
        // Non-recursive on purpose: `prompts/panels/templates/` is the
        // protocol-template library and must not surface here at all.
        return entries.filter { $0.pathExtension == "json" }
    }

    /// Legacy layout was a directory per scenario holding `scenario.json`.
    private static func legacyFiles(in root: URL) -> [URL] {
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return [] }
        return entries.map { $0.appending(component: "scenario.json") }
    }

    private static func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey])
            .contentModificationDate) ?? .distantPast
    }

    @discardableResult
    public static func save(_ scenario: MultiAgentScenario) throws -> MultiAgentScenarioRecord {
        try update(scenario, at: try uniqueURL(for: scenario.name))
    }

    /// `prompts/panels/<slug>.json`, suffixed if taken. Never overwrites a
    /// different panel that happens to slugify the same way.
    ///
    /// Uses `ExperimentStore.canonicalSlug` — the one rule both engines share
    /// — rather than `FineTuneStore.slugify`, which passes `_` through. Both
    /// engines now write into this directory, so a naming disagreement would
    /// mean two files for one panel on the cluster and a silent overwrite on a
    /// case-insensitive Mac.
    private static func uniqueURL(for name: String) throws -> URL {
        let root = directory
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let base = ExperimentStore.canonicalSlug(name)
        var candidate = root.appending(component: "\(base).json")
        var suffix = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = root.appending(component: "\(base)-\(suffix).json")
            suffix += 1
        }
        return candidate
    }

    @discardableResult
    public static func update(
        _ scenario: MultiAgentScenario,
        at url: URL
    ) throws -> MultiAgentScenarioRecord {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        // Stamped at the one place that writes panel files, from the turn
        // script itself: a file claims schema 2 exactly when it needs 2. An
        // in-memory scenario whose version is stale — built in code, or
        // decoded from a file that omitted the key — is corrected here rather
        // than shipping a version that misdescribes its own contents.
        var stamped = scenario
        stamped.schemaVersion = scenario.requiredSchemaVersion
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(stamped).write(to: url, options: .atomic)
        return MultiAgentScenarioRecord(url: url, scenario: stamped)
    }

    public static func delete(_ record: MultiAgentScenarioRecord) throws {
        // Canonical records are one flat file; legacy ones own their enclosing
        // directory, so removing only the file would leave an empty husk that
        // `scan` keeps listing as an undecodable entry.
        if record.url.lastPathComponent == "scenario.json" {
            try FileManager.default.removeItem(at: record.url.deletingLastPathComponent())
        } else {
            try FileManager.default.removeItem(at: record.url)
        }
    }

    public static func hash(_ url: URL) throws -> String {
        hash(try Data(contentsOf: url))
    }

    /// Hash of bytes already in hand. A caller that also needs the bytes —
    /// the run path snapshots the scenario file verbatim — must not read the
    /// file a second time to hash it: two reads can straddle an edit and
    /// certify bytes that are not the ones that ran.
    public static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

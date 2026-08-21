import Foundation

/// A deliberation PROTOCOL with the case taken out.
///
/// The reuse problem it solves: a panel protocol — the seats, the turn script,
/// the routing, the per-turn token caps, the endpoint declarations — is the
/// part a researcher wants to hold FIXED across many cases, and the case
/// record is the part that changes every time. Before this, the only way to
/// reuse a protocol was to keep a scenario around with `sharedMaterials` blank
/// ("three-judge-panel") and remember never to run it. That is a landmine, not
/// a solution: it is indistinguishable from a real panel in every picker, it
/// runs when told to, and a run of it produces a full transcript of three
/// judges deliberating about an empty record.
///
/// So a protocol template is its own artifact, marked by `kind`, stored in its
/// own library (`prompts/panels/templates/`), listed in its own picker, and
/// REFUSED by the scenario decoder on both engines. It cannot be run, because
/// nothing will parse it as a panel.
///
/// The file is otherwise a panel, flat, key for key. That is deliberate: a
/// protocol template is meant to be read and hand-edited in a diff exactly
/// like the panels it mints, and a nested envelope would have made every turn
/// script two levels deeper for no gain. The safety comes from the marker, not
/// from the shape.
///
/// Instantiation produces an ORDINARY scenario file, written through the
/// ordinary `MultiAgentScenarioStore.save`. There is no template-aware run
/// path, no template-aware pinning, no template-aware freeze, and the Python
/// engine knows nothing about templates except how to refuse one. Templates
/// are an authoring convenience that leaves NO trace in the measurement path
/// — deliberately not even a provenance stamp on the minted scenario, because
/// a panel file is hash-pinned by studies and round-tripped by the Python
/// engine, which does not emit unknown keys: a provenance field would survive
/// on the Mac, vanish on the cluster, and turn a no-op re-save into hash
/// drift on somebody's frozen study.
public struct ScenarioProtocolTemplate: Sendable, Equatable {
    /// The value of the `kind` key that makes this file not-a-panel. Both
    /// engines compare against this exact string.
    public static let kindMarker = "scenarioProtocolTemplate"
    public static let currentTemplateSchemaVersion = 1

    /// Version of the TEMPLATE envelope — `kind`, `materialsChecklist`, and
    /// this key. Separate from the panel body's own `schemaVersion`, which
    /// keeps meaning exactly what it means in a panel file. One shared number
    /// would have made a bump to the envelope silently claim a new panel
    /// schema, and the minted scenario would carry it.
    public var templateSchemaVersion: Int

    /// What the eventual shared materials have to contain, in the author's own
    /// words — "the case record", "procedural posture", "the disposition scale
    /// with anchors". Advisory by construction: see `review`.
    public var materialsChecklist: [String]

    /// The protocol itself: a panel in every respect, with `sharedMaterials`
    /// empty by design.
    public var protocolBody: MultiAgentScenario

    public init(
        templateSchemaVersion: Int = ScenarioProtocolTemplate.currentTemplateSchemaVersion,
        materialsChecklist: [String] = [],
        protocolBody: MultiAgentScenario
    ) {
        self.templateSchemaVersion = templateSchemaVersion
        self.materialsChecklist = materialsChecklist
        var body = protocolBody
        body.sharedMaterials = ""
        self.protocolBody = body
    }

    public var name: String {
        get { protocolBody.name }
        set { protocolBody.name = newValue }
    }

    public var templateDescription: String {
        get { protocolBody.description }
        set { protocolBody.description = newValue }
    }

    public var seats: [MultiAgentScenario.Agent] { protocolBody.agents }
    public var turns: [MultiAgentScenario.Turn] { protocolBody.turns }

    /// A one-line summary for a picker row.
    public var label: String {
        "\(name) · \(protocolBody.agents.count) seats · \(protocolBody.turns.count) turns"
    }

    /// Cheap marker test on raw bytes, for scanners that must classify a file
    /// before deciding which decoder to hand it to.
    ///
    /// Deliberately not "decode it and see": a file whose panel body is
    /// malformed is still a TEMPLATE if it says so, and mistaking it for a
    /// broken panel would put it in the wrong picker's broken list.
    public static func isTemplateFile(_ data: Data) -> Bool {
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let kind = object["kind"] as? String
        else { return false }
        return kind == kindMarker
    }
}

// MARK: - Codable (flat: the marker keys and the panel body share one object)

extension ScenarioProtocolTemplate: Codable {
    private enum MarkerKeys: String, CodingKey {
        case kind
        case templateSchemaVersion
        case materialsChecklist
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: MarkerKeys.self)
        let kind = try container.decodeIfPresent(String.self, forKey: .kind) ?? ""
        guard kind == Self.kindMarker else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        kind.isEmpty
                        ? "not a scenario protocol template: no \"kind\" marker "
                            + "(a plain panel file belongs in prompts/panels/, "
                            + "not prompts/panels/templates/)"
                        : "unknown artifact kind \"\(kind)\" — expected "
                            + "\"\(Self.kindMarker)\""))
        }
        templateSchemaVersion =
            try container.decodeIfPresent(Int.self, forKey: .templateSchemaVersion)
            ?? Self.currentTemplateSchemaVersion
        materialsChecklist =
            try container.decodeIfPresent([String].self, forKey: .materialsChecklist) ?? []
        // The same panel decoder the panels use, minus the marker refusal it
        // would otherwise hit on its way in.
        var body = try MultiAgentScenario(bodyFrom: decoder)
        // A template's materials are empty BY DEFINITION, not by hope. If a
        // file carries some — hand-edited, or minted before this rule — they
        // are dropped here rather than smuggled into every scenario the
        // template ever mints.
        body.sharedMaterials = ""
        protocolBody = body
    }

    public func encode(to encoder: any Encoder) throws {
        try protocolBody.encode(to: encoder)
        var container = encoder.container(keyedBy: MarkerKeys.self)
        try container.encode(Self.kindMarker, forKey: .kind)
        try container.encode(templateSchemaVersion, forKey: .templateSchemaVersion)
        try container.encode(materialsChecklist, forKey: .materialsChecklist)
    }
}

public struct ScenarioProtocolTemplateRecord: Identifiable, Sendable {
    public let url: URL
    public let template: ScenarioProtocolTemplate

    public init(url: URL, template: ScenarioProtocolTemplate) {
        self.url = url
        self.template = template
    }

    public var id: String { url.path }
    public var label: String { template.label }
}

/// Listing, minting and instantiating protocol templates.
///
/// Authoring-side in its entirety. Nothing here is read by a run, a freeze, an
/// analysis, or the Python engine.
public enum ScenarioProtocolTemplateStore {

    // MARK: - Location

    /// `prompts/panels/templates/` — beside the panels, one level down.
    ///
    /// A subdirectory rather than a sibling tree because the two belong
    /// together in a diff and in a workspace listing, and because
    /// `MultiAgentScenarioStore`'s scan is non-recursive, so a directory here
    /// is invisible to it for free. The `kind` marker is still the load-bearing
    /// guard — location is convenience, not safety.
    public static var directory: URL {
        MultiAgentScenarioStore.directory.appending(component: "templates")
    }

    // MARK: - Listing

    public struct Scan: Sendable {
        public let records: [ScenarioProtocolTemplateRecord]
        public let unreadable: [PanelFileIssue]

        public init(
            records: [ScenarioProtocolTemplateRecord], unreadable: [PanelFileIssue]
        ) {
            self.records = records
            self.unreadable = unreadable
        }
    }

    public static func list(directory root: URL = directory) -> [ScenarioProtocolTemplateRecord] {
        scanAll(directory: root).records
    }

    /// Templates, plus the files that claim to be templates and will not
    /// decode. Same rule as the panel picker: a file never disappears.
    ///
    /// It scans the panel directory too, for `kind`-marked files that ended up
    /// there — dropped in by hand, or written by an older build. Those are
    /// excluded from the scenario picker by their marker, so listing them here
    /// is the difference between "it moved" and "it vanished".
    public static func scanAll(
        directory root: URL = directory,
        alsoScanning panelRoot: URL? = MultiAgentScenarioStore.directory
    ) -> Scan {
        var records: [ScenarioProtocolTemplateRecord] = []
        var unreadable: [PanelFileIssue] = []
        var seen = Set<String>()
        for base in [root, panelRoot].compactMap({ $0 }) {
            guard
                let entries = try? FileManager.default.contentsOfDirectory(
                    at: base, includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles])
            else { continue }
            for url in entries.filter({ $0.pathExtension == "json" }).sorted(by: {
                $0.lastPathComponent < $1.lastPathComponent
            }) {
                guard let data = try? Data(contentsOf: url) else { continue }
                // In the panel directory only marked files are ours; in the
                // template directory everything is, so an unmarked file there
                // is a mistake worth naming.
                let claimed = ScenarioProtocolTemplate.isTemplateFile(data)
                guard claimed || base == root else { continue }
                guard seen.insert(url.resolvingSymlinksInPath().path).inserted else { continue }
                do {
                    records.append(
                        ScenarioProtocolTemplateRecord(
                            url: url,
                            template: try JSONDecoder().decode(
                                ScenarioProtocolTemplate.self, from: data)))
                } catch {
                    unreadable.append(
                        PanelFileIssue(
                            url: url,
                            reason: MultiAgentScenarioStore.decodeFailureReason(error)))
                }
            }
        }
        records.sort { $0.template.name.lowercased() < $1.template.name.lowercased() }
        unreadable.sort { $0.url.path < $1.url.path }
        return Scan(records: records, unreadable: unreadable)
    }

    // MARK: - Persistence

    @discardableResult
    public static func save(
        _ template: ScenarioProtocolTemplate
    ) throws -> ScenarioProtocolTemplateRecord {
        let trimmed = template.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ExperimentError(reason: "a protocol template needs a name")
        }
        let slug = ExperimentStore.canonicalSlug(trimmed)
        guard !slug.isEmpty else {
            throw ExperimentError(
                reason: "'\(trimmed)' has no letters or digits to make a file "
                    + "name from")
        }
        var stored = template
        stored.name = trimmed
        let root = directory
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var url = root.appending(component: "\(slug).json")
        var suffix = 2
        // Same-name means "update"; a different name that slugifies alike gets
        // a suffix rather than clobbering its neighbour — the panel store's
        // rule, so the two libraries behave the same way under collision.
        while FileManager.default.fileExists(atPath: url.path),
            existingName(at: url) != stored.name
        {
            url = root.appending(component: "\(slug)-\(suffix).json")
            suffix += 1
        }
        return try update(stored, at: url)
    }

    @discardableResult
    public static func update(
        _ template: ScenarioProtocolTemplate, at url: URL
    ) throws -> ScenarioProtocolTemplateRecord {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(template).write(to: url, options: .atomic)
        return ScenarioProtocolTemplateRecord(url: url, template: template)
    }

    public static func delete(_ record: ScenarioProtocolTemplateRecord) throws {
        try FileManager.default.removeItem(at: record.url)
    }

    private static func existingName(at url: URL) -> String? {
        guard let data = try? Data(contentsOf: url),
            let template = try? JSONDecoder().decode(ScenarioProtocolTemplate.self, from: data)
        else { return nil }
        return template.name
    }

    // MARK: - Minting a template from a scenario

    /// Strips a scenario to protocol form.
    ///
    /// What comes out: the seats, the turn script, the routing, the per-turn
    /// caps and the endpoint declarations, verbatim — turn ids included, so a
    /// template and the panel it came from are diffable line by line. What is
    /// removed: `sharedMaterials`, and whatever bindings the file carried
    /// (`PanelComposition.semanticForm` does that half, so a bound legacy panel
    /// yields a protocol rather than a protocol plus a model).
    ///
    /// The name defaults to the scenario's, which is nearly always wrong for a
    /// protocol — `appellate-study3-prec-deliberative` names a CASE — so
    /// every caller is expected to offer a rename. It is not enforced: naming
    /// is the researcher's, and refusing a name would be this store deciding
    /// what a protocol is called.
    public static func templateFromScenario(
        _ scenario: MultiAgentScenario,
        named name: String? = nil,
        materialsChecklist: [String] = [],
        description: String? = nil
    ) -> ScenarioProtocolTemplate {
        var body = PanelComposition.semanticForm(scenario)
        body.sharedMaterials = ""
        if let name { body.name = name.trimmingCharacters(in: .whitespacesAndNewlines) }
        if let description { body.description = description }
        return ScenarioProtocolTemplate(
            materialsChecklist: materialsChecklist.map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }.filter { !$0.isEmpty },
            protocolBody: body)
    }

    // MARK: - The checklist

    /// What the checklist review found. Never a refusal — see below.
    public struct ChecklistReview: Sendable, Equatable {
        /// Checklist elements the researcher has not confirmed.
        public let unconfirmed: [String]
        /// Everything worth showing, in display order.
        public let warnings: [String]

        public init(unconfirmed: [String], warnings: [String]) {
            self.unconfirmed = unconfirmed
            self.warnings = warnings
        }

        public var isClean: Bool { warnings.isEmpty }
    }

    /// Reviews shared materials against a template's checklist.
    ///
    /// **The checklist is confirmed, not detected.** "The case record",
    /// "procedural posture", "the disposition scale with anchors" are
    /// statements about MEANING, and no substring match decides them: a
    /// keyword scan would clear materials that mention the words and flag good
    /// materials that phrase them differently, and either failure teaches the
    /// researcher to ignore the warning. So the researcher ticks each element
    /// off and anything unticked is reported as unconfirmed — an honest "you
    /// did not say this was in there", never a claim about the prose.
    ///
    /// The one thing that IS checked mechanically is whether there are any
    /// materials at all, because that is a fact about the file and it is the
    /// exact failure mode the blank-`sharedMaterials` habit produced.
    ///
    /// Soft in every case: this returns warnings, and no caller may promote
    /// them to a refusal. A researcher instantiating a protocol for a case
    /// whose record genuinely has no procedural posture must not be blocked by
    /// a checklist somebody else wrote last month.
    public static func review(
        checklist: [String],
        sharedMaterials: String,
        confirmed: Set<String> = []
    ) -> ChecklistReview {
        var warnings: [String] = []
        let materials = sharedMaterials.trimmingCharacters(in: .whitespacesAndNewlines)
        if materials.isEmpty {
            warnings.append(
                "the shared materials are empty — a panel with no case record "
                    + "still runs, and every seat will deliberate about nothing")
        }
        let unconfirmed = checklist.filter { !confirmed.contains($0) }
        if !unconfirmed.isEmpty {
            warnings.append(
                "the protocol's materials checklist is not fully confirmed — "
                    + "unconfirmed: \(unconfirmed.joined(separator: "; "))")
        }
        return ChecklistReview(unconfirmed: unconfirmed, warnings: warnings)
    }

    // MARK: - Instantiate

    /// A minted scenario and everything the researcher should see about it.
    public struct Instantiation: Sendable {
        public let record: MultiAgentScenarioRecord
        public let review: ChecklistReview
        public var warnings: [String] { review.warnings }

        public init(record: MultiAgentScenarioRecord, review: ChecklistReview) {
            self.record = record
            self.review = review
        }
    }

    /// Template + name + shared materials → an ORDINARY scenario file.
    ///
    /// Written through `MultiAgentScenarioStore.save`, so the result is
    /// indistinguishable from a hand-authored panel: same directory, same
    /// slug rule, same uniquing, same bytes a study would pin. The checklist
    /// review rides along as warnings and never stops the write.
    @discardableResult
    public static func instantiate(
        _ template: ScenarioProtocolTemplate,
        name: String,
        sharedMaterials: String,
        confirmedChecklistItems: Set<String> = [],
        description: String? = nil
    ) throws -> Instantiation {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ExperimentError(reason: "the new scenario needs a name")
        }
        var scenario = template.protocolBody
        scenario.name = trimmed
        scenario.description = description ?? template.templateDescription
        scenario.sharedMaterials = sharedMaterials
        // Normalised on the way out for the same reason a save is: the panel
        // store must only ever be handed a semantic panel.
        scenario = PanelComposition.semanticForm(scenario)
        let record = try MultiAgentScenarioStore.save(scenario)
        return Instantiation(
            record: record,
            review: review(
                checklist: template.materialsChecklist,
                sharedMaterials: sharedMaterials,
                confirmed: confirmedChecklistItems))
    }
}

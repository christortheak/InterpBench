import CryptoKit
import Foundation

/// A study with the agents taken out.
///
/// The same replication gets run many times against different agent sets. Done
/// by hand that means duplicating a study and re-picking arms, and the parts
/// that must NOT vary — the task file and its hash, the instruments, the
/// judges and rubric, the sampling policy, the battery — are re-entered (or
/// silently inherited stale) on every repetition. A template makes the
/// invariant half a first-class artifact: it holds every generation and
/// measurement setting and holds NO agents, so the only thing an
/// instantiation decides is the casting.
///
/// The hard constraint is that instantiation produces an ORDINARY draft study.
/// There is no template-aware run path, no template-aware freeze, and no
/// server-side knowledge of any of this: a minted study is a manifest like any
/// other, and the whole firewall — pins, verify, freeze gates, epoch guard,
/// run stamps — applies to it unchanged. Templates are an AUTHORING
/// convenience that leaves no trace in the measurement path except one
/// provenance stamp.
///
/// Stored in the WORKSPACE (`templates/<name>/template.json`), never in the
/// code repo, for the same reason experiments and prompts are.
public struct StudyTemplate: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    /// The semantic panel a multi-agent template instantiates against, pinned
    /// by workspace-relative path + SHA-256 exactly like every other declared
    /// input. Instantiation COMPILES this plus a seat casting into an ordinary
    /// bound scenario (see `PanelComposition`).
    public struct SemanticScenarioRef: Codable, Sendable, Equatable {
        public var path: String
        public var hash: String

        public init(path: String, hash: String) {
            self.path = path
            self.hash = hash
        }
    }

    public var schemaVersion: Int
    /// Directory name under `templates/`. Not part of the content hash: a
    /// template renamed is the same template.
    public var name: String
    public var templateDescription: String
    public var createdAt: String
    /// Set when this template was minted from a study that had DIVERGED from
    /// an earlier one — the lineage that says "this is that replication, with
    /// changes", rather than a template that appeared from nowhere.
    public var parentTemplate: String?
    public var semanticScenario: SemanticScenarioRef?
    /// The partial manifest: every setting a study carries EXCEPT the ones
    /// that identify an instance or bind its agents. See
    /// `StudyTemplateStore.strippedBody` for the exact disposition of every
    /// manifest field.
    public var study: ExperimentManifest

    public init(
        name: String,
        templateDescription: String = "",
        createdAt: String = ISO8601DateFormatter().string(from: Date()),
        parentTemplate: String? = nil,
        semanticScenario: SemanticScenarioRef? = nil,
        study: ExperimentManifest
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.name = name
        self.templateDescription = templateDescription
        self.createdAt = createdAt
        self.parentTemplate = parentTemplate
        self.semanticScenario = semanticScenario
        self.study = study
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion =
            try container.decodeIfPresent(Int.self, forKey: .schemaVersion)
            ?? Self.currentSchemaVersion
        name = try container.decode(String.self, forKey: .name)
        templateDescription =
            try container.decodeIfPresent(String.self, forKey: .templateDescription) ?? ""
        createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt) ?? ""
        parentTemplate = try container.decodeIfPresent(String.self, forKey: .parentTemplate)
        semanticScenario = try container.decodeIfPresent(
            SemanticScenarioRef.self, forKey: .semanticScenario)
        study = try container.decode(ExperimentManifest.self, forKey: .study)
    }

    /// The study kind this template mints — the same derivation the app uses
    /// on an ordinary manifest, so a template's type is never a second
    /// declaration that can disagree with its content.
    public var intent: StudyIntent {
        StudyIntent.parse(study.studyType ?? "")
            ?? (study.studyKind == .multiAgent ? .multiAgent : .agentComparison)
    }
}

/// Minting, listing and instantiating study templates.
///
/// Everything here is authoring-side. Nothing in this type is read by a run,
/// a freeze, an analysis, or the Python engine.
public enum StudyTemplateStore {

    // MARK: - Location

    /// `templates/` beside `experiments/`, `prompts/` and `runs/` in the
    /// workspace. Resolved through `ExperimentStore.workspaceRoot`, so the
    /// test root override reaches templates as it reaches everything else.
    public static var directory: URL {
        ExperimentStore.workspaceRoot.appending(component: "templates")
    }

    static func templateURL(_ name: String) -> URL {
        directory.appending(components: name, "template.json")
    }

    // MARK: - Persistence

    public static func list() -> [StudyTemplate] {
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }
        return entries
            .compactMap { try? load(name: $0.lastPathComponent) }
            .sorted { $0.createdAt > $1.createdAt }
    }

    public static func load(name: String) throws -> StudyTemplate {
        let data = try Data(contentsOf: templateURL(name))
        return try JSONDecoder().decode(StudyTemplate.self, from: data)
    }

    public static func save(_ template: StudyTemplate) throws {
        guard !template.name.isEmpty else {
            throw ExperimentError(reason: "empty template name")
        }
        let url = templateURL(template.name)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(template).write(to: url, options: .atomic)
    }

    /// Overwrite a design that ALREADY EXISTS, by name.
    ///
    /// Separate from `save` on purpose. `save` is a create-or-overwrite upsert,
    /// which is right for minting and for the description edit but wrong for
    /// the round trip: "save this draft back to design 'x'" must be a decision
    /// about a design the researcher can see in the library, and a silent upsert
    /// would turn a typo'd or stale name into a brand-new design that looks like
    /// an edit. So this refuses a name that is not already there and says which
    /// one, and the create path stays exactly where it was
    /// (`templateFromStudy`).
    public static func update(_ template: StudyTemplate) throws {
        guard (try? load(name: template.name)) != nil else {
            throw ExperimentError(
                reason: "no design named '\(template.name)' — updating a design "
                    + "in place requires one to exist (use 'Save as new design' "
                    + "to create one)")
        }
        try save(template)
    }

    public static func delete(name: String) throws {
        try FileManager.default.removeItem(
            at: directory.appending(component: name))
    }

    /// Renames a template: the directory moves and the manifest's `name`
    /// follows.
    ///
    /// Unlike a study rename this is unconditional — a template is never
    /// frozen and nothing stamps its name into evidence. Studies already
    /// minted from it keep the OLD name in their `templateProvenance`, which
    /// is the same trade study renames make: provenance records what was true
    /// when it was written. Those instances stop resolving back to a live
    /// template and will mint a fresh one on the next "load as template".
    @discardableResult
    public static func rename(
        templateName oldName: String, to newName: String
    ) throws -> String {
        var template = try load(name: oldName)
        let sanitized = ExperimentStore.resolvedRenameTarget(newName)
        guard sanitized.contains(where: { $0.isLetter || $0.isNumber }) else {
            throw ExperimentError(
                reason: "empty name — a template name needs at least one "
                    + "letter or digit (it becomes the templates/<name>/ "
                    + "directory)")
        }
        guard sanitized != oldName else { return oldName }
        let destination = directory.appending(component: sanitized)
        guard (try? load(name: sanitized)) == nil,
            !FileManager.default.fileExists(atPath: destination.path)
        else {
            throw ExperimentError(reason: "template '\(sanitized)' already exists")
        }
        try FileManager.default.moveItem(
            at: directory.appending(component: oldName), to: destination)
        template.name = sanitized
        try save(template)
        return sanitized
    }

    /// A template name not yet taken: `base`, else `base-2`, `base-3`, … —
    /// the suffix shape studies already use, so a workspace reads the same
    /// way in both trees.
    public static func unusedTemplateName(base: String) -> String {
        let sanitized = ExperimentStore.sanitizedExperimentName(base)
        guard !sanitized.isEmpty else { return sanitized }
        var candidate = sanitized
        var counter = 1
        while (try? load(name: candidate)) != nil
            || FileManager.default.fileExists(
                atPath: directory.appending(component: candidate).path)
        {
            counter += 1
            candidate = "\(sanitized)-\(counter)"
        }
        return candidate
    }

    // MARK: - Content hash

    /// The template's content hash: its SETTINGS, and nothing about its label
    /// or its history.
    ///
    /// Name, description, creation time and lineage are excluded on purpose —
    /// this hash answers exactly one question, "do these two describe the same
    /// study minus agents?", and that is what the mint-dedup rule needs. Built
    /// on `ExperimentStore.manifestHash` for the body so template identity and
    /// study identity canonicalize the same way.
    public static func hash(_ template: StudyTemplate) -> String {
        var canonical = template
        canonical.name = ""
        canonical.templateDescription = ""
        canonical.createdAt = ""
        canonical.parentTemplate = nil
        let bodyHash = ExperimentStore.manifestHash(canonical.study)
        let scenario = canonical.semanticScenario.map { "\($0.path)\u{1}\($0.hash)" } ?? ""
        let material = "\(canonical.schemaVersion)\u{0}\(bodyHash)\u{0}\(scenario)"
        return SHA256.hash(data: Data(material.utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Stripping a study to template form

    /// The manifest with everything instance-specific removed.
    ///
    /// A DENYLIST, deliberately, where `createConfirmationDraft` uses an
    /// allowlist — the two are answering different questions. A confirmation
    /// draft inherits a chosen few scientific pins and must default new fields
    /// to NOT inherited. A template means "this study, minus the agents", so a
    /// manifest field added next year should be carried by default; forgetting
    /// to list it here would silently drop a setting the researcher believed
    /// the template held.
    ///
    /// Removed, and why:
    /// - `name`, `createdAt` — identity of an instance, not of the recipe.
    /// - `status`, `frozenAt`, `freezeHash`, `frozenBy`, `gitCommit`,
    ///   `appVersion`, `freezeForced`, `forcedGatesSkipped` — lifecycle
    ///   stamps; a template is never frozen and must never mint a study that
    ///   claims to be.
    /// - `variantConditions` — THE agents. The entire point.
    /// - `conditions` — steering arms are per-instance too, and a
    ///   sweep-stamped `<concept>-recommended` condition carries selection
    ///   provenance from one particular sweep run; copying it into every
    ///   sibling would attribute one run's evidence to studies that never ran.
    /// - `multiAgentScenarioPath`/`Hash` — the study's scenario is a COMPILED,
    ///   seat-bound file. The template holds the SEMANTIC panel instead, in
    ///   its own `semanticScenario` field.
    /// - `templateProvenance` — a template does not descend from itself.
    ///
    /// Everything else is carried verbatim: the task file and its pin, the
    /// instruments and their scope declaration, ordinal aggregation, the
    /// parser and registry pin, exclusion rules, sampling policy and seeds,
    /// prompt mode and system prompt, dtype, the battery, judges and rubric,
    /// human baseline and validation, readers, the sweep spec, the pipeline
    /// and J-lens blocks, concepts and their corpora, phase and case family,
    /// and the prose fields.
    static func strippedBody(_ manifest: ExperimentManifest) -> ExperimentManifest {
        var body = manifest
        body.name = ""
        body.createdAt = ""
        body.status = .draft
        body.frozenAt = nil
        body.freezeHash = nil
        body.frozenBy = nil
        body.gitCommit = nil
        body.appVersion = nil
        body.freezeForced = nil
        body.forcedGatesSkipped = nil
        body.variantConditions = []
        body.conditions = []
        body.multiAgentScenarioPath = nil
        body.multiAgentScenarioHash = nil
        body.multiAgentSemanticScenarioPath = nil
        body.multiAgentSemanticScenarioHash = nil
        body.templateProvenance = nil
        return body
    }

    // MARK: - Mint from a study ("Load as Template")

    /// The outcome of `templateFromStudy`.
    public struct Mint: Sendable {
        public let template: StudyTemplate
        public let hash: String
        /// False when the study was recognised as an unchanged instance of a
        /// template that already exists — no new template was written.
        public let minted: Bool
        /// The template this one diverged FROM, when a diverged study forced a
        /// new mint.
        public let divergedFrom: String?
        /// Things the researcher must see: a bound panel that had to be
        /// hoisted, seats that disagreed about the model.
        public let warnings: [String]

        public init(
            template: StudyTemplate, hash: String, minted: Bool,
            divergedFrom: String? = nil, warnings: [String] = []
        ) {
            self.template = template
            self.hash = hash
            self.minted = minted
            self.divergedFrom = divergedFrom
            self.warnings = warnings
        }
    }

    /// Strips a study to template form, minting a template only when it has to.
    ///
    /// The dedup rule matters because "load as template" is a button a
    /// researcher presses whenever they want to run the replication again. A
    /// naive implementation mints a near-identical template every time, and
    /// within a week the picker holds a dozen indistinguishable entries and
    /// nobody can say which one the last study came from.
    ///
    /// So: a study carrying `templateProvenance` whose stripped form still
    /// hashes to its template's hash IS that template — return it, write
    /// nothing. A study with no lineage, or one that has DIVERGED (any setting
    /// edited since it was minted), gets a new template, and a diverged one
    /// records the template it came from.
    public static func templateFromStudy(
        experimentName: String,
        named requestedName: String? = nil,
        description: String? = nil
    ) throws -> Mint {
        let study = try ExperimentStore.load(name: experimentName)
        let body = strippedBody(study)

        // The semantic panel, if this is a multi-agent study.
        var warnings: [String] = []
        var semanticScenario: StudyTemplate.SemanticScenarioRef?
        var semanticPanel: MultiAgentScenario?
        if let scenarioPath = study.multiAgentScenarioPath, !scenarioPath.isEmpty {
            let hoist = try PanelComposition.hoistLegacyScenario(path: scenarioPath)
            semanticPanel = hoist.semantic
            warnings += hoist.warnings
        }

        // An unchanged instance of a live template returns that template.
        if let provenance = study.templateProvenance,
            let existing = try? load(name: provenance.template)
        {
            // The panel is compared structurally (see `sameSemanticPanel`);
            // everything else is compared through the ordinary content hash.
            let panelMatches = semanticPanel.map { sameSemanticPanel($0, as: existing) }
                ?? (existing.semanticScenario == nil)
            var candidate = existing
            candidate.study = body
            if panelMatches, hash(candidate) == hash(existing) {
                return Mint(
                    template: existing, hash: hash(existing), minted: false,
                    warnings: warnings)
            }
        }

        // Diverged, or never had a lineage: mint.
        let base = requestedName
            ?? ExperimentStore.displayLabel(name: experimentName)
            ?? study.name
        let name = unusedTemplateName(base: ExperimentStore.canonicalSlug(base))
        guard !name.isEmpty else {
            throw ExperimentError(
                reason: "'\(base)' has no letters or digits to make a template "
                    + "directory name from")
        }
        if let semanticPanel {
            semanticScenario = try pinSemanticPanel(
                semanticPanel, slug: name,
                reusing: study.multiAgentSemanticScenarioPath)
        }
        let template = StudyTemplate(
            name: name,
            templateDescription: description ?? study.experimentDescription,
            parentTemplate: study.templateProvenance?.template,
            semanticScenario: semanticScenario,
            study: body)
        try save(template)
        return Mint(
            template: template, hash: hash(template), minted: true,
            divergedFrom: study.templateProvenance?.template, warnings: warnings)
    }

    // MARK: - Save a study back onto the design it names

    /// What an in-place design update did.
    public struct DesignUpdate: Sendable {
        public let design: String
        public let hashBefore: String
        public let hashAfter: String
        /// False when the draft's design form already matched — the write
        /// happened, nothing about the recipe moved.
        public var changed: Bool { hashBefore != hashAfter }
        /// Things the researcher must see (a hoisted panel, a dropped one).
        public let warnings: [String]

        public init(
            design: String, hashBefore: String, hashAfter: String,
            warnings: [String] = []
        ) {
            self.design = design
            self.hashBefore = hashBefore
            self.hashAfter = hashAfter
            self.warnings = warnings
        }
    }

    /// Strips a study to design form and OVERWRITES the design its lineage
    /// names.
    ///
    /// The other half of "Edit design…": a design is revised by instantiating
    /// it into an ordinary scratch draft, editing that draft in the ONE manifest
    /// editor, and writing the result back here. That round trip is why the
    /// Templates tab has no editor of its own (see `StudyDesignSummary`), and
    /// this is the return leg — without it the loop dead-ends at
    /// `templateFromStudy`, which can only mint a NEW design and so grows the
    /// library by one entry per revision.
    ///
    /// **Studies minted earlier are untouched, and their divergence display
    /// stays honest.** `agreement(of:)` compares a study's stripped form
    /// against `templateProvenance.templateHash` — the hash stamped at MINT
    /// time, not the design's current one — so bumping a design's content hash
    /// here cannot silently re-file an existing instance as matching or
    /// diverged. (One real coupling: for a PANEL study `agreement` also
    /// compares the study's compiled scenario against the design's CURRENT
    /// semantic panel, so replacing a design's panel does re-read older panel
    /// instances as diverged. That is the correct reading — they no longer run
    /// the panel that name denotes — and it is why the panel case emits a
    /// warning below.)
    ///
    /// What is NOT taken from the draft: the design's name, description,
    /// creation time and `parentTemplate`. Those are the design's identity and
    /// the researcher's own note about it — none is part of the content hash —
    /// and clobbering the library's note with a draft's "question or purpose"
    /// field is not what "save back to design" says it does.
    @discardableResult
    public static func saveStudyBackToDesign(
        experimentName: String
    ) throws -> DesignUpdate {
        let study = try ExperimentStore.load(name: experimentName)
        guard let provenance = study.templateProvenance else {
            throw ExperimentError(
                reason: "'\(experimentName)' was not minted from a design — "
                    + "there is nothing to save it back onto (save it as a NEW "
                    + "design instead)")
        }
        guard let existing = try? load(name: provenance.template) else {
            throw ExperimentError(
                reason: "design '\(provenance.template)' is no longer in the "
                    + "library (renamed or deleted) — save this study as a new "
                    + "design instead")
        }

        var warnings: [String] = []
        var updated = existing
        updated.study = strippedBody(study)

        if let scenarioPath = study.multiAgentScenarioPath, !scenarioPath.isEmpty {
            // The study's scenario is COMPILED and seat-bound; a design holds
            // the semantic panel, so the panel is hoisted and compared
            // structurally exactly as minting does.
            let hoist = try PanelComposition.hoistLegacyScenario(path: scenarioPath)
            warnings += hoist.warnings
            if !sameSemanticPanel(hoist.semantic, as: existing) {
                updated.semanticScenario = try pinSemanticPanel(
                    hoist.semantic, slug: existing.name,
                    reusing: study.multiAgentSemanticScenarioPath)
                warnings.append(
                    "design '\(existing.name)' now declares a different panel — "
                        + "studies minted from the old one read as diverged, "
                        + "which is what they are")
            }
        } else if existing.semanticScenario != nil {
            updated.semanticScenario = nil
            warnings.append(
                "'\(experimentName)' carries no panel, so design "
                    + "'\(existing.name)' no longer declares one — seat "
                    + "castings against it will refuse until a panel is saved "
                    + "back")
        }

        let before = hash(existing)
        try update(updated)
        return DesignUpdate(
            design: updated.name, hashBefore: before, hashAfter: hash(updated),
            warnings: warnings)
    }

    /// Struct equality, not byte equality: the template's semantic panel is a
    /// file the researcher may have hand-formatted, while the study's is the
    /// stripped form of a compiled one. Comparing bytes would call every
    /// instance diverged.
    /// Internal rather than private: the live divergence check
    /// (`StudyDesign.swift`) has to compare panels by exactly this rule, and a
    /// second copy of it would be a second answer to "is this the same panel?".
    static func sameSemanticPanel(
        _ panel: MultiAgentScenario, as template: StudyTemplate
    ) -> Bool {
        guard let ref = template.semanticScenario,
            let data = try? Data(
                contentsOf: ExperimentStore.resolveProjectPath(ref.path)),
            let stored = try? JSONDecoder().decode(MultiAgentScenario.self, from: data)
        else { return false }
        return PanelComposition.semanticForm(stored) == panel
    }

    /// Writes a semantic panel into the panel library and returns its pin.
    ///
    /// `reusing` is the scenario a CAST study records as the source of its
    /// casting (`multiAgentSemanticScenarioPath`). When that file is still in
    /// the library and still says the same thing, the design points at it
    /// rather than at a byte-identical copy: a workspace where every design
    /// mints its own `-semantic.json` twin makes the scenario picker unusable
    /// within a few replications.
    private static func pinSemanticPanel(
        _ panel: MultiAgentScenario, slug: String, reusing recorded: String? = nil
    ) throws -> StudyTemplate.SemanticScenarioRef {
        if let recorded, !recorded.isEmpty {
            let url = ExperimentStore.resolveProjectPath(recorded)
            if let data = try? Data(contentsOf: url),
                let stored = try? JSONDecoder().decode(MultiAgentScenario.self, from: data),
                PanelComposition.semanticForm(stored) == panel
            {
                return StudyTemplate.SemanticScenarioRef(
                    path: recorded, hash: MultiAgentScenarioStore.hash(data))
            }
        }
        let root = MultiAgentScenarioStore.directory
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        var url = root.appending(component: "\(slug)-semantic.json")
        var suffix = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = root.appending(component: "\(slug)-semantic-\(suffix).json")
            suffix += 1
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(panel)
        try data.write(to: url, options: .atomic)
        return StudyTemplate.SemanticScenarioRef(
            path: FineTuneStore.relativePath(for: url),
            hash: MultiAgentScenarioStore.hash(data))
    }

    // MARK: - Instantiate

    /// One study's worth of casting: the agents an instance runs.
    public enum Cell: Sendable {
        /// Compare-agents: the arms, in the order they should appear.
        case agents([ModelVariantRecord])
        /// Multi-agent: who sits in which seat of the template's panel.
        case seating(SeatAssignment)

        /// The name fragment this casting contributes to its study's name.
        /// Capped: the fragment becomes a directory name, and a six-agent
        /// casting would otherwise produce one nobody can read or type.
        var descriptor: String {
            let raw: String
            switch self {
            case .agents(let records):
                raw = records.isEmpty
                    ? "baseline"
                    : records.map { ExperimentStore.canonicalSlug($0.artifact.name) }
                        .joined(separator: "-")
            case .seating(let assignment):
                raw = ExperimentStore.canonicalSlug(assignment.descriptor)
            }
            guard raw.count > 48 else { return raw }
            // Truncation collides; `unusedExperimentName` resolves it with the
            // ordinary `-2` suffix, and the manifest holds the full casting.
            return String(raw.prefix(48))
        }
    }

    /// Mints an ordinary draft study from a template plus one casting.
    ///
    /// Ordinary is the whole contract. The result is a manifest the Studies
    /// panel, both CLIs, `verify`, `freeze` and the Python engine treat like
    /// any hand-authored draft; the only thing that marks it is the
    /// `templateProvenance` stamp, which is provenance rather than a pin.
    ///
    /// Two things are RE-DERIVED rather than copied, because a template is a
    /// recipe applied at a later date than it was written:
    ///
    /// 1. The task-prompt bytes are re-checked against the template's pin. A
    ///    drifted file refuses HERE — minting a study whose first `verify`
    ///    fails wastes the researcher's next hour.
    /// 2. `outcomeInstrumentScope` is re-pinned against that file through the
    ///    ordinary declaration path. A stale id-set pin is the failure that
    ///    motivated this: four Slurm shards refused after the model was
    ///    already loaded, because the pinned item ids no longer matched the
    ///    file the study named.
    @discardableResult
    public static func instantiate(
        templateName: String,
        cell: Cell,
        studyName: String? = nil,
        batchGroup: String? = nil
    ) throws -> ExperimentManifest {
        let template = try load(name: templateName)
        let templateHash = hash(template)

        var draft = template.study
        draft.createdAt = ISO8601DateFormatter().string(from: Date())
        draft.status = .draft
        draft.templateProvenance = ExperimentManifest.TemplateProvenance(
            template: template.name, templateHash: templateHash,
            batchGroup: batchGroup)

        let name = ExperimentStore.unusedExperimentName(
            base: studyName ?? "\(template.name)-\(cell.descriptor)")
        guard !name.isEmpty else {
            throw ExperimentError(
                reason: "could not derive a study directory name for this casting")
        }
        draft.name = name

        try refuseDriftedTaskPrompts(draft)

        switch cell {
        case .agents(let records):
            // The SAME path the Studies panel's "Add agent" uses, so a minted
            // arm and a clicked arm are byte-identical.
            for record in records {
                try ExperimentStore.attachAgent(record, into: &draft)
            }
        case .seating(let assignment):
            guard let ref = template.semanticScenario else {
                throw ExperimentError(
                    reason: "template '\(templateName)' declares no semantic "
                        + "panel — a seat casting has nothing to compile against")
            }
            let semantic = try loadSemanticPanel(ref)
            // The ONE compile-and-pin (`SeatCasting.compile`), shared with the
            // Studies editor's Seats section — a minted casting and a
            // hand-cast one are the same write, including the provenance that
            // lets the study's seats be re-listed and re-cast later.
            try SeatCasting.compile(
                assignment, semantic: semantic, semanticPath: ref.path,
                into: &draft)
        }

        try ExperimentStore.save(draft, allowCreate: true)
        // Re-pin the derived scope through the ordinary declaration path.
        if let scope = draft.outcomeInstrumentScope {
            return try ExperimentStore.declareOutcomeInstrumentScope(
                responseFormats: scope.responseFormats, experimentName: name)
        }
        return draft
    }

    // MARK: - The scratch draft "Edit design…" opens

    /// The name an edit draft takes: `<design>-edit`, uniquified the ordinary
    /// way. Recognisable in a picker of thirty, and disposable — an edit draft
    /// is a means of revising a design, not a study anyone intends to run.
    public static func editDraftName(templateName: String) -> String {
        ExperimentStore.unusedExperimentName(base: "\(templateName)-edit")
    }

    /// The agentless casting an edit draft is minted with.
    ///
    /// A compare-agents design takes zero agents — a legal draft whose only arm
    /// is the baseline (the same thing "Load Only" writes when no agent is
    /// cast). A PANEL design cannot: seats are compiled into the study's
    /// scenario file at mint time, and an `.agents([])` casting would produce a
    /// draft with no scenario at all — which, saved back, would strip the panel
    /// off the design. So a panel edit draft is cast all-baseline, which
    /// compiles the design's own panel and hoists back to it unchanged.
    static func editDraftCell(for template: StudyTemplate) throws -> Cell {
        guard template.intent == .multiAgent else { return .agents([]) }
        let seats = try semanticSeatIDs(templateName: template.name)
        return .seating(
            SeatAssignment(
                seatIDs: seats,
                occupants: Dictionary(
                    uniqueKeysWithValues: seats.map { ($0, SeatOccupant.baseline) })))
    }

    /// Mints the ordinary scratch draft that "Edit design…" opens.
    ///
    /// Nothing about it is special: it carries the design's `templateProvenance`
    /// like any instance, the full Studies editor applies, and it freezes and
    /// runs like any other draft if the researcher decides to keep it. The only
    /// thing that makes it an "edit" draft is what the researcher does next —
    /// `saveStudyBackToDesign`.
    @discardableResult
    public static func mintEditDraft(
        templateName: String
    ) throws -> ExperimentManifest {
        let template = try load(name: templateName)
        return try instantiate(
            templateName: templateName,
            cell: try editDraftCell(for: template),
            studyName: editDraftName(templateName: templateName))
    }

    /// Mints one draft per casting, all stamped with a shared batch id.
    ///
    /// Panels take the sibling-studies route by necessity: a manifest carries
    /// exactly ONE scenario on both engines, and its conditions are the fixed
    /// `baseline`/`configured` pair, so N castings cannot be N conditions of
    /// one study without changing the run loop. `batchGroup` is what puts them
    /// back together for analysis.
    ///
    /// Minting only — submission is the caller's decision, and a batch that
    /// mints drafts is reviewable before it costs GPU time.
    @discardableResult
    public static func instantiateBatch(
        templateName: String,
        cells: [Cell],
        batchID: String? = nil
    ) throws -> [ExperimentManifest] {
        let batch = batchID ?? Self.newBatchID()
        return try cells.map {
            try instantiate(templateName: templateName, cell: $0, batchGroup: batch)
        }
    }

    /// A batch id: short, sortable, and unique enough for one workspace.
    public static func newBatchID(date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "batch-\(formatter.string(from: date))-"
            + String(UUID().uuidString.prefix(6)).lowercased()
    }

    // MARK: - Standard batches

    /// The composition sweep as cells: all-baseline, each seat solo-treated,
    /// all-treated.
    public static func compositionSweepCells(
        templateName: String, agent: SeatOccupant
    ) throws -> [Cell] {
        let seats = try semanticSeatIDs(templateName: templateName)
        return PanelComposition.compositionSweep(seatIDs: seats, agent: agent)
            .map(Cell.seating)
    }

    /// Every distinct casting of an agent multiset across the panel's seats.
    public static func permutationCells(
        templateName: String, occupants: [SeatOccupant]
    ) throws -> [Cell] {
        let seats = try semanticSeatIDs(templateName: templateName)
        return try PanelComposition.distinctAssignments(
            seatIDs: seats, occupants: occupants
        ).map(Cell.seating)
    }

    /// The seat ids of a multi-agent template's panel, in panel order.
    public static func semanticSeatIDs(templateName: String) throws -> [String] {
        let template = try load(name: templateName)
        guard let ref = template.semanticScenario else {
            throw ExperimentError(
                reason: "template '\(templateName)' declares no semantic panel")
        }
        return PanelComposition.seatIDs(try loadSemanticPanel(ref))
    }

    // MARK: - Helpers

    private static func loadSemanticPanel(
        _ ref: StudyTemplate.SemanticScenarioRef
    ) throws -> MultiAgentScenario {
        let url = ExperimentStore.resolveProjectPath(ref.path)
        guard let data = try? Data(contentsOf: url) else {
            throw ExperimentError(reason: "semantic panel not found: \(url.path)")
        }
        let live = MultiAgentScenarioStore.hash(data)
        guard live == ref.hash else {
            throw ExperimentError(
                reason: "the template's semantic panel changed since it was "
                    + "pinned (\(ref.hash.prefix(12))… → \(live.prefix(12))…) "
                    + "— re-mint the template so its instances agree about "
                    + "what they are running")
        }
        return try JSONDecoder().decode(MultiAgentScenario.self, from: data)
    }

    /// Refuses a mint whose task prompts no longer match the template's pin.
    ///
    /// The alternative — re-pinning the hash from the current file — would
    /// launder drift: the minted study would verify cleanly while measuring
    /// items the template never described. Refusing sends the researcher back
    /// to re-mint the template deliberately.
    private static func refuseDriftedTaskPrompts(_ draft: ExperimentManifest) throws {
        guard let file = draft.taskPromptsFile, !file.isEmpty,
            let pinned = draft.taskPromptsHash
        else { return }
        let url = ExperimentStore.resolveProjectPath(file)
        guard let data = try? Data(contentsOf: url) else {
            throw ExperimentError(
                reason: "the template's task prompts are missing: \(url.path)")
        }
        let live = ExperimentStore.sha256Hex(data)
        guard live == pinned else {
            throw ExperimentError(
                reason: "'\(file)' changed since this template was minted "
                    + "(\(pinned.prefix(12))… → \(live.prefix(12))…) — "
                    + "re-mint the template rather than minting a study whose "
                    + "pins are already stale")
        }
    }
}

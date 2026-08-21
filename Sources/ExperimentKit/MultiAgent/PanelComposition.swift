import Foundation

/// Who occupies a panel seat in one casting: a named agent artifact, or the
/// plain model with no intervention.
///
/// The agent is carried by workspace-relative path + artifact hash — the same
/// pinning convention `ExperimentManifest.VariantCondition` uses — rather than
/// by a live record, so an assignment is a value that can be built, permuted,
/// stored and compared without touching the filesystem.
public enum SeatOccupant: Sendable, Equatable, Hashable, Codable {
    /// The study's base model, unsteered. Not "no one": the baseline seat is
    /// a condition, and a panel where every seat is baseline is the control
    /// composition.
    case baseline
    case agent(name: String, artifactPath: String, artifactHash: String)

    /// The label this occupant contributes to a study/scenario name.
    public var label: String {
        switch self {
        case .baseline: "baseline"
        case .agent(let name, _, _): name
        }
    }

    /// Total order for deterministic permutation output. Two occupants that
    /// sort equal ARE the same casting, which is what makes the multiset
    /// dedupe correct.
    var sortKey: String {
        switch self {
        case .baseline: "\u{0}baseline"
        case .agent(let name, let path, let hash): "\(name)\u{1}\(path)\u{1}\(hash)"
        }
    }
}

/// One casting of a semantic panel: every seat filled.
///
/// Seats are keyed by the scenario's agent `id`, not its display name — the id
/// is what turns reference as `speakerAgentID`, so keying by name would let a
/// rename silently re-cast the panel.
public struct SeatAssignment: Sendable, Equatable, Codable {
    /// Seat ids in the scenario's own order. Kept explicitly (rather than
    /// derived from the dictionary) so an assignment has a stable reading
    /// order for names, labels and hashes.
    public var seatIDs: [String]
    public var occupants: [String: SeatOccupant]

    public init(seatIDs: [String], occupants: [String: SeatOccupant]) {
        self.seatIDs = seatIDs
        self.occupants = occupants
    }

    /// Positional constructor — seat i takes occupant i. The shape the
    /// permutation expander produces.
    public init(seatIDs: [String], ordered: [SeatOccupant]) {
        self.seatIDs = seatIDs
        var map: [String: SeatOccupant] = [:]
        for (seat, occupant) in zip(seatIDs, ordered) { map[seat] = occupant }
        self.occupants = map
    }

    public subscript(seat: String) -> SeatOccupant? { occupants[seat] }

    /// Occupants in seat order — the reading order for labels and names.
    public var ordered: [SeatOccupant] {
        seatIDs.compactMap { occupants[$0] }
    }

    /// Short human descriptor for auto-naming: seat occupants in order, e.g.
    /// `baseline-sympathy-baseline`.
    public var descriptor: String {
        ordered.map(\.label).joined(separator: "-")
    }
}

/// Semantic panels, seat casting, and the COMPILE step that turns the two into
/// a scenario the existing engines already know how to run.
///
/// The split exists because a panel script conflates two different kinds of
/// decision. Roles, turn structure, visibility and case materials are the
/// EXPERIMENT — they should survive unchanged across every casting. Which
/// agent sits in which seat, and which model at which sampling settings, are
/// per-STUDY parameters that change on every run of the replication.
/// Editing them in the script means hand-maintaining N near-identical panel
/// files, and the difference between two of them is invisible in a diff of
/// 200 lines of prompt text.
///
/// Nothing downstream learns about any of this. `compile` emits the EXISTING
/// scenario schema with every binding filled in, and the run loop, the freeze
/// packager (which reaches into `agents[].variantArtifactPath` to enumerate
/// seats) and the Python engine see an ordinary panel file.
public enum PanelComposition {

    /// Where compiled panels are written: `prompts/panels/compiled/`.
    ///
    /// Under `prompts/` because a compiled scenario is a pinned study input
    /// like any other — the freeze cleanliness gate must be able to see it,
    /// which is exactly why panels moved out of gitignored `runs/`. In a
    /// SUBDIRECTORY because `MultiAgentScenarioStore.scan` reads
    /// `prompts/panels/` non-recursively: generated castings would otherwise
    /// flood the panel picker with N near-identical entries and bury the
    /// semantic panels a researcher actually edits.
    public static var compiledDirectory: URL {
        MultiAgentScenarioStore.directory.appending(component: "compiled")
    }

    /// True when this path names a scenario THIS compiler wrote.
    ///
    /// The directory is the marker, and it is enough: nothing else writes into
    /// `prompts/panels/compiled/`, and a bound scenario found anywhere else is
    /// a hand-authored one whose castings live in the file (see
    /// `hoistLegacyScenario`). Accepts the workspace-relative form a manifest
    /// pins as well as an absolute path.
    public static func isCompiledPath(_ path: String) -> Bool {
        guard !path.isEmpty else { return false }
        let url = ExperimentStore.resolveProjectPath(path).standardizedFileURL
        return url.deletingLastPathComponent().standardizedFileURL.path
            == compiledDirectory.standardizedFileURL.path
    }

    // MARK: - The semantic form

    /// A scenario is SEMANTIC when no seat declares a model.
    ///
    /// There is no v2 schema and no new file format: a semantic panel is an
    /// ordinary scenario JSON with the binding fields left empty. That is a
    /// deliberate choice over inventing a parallel type — the semantic form
    /// stays loadable by the existing decoder, editable by the existing
    /// editor, and hashable as the same kind of pinned input.
    ///
    /// It is also structurally NOT RUNNABLE, which is the property that makes
    /// the design safe: `MultiAgentRunner.validate` already refuses an agent
    /// with an empty `baseModelID`, so a semantic panel that reaches a run
    /// without being compiled fails at validation instead of quietly
    /// generating from whatever model happened to be resident.
    ///
    /// (`agents[].baseModelID` is a REQUIRED key in the Swift decoder, so the
    /// semantic form writes it as `""` rather than omitting it — a panel with
    /// the key missing would not decode here at all, and would vanish from
    /// the picker through `scan`'s `try?`.)
    public static func isSemantic(_ scenario: MultiAgentScenario) -> Bool {
        !scenario.agents.isEmpty && scenario.agents.allSatisfy {
            $0.baseModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    /// Strips every binding from a scenario, leaving roles, turns, visibility
    /// and materials. Sampling settings go too: temperature and maxTokens are
    /// study parameters, and a scenario that carries its own is a second
    /// source of truth for the run's sampling policy.
    ///
    /// `systemPrompt` deliberately SURVIVES — a seat's system prompt is its
    /// ROLE ("you represent Team South"), not a binding. Which agent plays
    /// that role is the thing being cast.
    ///
    /// So do `agents[].role` and every turn's `contract` (panel turn
    /// contracts, 2026-08-17), for the same reason and more strongly: the role
    /// noun phrase is the seat's identity in the contract renderer's opener,
    /// and a contract IS the turn's prompt. Stripping either would leave a
    /// panel that still compiles, still runs, and renders different prose than
    /// the one the researcher authored. Written as a per-field strip of a
    /// COPIED seat rather than a rebuilt one precisely so a field added to the
    /// schema survives by default and has to be deliberately removed here to
    /// be lost.
    public static func semanticForm(_ scenario: MultiAgentScenario) -> MultiAgentScenario {
        var semantic = scenario
        semantic.baseModelID = ""
        semantic.temperature = 0
        semantic.maxTokens = MultiAgentScenario.semanticMaxTokensPlaceholder
        semantic.agents = scenario.agents.map { seat in
            var stripped = seat
            stripped.baseModelID = ""
            stripped.variantArtifactPath = nil
            stripped.variantArtifactHash = nil
            return stripped
        }
        return semantic
    }

    /// The seats of a semantic panel, in scenario order.
    public static func seatIDs(_ scenario: MultiAgentScenario) -> [String] {
        scenario.agents.map(\.id)
    }

    /// The casting a BOUND scenario is carrying, recovered as data.
    ///
    /// Pure: no file reads, no hashing, no warnings. That is exactly what the
    /// study editor needs — a compiled scenario always pins its own artifact
    /// hashes, so reopening a cast study is a decode and this, and the seat
    /// pickers can be filled without touching the agent library.
    /// `hoistLegacyScenario` builds on it for the messier job of migrating a
    /// hand-authored bound panel, where a seat may name an artifact with no
    /// hash and the researcher has to be told.
    public static func assignment(in scenario: MultiAgentScenario) -> SeatAssignment {
        var occupants: [String: SeatOccupant] = [:]
        for seat in scenario.agents {
            if let artifactPath = seat.variantArtifactPath, !artifactPath.isEmpty {
                occupants[seat.id] = .agent(
                    name: seat.name,
                    artifactPath: artifactPath,
                    artifactHash: seat.variantArtifactHash ?? "")
            } else {
                occupants[seat.id] = .baseline
            }
        }
        return SeatAssignment(seatIDs: seatIDs(scenario), occupants: occupants)
    }

    // MARK: - Compile

    /// Binds a semantic panel to one casting and one set of study parameters,
    /// producing the ORDINARY scenario schema.
    ///
    /// Compile, not re-plumb: everything the engines already do — seat model
    /// resolution, variant loading, freeze-time seat enumeration, the
    /// baseline/configured arm pair — keeps working on the output because the
    /// output is the same file shape those paths have always read.
    ///
    /// The TURN SCRIPT is not touched at all: `bound.turns` is the semantic
    /// panel's own array, so a contract turn compiles to the identical
    /// contract. Casting changes who speaks, never what the prompt says.
    public static func compile(
        semantic: MultiAgentScenario,
        assignment: SeatAssignment,
        modelID: String,
        temperature: Double,
        maxTokens: Int,
        name: String? = nil
    ) throws -> MultiAgentScenario {
        let model = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            throw ExperimentError(
                reason: "compiling a panel needs the study's base model — the "
                    + "semantic scenario deliberately declares none")
        }
        let seats = seatIDs(semantic)
        let unknown = assignment.seatIDs.filter { !seats.contains($0) }
        guard unknown.isEmpty else {
            throw ExperimentError(
                reason: "seat assignment names seats this panel does not have: "
                    + unknown.joined(separator: ", "))
        }
        let uncast = seats.filter { assignment[$0] == nil }
        guard uncast.isEmpty else {
            throw ExperimentError(
                reason: "no agent assigned to seat(s): "
                    + uncast.joined(separator: ", ")
                    + " — every seat runs, so every seat must be cast "
                    + "(use baseline for an unsteered seat)")
        }

        var bound = semantic
        // Stamped from the turn script, the same rule
        // `MultiAgentScenarioStore.update` applies on save. `compileAndPin`
        // writes its file with a bare encoder rather than through the store,
        // so without this a semantic panel whose in-memory version was stale
        // — hand-edited to `1` while carrying a contract turn, or built in
        // code before the turn was added — would compile to a pinned file
        // that misdescribes its own contents.
        bound.schemaVersion = semantic.requiredSchemaVersion
        // The panel's NAME is part of the experiment, not of the casting: it
        // stays whatever the semantic panel calls itself so that stripping a
        // compiled panel back reproduces the semantic one exactly (which is
        // what lets "reload this study as a template" recognise an unchanged
        // instance). Castings are told apart by the STUDY name and by the
        // compiled file name.
        bound.name = name ?? semantic.name
        bound.baseModelID = model
        bound.temperature = temperature
        bound.maxTokens = maxTokens
        bound.agents = semantic.agents.map { seat in
            var cast = seat
            cast.baseModelID = model
            switch assignment[seat.id] {
            case .agent(_, let path, let hash):
                cast.variantArtifactPath = path
                cast.variantArtifactHash = hash
            case .baseline, nil:
                cast.variantArtifactPath = nil
                cast.variantArtifactHash = nil
            }
            return cast
        }
        // Refuse here rather than at run start: a compiled panel that cannot
        // validate is a bug in the casting, and the queue wait is a bad place
        // to learn it.
        try MultiAgentRunner.validate(bound)
        return bound
    }

    /// Compiles and writes the bound panel as pinned workspace data, returning
    /// its workspace-relative path and SHA-256 — the two values a manifest
    /// pins. Bytes are hashed as WRITTEN, from one encode, so the pin cannot
    /// describe different bytes than the file holds.
    public static func compileAndPin(
        semantic: MultiAgentScenario,
        assignment: SeatAssignment,
        modelID: String,
        temperature: Double,
        maxTokens: Int,
        fileSlug: String,
        name: String? = nil
    ) throws -> (path: String, hash: String, scenario: MultiAgentScenario) {
        let bound = try compile(
            semantic: semantic, assignment: assignment, modelID: modelID,
            temperature: temperature, maxTokens: maxTokens, name: name)
        let root = compiledDirectory
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        let base = ExperimentStore.canonicalSlug(fileSlug)
        var url = root.appending(component: "\(base).json")
        var suffix = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = root.appending(component: "\(base)-\(suffix).json")
            suffix += 1
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(bound)
        try data.write(to: url, options: .atomic)
        return (
            path: FineTuneStore.relativePath(for: url),
            hash: MultiAgentScenarioStore.hash(data),
            scenario: bound
        )
    }

    // MARK: - Expansion

    /// Every DISTINCT casting of `occupants` across `seatIDs`.
    ///
    /// Multiset permutations, not plain permutations: three different agents
    /// over three seats give 6 castings, but `[A, A, B]` gives 3 — swapping
    /// the two A's produces the same panel, and running it twice would spend
    /// GPU hours measuring the same condition and then double-count it in the
    /// analysis.
    ///
    /// Pure, deterministic, and total-ordered by occupant sort key, so the
    /// same inputs give the same castings in the same order on both machines.
    /// Refuses a count mismatch rather than truncating: a partial casting is
    /// never what the caller meant.
    public static func distinctAssignments(
        seatIDs: [String], occupants: [SeatOccupant]
    ) throws -> [SeatAssignment] {
        guard seatIDs.count == occupants.count else {
            throw ExperimentError(
                reason: "\(occupants.count) agent(s) for \(seatIDs.count) seat(s) "
                    + "— a casting fills every seat exactly once")
        }
        guard !seatIDs.isEmpty else { return [] }
        let pool = occupants.sorted { $0.sortKey < $1.sortKey }
        var used = [Bool](repeating: false, count: pool.count)
        var current: [SeatOccupant] = []
        var out: [SeatAssignment] = []

        func recurse() {
            if current.count == pool.count {
                out.append(SeatAssignment(seatIDs: seatIDs, ordered: current))
                return
            }
            var lastKey: String?
            for index in pool.indices where !used[index] {
                // The pool is sorted, so equal occupants are adjacent: at each
                // position take only the FIRST unused member of each run of
                // equals, and the identical castings never get generated.
                if pool[index].sortKey == lastKey { continue }
                lastKey = pool[index].sortKey
                used[index] = true
                current.append(pool[index])
                recurse()
                current.removeLast()
                used[index] = false
            }
        }
        recurse()
        return out
    }

    /// The standard composition sweep for one agent: all-baseline, then each
    /// seat solo-treated, then all-treated.
    ///
    /// This is the panel-propagation question in its minimal form — does one
    /// treated member move the group, and does the effect scale with how many
    /// members carry it — so it is a named helper rather than something each
    /// study re-derives by hand.
    public static func compositionSweep(
        seatIDs: [String], agent: SeatOccupant
    ) -> [SeatAssignment] {
        guard !seatIDs.isEmpty else { return [] }
        var out: [SeatAssignment] = [
            SeatAssignment(
                seatIDs: seatIDs, ordered: Array(repeating: .baseline, count: seatIDs.count))
        ]
        for solo in seatIDs.indices {
            var ordered = Array(repeating: SeatOccupant.baseline, count: seatIDs.count)
            ordered[solo] = agent
            out.append(SeatAssignment(seatIDs: seatIDs, ordered: ordered))
        }
        // A one-seat panel's "all treated" IS its solo casting; emitting it
        // twice would mint two identical studies.
        if seatIDs.count > 1 {
            out.append(
                SeatAssignment(
                    seatIDs: seatIDs, ordered: Array(repeating: agent, count: seatIDs.count)))
        }
        return out
    }

    // MARK: - Legacy hoist

    /// What a legacy (fully bound) panel splits into, so the UI can migrate it
    /// LOUDLY rather than guessing.
    public struct LegacyHoist: Sendable {
        /// The panel with every binding removed — the reusable experiment.
        public let semantic: MultiAgentScenario
        /// The casting the legacy file was carrying, recovered as data.
        public let assignment: SeatAssignment
        /// The model the legacy file ran on. When seats disagreed, this is the
        /// one the majority of seats named — and `warnings` says so.
        public let modelID: String
        public let temperature: Double
        public let maxTokens: Int
        /// Non-empty means the hoist LOST information the researcher must
        /// decide about. Never resolved silently.
        public let warnings: [String]

        public init(
            semantic: MultiAgentScenario, assignment: SeatAssignment,
            modelID: String, temperature: Double, maxTokens: Int,
            warnings: [String]
        ) {
            self.semantic = semantic
            self.assignment = assignment
            self.modelID = modelID
            self.temperature = temperature
            self.maxTokens = maxTokens
            self.warnings = warnings
        }
    }

    /// Splits a legacy bound panel into its semantic half and the casting +
    /// sampling settings it was carrying.
    ///
    /// A DIVERGENT per-seat model set is surfaced as an explicit warning and
    /// never unified in silence. Single-model panels are the design — a
    /// cross-model panel confounds the intervention with the model that
    /// produced the text, and the run's `generations.jsonl` already labels
    /// every turn with the scenario's root model, so a mixed panel's records
    /// are mislabelled at the top level whatever we do here.
    public static func hoistLegacyScenario(path: String) throws -> LegacyHoist {
        let url = ExperimentStore.resolveProjectPath(path)
        guard let data = try? Data(contentsOf: url) else {
            throw ExperimentError(reason: "panel scenario not found: \(url.path)")
        }
        let scenario = try JSONDecoder().decode(MultiAgentScenario.self, from: data)
        guard !scenario.agents.isEmpty else {
            throw ExperimentError(
                reason: "panel '\(scenario.name)' declares no seats — nothing to hoist")
        }

        var warnings: [String] = []
        var occupants = assignment(in: scenario).occupants
        for seat in scenario.agents {
            guard let artifactPath = seat.variantArtifactPath, !artifactPath.isEmpty,
                seat.variantArtifactHash == nil
            else { continue }
            // A legacy seat may name an artifact with no hash. Pin the file as
            // it stands now rather than inventing one; an unreadable artifact
            // is reported, not papered over.
            occupants[seat.id] = .agent(
                name: seat.name,
                artifactPath: artifactPath,
                artifactHash: (try? ModelVariantStore.hash(
                    ModelVariantStore.absoluteURL(artifactPath))) ?? "")
            warnings.append(
                "seat '\(seat.name)' named an agent artifact with no "
                    + "pinned hash; it was hashed from the file as it "
                    + "stands today")
        }

        let seatModels = scenario.agents.map {
            (seat: $0.name, model: $0.baseModelID.trimmingCharacters(in: .whitespaces))
        }
        let declared = seatModels.map(\.model).filter { !$0.isEmpty }
        var counts: [String: Int] = [:]
        for model in declared { counts[model, default: 0] += 1 }
        if counts.count > 1 {
            let listed = seatModels
                .map { "\($0.seat) → \($0.model.isEmpty ? "(none)" : $0.model)" }
                .joined(separator: "; ")
            warnings.append(
                "this panel binds DIFFERENT models to different seats "
                    + "(\(listed)). A study has one base model, so hoisting "
                    + "collapses them — decide deliberately which model the "
                    + "panel runs on before instantiating; a cross-model panel "
                    + "confounds the intervention with the model.")
        }
        // Root model first (it is what the run records stamp), else the
        // majority seat model, ties broken by seat order for determinism.
        let rootModel = scenario.baseModelID.trimmingCharacters(in: .whitespaces)
        let majority = declared.max { lhs, rhs in
            let (l, r) = (counts[lhs] ?? 0, counts[rhs] ?? 0)
            return l == r ? false : l < r
        }
        let modelID = rootModel.isEmpty ? (majority ?? "") : rootModel
        if modelID.isEmpty {
            warnings.append(
                "this panel declares no model at all — pick the study's base "
                    + "model before instantiating")
        }

        return LegacyHoist(
            semantic: semanticForm(scenario),
            assignment: SeatAssignment(
                seatIDs: seatIDs(scenario), occupants: occupants),
            modelID: modelID,
            temperature: scenario.temperature,
            maxTokens: scenario.maxTokens,
            warnings: warnings)
    }
}

extension MultiAgentScenario {
    /// What `semanticForm` writes for `maxTokens`.
    ///
    /// The field is not optional and its absent-default DIVERGES across the
    /// engines (2048 on Swift, 512 on the server), so leaving it out would
    /// make the same semantic panel mean two different things. Writing the
    /// Swift default explicitly makes the byte content agree; compilation
    /// overwrites it with the study's declared budget regardless, so the value
    /// never reaches a run.
    static let semanticMaxTokensPlaceholder = 2048
}

import Foundation

/// The seat casting a multi-agent STUDY carries: which agent sits in which
/// seat of the scenario it runs.
///
/// Casting used to exist in exactly one place — the design instantiation table
/// — so a study that picked a semantic scenario directly in the Studies editor
/// had no way to fill its seats, and refused at run time
/// (`MultiAgentRunner.validate` rejects a seat with no model, deliberately: an
/// uncast panel must never quietly generate from whatever model happened to be
/// resident). This is the same machinery reached from the study editor.
///
/// Nothing here is new measurement plumbing. Reading a casting is a decode of
/// the file the study already pins (`PanelComposition.assignment(in:)`);
/// writing one is `PanelComposition.compileAndPin`, the identical call
/// `StudyTemplateStore.instantiate` makes for a `.seating` cell. A study cast
/// here and a study minted from a design are the same artifact, and the run
/// loop, the freeze packager and the Python engine see an ordinary bound
/// scenario in both cases.
public enum SeatCasting {

    // MARK: - What a study is holding

    /// The KIND of scenario a study pins, which is what decides whether its
    /// seats can be edited at all.
    public enum Form: Sendable, Equatable {
        /// A semantic scenario is chosen and no casting is compiled yet. The
        /// seats are editable, and the study is NOT runnable until they are
        /// saved — the scenario declares no model for any seat.
        case uncast
        /// The pinned scenario is a compiled casting: editable, and saving
        /// recompiles it.
        case cast
        /// The pinned scenario carries its own bindings and was not compiled
        /// from a semantic one — a panel hand-authored before the
        /// semantic/casting split. READ-ONLY here: its casting lives in the
        /// file, other studies may pin the same file, and re-casting it from a
        /// study editor would rewrite an input under them. The migration path
        /// (Panels → hoist) is the deliberate way to convert it.
        case legacyBound
    }

    public struct Seat: Identifiable, Sendable, Equatable {
        /// The scenario's agent `id` — what turns reference as
        /// `speakerAgentID`. Keying by name would let a rename silently re-cast
        /// the panel.
        public let id: String
        /// The role's display name, for the picker's label.
        public let name: String

        public init(id: String, name: String) {
            self.id = id
            self.name = name
        }
    }

    /// Everything the Seats section renders, read once from the manifest.
    public struct State: Sendable, Equatable {
        public var form: Form
        public var seats: [Seat]
        public var occupants: [String: SeatOccupant]
        /// The semantic scenario a save COMPILES against.
        ///
        /// For a cast study this is the stripped form of the file the study
        /// actually runs, not a re-read of the source: the compiled bytes are
        /// what the last run measured, and re-deriving the casting from a
        /// source file that has since been edited would silently change the
        /// experiment behind a "save".
        public var semantic: MultiAgentScenario
        /// Workspace-relative path of the semantic scenario this casting came
        /// from, when the study records one. Provenance — see
        /// `ExperimentManifest.multiAgentSemanticScenarioPath`.
        public var semanticPath: String?
        /// Non-blocking things the researcher must see (a source scenario that
        /// has drifted, a legacy file that cannot be cast here).
        public var advisories: [String]

        public init(
            form: Form, seats: [Seat], occupants: [String: SeatOccupant],
            semantic: MultiAgentScenario, semanticPath: String?,
            advisories: [String] = []
        ) {
            self.form = form
            self.seats = seats
            self.occupants = occupants
            self.semantic = semantic
            self.semanticPath = semanticPath
            self.advisories = advisories
        }

        public var seatIDs: [String] { seats.map(\.id) }

        /// The casting as the value `compile` takes. Seats nobody chose read as
        /// baseline: the pickers default there, and an all-baseline panel is
        /// the control composition rather than an absence.
        public var assignment: SeatAssignment {
            SeatAssignment(
                seatIDs: seatIDs,
                occupants: Dictionary(
                    uniqueKeysWithValues: seats.map {
                        ($0.id, occupants[$0.id] ?? .baseline)
                    }))
        }

        /// True when the seats may be edited and saved.
        public var isEditable: Bool { form != .legacyBound }
    }

    // MARK: - Reading

    /// The casting a study is currently carrying.
    ///
    /// `selected` is the scenario the editor's picker names, which may be
    /// ahead of what the manifest pins — the researcher picks a scenario, casts
    /// its seats and saves once. A selection of a SEMANTIC scenario therefore
    /// wins over the pinned file, and the pinned file still supplies the
    /// current occupants when it is a casting of that same scenario (same seat
    /// ids), so choosing the scenario a study already runs does not silently
    /// blank its cast.
    ///
    /// `overlay` is the editor's unsaved per-seat edits, keyed by seat id.
    /// Seats it does not name keep what the study is carrying; ids belonging to
    /// some other scenario are ignored, which is what makes switching scenarios
    /// in the picker read as an all-baseline panel rather than a half-cast one.
    public static func state(
        of manifest: ExperimentManifest,
        selected: (scenario: MultiAgentScenario, path: String)? = nil,
        overlay: [String: SeatOccupant] = [:]
    ) -> State? {
        guard manifest.studyKind == .multiAgent else { return nil }
        var advisories: [String] = []
        let pinned = loadScenario(manifest.multiAgentScenarioPath)

        // The pinned file, split into the two halves the section shows.
        var pinnedSemantic: MultiAgentScenario?
        var pinnedOccupants: [String: SeatOccupant] = [:]
        var pinnedForm: Form?
        if let pinned {
            if PanelComposition.isSemantic(pinned.scenario) {
                pinnedSemantic = pinned.scenario
                pinnedForm = .uncast
            } else if PanelComposition.isCompiledPath(pinned.path)
                || manifest.multiAgentSemanticScenarioPath != nil
            {
                pinnedSemantic = PanelComposition.semanticForm(pinned.scenario)
                pinnedOccupants = PanelComposition.assignment(in: pinned.scenario).occupants
                pinnedForm = .cast
            } else {
                pinnedSemantic = PanelComposition.semanticForm(pinned.scenario)
                pinnedOccupants = PanelComposition.assignment(in: pinned.scenario).occupants
                pinnedForm = .legacyBound
            }
        }

        // What the editor is pointing AT wins, when it points at something
        // castable.
        var semantic: MultiAgentScenario
        var semanticPath: String?
        var occupants: [String: SeatOccupant]
        var form: Form
        if let selected, PanelComposition.isSemantic(selected.scenario) {
            semantic = selected.scenario
            semanticPath = selected.path
            // A pinned casting of THIS scenario keeps its occupants; a casting
            // of a different one contributes nothing (its seat ids do not
            // occur here).
            occupants = pinnedForm == .legacyBound ? [:] : pinnedOccupants
            form = pinnedForm == .cast
                && PanelComposition.seatIDs(semantic)
                    == PanelComposition.seatIDs(pinnedSemantic ?? semantic)
                ? .cast : .uncast
        } else if let pinnedSemantic, let pinnedForm {
            semantic = pinnedSemantic
            semanticPath = manifest.multiAgentSemanticScenarioPath
            occupants = pinnedOccupants
            form = pinnedForm
        } else if let selected {
            // A bound scenario chosen in the picker and not yet pinned: the
            // same read-only rule as a pinned one.
            semantic = PanelComposition.semanticForm(selected.scenario)
            occupants = PanelComposition.assignment(in: selected.scenario).occupants
            form = .legacyBound
        } else {
            return nil
        }

        if form == .legacyBound {
            advisories.append(legacyAdvisory)
        }
        if form == .cast, let drift = sourceDriftAdvisory(manifest) {
            advisories.append(drift)
        }
        if pinned == nil, manifest.multiAgentScenarioPath?.isEmpty == false {
            advisories.append(
                "the scenario this study pins (\(manifest.multiAgentScenarioPath ?? "")) "
                    + "is missing — restore it, or pick a scenario and save the "
                    + "casting to pin a new one")
        }

        let seats = semantic.agents.map { Seat(id: $0.id, name: $0.name) }
        let seatIDs = Set(seats.map(\.id))
        for (seat, occupant) in overlay where seatIDs.contains(seat) {
            occupants[seat] = occupant
        }
        return State(
            form: form, seats: seats,
            occupants: occupants.filter { seatIDs.contains($0.key) },
            semantic: semantic, semanticPath: semanticPath,
            advisories: advisories)
    }

    // MARK: - Writing

    /// Compiles a casting against the study's OWN settings and pins the result
    /// as the study's scenario.
    ///
    /// The compile inputs are manifest fields — base model, temperature, max
    /// tokens — so this is also the recompile path: a study whose settings move
    /// after it was cast is re-compiled from the same casting rather than left
    /// pinning a scenario that binds the previous model. (Never the reverse:
    /// the manifest is the one place those three values are decided, and a
    /// compiled scenario that disagreed with it would be a second answer.)
    ///
    /// Refuses through `PanelComposition.compile`, which validates the bound
    /// scenario before anything is written — a casting that cannot run is a bug
    /// in the cast, and the queue is a bad place to learn it.
    /// - Parameter fileSlug: what the compiled file is named, before the
    ///   uniquing suffix. Defaults to the study's name — the value every UI
    ///   caller has always passed, so the default IS the historical behaviour
    ///   and the parameter only exists for `panel compile --file-slug`, where
    ///   a batch of castings into sibling drafts wants to name its own files.
    @discardableResult
    public static func compile(
        _ assignment: SeatAssignment,
        semantic: MultiAgentScenario,
        semanticPath: String?,
        into manifest: inout ExperimentManifest,
        fileSlug: String? = nil
    ) throws -> (path: String, hash: String) {
        let compiled = try PanelComposition.compileAndPin(
            semantic: semantic,
            assignment: assignment,
            modelID: manifest.modelID,
            temperature: manifest.temperature,
            maxTokens: manifest.maxTokens,
            fileSlug: fileSlug ?? manifest.name)
        manifest.multiAgentScenarioPath = compiled.path
        manifest.multiAgentScenarioHash = compiled.hash
        manifest.multiAgentSemanticScenarioPath = semanticPath
        manifest.multiAgentSemanticScenarioHash = semanticPath.flatMap {
            try? MultiAgentScenarioStore.hash(ExperimentStore.resolveProjectPath($0))
        }
        return (compiled.path, compiled.hash)
    }

    // MARK: - Occupants

    /// The occupant a library agent contributes to a seat, pinned by path and
    /// artifact hash exactly as a variant condition is.
    ///
    /// THE derivation, in one place. It used to be written out twice — once in
    /// the study editor (`ExperimentPanel.seatOccupant(forAgentID:)`) and once
    /// in the design instantiation table (`TemplateInstantiation.occupant(for:)`)
    /// — with a comment on each saying it matched the other. `panel compile`
    /// would have made three, so the two callers now call this and the CLI
    /// calls it through `occupant(artifactPath:)`: a seat cast from the app and
    /// a seat cast headlessly are the same value by construction rather than by
    /// review.
    ///
    /// An unreadable artifact yields an EMPTY hash rather than a throw, which
    /// is what both callers have always done: the pin records what could be
    /// read, and `compile` → `MultiAgentRunner.validate` is where an unusable
    /// seat is refused.
    public static func occupant(for record: ModelVariantRecord) -> SeatOccupant {
        .agent(
            name: record.artifact.name,
            artifactPath: ModelVariantStore.relativePath(for: record),
            artifactHash: (try? ModelVariantStore.hash(record.url)) ?? "")
    }

    /// The occupant a workspace-relative (or absolute) agent-artifact PATH
    /// contributes to a seat — the headless spelling of the picker.
    ///
    /// Refuses an absent or undecodable artifact rather than pinning an empty
    /// hash: the CLI caller typed this path, so "the file is not there" is the
    /// answer it needs, where a picker could only offer artifacts it had
    /// already read.
    public static func occupant(artifactPath: String) throws -> SeatOccupant {
        let url = ModelVariantStore.absoluteURL(artifactPath)
        guard let data = try? Data(contentsOf: url) else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "agent artifact not found: \(artifactPath)",
                repair: "agent artifacts live under runs/model-variants/ — mint "
                    + "one with steerlab-cli experiment promote <name> <concept>, "
                    + "or name an existing artifact's model-variant.json")
        }
        guard
            let artifact = try? JSONDecoder().decode(
                ModelVariantArtifact.self, from: data)
        else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "not an agent artifact: \(artifactPath)",
                repair: "name the model-variant.json of a promoted agent — "
                    + "steerlab-cli experiment promote <name> <concept> mints one")
        }
        return occupant(for: ModelVariantRecord(url: url, artifact: artifact))
    }

    /// The casting with every agent occupant dropped, for the one case that
    /// needs it: the study's base model changed, so no agent built on the old
    /// model is eligible for any seat.
    ///
    /// The same rule `saveProtocol` already applies to a comparison study's
    /// arms (a model change clears them), applied to seats. Silent re-use would
    /// bind adapters trained on one model into a panel running another.
    public static func resetToBaseline(_ assignment: SeatAssignment) -> SeatAssignment {
        SeatAssignment(
            seatIDs: assignment.seatIDs,
            ordered: Array(repeating: .baseline, count: assignment.seatIDs.count))
    }

    /// True when the casting binds at least one agent — the difference between
    /// a treated panel and the control composition.
    public static func isTreated(_ assignment: SeatAssignment) -> Bool {
        assignment.ordered.contains { $0 != .baseline }
    }

    // MARK: - Copy

    static let legacyAdvisory =
        "this scenario carries its own seat bindings (authored before seats "
        + "and scenarios were separated), so its casting is shown read-only "
        + "here. Migrate it in the Panels editor — that splits it into a "
        + "reusable scenario plus the casting it was carrying — then pick the "
        + "migrated scenario to cast seats from this study."

    /// Says the drift and what it means, without pretending it is a violation:
    /// the study runs the compiled bytes, which is the honest reading.
    private static func sourceDriftAdvisory(_ manifest: ExperimentManifest) -> String? {
        guard let path = manifest.multiAgentSemanticScenarioPath, !path.isEmpty
        else { return nil }
        guard let pinnedHash = manifest.multiAgentSemanticScenarioHash else { return nil }
        let url = ExperimentStore.resolveProjectPath(path)
        guard let live = try? MultiAgentScenarioStore.hash(url) else {
            return "the scenario this casting came from (\(path)) is no longer "
                + "in the workspace. The study still runs its compiled "
                + "casting; saving the seats recompiles from the compiled "
                + "scenario's own roles and turns."
        }
        guard live != pinnedHash else { return nil }
        return "'\(path)' has changed since this casting was compiled "
            + "(\(pinnedHash.prefix(12))… → \(live.prefix(12))…). This study "
            + "runs the COMPILED scenario, so nothing moved under it — pick "
            + "the scenario again and save the seats to adopt the new version."
    }

    // MARK: - Helpers

    private static func loadScenario(
        _ path: String?
    ) -> (scenario: MultiAgentScenario, path: String)? {
        guard let path, !path.isEmpty else { return nil }
        let url = ExperimentStore.resolveProjectPath(path)
        guard let data = try? Data(contentsOf: url),
            let scenario = try? JSONDecoder().decode(MultiAgentScenario.self, from: data)
        else { return nil }
        return (scenario, path)
    }
}

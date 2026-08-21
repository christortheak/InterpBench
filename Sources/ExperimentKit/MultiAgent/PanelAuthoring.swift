import Foundation

/// The authoring half of the semantic-panel design: what the Scenario tab is
/// allowed to write, what it must refuse to edit, and how a legacy bound panel
/// is split apart in front of the researcher rather than behind them.
///
/// `PanelComposition` owns the science-facing direction — semantic panel plus
/// casting COMPILES to a runnable scenario. This owns the inverse, editorial
/// direction: a scenario file is an ENVIRONMENT (roles, turn structure, case
/// materials, visibility), and every binding a file happens to carry —
/// model, sampling settings, seat→agent casting — belongs to the STUDY that
/// runs it. The two rules below are the whole contract:
///
///   1. A save writes the semantic form, always. Not "the editor stops
///      offering bindings" — that is a UI promise, and UI promises leak. The
///      bytes are normalised on the way to disk, so a binding that survived in
///      editor state (loaded from an old file, set by a future code path)
///      still cannot reach the file.
///   2. A file that carries bindings is not silently edited. Opening one puts
///      the editor in `legacyBound` mode; the researcher migrates it — loudly,
///      seeing exactly what was extracted — or leaves it alone.
///
/// Nothing here changes what RUNS. A frozen study pinning a bound panel by
/// path + hash keeps resolving to the same bytes, because migration never
/// rewrites the file it read (see `migrateLegacyPanel`).
public enum PanelAuthoring {

    // MARK: - Mode

    /// What the editor is looking at.
    public enum Mode: Sendable, Equatable {
        /// An environment: safe to edit, safe to save.
        case semantic
        /// A pre-split file that still embeds model/sampling/casting. The
        /// editor renders it read-only until it is migrated.
        case legacyBound
    }

    /// Does this scenario carry anything the editor no longer owns?
    ///
    /// The test is deliberately the exact one, not an approximation of it: a
    /// file carries bindings iff it DIFFERS FROM ITS OWN SEMANTIC FORM. That
    /// keeps the banner condition and the save normalisation defined by a
    /// single function — if `semanticForm` ever learns to strip one more
    /// field, the migration banner learns about it in the same commit, and
    /// there is no window where a save would quietly drop something the editor
    /// never warned about.
    ///
    /// (`PanelComposition.isSemantic` answers a narrower question — do the
    /// SEATS name a model — which is the one the run path cares about. A root
    /// `baseModelID`, a stashed temperature or a seat-level agent reference
    /// with no seat model would slip past it and then vanish on the next
    /// save.)
    public static func carriesBindings(_ scenario: MultiAgentScenario) -> Bool {
        PanelComposition.semanticForm(scenario) != scenario
    }

    public static func mode(for scenario: MultiAgentScenario) -> Mode {
        carriesBindings(scenario) ? .legacyBound : .semantic
    }

    // MARK: - Save normalisation

    /// The scenario a save writes, assembled from the editor's semantic
    /// surface and then normalised through `PanelComposition.semanticForm`.
    ///
    /// The normalisation is not redundant with the editor no longer offering
    /// bindings. It is the enforcement point: this is the only expression the
    /// panel store is ever handed, so "a saved panel is semantic" is a
    /// property of the code path rather than a property of every future view
    /// that touches `agents`.
    public static func scenarioForSave(
        name: String,
        description: String,
        sharedMaterials: String,
        agents: [MultiAgentScenario.Agent],
        turns: [MultiAgentScenario.Turn]
    ) -> MultiAgentScenario {
        PanelComposition.semanticForm(
            MultiAgentScenario(
                name: name.trimmingCharacters(in: .whitespacesAndNewlines),
                description: description,
                // Written empty on purpose and then normalised anyway: the
                // initialiser requires the key, the semantic form defines its
                // value.
                baseModelID: "",
                sharedMaterials: sharedMaterials,
                agents: agents,
                turns: turns))
    }

    // MARK: - Rehearsal

    /// The scenario an AD-HOC play-through from the authoring tab runs.
    ///
    /// A rehearsal needs the three things the environment deliberately does
    /// not carry, so the tab asks for them as throwaway settings and compiles
    /// them in here. Every seat is cast BASELINE: casting is the study's
    /// decision, and a rehearsal that quietly seated an agent would be exactly
    /// the confusion this rework exists to remove — a transcript that looks
    /// like a measurement and is not one.
    ///
    /// It goes through `PanelComposition.compile`, not a private shortcut, so
    /// a rehearsal exercises the same binding path a measured run does.
    public static func rehearsalScenario(
        _ semantic: MultiAgentScenario,
        modelID: String,
        temperature: Double,
        maxTokens: Int
    ) throws -> MultiAgentScenario {
        let seats = PanelComposition.seatIDs(semantic)
        return try PanelComposition.compile(
            semantic: semantic,
            assignment: SeatAssignment(
                seatIDs: seats,
                ordered: Array(repeating: .baseline, count: seats.count)),
            modelID: modelID,
            temperature: temperature,
            maxTokens: maxTokens)
    }

    // MARK: - Contract authoring (spec §1–2)

    /// One earlier turn's output, as the contract-inputs picker needs it.
    ///
    /// A contract's `inputs` are output LABELS, and a label is meaningless on
    /// its own in a picker — `r1_judge-2` says nothing about which turn wrote
    /// it or who spoke. The label is what the file stores; the title and
    /// speaker are what makes the choice reviewable.
    public struct AvailableInput: Sendable, Equatable, Identifiable {
        public let label: String
        public let turnTitle: String
        public let speakerName: String

        public init(label: String, turnTitle: String, speakerName: String) {
            self.label = label
            self.turnTitle = turnTitle
            self.speakerName = speakerName
        }

        public var id: String { label }
        public var summary: String { "\(label) — \(turnTitle) (\(speakerName))" }
    }

    /// Output labels a contract turn may name as inputs: the ones produced by
    /// turns STRICTLY EARLIER in the script, in script order.
    ///
    /// Same rule as `MultiAgentRunner.validate`'s third refusal, computed from
    /// the same normalization (`turn_<n>` when a turn declares no label), so
    /// the picker cannot offer something the validator would then refuse.
    public static func availableInputs(
        in scenario: MultiAgentScenario, before turnID: String
    ) -> [AvailableInput] {
        let names = Dictionary(
            scenario.agents.map { ($0.id, $0.name) }, uniquingKeysWith: { first, _ in first })
        let limit = scenario.turns.firstIndex { $0.id == turnID } ?? scenario.turns.count
        var out: [AvailableInput] = []
        for (index, turn) in scenario.turns.enumerated() where index < limit {
            out.append(
                AvailableInput(
                    label: MultiAgentRunner.normalizedOutputLabel(turn: turn, index: index),
                    turnTitle: turn.title,
                    speakerName: names[turn.speakerAgentID] ?? turn.speakerAgentID))
        }
        return out
    }

    /// The canonical layout a contract turn will render, block by block, as
    /// display lines.
    ///
    /// Read-only ON PURPOSE. A contract turn declares its CONTENT and the
    /// renderer owns its ARRANGEMENT — that split is the whole point of the
    /// contract — so the editor must show the arrangement rather than offer to
    /// edit it. Derived from the turn's own flags and the scenario it sits in,
    /// never from a hardcoded picture of a typical panel: a summary that could
    /// disagree with the renderer would be worse than none.
    public static func contractLayoutSummary(
        scenario: MultiAgentScenario, turn: MultiAgentScenario.Turn
    ) -> [String] {
        guard let contract = turn.contract else { return [] }
        let speaker = scenario.agents.first { $0.id == turn.speakerAgentID }
        let speakerName = speaker?.name ?? turn.speakerAgentID
        let colleagues = scenario.agents.filter { $0.id != turn.speakerAgentID }.map(\.name)
        var out: [String] = []

        var opener = "1. Identity — \"You are \(speakerName)"
        if let role = MultiAgentScenario.Agent.normalizedRole(speaker?.role) {
            opener += ", \(role)"
        }
        opener += ".\""
        if colleagues.isEmpty {
            opener += " (one seat: no roll call)"
        } else {
            opener += " + the other participants"
        }
        opener +=
            contract.stage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? " (no stage sentence)"
            : " + your stage sentence"
        out.append(opener)

        out.append(
            turn.includeScenarioMaterials
                ? "2. Shared materials, fenced as \"\(contract.materialsTitle)\", then the "
                    + "\"every participant has read it\" line"
                : "2. Shared materials — OFF for this turn (nothing rendered)")

        // Attribution comes from the PRODUCING turn's speaker, resolved the way
        // the renderer resolves it, so the summary cannot claim an ownership
        // the rendered prompt will not show.
        let producers = MultiAgentRunner.earlierProducers(scenario: scenario, before: turn)
        let names = Dictionary(
            scenario.agents.map { ($0.id, $0.name) }, uniquingKeysWith: { first, _ in first })
        if contract.inputs.isEmpty {
            out.append("3. Earlier outputs — none declared")
        } else {
            let described = contract.inputs.map { label -> String in
                guard let producer = producers[label] else {
                    return "\(label) (no earlier turn produces this — validate refuses)"
                }
                guard producer.speakerAgentID != turn.speakerAgentID else {
                    return "\(label) (your own earlier output)"
                }
                return "\(label) (\(names[producer.speakerAgentID] ?? producer.speakerAgentID))"
            }
            out.append("3. Earlier outputs, fenced and attributed: " + described.joined(separator: ", "))
        }

        out.append(
            turn.includeSpeakerContext
                ? "4. Routed transcript so far, fenced (when this seat has one)"
                : "4. Routed transcript — OFF for this turn (nothing rendered)")
        out.append("5. \"===== YOUR TASK =====\" then \"You are \(speakerName).\" + your task")
        if contract.ownVoice, scenario.agents.count > 1 {
            out.append(
                "6. Own-voice constraint naming \(MultiAgentRunner.nameList(colleagues, conjunction: "or"))")
        } else if contract.ownVoice {
            out.append("6. Own-voice constraint — omitted: a one-seat panel has no one to speak for")
        } else {
            out.append("6. Own-voice constraint — OFF for this turn")
        }
        out.append(
            contract.format.isEmpty
                ? "7. Format block — none declared"
                : "7. Your format block, verbatim")
        out.append("8. Closing reminder: \"you are \(speakerName)… and no one else.\"")
        return out
    }

    /// The prompt this turn would render, through the RUNNER's own renderer.
    ///
    /// Not a preview implementation: the actual `MultiAgentRunner.renderPrompt`,
    /// which is pure over strings. An editor preview that re-derived the layout
    /// would be a second renderer, and the day the two disagreed the preview
    /// would be the one the researcher believed.
    ///
    /// The two runtime-only inputs — earlier outputs and the routed transcript
    /// — are stood in for with obviously-placeholder text, since neither
    /// exists until a run: a preview that showed them empty would silently drop
    /// exactly the blocks the contract exists to arrange.
    public static func previewPrompt(
        scenario: MultiAgentScenario,
        turn: MultiAgentScenario.Turn,
        placeholderRuntimeText: Bool = true
    ) -> String {
        let speakerName =
            scenario.agents.first { $0.id == turn.speakerAgentID }?.name ?? turn.speakerAgentID
        var outputs: [String: String] = [:]
        if placeholderRuntimeText {
            for input in availableInputs(in: scenario, before: turn.id) {
                outputs[input.label] = "«\(input.label): what \(input.speakerName) wrote in "
                    + "\"\(input.turnTitle)\" appears here»"
            }
        }
        let context =
            placeholderRuntimeText
            ? "«the turns routed to \(speakerName) so far appear here»"
            : ""
        return MultiAgentRunner.renderPrompt(
            scenario: scenario,
            turn: turn,
            speakerName: speakerName,
            speakerContext: context,
            outputsByLabel: outputs)
    }

    /// A turn converted between the two renderers, plus what the conversion
    /// did to the author's text.
    ///
    /// The notes are the point. A turn has exactly one renderer, so switching
    /// necessarily moves text between fields that mean different things, and
    /// doing that silently is how an author loses a paragraph they wrote.
    public struct TurnConversion: Sendable, Equatable {
        public let turn: MultiAgentScenario.Turn
        public let notes: [String]

        public init(turn: MultiAgentScenario.Turn, notes: [String]) {
            self.turn = turn
            self.notes = notes
        }
    }

    /// Template turn → contract turn.
    ///
    /// The template body becomes the contract's `task`, minus the layout
    /// placeholders — those name slots the contract renderer fills in its own
    /// canonical positions, and `validate` refuses them inside contract text
    /// (spec §1.3). Everything else is carried verbatim: this is a conversion,
    /// not a rewrite, and the author reads the result in the task field.
    ///
    /// A turn that already carries a contract is returned unchanged.
    public static func asContractTurn(_ turn: MultiAgentScenario.Turn) -> TurnConversion {
        if turn.contract != nil { return TurnConversion(turn: turn, notes: []) }
        let (task, dropped) = contractTaskDraft(from: turn.promptTemplate)
        var converted = turn
        converted.promptTemplate = ""
        converted.contract = TurnContract(task: task)
        var notes: [String] = []
        if !dropped.isEmpty {
            notes.append(
                "removed \(dropped.joined(separator: ", ")) from the task text — a contract "
                    + "turn gets the materials, the transcript and every declared input in "
                    + "the renderer's own canonical slots, and a hand-placed copy is refused")
        }
        if task.isEmpty {
            notes.append("the task is empty — write what this turn asks for; validate refuses a contract with no task")
        }
        return TurnConversion(turn: converted, notes: notes)
    }

    /// Contract turn → free-template turn.
    ///
    /// `stage`, `task` and `format` are reassembled into the template body in
    /// render order so no authored sentence is lost. `inputs` cannot survive —
    /// a template reads earlier outputs through `{{outputs.label}}` — so they
    /// are named in the notes rather than dropped in silence.
    public static func asTemplateTurn(_ turn: MultiAgentScenario.Turn) -> TurnConversion {
        guard let contract = turn.contract else { return TurnConversion(turn: turn, notes: []) }
        var converted = turn
        converted.contract = nil
        converted.promptTemplate = [contract.stage, contract.task, contract.format]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        var notes: [String] = [
            "the canonical layout is gone: a template turn renders exactly what you write, "
                + "and the identity opener, the fences, the own-voice constraint and the "
                + "closing reminder are no longer added for you",
        ]
        if !contract.inputs.isEmpty {
            notes.append(
                "the declared inputs (\(contract.inputs.joined(separator: ", "))) are not "
                    + "carried — reference them as {{outputs.<label>}} if this turn still needs them")
        }
        if converted.promptTemplate.isEmpty {
            notes.append("the template is empty — validate refuses a template turn with no body")
        }
        return TurnConversion(turn: converted, notes: notes)
    }

    /// A template body reduced to contract-legal task text: the layout
    /// placeholders removed, and lines left blank by their removal dropped.
    ///
    /// Internal seam so the conversion's text handling is testable on its own.
    static func contractTaskDraft(from template: String) -> (task: String, dropped: [String]) {
        var dropped: [String] = []
        var text = template
        for placeholder in ["{{scenario.materials}}", "{{agent.context}}"]
        where text.contains(placeholder) {
            dropped.append(placeholder)
            text = text.replacingOccurrences(of: placeholder, with: "")
        }
        for label in MultiAgentRunner.outputReferences(in: text) {
            let placeholder = "{{outputs.\(label)}}"
            guard text.contains(placeholder) else { continue }
            dropped.append(placeholder)
            text = text.replacingOccurrences(of: placeholder, with: "")
        }
        // Removing a placeholder leaves the line it sat on blank; two of them
        // in a row leave a blank run. Collapse both, so the author reads their
        // own prose rather than the holes the substitution made.
        var lines: [String] = []
        for line in text.components(separatedBy: "\n") {
            let blank = line.trimmingCharacters(in: .whitespaces).isEmpty
            if blank, lines.last?.trimmingCharacters(in: .whitespaces).isEmpty ?? true { continue }
            lines.append(blank ? "" : line)
        }
        return (
            lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines),
            dropped
        )
    }

    // MARK: - Migration

    /// One seat's recovered casting, in display terms.
    public struct SeatBinding: Sendable, Equatable {
        public let seatID: String
        public let seatName: String
        /// The agent artifact's name, or `nil` for an unsteered seat.
        public let agentName: String?
        public let artifactPath: String?

        public var summary: String {
            guard let agentName else { return "\(seatName) → baseline (no agent)" }
            return "\(seatName) → \(agentName)"
        }
    }

    /// What a migration produced, for the UI to show verbatim.
    public struct Migration: Sendable {
        /// Workspace-relative path of the NEW semantic file.
        public let semanticPath: String
        public let semanticFileName: String
        /// The file the bindings were read out of — untouched, still on disk.
        public let sourcePath: String
        public let scenario: MultiAgentScenario
        /// Extracted study parameters. These now belong to a study/template.
        public let modelID: String
        public let temperature: Double
        public let maxTokens: Int
        public let seatBindings: [SeatBinding]
        /// Straight from `PanelComposition.LegacyHoist` — surfaced word for
        /// word, never summarised away.
        public let warnings: [String]
    }

    /// Splits a bound panel into an environment file plus the study parameters
    /// it was carrying, and writes the environment as a NEW panel.
    ///
    /// New file, not an in-place rewrite, and the reason is a pin: a study
    /// names its panel by `multiAgentScenarioPath` + `multiAgentScenarioHash`,
    /// and the run path REFUSES a scenario whose bytes moved since pinning
    /// ("changed since pinning"). Rewriting the file in place would therefore
    /// break every study — frozen or draft — that already points at it, and it
    /// would do so at run start on a compute node rather than here. Leaving
    /// the source bytes alone means the old studies keep loading and running
    /// exactly as they did, and the migrated environment is a new input a NEW
    /// study opts into.
    ///
    /// The new file keeps the panel's NAME (the name is part of the
    /// environment, and changing it would make the migrated panel look like a
    /// different experiment); the store's own uniquing gives it a distinct
    /// file name. The caller is expected to say all of this out loud.
    public static func migrateLegacyPanel(at url: URL) throws -> Migration {
        let sourcePath = FineTuneStore.relativePath(for: url)
        let hoist = try PanelComposition.hoistLegacyScenario(path: sourcePath)
        guard carriesBindings(try loadScenario(at: url)) else {
            throw ExperimentError(
                reason: "panel '\(hoist.semantic.name)' already carries no "
                    + "bindings — there is nothing to migrate")
        }

        let record = try MultiAgentScenarioStore.save(hoist.semantic)
        let seatNames = Dictionary(
            hoist.semantic.agents.map { ($0.id, $0.name) },
            uniquingKeysWith: { first, _ in first })
        let bindings = hoist.assignment.seatIDs.map { seatID -> SeatBinding in
            let name = seatNames[seatID] ?? seatID
            switch hoist.assignment[seatID] {
            case .agent(let agentName, let path, _):
                return SeatBinding(
                    seatID: seatID, seatName: name,
                    agentName: agentName, artifactPath: path)
            case .baseline, nil:
                return SeatBinding(
                    seatID: seatID, seatName: name,
                    agentName: nil, artifactPath: nil)
            }
        }
        return Migration(
            semanticPath: FineTuneStore.relativePath(for: record.url),
            semanticFileName: record.url.lastPathComponent,
            sourcePath: sourcePath,
            scenario: hoist.semantic,
            modelID: hoist.modelID,
            temperature: hoist.temperature,
            maxTokens: hoist.maxTokens,
            seatBindings: bindings,
            warnings: hoist.warnings)
    }

    private static func loadScenario(at url: URL) throws -> MultiAgentScenario {
        guard let data = try? Data(contentsOf: url) else {
            throw ExperimentError(reason: "panel scenario not found: \(url.path)")
        }
        return try JSONDecoder().decode(MultiAgentScenario.self, from: data)
    }
}

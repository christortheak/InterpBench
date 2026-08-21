import Foundation
import Observation
import SteeringKit

@Observable @MainActor
public final class MultiAgentPanel {
    public struct LiveTurn: Identifiable, Sendable, Equatable {
        public var id: String { "\(index)-\(title)-\(speaker)" }
        public let index: Int
        public let title: String
        public let speaker: String
        public var output: String

        public var summary: String {
            "\(index). \(title) — \(speaker)"
        }
    }

    public internal(set) weak var host: ChatService?

    public private(set) var scenarios: [MultiAgentScenarioRecord] = []
    /// Panel files the scan could not read. Carried into the picker as
    /// disabled rows: a file that will not decode must be VISIBLE and say why,
    /// never silently absent (see `PanelFileIssue`).
    public private(set) var brokenScenarios: [PanelFileIssue] = []
    /// The protocol-template library (`prompts/panels/templates/`).
    public private(set) var protocolTemplates: [ScenarioProtocolTemplateRecord] = []
    public private(set) var brokenProtocolTemplates: [PanelFileIssue] = []
    public var selectedScenarioID: MultiAgentScenarioRecord.ID? {
        didSet {
            if oldValue != selectedScenarioID {
                loadEditorFromSelection()
            }
        }
    }
    public private(set) var status: String?
    public private(set) var isRunning = false
    public private(set) var liveRunDirectory: String?
    public private(set) var liveRunFailure: String?
    public private(set) var liveRunWarnings: [MultiAgentRunWarning] = []
    public private(set) var liveActiveTurn: LiveTurn?
    public private(set) var liveTurnResults: [MultiAgentTurnResult] = []
    private var runTask: Task<Void, Never>?
    private var stopRequested = false

    // MARK: - The environment (what a save writes)

    public var name = "three-agent-panel"
    public var scenarioDescription = ""
    public var sharedMaterials = ""
    public var agents: [MultiAgentScenario.Agent] = []
    public var turns: [MultiAgentScenario.Turn] = []

    /// One element of a protocol's materials checklist, as the editor holds
    /// it: the author's phrase, plus whether the researcher has confirmed the
    /// materials cover it.
    public struct ChecklistItem: Identifiable, Sendable, Equatable {
        public let text: String
        public var confirmed: Bool

        public init(text: String, confirmed: Bool = false) {
            self.text = text
            self.confirmed = confirmed
        }

        public var id: String { text }
    }

    /// The checklist of the protocol template this draft was started from.
    ///
    /// Editor state, not scenario content: it is never written to the panel
    /// file. A protocol's checklist describes what its materials must contain,
    /// and once the materials are written the concrete panel has no use for
    /// it — carrying it in the file would put an authoring note inside a
    /// hash-pinned measurement input.
    public private(set) var materialsChecklist: [ChecklistItem] = []
    /// The template this draft came from, for display only.
    public private(set) var sourceProtocolTemplate: String?
    /// Soft warnings from the last save. Shown; never a blocker.
    public private(set) var checklistWarnings: [String] = []

    // MARK: - Rehearsal settings (never saved)

    /// Model, sampling and budget for an AD-HOC play-through from this tab.
    ///
    /// These are NOT scenario content and never reach the panel file — a
    /// measured study declares its own, and the whole point of the semantic
    /// form is that one environment is reusable across models and settings.
    /// They exist only because a rehearsal has to bind something to generate,
    /// and `PanelAuthoring.rehearsalScenario` compiles them in at run time.
    public var rehearsalModelID = ChatService.availableModels.first?.id ?? ""
    public var rehearsalTemperature: Double = 0
    public var rehearsalMaxTokens: Int = 2048

    /// The bound file the editor is showing, when the selection is a
    /// pre-split panel. Non-nil ⇒ `authoringMode == .legacyBound` and the
    /// editor is read-only.
    public private(set) var legacySource: MultiAgentScenario?
    /// The last migration's report, kept for display until the researcher
    /// moves on.
    public private(set) var migration: PanelAuthoring.Migration?

    /// Read-only until migrated: a bound file is somebody's pinned input, and
    /// editing it in place is how a frozen study loses its bytes.
    public var authoringMode: PanelAuthoring.Mode {
        legacySource == nil ? .semantic : .legacyBound
    }

    public var isLegacyBound: Bool { authoringMode == .legacyBound }

    public var selectedScenario: MultiAgentScenarioRecord? {
        scenarios.first { $0.id == selectedScenarioID }
    }

    /// Is the app's active compute workspace a server (cluster) rather than
    /// local MLX? Same derivation as `ExperimentPanel.isServerWorkspace`.
    public var isServerWorkspace: Bool {
        if case .server = host?.cluster.activeWorkspace { return true }
        return false
    }

    /// Model options for the REHEARSAL only — the ACTIVE workspace's
    /// inventory, through the one shared rule
    /// (`WorkspaceScoping.studyBaselineModelOptions`). A server target offers
    /// the SERVER's installed models; offering the local MLX tiers there was
    /// the reported bug — their repo ids name weights the cluster cannot
    /// load. A current selection outside the returned inventory is the view's
    /// "(not installed)" row, not an extra pickable option here.
    ///
    /// The scenario itself names no model: a measured study picks one, and
    /// this list is only what an ad-hoc play-through can bind.
    public var modelOptions: [String] {
        WorkspaceScoping.studyBaselineModelOptions(
            workspaceIsServer: isServerWorkspace,
            localOptions: localModelOptions,
            serverOptions: host?.workspaceModelOptions ?? [])
    }

    private var localModelOptions: [String] {
        var seen = Set<String>()
        var options: [String] = []
        func append(_ value: String?) {
            guard let value else { return }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !seen.contains(trimmed) else { return }
            seen.insert(trimmed)
            options.append(trimmed)
        }
        append(rehearsalModelID)
        append(host?.selectedModelID)
        for model in ChatService.availableModels.map(\.id) { append(model) }
        for model in SteeredContainerLoader.localModelIDs() { append(model) }
        for agent in ModelVariantStore.scan() { append(agent.artifact.baseModelID) }
        return options
    }

    // Seat→agent casting is deliberately absent from this panel. Which agent
    // sits in which seat is a STUDY parameter: the Studies surface builds the
    // casting and `PanelComposition.compile` binds it, so the authoring tab
    // has no agent picker to scope and no variant list to filter.

    /// Where the Run button's work would execute for the active workspace.
    /// Scenario runs are in-process MLX today; under a server target the view
    /// disables Run and shows `localOnlyCaption` rather than silently
    /// computing on the Mac.
    public var runRoute: WorkspaceScoping.Route {
        WorkspaceScoping.route(
            for: .multiAgentScenario,
            workspace: host?.cluster.activeWorkspace ?? .local)
    }

    /// Non-nil when Run is unavailable in this workspace; the string is the
    /// user-facing reason and names the next action.
    public var runUnavailableReason: String? {
        if case .localOnly(let caption) = runRoute { return caption }
        return nil
    }

    public init() {
        refresh()
        if agents.isEmpty {
            resetToPanelTemplate()
        }
    }

    public func refresh() {
        let panels = MultiAgentScenarioStore.scanAll()
        scenarios = panels.records
        brokenScenarios = panels.unreadable
        let templates = ScenarioProtocolTemplateStore.scanAll()
        protocolTemplates = templates.records
        brokenProtocolTemplates = templates.unreadable
        if let selectedScenarioID,
            !scenarios.contains(where: { $0.id == selectedScenarioID })
        {
            self.selectedScenarioID = nil
        }
    }

    /// The scanned record for a file we just wrote, matched by RESOLVED path.
    ///
    /// A record's id is its `url.path`, and the two sides disagree whenever
    /// the workspace root sits behind a symlink: `scan` reads through
    /// `contentsOfDirectory`, which resolves (`/private/var/…`), while the
    /// store's writer builds its URL by appending to the configured root
    /// (`/var/…`). Selecting by the writer's id then silently matched
    /// nothing — the panel a researcher had just saved came back
    /// unselected, and the editor's saved-snapshot bookkeeping went with it.
    private func recordID(forFileAt url: URL) -> MultiAgentScenarioRecord.ID? {
        let target = url.resolvingSymlinksInPath().standardizedFileURL.path
        return scenarios.first {
            $0.url.resolvingSymlinksInPath().standardizedFileURL.path == target
        }?.id
    }

    /// Snapshot of the editor as last loaded or saved. `nil` means "a fresh
    /// draft that has never been saved", which is dirty as soon as it differs
    /// from the template it started from.
    private var savedSnapshot: MultiAgentScenario?

    /// Does the editor hold work that is not on disk?
    ///
    /// `newScenario()` used to overwrite everything unconditionally, so a
    /// panel a researcher had just built — agents, turns, materials — vanished
    /// with no warning and no undo. That is data loss, not a papercut.
    public var hasUnsavedChanges: Bool {
        guard let savedSnapshot else {
            // Never saved: dirty once it carries anything a template would not.
            return !agents.isEmpty || !turns.isEmpty
                || !sharedMaterials.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return scenarioFromEditor() != savedSnapshot
    }

    /// Start a fresh draft. Refuses when the editor holds unsaved work unless
    /// `discardingChanges` is set — the view asks first.
    @discardableResult
    public func newScenario(discardingChanges: Bool) -> Bool {
        guard discardingChanges || !hasUnsavedChanges else {
            status = "this panel has unsaved changes — save it first, or "
                + "confirm discarding them"
            return false
        }
        newScenario()
        return true
    }

    public func newScenario() {
        selectedScenarioID = nil
        legacySource = nil
        migration = nil
        name = "three-agent-panel"
        scenarioDescription = "Scripted multi-agent deliberation scenario."
        // Rehearsal default: something the ACTIVE workspace can actually load
        // (the local selection is only a sensible seed under a local target).
        // The scenario itself stays model-free.
        rehearsalModelID =
            isServerWorkspace
            ? (modelOptions.first ?? "")
            : (host?.selectedModelID ?? ChatService.availableModels.first?.id ?? "")
        sharedMaterials = ""
        rehearsalTemperature = 0
        rehearsalMaxTokens = 2048
        resetToPanelTemplate()
        savedSnapshot = nil
        materialsChecklist = []
        sourceProtocolTemplate = nil
        checklistWarnings = []
        turnNotices = [:]
        status = "started a new multi-agent scenario draft"
    }

    // MARK: - Protocol templates

    /// Starts a draft from a PROTOCOL: the seats, turn script, routing, caps
    /// and endpoint declarations arrive filled in; the shared materials do
    /// not.
    ///
    /// Refuses on unsaved work exactly like `newScenario(discardingChanges:)`
    /// — this is the same "throw away what is in the editor" action wearing a
    /// different label, and it would be a strange place to lose a panel.
    ///
    /// The NAME is left blank on purpose. Prefilling it with the protocol's
    /// name would produce a library where the protocol and the first case that
    /// used it are called the same thing, and Save is already disabled until a
    /// name is typed — so the one decision instantiation actually requires is
    /// the one it asks for.
    @discardableResult
    public func newScenarioFromProtocolTemplate(
        _ record: ScenarioProtocolTemplateRecord, discardingChanges: Bool = false
    ) -> Bool {
        guard discardingChanges || !hasUnsavedChanges else {
            status = "this panel has unsaved changes — save it first, or "
                + "confirm discarding them"
            return false
        }
        let template = record.template
        selectedScenarioID = nil
        legacySource = nil
        migration = nil
        name = ""
        scenarioDescription = template.templateDescription
        sharedMaterials = ""
        agents = template.protocolBody.agents
        turns = template.protocolBody.turns
        rehearsalTemperature = 0
        rehearsalMaxTokens = 2048
        savedSnapshot = nil
        turnNotices = [:]
        materialsChecklist = template.materialsChecklist.map { ChecklistItem(text: $0) }
        sourceProtocolTemplate = template.name
        checklistWarnings = ScenarioProtocolTemplateStore.review(
            checklist: template.materialsChecklist,
            sharedMaterials: "",
            confirmed: []
        ).warnings
        status = "started a scenario from protocol '\(template.name)' — name it "
            + "for the case it runs, then write the shared materials"
        return true
    }

    /// Ticks (or unticks) one checklist element. Confirmation is the
    /// researcher's statement about their own prose — nothing here inspects
    /// the materials, and nothing here blocks a save.
    public func setChecklistItem(_ text: String, confirmed: Bool) {
        guard let index = materialsChecklist.firstIndex(where: { $0.text == text })
        else { return }
        materialsChecklist[index].confirmed = confirmed
        checklistWarnings = currentChecklistReview().warnings
    }

    /// Mints a protocol template from what the editor is holding: seats, turn
    /// script, caps and endpoints kept verbatim, materials dropped.
    ///
    /// Works on an unsaved draft as well as a saved panel — the protocol is
    /// the editor's content, and requiring a save first would mean a
    /// researcher who has just built a turn script cannot keep the script
    /// without first inventing a case to attach it to.
    @discardableResult
    public func saveAsProtocolTemplate(
        named templateName: String, checklist: [String]
    ) -> Bool {
        do {
            let template = ScenarioProtocolTemplateStore.templateFromScenario(
                scenarioFromEditor(),
                named: templateName,
                materialsChecklist: checklist)
            let record = try ScenarioProtocolTemplateStore.save(template)
            refresh()
            // A checklist element the researcher had already ticked stays
            // ticked: minting a protocol out of the draft they are editing
            // must not quietly reset their own confirmations.
            let alreadyConfirmed = Set(materialsChecklist.filter(\.confirmed).map(\.text))
            materialsChecklist = record.template.materialsChecklist.map { text in
                ChecklistItem(text: text, confirmed: alreadyConfirmed.contains(text))
            }
            sourceProtocolTemplate = record.template.name
            status = "saved protocol template '\(record.template.name)' "
                + "(\(record.url.lastPathComponent)) — its seats and turn script "
                + "are reusable; its materials are empty by design"
            return true
        } catch {
            status = "\(error)"
            return false
        }
    }

    /// The current editor state reviewed against the live checklist.
    private func currentChecklistReview() -> ScenarioProtocolTemplateStore.ChecklistReview {
        ScenarioProtocolTemplateStore.review(
            checklist: materialsChecklist.map(\.text),
            sharedMaterials: sharedMaterials,
            confirmed: Set(materialsChecklist.filter(\.confirmed).map(\.text)))
    }

    /// Writes the editor's environment to disk.
    ///
    /// Always the SEMANTIC form (`PanelAuthoring.scenarioForSave`), and it
    /// refuses outright while a bound file is selected — overwriting one in
    /// place would move bytes a study pins by hash.
    public func saveScenario() {
        guard !isLegacyBound else {
            status = "this panel still embeds model and agent bindings — "
                + "migrate it first; saving over it would change bytes a "
                + "study may already pin"
            return
        }
        do {
            let scenario = scenarioFromEditor()
            let record: MultiAgentScenarioRecord
            if let selected = selectedScenario {
                // No timestamps to carry across: the panel file is the recipe
                // only (B2), so re-saving an unedited panel is a byte-for-byte
                // no-op rather than gratuitous hash drift on a pinned input.
                record = try MultiAgentScenarioStore.update(scenario, at: selected.url)
            } else {
                record = try MultiAgentScenarioStore.save(scenario)
            }
            refresh()
            selectedScenarioID = recordID(forFileAt: record.url) ?? record.id
            savedSnapshot = scenario
            // SOFT: the file is already written. A checklist somebody wrote
            // last month must not be able to refuse this month's case, so the
            // review is reported after the fact and never gates the write.
            let review = currentChecklistReview()
            checklistWarnings = materialsChecklist.isEmpty && !sharedMaterials.isEmpty
                ? []
                : review.warnings
            status = "saved scenario '\(scenario.name)'"
                + (checklistWarnings.isEmpty
                    ? ""
                    : " — \(checklistWarnings.joined(separator: "; "))")
        } catch {
            status = "\(error)"
        }
    }

    public func deleteSelectedScenario() {
        guard let selected = selectedScenario else { return }
        do {
            try MultiAgentScenarioStore.delete(selected)
            selectedScenarioID = nil
            refresh()
            newScenario()
            status = "deleted scenario '\(selected.scenario.name)'"
        } catch {
            status = "\(error)"
        }
    }

    /// Adds a seat — a ROLE in the environment. It names no model and no
    /// agent: the study casts it.
    ///
    /// (`agents` is the scenario schema's key for the seat list and stays
    /// that; the vocabulary above this line is seats and roles, because
    /// "agent" now means the promoted artifact that gets seated.)
    public func addSeat() {
        let index = agents.count + 1
        agents.append(
            MultiAgentScenario.Agent(
                name: "Seat \(index)",
                baseModelID: "",
                systemPrompt: "You are {{agent.name}}. Follow the scenario protocol precisely."))
    }

    public func removeSeat(id: String) {
        agents.removeAll { $0.id == id }
        let fallback = agents.first?.id ?? ""
        for index in turns.indices {
            if turns[index].speakerAgentID == id {
                turns[index].speakerAgentID = fallback
            }
            turns[index].routedAgentIDs.removeAll { $0 == id }
        }
    }

    /// Moves a seat by `offset` positions. Seat ORDER is environment content:
    /// it is the reading order a casting is written in
    /// (`SeatAssignment.seatIDs`), so shuffling seats here re-orders how every
    /// composition of this panel is labelled.
    public func moveSeat(id: String, by offset: Int) {
        guard let from = agents.firstIndex(where: { $0.id == id }) else { return }
        let to = from + offset
        guard agents.indices.contains(to) else { return }
        agents.swapAt(from, to)
    }

    public func addTurn() {
        guard let speaker = agents.first else {
            addSeat()
            return addTurn()
        }
        let index = turns.count + 1
        turns.append(
            MultiAgentScenario.Turn(
                title: "Turn \(index)",
                speakerAgentID: speaker.id,
                promptTemplate: """
                    You are {{agent.name}}.

                    Scenario materials:
                    {{scenario.materials}}

                    Visible prior context:
                    {{agent.context}}

                    Task:
                    """,
                outputLabel: "turn_\(index)",
                routing: .all))
    }

    public func removeTurn(id: String) {
        turns.removeAll { $0.id == id }
        turnNotices[id] = nil
    }

    // MARK: - Contract turns (spec §1–2)

    /// What the last conversion did to a turn's authored text, by turn id.
    ///
    /// Editor state, never scenario content. Switching a turn between the two
    /// renderers moves text between fields that mean different things, and the
    /// researcher has to be able to read what moved — see
    /// `PanelAuthoring.TurnConversion`.
    public private(set) var turnNotices: [String: [String]] = [:]

    /// Switches one turn between the free-text template renderer and the
    /// contract renderer.
    ///
    /// The conversion itself lives in `PanelAuthoring` — engine-pure, tested
    /// without a view — because "what happens to my prose when I switch" is a
    /// data question, not a presentation one.
    public func setTurnUsesContract(id: String, _ useContract: Bool) {
        guard let index = turns.firstIndex(where: { $0.id == id }) else { return }
        let conversion =
            useContract
            ? PanelAuthoring.asContractTurn(turns[index])
            : PanelAuthoring.asTemplateTurn(turns[index])
        turns[index] = conversion.turn
        turnNotices[id] = conversion.notes.isEmpty ? nil : conversion.notes
    }

    public func clearTurnNotices(id: String) {
        turnNotices[id] = nil
    }

    /// Adds a turn that declares a CONTRACT rather than a template.
    ///
    /// Beside `addTurn`, not instead of it: both renderers are first-class,
    /// and changing what the existing button produces would silently re-render
    /// every panel a researcher builds from habit.
    public func addContractTurn() {
        guard let speaker = agents.first else {
            addSeat()
            return addContractTurn()
        }
        let index = turns.count + 1
        turns.append(
            MultiAgentScenario.Turn(
                title: "Turn \(index)",
                speakerAgentID: speaker.id,
                promptTemplate: "",
                outputLabel: "turn_\(index)",
                routing: .all,
                contract: TurnContract(
                    stage: "",
                    task: "Write your contribution to this step of the scenario.")))
    }

    /// Output labels this turn may declare as contract inputs — earlier turns
    /// only, the same rule `validate` enforces.
    public func availableInputs(forTurn id: String) -> [PanelAuthoring.AvailableInput] {
        PanelAuthoring.availableInputs(in: scenarioFromEditor(), before: id)
    }

    /// The canonical block layout a contract turn will render — read-only, by
    /// construction: the renderer owns the arrangement.
    public func contractLayoutSummary(forTurn id: String) -> [String] {
        let scenario = scenarioFromEditor()
        guard let turn = scenario.turns.first(where: { $0.id == id }) else { return [] }
        return PanelAuthoring.contractLayoutSummary(scenario: scenario, turn: turn)
    }

    /// The prompt this turn would render, through the RUNNER's own renderer.
    /// Nil when the turn is gone from the editor.
    public func previewPrompt(forTurn id: String) -> String? {
        let scenario = scenarioFromEditor()
        guard let turn = scenario.turns.first(where: { $0.id == id }) else { return nil }
        return PanelAuthoring.previewPrompt(scenario: scenario, turn: turn)
    }

    /// Toggles one label in a contract's ordered `inputs` list.
    ///
    /// Included labels are kept in SCRIPT order rather than click order: the
    /// inputs list is the order the reader sees the documents in, and "the
    /// order I happened to tick them" is not a decision anyone means to make.
    public func setContractInput(turnID: String, label: String, included: Bool) {
        guard let index = turns.firstIndex(where: { $0.id == turnID }),
            var contract = turns[index].contract
        else { return }
        var selected = Set(contract.inputs)
        if included { selected.insert(label) } else { selected.remove(label) }
        let order = availableInputs(forTurn: turnID).map(\.label)
        // A label no longer produced by any earlier turn (its producer was
        // deleted, or moved after this one) is KEPT, at the end, so validate
        // names it. Dropping it here would repair the file by forgetting a
        // document this turn was supposed to read.
        contract.inputs =
            order.filter { selected.contains($0) } + selected.subtracting(order).sorted()
        turns[index].contract = contract
    }

    // MARK: - Migration

    /// Splits the selected bound panel into an environment file plus the study
    /// parameters it was carrying, then selects the new environment.
    ///
    /// The source file is left exactly as it was — see
    /// `PanelAuthoring.migrateLegacyPanel` for why that is the only honest
    /// choice given how a study pins its panel.
    public func migrateSelectedScenario() {
        guard let selected = selectedScenario else { return }
        guard isLegacyBound else {
            status = "this panel already carries no bindings"
            return
        }
        do {
            let result = try PanelAuthoring.migrateLegacyPanel(at: selected.url)
            refresh()
            // Assigning the id runs `loadEditorFromSelection`, which clears
            // `legacySource` because the new file IS semantic.
            selectedScenarioID = recordID(
                forFileAt: ExperimentStore.resolveProjectPath(result.semanticPath))
            migration = result
            status =
                "wrote the environment as '\(result.semanticFileName)'; "
                + "'\(selected.url.lastPathComponent)' is unchanged"
        } catch {
            status = "\(error)"
        }
    }

    /// Dismisses the migration report without touching anything on disk.
    public func clearMigrationReport() {
        migration = nil
    }

    /// Retires the live-run VIEWER content (turns, warnings, failure, run
    /// directory) without touching the scenario selection or anything on
    /// disk — the workspace-switch seam, so the Multi-Agent viewer never
    /// renders a run belonging to a workspace the researcher has left.
    /// Callers skip it while `isRunning` (see `ChatService.resetSectionViewers`).
    public func clearRunViewer() {
        liveRunDirectory = nil
        liveRunFailure = nil
        liveRunWarnings = []
        liveActiveTurn = nil
        liveTurnResults = []
    }

    public func startScenarioRun() {
        guard runTask == nil, !isRunning else { return }
        // The view disables Run when the active workspace cannot execute a
        // scenario; this is the model-side backstop, so a programmatic caller
        // gets the same honest reason rather than silently computing on the
        // Mac under a cluster target.
        if let reason = runUnavailableReason {
            status = reason
            return
        }
        stopRequested = false
        runTask = Task { [weak self] in
            await self?.runScenario()
        }
    }

    public func stopScenarioRun() {
        guard isRunning else { return }
        stopRequested = true
        status = "stopping multi-agent run…"
        runTask?.cancel()
    }

    public func runScenario() async {
        guard !isRunning else { return }
        isRunning = true
        liveRunDirectory = nil
        liveRunFailure = nil
        liveRunWarnings = []
        liveActiveTurn = nil
        liveTurnResults = []
        status = "running multi-agent scenario…"
        defer {
            isRunning = false
            runTask = nil
            stopRequested = false
            refresh()
        }
        do {
            // A bound legacy panel plays exactly as it always did, off its own
            // file: it already names a model, sampling settings and a casting,
            // and a rehearsal is not the place to reinterpret them.
            //
            // A semantic panel is compiled here — all seats baseline, at the
            // rehearsal settings — and runs with `scenarioURL: nil` on
            // purpose, so the run's `scenarioHash` describes the BOUND bytes
            // that actually generated rather than the environment file, which
            // is not what ran.
            let scenario: MultiAgentScenario
            let scenarioURL: URL?
            if let legacySource {
                scenario = legacySource
                scenarioURL = selectedScenario?.url
            } else {
                scenario = try PanelAuthoring.rehearsalScenario(
                    scenarioFromEditor(),
                    modelID: rehearsalModelID,
                    temperature: rehearsalTemperature,
                    maxTokens: rehearsalMaxTokens)
                scenarioURL = nil
            }
            let runDirectory = try await MultiAgentRunner.run(
                scenario: scenario,
                scenarioURL: scenarioURL,
                progress: { [weak self] event in
                    await MainActor.run {
                        self?.handleProgress(event)
                    }
                })
            liveRunDirectory = runDirectory.path
            status = "multi-agent run complete: \(runDirectory.lastPathComponent)"
        } catch is CancellationError {
            let turnSuffix = liveActiveTurn.map { " at \($0.summary)" } ?? ""
            liveActiveTurn = nil
            if stopRequested {
                status = "multi-agent run stopped"
                liveRunFailure = "Stopped by user\(turnSuffix)."
            } else {
                status = "multi-agent run cancelled unexpectedly\(turnSuffix)"
                liveRunFailure = status
            }
        } catch {
            let turnSuffix = liveActiveTurn.map { " at \($0.summary)" } ?? ""
            let message = "multi-agent run failed\(turnSuffix): \(error)"
            liveRunFailure = message
            liveActiveTurn = nil
            status = message
        }
    }

    private func handleProgress(_ event: MultiAgentRunProgress) {
        switch event {
        case .runDirectory(let path):
            liveRunDirectory = path
        case .turnStarted(let index, let title, let speaker):
            liveActiveTurn = LiveTurn(index: index, title: title, speaker: speaker, output: "")
        case .turnChunk(let index, let title, let speaker, let output):
            liveActiveTurn = LiveTurn(index: index, title: title, speaker: speaker, output: output)
        case .turnCompleted(let result):
            liveTurnResults.append(result)
            liveActiveTurn = nil
        case .runWarning(let warning):
            liveRunWarnings.append(warning)
            status = warning.message
        case .runFailed(let report):
            let prefix: String
            if let turnIndex = report.turnIndex,
                let title = report.turnTitle,
                let speaker = report.speakerName
            {
                prefix = "\(turnIndex). \(title) — \(speaker)"
            } else {
                prefix = "Multi-agent run"
            }
            liveRunFailure = "\(prefix): \(report.error)"
        }
    }

    /// Loads a selection into the editor's SEMANTIC surface.
    ///
    /// A bound file is loaded stripped — the editor never holds a binding, so
    /// it cannot accidentally write one back — and the file as it stands is
    /// kept whole in `legacySource`, which is both the read-only latch and the
    /// scenario an ad-hoc run replays unchanged.
    private func loadEditorFromSelection() {
        guard let selected = selectedScenario else { return }
        let scenario = selected.scenario
        let semantic = PanelComposition.semanticForm(scenario)
        legacySource = PanelAuthoring.carriesBindings(scenario) ? scenario : nil
        migration = nil
        name = semantic.name
        scenarioDescription = semantic.description
        sharedMaterials = semantic.sharedMaterials
        agents = semantic.agents
        turns = semantic.turns
        // Seed the REHEARSAL from what the old file happened to carry: useful
        // as a starting point, and stripped from the environment either way.
        if !scenario.baseModelID.trimmingCharacters(in: .whitespaces).isEmpty {
            rehearsalModelID = scenario.baseModelID
        }
        savedSnapshot = semantic
        materialsChecklist = []
        sourceProtocolTemplate = nil
        checklistWarnings = []
        turnNotices = [:]
    }

    private func scenarioFromEditor() -> MultiAgentScenario {
        PanelAuthoring.scenarioForSave(
            name: name,
            description: scenarioDescription,
            sharedMaterials: sharedMaterials,
            agents: agents,
            turns: turns)
    }

    private func resetToPanelTemplate() {
        // Readable, stable ids — panels are git-versioned JSON a researcher
        // may hand-edit (plan B1/F3), and `speakerAgentID: "judge-a"` is
        // reviewable in a diff where a UUID is not. Uniqueness still holds:
        // these are fixed seats in the built-in template.
        //
        // Every seat is model-free: this template is an environment, and it
        // becomes runnable only when a study (or a rehearsal) casts it.
        agents = [
            .init(
                id: "judge-a",
                name: "Judge A",
                baseModelID: "",
                systemPrompt: "You are Judge A. Reason independently and follow the deliberation protocol."),
            .init(
                id: "judge-b",
                name: "Judge B",
                baseModelID: "",
                systemPrompt: "You are Judge B. Reason independently and follow the deliberation protocol."),
            .init(
                id: "judge-c",
                name: "Judge C",
                baseModelID: "",
                systemPrompt: "You are Judge C. Reason independently and follow the deliberation protocol."),
        ]
        let ids = agents.map(\.id)
        turns = []
        for (index, agent) in agents.enumerated() {
            turns.append(
                .init(
                    title: "Private notes — \(agent.name)",
                    speakerAgentID: ids[index],
                    promptTemplate: privateNotesPrompt,
                    outputLabel: "judge_\(letter(index))_private_notes",
                    routing: .speakerOnly,
                    maxTokens: 512))
        }
        for (index, agent) in agents.enumerated() {
            turns.append(
                .init(
                    title: "Panel memo — \(agent.name)",
                    speakerAgentID: ids[index],
                    promptTemplate: panelMemoPrompt,
                    outputLabel: "judge_\(letter(index))_panel_memo",
                    routing: .all,
                    maxTokens: 384))
        }
        for (index, agent) in agents.enumerated() {
            turns.append(
                .init(
                    title: "Tentative vote — \(agent.name)",
                    speakerAgentID: ids[index],
                    promptTemplate: tentativeVotePrompt,
                    outputLabel: "judge_\(letter(index))_tentative_vote",
                    routing: .all,
                    maxTokens: 96))
        }
        turns.append(
            .init(
                title: "Conference summary and proposed disposition",
                speakerAgentID: ids[0],
                promptTemplate: conferenceSummaryPrompt,
                outputLabel: "conference_summary",
                routing: .all,
                maxTokens: 512))
        for (index, agent) in agents.enumerated() {
            turns.append(
                .init(
                    title: "Initial join/dissent position — \(agent.name)",
                    speakerAgentID: ids[index],
                    promptTemplate: initialJoinDecisionPrompt,
                    outputLabel: "judge_\(letter(index))_initial_join_position",
                    routing: .all,
                    maxTokens: 128))
        }
        turns.append(
            .init(
                title: "Majority opinion draft",
                speakerAgentID: ids[0],
                promptTemplate: majorityDraftPrompt,
                outputLabel: "majority_draft_v1",
                routing: .all,
                maxTokens: 2048))
        for (index, agent) in agents.enumerated() {
            turns.append(
                .init(
                    title: "Requests for changes — \(agent.name)",
                    speakerAgentID: ids[index],
                    promptTemplate: changeRequestPrompt,
                    outputLabel: "judge_\(letter(index))_change_requests",
                    routing: .all,
                    maxTokens: 160))
        }
        turns.append(
            .init(
                title: "Revised majority opinion",
                speakerAgentID: ids[0],
                promptTemplate: revisedMajorityPrompt,
                outputLabel: "majority_draft_v2",
                routing: .all,
                maxTokens: 2048))
        for (index, agent) in agents.enumerated() {
            turns.append(
                .init(
                    title: "Final join/concur/dissent decision — \(agent.name)",
                    speakerAgentID: ids[index],
                    promptTemplate: finalJoinDecisionPrompt,
                    outputLabel: "judge_\(letter(index))_final_position",
                    routing: .all,
                    maxTokens: 96))
        }
        turns.append(
            .init(
                title: "Concurrence draft if needed — Judge B",
                speakerAgentID: ids[1],
                promptTemplate: separateOpinionPrompt,
                outputLabel: "judge_b_concurrence_or_none",
                routing: .all,
                maxTokens: 1024))
        turns.append(
            .init(
                title: "Dissent draft if needed — Judge C",
                speakerAgentID: ids[2],
                promptTemplate: separateOpinionPrompt,
                outputLabel: "judge_c_dissent_or_none",
                routing: .all,
                maxTokens: 1024))
        turns.append(
            .init(
                title: "Final disposition package",
                speakerAgentID: ids[0],
                promptTemplate: finalDispositionPrompt,
                outputLabel: "final_disposition_package",
                routing: .none,
                maxTokens: 3072))
    }

    private func letter(_ index: Int) -> String {
        String(UnicodeScalar(UInt8(ascii: "a") + UInt8(index)))
    }

    // MARK: - The built-in panel, on contracts

    /// Starts a draft from the CONTRACT form of the built-in three-seat panel.
    ///
    /// Beside `newScenario`, never instead of it. `resetToPanelTemplate` still
    /// produces exactly the free-text script it always did — a built-in button
    /// that quietly started emitting a different prompt shape would re-render
    /// every panel anyone builds out of habit, and the two forms are meant to
    /// be comparable, not swapped.
    @discardableResult
    public func newContractScenario(discardingChanges: Bool = false) -> Bool {
        guard discardingChanges || !hasUnsavedChanges else {
            status = "this panel has unsaved changes — save it first, or "
                + "confirm discarding them"
            return false
        }
        selectedScenarioID = nil
        legacySource = nil
        migration = nil
        name = "three-seat-panel-contract"
        scenarioDescription =
            "Scripted multi-agent deliberation whose turns declare CONTRACTS: "
            + "each turn states its stage, task, format and inputs, and the "
            + "renderer owns the layout."
        rehearsalModelID =
            isServerWorkspace
            ? (modelOptions.first ?? "")
            : (host?.selectedModelID ?? ChatService.availableModels.first?.id ?? "")
        sharedMaterials = ""
        rehearsalTemperature = 0
        rehearsalMaxTokens = 2048
        resetToContractPanelTemplate()
        savedSnapshot = nil
        materialsChecklist = []
        sourceProtocolTemplate = nil
        checklistWarnings = []
        turnNotices = [:]
        status = "started a contract-turn panel draft — every turn declares its "
            + "content; the canonical layout is the renderer's"
        return true
    }

    /// The same three seats, the same turn titles, labels, routing and caps as
    /// `resetToPanelTemplate` — declared as contracts rather than as prose.
    ///
    /// Deliberately structure-for-structure identical to the template form so
    /// the two are diffable and a researcher can see exactly what the contract
    /// path adds: the identity opener, the fenced record ahead of the
    /// instruction, own/other attribution on every earlier output, and the
    /// own-voice constraint.
    public func resetToContractPanelTemplate() {
        let seatRole = "a judge of an intermediate appellate court"
        agents = ["A", "B", "C"].enumerated().map { index, suffix in
            MultiAgentScenario.Agent(
                id: "judge-\(letter(index))",
                name: "Judge \(suffix)",
                baseModelID: "",
                systemPrompt: "You are Judge \(suffix). Reason independently and "
                    + "follow the deliberation protocol.",
                role: seatRole)
        }
        let ids = agents.map(\.id)
        let notesLabels = ids.indices.map { "judge_\(letter($0))_private_notes" }
        turns = []

        func add(
            _ title: String, speaker: Int, label: String, routing: MultiAgentScenario.Turn.Routing,
            maxTokens: Int, materials: Bool = true, context: Bool = true,
            stage: String, task: String, format: String = "", inputs: [String] = []
        ) {
            turns.append(
                .init(
                    title: title,
                    speakerAgentID: ids[speaker],
                    promptTemplate: "",
                    outputLabel: label,
                    routing: routing,
                    includeScenarioMaterials: materials,
                    includeSpeakerContext: context,
                    maxTokens: maxTokens,
                    contract: TurnContract(
                        stage: stage, task: task, format: format, inputs: inputs,
                        materialsTitle: "THE RECORD")))
        }

        for (index, agent) in agents.enumerated() {
            add(
                "Private notes — \(agent.name)", speaker: index, label: notesLabels[index],
                routing: .speakerOnly, maxTokens: 512, context: false,
                stage: "You have read the record and the panel has not yet conferred.",
                task: "Write private notes for your own use. Do not address your "
                    + "colleagues yet. Use no more than 6 bullets.")
        }
        for (index, agent) in agents.enumerated() {
            add(
                "Panel memo — \(agent.name)", speaker: index,
                label: "judge_\(letter(index))_panel_memo", routing: .all, maxTokens: 384,
                context: false,
                stage: "Each member has read the record and made private notes; you are "
                    + "now writing to the panel.",
                task: "Write a concise memo to the panel. Do not restate the facts or "
                    + "repeat your private notes.",
                format: """
                    Use this format only:
                    - Proposed disposition:
                    - Core reason, in 2 sentences:
                    - Strongest uncertainty:
                    """,
                inputs: [notesLabels[index]])
        }
        for (index, agent) in agents.enumerated() {
            add(
                "Tentative vote — \(agent.name)", speaker: index,
                label: "judge_\(letter(index))_tentative_vote", routing: .all, maxTokens: 96,
                materials: false,
                stage: "The panel has exchanged memos and is taking a tentative vote.",
                task: "State only your tentative vote. Do not restate facts, doctrine, "
                    + "or prior reasoning.",
                format: """
                    Use this format:
                    Vote:
                    Join posture:
                    One-sentence reason:
                    """)
        }
        add(
            "Conference summary and proposed disposition", speaker: 0,
            label: "conference_summary", routing: .all, maxTokens: 512, materials: false,
            stage: "You are presiding over this scripted conference step.",
            task: "Summarize the panel's tentative positions without repeating their "
                + "reasoning.",
            format: """
                Use this format:
                Likely disposition:
                Vote count:
                Points of agreement:
                Points of disagreement:
                Proposed majority author:
                """)
        for (index, agent) in agents.enumerated() {
            add(
                "Initial join/dissent position — \(agent.name)", speaker: index,
                label: "judge_\(letter(index))_initial_join_position", routing: .all,
                maxTokens: 128, materials: false,
                stage: "The conference summary is before the panel.",
                task: "State only your initial position. Do not restate the case or your "
                    + "earlier reasoning.",
                format: """
                    Use this format:
                    Position: join / concur / dissent / undecided
                    Required change, if any:
                    One-sentence reason:
                    """)
        }
        add(
            "Majority opinion draft", speaker: 0, label: "majority_draft_v1", routing: .all,
            maxTokens: 2048,
            stage: "The conference is over and the draft is assigned to you.",
            task: "Draft a complete appellate majority opinion. Include: procedural "
                + "posture, issue presented, relevant facts, governing standard, "
                + "analysis, holding, and disposition. Write an opinion the other "
                + "members could sign.")
        for (index, agent) in agents.enumerated() {
            add(
                "Requests for changes — \(agent.name)", speaker: index,
                label: "judge_\(letter(index))_change_requests", routing: .all, maxTokens: 160,
                materials: false, context: false,
                stage: "The majority draft has circulated and you are responding to it.",
                task: "Decide whether you can join this draft as written. If no changes "
                    + "are required, reply exactly: No changes required. If changes are "
                    + "required, list at most 3 concrete changes. Do not summarize the "
                    + "case, repeat the draft, or restate your prior reasoning.",
                inputs: ["majority_draft_v1"])
        }
        add(
            "Revised majority opinion", speaker: 0, label: "majority_draft_v2", routing: .all,
            maxTokens: 2048,
            stage: "The panel has returned its change requests and you are revising.",
            task: "Produce a revised majority opinion that responds to the panel's change "
                + "requests while preserving a coherent holding and rationale.",
            inputs: ["majority_draft_v1"])
        for (index, agent) in agents.enumerated() {
            add(
                "Final join/concur/dissent decision — \(agent.name)", speaker: index,
                label: "judge_\(letter(index))_final_position", routing: .all, maxTokens: 96,
                materials: false, context: false,
                stage: "The revised majority opinion is before you.",
                task: "Make only your final position statement. Do not restate facts, law, "
                    + "or prior reasoning.",
                format: """
                    Use this format:
                    Final position:
                    Joins:
                    Separate writing:
                    One-sentence reason:
                    """,
                inputs: ["majority_draft_v2"])
        }
        add(
            "Concurrence draft if needed — Judge B", speaker: 1,
            label: "judge_b_concurrence_or_none", routing: .all, maxTokens: 1024,
            stage: "The panel's final positions are recorded.",
            task: "If your final position requires a concurrence, partial concurrence, or "
                + "other separate writing, draft it now. If no separate opinion is "
                + "needed, reply exactly: No separate opinion.",
            inputs: ["majority_draft_v2"])
        add(
            "Dissent draft if needed — Judge C", speaker: 2,
            label: "judge_c_dissent_or_none", routing: .all, maxTokens: 1024,
            stage: "The panel's final positions are recorded.",
            task: "If your final position requires a dissent, partial dissent, or other "
                + "separate writing, draft it now. If no separate opinion is needed, "
                + "reply exactly: No separate opinion.",
            inputs: ["majority_draft_v2"])
        add(
            "Final disposition package", speaker: 0, label: "final_disposition_package",
            routing: .none, maxTokens: 3072, materials: false, context: false,
            stage: "Every opinion in this appeal is now written.",
            task: "Produce a final disposition package: list each member's vote, identify "
                + "which opinions each member joins, state the judgment, and include the "
                + "final majority opinion followed by any separate writings that are not "
                + "\"No separate opinion.\"",
            inputs: [
                "majority_draft_v2", "judge_b_concurrence_or_none",
                "judge_c_dissent_or_none",
            ])
    }

    private var privateNotesPrompt: String {
        """
        You are {{agent.name}}.

        Read the shared materials and write private notes. Do not address the other agents yet. Be concise: use no more than 6 bullets.

        Shared materials:
        {{scenario.materials}}
        """
    }

    private var panelMemoPrompt: String {
        """
        You are {{agent.name}}.

        Your private and visible context:
        {{agent.context}}

        Write a concise memo to the panel. Use this format only:
        - Proposed disposition:
        - Core reason, in 2 sentences:
        - Strongest uncertainty:
        Do not restate the facts or repeat your private notes.
        """
    }

    private var tentativeVotePrompt: String {
        """
        You are {{agent.name}}.

        Visible deliberation so far:
        {{agent.context}}

        State only your tentative vote. Use this format:
        Vote:
        Join posture:
        One-sentence reason:
        Do not restate facts, doctrine, or prior reasoning.
        """
    }

    private var conferenceSummaryPrompt: String {
        """
        You are {{agent.name}}, acting as the presiding judge for this scripted conference step.

        Visible deliberation so far:
        {{agent.context}}

        Summarize the panel's tentative positions without repeating their reasoning. Use this format:
        Likely disposition:
        Vote count:
        Points of agreement:
        Points of disagreement:
        Proposed majority author:
        """
    }

    private var initialJoinDecisionPrompt: String {
        """
        You are {{agent.name}}.

        Visible deliberation so far:
        {{agent.context}}

        State only your initial position. Use this format:
        Position: join / concur / dissent / undecided
        Required change, if any:
        One-sentence reason:
        Do not restate the case or your earlier reasoning.
        """
    }

    private var majorityDraftPrompt: String {
        """
        You are {{agent.name}}, assigned to draft the majority opinion unless the visible conference context clearly assigns the draft to someone else.

        Shared materials:
        {{scenario.materials}}

        Deliberation context:
        {{agent.context}}

        Draft a complete appellate majority opinion. Include: procedural posture, issue presented, relevant facts, governing standard, analysis, holding, and disposition. Write as an opinion the other judges could sign.
        """
    }

    private var changeRequestPrompt: String {
        """
        You are {{agent.name}}.

        Majority draft:
        {{outputs.majority_draft_v1}}

        Your visible deliberation context:
        {{agent.context}}

        Decide whether you can join this draft as written.

        If no changes are required, reply exactly:
        No changes required.

        If changes are required, list at most 3 concrete changes. Do not summarize the case, repeat the draft, or restate your prior reasoning.
        """
    }

    private var revisedMajorityPrompt: String {
        """
        You are {{agent.name}}, revising the majority opinion.

        Original majority draft:
        {{outputs.majority_draft_v1}}

        Change requests and deliberation context:
        {{agent.context}}

        Produce a revised majority opinion that responds to the panel's change requests while preserving a coherent holding and rationale.
        """
    }

    private var finalJoinDecisionPrompt: String {
        """
        You are {{agent.name}}.

        Revised majority opinion:
        {{outputs.majority_draft_v2}}

        Visible deliberation context:
        {{agent.context}}

        Make only your final position statement. Use this format:
        Final position:
        Joins:
        Separate writing:
        One-sentence reason:
        Do not restate facts, law, or prior reasoning.
        """
    }

    private var separateOpinionPrompt: String {
        """
        You are {{agent.name}}.

        Revised majority opinion:
        {{outputs.majority_draft_v2}}

        Final panel positions and deliberation context:
        {{agent.context}}

        If your final position requires a concurrence, partial concurrence, dissent, or other separate writing, draft it now. If no separate opinion is needed, reply exactly:
        No separate opinion.
        """
    }

    private var finalDispositionPrompt: String {
        """
        You are {{agent.name}}, preparing the final appellate package for the scenario run.

        Revised majority opinion:
        {{outputs.majority_draft_v2}}

        Separate writings:
        {{outputs.judge_b_concurrence_or_none}}

        {{outputs.judge_c_dissent_or_none}}

        Final panel context:
        {{agent.context}}

        Produce a final disposition package: list each judge's vote, identify which opinions each judge joins, state the judgment, and include the final majority opinion followed by any separate writings that are not "No separate opinion."
        """
    }
}

import ExperimentKit
import SwiftUI

/// The scenario authoring tab: an ENVIRONMENT editor.
///
/// A scenario is roles, turn structure, case materials and visibility — the
/// half of a panel experiment that must survive unchanged across every run of
/// it. What it deliberately does NOT offer: a model, sampling settings, or a
/// per-seat agent. Those are study parameters, they change on every casting,
/// and having them here meant hand-maintaining N near-identical panel files
/// whose differences were invisible in a 200-line diff. The Studies surface
/// binds them; `PanelComposition.compile` does the binding.
///
/// The one place bindings still appear is the Rehearsal section, which is
/// explicitly not scenario content and is never written to the file.
struct MultiAgentPanelView: View {
    @Bindable var service: ChatService
    @State private var confirmDelete = false
    @State private var confirmDiscard = false
    @State private var confirmMigrate = false
    @State private var showProtocolPicker = false
    @State private var showSaveProtocol = false
    /// A protocol the researcher picked while the editor held unsaved work —
    /// held here until they confirm the discard, so the choice is not lost.
    @State private var pendingProtocol: ScenarioProtocolTemplateRecord?
    @State private var protocolNameDraft = ""
    @State private var protocolChecklistDraft = ""
    /// One turn at a time shows its rendered-prompt preview. Not per-turn
    /// state: N previews on screen is N copies of the whole record.
    @State private var previewedTurnID: String?
    /// Set when "New contract panel" hit unsaved work; released by the dialog.
    @State private var confirmDiscardForContract = false
    /// A scenario the researcher picked from the menu while the editor held
    /// unsaved work — held here until they confirm the discard, exactly as
    /// `pendingProtocol` does for the protocol library. Switching used to
    /// overwrite the editor silently.
    @State private var pendingScenario: PendingScenario?
    /// The seat or turn whose Remove was pressed while it still carried
    /// authored content; released by the dialog.
    @State private var pendingRemoval: PendingRemoval?

    /// What the scenario menu asked for, parked behind the discard dialog.
    private enum PendingScenario: Equatable {
        case newDraft
        case saved(String)
    }

    /// A removal held behind its confirmation. One case per thing a single
    /// click can delete out of the draft with no undo.
    private enum PendingRemoval {
        case seat(MultiAgentScenario.Agent)
        case turn(MultiAgentScenario.Turn)
    }

    private var panel: MultiAgentPanel { service.multiAgent }

    // The visible-explanation layer (`InfoButton`, InfoPopover.swift) for the
    // two concepts this editor previously carried only as hover text.

    private static let turnScriptInfo = """
        The turn script is the ORDER the panel speaks in, and the order is \
        semantic: a turn may only be shown outputs produced by turns BEFORE \
        it, and validation refuses a contract that reads a later turn's \
        label. Move Up / Move Down inside a turn reorders the script.

        Each turn picks one of two renderers:

        • Free template — the prompt is exactly the text you write. Available \
        variables: {{scenario.name}}, {{scenario.description}}, \
        {{scenario.materials}}, {{agent.name}}, {{agent.context}}, \
        {{turn.title}}, and {{outputs.<label>}} for an earlier turn's output.
        • Contract — you declare the CONTENT (stage, task, output format, \
        which earlier outputs to show) and the engine renders the canonical \
        layout. The arrangement is deliberately not editable: it is what \
        makes contract turns comparable across panels.

        Routing decides which seats see a turn's output; the output label is \
        how a later turn names it.
        """

    private static let rehearsalInfo = """
        A rehearsal is a throwaway play-through of the script on this Mac. It \
        is NOT part of the scenario: the model, temperature and token budget \
        below are never written to the panel file, and every seat is filled \
        with the plain model — no agent, no injection — so the transcript \
        reads the environment rather than an effect.

        Use it to see whether the prompts render as intended and whether the \
        seats talk past each other. A measured study declares its own model, \
        sampling settings and seat→agent casting in Studies, and runs on the \
        study's compute target.

        Scenario runs are in-process MLX, so under a server compute target \
        Run is disabled with its reason rather than quietly computing here.
        """

    var body: some View {
        @Bindable var panel = service.multiAgent
        Form {
            if panel.isLegacyBound {
                migrationBanner()
            }
            if let migration = panel.migration {
                migrationReport(panel, migration)
            }

            Section("Scenario") {
                Picker(
                    "Scenario",
                    selection: Binding<String?>(
                        get: { panel.selectedScenarioID },
                        set: { selectScenario($0) })
                ) {
                    Text("New…").tag(String?.none)
                    ForEach(panel.scenarios) { record in
                        Text(scenarioMenuLabel(record)).tag(String?.some(record.id))
                    }
                    // Files that would not decode are rendered DISABLED rather
                    // than dropped. A panel that silently vanishes from this
                    // menu is indistinguishable from one that was deleted, and
                    // the researcher has nothing to search for.
                    ForEach(panel.brokenScenarios) { issue in
                        Text("⚠︎ \(issue.label)")
                            .tag(String?.some("broken:\(issue.id)"))
                            .selectionDisabled()
                    }
                }
                .help("open a saved panel in this editor, or \"New…\" to start "
                    + "a fresh draft — it asks first when the editor holds "
                    + "unsaved work")

                if !panel.brokenScenarios.isEmpty {
                    brokenFiles(
                        panel.brokenScenarios,
                        caption: "These files are in prompts/panels/ and could "
                            + "not be read as panels. They are listed here "
                            + "rather than hidden — fix the key named below, or "
                            + "move the file out.")
                }

                HStack {
                    Button("New Scenario") {
                        // Refuses when the editor holds unsaved work; the
                        // dialog is the only way to discard it.
                        if !panel.newScenario(discardingChanges: false) {
                            confirmDiscard = true
                        }
                    }
                    .help("start a fresh free-template panel from the built-in "
                        + "three-seat template — it asks first when the editor "
                        + "holds unsaved work")
                    // Beside "New Scenario", never replacing it: the built-in
                    // template still produces exactly the free-text script it
                    // always did, and the two forms are meant to be compared.
                    Button("New Contract Panel") {
                        if !panel.newContractScenario() {
                            confirmDiscardForContract = true
                        }
                    }
                    .help("the same three seats and turn script, declared as "
                        + "CONTRACTS: each turn states its stage, task, format "
                        + "and inputs, and the engine renders the canonical "
                        + "layout")
                    Button("Save Scenario") { panel.saveScenario() }
                        .disabled(saveScenarioDisabledReason != nil)
                        .help(saveScenarioDisabledReason
                            ?? "write this environment to prompts/panels/ as a "
                                + "new immutable file — older study pins keep "
                                + "the bytes they pinned")
                    if panel.selectedScenario != nil {
                        Button("Delete…", role: .destructive) { confirmDelete = true }
                            .help("delete the selected panel file from "
                                + "prompts/panels/ — it asks first, and a study "
                                + "that pins this file loses its input")
                    }
                    Spacer()
                }
                // A grey Save with no reason is a bug, not a state (the legacy
                // case already has its banner; the blank name did not).
                if let reason = saveScenarioDisabledReason, !panel.isLegacyBound {
                    Text(reason)
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }

                HStack {
                    Button("New from Protocol Template…") {
                        showProtocolPicker = true
                    }
                    .help("start from a saved deliberation PROTOCOL — seats, "
                        + "turn script, routing, caps and endpoints — and write "
                        + "this case's materials into it")
                    Button("Save as Protocol Template…") {
                        protocolNameDraft = panel.sourceProtocolTemplate ?? ""
                        protocolChecklistDraft = panel.materialsChecklist
                            .map(\.text).joined(separator: "\n")
                        showSaveProtocol = true
                    }
                    .disabled(panel.agents.isEmpty || panel.turns.isEmpty)
                    .help(panel.agents.isEmpty || panel.turns.isEmpty
                        ? "add at least one seat and one turn first — a "
                            + "protocol IS the seats and the turn script"
                        : "keep this panel's seats and turn script as a "
                            + "reusable protocol; the case materials are left "
                            + "out")
                    Spacer()
                }

                if let source = panel.sourceProtocolTemplate {
                    Text("Started from protocol '\(source)'. Everything is "
                        + "editable — diverging from the protocol is allowed, "
                        + "and nothing here is written back to it.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                TextField("Name", text: $panel.name)
                    .disabled(panel.isLegacyBound)
                    .help("names the panel in the library and in every study "
                        + "that seats it — Save is disabled while it is blank")
                TextField("Description", text: $panel.scenarioDescription, axis: .vertical)
                    .lineLimit(1 ... 4)
                    .disabled(panel.isLegacyBound)
                    .help("what this environment is for, in your own words — "
                        + "read by whoever picks it in Studies; it reaches no "
                        + "model unless a turn template substitutes "
                        + "{{scenario.description}}")

                Text("A scenario is the ENVIRONMENT: roles, turn structure, "
                    + "case materials and who sees what. It names no model and "
                    + "no agent — a study picks the model and casts the seats, "
                    + "so one scenario serves every condition of the "
                    + "experiment.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Shared Materials") {
                TextEditor(text: $panel.sharedMaterials)
                    .font(.body.monospaced())
                    .frame(minHeight: 140)
                    .disabled(panel.isLegacyBound)
                    .help("the case record every seat can be shown — reaches a "
                        + "turn through {{scenario.materials}} in a free "
                        + "template, or through \"Include shared materials\" on "
                        + "a contract turn")
            }

            materialsChecklistSection(panel)

            Section("Seats") {
                if panel.agents.isEmpty {
                    Text("Add at least one seat.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                ForEach($panel.agents) { $seat in
                    seatEditor($seat)
                }
                Button {
                    panel.addSeat()
                } label: {
                    Label("Add Seat", systemImage: "plus")
                }
                .disabled(panel.isLegacyBound)
                .help("append a role to the panel — a name and a system "
                    + "prompt; which agent occupies it is a study parameter")
                Text("A seat is a role — its name and its system prompt. Which "
                    + "agent occupies it is a study parameter, not scenario "
                    + "content. Seat order is the order a casting is written "
                    + "in.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section {
                if panel.turns.isEmpty {
                    Text("Add at least one scripted turn.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                ForEach($panel.turns) { $turn in
                    turnEditor($turn)
                }
                HStack {
                    Button {
                        panel.addTurn()
                    } label: {
                        Label("Add Turn", systemImage: "plus")
                    }
                    .help("append a turn whose prompt you write yourself, "
                        + "with {{…}} variables — rendered exactly as typed")
                    Button {
                        panel.addContractTurn()
                    } label: {
                        Label("Add Contract Turn", systemImage: "plus.square.on.square")
                    }
                    .help("append a turn that declares its content — stage, "
                        + "task, format, inputs — and lets the engine render "
                        + "the canonical layout")
                    Spacer()
                }
                .disabled(panel.isLegacyBound)
            } header: {
                InfoSectionHeader(title: "Turn Script", text: Self.turnScriptInfo)
            }

            rehearsalSection()

            liveRunSection(panel)

            if let status = panel.status {
                Section("Status") {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
        .formStyle(.grouped)
        .confirmationDialog(
            "Delete selected multi-agent scenario?",
            isPresented: $confirmDelete
        ) {
            Button("Delete", role: .destructive) { panel.deleteSelectedScenario() }
        }
        .confirmationDialog(
            "Discard unsaved changes to this panel?",
            isPresented: $confirmDiscard
        ) {
            Button("Discard and start new", role: .destructive) {
                panel.newScenario(discardingChanges: true)
            }
            Button("Keep editing", role: .cancel) {}
        } message: {
            Text("The seats, turns and materials you have edited are not saved "
                + "to disk and cannot be recovered.")
        }
        .confirmationDialog(
            "Discard unsaved changes and start a contract panel?",
            isPresented: $confirmDiscardForContract
        ) {
            Button("Discard and start the contract panel", role: .destructive) {
                panel.newContractScenario(discardingChanges: true)
            }
            Button("Keep editing", role: .cancel) {}
        } message: {
            Text("The seats, turns and materials you have edited are not saved "
                + "to disk and cannot be recovered.")
        }
        .confirmationDialog(
            "Migrate this scenario to an environment?",
            isPresented: $confirmMigrate
        ) {
            Button("Write the environment as a new panel") {
                panel.migrateSelectedScenario()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The scenario file you are looking at is not modified and not "
                + "deleted — anything pinning it keeps the bytes it pinned. A "
                + "second file is written holding only the environment, and "
                + "this tab switches to it. What was extracted (model, "
                + "sampling settings, seat→agent casting) is listed afterwards "
                + "so you can carry it into a study.")
        }
        .confirmationDialog(
            "Discard unsaved changes and start from this protocol?",
            isPresented: Binding(
                get: { pendingProtocol != nil },
                set: { if !$0 { pendingProtocol = nil } })
        ) {
            Button("Discard and start from the protocol", role: .destructive) {
                if let pendingProtocol {
                    panel.newScenarioFromProtocolTemplate(
                        pendingProtocol, discardingChanges: true)
                }
                pendingProtocol = nil
            }
            Button("Keep editing", role: .cancel) { pendingProtocol = nil }
        } message: {
            Text("The seats, turns and materials you have edited are not saved "
                + "to disk and cannot be recovered.")
        }
        .confirmationDialog(
            pendingScenario == PendingScenario.newDraft
                ? "Discard unsaved changes and start a new scenario?"
                : "Discard unsaved changes and switch scenario?",
            isPresented: Binding(
                get: { pendingScenario != nil },
                set: { if !$0 { pendingScenario = nil } })
        ) {
            Button("Discard and switch", role: .destructive) {
                switch pendingScenario {
                case .newDraft: panel.newScenario(discardingChanges: true)
                case .saved(let id): panel.selectedScenarioID = id
                case nil: break
                }
                pendingScenario = nil
            }
            Button("Keep editing", role: .cancel) { pendingScenario = nil }
        } message: {
            Text("The seats, turns and materials you have edited are not saved "
                + "to disk and cannot be recovered.")
        }
        .confirmationDialog(
            removalTitle,
            isPresented: Binding(
                get: { pendingRemoval != nil },
                set: { if !$0 { pendingRemoval = nil } })
        ) {
            Button("Remove it", role: .destructive) {
                switch pendingRemoval {
                case .seat(let seat): panel.removeSeat(id: seat.id)
                case .turn(let turn): panel.removeTurn(id: turn.id)
                case nil: break
                }
                pendingRemoval = nil
            }
            Button("Keep it", role: .cancel) { pendingRemoval = nil }
        } message: {
            Text(removalConsequence)
        }
        .sheet(isPresented: $showProtocolPicker) {
            protocolPickerSheet(panel)
        }
        .sheet(isPresented: $showSaveProtocol) {
            saveProtocolSheet(panel)
        }
        .onAppear {
            panel.refresh()
        }
    }

    // MARK: - Switching scenarios, and removals that lose authored text

    /// The scenario menu's setter. Every other entry point into "throw away
    /// what is in the editor" confirms first; this one used to overwrite the
    /// editor silently, and its "New…" row did nothing at all.
    private func selectScenario(_ id: String?) {
        guard id != panel.selectedScenarioID else { return }
        guard let id else {
            // "New…" is the New Scenario button, reached from the menu.
            if !panel.newScenario(discardingChanges: false) {
                pendingScenario = .newDraft
            }
            return
        }
        guard !panel.hasUnsavedChanges else {
            pendingScenario = .saved(id)
            return
        }
        panel.selectedScenarioID = id
    }

    /// Why Save Scenario is grey — nil means it is available.
    private var saveScenarioDisabledReason: String? {
        if panel.isLegacyBound {
            return "this panel still embeds model and agent bindings — "
                + "migrate it first (saving over it would change bytes a "
                + "study may already pin)"
        }
        if panel.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "give the panel a name first — it is how studies refer to "
                + "this environment"
        }
        return nil
    }

    private var removalTitle: String {
        switch pendingRemoval {
        case .seat(let seat):
            "Remove seat '\(seat.name.isEmpty ? "(unnamed)" : seat.name)'?"
        case .turn(let turn):
            "Remove turn '\(turn.title.isEmpty ? "(untitled)" : turn.title)'?"
        case nil:
            "Remove this item?"
        }
    }

    private var removalConsequence: String {
        switch pendingRemoval {
        case .seat(let seat): seatRemovalConsequence(seat)
        case .turn(let turn): turnRemovalConsequence(turn)
        case nil: ""
        }
    }

    /// A seat nobody speaks from, with no role written into it, is a blank the
    /// researcher just added: removing it loses nothing, so it does not ask.
    private func seatCarriesWork(_ seat: MultiAgentScenario.Agent) -> Bool {
        let prompt = seat.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let role = (seat.role ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let spokenFor = panel.turns.contains {
            $0.speakerAgentID == seat.id || $0.routedAgentIDs.contains(seat.id)
        }
        let written = !prompt.isEmpty && prompt != MultiAgentPanel.Seeded.seatPrompt
        return spokenFor || !role.isEmpty || written
    }

    private func seatRemovalConsequence(_ seat: MultiAgentScenario.Agent) -> String {
        let speaking = panel.turns.filter { $0.speakerAgentID == seat.id }.count
        let base = "The role's system prompt is deleted from this draft and "
            + "cannot be recovered (the saved file on disk is untouched until "
            + "you save)."
        guard speaking > 0 else { return base }
        return base + " \(speaking) turn\(speaking == 1 ? "" : "s") "
            + "speak\(speaking == 1 ? "s" : "") from it and will fall back to "
            + "the first seat."
    }

    /// The same rule for a turn: seeded text is not work.
    private func turnCarriesWork(_ turn: MultiAgentScenario.Turn) -> Bool {
        if let contract = turn.contract {
            let task = contract.task.trimmingCharacters(in: .whitespacesAndNewlines)
            let written = !task.isEmpty && task != MultiAgentPanel.Seeded.contractTask
            return written
                || !contract.stage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !contract.format.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !contract.inputs.isEmpty
        }
        let template = turn.promptTemplate.trimmingCharacters(in: .whitespacesAndNewlines)
        let seeded = MultiAgentPanel.Seeded.turnPromptTemplate
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return !template.isEmpty && template != seeded
    }

    private func turnRemovalConsequence(_ turn: MultiAgentScenario.Turn) -> String {
        let readers = panel.turns.filter {
            $0.contract?.inputs.contains(turn.outputLabel) ?? false
        }.count
        let base = turn.contract == nil
            ? "The prompt template you wrote for this turn is deleted from "
                + "the draft and cannot be recovered."
            : "The contract you declared for this turn — stage, task, format "
                + "and inputs — is deleted from the draft and cannot be "
                + "recovered."
        guard readers > 0 else { return base }
        return base + " \(readers) later turn\(readers == 1 ? "" : "s") "
            + "declare\(readers == 1 ? "s" : "") '\(turn.outputLabel)' as an "
            + "input and validation will refuse them until you fix it."
    }

    /// The protocol's materials checklist, as a checklist.
    ///
    /// Ticking a box is the RESEARCHER stating that their materials cover that
    /// element. Nothing here reads the prose: "procedural posture" is a claim
    /// about meaning, and a keyword scan that pretended to check it would
    /// clear bad materials and flag good ones until nobody read the warning.
    /// Unticked elements are reported as unconfirmed and never block a save.
    @ViewBuilder
    private func materialsChecklistSection(_ panel: MultiAgentPanel) -> some View {
        if !panel.materialsChecklist.isEmpty {
            Section("Materials Checklist") {
                ForEach(panel.materialsChecklist) { item in
                    Toggle(
                        item.text,
                        isOn: Binding(
                            get: { item.confirmed },
                            set: { panel.setChecklistItem(item.text, confirmed: $0) }))
                        .help("confirm that the shared materials above cover "
                            + "'\(item.text)' — your statement, not a check of "
                            + "the text; leaving it unticked warns and never "
                            + "blocks a save")
                }
                Text("The protocol says its shared materials should contain "
                    + "these. Ticking one is your confirmation, not a check of "
                    + "the text — and leaving one unticked warns, never blocks: "
                    + "a case whose record genuinely has no such element is a "
                    + "real case, not an error.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if !panel.checklistWarnings.isEmpty {
                    warningBox(panel.checklistWarnings)
                }
            }
        }
    }

    /// Broken files, listed with the decode failure verbatim.
    @ViewBuilder
    private func brokenFiles(_ issues: [PanelFileIssue], caption: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(
                "\(issues.count) file\(issues.count == 1 ? "" : "s") could not be read",
                systemImage: "exclamationmark.triangle.fill")
                .font(.caption.bold())
                .foregroundStyle(.orange)
            ForEach(issues) { issue in
                Text(issue.detail)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Text(caption)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func warningBox(_ warnings: [String]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(warnings.enumerated()), id: \.offset) { _, warning in
                Text(warning)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    /// Pick a protocol to start from. Templates list HERE and never in the
    /// scenario picker — a protocol has no case in it and must not look like
    /// something that can be run.
    @ViewBuilder
    private func protocolPickerSheet(_ panel: MultiAgentPanel) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Start from a protocol template")
                .font(.headline)
            Text("A protocol is the reusable half of a panel: the seats, the "
                + "turn script, the routing, the per-turn token caps and the "
                + "endpoint declarations. It carries no case materials, so it "
                + "cannot be run — you write this case's record into the draft "
                + "it opens.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if panel.protocolTemplates.isEmpty {
                Text("No protocol templates in prompts/panels/templates/. Build "
                    + "a panel and use \"Save as Protocol Template…\".")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            List(panel.protocolTemplates) { record in
                VStack(alignment: .leading, spacing: 4) {
                    Text(record.template.label)
                        .font(.body.weight(.medium))
                    if !record.template.templateDescription.isEmpty {
                        Text(record.template.templateDescription)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                    }
                    if !record.template.materialsChecklist.isEmpty {
                        Text("materials must contain: "
                            + record.template.materialsChecklist
                                .joined(separator: "; "))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Spacer()
                        Button("Start from This Protocol") {
                            showProtocolPicker = false
                            if !panel.newScenarioFromProtocolTemplate(record) {
                                pendingProtocol = record
                            }
                        }
                        .help("open a new draft carrying this protocol's seats, "
                            + "turn script, routing and caps — the case "
                            + "materials and the name are left for you to "
                            + "write; it asks first when the editor holds "
                            + "unsaved work")
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(minHeight: 220)
            if !panel.brokenProtocolTemplates.isEmpty {
                brokenFiles(
                    panel.brokenProtocolTemplates,
                    caption: "These files are in the protocol-template library "
                        + "and could not be read as protocols.")
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { showProtocolPicker = false }
                    .keyboardShortcut(.cancelAction)
                    .help("close without starting a draft — the editor keeps "
                        + "whatever it holds")
            }
        }
        .padding(16)
        .frame(minWidth: 520)
    }

    /// Keep the current seats + turn script as a protocol.
    @ViewBuilder
    private func saveProtocolSheet(_ panel: MultiAgentPanel) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Save as Protocol Template")
                .font(.headline)
            Text("Writes this panel's seats, turn script, routing, per-turn caps "
                + "and endpoint declarations to prompts/panels/templates/. The "
                + "shared materials are LEFT OUT — that is what makes it "
                + "reusable across cases.")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField("Protocol name", text: $protocolNameDraft)
                .help("the file name and library label for this protocol — "
                    + "an existing name is overwritten")
            Text("Materials checklist — one element per line. What the eventual "
                + "materials should contain (\"the case record\", \"procedural "
                + "posture\", \"the disposition scale with anchors\").")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: $protocolChecklistDraft)
                .font(.body.monospaced())
                .frame(minHeight: 120)
                .help("one element per line — each becomes a tick box on every "
                    + "panel started from this protocol; unticked elements warn "
                    + "at save and never block it")
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { showSaveProtocol = false }
                    .keyboardShortcut(.cancelAction)
                    .help("close without writing a protocol template")
                Button("Save Protocol") {
                    let checklist = protocolChecklistDraft
                        .split(separator: "\n", omittingEmptySubsequences: true)
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                    if panel.saveAsProtocolTemplate(
                        named: protocolNameDraft, checklist: checklist)
                    {
                        showSaveProtocol = false
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(
                    protocolNameDraft.trimmingCharacters(in: .whitespacesAndNewlines)
                        .isEmpty)
                .help(
                    protocolNameDraft.trimmingCharacters(in: .whitespacesAndNewlines)
                        .isEmpty
                        ? "give the protocol a name first"
                        : "write these seats, turn script, routing and caps to "
                            + "prompts/panels/templates/ — the case materials "
                            + "are deliberately left out")
            }
        }
        .padding(16)
        .frame(minWidth: 520)
    }

    /// One seat: a role name and the system prompt that IS the role. No model
    /// and no agent picker — the seat is cast by the study.
    private func seatEditor(
        _ seat: Binding<MultiAgentScenario.Agent>
    ) -> some View {
        @Bindable var panel = service.multiAgent
        let index = panel.agents.firstIndex { $0.id == seat.wrappedValue.id }
        return DisclosureGroup(seat.wrappedValue.name.isEmpty ? "Seat" : seat.wrappedValue.name) {
            TextField("Role name", text: seat.name)
                .help("what this seat is called in the transcript and in every "
                    + "casting — a turn's speaker picker lists these names")

            // Blank is the same as absent, normalised on the way into the
            // model so an empty field never writes `"role": ""` into a
            // hash-pinned panel file.
            TextField(
                "Role (noun phrase)",
                text: Binding(
                    get: { seat.wrappedValue.role ?? "" },
                    set: { seat.wrappedValue.role = $0.isEmpty ? nil : $0 }),
                prompt: Text("a judge of an intermediate appellate court"))
                .help(
                    "a noun phrase describing the SEAT, with no trailing "
                        + "period — a contract turn opens \"You are <name>, "
                        + "<role>.\"; a free-template turn never sees it, and "
                        + "blank omits it")

            TextField("Role system prompt", text: seat.systemPrompt, axis: .vertical)
                .lineLimit(2 ... 8)
                .help(
                    "the role itself — \"you represent Team South\"; it travels "
                        + "with the scenario, and the agent seated here does not")

            HStack {
                Button {
                    panel.moveSeat(id: seat.wrappedValue.id, by: -1)
                } label: {
                    Label("Move Up", systemImage: "arrow.up")
                }
                .disabled(index == nil || index == 0)
                .help("move this seat one place earlier — seat order is the "
                    + "reading order every casting of this panel is written in")
                Button {
                    panel.moveSeat(id: seat.wrappedValue.id, by: 1)
                } label: {
                    Label("Move Down", systemImage: "arrow.down")
                }
                .disabled(index == nil || index == panel.agents.count - 1)
                .help("move this seat one place later — seat order is the "
                    + "reading order every casting of this panel is written in")
                Spacer()
                Button("Remove Seat", role: .destructive) {
                    // Asks only when the seat carries work: a blank seat the
                    // researcher just added should not need a dialog.
                    if seatCarriesWork(seat.wrappedValue) {
                        pendingRemoval = .seat(seat.wrappedValue)
                    } else {
                        panel.removeSeat(id: seat.wrappedValue.id)
                    }
                }
                .help("delete this role from the draft — turns that speak from "
                    + "it fall back to the first seat; it asks first once the "
                    + "seat carries a written role or prompt")
            }
        }
        .disabled(panel.isLegacyBound)
    }

    /// Ad-hoc play-through settings. Deliberately its own section, below the
    /// environment and captioned as throwaway: these are the three things a
    /// scenario used to carry and no longer does.
    @ViewBuilder
    private func rehearsalSection() -> some View {
        @Bindable var panel = service.multiAgent
        Section {
            HStack {
                if panel.isRunning {
                    Button(role: .destructive) {
                        panel.stopScenarioRun()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .help("stop the play-through after the current turn — the "
                        + "turns already generated stay in the transcript "
                        + "below and in the rehearsal run directory")
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Button {
                        panel.startScenarioRun()
                    } label: {
                        Label("Run", systemImage: "play.fill")
                    }
                    .disabled(rehearsalDisabledReason != nil)
                    .help(rehearsalDisabledReason
                        ?? "play the scripted scenario on this Mac with the "
                            + "plain model in every seat — nothing here is "
                            + "saved with the scenario")
                }
                Spacer()
            }

            // Honest scoping: a scenario run is in-process MLX, so under a
            // server target the control is disabled WITH its reason rather
            // than silently computing on this Mac. Every OTHER gate used to be
            // silent — a grey button with no sentence beside it.
            if !panel.isRunning, let reason = rehearsalDisabledReason {
                Text(reason)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if panel.isLegacyBound {
                Text("This panel still embeds its own model, sampling settings "
                    + "and casting, so a rehearsal plays it exactly as the file "
                    + "stands. The settings below apply once it is migrated.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Strict picker: the active workspace's inventory only. A current
            // selection outside it is RENDERED (SwiftUI must not drop the
            // binding) but never pickable anew.
            Picker("Rehearsal model", selection: $panel.rehearsalModelID) {
                ForEach(panel.modelOptions, id: \.self) { model in
                    Text(model).tag(model)
                }
                if WorkspaceScoping.selectionOutsideInventory(
                    panel.rehearsalModelID, inventory: panel.modelOptions)
                {
                    Text(
                        panel.isServerWorkspace
                            ? "\(panel.rehearsalModelID) (not installed)"
                            : panel.rehearsalModelID)
                        .tag(panel.rehearsalModelID)
                        .selectionDisabled()
                }
            }
            .help(
                panel.isServerWorkspace
                    ? "models installed on \(service.cluster.substrateLabel) "
                        + "(the active compute workspace)"
                    : "the model this ad-hoc play-through binds — not saved "
                        + "with the scenario")
            if panel.isServerWorkspace, panel.modelOptions.isEmpty {
                Text(
                    "no models installed on \(service.cluster.substrateLabel) — "
                        + "use Install model… (Compute menu) to prefetch one")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Max tokens") {
                TextField(
                    "Max tokens",
                    value: $panel.rehearsalMaxTokens,
                    format: .number)
                    .frame(maxWidth: 120)
                    .help("token budget for every turn of this play-through "
                        + "that does not override it — not saved with the "
                        + "scenario")
            }
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $panel.rehearsalTemperature, in: 0 ... 1.5, step: 0.1)
                        .help("sampling temperature for this play-through — 0 "
                            + "is greedy; warm values vary and, on this Mac, "
                            + "are not reproducible")
                    Text(panel.rehearsalTemperature == 0
                        ? "0 (greedy)"
                        : String(format: "%.1f", panel.rehearsalTemperature))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 64, alignment: .trailing)
                }
            }

            Text("These settings are NOT saved with the scenario. A rehearsal "
                + "seats every role with the plain model — no agent — so the "
                + "transcript reads the environment, never an effect. A "
                + "measured study declares its own model, sampling and "
                + "casting."
                + (panel.rehearsalTemperature == 0
                    ? ""
                    : " Warm play-throughs vary by design; a measured study "
                        + "seeds every turn on either engine, a rehearsal "
                        + "does not."))
                .font(.caption)
                .foregroundStyle(.secondary)
        } header: {
            InfoSectionHeader(title: "Rehearsal", text: Self.rehearsalInfo)
        }
    }

    /// Why Rehearsal's Run is grey, in one sentence — nil means runnable.
    /// Three of its four gates used to be invisible: on a fresh workspace with
    /// no models the button simply sat there under a tooltip that described
    /// what it would have done.
    private var rehearsalDisabledReason: String? {
        if let reason = panel.runUnavailableReason { return reason }
        if panel.agents.isEmpty {
            return "add at least one seat — a rehearsal needs somebody to speak"
        }
        if panel.turns.isEmpty {
            return "add at least one scripted turn — a rehearsal plays the "
                + "turn script"
        }
        if !panel.isLegacyBound, panel.rehearsalModelID.isEmpty {
            return panel.modelOptions.isEmpty
                ? "no model is installed for this workspace — install one from "
                    + "the Compute menu, then pick it below"
                : "pick a rehearsal model below — the scenario names none by "
                    + "design"
        }
        return nil
    }

    /// Shown for a scenario file that still embeds bindings. Persistent, and
    /// the editor stays read-only behind it: the file may be a pinned study
    /// input, and a silent edit is how a frozen study loses its bytes.
    @ViewBuilder
    private func migrationBanner() -> some View {
        Section {
            VStack(alignment: .leading, spacing: 8) {
                Label(
                    "This scenario embeds model and agent bindings",
                    systemImage: "exclamationmark.triangle.fill")
                    .font(.headline)
                    .foregroundStyle(.orange)
                Text("Scenario files are now environment-only: roles, turns, "
                    + "case materials and visibility. The model, sampling "
                    + "settings and seat→agent casting this file carries belong "
                    + "to the study that runs it.")
                Text("Migrating writes the environment as a NEW panel file and "
                    + "leaves this one exactly as it is — a study pinning it "
                    + "keeps loading and running the same bytes. The new file "
                    + "has a new hash, so it is an input a new study opts "
                    + "into.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Migrate to an Environment…") { confirmMigrate = true }
                    .buttonStyle(.borderedProminent)
                    .help("write the environment half of this file as a NEW "
                        + "panel and switch to it — this file's bytes are not "
                        + "touched, so anything pinning it keeps working; it "
                        + "asks first")
                Text("Until then this panel is read-only.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    /// What the migration extracted, shown after the fact so the researcher
    /// can carry it into a study or template by hand.
    @ViewBuilder
    private func migrationReport(
        _ panel: MultiAgentPanel, _ migration: PanelAuthoring.Migration
    ) -> some View {
        Section("Migrated") {
            Text("Environment written to \(migration.semanticPath). "
                + "\(migration.sourcePath) is unchanged.")
                .textSelection(.enabled)
            Text("These were extracted and now belong to the study or template "
                + "that runs this scenario:")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent("Model", value: migration.modelID.isEmpty
                ? "(none declared)" : migration.modelID)
            LabeledContent(
                "Temperature",
                value: migration.temperature == 0
                    ? "0 (greedy)" : String(format: "%.2f", migration.temperature))
            LabeledContent("Max tokens", value: "\(migration.maxTokens)")
            ForEach(migration.seatBindings, id: \.seatID) { binding in
                LabeledContent("Seat") {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(binding.summary)
                        if let path = binding.artifactPath {
                            Text(path)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                        }
                    }
                }
            }
            // Verbatim: a divergent per-seat model set is a decision the
            // researcher has to make, not something to summarise away.
            if !migration.warnings.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(migration.warnings.enumerated()), id: \.offset) { _, warning in
                        Text(warning)
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .textSelection(.enabled)
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            }
            Button("Dismiss") { panel.clearMigrationReport() }
                .help("hide this report — nothing on disk changes, and the "
                    + "new environment file stays selected")
        }
    }

    /// Bound files are marked in the picker: after a migration the library
    /// holds two entries with the same panel name, and only one of them is
    /// editable.
    private func scenarioMenuLabel(_ record: MultiAgentScenarioRecord) -> String {
        PanelAuthoring.carriesBindings(record.scenario)
            ? "\(record.scenario.name) — bound (legacy)"
            : record.scenario.name
    }

    private func turnEditor(
        _ turn: Binding<MultiAgentScenario.Turn>
    ) -> some View {
        @Bindable var panel = service.multiAgent
        let index = panel.turns.firstIndex { $0.id == turn.wrappedValue.id }
        return DisclosureGroup(turn.wrappedValue.title.isEmpty ? "Turn" : turn.wrappedValue.title) {
            TextField("Title", text: turn.title)
                .help("what this step of the script is called — shown in the "
                    + "transcript and, for a contract turn, in the rendered "
                    + "prompt as {{turn.title}}")

            Picker("Speaking seat", selection: turn.speakerAgentID) {
                ForEach(panel.agents) { seat in
                    Text(seat.name).tag(seat.id)
                }
            }
            .help("which role generates this turn — its system prompt is what "
                + "the model is given as its identity")

            TextField("Output label", text: turn.outputLabel)
                .help("names this turn's output so later turns can ask for it "
                    + "— {{outputs.<label>}} in a template, or a tick box in a "
                    + "later contract turn's inputs")

            Picker("Route output to", selection: turn.routing) {
                ForEach(MultiAgentScenario.Turn.Routing.allCases, id: \.self) { routing in
                    Text(routing.label).tag(routing)
                }
            }
            .help("who is shown this output afterwards — all seats, nobody, or "
                + "the seats ticked below")

            if turn.wrappedValue.routing == .selected {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Seats that see this output")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(panel.agents) { seat in
                        Toggle(
                            seat.name,
                            isOn: Binding(
                                get: { turn.wrappedValue.routedAgentIDs.contains(seat.id) },
                                set: { include in
                                    if include {
                                        if !turn.wrappedValue.routedAgentIDs.contains(seat.id) {
                                            turn.wrappedValue.routedAgentIDs.append(seat.id)
                                        }
                                    } else {
                                        turn.wrappedValue.routedAgentIDs.removeAll { $0 == seat.id }
                                    }
                                }))
                            .help("show this turn's output to '\(seat.name)' in "
                                + "its later turns")
                    }
                }
            }

            Toggle("Include shared materials", isOn: turn.includeScenarioMaterials)
                .help("put the panel's Shared Materials in front of this "
                    + "speaker — a contract turn fences them under the "
                    + "materials title")
            Toggle("Include speaker context", isOn: turn.includeSpeakerContext)
                .help("include what this seat has already been routed — the "
                    + "conversation it can see, as opposed to the case record")

            // A per-turn budget IS environment content — "the vote is one
            // line, the opinion is long" is part of the protocol. Leaving it
            // unset defers to whatever budget the RUN declares (the study's,
            // or the rehearsal's), which is why the PLACEHOLDER names that
            // budget: showing it as a value made "no override" and "overridden
            // to the same number" look identical.
            LabeledContent("Max tokens") {
                HStack {
                    TextField(
                        "run's budget: \(panel.rehearsalMaxTokens)",
                        text: Binding<String>(
                            get: {
                                turn.wrappedValue.maxTokens.map(String.init) ?? ""
                            },
                            set: { text in
                                let trimmed = text.trimmingCharacters(
                                    in: .whitespaces)
                                if trimmed.isEmpty {
                                    turn.wrappedValue.maxTokens = nil
                                } else if let value = Int(trimmed) {
                                    turn.wrappedValue.maxTokens = max(1, value)
                                }
                                // Anything else is dropped: the field
                                // re-renders the last good value.
                            }))
                        .frame(maxWidth: 130)
                        .help("this turn's own token budget — leave it empty to "
                            + "use whatever budget the run declares (the "
                            + "study's, or the rehearsal's)")
                    if turn.wrappedValue.maxTokens != nil {
                        Button("Use the Run's Budget") {
                            turn.wrappedValue.maxTokens = nil
                        }
                        .buttonStyle(.borderless)
                        .help("clear this turn's override so it follows the "
                            + "run's budget again")
                    }
                }
            }

            turnPromptSection(turn)

            HStack {
                Button {
                    panel.moveTurn(id: turn.wrappedValue.id, by: -1)
                } label: {
                    Label("Move Up", systemImage: "arrow.up")
                }
                .disabled(index == nil || index == 0)
                .help("move this turn one step earlier in the script — a turn "
                    + "may only read outputs produced BEFORE it")
                Button {
                    panel.moveTurn(id: turn.wrappedValue.id, by: 1)
                } label: {
                    Label("Move Down", systemImage: "arrow.down")
                }
                .disabled(index == nil || index == panel.turns.count - 1)
                .help("move this turn one step later in the script — a turn "
                    + "may only read outputs produced BEFORE it")
                Spacer()
                Button("Remove Turn", role: .destructive) {
                    if turnCarriesWork(turn.wrappedValue) {
                        pendingRemoval = .turn(turn.wrappedValue)
                    } else {
                        panel.removeTurn(id: turn.wrappedValue.id)
                    }
                }
                .help("delete this step from the script — it asks first once "
                    + "the turn carries an authored prompt or contract")
            }
        }
        .disabled(panel.isLegacyBound)
    }

    /// The renderer choice and whichever editor it selects.
    ///
    /// Its own function so `turnEditor` stays inside SwiftUI's ten-child
    /// ViewBuilder limit — grouping, not a second layer of logic.
    @ViewBuilder
    private func turnPromptSection(
        _ turn: Binding<MultiAgentScenario.Turn>
    ) -> some View {
        @Bindable var panel = service.multiAgent
        Group {
            // Which renderer this turn goes through. One or the other, never
            // both — `MultiAgentRunner.validate` refuses a turn carrying a
            // contract AND a template, so the choice is a picker rather than
            // two independently editable fields.
            Picker(
                "Prompt",
                selection: Binding(
                    get: { turn.wrappedValue.contract != nil },
                    set: { panel.setTurnUsesContract(id: turn.wrappedValue.id, $0) })
            ) {
                Text("Free template").tag(false)
                Text("Contract").tag(true)
            }
            .pickerStyle(.segmented)
            .help("a contract turn declares its CONTENT — stage, task, format, "
                + "inputs — and the engine renders the canonical layout; a "
                + "template turn renders exactly the text you write")

            if let notes = panel.turnNotices[turn.wrappedValue.id], !notes.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(notes.enumerated()), id: \.offset) { _, note in
                        Text(note)
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .textSelection(.enabled)
                    }
                    Button("Dismiss") { panel.clearTurnNotices(id: turn.wrappedValue.id) }
                        .buttonStyle(.borderless)
                        .font(.caption)
                        .help("hide these notices about the last renderer "
                            + "switch — nothing in the turn changes")
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            }

            if turn.wrappedValue.contract != nil {
                contractEditor(turn)
            } else {
                TextEditor(text: turn.promptTemplate)
                    .font(.body.monospaced())
                    .frame(minHeight: 150)
                    .help("rendered exactly as typed, with the variables below "
                        + "substituted")
                // The one thing a template author must know, VISIBLE. It used
                // to exist only as hover text on the editor.
                Text("Variables: {{scenario.materials}} · {{scenario.name}} · "
                    + "{{scenario.description}} · {{agent.name}} · "
                    + "{{agent.context}} · {{turn.title}} · "
                    + "{{outputs.<label>}} for an earlier turn's output. "
                    + "Anything else is left as typed.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }

            promptPreview(turn)
        }
    }

    /// The contract half of the turn editor: content only.
    ///
    /// There is no field here for the arrangement — where the record sits
    /// relative to the instruction, how a colleague's output is fenced, where
    /// the own-voice constraint lands. That is the contract's whole point, so
    /// the layout appears as a read-only summary of what the engine WILL
    /// render, computed by `PanelAuthoring` from this turn's own state.
    @ViewBuilder
    private func contractEditor(
        _ turn: Binding<MultiAgentScenario.Turn>
    ) -> some View {
        @Bindable var panel = service.multiAgent
        let turnID = turn.wrappedValue.id

        LabeledContent("Stage") {
            TextField(
                "Stage",
                text: contractField(turn, \.stage),
                prompt: Text("where the scenario stands right now"),
                axis: .vertical)
                .lineLimit(1 ... 3)
        }
        .help("one or two sentences saying where the scenario stands, appended "
            + "to the identity opener")

        Text("Task — what this turn asks for. Required.")
            .font(.caption)
            .foregroundStyle(.secondary)
        TextEditor(text: contractField(turn, \.task))
            .font(.body.monospaced())
            .frame(minHeight: 90)
            .help("the instruction, rendered under \"===== YOUR TASK =====\" "
                + "after \"You are <name>.\" — only {{scenario.name}}, "
                + "{{scenario.description}}, {{agent.name}} and {{turn.title}} "
                + "substitute here")

        Text("Output format — rendered verbatim, after the own-voice block.")
            .font(.caption)
            .foregroundStyle(.secondary)
        TextEditor(text: contractField(turn, \.format))
            .font(.body.monospaced())
            .frame(minHeight: 60)
            .help("the shape the answer must take — rendered verbatim with no "
                + "substitution; leave it empty to ask for nothing in "
                + "particular")

        TextField("Materials fence title", text: contractField(turn, \.materialsTitle))
            .help("the fence label around the shared materials — \"THE RECORD "
                + "ON APPEAL\", verbatim")

        Toggle("Own-voice constraint", isOn: contractBool(turn, \.ownVoice))
            .help("add the block forbidding this speaker from writing on behalf "
                + "of the other participants — omitted automatically on a "
                + "one-seat panel")

        VStack(alignment: .leading, spacing: 4) {
            Text("Earlier outputs to show this speaker, fenced and attributed")
                .font(.caption)
                .foregroundStyle(.secondary)
            let available = panel.availableInputs(forTurn: turnID)
            if available.isEmpty {
                Text("No earlier turn produces an output yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(available) { input in
                Toggle(
                    input.summary,
                    isOn: Binding(
                        get: { turn.wrappedValue.contract?.inputs.contains(input.label) ?? false },
                        set: {
                            panel.setContractInput(
                                turnID: turnID, label: input.label, included: $0)
                        }))
                    .help("show '\(input.label)' to this speaker, fenced and "
                        + "attributed — it bypasses routing by design, so "
                        + "declaring it here is what makes it visible")
            }
            // A declared input whose producer has since been deleted or moved
            // after this turn: shown so `validate`'s refusal is not a surprise.
            let orphans = (turn.wrappedValue.contract?.inputs ?? [])
                .filter { label in !available.contains { $0.label == label } }
            ForEach(orphans, id: \.self) { label in
                Text("\(label) — no earlier turn produces this; validate refuses it")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }

        VStack(alignment: .leading, spacing: 4) {
            Text("The engine renders these blocks, in this order:")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(
                Array(panel.contractLayoutSummary(forTurn: turnID).enumerated()),
                id: \.offset
            ) { _, line in
                Text(line)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    /// The prompt this turn would render, from the RUNNER's own renderer.
    ///
    /// The height is a CONSTANT: variable content lives inside a scroll view
    /// rather than growing the container, because a SwiftUI minimum that
    /// changes while it is on screen is fatal on this macOS beta.
    @ViewBuilder
    private func promptPreview(
        _ turn: Binding<MultiAgentScenario.Turn>
    ) -> some View {
        @Bindable var panel = service.multiAgent
        let turnID = turn.wrappedValue.id
        Toggle(
            "Preview the rendered prompt",
            isOn: Binding(
                get: { previewedTurnID == turnID },
                set: { previewedTurnID = $0 ? turnID : nil }))
            .toggleStyle(.switch)
            .help("show the prompt this turn would send, rendered by the "
                + "engine's own renderer and not a second copy of it — "
                + "runtime-only text (earlier outputs, the routed transcript) "
                + "is stood in for")
        if previewedTurnID == turnID, let preview = panel.previewPrompt(forTurn: turnID) {
            ScrollView {
                Text(preview)
                    .font(.caption2.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .frame(height: 240)
            .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    /// A binding into one string field of a turn's contract. Writes are
    /// dropped when the turn has no contract — the picker is what creates and
    /// removes one, through `MultiAgentPanel.setTurnUsesContract`.
    private func contractField(
        _ turn: Binding<MultiAgentScenario.Turn>,
        _ keyPath: WritableKeyPath<TurnContract, String>
    ) -> Binding<String> {
        Binding(
            get: { turn.wrappedValue.contract?[keyPath: keyPath] ?? "" },
            set: { value in
                guard var contract = turn.wrappedValue.contract else { return }
                contract[keyPath: keyPath] = value
                turn.wrappedValue.contract = contract
            })
    }

    private func contractBool(
        _ turn: Binding<MultiAgentScenario.Turn>,
        _ keyPath: WritableKeyPath<TurnContract, Bool>
    ) -> Binding<Bool> {
        Binding(
            get: { turn.wrappedValue.contract?[keyPath: keyPath] ?? true },
            set: { value in
                guard var contract = turn.wrappedValue.contract else { return }
                contract[keyPath: keyPath] = value
                turn.wrappedValue.contract = contract
            })
    }

    @ViewBuilder
    private func liveRunSection(_ panel: MultiAgentPanel) -> some View {
        // Named for the activity it belongs to: Run/Stop live under
        // "Rehearsal" above, and this is where that rehearsal's turns appear.
        // Two names for one activity read as two features.
        Section("Rehearsal Transcript") {
            if let liveRunDirectory = panel.liveRunDirectory {
                Text(liveRunDirectory)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let liveActiveTurn = panel.liveActiveTurn {
                HStack {
                    ProgressView()
                        .controlSize(.small)
                    Text("\(liveActiveTurn.summary) · \(liveActiveTurn.output.count) chars")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if let failure = panel.liveRunFailure {
                Text(failure)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            }
            if !panel.liveRunWarnings.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(Array(panel.liveRunWarnings.enumerated()), id: \.offset) { _, warning in
                        Text(warning.message)
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .textSelection(.enabled)
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            }
            if panel.liveTurnResults.isEmpty {
                Text("Start a rehearsal above to watch turn outputs appear "
                    + "here.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(panel.liveTurnResults) { result in
                    DisclosureGroup("\(result.turnIndex). \(result.title) — \(result.speakerName)") {
                        LabeledContent("Output label", value: result.outputLabel)
                        LabeledContent("Routed to", value: result.routedAgentIDs.joined(separator: ", "))
                        Text(result.output)
                            .textSelection(.enabled)
                    }
                }
            }
        }
    }
}

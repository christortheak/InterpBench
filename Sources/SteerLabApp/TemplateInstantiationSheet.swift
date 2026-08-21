import ExperimentKit
import SwiftUI

/// What "instantiate a template" was opened on. Captured at open time, like
/// `RenameStudySheet`, so the sheet never re-reads the store while it is up.
struct TemplateInstantiationRequest: Identifiable {
    let id = UUID()
    let templateName: String
    /// Mirrored from the panel's Remote options so the totals line counts the
    /// jobs this batch will really create, not a default.
    let shardsPerStudy: Int
    let jobNoun: String
    /// False when no server is connected — "Mint & Submit All" is not offered,
    /// and the sheet says why rather than failing six times in a row.
    let canSubmit: Bool
    /// Occupants to preload as permutation rows — the Seats section's "Create
    /// permuted siblings…" arrives with the study's own casting in hand. Empty
    /// for the ordinary "instantiate this design" opening.
    var permuting: [SeatOccupant] = []
}

/// The cell table: one row per study to mint.
///
/// The sheet exists because the choice a researcher is actually making here is
/// a batch SIZE, and the row count hides it. A composition sweep over four
/// seats is five rows and, at twenty play-throughs against a two-arm manifest,
/// two hundred transcripts — so the totals line is not decoration, it is the
/// control that stops a click from queueing a week of cluster time.
///
/// Every rule it renders lives in `TemplateInstantiation` /
/// `TemplateBatchTotals` (ExperimentKit, unit-tested). This file decides
/// nothing.
@MainActor
struct TemplateInstantiationSheet: View {
    let request: TemplateInstantiationRequest
    let panel: ExperimentPanel

    @Environment(\.dismiss) private var dismiss
    @State private var model: TemplateInstantiation

    /// Tag for the baseline entry in a seat picker. Not "": an empty tag reads
    /// as "nothing selected", and an all-baseline casting is a real condition
    /// (the control composition), not an absence.
    private static let baselineTag = "\u{0}baseline"

    init(request: TemplateInstantiationRequest, panel: ExperimentPanel) {
        self.request = request
        self.panel = panel
        let instantiation = TemplateInstantiation(templateName: request.templateName)
        instantiation.shardsPerStudy = request.shardsPerStudy
        instantiation.jobNoun = request.jobNoun
        // Opened from a study's Seats section: the table arrives holding one
        // row per distinct re-seating of that study's cast.
        instantiation.preloadPermutations(occupants: request.permuting)
        _model = State(initialValue: instantiation)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            if let failure = model.loadFailure {
                Label(failure, systemImage: "xmark.octagon.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            } else {
                Form {
                    ForEach(model.advisories, id: \.self) { advisory in
                        Label(advisory, systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if model.template?.intent == .multiAgent {
                        presetsSection
                    }
                    rowsSection
                }
                .formStyle(.grouped)
            }
            totalsLine
            footer
        }
        .padding(18)
        // Constant frame: a sheet whose minimum size changes while it is on
        // screen is fatal on this macOS beta (see the split-view minimum-size
        // note) — the table scrolls inside instead.
        .frame(minWidth: 720, idealWidth: 860, minHeight: 520, idealHeight: 640)
    }

    // MARK: Header

    @ViewBuilder
    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            // What this sheet MAKES, named in the title: every row is a new
            // ordinary draft study, and "instantiate a template" did not say so.
            Text("New studies from '\(request.templateName)'")
                .font(.title2.weight(.semibold))
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            if let template = model.template {
                Text(templateSummary(template))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func templateSummary(_ template: StudyTemplate) -> String {
        var parts = [template.intent.displayName, template.study.modelID]
        if let file = template.study.taskPromptsFile, !file.isEmpty {
            parts.append(file)
        }
        if !model.seatIDs.isEmpty {
            parts.append("\(model.seatIDs.count) seats")
        }
        return parts.joined(separator: "  ·  ")
    }

    // MARK: Presets (multi-agent only)

    @ViewBuilder
    private var presetsSection: some View {
        @Bindable var model = model
        Section("Standard batches") {
            HStack {
                Picker("Treated agent", selection: $model.sweepAgentID) {
                    Text("baseline").tag(String?.none)
                    ForEach(model.agents) { agent in
                        Text(agent.artifact.name).tag(String?.some(agent.id))
                    }
                }
                Button("Add composition sweep") { model.addCompositionSweep() }
                    .disabled(model.sweepAgentID == nil)
            }
            .help(
                "the panel-propagation question in its minimal form: all seats "
                    + "baseline, then each seat treated on its own, then every "
                    + "seat treated — does one treated member move the group, "
                    + "and does the effect scale with how many carry it")

            HStack {
                Menu(permutationMenuLabel) {
                    ForEach(model.agents) { agent in
                        Toggle(
                            agent.artifact.name,
                            isOn: permutationBinding(agentID: agent.id))
                    }
                }
                Toggle("pad with baseline", isOn: $model.permutationPadsWithBaseline)
                    .help(
                        "fills the remaining seats with the unsteered model, so "
                            + "a two-agent set still casts a three-seat panel")
                Button("Add all permutations") { model.addAllPermutations() }
                    .disabled(model.permutationRefusal != nil)
            }
            if let refusal = model.permutationRefusal {
                Text(refusal)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("\(model.permutationCount) distinct casting(s) — swapping "
                    + "two identical occupants gives the same panel, so those "
                    + "are deduped rather than run (and double-counted) twice")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var permutationMenuLabel: String {
        let chosen = model.permutationAgentIDs.count
        return chosen == 0 ? "Agents to permute…" : "\(chosen) agent(s) chosen"
    }

    private func permutationBinding(agentID: String) -> Binding<Bool> {
        Binding(
            get: { model.permutationAgentIDs.contains(agentID) },
            set: { isOn in
                if isOn {
                    model.permutationAgentIDs.append(agentID)
                } else if let index = model.permutationAgentIDs.firstIndex(of: agentID) {
                    model.permutationAgentIDs.remove(at: index)
                }
            })
    }

    // MARK: The cell table

    @ViewBuilder
    private var rowsSection: some View {
        Section("Studies to mint") {
            ForEach(model.rows) { row in
                rowView(row)
            }
            Button {
                model.addRow()
            } label: {
                Label("Add Study", systemImage: "plus")
            }
            .disabled(model.isWorking)
            .help("one more study from this design, with its own casting")
        }
    }

    @ViewBuilder
    private func rowView(_ row: TemplateCellRow) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                TextField(model.defaultName(for: row), text: nameBinding(row: row.id))
                    .font(.caption.monospaced())
                    .help(
                        "the study directory this casting mints; empty takes the "
                            + "template's own naming, and a collision resolves "
                            + "with the usual -2 suffix")
                Spacer()
                Button(role: .destructive) {
                    model.removeRow(row.id)
                } label: {
                    Image(systemName: "minus.circle")
                }
                .buttonStyle(.borderless)
                .disabled(model.isWorking)
            }

            if model.template?.intent == .multiAgent {
                ForEach(model.seatIDs, id: \.self) { seat in
                    Picker(seat, selection: seatBinding(row: row.id, seat: seat)) {
                        Text("baseline").tag(Self.baselineTag)
                        ForEach(model.agents) { agent in
                            Text(agent.artifact.name).tag(agent.id)
                        }
                    }
                }
            } else {
                // A compare-agents row casts as many agents as it likes, so
                // the affordance is "Add agent" (a panel row has one picker
                // per SEAT instead — a casting fills every seat exactly once).
                HStack(spacing: 8) {
                    Menu("Add agent") {
                        ForEach(model.agents) { agent in
                            Toggle(
                                agent.artifact.name,
                                isOn: agentBinding(row: row.id, agentID: agent.id))
                        }
                    }
                    .fixedSize()
                    .help(
                        "each chosen agent becomes one arm; the run loop always "
                            + "prepends the unsteered baseline arm, so it is "
                            + "never picked here")
                    Text(agentMenuLabel(row))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            rowState(row)
        }
        .padding(.vertical, 2)
    }

    private func agentMenuLabel(_ row: TemplateCellRow) -> String {
        let names = row.agentIDs.compactMap { id in
            model.agents.first { $0.id == id }?.artifact.name
        }
        return names.isEmpty ? "baseline arm only" : names.joined(separator: ", ")
    }

    @ViewBuilder
    private func rowState(_ row: TemplateCellRow) -> some View {
        switch row.state {
        case .pending:
            if let refusal = model.refusal(for: row) {
                Label(refusal, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            } else if let advisory = model.advisory(for: row) {
                // Non-blocking: a zero-agent draft is legal and Load Only will
                // write it. Only submission requires a runnable casting.
                Label(advisory, systemImage: "info.circle")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            } else {
                Label("ready", systemImage: "checkmark.circle")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        case .minted(let study):
            Label("minted \(study)", systemImage: "doc.badge.plus")
                .font(.caption2)
                .foregroundStyle(.secondary)
        case .submitted(let study, let jobID):
            Label("submitted \(study) — job \(jobID)", systemImage: "paperplane")
                .font(.caption2.monospaced())
                .foregroundStyle(.green)
                .textSelection(.enabled)
        case .failed(let message):
            // Verbatim: `instantiate`'s refusals name the file that drifted and
            // say what to do about it, and a paraphrase costs the instruction.
            Label(message, systemImage: "xmark.octagon.fill")
                .font(.caption2)
                .foregroundStyle(.red)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }

    // MARK: Bindings into the row table

    private func nameBinding(row id: TemplateCellRow.ID) -> Binding<String> {
        Binding(
            get: { model.rows.first { $0.id == id }?.name ?? "" },
            set: { newValue in
                guard let index = model.rows.firstIndex(where: { $0.id == id })
                else { return }
                model.rows[index].name = newValue
            })
    }

    private func seatBinding(
        row id: TemplateCellRow.ID, seat: String
    ) -> Binding<String> {
        Binding(
            get: {
                guard let row = model.rows.first(where: { $0.id == id }),
                    case .agent(_, let path, _) = row.seating[seat] ?? .baseline
                else { return Self.baselineTag }
                return model.agents.first {
                    ModelVariantStore.relativePath(for: $0) == path
                }?.id ?? Self.baselineTag
            },
            set: { newValue in
                guard let index = model.rows.firstIndex(where: { $0.id == id })
                else { return }
                model.rows[index].seating[seat] =
                    newValue == Self.baselineTag
                    ? .baseline
                    : model.occupant(forAgentID: newValue)
            })
    }

    private func agentBinding(
        row id: TemplateCellRow.ID, agentID: String
    ) -> Binding<Bool> {
        Binding(
            get: {
                model.rows.first { $0.id == id }?.agentIDs.contains(agentID) ?? false
            },
            set: { isOn in
                guard let index = model.rows.firstIndex(where: { $0.id == id })
                else { return }
                if isOn {
                    model.rows[index].agentIDs.append(agentID)
                } else {
                    model.rows[index].agentIDs.removeAll { $0 == agentID }
                }
            })
    }

    // MARK: Totals and actions

    @ViewBuilder
    private var totalsLine: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(model.totals.summary)
                .font(.callout.weight(.medium))
                .fixedSize(horizontal: false, vertical: true)
            if let summary = model.lastSummary {
                Text(summary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
        }
    }

    @ViewBuilder
    private var footer: some View {
        HStack {
            if model.isWorking { ProgressView().controlSize(.small) }
            Spacer()
            Button("Close", role: .cancel) { dismiss() }
            Button("Load Only") {
                Task {
                    await model.mint()
                    panel.refresh()
                    // LOADS the first minted draft in the Studies editor: the
                    // point of minting without submitting is to keep editing,
                    // and leaving the researcher to find the study in a picker
                    // of thirty is the same as not opening it.
                    if let first = model.firstMintedStudy {
                        panel.selectedName = first
                    }
                    panel.note(model.lastSummary ?? "", severity: .success)
                    // Failures stay on screen to be read; a clean batch has
                    // nothing left to say here.
                    if model.lastMintWasClean { dismiss() }
                }
            }
            .disabled(!model.readyToMint || model.isWorking)
            .help(
                "mints one ordinary draft per row (shared batch id) and opens "
                    + "the first one in the Studies editor. Agents optional — a "
                    + "zero-agent draft is legal, and the study's readiness "
                    + "check surfaces the missing casting")
            Button("Load and Submit") {
                Task {
                    await model.mint(submit: { study in
                        await panel.submitStudyBundle(named: study)
                    })
                    panel.refresh()
                    if let first = model.firstMintedStudy {
                        panel.selectedName = first
                    }
                    panel.note(model.lastSummary ?? "", severity: .info)
                    if model.lastMintWasClean { dismiss() }
                }
            }
            .keyboardShortcut(.defaultAction)
            .disabled(!model.readyToSubmit || model.isWorking || !request.canSubmit)
            .help(submitHelp)
        }
    }

    /// Why "Load and Submit" is off, or what it does. Naming the specific
    /// blocker matters: "disabled" alone sends the researcher hunting between
    /// a missing server connection and a row with no agents cast.
    private var submitHelp: String {
        guard request.canSubmit else {
            return "no server connection — connect one in Compute, or Load "
                + "Only and submit from the study list"
        }
        guard model.readyToSubmit else {
            return "every study needs a runnable casting before the batch can "
                + "be queued — submitting a baseline-only arm spends cluster "
                + "time measuring nothing against nothing"
        }
        return "mints every row, then submits each minted draft to the server "
            + "IN TURN — one failed submission does not stop the rest, and the "
            + "summary names the ones that failed"
    }
}

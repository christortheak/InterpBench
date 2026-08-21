import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// The Data section's role-first New Dataset flow (WP-Data phase 2) — and,
/// since phase 4 retired the Concepts & Vectors builder's own from-scratch
/// concept form, the ONE way a dataset comes into existence in this app.
/// `NewDatasetButton` (below) is how every surface reaches it.
///
/// Three steps, in the order the decision actually has to be made:
///
/// 1. **Role** — what the dataset IS, and therefore what reads it.
/// 2. **Name + destination** — the canonical path, PREVIEWED before anything
///    is written, computed by `DatasetCreationPlanner` from the engine's own
///    path authorities.
/// 3. **Create by** — author it in the existing Concept Builder, or import
///    files that are copied into that destination after validation.
///
/// This view is thin on purpose: every destination, requirement, collision,
/// and parse rule lives in `DatasetCreationPlan`. The view renders the plan
/// and calls `apply`.
///
/// Layout note (macOS 27 beta hazard, project memory "split-view min-size
/// crashes"): a plain sheet with a CONSTANT frame — no split view, and no
/// minimum that varies with the step or the role.
struct DatasetCreationSheet: View {
    @Bindable var service: ChatService
    /// Hand back the `DatasetInventoryEntry.ID` of what was just created, so
    /// the inventory re-scans and lands on the new row.
    let onCreated: (String) -> Void
    /// The same routing seam the inventory's detail pane uses.
    let openInConceptBuilder: (String?) -> Void

    @Environment(\.dismiss) private var dismiss

    /// Constants — never step- or role-dependent (see the layout note).
    private static let sheetWidth: CGFloat = 660
    private static let sheetHeight: CGFloat = 560

    private enum Step: Int, CaseIterable {
        case role
        case destination
        case method

        var title: String {
            switch self {
            case .role: "What is this dataset?"
            case .destination: "Name and destination"
            case .method: "How do you want to create it?"
            }
        }
    }

    @State private var step: Step = .role
    @State private var role: DatasetRole?
    @State private var rawName = ""
    @State private var recipeFamily: DatasetRecipeFamily?
    @State private var neutralTarget: NeutralCorpusTarget = .normCalibration
    @State private var pairedFamily: VectorCatalog.PairedStimulusFamily?
    @State private var chosen: [DatasetFileSlot: URL] = [:]
    @State private var importingSlot: DatasetFileSlot?
    @State private var confirmingReplace = false
    @State private var failure: String?
    @State private var isApplying = false

    // MARK: Plan

    private var plan: DatasetCreationPlan? {
        guard let role else { return nil }
        return DatasetCreationPlanner.plan(
            DatasetCreationRequest(
                role: role,
                rawName: rawName,
                recipeFamily: role == .validationSet ? recipeFamily : nil,
                neutralTarget: role == .neutralCorpus ? neutralTarget : nil,
                pairedFamily: role == .pairedStimuli ? pairedFamily : nil))
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    switch step {
                    case .role: roleStep
                    case .destination: destinationStep
                    case .method: methodStep
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
            }
            .frame(maxHeight: .infinity)
            Divider()
            footer
        }
        .frame(width: Self.sheetWidth, height: Self.sheetHeight)
        .fileImporter(
            isPresented: Binding(
                get: { importingSlot != nil },
                set: { if !$0 { importingSlot = nil } }),
            allowedContentTypes: [.json, .plainText, .data]
        ) { result in
            let slot = importingSlot
            importingSlot = nil
            guard let slot, case .success(let url) = result else { return }
            chosen[slot] = url
            failure = nil
        }
        .alert("Replace existing file?", isPresented: $confirmingReplace) {
            Button("Replace", role: .destructive) { commit(replacing: true) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(replacementMessage)
        }
    }

    // MARK: Chrome

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text("New Dataset")
                    .font(.headline)
                Spacer()
                Text("Step \(step.rawValue + 1) of \(Step.allCases.count)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Text(step.title)
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var footer: some View {
        HStack(spacing: 10) {
            if let failure {
                Label(failure, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            if step != .role {
                Button("Back") { goBack() }
            }
            if step != .method {
                Button("Continue") { goForward() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canContinue)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var canContinue: Bool {
        switch step {
        case .role: role != nil
        case .destination: plan?.isResolved == true
        case .method: false
        }
    }

    private func goForward() {
        guard canContinue else { return }
        failure = nil
        switch step {
        case .role: step = .destination
        case .destination: step = .method
        case .method: break
        }
    }

    private func goBack() {
        failure = nil
        switch step {
        case .role: break
        case .destination: step = .role
        case .method: step = .destination
        }
    }

    // MARK: Step 1 — role

    private var roleStep: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(
                "Declare what the dataset is before it exists. The role picks "
                    + "the recipe that reads it and the one place it belongs — "
                    + "so a set filed here is a set the engine will find."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)

            ForEach(DatasetRole.allCases) { candidate in
                roleCard(candidate)
            }
        }
    }

    private func roleCard(_ candidate: DatasetRole) -> some View {
        let selected = role == candidate
        return Button {
            if role != candidate {
                role = candidate
                // A role change invalidates every downstream choice.
                chosen = [:]
                recipeFamily = nil
                neutralTarget = .normCalibration
                pairedFamily = nil
            }
            step = .destination
        } label: {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Image(
                        systemName: selected
                            ? "largecircle.fill.circle" : "circle")
                        .foregroundStyle(selected ? Color.accentColor : .secondary)
                    Text(candidate.title)
                        .font(.callout.weight(.medium))
                    Spacer()
                }
                Text(candidate.feeds)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(candidate.canonicalLocationHint)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(selected ? Color.accentColor.opacity(0.10) : Color.clear))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(
                        selected ? Color.accentColor : Color.secondary.opacity(0.25)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: Step 2 — name and destination

    @ViewBuilder
    private var destinationStep: some View {
        if let role {
            VStack(alignment: .leading, spacing: 14) {
                Label(role.title, systemImage: "square.stack.3d.up")
                    .font(.callout.weight(.medium))

                if role == .validationSet { recipeFamilyPicker }
                if role == .neutralCorpus { neutralTargetPicker }
                if role == .pairedStimuli { pairedFamilyPicker }

                if role.requiresName(
                    neutralTarget: role == .neutralCorpus ? neutralTarget : nil)
                {
                    nameField
                }

                destinationPreview
            }
        }
    }

    private var recipeFamilyPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Which recipe does this validate?")
                .font(.callout.weight(.medium))
            Text(
                "Not a label — it decides the canonical root. The paired "
                    + "recipes read prompts/concepts/, the grand-mean recipe "
                    + "reads prompts/emotions/."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            Picker("", selection: $recipeFamily) {
                Text("Choose…").tag(DatasetRecipeFamily?.none)
                ForEach(DatasetRecipeFamily.allCases) { family in
                    Text(family.label).tag(DatasetRecipeFamily?.some(family))
                }
            }
            .labelsHidden()
            .pickerStyle(.radioGroup)
            if let recipeFamily {
                Text(recipeFamily.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Paired sets: the family is not a label either — it picks the root
    /// AND the row shape the import is parsed with. Two `pairs.jsonl` files
    /// with the same name under the two roots are different datasets, and
    /// neither loader reads the other's rows.
    private var pairedFamilyPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Which paired recipe reads this?")
                .font(.callout.weight(.medium))
            Picker("", selection: $pairedFamily) {
                Text("Choose…").tag(VectorCatalog.PairedStimulusFamily?.none)
                ForEach(VectorCatalog.PairedStimulusFamily.allCases) { family in
                    Text(family.title)
                        .tag(VectorCatalog.PairedStimulusFamily?.some(family))
                }
            }
            .labelsHidden()
            .pickerStyle(.radioGroup)
            if let pairedFamily {
                Text(pairedFamily.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var neutralTargetPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Which neutral corpus?")
                .font(.callout.weight(.medium))
            Picker("", selection: $neutralTarget) {
                ForEach(NeutralCorpusTarget.allCases) { target in
                    Text(target.label).tag(target)
                }
            }
            .labelsHidden()
            .pickerStyle(.radioGroup)
        }
    }

    private var nameField: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(nameFieldLabel)
                .font(.callout.weight(.medium))
            TextField(nameFieldPlaceholder, text: $rawName)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 320)
            if let plan, plan.wasSanitized {
                Text("filed as \(plan.name)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help(
                        "directory names are sanitized to letters, numbers, "
                            + "and hyphens so the recipe and the manifest name "
                            + "the same thing")
            }
        }
    }

    /// Batteries are FLAT files in a shared directory, so the name is the
    /// filename rather than a folder — say which, so the preview below is
    /// not a surprise.
    private var nameFieldLabel: String {
        switch role {
        case .neutralCorpus: "Projection basis name"
        case .capabilityBattery: "Battery name (becomes the filename)"
        default: "Concept name"
        }
    }

    private var nameFieldPlaceholder: String {
        switch role {
        case .neutralCorpus: "assistant-dialogue-neutral"
        case .capabilityBattery: "study-guardrail"
        default: "sympathy"
        }
    }

    @ViewBuilder
    private var destinationPreview: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Destination")
                .font(.callout.weight(.medium))
            if let plan, plan.isResolved {
                ForEach(plan.files) { file in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Image(
                            systemName: file.exists
                                ? "exclamationmark.circle.fill" : "doc.badge.plus")
                            .foregroundStyle(file.exists ? .orange : .secondary)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(file.relativePath)
                                .font(.caption.monospaced())
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                            if file.exists {
                                Text(
                                    "already exists"
                                        + (file.existingByteSize.map {
                                            " · \(DatasetInventoryEntry.formattedSize($0))"
                                        } ?? ""))
                                    .font(.caption2)
                                    .foregroundStyle(.orange)
                            }
                        }
                        Spacer(minLength: 0)
                    }
                }
                if plan.verb != .create {
                    Text(verbNote(plan))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(plan.advisories, id: \.self) { advisory in
                    Label(advisory, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else if let plan {
                ForEach(plan.requirements.map(\.message), id: \.self) { message in
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func verbNote(_ plan: DatasetCreationPlan) -> String {
        switch plan.verb {
        case .create:
            ""
        case .addTo:
            "\(plan.directoryRelativePath) already exists — this ADDS to it. "
                + "Nothing already there is touched."
        case .replace:
            "A file is already filed here. Importing REPLACES it, and asks "
                + "before it does. Nothing is overwritten silently."
        }
    }

    // MARK: Step 3 — create by

    @ViewBuilder
    private var methodStep: some View {
        if let plan, plan.isResolved {
            VStack(alignment: .leading, spacing: 16) {
                summaryRow(plan)

                if plan.role.authorsInConceptBuilder {
                    authorInBuilderCard(plan)
                    Divider()
                }
                if plan.role == .storyCorpus {
                    llmAuthoringNote
                }
                if plan.role == .validationSet {
                    filingNote
                }
                if plan.role == .capabilityBattery {
                    batteryAuthoringNote
                }

                importCard(plan)
            }
        }
    }

    private func summaryRow(_ plan: DatasetCreationPlan) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(plan.role.title) · \(plan.name)")
                .font(.callout.weight(.medium))
            Text(plan.directoryRelativePath)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The hand-back to the builder. Offered for every role the Concept
    /// Builder can author ROWS for (`DatasetRole.authorsInConceptBuilder`) —
    /// which, since phase 4 retired the builder's own new-concept field, is
    /// the only way those datasets get started at all.
    private func authorInBuilderCard(_ plan: DatasetCreationPlan) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Author in Concept Builder")
                .font(.callout.weight(.medium))
            Text(
                "Creates \(plan.directoryRelativePath) and registers the "
                    + "concept — empty structure, no example rows — then opens "
                    + "Concepts & Vectors with it selected so you can paste, "
                    + "generate, or type the rows there."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            Button("Create and open the builder") { authorInBuilder(plan) }
                .disabled(isApplying)
        }
    }

    private var llmAuthoringNote: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Authoring the rows with an LLM")
                .font(.callout.weight(.medium))
            Text(
                "The recipe-specific prompts — including the Claude Cowork "
                    + "corpus-generation instructions — live in Concepts & "
                    + "Vectors under the grand-mean recipe (\"Copy LLM prompt\" "
                    + "and \"Copy Claude Cowork prompt\"). They are generated "
                    + "against the builder's current concept and recipe, so "
                    + "they are copied there rather than duplicated here. "
                    + "Import the JSONL you get back below."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The battery role's step-3 note: what to author, and the two verbs
    /// that author and check it. The engine has ONE format-2 shape checker
    /// (`PinShapeValidation`), and the server CLI exposes it — so the honest
    /// instruction is to use those, not to restate the schema here.
    private var batteryAuthoringNote: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Authoring and checking the battery")
                .font(.callout.weight(.medium))
            Text(
                "steerlab-server battery generation-prompt <file> emits the "
                    + "LLM prompt for a format-2 battery, and steerlab-server "
                    + "battery lint <file> reports every shape problem before "
                    + "the file is pinned. Import below applies the same "
                    + "check: the engine's own battery loader parses the "
                    + "file, so a header or item that would fail the pin is "
                    + "refused here instead of at freeze."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            Text(
                "Legacy headerless batteries still import and still run — "
                    + "their pinned hashes are unchanged. Format 2 is worth "
                    + "the repair because it declares its own scoring, prompt "
                    + "mode, and token cap, so a capability reading is "
                    + "comparable across instruments."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var filingNote: some View {
        Text(
            "The inventory labels validation sets with the root they were "
                + "found under, so a misfiled one is visible. This flow is how "
                + "they are filed right the first time."
        )
        .font(.caption)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func importCard(_ plan: DatasetCreationPlan) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(plan.files.count > 1 ? "Import files" : "Import file")
                .font(.callout.weight(.medium))
            Text(
                "The chosen bytes are copied to the canonical name and place, "
                    + "after the family's own parser accepts them. A malformed "
                    + "file is refused with its error and nothing is written."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)

            ForEach(plan.files) { file in
                slotRow(file)
            }

            HStack(spacing: 10) {
                Button(importButtonTitle(plan)) { attemptCommit(plan) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(chosen.isEmpty || isApplying)
                if isApplying {
                    ProgressView().controlSize(.small)
                }
                Spacer()
            }
            .padding(.top, 2)
        }
    }

    private func importButtonTitle(_ plan: DatasetCreationPlan) -> String {
        switch plan.verb {
        case .create: "Create dataset"
        case .addTo: "Add to dataset"
        case .replace: "Replace…"
        }
    }

    private func slotRow(_ file: DatasetPlannedFile) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("\(file.slot.label)  →  \(file.filename)")
                    .font(.caption.weight(.medium))
                if let source = chosen[file.slot] {
                    Text(source.lastPathComponent)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(source.path)
                } else {
                    Text(file.slot.rowShapeHint)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            Button(chosen[file.slot] == nil ? "Choose…" : "Change…") {
                importingSlot = file.slot
            }
            .controlSize(.small)
            if chosen[file.slot] != nil {
                Button("Clear") { chosen[file.slot] = nil }
                    .controlSize(.small)
            }
        }
    }

    // MARK: Actions

    /// Create the destination directory, drive the builder's OWN
    /// new-concept seam (`saveNewConcept` — kept precisely so this flow can
    /// call it after phase 4 retired its in-builder button), then route.
    /// `saveNewConcept` lands both of the panel's concept selections on the
    /// new concept.
    ///
    /// The concept is registered under `prompts/concepts/<name>/` even when
    /// the role's own destination is elsewhere (a story corpus, a probe set,
    /// a paired mirror): that metadata file is the concept PRIMITIVE, which
    /// is exactly what the retired button wrote, and it is what makes the
    /// name selectable in the builder a moment later. No dataset row is
    /// invented — `DatasetInventory` lists a concept-stimuli row only when
    /// positive/negative actually exist.
    private func authorInBuilder(_ plan: DatasetCreationPlan) {
        do {
            try plan.createDirectory()
        } catch {
            failure = "\(error)"
            return
        }
        let builder = service.concepts
        builder.conceptName = plan.name
        builder.saveNewConcept()
        openInConceptBuilder(plan.name)
        onCreated(plan.inventoryEntryID)
        dismiss()
    }

    private func attemptCommit(_ plan: DatasetCreationPlan) {
        let replacing = plan.files.contains { chosen[$0.slot] != nil && $0.exists }
        if replacing {
            confirmingReplace = true
        } else {
            commit(replacing: false)
        }
    }

    private var replacementMessage: String {
        guard let plan else { return "" }
        let paths = plan.files
            .filter { chosen[$0.slot] != nil && $0.exists }
            .map(\.relativePath)
            .joined(separator: ", ")
        return "This overwrites \(paths) with the file you chose. Vectors and "
            + "runs already extracted from the old bytes keep their recorded "
            + "hash, so they stay traceable — but a frozen experiment pinned "
            + "to those bytes will now fail verification."
    }

    private func commit(replacing: Bool) {
        guard let plan else { return }
        isApplying = true
        failure = nil
        // The file-importer hands back security-scoped URLs; the copy happens
        // inside `apply`, so access is held across the whole call.
        let scoped = chosen.values.filter { $0.startAccessingSecurityScopedResource() }
        defer {
            for url in scoped { url.stopAccessingSecurityScopedResource() }
            isApplying = false
        }
        do {
            let outcome = try plan.apply(
                imports: chosen, replacingExisting: replacing)
            onCreated(outcome.inventoryEntryID)
            dismiss()
        } catch {
            failure = "\(error)"
        }
    }
}

/// The ONE way to open the role-first New Dataset flow.
///
/// Phase 2 put the flow behind the Inventory header's button; phase 4 made it
/// the one creation entry in the Data section, which means the Concepts &
/// Vectors tool needs it too — its own from-scratch concept field was
/// retired. Rather than a second `@State private var showCreationSheet` and a
/// second `.sheet` modifier per surface, the presentation is packaged here:
/// one button, one sheet, one set of callbacks.
///
/// `openInConceptBuilder` is the caller's routing seam because the two
/// surfaces mean different things by it — from the Inventory it also switches
/// the tool tab, from inside the builder it only moves the selection.
struct NewDatasetButton: View {
    @Bindable var service: ChatService
    /// The `DatasetInventoryEntry.ID` of what was created — the Inventory
    /// lands on that row; the builder ignores it and just re-scans.
    let onCreated: (String) -> Void
    let openInConceptBuilder: (String?) -> Void
    var title = "New Dataset…"
    var systemImage: String? = "plus"
    var help =
        "declare what the dataset is, then file it in the one place its "
        + "recipe reads"

    @State private var isPresented = false

    var body: some View {
        Button {
            isPresented = true
        } label: {
            if let systemImage {
                Label(title, systemImage: systemImage)
            } else {
                Text(title)
            }
        }
        .help(help)
        .sheet(isPresented: $isPresented) {
            DatasetCreationSheet(
                service: service,
                onCreated: onCreated,
                openInConceptBuilder: openInConceptBuilder)
        }
    }
}

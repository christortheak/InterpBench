import AppKit
import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// "Data Readiness" — what study data the selected manifest still needs and
/// where each file goes, with one-click template scaffolding and a
/// field-preserving JSONL editor. Thin view: every rule lives in
/// `StudyDataReadiness` / `DataTemplates` (ExperimentKit, unit-tested); this
/// renders the derived rows and forwards actions.
struct DataReadinessSection: View {
    let manifest: ExperimentManifest
    /// Task prompts already have a full editor in the Input Data section —
    /// the row's Edit button routes there instead of the minimal sheet.
    var onEditTaskPrompts: () -> Void = {}
    /// Categories relevant to the current Study Type. Groups outside this
    /// set are hidden ONLY when every row in them is `.optional` — a group
    /// carrying real data or a blocker always renders.
    var relevantCategories: Set<DataCategory> = Set(DataCategory.allCases)
    /// When present, the comparison group offers the human-baseline
    /// pin/unpin controls (moved here from the dissolved Science Manifest
    /// — the pin is data curation, so it lives with the data).
    var panel: ExperimentPanel?
    /// The task-prompts CONTENT editor (owned by the panel view — it
    /// carries the import-JSONL sheet state), rendered collapsed under the
    /// task-prompts row; the row's Edit button expands it. One pane: the
    /// row tracks the file, the disclosure edits the same file.
    var taskPromptsEditor: (() -> AnyView)?

    @State private var requirements: [DataRequirement] = []
    @State private var message: String?
    @State private var editTarget: JSONLEditTarget?
    @State private var showTaskPromptsEditor = false
    @State private var batteryPathField = ""
    @State private var humanValidationPathField = ""

    private var summary: ReadinessSummary { StudyDataReadiness.summary(requirements) }

    /// Blockers first: invalid (red — the run refuses the file), missing
    /// (red), partial (amber), present (green), optional (grey).
    private static let displayOrder: [DataRequirement.Status] = [
        .invalid, .missing, .partial, .present, .optional,
    ]

    var body: some View {
        Section {
            HStack {
                Text(summary.line)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.plain)
                .help("re-scan the workspace for the files this study needs")
            }
            // One pane, subdivided by what each file IS FOR (2026-07-19
            // feedback) — blockers first within each group.
            ForEach(DataCategory.allCases) { category in
                let rows = groupRows(category)
                if !rows.isEmpty {
                    HStack(spacing: 6) {
                        Text(category.title)
                            .font(.caption.bold())
                        InfoButton(text: StudyInfo.dataCategory(category))
                    }
                    .padding(.top, 4)
                    ForEach(rows) { requirement in
                        requirementRow(requirement)
                    }
                    if category == .taskPrompts, let taskPromptsEditor {
                        DisclosureGroup(
                            "Edit prompt contents",
                            isExpanded: $showTaskPromptsEditor
                        ) {
                            taskPromptsEditor()
                        }
                        .font(.caption)
                    }
                    if category == .comparison, let panel {
                        humanBaselineControls(panel: panel)
                    }
                    if category == .battery, panel != nil {
                        batteryControls
                    }
                }
            }
            if let message {
                Text(message)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        } header: {
            InfoSectionHeader(
                title: "Data & Prompts", text: StudyInfo.dataPrompts)
        }
        .onAppear { refresh() }
        .onChange(of: manifest) { refresh() }
        // Review 2026-08-01 (P2): the pin path fields seeded only on
        // appear, so switching studies while the view stayed mounted could
        // display — and pin — the PREVIOUS study's path.
        .onChange(of: manifest.name) {
            batteryPathField = manifest.capabilityBatteryFile ?? ""
            humanValidationPathField = manifest.humanValidation?.path ?? ""
        }
        .sheet(item: $editTarget) { target in
            JSONLRecordEditorSheet(target: target) {
                editTarget = nil
                refresh()
            }
        }
    }

    private func refresh() {
        requirements = StudyDataReadiness.requirements(
            for: manifest, workspaceRoot: VectorCatalog.projectRoot,
            workspaceIsServer: panel?.isServerWorkspace ?? false)
    }

    /// A category's rows, blockers first. Categories outside the current
    /// focus render only when they carry something real (never hide a
    /// blocker or existing data behind a view filter).
    private func groupRows(_ category: DataCategory) -> [DataRequirement] {
        let rows = requirements.filter {
            $0.kind.authoringCategory == category
        }
        guard !rows.isEmpty else { return [] }
        if !relevantCategories.contains(category),
            rows.allSatisfy({ $0.status == .optional })
        {
            return []
        }
        return Self.displayOrder.flatMap { status in
            rows.filter { $0.status == status }
        }
    }

    private func color(_ status: DataRequirement.Status) -> Color {
        switch status {
        case .present: .green
        case .partial: .orange
        case .invalid: .red
        case .missing: .red
        case .optional: .secondary
        }
    }

    private func resolvedURL(_ requirement: DataRequirement) -> URL {
        requirement.path.hasPrefix("/")
            ? URL(filePath: requirement.path)
            : VectorCatalog.projectRoot.appending(path: requirement.path)
    }

    /// The on-disk file behind a row, when it exists (stimuli rows carry an
    /// annotated path; use its directory part for reveal).
    private func revealURL(_ requirement: DataRequirement) -> URL? {
        var url = resolvedURL(requirement)
        if requirement.kind == .conceptStimuli {
            // Path is annotated ("…/positive.jsonl (+ negative.jsonl)");
            // reveal the concept directory instead.
            url = url.deletingLastPathComponent()
        }
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    @ViewBuilder
    private func requirementRow(_ requirement: DataRequirement) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Image(systemName: "circle.fill")
                    .font(.system(size: 8))
                    .foregroundStyle(color(requirement.status))
                Text(requirement.title)
                    .font(.caption)
                Text(requirement.status.rawValue)
                    .font(.caption2)
                    .foregroundStyle(color(requirement.status))
                Spacer()
                rowButtons(requirement)
            }
            Text(requirement.detail)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.leading, 16)
            if requirement.kind == .conceptStimuli {
                // Annotated path ("… (+ negative.jsonl)") — plain text;
                // the row's folder button reveals the concept directory.
                Text(requirement.path)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
                    .padding(.leading, 16)
            } else {
                // Every data source opens IN the app (eye = read-only
                // viewer with hash), reveals in Finder (folder), and opens
                // in the default editor (pencil) — 2026-07-19 feedback.
                FileReferenceRow(
                    label: "file", path: requirement.path,
                    allowsOpenInEditor: true)
                    .padding(.leading, 16)
            }
        }
    }

    @ViewBuilder
    private func rowButtons(_ requirement: DataRequirement) -> some View {
        let onDisk = revealURL(requirement)
        if requirement.templateID != nil, onDisk == nil {
            Button("Create from template") {
                do {
                    let created = try StudyDataReadiness.scaffold(
                        requirement: requirement, in: VectorCatalog.projectRoot)
                    // Some requirements are satisfied by the FILE; the panel
                    // script is satisfied by the PIN, so creating it without
                    // pinning left the blocker standing with the file right
                    // there. Pin it here and say so.
                    let pinned = panel?.pinScaffoldedFile(
                        requirement: requirement, createdPath: created.path) ?? false
                    message = pinned
                        ? "created and pinned \(created.path) — replace the example "
                            + "content with your study's data"
                        : "created \(created.path) — replace the example rows "
                            + "with your study's data"
                    refresh()
                } catch {
                    message = "\(error)"
                }
            }
            .font(.caption)
            .help("copy the \(requirement.templateID ?? "") template to "
                + "\(requirement.path) (never overwrites)")
        }
        if let onDisk {
            if isJSONLEditable(requirement) {
                Button("Edit…") {
                    if requirement.kind == .taskPrompts {
                        onEditTaskPrompts()
                        showTaskPromptsEditor = true
                    } else {
                        editTarget = JSONLEditTarget(
                            title: requirement.title, url: onDisk)
                    }
                }
                .font(.caption)
                .help(requirement.kind == .taskPrompts
                    ? "load this file into the prompt-contents editor just below"
                    : "edit record texts; all other fields are preserved on save")
            }
            if requirement.kind == .conceptStimuli {
                // Non-stimuli rows get folder/eye/pencil from their
                // FileReferenceRow; the stimuli row reveals its directory.
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([onDisk])
                } label: {
                    Image(systemName: "folder")
                }
                .buttonStyle(.plain)
                .help("reveal the concept directory in Finder")
            }
        }
    }

    private func isJSONLEditable(_ requirement: DataRequirement) -> Bool {
        switch requirement.kind {
        case .taskPrompts, .conceptValidation: true
        default: false
        }
    }

    /// Pin/unpin a NAMED capability battery, right under the row that
    /// tracks it. `ExperimentStore.pinCapabilityBattery(_:into:)` existed
    /// since the study-pack importer but had no picker (audit 2026-08-01:
    /// a UI-orphaned setter) — an unpinned study silently runs the engine
    /// default battery, which is fine until a study means to run its own.
    /// Writes through the draft-gated `setCapabilityBatteryFile`.
    @ViewBuilder
    private var batteryControls: some View {
        let isDraft = manifest.status == .draft
        HStack(spacing: 8) {
            TextField(
                "battery JSONL path (prompts/batteries/…)",
                text: $batteryPathField)
            WorkspacePathChooseButton(
                message: "Choose the capability-battery JSONL (workspace "
                    + "files only — the path pins on selection)",
                allowedTypes: [.json, .plainText],
                startingSubdirectory: "prompts/batteries",
                onChoose: { relativePath in
                    batteryPathField = relativePath
                    pinBattery()
                },
                onProblem: { message = $0 })
                .disabled(!isDraft)
            Button("Pin") { pinBattery() }
                .disabled(!isDraft
                    || batteryPathField
                        .trimmingCharacters(in: .whitespaces).isEmpty)
                .help(
                    "pins the battery at its CURRENT SHA-256 after shape "
                        + "validation — drift afterwards is a verify finding")
            if manifest.capabilityBatteryFile != nil {
                Button("Unpin") { setBattery(nil) }
                    .disabled(!isDraft)
                    .help(
                        "clears the named battery — arms fall back to the "
                            + "engine's default battery")
            }
        }
        .font(.caption)
        .disabled(!isDraft)
        .onAppear { batteryPathField = manifest.capabilityBatteryFile ?? "" }
    }

    private func pinBattery() {
        let trimmed = batteryPathField.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        setBattery(trimmed)
    }

    private func setBattery(_ file: String?) {
        do {
            _ = try ExperimentStore.setCapabilityBatteryFile(
                file, experimentName: manifest.name)
            if file == nil { batteryPathField = "" }
            message = nil
            panel?.refresh()
            refresh()
        } catch {
            message = "Couldn't pin the battery: \(error)"
        }
    }

    /// Pin/unpin the human-baseline CSV, right under the row that tracks
    /// it. Pinning records the file's CURRENT SHA-256; unpinning is an
    /// explicit button (never a side effect of an emptied text field).
    @ViewBuilder
    private func humanBaselineControls(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        let isDraft = manifest.status == .draft
        // The row ABOVE tracks the currently pinned CSV (if any); this
        // field is how a pin is made — type/paste the CSV's path, press
        // Pin, and its SHA-256 is recorded (2026-07-19: the field read as
        // a mysterious duplicate of the row).
        Text("To pin (or repin after edits): enter the CSV's "
            + "workspace-relative path, then Pin.")
            .font(.caption2)
            .foregroundStyle(.secondary)
        HStack(spacing: 8) {
            TextField(
                "human-baseline CSV path (prompts/baselines/…)",
                text: $panel.humanBaselinePathField)
            // Phase 3 item 12: pick the CSV instead of typing its path —
            // the choice writes the workspace-relative path AND makes the
            // pin through the same shape-validating pin flow.
            WorkspacePathChooseButton(
                message: "Choose the human-baseline CSV (workspace files "
                    + "only — the path pins on selection)",
                allowedTypes: [.commaSeparatedText, .plainText],
                startingSubdirectory: "prompts/baselines",
                onChoose: { relativePath in
                    panel.humanBaselinePathField = relativePath
                    panel.repinHumanBaseline()
                },
                onProblem: { message = $0 })
                .disabled(!isDraft)
            Button("Pin") { panel.repinHumanBaseline() }
                .disabled(!isDraft
                    || panel.humanBaselinePathField
                        .trimmingCharacters(in: .whitespaces).isEmpty)
                .help(
                    "pins the file at its CURRENT SHA-256 — drift afterwards "
                        + "is a readiness/verify finding")
            if manifest.humanBaseline != nil {
                Button("Unpin") { panel.clearHumanBaseline() }
                    .disabled(!isDraft)
                    .help(
                        "removes the pin — results stay model-internal "
                            + "without a human baseline (R claims need one)")
            }
        }
        .disabled(!isDraft)
        // Phase 3 item 13: a paper's effect table (JSON array / CSV with
        // any column names) enters through a column-mapping sheet and
        // lands as the loader-format CSV, pinned — no hand-built CSV.
        TabularImportButton(
            target: .humanBaseline, panel: panel, disabled: !isDraft)
        // The human-validation subset (per-judge vs-human agreement rows).
        // Until 2026-08-01 this pin had NO writer in the app — display
        // only, set by pasted JSON. Now: same pin/unpin discipline as the
        // baseline above, shape-validated at pin time (JSONL rows
        // {"condition", "promptID", "outcome": baseline|variant|tie}).
        if let humanValidation = manifest.humanValidation {
            FileReferenceRow(
                label: "human validation",
                path: humanValidation.path,
                pinnedHash: humanValidation.hash)
        }
        HStack(spacing: 8) {
            TextField(
                "human-validation JSONL path (optional — per-judge "
                    + "vs-human agreement)",
                text: $humanValidationPathField)
            WorkspacePathChooseButton(
                message: "Choose the human-validation JSONL (workspace "
                    + "files only — the path pins on selection)",
                allowedTypes: [.json, .plainText],
                startingSubdirectory: "prompts",
                onChoose: { relativePath in
                    humanValidationPathField = relativePath
                    pinHumanValidation()
                },
                onProblem: { message = $0 })
                .disabled(!isDraft)
            Button("Pin") { pinHumanValidation() }
                .disabled(!isDraft
                    || humanValidationPathField
                        .trimmingCharacters(in: .whitespaces).isEmpty)
                .help(
                    "pins the subset at its CURRENT SHA-256 after shape "
                        + "validation — when pinned, the evaluation report "
                        + "adds per-judge vs-human agreement (percent "
                        + "agreement, Cohen's kappa)")
            if manifest.humanValidation != nil {
                Button("Unpin") { clearHumanValidation() }
                    .disabled(!isDraft)
                    .help(
                        "removes the pin — evaluation keeps inter-judge "
                            + "agreement but loses the vs-human column")
            }
        }
        .font(.caption)
        .disabled(!isDraft)
        .onAppear {
            humanValidationPathField = manifest.humanValidation?.path ?? ""
        }
    }

    private func pinHumanValidation() {
        let trimmed = humanValidationPathField
            .trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        do {
            _ = try ExperimentStore.pinHumanValidation(
                path: trimmed, experimentName: manifest.name)
            message = nil
            panel?.refresh()
            refresh()
        } catch {
            message = "Couldn't pin the human-validation subset: \(error)"
        }
    }

    private func clearHumanValidation() {
        do {
            _ = try ExperimentStore.clearHumanValidation(
                experimentName: manifest.name)
            humanValidationPathField = ""
            message = nil
            panel?.refresh()
            refresh()
        } catch {
            message = "Couldn't unpin the human-validation subset: \(error)"
        }
    }
}

/// Sheet target: a JSONL file to edit through the field-preserving document.
struct JSONLEditTarget: Identifiable {
    let title: String
    let url: URL
    var id: String { url.path }
}

/// Minimal field-preserving JSONL editor for readiness rows that have no
/// dedicated editor (e.g. validation sets): records load through
/// `TaskPromptsDocument`, only the text is editable, and every other key of
/// each line is written back verbatim on save.
struct JSONLRecordEditorSheet: View {
    let target: JSONLEditTarget
    var onDone: () -> Void

    @State private var document: TaskPromptsDocument?
    @State private var editorText = ""
    @State private var status: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(target.title)
                .font(.headline)
            Text(target.url.path)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            TextEditor(text: $editorText)
                .font(.system(.caption, design: .monospaced))
                .frame(minWidth: 520, minHeight: 320)
            if let summary = document?.instrumentSummary {
                Label(summary, systemImage: "list.bullet.rectangle")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            Text("One record per block; separate records with a line containing "
                + "only ---. Non-text fields of each record are preserved on save. "
                + "If this file is pinned by a manifest, saving changes will "
                + "surface as drift until re-attached.")
                .font(.caption2)
                .foregroundStyle(.secondary)
            if let status {
                Text(status)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Spacer()
                Button("Cancel") { onDone() }
                Button("Save") { save() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(document == nil)
            }
        }
        .padding()
        .onAppear { load() }
    }

    private func load() {
        do {
            let data = try Data(contentsOf: target.url)
            let loaded = try TaskPromptsDocument.load(data)
            document = loaded
            editorText = loaded.editorText
        } catch {
            status = "cannot load: \(error)"
        }
    }

    private func save() {
        guard let document else { return }
        do {
            let blocks = TaskPromptsDocument.editorBlocks(editorText)
            let updated = document.applyingEditedTexts(blocks)
            try updated.serialized().write(to: target.url, options: .atomic)
            onDone()
        } catch {
            status = "cannot save: \(error)"
        }
    }
}

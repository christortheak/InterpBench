import AppKit
import ExperimentKit
import SwiftUI

struct FineTuningPanelView: View {
    @Bindable var service: ChatService
    /// Item 2 (cluster-testing): a server LoRA training submission parked
    /// while the shared no-GPU-session dialog asks.
    @State private var pendingModelJob: PendingModelJob?

    private var panel: FineTuningPanel { service.fineTuning }

    var body: some View {
        @Bindable var panel = service.fineTuning
        Form {
            Section("New Adapter") {
                // Workspace-scoped: local tiers in Local, the server's
                // installed models in a server workspace.
                Picker("Base model", selection: $panel.newAdapterBaseModelID) {
                    scopedModelRows(current: panel.newAdapterBaseModelID)
                }
                .help("Choose the base model this adapter will be trained for — restricted to the active workspace's installed models. Adapters are model-specific and should not be reused across different base models.")
                if service.cluster.computeTarget == .server {
                    HStack {
                        InstallModelButton(cluster: service.cluster)
                        Spacer()
                    }
                }
                TextField("Name", text: $panel.newAdapterName)
                    .textFieldStyle(.roundedBorder)
                    .help("Name for the adapter project and artifact. Use a stable research label, such as the corpus, method, or intended intervention.")
                DirectoryPathRow(
                    title: "Project directory",
                    path: resolvedDisplayPath(panel.newAdapterProjectDirectory),
                    placeholder: "Default: adapters/ in the workspace",
                    chooseTitle: "Choose...",
                    help: "Parent folder where SteerLab will create the adapter's home. Leave blank to store it under the workspace's adapters/ folder."
                ) {
                    openFolder(panel.newAdapterProjectDirectory)
                } choose: {
                    chooseDirectory { panel.newAdapterProjectDirectory = $0 }
                }
                Button("Create Adapter") { panel.createAdapterProject() }
                    .help("create adapters/<name>/ in the workspace with training/ and validation/ data folders, and add the adapter to the library")
            }

            Section("Adapter Library") {
                if panel.adapters.isEmpty {
                    Text("No adapter artifacts registered.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .help("Adapter projects you create or register will appear here.")
                } else {
                    ForEach(panel.adapters) { adapter in
                        Button {
                            panel.chooseAdapter(adapter)
                        } label: {
                            adapterRow(adapter, selected: panel.selectedAdapterID == adapter.id)
                        }
                        .buttonStyle(.plain)
                        .help("Select this adapter artifact to inspect or edit its training settings.")
                    }
                }
            }

            Section("Adapter Trainer") {
                if let selected = panel.selectedAdapter {
                    LabeledContent("Artifact", value: selected.artifact.name)
                        .help("The saved adapter artifact currently loaded into the trainer panel.")
                    Picker("Base model", selection: $panel.adapterBaseModelID) {
                        scopedModelRows(current: panel.adapterBaseModelID)
                    }
                    .help("Base model this adapter belongs to — restricted to the active workspace's installed models. Changing this changes provenance; the trained adapter should only be applied to the matching model.")
                    TextField("Name", text: $panel.adapterName)
                        .textFieldStyle(.roundedBorder)
                        .help("Editable display name stored in the adapter artifact sidecar.")
                    DirectoryPathRow(
                        title: "Project",
                        path: resolvedDisplayPath(panel.adapterProjectDirectory),
                        placeholder: "No project directory",
                        chooseTitle: nil,
                        help: "Root folder for this adapter project. Click the folder label to reveal it in Finder."
                    ) {
                        openFolder(panel.adapterProjectDirectory)
                    } choose: {}
                    DirectoryPathRow(
                        title: "Adapter output",
                        path: resolvedDisplayPath(panel.adapterDirectory),
                        placeholder: "Choose adapter output folder",
                        chooseTitle: "Choose...",
                        help: "Folder where the trained adapter files live. A completed MLX adapter should contain adapter_config.json and adapters.safetensors."
                    ) {
                        openFolder(panel.adapterDirectory)
                    } choose: {
                        chooseDirectory { panel.adapterDirectory = $0 }
                    }
                    DirectoryPathRow(
                        title: "Training workspace",
                        path: resolvedDisplayPath(panel.trainingWorkspacePath),
                        placeholder: "Choose training workspace folder",
                        chooseTitle: "Choose...",
                        help: "Workspace for raw source material and instruction templates before they are converted into train/validation datasets."
                    ) {
                        openFolder(panel.trainingWorkspacePath)
                    } choose: {
                        chooseDirectory { panel.trainingWorkspacePath = $0 }
                    }
                    DirectoryPathRow(
                        title: "Training data",
                        path: directoryDisplayPath(panel.trainingDataPath),
                        placeholder: "Choose training data folder",
                        chooseTitle: "Choose...",
                        help: "Folder containing the examples the adapter learns from (.txt, .md, .json, .jsonl, .pdf). Drop files onto this row to copy them into the folder."
                    ) {
                        openFolder(panel.trainingDataPath)
                    } choose: {
                        chooseDirectory { panel.trainingDataPath = $0 }
                    }
                    .dropDestination(for: URL.self) { urls, _ in
                        panel.importDroppedFiles(urls, to: .training)
                        return true
                    }
                    DirectoryPathRow(
                        title: "Validation data",
                        path: directoryDisplayPath(panel.validationDataPath),
                        placeholder: "Choose validation data folder",
                        chooseTitle: "Choose...",
                        help: "Folder containing held-out examples used to monitor validation loss. Drop files onto this row to copy them into the folder."
                    ) {
                        openFolder(panel.validationDataPath)
                    } choose: {
                        chooseDirectory { panel.validationDataPath = $0 }
                    }
                    .dropDestination(for: URL.self) { urls, _ in
                        panel.importDroppedFiles(urls, to: .validation)
                        return true
                    }

                    Divider()

                    Picker("Training data type", selection: $panel.trainingMode) {
                        ForEach(FineTuneTrainingMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .help("Document adaptation trains next-token prediction over plain document chunks. Instruction/chat tuning expects structured user/assistant JSONL and trains loss only on assistant tokens.")
                    Picker("Adapter type", selection: $panel.fineTuneType) {
                        ForEach(FineTuningPanel.FineTuneType.allCases) { type in
                            Text(type.label).tag(type)
                        }
                    }
                    .help("Choose the adapter training method. LoRA is the default low-rank adapter; DoRA is reserved for compatible training support.")
                    VStack(alignment: .leading, spacing: 10) {
                        IntSliderField(
                            "Rank",
                            value: $panel.rank,
                            range: 1 ... 128,
                            help: "LoRA rank controls adapter capacity. Higher rank can learn more but uses more memory and is easier to overfit.")
                        DoubleSliderField(
                            "Scale",
                            value: $panel.scale,
                            range: 0.1 ... 128,
                            step: 0.5,
                            fractionDigits: 1,
                            help: "LoRA scale controls the strength of the learned low-rank update during training and inference.")
                        IntSliderField(
                            "Layers",
                            value: $panel.adaptedLayers,
                            range: 1 ... 128,
                            help: "Number of transformer layers to adapt. Fewer layers are cheaper and more localized; more layers give the adapter more reach.")
                        IntSliderField(
                            "Batch",
                            value: $panel.batchSize,
                            range: 1 ... 32,
                            help: "Training batch size. Larger batches can be steadier but require more memory.")
                        IntSliderField(
                            "Iterations",
                            value: $panel.iterations,
                            range: 1 ... 100_000,
                            help: "Number of optimizer steps. More iterations train longer but increase overfitting risk.")
                        LogDoubleSliderField(
                            "Learning rate",
                            value: $panel.learningRate,
                            range: 0.000001 ... 0.001,
                            fractionDigits: 7,
                            help: "Optimizer learning rate on a logarithmic slider. 1e-5 is a conservative default for LoRA fine-tuning.")
                    }
                    HStack {
                        Button("Analyze Training Plan") { panel.analyzeTrainingPlan() }
                            .help("Inspect the selected train/validation files, estimate data size, and prepare hyperparameter recommendations.")
                        Button("Apply Plan") { panel.applyTrainingPlan() }
                            .disabled(panel.trainingPlan == nil)
                            .help("Copy the latest recommended hyperparameters into the trainer controls. You can still edit them afterward.")
                    }
                    if let plan = panel.trainingPlan {
                        TrainingPlanView(plan: plan)
                    }
                    TextField("Notes", text: $panel.notes)
                        .textFieldStyle(.roundedBorder)
                        .help("Free-form notes about corpus, intended effect, training decisions, or caveats to preserve with the adapter artifact.")
                    HStack {
                        Button("Save Adapter") { panel.saveSelectedAdapter() }
                            .disabled(panel.isTraining)
                            .help("Save the current adapter metadata, paths, hyperparameters, and refreshed dataset hashes.")
                        // Training routes to the owning engine: in-process MLX
                        // in Local, a durable /api/finetune/train job on a
                        // server workspace (adapter stays server-side).
                        if service.cluster.computeTarget == .server {
                            Button("Begin Training on Server") {
                                // Item 2: LoRA training is the archetypal
                                // model-running job — the shared gate warns
                                // when no GPU session is up.
                                let panel = panel
                                ModelJobGPUGate.submit(
                                    "server LoRA training", service: service,
                                    pending: $pendingModelJob
                                ) { await panel.beginServerTraining() }
                            }
                            .disabled(panel.isTraining)
                            .help(
                                "queue document-adaptation LoRA on "
                                    + "\(service.cluster.substrateLabel): the training "
                                    + "folder's text files (or the single training "
                                    + "file) are uploaded inline and the adapter "
                                    + "lands in the server's runs tree (progress "
                                    + "under Compute)")
                        } else {
                            Button("Begin Training") { panel.beginTraining() }
                                .disabled(panel.isTraining)
                                .help("Start LoRA/DoRA adapter training with the selected base model, data folders, and hyperparameters.")
                        }
                        if panel.isTraining {
                            ProgressView()
                                .controlSize(.small)
                                .help("Adapter training is running.")
                            Button("Cancel") { panel.cancelTraining() }
                                .help(
                                    panel.serverTrainingJobID == nil
                                        ? "Request cancellation. Local training "
                                            + "stops at the next reporting checkpoint."
                                        : "Send a cancel request for cluster job "
                                            + "\(panel.serverTrainingJobID ?? "") — the "
                                            + "cluster decides when it stops; logs keep "
                                            + "streaming until it does.")
                        }
                    }
                    if let progress = panel.trainingProgress {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(progress)
                                .font(.caption)
                                .foregroundStyle(panel.isTraining ? .secondary : .primary)
                                .textSelection(.enabled)
                            if !panel.trainingLog.isEmpty {
                                DisclosureGroup("Training log") {
                                    VStack(alignment: .leading, spacing: 3) {
                                        ForEach(Array(panel.trainingLog.enumerated()), id: \.offset) { _, line in
                                            Text(line)
                                                .font(.caption2.monospaced())
                                                .foregroundStyle(.secondary)
                                                .textSelection(.enabled)
                                        }
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .help("Recent training progress messages, including validation, loss reports, and checkpoints.")
                            }
                        }
                        .padding(8)
                        .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
                    }
                } else {
                    Text("Select an adapter in the library to configure training.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .help("Choose an adapter from the library above to show its trainer settings.")
                }
                Button("Refresh Library") { panel.refresh() }
                    .help("Rescan saved adapter and agent (variant) artifacts from disk.")
            }

            if let status = panel.status {
                Section("Status") {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .help("Latest Fine-Tuning action result or error message.")
                }
            }
        }
        .formStyle(.grouped)
        // Item 2 (cluster-testing): the shared no-GPU-session warning for
        // the server training submission above.
        .modelJobGPUWarning(pending: $pendingModelJob, service: service)
        // The explicit-split route parks the server's normalized training
        // plan here; nothing is scheduled until the researcher confirms it
        // (docs/CLUSTER-LORA-READINESS.md §3 — the plan confirmed and the
        // plan run are provably the same one via expectedPlanHash).
        .sheet(item: Binding(
            get: { panel.pendingServerTrainingPlan },
            set: { if $0 == nil { panel.cancelServerTrainingPlan() } }
        )) { pending in
            ServerTrainingPlanSheet(pending: pending, panel: panel)
        }
        .onAppear { panel.refresh() }
    }

    /// Installed models for the active workspace (through the shared
    /// catalog). A current selection that is not in the workspace's inventory
    /// (e.g. an adapter defined against a model the server has not installed)
    /// is still rendered so the picker never silently drops it, but it is
    /// labeled honestly and selection-disabled — strict availability, same
    /// rule as `WorkspaceModelPicker`.
    @ViewBuilder
    private func scopedModelRows(current: String) -> some View {
        let installed = service.workspaceModelOptions
        let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
        ForEach(installed, id: \.self) { model in
            Text(model).tag(model)
        }
        if !trimmed.isEmpty, !installed.contains(trimmed) {
            Text(
                service.cluster.computeTarget == .server
                    ? "\(trimmed) (not installed)" : trimmed
            )
            .tag(trimmed)
            .selectionDisabled()
        }
    }

    private func adapterRow(_ record: FineTuneArtifactRecord, selected: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(record.artifact.name)
                    .font(.headline)
                Spacer()
                if selected {
                    Text("selected")
                        .font(.caption)
                        .foregroundStyle(.blue)
                }
                Text(record.artifact.fineTuneType.uppercased())
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            Text(record.artifact.baseModelID)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text((FineTuneTrainingMode(rawValue: record.artifact.trainingMode ?? "") ?? .document).label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 2) {
                GridRow {
                    metric("rank", "\(record.artifact.rank)")
                    metric("scale", record.artifact.scale.formatted(.number.precision(.fractionLength(1))))
                    metric("layers", "\(record.artifact.adaptedLayers)")
                    metric("iters", "\(record.artifact.iterations)")
                }
                GridRow {
                    metric("adapter", shortHash(record.artifact.adapterHash))
                    metric("config", shortHash(record.artifact.configHash))
                    metric("train", shortHash(record.artifact.trainingDataHash))
                    metric("valid", shortHash(record.artifact.validationDataHash))
                }
            }
            Text(resolvedDisplayPath(record.artifact.adapterDirectory))
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        .padding(.vertical, 4)
        .help("Adapter \(record.artifact.name), trained for \(record.artifact.baseModelID). Click to load it into the trainer.")
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.monospaced())
        }
        .help("\(label): \(value)")
    }

    private func shortHash(_ value: String?) -> String {
        guard let value, !value.isEmpty else { return "-" }
        return String(value.prefix(10))
    }

    private func chooseDirectory(_ receive: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            receive(url.path)
        }
    }

    // Stored adapter paths are workspace-relative for anything inside the
    // workspace. Every display/open below resolves them through
    // `FineTuneStore.absoluteURL` — never `URL(filePath:)`, which would
    // resolve against the process CWD (the code checkout when launched from
    // Xcode) and show the researcher a path their data does not live at.

    /// The resolved absolute location of a stored path, for display.
    private func resolvedDisplayPath(_ path: String) -> String {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        return FineTuneStore.absoluteURL(trimmed).standardizedFileURL.path
    }

    private func openFolder(_ path: String) {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var url = FineTuneStore.absoluteURL(trimmed).standardizedFileURL
        var isDirectory = ObjCBool(false)
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) {
            if !isDirectory.boolValue {
                url = url.deletingLastPathComponent()
            }
        } else if !url.pathExtension.isEmpty {
            url = url.deletingLastPathComponent()
        }
        NSWorkspace.shared.open(url)
    }

    /// Folder-shaped display for the data rows: the resolved folder itself,
    /// or the containing folder for a legacy stored `<folder>/train.jsonl`.
    private func directoryDisplayPath(_ path: String) -> String {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let url = FineTuneStore.absoluteURL(trimmed).standardizedFileURL
        var isDirectory = ObjCBool(false)
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) {
            return isDirectory.boolValue ? url.path : url.deletingLastPathComponent().path
        }
        return url.pathExtension.isEmpty ? url.path : url.deletingLastPathComponent().path
    }
}

private struct DirectoryPathRow: View {
    let title: String
    let path: String
    let placeholder: String
    let chooseTitle: String?
    let help: String
    let open: () -> Void
    let choose: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Button(action: open) {
                Label(title, systemImage: "folder")
                    .labelStyle(.titleAndIcon)
                    .frame(width: 150, alignment: .leading)
            }
            .buttonStyle(.plain)
            .disabled(path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .help(path.isEmpty ? help : "\(help) Click to open this folder in Finder.")

            Text(path.isEmpty ? placeholder : path)
                .font(.caption.monospaced())
                .foregroundStyle(path.isEmpty ? .secondary : .primary)
                .lineLimit(1)
                .truncationMode(.middle)
                .help(path.isEmpty ? help : path)

            Spacer(minLength: 8)

            if let chooseTitle {
                Button(chooseTitle, action: choose)
                    .help("Choose a different folder for \(title.lowercased()).")
            }
        }
        .help(help)
    }
}

private struct TrainingPlanView: View {
    let plan: FineTuningPanel.TrainingPlan

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Recommended plan")
                    .font(.headline)
                Spacer()
                Text(plan.dataScale)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 4) {
                GridRow {
                    planMetric("train", "\(plan.train.exampleCount) ex / ~\(plan.train.estimatedTokens) tok")
                    planMetric("valid", "\(plan.validation.exampleCount) ex / ~\(plan.validation.estimatedTokens) tok")
                }
                GridRow {
                    planMetric("kind", plan.train.kind)
                    planMetric("model", modelLabel)
                }
                GridRow {
                    planMetric("rank", "\(plan.recommendedRank)")
                    planMetric("layers", "\(plan.recommendedLayers)")
                }
                GridRow {
                    planMetric("batch", "\(plan.recommendedBatchSize)")
                    planMetric("iters", "\(plan.recommendedIterations)")
                }
                GridRow {
                    planMetric("scale", plan.recommendedScale.formatted(.number.precision(.fractionLength(1))))
                    planMetric("lr", plan.recommendedLearningRate.formatted(.number.precision(.fractionLength(7))))
                }
            }
            Text(plan.rationale)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            ForEach(plan.warnings, id: \.self) { warning in
                Text(warning)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
        .help("Training plan generated from model size and estimated train/validation data size. Apply it to update the editable hyperparameters.")
    }

    private var modelLabel: String {
        plan.modelSizeBillions.map {
            "\($0.formatted(.number.precision(.fractionLength(0 ... 1))))B"
        } ?? "unknown"
    }

    private func planMetric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.monospaced())
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .help("\(label): \(value)")
    }
}

private struct IntSliderField: View {
    let title: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let step: Int
    let help: String

    init(
        _ title: String,
        value: Binding<Int>,
        range: ClosedRange<Int>,
        step: Int = 1,
        help: String
    ) {
        self.title = title
        self._value = value
        self.range = range
        self.step = step
        self.help = help
    }

    var body: some View {
        HStack(spacing: 10) {
            Text(title)
                .frame(width: 95, alignment: .leading)
                .help(help)
            Slider(
                value: Binding(
                    get: { Double(value) },
                    set: { value = clamped(Int($0.rounded())) }),
                in: Double(range.lowerBound) ... Double(range.upperBound),
                step: Double(step))
                .help("\(help) Valid range: \(range.lowerBound)-\(range.upperBound).")
            TextField(title, value: $value, format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(width: 86)
                .monospacedDigit()
                .help("Type an exact \(title.lowercased()) value. Values outside \(range.lowerBound)-\(range.upperBound) are clamped.")
        }
        .help(help)
        .onChange(of: value) { _, newValue in
            let next = clamped(newValue)
            if next != newValue { value = next }
        }
    }

    private func clamped(_ candidate: Int) -> Int {
        min(max(candidate, range.lowerBound), range.upperBound)
    }
}

private struct DoubleSliderField: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let fractionDigits: Int
    let help: String

    init(
        _ title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        fractionDigits: Int,
        help: String
    ) {
        self.title = title
        self._value = value
        self.range = range
        self.step = step
        self.fractionDigits = fractionDigits
        self.help = help
    }

    var body: some View {
        HStack(spacing: 10) {
            Text(title)
                .frame(width: 95, alignment: .leading)
                .help(help)
            Slider(
                value: Binding(
                    get: { value },
                    set: { value = stepped($0) }),
                in: range,
                step: step)
                .help("\(help) Valid range: \(range.lowerBound)-\(range.upperBound).")
            TextField(
                title,
                value: Binding(
                    get: { value },
                    set: { value = clamped($0) }),
                format: .number.precision(.fractionLength(fractionDigits)))
                .textFieldStyle(.roundedBorder)
                .frame(width: 86)
                .monospacedDigit()
                .help("Type an exact \(title.lowercased()) value. Values outside the valid range are clamped.")
        }
        .help(help)
        .onChange(of: value) { _, newValue in
            let next = clamped(newValue)
            if next != newValue { value = next }
        }
    }

    private func stepped(_ candidate: Double) -> Double {
        let rounded = (candidate / step).rounded() * step
        return clamped(rounded)
    }

    private func clamped(_ candidate: Double) -> Double {
        min(max(candidate, range.lowerBound), range.upperBound)
    }
}

private struct LogDoubleSliderField: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let fractionDigits: Int
    let help: String

    init(
        _ title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        fractionDigits: Int,
        help: String
    ) {
        self.title = title
        self._value = value
        self.range = range
        self.fractionDigits = fractionDigits
        self.help = help
    }

    var body: some View {
        HStack(spacing: 10) {
            Text(title)
                .frame(width: 95, alignment: .leading)
                .help(help)
            Slider(
                value: Binding(
                    get: { log10(clamped(value)) },
                    set: { value = clamped(pow(10.0, $0)) }),
                in: log10(range.lowerBound) ... log10(range.upperBound))
                .help("\(help) Slider is logarithmic over \(range.lowerBound)-\(range.upperBound).")
            TextField(
                title,
                value: Binding(
                    get: { value },
                    set: { value = clamped($0) }),
                format: .number.precision(.fractionLength(fractionDigits)))
                .textFieldStyle(.roundedBorder)
                .frame(width: 104)
                .monospacedDigit()
                .help("Type an exact learning rate. Values outside \(range.lowerBound)-\(range.upperBound) are clamped.")
        }
        .help(help)
        .onChange(of: value) { _, newValue in
            let next = clamped(newValue)
            if next != newValue { value = next }
        }
    }

    private func clamped(_ candidate: Double) -> Double {
        min(max(candidate, range.lowerBound), range.upperBound)
    }
}

/// The researcher's yes/no on the server's normalized training plan. The
/// summary lines are exactly what `FineTuningPanel.serverTrainingPlanSummary`
/// produced from the plan the server returned — resolved revision, split
/// sizes, schedule, dtype, and the plan hash that rides back as
/// `expectedPlanHash` on confirmation.
private struct ServerTrainingPlanSheet: View {
    let pending: FineTuningPanel.PendingServerTrainingPlan
    let panel: FineTuningPanel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Confirm Server Training Plan")
                .font(.headline)
            Text(
                "The server resolved this plan from the uploaded splits. "
                    + "Nothing is scheduled until you confirm it.")
                .font(.caption)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(pending.summary.enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
            HStack {
                Spacer()
                Button("Cancel") {
                    panel.cancelServerTrainingPlan()
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                Button("Confirm & Train") {
                    let panel = panel
                    Task { await panel.confirmServerTrainingPlan() }
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 460)
    }
}

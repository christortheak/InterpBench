import AppKit
import ExperimentKit
import SwiftUI

struct FineTuningPanelView: View {
    @Bindable var service: ChatService
    /// Item 2 (cluster-testing): a server LoRA training submission parked
    /// while the shared no-GPU-session dialog asks.
    @State private var pendingModelJob: PendingModelJob?
    /// The cluster job a "Cancel Training" click is asking about — cancelling
    /// a server job is irreversible and loses the queue slot, so it confirms.
    @State private var confirmingCancelJobID: String?

    private var panel: FineTuningPanel { service.fineTuning }

    /// The panel's own Create Adapter gates, so the refusal can be shown
    /// BESIDE the button instead of as a note at the bottom of the form.
    /// Mirrors `FineTuningPanel.createAdapterProject`, including its fallback
    /// to the host's loaded/selected model when the picker is empty.
    private var createAdapterDisabledReason: String? {
        if panel.newAdapterName.trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        {
            return "name the adapter first — the name becomes adapters/<name>/"
        }
        let picked = panel.newAdapterBaseModelID
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let effective =
            picked.isEmpty
            ? (service.loadedModelID ?? service.selectedModelID) : picked
        if effective.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "choose a base model first — an adapter is trained for one model"
        }
        return nil
    }

    /// Instruction/chat tuning needs the structured upload route; on a server
    /// whose capabilities say it has none, the click was refused into the
    /// Status section at the bottom of the form. Only asserted when the
    /// capabilities are actually known.
    private var legacyServerModeRefusal: String? {
        guard service.cluster.computeTarget == .server,
            panel.trainingMode == .instructionChat,
            let capabilities = service.cluster.capabilities,
            !capabilities.supportsStructuredFineTuneUpload
        else { return nil }
        return
            "\(service.cluster.substrateLabel) accepts document adaptation "
            + "only — instruction/chat tuning trains in the Local workspace"
    }

    /// The wire key and the server's derivation, on screen rather than only
    /// in the Scale tooltip.
    private var scaleConventionNote: String {
        let alpha = Double(panel.scale) * Double(panel.rank)
        return
            "Scale is sent as adapterScale; the server resolves lora_alpha = "
            + "scale × rank = "
            + "\(panel.scale.formatted(.number.precision(.fractionLength(0 ... 1)))) × "
            + "\(panel.rank) = "
            + "\(alpha.formatted(.number.precision(.fractionLength(0 ... 1))))."
    }

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
                TextField(
                    "Name", text: $panel.newAdapterName,
                    prompt: Text("my-concept-lora")
                )
                .textFieldStyle(.roundedBorder)
                .help("Name for the adapter project and artifact. Use a stable research label, such as the corpus, method, or intended intervention.")
                DisclosureGroup("Storage details") {
                    DirectoryPathRow(
                        title: "Project directory",
                        path: resolvedDisplayPath(panel.newAdapterProjectDirectory),
                        placeholder: "Default: adapters/ in the workspace",
                        chooseTitle: "Choose…",
                        help: "Parent folder where SteerLab will create the adapter's home. Leave blank to store it under the workspace's adapters/ folder."
                    ) {
                        openFolder(panel.newAdapterProjectDirectory)
                    } choose: {
                        chooseDirectory { panel.newAdapterProjectDirectory = $0 }
                    }
                }
                Button("Create Adapter") { panel.createAdapterProject() }
                    .disabled(createAdapterDisabledReason != nil)
                    .help("create adapters/<name>/ in the workspace with training/ and validation/ data folders, and add the adapter to the library")
                // The panel's own gates, said beside the button instead of as
                // a note at the bottom of the form (audit 2026-09-06).
                if let reason = createAdapterDisabledReason {
                    Text(reason)
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
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
                    DisclosureGroup("Storage details") {
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
                            title: "Saved adapter files",
                            path: resolvedDisplayPath(panel.adapterDirectory),
                            placeholder: "Choose adapter output folder",
                            chooseTitle: "Choose…",
                            help: "Folder where the trained adapter files live. A completed MLX adapter should contain adapter_config.json and adapters.safetensors."
                        ) {
                            openFolder(panel.adapterDirectory)
                        } choose: {
                            chooseDirectory { panel.adapterDirectory = $0 }
                        }
                        DirectoryPathRow(
                            title: "Source materials (optional)",
                            path: resolvedDisplayPath(panel.trainingWorkspacePath),
                            placeholder: "Not set — optional",
                            chooseTitle: "Choose…",
                            help: "Optional reference to source material. This folder is not read by training; only Training data and Validation data supply examples."
                        ) {
                            openFolder(panel.trainingWorkspacePath)
                        } choose: {
                            chooseDirectory { panel.trainingWorkspacePath = $0 }
                        }
                    }
                    DirectoryPathRow(
                        title: "Training data",
                        path: directoryDisplayPath(panel.trainingDataPath),
                        placeholder: "Choose training data folder",
                        chooseTitle: "Choose…",
                        help: "Examples used to update the adapter. Document adaptation learns from documents; instruction/chat tuning needs structured JSONL conversations with desired assistant responses. Drop files here to copy them into the training folder.",
                        acceptsDrop: true,
                        previewPath: resolvedDisplayPath(panel.trainingDataPath)
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
                        chooseTitle: "Choose…",
                        help: "Separate representative examples used to monitor loss without training on them. Use the same format as training and avoid duplicate source material. If you use validation to choose settings, reserve separate test examples for the final study.",
                        acceptsDrop: true,
                        previewPath: resolvedDisplayPath(panel.validationDataPath)
                    ) {
                        openFolder(panel.validationDataPath)
                    } choose: {
                        chooseDirectory { panel.validationDataPath = $0 }
                    }
                    .dropDestination(for: URL.self) { urls, _ in
                        panel.importDroppedFiles(urls, to: .validation)
                        return true
                    }

                    Text("Training examples teach the adapter; separate validation examples check how it performs on material it has not learned from.")
                        .font(.caption).foregroundStyle(.secondary)
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
                            help: "LoRA scale is the direct multiplier on the learned low-rank update, during training and inference (this engine's MLX convention). A server run receives it as adapterScale and resolves PEFT's lora_alpha = scale × rank itself, so the number means the same strength on both engines.")
                        // The convention is visible, not tooltip-only: with
                        // the defaults (rank 8 × scale 10) a PEFT reader would
                        // otherwise assume alpha = 10 where the server gets 80.
                        Text(scaleConventionNote)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
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
                            .disabled(
                                panel.isTraining
                                    || legacyServerModeRefusal != nil)
                            .help(
                                "queue \(panel.trainingMode.label.lowercased()) LoRA on "
                                    + "\(service.cluster.substrateLabel): the training "
                                    + "folder's text files (or the single training "
                                    + "file) are uploaded and the adapter "
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
                            // A server cancel is an irreversible cluster job
                            // cancel, so it asks and names the job; the local
                            // stop is cooperative and keeps its checkpoints,
                            // so it stays unconfirmed (audit 2026-09-06).
                            if let jobID = panel.serverTrainingJobID {
                                Button("Cancel Training", role: .destructive) {
                                    confirmingCancelJobID = jobID
                                }
                                .help(
                                    "send a cancel request for cluster job \(jobID) — the "
                                        + "cluster decides when it stops; logs keep "
                                        + "streaming until it does")
                            } else {
                                Button("Cancel Training") { panel.cancelTraining() }
                                    .help(
                                        "request cancellation — local training stops at "
                                            + "the next reporting checkpoint and the "
                                            + "adapter written so far stays on disk")
                            }
                        }
                    }
                    if let refusal = legacyServerModeRefusal {
                        Text(refusal)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
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
                    .help("Rescan saved adapter and agent artifacts from disk.")
            }

            // A CONSTANT slot: the section used to appear and disappear with
            // the status, shifting the whole form (audit 2026-09-06).
            Section("Status") {
                Text(panel.status ?? "no Adapter Training action yet this session")
                    .font(.caption)
                    .foregroundStyle(panel.status == nil ? .tertiary : .secondary)
                    .textSelection(.enabled)
                    .help("Latest Adapter Training action result or error message.")
            }
        }
        .formStyle(.grouped)
        // Item 2 (cluster-testing): the shared no-GPU-session warning for
        // the server training submission above.
        .modelJobGPUWarning(pending: $pendingModelJob, service: service)
        .confirmationDialog(
            "Cancel training job \(confirmingCancelJobID ?? "") on "
                + "\(service.cluster.substrateLabel)?",
            isPresented: Binding(
                get: { confirmingCancelJobID != nil },
                set: { if !$0 { confirmingCancelJobID = nil } }),
            presenting: confirmingCancelJobID
        ) { jobID in
            Button("Cancel Job \(jobID)", role: .destructive) {
                confirmingCancelJobID = nil
                panel.cancelTraining()
            }
            Button("Keep Training", role: .cancel) { confirmingCancelJobID = nil }
        } message: { jobID in
            Text(
                "The cluster decides when job \(jobID) stops, and the queue "
                    + "slot is lost. Checkpoints already written to the "
                    + "server's runs tree stay there; training does not resume "
                    + "from here.")
        }
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
                        .foregroundStyle(Color.accentColor)
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
        .help(Self.metricHelp[label] ?? label)
    }

    /// What each abbreviated cell in an adapter row MEANS — the tooltip used
    /// to repeat the two words already on screen (audit 2026-09-06).
    private static let metricHelp: [String: String] = [
        "rank": "LoRA rank the adapter was trained at",
        "scale": "LoRA scale (adapterScale; the server resolves lora_alpha = scale × rank)",
        "layers": "how many transformer layers the adapter touches",
        "iters": "optimizer steps the training ran for",
        "adapter": "first 10 characters of the adapter weights' sha256",
        "config": "first 10 characters of adapter_config.json's sha256",
        "train": "first 10 characters of the training data's sha256",
        "valid": "first 10 characters of the validation data's sha256",
    ]

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
    /// Rows that accept a file drop say so on screen: the drop used to be
    /// discoverable only through the tooltip (audit 2026-09-06).
    var acceptsDrop: Bool = false
    var previewPath: String? = nil
    let open: () -> Void
    let choose: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            // Link style, not plain: this opens Finder, and as a plain label
            // it read as static text.
            Button(action: open) {
                Label(title, systemImage: "folder")
                    .labelStyle(.titleAndIcon)
                    .frame(width: 150, alignment: .leading)
            }
            .buttonStyle(.link)
            .disabled(path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .help(path.isEmpty ? help : "\(help) Click to reveal this folder in Finder.")
            .accessibilityLabel("Reveal \(title.lowercased()) in Finder")

            Text(path.isEmpty ? placeholder : path)
                .font(.caption.monospaced())
                .foregroundStyle(path.isEmpty ? .secondary : .primary)
                .lineLimit(1)
                .truncationMode(.middle)
                .help(path.isEmpty ? help : path)

            Spacer(minLength: 8)

            if acceptsDrop {
                Label("drop files to copy", systemImage: "arrow.down.doc")
                    .labelStyle(.titleAndIcon)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .help(
                        "drag files onto this row and they are COPIED into "
                            + "\(title.lowercased()) — the originals are left alone")
            }

            if let previewPath { TrainingDataBrowseButton(path: previewPath) }
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
        // The value truncates, so the tooltip carries it in full alongside
        // what the abbreviation means.
        .help("\(Self.planMetricHelp[label] ?? label) — \(value)")
    }

    private static let planMetricHelp: [String: String] = [
        "train": "training examples found, and their estimated token count",
        "valid": "held-out examples found, and their estimated token count",
        "kind": "row shape the training files parsed as",
        "model": "parameter count read from the base model's name",
        "rank": "recommended LoRA rank",
        "layers": "recommended number of adapted layers",
        "batch": "recommended batch size",
        "iters": "recommended optimizer steps",
        "scale": "recommended LoRA scale (adapterScale)",
        "lr": "recommended learning rate",
    ]
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
                .help("\(help) Valid range: \(range.lowerBound) to \(range.upperBound).")
            TextField(title, value: $value, format: .number)
                .textFieldStyle(.roundedBorder)
                .frame(width: 86)
                .monospacedDigit()
                .help("Type an exact \(title.lowercased()) value. Values outside \(range.lowerBound) to \(range.upperBound) are clamped.")
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
                .help("\(help) Valid range: \(SliderRangeText.of(range)).")
            TextField(
                title,
                value: Binding(
                    get: { value },
                    set: { value = displayRounded(clamped($0)) }),
                format: .number.precision(.fractionLength(fractionDigits)))
                .textFieldStyle(.roundedBorder)
                .frame(width: 86)
                .monospacedDigit()
                .help("Type an exact \(title.lowercased()) value. Values outside \(SliderRangeText.of(range)) are clamped, and the value is kept at the precision shown.")
        }
        .help(help)
        .onChange(of: value) { _, newValue in
            let next = clamped(newValue)
            if next != newValue { value = next }
        }
    }

    /// The slider's own notches are `lowerBound + k·step`; rounding to
    /// multiples of `step` instead put the stored value off the thumb by
    /// `lowerBound % step` at every notch (audit 2026-09-06).
    private func stepped(_ candidate: Double) -> Double {
        let offset = candidate - range.lowerBound
        let rounded = range.lowerBound + (offset / step).rounded() * step
        return clamped(rounded)
    }

    /// A typed value is stored at the precision the field displays, so the
    /// number on screen is the number that trains.
    private func displayRounded(_ candidate: Double) -> Double {
        let scale = pow(10.0, Double(fractionDigits))
        return (candidate * scale).rounded() / scale
    }

    private func clamped(_ candidate: Double) -> Double {
        min(max(candidate, range.lowerBound), range.upperBound)
    }
}

/// Range bounds as readable text: `"\(0.000001)-\(0.001)"` printed
/// "1e-06-0.001", where the range hyphen collides with the exponent's sign
/// (audit 2026-09-06).
private enum SliderRangeText {
    static func of(_ range: ClosedRange<Double>) -> String {
        "\(number(range.lowerBound)) to \(number(range.upperBound))"
    }

    static func number(_ value: Double) -> String {
        value.formatted(.number.precision(.significantDigits(1 ... 4)))
    }
}

private struct LogDoubleSliderField: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let help: String

    init(
        _ title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        help: String
    ) {
        self.title = title
        self._value = value
        self.range = range
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
                .help("\(help) Slider is logarithmic over \(SliderRangeText.of(range)).")
            TextField(
                title,
                value: Binding(
                    get: { value },
                    set: { value = clamped($0) }),
                // Significant digits rather than 7 fraction digits: 1e-5 used
                // to render "0.0000100".
                format: .number.precision(.significantDigits(1 ... 3)))
                .textFieldStyle(.roundedBorder)
                .frame(width: 104)
                .monospacedDigit()
                .help("Type an exact learning rate. Values outside \(SliderRangeText.of(range)) are clamped.")
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
                Button("Cancel", role: .cancel) {
                    panel.cancelServerTrainingPlan()
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                .help("discard this plan — nothing is scheduled and the uploaded splits are dropped")
                Button("Confirm & Train") {
                    let panel = panel
                    Task { await panel.confirmServerTrainingPlan() }
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .help("schedule exactly this plan — the confirmation carries its hash, so the plan run is provably the plan shown")
            }
        }
        .padding(20)
        .frame(minWidth: 460)
    }
}

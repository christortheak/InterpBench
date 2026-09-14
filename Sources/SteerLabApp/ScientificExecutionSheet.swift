import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// A captured client keeps planning, submitting and observation on one endpoint.
///
/// The form offers what the workspace and the server already know instead of
/// blank fields (researcher request 2026-09-13): the workspace's batteries by
/// name with a Finder chooser behind them, the server's installed models, and
/// the pinned revision read from the server's own model cache. The rules the
/// captions state live in `ScientificDiagnosticInputs` (ExperimentKit, tested).
struct ScientificExecutionSheet: View {
    let client: ClusterClient
    let endpoint: String
    /// The local workspace whose `prompts/batteries/` is listed. The server
    /// resolves the SAME relative path inside its workspace, so a seed
    /// battery exists on both sides and a Mac-authored one must be staged.
    let root: URL
    @Environment(\.dismiss) private var dismiss
    @State private var operation = "battery"
    @State private var batteryFile = ""
    @State private var batteryOptions: [String] = []
    @State private var batteryNote: String?
    @State private var showingBatteryImporter = false
    @State private var agents = "baseline"
    @State private var model = ""
    @State private var modelOptions: [String] = []
    @State private var modelNotice: String?
    @State private var revision = ""
    @State private var revisionNote: String?
    @State private var resolvingRevision = false
    @State private var study = ""
    @State private var concept = ""
    @State private var resamples = 32
    @State private var fraction = 0.5
    @State private var seed = "0"
    @State private var orderShuffles = 8
    @State private var alphaUnits = "norm"
    @State private var dtype = "auto"
    @State private var device = ""
    @State private var gpuOptions: ScientificGPUPlacement?
    @State private var gpuType = ""
    @State private var plannedGPUType = ""
    @State private var placementMessage = "Loading the controller’s GPU choices…"
    @State private var gpuReview: [String] = []
    @State private var plan: JSONValue?
    @State private var plannedRequest: JSONValue?
    @State private var jobID: String?
    @State private var output = ""
    @State private var busy = false
    /// The last refusal, in its own line: it used to be pasted into the
    /// output box on top of the plan the researcher was reading.
    @State private var failure: String?
    @State private var confirmingCancel = false

    private var agentLines: [String] { ScientificDiagnosticInputs.agentLines(agents) }

    private var request: JSONValue {
        var parameters: [String: JSONValue] = operation == "battery"
            ? ["batteryFile": .string(batteryFile), "agents": .array(agentLines.map { .string($0) }),
               "alphaUnits": .string(alphaUnits)]
            : ["experiment": .string(study), "concept": .string(concept),
               "resamples": .number(Double(resamples)), "fraction": .number(fraction),
               "seed": .string(seed), "orderShuffles": .number(Double(orderShuffles))]
        if operation == "battery", !model.isEmpty { parameters["modelID"] = .string(model) }
        if operation == "battery", !revision.isEmpty { parameters["revision"] = .string(revision) }
        if !dtype.isEmpty { parameters["dtype"] = .string(dtype) }
        if !device.isEmpty { parameters["device"] = .string(device) }
        return .object(["operation": .string(operation), "parameters": .object(parameters)])
    }
    private var planHash: String? {
        guard case .object(let object) = plan, case .string(let hash) = object["planSHA256"] else { return nil }
        return hash
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            form
            ForEach(gpuReview, id: \.self) { Text($0).font(.caption).textSelection(.enabled) }
            controls
            failureLine
            jobLine
            ScrollView {
                Text(output)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
        .frame(minWidth: 820, minHeight: 640)
        .task {
            batteryOptions = ScientificDiagnosticInputs.batteryFiles(root: root)
            do {
                gpuOptions = try await client.scientificGPUPlacement()
                placementMessage = gpuOptions == nil ? "This server does not advertise GPU choices. The server default will be used." : ""
            } catch { placementMessage = "GPU choices could not be read. The server default remains available; reopen this sheet to retry." }
            do {
                modelOptions = try await client.state().models
                if modelOptions.isEmpty {
                    modelNotice = "No models are installed on this server yet — use Install model… in the Compute header, or enter a model identifier under \"Use another model\"."
                }
            } catch {
                modelNotice = "Could not list this server's models (" + Self.describe(error)
                    + "). You can still enter a verified model identifier under \"Use another model\"."
            }
        }
        .onChange(of: gpuType) { _, _ in plan = nil; gpuReview = [] }
        .onChange(of: model) { _, _ in
            revision = ""
            revisionNote = nil
            // A pick from the server's inventory resolves at once; a typed
            // identifier waits for the button, so keystrokes are not requests.
            if modelOptions.contains(model) { resolveRevision() }
        }
        .fileImporter(isPresented: $showingBatteryImporter, allowedContentTypes: [.data]) { result in
            do {
                let url = try result.get()
                guard let relative = ScientificDiagnosticInputs.workspaceRelativePath(url, root: root) else {
                    batteryNote = "Choose a battery inside this workspace (\(root.path)) — a path outside it means nothing to the server."
                    return
                }
                batteryFile = relative
                batteryNote = relative.hasPrefix(VectorCatalog.batteriesRelativeDirectory + "/")
                    ? nil
                    : "Outside the conventional prompts/batteries/ directory — that is allowed, as long as the same relative path exists in the server workspace."
            } catch { batteryNote = error.localizedDescription }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Run a scientific diagnostic").font(.title2)
            Text(endpoint).font(.caption).textSelection(.enabled)
            Text("Plan first, then submit exactly the reviewed plan. The server "
                + "reads inputs from ITS workspace by the same relative paths this "
                + "workspace uses; a battery authored only on this Mac reaches it "
                + "through Stage inputs, collect evidence, clean up… in the jobs "
                + "panel's Actions menu. Diagnostics keep their own output type "
                + "and do not resume from checkpoints.")
                .font(.caption)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // Split out of `body`: the form's conditionals plus a `.help` on every
    // control is more than one expression should ask of the type checker.
    private var form: some View {
        Form {
            Picker("Diagnostic", selection: $operation) {
                Text("Capability battery").tag("battery")
                Text("Extraction stability").tag("stability")
            }
            .help("which diagnostic to plan — a capability battery scores "
                + "agents against a battery file; an extraction-stability "
                + "check resamples one concept's extraction")
            if operation == "battery" {
                batteryRows
                agentRows
                modelRows
            } else {
                TextField("Study name on server", text: $study)
                    .help("the study whose extraction is being resampled, by "
                        + "the name it has on this server")
                TextField("Concept", text: $concept)
                    .help("which of that study's concepts to resample")
            }
            if let gpuOptions, gpuOptions.available {
                ScientificGPUSelection(options: gpuOptions, selection: $gpuType)
            } else if !placementMessage.isEmpty { Text(placementMessage).font(.caption) }
            DisclosureGroup("Execution parameters") { parameters }
        }
        .disabled(busy || jobID != nil)
    }

    /// The battery by name, from this workspace's `prompts/batteries/`, with
    /// a Finder chooser for one filed elsewhere in the workspace and a typed
    /// path only for a file that exists on the server but not here.
    @ViewBuilder
    private var batteryRows: some View {
        Picker("Capability battery", selection: $batteryFile) {
            Text("Choose a battery…").tag("")
            ForEach(batteryOptions, id: \.self) { path in
                Text(path.replacingOccurrences(
                    of: VectorCatalog.batteriesRelativeDirectory + "/", with: "")).tag(path)
            }
            if !batteryFile.isEmpty, !batteryOptions.contains(batteryFile) {
                Text(batteryFile).tag(batteryFile)
            }
        }
        .help("the battery file to score against — listed from this workspace's "
            + "prompts/batteries/; the server reads the same relative path in "
            + "its own workspace, and Review plan refuses one it cannot find")
        HStack(spacing: 8) {
            Button("Choose file…") { showingBatteryImporter = true }
                .help("pick a .jsonl battery anywhere inside this workspace — "
                    + "it is recorded by its workspace-relative path")
            Text(batteryNote ?? (batteryFile.isEmpty ? " " : batteryFile))
                .font(.caption)
                .foregroundStyle(batteryNote == nil ? Color.secondary : Color.orange)
                .lineLimit(2)
                .truncationMode(.middle)
                .textSelection(.enabled)
                .help(batteryNote ?? batteryFile)
        }
        DisclosureGroup("Use a path that exists only on the server") {
            TextField("Path relative to the server workspace", text: $batteryFile)
                .help("for a battery staged on the server that this workspace "
                    + "does not hold — the path is relative to the server's "
                    + "workspace root, never absolute")
        }
    }

    @ViewBuilder
    private var agentRows: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Agents, one per line")
            Text("baseline · <concept>:<layer>:<alpha> (for example kindness:17:0.28) "
                + "· runs/model-variants/<name>.json · optional <name>=<reference>")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        TextEditor(text: $agents)
            .frame(height: 60)
            .accessibilityLabel("Agents, one per line")
            .help("one agent per line — the pinned model itself (baseline), a "
                + "steering condition on it (concept:layer:alpha), or a variant "
                + "artifact this server holds, which brings its own base model")
    }

    /// The model from the server's installed inventory (`/api/state`), with
    /// a typed identifier behind a disclosure, and the revision auto-filled
    /// from the server's model cache — never guessed.
    @ViewBuilder
    private var modelRows: some View {
        Picker("Model", selection: $model) {
            Text("Choose a model…").tag("")
            ForEach(modelOptions, id: \.self) { Text($0).tag($0) }
            if !model.isEmpty, !modelOptions.contains(model) {
                Text(model + " (not installed on this server)").tag(model)
            }
        }
        .help("the model a baseline or condition agent runs on, from the models "
            + "installed on this server — an artifact agent brings its own and "
            + "needs no choice here")
        Text(ScientificDiagnosticInputs.modelCaption(agents: agentLines))
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        if let modelNotice {
            Text(modelNotice).font(.caption).foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
        DisclosureGroup("Use another model") {
            TextField("Model identifier, such as owner/model-name", text: $model)
                .help("a Hugging Face model identifier the server can resolve — "
                    + "it must be installed there before the job can run")
        }
        HStack(spacing: 8) {
            TextField("Pinned model revision", text: $revision)
                .font(.system(.body, design: .monospaced))
                .help("the exact 40-character commit the reading is pinned to — "
                    + "filled from the server's model cache when a model is "
                    + "chosen; paste one only when the server cannot supply it")
            if resolvingRevision {
                ProgressView().controlSize(.small)
            }
            Button("Resolve from server") { resolveRevision() }
                .disabled(model.isEmpty || resolvingRevision || busy)
                .help("ask the server which commit of this model it holds — "
                    + "reads its cache only, downloads nothing")
        }
        Text(revisionNote ?? "The server refuses an agent without a pinned revision.")
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// The commit from the SERVER's cache for the chosen model — the same
    /// lookup the study judge picker uses — reported honestly when the server
    /// does not hold the model, never substituted from elsewhere.
    private func resolveRevision() {
        let wanted = model
        resolvingRevision = true
        Task {
            defer { resolvingRevision = false }
            do {
                let result = try await client.modelLoadPreflight(wanted)
                guard model == wanted else { return }
                if let commit = result.revision, commit.count == 40 {
                    revision = commit
                    revisionNote = result.cached
                        ? "From this server's model cache; no weights downloaded."
                        : "Resolved by the server, but the weights are not cached there yet — prepare the model in Compute before the job can run."
                } else {
                    revisionNote = "The server returned no exact commit for \(wanted) — install it there, or paste a verified commit."
                }
            } catch {
                guard model == wanted else { return }
                revisionNote = "Could not ask the server for this model's commit: " + Self.describe(error)
            }
        }
    }

    @ViewBuilder
    private var parameters: some View {
        if operation == "battery" {
            Picker("Alpha units", selection: $alphaUnits) {
                Text("Residual norm").tag("norm")
                Text("Raw").tag("raw")
            }
            .help("how each condition's alpha is read — as a fraction of the "
                + "residual norm, or as a raw multiplier")
        } else {
            Stepper("Resamples: \(resamples)", value: $resamples, in: 2...1_000_000)
                .help("how many resampled extractions to run — more costs "
                    + "proportionally more compute and narrows the interval")
            TextField("Fraction", value: $fraction, format: .number)
                .help("the share of the concept's pairs each resample draws")
            TextField("Seed (UInt64 decimal)", text: $seed)
                .help("the seed the resampling starts from — the same seed "
                    + "reproduces the same draw")
            Stepper("Order shuffles: \(orderShuffles)", value: $orderShuffles, in: 0...1_000_000)
                .help("how many times to re-order the pairs, to separate "
                    + "order effects from the extraction itself")
        }
        Picker("Data type", selection: $dtype) {
            Text("Automatic").tag("auto")
            Text("Float32").tag("float32")
            Text("BFloat16").tag("bfloat16")
            Text("Float16").tag("float16")
        }
        .help("the numeric precision the engine loads weights at — Automatic "
            + "leaves the engine's own choice in place")
        TextField("Device (blank for engine default)", text: $device)
            .help("which device to place the model on; blank leaves the "
                + "engine's default placement")
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Why Submit is dim, in the sheet rather than only on hover.
            Text(submitBlockedReason ?? " ")
                .font(.caption)
                .foregroundStyle(.orange)
                .lineLimit(1)
            HStack(spacing: 8) {
                Button(busy ? "Working…" : "Review plan") {
                    let captured = request
                    perform {
                        plan = try await client.scientificPlan(captured, gpuType: gpuType.isEmpty ? nil : gpuType)
                        plannedGPUType = gpuType
                        gpuReview = ScientificGPUPlacement.reviewLines(plan!)
                        plannedRequest = captured
                        output = describe(plan!)
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(busy || jobID != nil)
                .help("ask the server what this diagnostic would do and what "
                    + "it would cost — it plans only; nothing is submitted "
                    + "and nothing runs")
                if busy {
                    ProgressView()
                        .controlSize(.small)
                }
                Button("Submit reviewed diagnostic") {
                    guard let plannedRequest, let planHash, plannedRequest == request, plannedGPUType == gpuType else { return }
                    perform {
                        let result = try await client.scientificSubmit(plannedRequest, planSHA256: planHash, gpuType: plannedGPUType.isEmpty ? nil : plannedGPUType)
                        if case .object(let object) = result, case .string(let id) = object["jobId"] { jobID = id }
                        // Never retry an ambiguous submission automatically.
                        plan = nil
                        output = describe(result)
                    }
                }
                .disabled(submitBlockedReason != nil)
                .help("submit exactly the plan above, pinned by its hash — "
                    + "the job queues on this server and appears in the jobs "
                    + "list; editing any field first invalidates the plan")
                if let jobID {
                    Button("Refresh job") {
                        perform { output = describe(try await client.job(jobID)) }
                    }
                    .disabled(busy)
                    .help("re-read this job's record from the server")
                    Button("Cancel job", role: .destructive) { confirmingCancel = true }
                        .disabled(busy)
                        .help("cancel this diagnostic on the server — asks "
                            + "first; the queue slot is lost")
                        .confirmationDialog(
                            "Cancel diagnostic job \(jobID)?",
                            isPresented: $confirmingCancel,
                            titleVisibility: .visible
                        ) {
                            Button("Cancel job", role: .destructive) {
                                perform {
                                    try await client.cancelJob(jobID)
                                    output = describe(try await client.job(jobID))
                                }
                            }
                            Button("Keep running", role: .cancel) {}
                        } message: {
                            Text("The allocation is cancelled and its queue "
                                + "slot is lost. A diagnostic does not resume "
                                + "from a checkpoint, so it would have to be "
                                + "planned and submitted again from the top.")
                        }
                }
                Spacer()
                Button("Close", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close this sheet — a submitted job keeps running "
                        + "and stays in the jobs list")
            }
        }
    }

    /// One reason at a time, in the order the researcher meets them.
    private var submitBlockedReason: String? {
        if jobID != nil { return "This diagnostic has been submitted." }
        if busy { return "Waiting for the server…" }
        if planHash == nil { return "Review the plan first — submission is pinned to its hash." }
        if plannedRequest != request || plannedGPUType != gpuType {
            return "The fields changed since the plan was reviewed — review it again."
        }
        return nil
    }

    private var failureLine: some View {
        Label(failure ?? " ", systemImage: "exclamationmark.triangle.fill")
            .font(.caption)
            .foregroundStyle(.orange)
            .textSelection(.enabled)
            .lineLimit(2)
            .opacity(failure == nil ? 0 : 1)
            .accessibilityHidden(failure == nil)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var jobLine: some View {
        Text(jobID.map { "Job: " + $0 + " — retained in the jobs list after this window closes." }
            ?? " ")
            .font(.caption)
            .textSelection(.enabled)
            .lineLimit(1)
    }

    private func perform(_ operation: @escaping @MainActor () async throws -> Void) {
        guard !busy else { return }
        busy = true
        Task {
            defer { busy = false }
            do {
                try await operation()
                failure = nil
            } catch {
                // The plan is invalidated, but the output box keeps what the
                // researcher was reading; the refusal gets its own line.
                plan = nil
                failure = Self.describe(error)
                    + " — inspect the jobs list on this server before retrying "
                    + "a submission."
            }
        }
    }

    private func describe(_ value: some Encodable) -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        return (try? String(decoding: encoder.encode(value), as: UTF8.self)) ?? "Could not display response."
    }

    /// The server's own words for a refusal, never the Swift debug form.
    private static func describe(_ error: any Error) -> String {
        if let client = error as? ClusterClient.ClientError {
            return ClusterClient.unwrappingDetail(client).description
        }
        return error.localizedDescription
    }
}

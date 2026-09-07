import ExperimentKit
import SwiftUI

/// A captured client keeps planning, submitting and observation on one endpoint.
struct ScientificExecutionSheet: View {
    let client: ClusterClient
    let endpoint: String
    @Environment(\.dismiss) private var dismiss
    @State private var operation = "battery"
    @State private var batteryFile = ""
    @State private var agents = "baseline"
    @State private var model = ""
    @State private var revision = ""
    @State private var study = ""
    @State private var concept = ""
    @State private var resamples = 32
    @State private var fraction = 0.5
    @State private var seed = "0"
    @State private var orderShuffles = 8
    @State private var alphaUnits = "norm"
    @State private var dtype = "auto"
    @State private var device = ""
    @State private var plan: JSONValue?
    @State private var plannedRequest: JSONValue?
    @State private var jobID: String?
    @State private var output = ""
    @State private var busy = false
    /// The last refusal, in its own line: it used to be pasted into the
    /// output box on top of the plan the researcher was reading.
    @State private var failure: String?
    @State private var confirmingCancel = false

    private var request: JSONValue {
        var parameters: [String: JSONValue] = operation == "battery"
            ? ["batteryFile": .string(batteryFile), "agents": .array(agents.split(separator: "\n").map { .string(String($0)) }),
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
        .frame(minWidth: 820, minHeight: 610)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Run a scientific diagnostic").font(.title2)
            Text(endpoint).font(.caption).textSelection(.enabled)
            Text("Inputs must already be staged on this server through the "
                + "site's permitted transfer workflow. Review the plan before "
                + "submitting. Diagnostics retain their own output type and do "
                + "not resume from checkpoints.")
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
                TextField("Battery path relative to server workspace", text: $batteryFile)
                    .help("path to the battery file inside the server's "
                        + "workspace — this sheet transfers nothing, so it "
                        + "must already be staged there")
                TextField("Model ID", text: $model)
                    .help("the model to run the battery on; blank leaves the "
                        + "choice to the server's own default")
                TextField("Pinned model revision", text: $revision)
                    .help("the exact model revision to pin — blank means the "
                        + "server records whatever it resolves")
                Text("Agents: one baseline, condition or artifact reference per line")
                TextEditor(text: $agents)
                    .frame(height: 60)
                    .accessibilityLabel("Agents, one per line")
                    .help("one agent per line — a baseline, a condition, or a "
                        + "reference to an artifact this server holds")
            } else {
                TextField("Study name on server", text: $study)
                    .help("the study whose extraction is being resampled, by "
                        + "the name it has on this server")
                TextField("Concept", text: $concept)
                    .help("which of that study's concepts to resample")
            }
            DisclosureGroup("Execution parameters") { parameters }
        }
        .disabled(busy || jobID != nil)
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
                        plan = try await client.scientificPlan(captured)
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
                    guard let plannedRequest, let planHash, plannedRequest == request else { return }
                    perform {
                        let result = try await client.scientificSubmit(plannedRequest, planSHA256: planHash)
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
        if plannedRequest != request {
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

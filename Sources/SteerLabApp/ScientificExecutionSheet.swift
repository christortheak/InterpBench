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
            Text("Run a scientific diagnostic").font(.title2)
            Text(endpoint).font(.caption).textSelection(.enabled)
            Text("Inputs must already be staged on this server through the site's permitted transfer workflow. Review the plan before submitting. Diagnostics retain their own output type and do not resume from checkpoints.")
                .font(.caption)
            Form {
                Picker("Diagnostic", selection: $operation) {
                    Text("Capability battery").tag("battery")
                    Text("Extraction stability").tag("stability")
                }
                if operation == "battery" {
                    TextField("Battery path relative to server workspace", text: $batteryFile)
                    TextField("Model ID", text: $model)
                    TextField("Pinned model revision", text: $revision)
                    Text("Agents: one baseline, condition or artifact reference per line")
                    TextEditor(text: $agents).frame(height: 60)
                } else {
                    TextField("Study name on server", text: $study)
                    TextField("Concept", text: $concept)
                }
                DisclosureGroup("Execution parameters") {
                    if operation == "battery" {
                        Picker("Alpha units", selection: $alphaUnits) {
                            Text("Residual norm").tag("norm")
                            Text("Raw").tag("raw")
                        }
                    } else {
                        Stepper("Resamples: \(resamples)", value: $resamples, in: 2...1_000_000)
                        TextField("Fraction", value: $fraction, format: .number)
                        TextField("Seed (UInt64 decimal)", text: $seed)
                        Stepper("Order shuffles: \(orderShuffles)", value: $orderShuffles, in: 0...1_000_000)
                    }
                    Picker("Data type", selection: $dtype) {
                        Text("Automatic").tag("auto")
                        Text("Float32").tag("float32")
                        Text("BFloat16").tag("bfloat16")
                        Text("Float16").tag("float16")
                    }
                    TextField("Device (blank for engine default)", text: $device)
                }
            }.disabled(busy || jobID != nil)
            HStack {
                Button("Review plan") {
                    let captured = request
                    perform {
                        plan = try await client.scientificPlan(captured)
                        plannedRequest = captured
                        output = describe(plan!)
                    }
                }.disabled(busy || jobID != nil)
                Button("Submit reviewed diagnostic") {
                    guard let plannedRequest, let planHash, plannedRequest == request else { return }
                    perform {
                        let result = try await client.scientificSubmit(plannedRequest, planSHA256: planHash)
                        if case .object(let object) = result, case .string(let id) = object["jobId"] { jobID = id }
                        // Never retry an ambiguous submission automatically.
                        plan = nil
                        output = describe(result)
                    }
                }.disabled(busy || planHash == nil || plannedRequest != request || jobID != nil)
                if let jobID {
                    Button("Refresh job") { perform { output = describe(try await client.job(jobID)) } }.disabled(busy)
                    Button("Cancel job") {
                        perform { try await client.cancelJob(jobID); output = describe(try await client.job(jobID)) }
                    }.disabled(busy)
                }
                Spacer()
                Button("Close") { dismiss() }
            }
            if let jobID { Text("Job: " + jobID + " — retained in Server Jobs after this window closes.").font(.caption).textSelection(.enabled) }
            ScrollView { Text(output).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
        }.padding().frame(minWidth: 820, minHeight: 610)
    }

    private func perform(_ operation: @escaping @MainActor () async throws -> Void) {
        busy = true
        Task {
            defer { busy = false }
            do { try await operation() }
            catch { plan = nil; output = error.localizedDescription + "\nInspect Server Jobs on " + endpoint + " before retrying a submission." }
        }
    }
    private func describe(_ value: some Encodable) -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        return (try? String(decoding: encoder.encode(value), as: UTF8.self)) ?? "Could not display response."
    }
}

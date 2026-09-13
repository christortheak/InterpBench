import ExperimentKit
import SwiftUI

struct StudyMeasurementsView: View {
    let manifest: ExperimentManifest
    let root: URL
    let didSave: () -> Void
    @State private var showing = false
    var body: some View {
        GroupBox("Measurements") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Read trained probes during study responses without changing the agent’s behavior. Scores are saved with each response.").font(.caption)
                Text("\(ProbeMeasurements.references(manifest.probeMeasurements).count) probe(s) selected · executes on Python Compute").font(.caption).foregroundStyle(.secondary)
                Button("Choose study probes…") { showing = true }.disabled(manifest.status != .draft)
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
        .sheet(isPresented: $showing) {
            StudyMeasurementsEditor(manifest: manifest, root: root, didSave: didSave)
        }
    }
}

private struct StudyMeasurementsEditor: View {
    let manifest: ExperimentManifest
    let root: URL
    let didSave: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var probes: [ProbeLibrary.Record] = []
    @State private var selected = ""
    @State private var entries: [JSONValue] = []
    @State private var conditions = ""
    @State private var agents = ""
    @State private var prefill = true
    @State private var decode = true
    @State private var recordingStage = "postAction"
    @State private var retain = false
    @State private var maxReadings = 4096
    @State private var maxBytes = 1048576
    @State private var onError = "recordMissing"
    @State private var review: JSONValue?
    @State private var reviewedSettings: JSONValue?
    @State private var status = ""
    @State private var busy = false
    private func names(_ value: String) -> JSONValue {
        .array(value.split(separator: ",").map { .string($0.trimmingCharacters(in: .whitespacesAndNewlines)) })
    }
    private var settings: JSONValue {
        .object(["schemaVersion": .number(1), "probes": .array(entries), "onError": .string(onError),
            "retainActivations": .bool(retain), "maxReadings": .number(Double(maxReadings)), "maxActivationBytes": .number(Double(maxBytes))])
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Study measurements").font(.title2)
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Choose a probe trained for the executing model, layer, precision, and rendering. Prompt readings observe the input; decode readings observe the growing response. A probe trained on full examples may not generalize to partial responses.")
                    GroupBox("Add a measurement") {
                        VStack(alignment: .leading, spacing: 8) {
                            Picker("Probe", selection: $selected) {
                                Text("Choose…").tag("")
                                ForEach(probes) { probe in Text("\(probe.label) · \(probe.modelID) · layer \(probe.layer)").tag(probe.path) }
                            }
                            TextField("Conditions (blank means all)", text: $conditions)
                                .help("Comma-separated condition names. This selects where to record; it does not change those conditions.")
                            TextField("Seats (blank means all; panel studies)", text: $agents)
                                .help("Comma-separated seat IDs for panel studies. For ordinary studies, the agent instance is identified by its condition name.")
                            Toggle("Read the prompt", isOn: $prefill)
                            Toggle("Read during response generation", isOn: $decode)
                            Picker("Read at the selected site", selection: $recordingStage) {
                                Text("After steering at this site").tag("postAction")
                                Text("Before steering at this site").tag("preAction")
                            }.help("Earlier layers and tokens may already have been steered. The probe's artifact determines block input versus block output and which prompt positions are read.")
                            Button("Add probe") {
                                guard let probe = probes.first(where: { $0.path == selected }) else { return }
                                entries.append(.object(["id": .string(UUID().uuidString), "probe": .object(["path": .string(probe.path), "sha256": .string(probe.sha256)]), "conditions": names(conditions), "agents": names(agents), "stages": .array((prefill ? [JSONValue.string("prefill")] : []) + (decode ? [.string("decode")] : [])), "recordingStage": .string(recordingStage)]))
                                review = nil
                            }.disabled(selected.isEmpty || (!prefill && !decode))
                        }.padding(6)
                    }
                    ForEach(entries.indices, id: \.self) { index in
                        HStack {
                            Text(entryLabel(entries[index])).font(.caption).textSelection(.enabled)
                            Spacer()
                            Button("Remove") { entries.remove(at: index); review = nil }
                        }
                    }
                    DisclosureGroup("Recording limits and error handling") {
                        Toggle("Also retain activations", isOn: $retain)
                            .help("Scores alone cannot be used to fit another probe later. Activations use much more storage; retention is bounded per response.")
                        Stepper("Up to \(maxReadings) readings per response", value: $maxReadings, in: 1...65536)
                        Stepper("Activation storage budget: \(maxBytes) bytes per response", value: $maxBytes, in: 1...16777216)
                        Picker("If a score cannot be computed", selection: $onError) {
                            Text("Record a missing reading and continue").tag("recordMissing")
                            Text("Stop the run").tag("stop")
                        }
                    }
                    Text("No extra model pass reads the final emitted token. Missing scores and budget omissions are reported explicitly. These readings are not calibrated probabilities or evidence of a causal effect.").font(.caption)
                    if case .object(let detail) = review, case .array(let advisories) = detail["advisories"] {
                        ForEach(advisories.indices, id: \.self) { index in
                            if case .string(let message) = advisories[index] { Text(message).font(.caption) }
                        }
                    }
                    if review != nil { Text("Review ready: \(entries.count) selected measurement(s). The review pins the settings, study bytes, and probe bytes. Saving does not run a model.") }
                    if !status.isEmpty { Text(status).foregroundStyle(.secondary).textSelection(.enabled) }
                }.disabled(busy)
            }
            HStack {
                Button("Close") { dismiss() }
                Spacer()
                if busy { ProgressView().controlSize(.small) }
                Button("Review measurements") { performReview() }.disabled(busy)
                Button("Save to study") { save() }.disabled(busy || review == nil || reviewedSettings != settings)
            }
        }.padding(24).frame(minWidth: 650, idealWidth: 720, minHeight: 600)
        .task {
            if case .object(let config) = manifest.probeMeasurements {
                if case .array(let values) = config["probes"] { entries = values }
                if case .bool(let value) = config["retainActivations"] { retain = value }
                if case .number(let value) = config["maxReadings"] { maxReadings = Int(value) }
                if case .number(let value) = config["maxActivationBytes"] { maxBytes = Int(value) }
                if case .string(let value) = config["onError"] { onError = value }
            }
            do { probes = try await ProbeLibrary.inventory(root: root).probes.filter { $0.format == "activation-probe-v1" } }
            catch { status = error.localizedDescription }
        }
    }
    private func entryLabel(_ value: JSONValue) -> String {
        guard case .object(let item) = value, case .object(let probe) = item["probe"], case .string(let path) = probe["path"] else { return "Invalid measurement; remove and add it again." }
        let label = probes.first(where: { $0.path == path })?.label ?? path
        func list(_ key: String, empty: String) -> String {
            guard case .array(let values) = item[key] else { return empty }
            let names = values.compactMap { if case .string(let s) = $0 { return s }; return nil }
            return names.isEmpty ? empty : names.joined(separator: ", ")
        }
        let stage = item["recordingStage"] == .string("preAction") ? "before steering at this site" : "after steering at this site"
        return label + "\nConditions: " + list("conditions", empty: "all") + " · seats/agents: " + list("agents", empty: "all")
            + "\nRead: " + list("stages", empty: "none") + " · " + stage + "\n" + path
    }
    private func performReview() {
        busy = true; let captured = settings
        Task { defer { busy = false }
            do { review = try await ProbeMeasurements.request("measurements-review", experiment: manifest.name, settings: captured, root: root); reviewedSettings = captured; status = "Review complete. Check the selected probes and observation schedule above." }
            catch { review = nil; status = error.localizedDescription }
        }
    }
    private func save() {
        guard case .object(let value) = review, case .string(let hash) = value["planSHA256"], let reviewedSettings else { return }
        busy = true
        Task { defer { busy = false }
            do { _ = try await ProbeMeasurements.request("measurements-save", experiment: manifest.name, settings: reviewedSettings, root: root, planSHA256: hash); didSave(); dismiss() }
            catch { review = nil; status = error.localizedDescription }
        }
    }
}

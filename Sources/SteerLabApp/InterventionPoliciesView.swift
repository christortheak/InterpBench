import AppKit
import ExperimentKit
import SwiftUI

struct InterventionPoliciesView: View {
    let root: URL
    @Environment(\.dismiss) private var dismiss
    @State private var probes: [ProbeLibrary.Record] = []
    @State private var agents: [AgentArtifactDocument] = []
    @State private var policies: [PolicyRow] = []
    @State private var probePath = ""
    @State private var agentPath = ""
    @State private var selectedPolicies = Set<String>()
    @State private var name = "conditional-agent"
    @State private var policyName = "conditional-steering"
    @State private var rule = "threshold"
    @State private var action = "add"
    @State private var threshold = 0.0
    @State private var strength = 1.0
    @State private var lower = 0.0
    @State private var upper = 1.0
    @State private var slope = 1.0
    @State private var intercept = 0.0
    @State private var vectorPath = ""
    @State private var tokenText = ""
    @State private var expertSettings: JSONValue?
    @State private var review: JSONValue?
    @State private var reviewedSettings: JSONValue?
    @State private var publishAction = "policy-publish"
    @State private var status = ""
    @State private var busy = false

    private struct PolicyRow: Decodable, Identifiable {
        let path: String
        let name: String
        var id: String { path }
    }
    private struct Inventory: Decodable { let policies: [PolicyRow]; let issues: [ProbeLibrary.InspectionIssue] }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack { Text("Intervention policies").font(.title2); Spacer(); Button("Done") { dismiss() } }
            Text("A probe reads an agent. A policy uses a reading to change that agent’s behavior. Save the policy, attach it to a new agent version, and compare that agent with a baseline in a study using Python Compute.")
            Text("Workspace: \(root.path)").font(.caption).textSelection(.enabled)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    GroupBox("1. Decide when to intervene") {
                        VStack(alignment: .leading) {
                            TextField("Policy name", text: $policyName)
                            Picker("Fitted probe", selection: $probePath) {
                                Text("Choose a portable probe…").tag("")
                                ForEach(probes.filter { $0.format == "activation-probe-v1" }) { Text("\($0.label) · \($0.modelID) · layer \($0.layer)").tag($0.path) }
                            }
                            Picker("Rule", selection: $rule) {
                                Text("Above a score threshold").tag("threshold")
                                Text("Strength follows the score, within bounds").tag("affine")
                                Text("Fixed strength (comparison control)").tag("fixed")
                            }
                            if rule == "threshold" {
                                TextField("Act when score is greater than", value: $threshold, format: .number)
                                Text("At or below the threshold, the action is off. Choose the threshold using validation data, then evaluate on untouched examples.").font(.caption)
                            }
                            if rule == "affine" {
                                TextField("Score multiplier", value: $slope, format: .number)
                                TextField("Added strength", value: $intercept, format: .number)
                                Text("Strength = score × multiplier + added strength, clipped to your bounds.").font(.caption)
                            } else { TextField("Action strength", value: $strength, format: .number) }
                        }.padding(8)
                    }
                    GroupBox("2. Choose the change") {
                        VStack(alignment: .leading) {
                            Picker("Action", selection: $action) {
                                Text("Add a steering vector").tag("add")
                                Text("Remove a direction").tag("ablate")
                                Text("Bias token logits").tag("logitBias")
                                Text("Allow only selected tokens").tag("allowTokens")
                                Text("Force one token").tag("forceToken")
                            }
                            if action == "add" || action == "ablate" {
                                HStack { Text(vectorPath.isEmpty ? "Choose a Python vector artifact for this model." : vectorPath).textSelection(.enabled)
                                    Button("Choose vector…") { chooseVector() } }
                                Text("The action runs at the probe’s layer and input/output site. Addition uses the vector’s stored units; removal strength 1 removes its component, and 0 leaves it alone.").font(.caption)
                            } else {
                                TextField("Exact token IDs, separated by commas", text: $tokenText)
                                Text("IDs belong to this probe’s pinned tokenizer. Bias changes preference; token constraints are on above zero and never override existing forbidden tokens. Constraints use bounds 0 and 1.").font(.caption)
                            }
                            HStack {
                                TextField("Minimum strength", value: $lower, format: .number)
                                TextField("Maximum strength", value: $upper, format: .number)
                            }
                            Text("This starter acts at the last prompt position and each generated-prefix position. Advanced settings can change the schedule, combine probe scores, or declare a trusted Python provider.").font(.caption)
                        }.padding(8)
                    }
                    HStack {
                        Button("Review policy") { work { try await reviewPolicy() } }.disabled(probePath.isEmpty)
                        Button("Choose advanced settings…") { chooseSettings() }
                        if expertSettings != nil { Button("Review advanced settings") { work { try await reviewPolicy(advanced: true) } } }
                    }
                    GroupBox("3. Attach saved policies to an agent") {
                        VStack(alignment: .leading) {
                            Picker("Source agent", selection: $agentPath) {
                                Text("Choose an existing agent…").tag("")
                                ForEach(agents, id: \.path) { Text($0.artifact.name).tag($0.path) }
                            }
                            TextField("New agent name", text: $name)
                            ForEach(policies) { policy in
                                Toggle(policy.name, isOn: Binding(get: { selectedPolicies.contains(policy.path) }, set: { if $0 { selectedPolicies.insert(policy.path) } else { selectedPolicies.remove(policy.path) } }))
                                Button("Inspect \(policy.name)") { work { review = try await InterventionPolicyLibrary.call("policy-inspect", root: root, path: policy.path); reviewedSettings = nil } }.font(.caption)
                            }
                            Text("The selected policies replace the new version’s policy list. Select none to create a comparison agent with the same static vectors and adapters, but no policies.").font(.caption)
                            Button("Review new agent") { work { try await reviewAttachment() } }.disabled(agentPath.isEmpty)
                        }.padding(8)
                    }
                    if let review {
                        GroupBox("Reviewed definition") {
                            VStack(alignment: .leading) {
                                Text("Saving uses this reviewed definition. To use edited fields, review them again. No model runs when a policy is saved.").font(.caption)
                                if case .object(let body) = review, case .array(let notes) = body["limitations"] {
                                    ForEach(notes.indices, id: \.self) { i in if case .string(let text) = notes[i] { Text(text) } }
                                }
                                DisclosureGroup("Exact settings and input hashes") {
                                    PolicyJSONView(value: review)
                                }
                                if reviewedSettings != nil { Button(publishAction == "policy-attach" ? "Save new agent version" : "Save policy") { work { try await publish() } } }
                            }.padding(8)
                        }
                    }
                }
            }.disabled(busy)
            if busy { ProgressView("Reading and checking the selected inputs…") }
            if !status.isEmpty { Text(status).textSelection(.enabled) }
        }.padding(24).frame(minWidth: 760, idealWidth: 840, minHeight: 700)
        .task { work { try await refresh() } }
    }

    private func work(_ operation: @escaping @MainActor () async throws -> Void) {
        guard !busy else { return }; busy = true; status = ""
        Task { defer { busy = false }; do { try await operation() } catch { status = error.localizedDescription } }
    }
    private func refresh() async throws {
        probes = try await ProbeLibrary.inventory(root: root).probes
        let inventory: Inventory = try ProbeLibrary.decode(await InterventionPolicyLibrary.call("policy-list", root: root))
        policies = inventory.policies
        agents = try await Task.detached { try StudyAgentAuthoring.list(workspaceRoot: root).agents }.value
        if !inventory.issues.isEmpty { status = inventory.issues.map(\.reason).joined(separator: "\n") }
    }
    private func reviewPolicy(advanced: Bool = false) async throws {
        let settings: JSONValue
        if advanced, let expertSettings { settings = expertSettings }
        else {
            let probe = try await ProbeLibrary.inspect(path: probePath, root: root)
            let tokens = try tokenText.split(separator: ",", omittingEmptySubsequences: false).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.map { word -> Int in
                let text = word.trimmingCharacters(in: .whitespaces)
                guard text.allSatisfy({ "0123456789".contains($0) }), let token = Int(text) else { throw ExperimentError(reason: "Use comma-separated nonnegative token IDs.") }; return token
            }
            settings = try InterventionPolicyLibrary.starter(probe: probe, name: policyName, rule: rule, action: action,
                threshold: threshold, strength: strength, lower: lower, upper: upper, slope: slope, intercept: intercept, vectorPath: vectorPath, tokens: tokens)
        }
        review = try await InterventionPolicyLibrary.call("policy-review", root: root, settings: settings)
        reviewedSettings = settings; publishAction = "policy-publish"
    }
    private func reviewAttachment() async throws {
        let settings: JSONValue = .object(["agentPath": .string(agentPath), "name": .string(name), "policyPaths": .array(selectedPolicies.sorted().map(JSONValue.string))])
        review = try await InterventionPolicyLibrary.call("policy-attach-review", root: root, settings: settings)
        reviewedSettings = settings; publishAction = "policy-attach"
    }
    private func publish() async throws {
        guard let settings = reviewedSettings, case .object(let body) = review, case .string(let hash) = body["planSHA256"] else { return }
        let result = try await InterventionPolicyLibrary.call(publishAction, root: root, settings: settings, hash: hash)
        reviewedSettings = nil; review = result
        try await refresh()
        status = "Saved in this workspace. Refresh the agent library to use a newly created version in a study."
    }
    private func chooseVector() {
        let picker = NSOpenPanel(); picker.directoryURL = root.appending(path: "runs"); picker.canChooseDirectories = false
        guard picker.runModal() == .OK, let url = picker.url else { return }
        vectorPath = url.deletingPathExtension().path
    }
    private func chooseSettings() {
        let picker = NSOpenPanel(); picker.directoryURL = root; picker.canChooseDirectories = false
        guard picker.runModal() == .OK, let url = picker.url else { return }
        work {
            expertSettings = try await Task.detached { try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url)) }.value
            status = "Advanced settings loaded. Review displays the exact code and inputs before saving; providers are trusted code, not sandboxed plugins."
        }
    }
}

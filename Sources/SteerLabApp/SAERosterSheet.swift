import ExperimentKit
import SwiftUI

struct SAERosterSheet: View {
    let root: URL
    @Environment(\.dismiss) private var dismiss
    @State private var path = ""
    @State private var experiment = ""
    @State private var planHash: String?
    @State private var output = ""
    @State private var busy = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("SAE candidates and qualification records").font(.title2)
            Text(root.path).font(.caption).textSelection(.enabled)
            Text("Inspect nominations and their recorded evidence, then pin a roster to a draft. Inspection does not qualify a feature; pinning does not promote or seat it.")
            TextField("Workspace-relative roster or qualification path", text: $path)
                .onChange(of: path) { _, _ in planHash = nil }
            TextField("Draft study name (for pinning)", text: $experiment)
                .onChange(of: experiment) { _, _ in planHash = nil }
            HStack {
                Button("Check candidate roster") { run("sae-check") }
                Button("Show qualification record") { run("sae-show") }
                Button("Review roster pin") { run("sae-pin-plan") }.disabled(experiment.isEmpty)
                Button("Pin reviewed roster") { run("sae-pin") }.disabled(planHash == nil)
            }.disabled(path.isEmpty || busy)
            if busy { ProgressView() }
            ScrollView { Text(output).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
            HStack { Spacer(); Button("Done") { dismiss() }.disabled(busy) }
        }.padding().frame(minWidth: 800, minHeight: 600).interactiveDismissDisabled(busy)
    }
    private func run(_ action: String) {
        var payload: [String: JSONValue] = ["workspaceRoot": .string(root.path), "path": .string(path)]
        if ["sae-pin-plan", "sae-pin"].contains(action) { payload["experiment"] = .string(experiment) }
        if action == "sae-pin", let planHash { payload["planSHA256"] = .string(planHash) }
        busy = true; planHash = nil
        Task {
            defer { busy = false }
            do {
                let result = try await DiagnosticWorkspace.perform(action, payload: payload)
                if case .object(let object) = result, case .string(let text) = object["planSHA256"] { planHash = text }
                let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                output = String(decoding: try encoder.encode(result), as: UTF8.self)
            } catch { output = error.localizedDescription }
        }
    }
}

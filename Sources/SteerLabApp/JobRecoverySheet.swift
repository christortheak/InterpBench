import ExperimentKit
import SwiftUI

struct JobRecoverySheet: View {
    let client: ClusterClient
    let jobID: String
    @Environment(\.dismiss) private var dismiss
    @State private var review: [String: JSONValue] = [:]
    @State private var report = ""
    @State private var reason = ""
    @State private var confirmed = false
    @State private var busy = false
    @State private var recovered = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Review controller recovery").font(.title2)
            Text(client.profile.baseURL.absoluteString + " · " + jobID).font(.caption).textSelection(.enabled)
            Text("Recovery records your assertion that the original controller has exited. Inspect its ownership evidence first. It does not restart the computation or authorize a second submission.")
            ScrollView { Text(report).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
            TextField("Reason and evidence that the owner exited", text: $reason)
            Toggle("I have established that the recorded owner has exited", isOn: $confirmed)
            HStack {
                Button("Refresh review") { Task { await load() } }.disabled(busy)
                Button("Recover this job") {
                    guard case .string(let token) = review["reviewToken"] else { return }
                    busy = true
                    Task {
                        defer { busy = false }
                        do {
                            let result = try await client.recoverJob(jobID, reviewToken: token, reason: reason)
                            report = describe(result); recovered = true
                        } catch { review = [:]; report = error.localizedDescription }
                    }
                }.disabled(busy || recovered || !confirmed || reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || review["eligible"] != .bool(true))
                Spacer()
                Button("Close") { dismiss() }
            }
        }.padding().frame(minWidth: 760, minHeight: 540)
        .task { await load() }
    }

    private func load() async {
        busy = true; confirmed = false
        defer { busy = false }
        do {
            let result = try await client.recoveryReview(jobID)
            if case .object(let object) = result { review = object }
            report = describe(result)
        } catch { review = [:]; report = error.localizedDescription }
    }
    private func describe(_ value: JSONValue) -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return (try? String(decoding: encoder.encode(value), as: UTF8.self)) ?? "Could not display recovery response."
    }
}

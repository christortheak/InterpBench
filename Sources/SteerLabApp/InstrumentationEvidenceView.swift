import ExperimentKit
import SwiftUI

/// Reads the complete source through the shared offline owner, independently of preview limits.
struct InstrumentationEvidenceView: View {
    let root: URL
    let path: String
    @Environment(\.dismiss) private var dismiss
    @State private var report: Report?
    @State private var raw: JSONValue?
    @State private var filter = ""
    @State private var error = ""
    private struct Moments: Decodable {
        let count: Int
        let mean: Double?
        let minimum: Double?
        let maximum: Double?
        let nonzeroCount: Int
    }
    private struct Group: Decodable {
        let condition: String
        let agent: String
        let kind: String
        let artifactSHA256: String
        let label: String
        let site: JSONValue?
        let recordingStage: String
        let statuses: [String: Int]
        let scores: [String: Moments]
        let requested: [String: Moments]
        let applied: [String: Moments]
        var heading: String { "\(condition) · \(agent.isEmpty ? "agent" : agent) · \(label) · \(artifactSHA256.prefix(12)) · \(recordingStage)" }
    }
    private struct Report: Decodable {
        let responses: Int
        let uninstrumentedResponses: Int
        let partialResponses: Int
        let omittedReadings: Int
        let omittedDecisions: Int
        let groups: [Group]
        let limitations: [String]
        let sourceSHA256: String
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack { Text("Probe readings and policy actions").font(.title2); Spacer(); Button("Done") { dismiss() } }
            Text("Compare what probes read, what policies requested, and which actions were applied. These summaries describe the computation; use separate behavioral outcomes to judge whether an intervention helped.")
            if let report {
                Text("\(report.responses) responses · \(report.uninstrumentedResponses) without instrumentation · \(report.partialResponses) with partial evidence")
                Text("Recording limits omitted \(report.omittedReadings) readings and \(report.omittedDecisions) decisions.").font(.caption)
                TextField("Filter by condition, agent, artifact hash, or reading stage", text: $filter)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(report.groups.indices, id: \.self) { i in
                            let group = report.groups[i]
                            if filter.isEmpty || group.heading.localizedCaseInsensitiveContains(filter) {
                                GroupBox(group.heading) {
                                    VStack(alignment: .leading, spacing: 6) {
                                        if let site = group.site { Text((try? InterventionPolicyLibrary.formatted(site)) ?? "").font(.caption.monospaced()) }
                                        Text(group.statuses.keys.sorted().map { "\($0): \(group.statuses[$0]!)" }.joined(separator: ", ")).font(.caption)
                                        metrics("Probe scores", group.scores)
                                        metrics("Requested strengths", group.requested)
                                        metrics("Applied strengths", group.applied)
                                    }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                                }
                            }
                        }
                        ForEach(report.limitations, id: \.self) { Text($0).font(.caption) }
                        if let raw { DisclosureGroup("Complete summary and source hash") { PolicyJSONView(value: raw) } }
                    }
                }
            } else if error.isEmpty { ProgressView("Reading the complete evidence file…") }
            if !error.isEmpty { Text(error).textSelection(.enabled) }
        }.padding(24).frame(minWidth: 760, minHeight: 600)
        .task {
            do {
                let value = try await DiagnosticWorkspace.perform("evidence-analyze", payload: ["workspaceRoot": .string(root.path), "path": .string(path)])
                report = try ProbeLibrary.decode(value); raw = value
            } catch { self.error = error.localizedDescription }
        }
    }
    @ViewBuilder private func metrics(_ title: String, _ values: [String: Moments]) -> some View {
        if !values.isEmpty {
            Text(title).font(.headline)
            ForEach(values.keys.sorted(), id: \.self) { name in
                if let value = values[name] {
                    Text("\(name): \(value.count) observations; mean \(value.mean.map { String(format: "%.4g", $0) } ?? "—"); range \(value.minimum.map { String(format: "%.4g", $0) } ?? "—") to \(value.maximum.map { String(format: "%.4g", $0) } ?? "—"); \(value.nonzeroCount) nonzero.")
                        .font(.caption).textSelection(.enabled)
                }
            }
        }
    }
}

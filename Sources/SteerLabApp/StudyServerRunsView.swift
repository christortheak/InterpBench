import ExperimentKit
import SwiftUI

/// Read-only listing of immutable run directories on the active server.
struct StudyServerRunsView: View {
    let runs: [RemoteRunRecord]
    let substrateLabel: String
    let refresh: () async -> Void

    @ViewBuilder
    var body: some View {
        Group {
            Text("Server runs — \(substrateLabel)")
                .font(.caption.bold())
                .padding(.top, 4)
            Text(
                "Every immutable run directory on the server — any verb, "
                    + "any study. Pipelines (below) is the chain-level view: one "
                    + "row per chain with per-stage status and gate aborts."
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            Button("Refresh Server Runs") {
                Task { await refresh() }
            }
            .help("list the immutable run directories in the active server's runs/ tree")
            if runs.isEmpty {
                Text("No server runs listed — refresh, or run something on this server first.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(runs.prefix(40)) { run in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(run.id)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                        HStack(spacing: 8) {
                            if run.hasReport { Text("report") }
                            if run.hasGenerations { Text("generations") }
                            if !run.vectorNames.isEmpty {
                                Text(
                                    "\(run.vectorNames.count) vector\(run.vectorNames.count == 1 ? "" : "s")"
                                )
                            }
                            if let task = run.task, !task.isEmpty {
                                Text(task).lineLimit(1)
                            }
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 2)
                }
                if runs.count > 40 {
                    Text("… and \(runs.count - 40) more")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

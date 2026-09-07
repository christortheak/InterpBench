import ExperimentKit
import SwiftUI

/// Ledger evidence and scientific gate determinations for the selected study.
struct StudyPipelinesView: View {
    let pipelines: StudyPipelineController
    let substrateLabel: String
    let refresh: () async -> Void
    let duplicateStudy: () -> Void

    /// How many rows each list renders before the "N more" line. The listing
    /// is a summary; silently dropping the tail made a truncated list read as
    /// a complete one.
    private static let serverRowLimit = 20
    private static let localRowLimit = 10

    @State private var isRefreshing = false

    @ViewBuilder
    var body: some View {
        Group {
            Text("Pipelines — \(substrateLabel)")
                .font(.caption.bold())
                .padding(.top, 4)
            HStack(spacing: 8) {
                Button(isRefreshing ? "Refreshing…" : "Refresh Pipelines") {
                    guard !isRefreshing else { return }
                    isRefreshing = true
                    Task {
                        await refresh()
                        isRefreshing = false
                    }
                }
                .disabled(isRefreshing)
                .help(
                    "list this experiment's chain-runner runs on the active "
                        + "server: per-stage status, gate aborts, and promoted "
                        + "agents")
                if isRefreshing {
                    ProgressView()
                        .controlSize(.small)
                }
                Spacer()
            }
            if pipelines.pipelineRuns.isEmpty {
                Text(
                    "No pipelines listed — refresh, or submit the "
                        + "'pipeline' verb from Remote options."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            } else {
                ForEach(pipelines.pipelineRuns.prefix(Self.serverRowLimit)) { pipeline in
                    pipelineRunRow(pipeline)
                }
                moreRow(
                    total: pipelines.pipelineRuns.count,
                    shown: Self.serverRowLimit,
                    location: "on \(substrateLabel)")
            }
            if !pipelines.localPipelineRuns.isEmpty {
                Text("Imported / local")
                    .font(.caption2.bold())
                    .foregroundStyle(.secondary)
                ForEach(pipelines.localPipelineRuns.prefix(Self.localRowLimit)) { pipeline in
                    pipelineRunRow(pipeline)
                }
                moreRow(
                    total: pipelines.localPipelineRuns.count,
                    shown: Self.localRowLimit,
                    location: "in this workspace's runs/")
            }
        }
    }

    /// The tail this listing does not draw, stated rather than dropped.
    @ViewBuilder
    private func moreRow(total: Int, shown: Int, location: String) -> some View {
        if total > shown {
            Text("\(total - shown) more chain run\(total - shown == 1 ? "" : "s") "
                + "not listed — this summary draws the first \(shown)")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .help(
                    "\(total) chain runs exist \(location); the rest stay in "
                        + "the ledger and are unaffected by what this summary "
                        + "shows")
        }
    }

    @ViewBuilder
    private func pipelineRunRow(
        _ pipeline: ClusterClient.PipelineRunSummary
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Text(pipeline.run)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .lineLimit(1)
                    // The row truncates; the whole run id has to stay
                    // reachable, because it is what names the evidence.
                    .help(pipeline.run)
                Text(pipeline.stateLabel)
                    .font(.caption2.bold())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .background(
                        pipelineStateColor(pipeline).opacity(0.18),
                        in: Capsule())
                if pipeline.manifestStatus == "draft" {
                    Text("draft (exploratory)")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }
            Text(pipeline.stageSummaryLine)
                .font(.caption2)
                .foregroundStyle(.secondary)
            if let updated = pipeline.updatedAt {
                // For "unfinished" chains this is the evidence for judging
                // running-vs-abandoned — the listing cannot know. Rendered in
                // the reader's own locale and time zone; the tooltip keeps the
                // server's verbatim UTC stamp.
                Text("last ledger write: \(Self.timestampLabel(updated))")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .help("recorded by the server as \(updated) (UTC)")
            }
            if let agents = pipeline.promotedAgents, !agents.isEmpty {
                ForEach(agents.keys.sorted(), id: \.self) { concept in
                    if let agent = agents[concept] {
                        Text(promotedAgentLine(concept: concept, agent: agent))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }
            }
            if let abort = pipeline.abort {
                pipelineAbortCard(abort)
            }
        }
        .padding(.vertical, 2)
    }

    /// The ledger's UTC ISO stamp, in the reader's locale. Unparseable
    /// stamps render verbatim rather than disappearing.
    private static func timestampLabel(_ stamp: String) -> String {
        guard let date = HousekeepingDates.parse(stamp) else { return stamp }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    private func pipelineStateColor(
        _ pipeline: ClusterClient.PipelineRunSummary
    ) -> Color {
        switch pipeline.disposition {
        case "completed": .green
        case "aborted": .orange
        default: .blue
        }
    }

    private func promotedAgentLine(
        concept: String,
        agent: ClusterClient.PipelineRunSummary.PromotedAgent
    ) -> String {
        var line = "\(concept) → \(agent.artifact ?? "?")"
        if let cell = agent.winningCell, let layer = cell.layer,
            let alpha = cell.alpha
        {
            line += " (L\(layer), α\(alpha.formatted()))"
        }
        return line
    }

    /// The abort record, rendered as the determination it is — never as a
    /// job failure. Detail strings come verbatim from the server's
    /// GateResult (researcher-facing prose).
    @ViewBuilder
    private func pipelineAbortCard(
        _ abort: ClusterClient.PipelineRunSummary.Abort
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(
                "Stopped at '\(abort.stage ?? "?")' — a gate said stop. "
                    + "Nothing after it ran."
            )
            .font(.caption.bold())
            ForEach(
                Array((abort.gates ?? []).enumerated()), id: \.offset
            ) { _, gate in
                VStack(alignment: .leading, spacing: 1) {
                    if let detail = gate.detail {
                        Text(detail)
                            .font(.caption2)
                            .textSelection(.enabled)
                    }
                    if let measured = gate.measured,
                        let threshold = gate.threshold
                    {
                        Text(
                            "measured \(measured.formatted()) vs threshold "
                                + "\(threshold.formatted())"
                        )
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                    }
                }
            }
            if let evidence = abort.evidenceRunID {
                Text("evidence: \(evidence)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Button("Duplicate & Adjust") { duplicateStudy() }
                .font(.caption)
                .help(
                    "iterate by duplicating, never by editing: creates a "
                        + "draft copy of this experiment to adjust "
                        + "stimuli/gates/grid, leaving the preregistered "
                        + "chain and its abort record intact")
        }
        .padding(6)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }
}

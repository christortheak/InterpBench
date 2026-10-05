import ExperimentKit
import SwiftUI

/// The custom-code notice at the top of a study's page.
///
/// An intervention policy may carry an expert provider: Python the engine runs
/// with the researcher's permissions when the study runs. When the selected
/// study carries one that nobody has acknowledged in this workspace — most
/// often right after a shared pack was pasted in or an agent was added — this
/// section says so, names the code by its SHA-256, shows it on request, and
/// records an acknowledgement when the researcher says they trust the source.
/// Until then the study is not sent to run. Once acknowledged, one quiet line
/// remains saying who acknowledged it and when.
///
/// The review is read once per version of the manifest, not on every redraw:
/// a study's agents can embed large policy files.
struct CustomCodeNoticeSection: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel

    /// Not observable on purpose: refilled during a redraw, it must not ask
    /// for another one.
    private final class Cache {
        var key: ExperimentManifest?
        var state: ExperimentPanel.CustomCodeState?
    }

    @State private var cache = Cache()
    @State private var showCode = false
    @State private var refreshes = 0

    private var state: ExperimentPanel.CustomCodeState {
        _ = refreshes
        if cache.key != manifest || cache.state == nil {
            cache.key = manifest
            cache.state = panel.customCodeState(for: manifest.name)
        }
        return cache.state!
    }

    var body: some View {
        let current = state
        if let review = current.review, review.needsAcknowledgement {
            Section {
                Label(CustomCodeNotice.notice, systemImage: "exclamationmark.shield")
                    .foregroundStyle(.orange)
                ForEach(review.pending, id: \.sha256) { row in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(row.policyNames.isEmpty
                            ? "Policy without a name" : row.policyNames.joined(separator: ", "))
                        Text("SHA-256 \(row.sha256)")
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }
                DisclosureGroup("Show the code", isExpanded: $showCode) {
                    ForEach(review.pending, id: \.sha256) { row in
                        ScrollView(.horizontal) {
                            Text(row.sourceText)
                                .font(.caption.monospaced())
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                if let problem = current.problem {
                    Text(problem)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                Button("I trust this source: acknowledge") {
                    if panel.acknowledgeCustomCode(review) {
                        cache.state = nil
                        refreshes += 1
                    }
                }
                .help(
                    "Records your account name, the code's SHA-256, and the time in "
                        + "\(CustomCodeNotice.fileName) in this workspace. After that "
                        + "the study can be sent to run. The code is not sandboxed.")
            } header: {
                Text("Custom code")
            } footer: {
                Text(
                    "This study will not be sent to run until the code is "
                        + "acknowledged. SteerLab does not sandbox it.")
            }
        } else if let review = current.review {
            Section("Custom code") {
                ForEach(review.providers, id: \.sha256) { row in
                    Text(
                        "Acknowledged by \(row.acknowledgedBy ?? "an unrecorded account") "
                            + "on \(row.acknowledgedAt ?? "an unrecorded date"), "
                            + "SHA-256 \(row.sha256.prefix(12))…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .help("SHA-256 \(row.sha256)")
                }
            }
        }
    }
}

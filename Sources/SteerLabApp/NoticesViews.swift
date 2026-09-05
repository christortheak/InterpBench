import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct NoticesBellButton: View {
    @State private var showFeed = false

    var body: some View {
        let notices = PanelNotices.shared
        Button {
            showFeed = true
            notices.markViewed()
        } label: {
            Image(
                systemName: notices.hasUnseenErrors
                    ? "bell.badge.fill" : "bell"
            )
            .foregroundStyle(
                notices.hasUnseenErrors
                    ? AnyShapeStyle(.red) : AnyShapeStyle(.secondary)
            )
            .imageScale(.medium)
        }
        .buttonStyle(.plain)
        .help(
            "panel notices — every Studies/Agents status event, kept (last "
                + "\(PanelNotices.capacity)) and persisted per workspace; "
                + "errors badge the bell until viewed"
        )
        .popover(isPresented: $showFeed, arrowEdge: .bottom) {
            NoticesFeedView()
        }
    }
}

/// The feed popover: newest first, severity icons, source + timestamp, and a
/// Clear action. Read-only over the store — the panels append, this renders.
struct NoticesFeedView: View {
    var body: some View {
        let notices = PanelNotices.shared
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Notices")
                    .font(.headline)
                Spacer()
                Button("Clear") { notices.clear() }
                    .controlSize(.small)
                    .disabled(notices.notices.isEmpty)
                    .help("empties the notices ring (and its persisted file)")
            }
            if notices.notices.isEmpty {
                Text(
                    "No notices yet — Studies and Agents events land here "
                        + "and persist across the session."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(notices.recentFirst) { notice in
                            noticeRow(notice)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 360)
            }
        }
        .padding(12)
        .frame(width: 460)
    }

    private func noticeRow(_ notice: PanelNotice) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: notice.severity.symbolName)
                .foregroundStyle(severityColor(notice.severity))
                .imageScale(.small)
            VStack(alignment: .leading, spacing: 1) {
                Text(notice.message)
                    .font(.caption)
                    .textSelection(.enabled)
                Text(
                    "\(notice.source) · "
                        + notice.timestamp.formatted(
                            date: .abbreviated, time: .standard)
                )
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
        }
    }

    private func severityColor(_ severity: PanelNotice.Severity) -> Color {
        switch severity {
        case .info: .secondary
        case .success: .green
        case .warning: .orange
        case .error: .red
        }
    }
}

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
        // A bare `Image` reads as its symbol name to VoiceOver, and the
        // unseen-error state was carried by the glyph alone.
        .accessibilityLabel(
            notices.hasUnseenErrors
                ? "Notices — unseen errors" : "Notices")
        .frame(minWidth: 20, minHeight: 20)
        .contentShape(Rectangle())
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
    @State private var confirmingClear = false

    var body: some View {
        let notices = PanelNotices.shared
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Notices")
                    .font(.headline)
                Spacer()
                // Clear deletes the persisted file as well as the in-memory
                // ring — the only record of what the panels reported this
                // session — so it asks first (2026-09-06 audit, headline 7).
                Button("Clear", role: .destructive) { confirmingClear = true }
                    .controlSize(.small)
                    .disabled(notices.notices.isEmpty)
                    .help("empties the notices ring (and its persisted file)")
                    .confirmationDialog(
                        "Clear all \(notices.notices.count) notices?",
                        isPresented: $confirmingClear
                    ) {
                        Button("Clear Notices", role: .destructive) { notices.clear() }
                        Button("Cancel", role: .cancel) {}
                    } message: {
                        Text(
                            "Every recorded Studies and Agents event is deleted, "
                                + "here and in this workspace's saved notices file. "
                                + "Nothing else changes — no study, run, or "
                                + "artifact is touched.")
                    }
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

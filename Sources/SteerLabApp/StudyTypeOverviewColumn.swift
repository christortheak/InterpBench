import ExperimentKit
import SwiftUI

/// The display-pane Study Guide (2026-07-19): a real, readable explanation
/// of the selected study's type — what it is, what the researcher
/// provides, what it measures — rendered where there is room, instead of
/// caption-sized text squeezed into the form. Follows the Studies panel's
/// selection and type picker live.
struct StudyTypeOverviewColumn: View {
    let service: ChatService

    private var panel: ExperimentPanel { service.experiments }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if let manifest = panel.selected {
                    guide(for: panel.studyFocus, manifest: manifest)
                } else {
                    emptyState
                }
            }
            .padding(20)
            .frame(maxWidth: 560, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("No study selected", systemImage: "square.dashed")
                .font(.title3.weight(.semibold))
            Text("Select or create a study on the left. This pane explains "
                + "the selected study's type: what it is, what you provide, "
                + "and what it measures.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func guide(
        for intent: StudyIntent, manifest: ExperimentManifest
    ) -> some View {
        // Header: what kind of study this is, at a glance.
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: intent.systemImage)
                .font(.system(size: 30, weight: .medium))
                .foregroundStyle(.tint)
                .frame(width: 44, height: 44)
                .background(.tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
            VStack(alignment: .leading, spacing: 3) {
                Text(intent.displayName)
                    .font(.title2.weight(.semibold))
                Text(intent.tagline)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }

        LabeledContent("Study") {
            Text("\(manifest.name) — \(manifest.status.rawValue)")
                .font(.callout.monospaced())
        }
        .font(.callout)

        guideSection("What it is", systemImage: "questionmark.circle") {
            Text(intent.whatItIs)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }

        guideSection("You provide", systemImage: "tray.and.arrow.down") {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(intent.youProvide) { item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(item.required ? "required" : "optional")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(
                                (item.required ? Color.orange : Color.secondary)
                                    .opacity(0.15),
                                in: Capsule())
                            .foregroundStyle(
                                item.required ? .orange : .secondary)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(item.name)
                                .font(.callout.weight(.medium))
                            Text(item.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        }

        guideSection("It measures", systemImage: "chart.bar.xaxis") {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(intent.itMeasures, id: \.self) { line in
                    HStack(alignment: .firstTextBaseline, spacing: 7) {
                        Image(systemName: "arrow.right")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Text(line)
                            .font(.callout)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }

        if let hidden = intent.hiddenContentNote(for: manifest) {
            Label(hidden, systemImage: "eye.slash")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }

        Text("Description, task description, outcome measures, phase, and "
            + "case family are notes for the record — never sent to any "
            + "model. The Issues box and Data & Prompts pane in the Studies "
            + "panel list exactly what this study still needs.")
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private func guideSection(
        _ title: String, systemImage: String,
        @ViewBuilder content: () -> some View
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
    }
}

import AppKit
import ExperimentKit
import SwiftUI

/// The display pane's Studies mode: the SELECTED STUDY, in two readings.
///
/// **JSON** (2026-08-26, the default) is the study itself — its manifest,
/// pretty-printed and read-only: the same `experiment.json` document every
/// engine reads, re-encoded from the manifest the panel already decoded
/// (`ExperimentPanel.selectedStudyJSON`), never a second read of the file.
/// **Guide** is the pre-existing explanation of the study's TYPE (2026-07-19)
/// — what it is, what you provide, what it measures — which is about the kind
/// of study, not this one, and is kept a click away rather than displaced.
struct StudyViewerColumn: View {
    let service: ChatService

    private enum Reading: String, CaseIterable, Identifiable {
        case json = "JSON"
        case guide = "Guide"
        var id: String { rawValue }
    }

    @State private var reading: Reading = .json

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $reading) {
                ForEach(Reading.allCases) { item in
                    Text(item.rawValue).tag(item)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .help(
                "JSON: the selected study's manifest — the experiment.json "
                    + "the CLI and server read. Guide: what this study's TYPE "
                    + "is, what you provide, and what it measures.")
            Divider()
            switch reading {
            case .json: StudyManifestJSONColumn(service: service)
            case .guide: StudyTypeOverviewColumn(service: service)
            }
        }
    }
}

/// The selected study's manifest as one read-only, scrollable JSON document.
/// Nothing is parsed or re-read here: the panel hands over the decoded
/// manifest re-encoded pretty-printed with sorted keys — the same encoder
/// (`ExperimentStore.exportStudyJSON`) that Copy Study JSON uses, so what is
/// read here and what is pasted elsewhere are one document.
struct StudyManifestJSONColumn: View {
    let service: ChatService

    private var panel: ExperimentPanel { service.experiments }

    var body: some View {
        if let json = panel.selectedStudyJSON, let manifest = panel.selected {
            VStack(spacing: 0) {
                header(for: manifest)
                Divider()
                // Horizontal scrolling too: manifest values (paths, hashes,
                // pinned prompts) are long, and wrapping a JSON document
                // makes its structure unreadable.
                ScrollView([.vertical, .horizontal]) {
                    Text(json)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        // Unwrapped, so the indentation keeps meaning the
                        // structure; the horizontal axis above is what long
                        // values (paths, hashes, pinned prompts) scroll on.
                        .fixedSize(horizontal: true, vertical: false)
                        .padding(12)
                }
            }
        } else {
            ContentUnavailableView {
                Label("No study selected", systemImage: "curlybraces")
            } description: {
                Text(
                    "Select a study in the Studies pane; its manifest — the "
                        + "experiment.json every engine reads — renders here.")
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func header(for manifest: ExperimentManifest) -> some View {
        HStack(spacing: 8) {
            Text("\(manifest.name) — \(manifest.status.rawValue)")
                .font(.caption.monospaced().weight(.semibold))
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button {
                copy(panel.selectedStudyJSON ?? "")
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .help("copy this manifest JSON")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .help(
            "read-only — a study is edited through the Studies pane's own "
                + "controls (and a frozen study not at all); Paste Study JSON "
                + "imports an edited document as a NEW draft")
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

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

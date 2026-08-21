import ExperimentKit
import SwiftUI

/// Evaluation-section controls for pinning fitted RepE reader artifacts
/// (`readerRefs`).
///
/// Audit 2026-08-01: the "Add reader instrument" button gated on a
/// non-empty `readerRefs` list that nothing in the app could populate — an
/// affordance behind a door with no key. This view is the key: pinned
/// readers render as removable rows, and a picker over
/// `VectorCatalog.scanReaders()` (the fitted-reader artifacts already on
/// disk under runs/) pins one per concept through the draft gate. When the
/// scan finds nothing, it says where readers come from instead of hiding.
///
/// Pattern: `OrdinalScaleInstrumentControls` — the store's draft-edit gate,
/// never a parallel save path.
struct ReaderPinControls: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @State private var available: [VectorCatalog.ReaderArtifactRecord] = []
    @State private var errorText: String?

    private var isDraft: Bool { manifest.status == .draft }
    private var pinned: [ExperimentManifest.ReaderRef] {
        manifest.readerRefs ?? []
    }

    var body: some View {
        Group {
            pinnedRows
            if isDraft {
                pickerOrHint
            }
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        .onAppear { available = VectorCatalog.scanReaders() }
    }

    @ViewBuilder private var pinnedRows: some View {
        ForEach(pinned, id: \.concept) { ref in
            HStack(spacing: 6) {
                Label(
                    "reader '\(ref.concept)'",
                    systemImage: "waveform.path.ecg")
                    .font(.caption)
                Text(ref.path)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                if isDraft {
                    Button("Remove") { remove(ref.concept) }
                        .font(.caption)
                        .help(
                            "unpins this reader; if it was the last one, the "
                                + "reader instrument is removed in the same "
                                + "edit (verify refuses repeReaderScore with "
                                + "no pinned readers)")
                }
            }
        }
    }

    @ViewBuilder private var pickerOrHint: some View {
        if available.isEmpty {
            if pinned.isEmpty {
                Text(
                    "No fitted reader artifacts found under runs/. Readers "
                        + "are fitted from paired reader data (RepE reading "
                        + "probes) — fit one from the Concept Lab, or import "
                        + "a server evidence bundle that carries one; it "
                        + "will appear here to pin.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        } else {
            Menu("Pin fitted reader…") {
                ForEach(available) { record in
                    Button(menuLabel(record)) { pin(record) }
                }
            }
            .font(.caption)
            .help(
                "pins a fitted reader artifact (path + SHA-256) for its "
                    + "concept — re-pinning a concept replaces its entry. "
                    + "Readers are substrate-specific: the engine that "
                    + "freezes checks the pinned reader was fitted on its "
                    + "own substrate")
        }
    }

    private func menuLabel(_ record: VectorCatalog.ReaderArtifactRecord) -> String {
        var label = record.label
        if record.artifact.modelID != manifest.modelID {
            label += " — ⚠︎ fitted on \(record.artifact.modelID)"
        }
        return label
    }

    private func pin(_ record: VectorCatalog.ReaderArtifactRecord) {
        let path = workspaceRelativePath(record.url)
        write {
            try ExperimentStore.pinReader(
                path: path, experimentName: manifest.name)
        }
    }

    private func remove(_ concept: String) {
        write {
            try ExperimentStore.removeReader(
                concept: concept, experimentName: manifest.name)
        }
    }

    /// Store workspace-relative paths where the artifact lives inside the
    /// workspace (portable across the bundle path), absolute otherwise —
    /// the same rule the store's other pins follow.
    private func workspaceRelativePath(_ url: URL) -> String {
        let root = VectorCatalog.projectRoot.standardizedFileURL.path + "/"
        let path = url.standardizedFileURL.path
        return path.hasPrefix(root)
            ? String(path.dropFirst(root.count)) : path
    }

    /// One draft-edit write path: store write, panel refresh, plain-language
    /// failure line with the raw detail underneath.
    private func write(_ edit: () throws -> Void) {
        do {
            try edit()
            errorText = nil
            panel.refresh()
        } catch {
            errorText =
                "Couldn't update the pinned readers — the study must still "
                + "be a draft, and the artifact must be a reader fitted on "
                + "the study's model. Details: \(error)"
        }
    }
}

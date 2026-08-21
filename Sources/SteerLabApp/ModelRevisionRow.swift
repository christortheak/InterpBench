import ExperimentKit
import SwiftUI

/// Study Setup row for the draft's model-revision pin.
///
/// Audit 2026-08-01: the revision was settable at create time and read-only
/// forever after — changing it meant duplicating the study or pasting JSON,
/// for a field that is an ordinary draft pin until freeze makes it
/// evidence. Draft-edits go through `ExperimentStore.setModelRevision`;
/// empty clears back to the auto-pin path (resolved at freeze from the
/// local HF cache).
///
/// Pattern: `OrdinalScaleInstrumentControls` — the store's draft-edit gate,
/// never a parallel save path.
struct ModelRevisionControls: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @State private var revisionText: String = ""
    @State private var errorText: String?

    private var isDraft: Bool { manifest.status == .draft }

    private static let revisionHelp: String =
        "the model commit hash every run and extraction pins to. Empty = "
        + "auto-pin (freeze resolves the locally cached revision). Changing "
        + "it on a draft is safe by design: validate and battery evidence "
        + "are scope-matched by revision, so evidence for the old revision "
        + "reclassifies as stale instead of silently carrying over"

    var body: some View {
        Group {
            if isDraft {
                HStack(spacing: 8) {
                    TextField(
                        "model revision (optional commit hash; empty = "
                            + "auto-pin at freeze)",
                        text: $revisionText)
                        .onSubmit { commit() }
                    Button("Set") { commit() }
                        .disabled(revisionText == storedRevision)
                        .help(Self.revisionHelp)
                }
                .font(.caption)
            }
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        .onAppear { revisionText = storedRevision }
        .onChange(of: manifest.name) { revisionText = storedRevision }
    }

    private var storedRevision: String { manifest.modelRevision ?? "" }

    private func commit() {
        do {
            _ = try ExperimentStore.setModelRevision(
                revisionText, experimentName: manifest.name)
            errorText = nil
            panel.refresh()
        } catch {
            errorText =
                "Couldn't set the model revision — the study must still be "
                + "a draft (frozen studies are read-only). Details: \(error)"
        }
    }
}

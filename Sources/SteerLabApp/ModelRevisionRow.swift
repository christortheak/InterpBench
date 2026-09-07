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
                    // Short placeholder: the sentence lives in the tooltip,
                    // where it does not truncate at narrow widths.
                    TextField("commit hash (optional)", text: $revisionText)
                        .onSubmit { commit() }
                        .help(Self.revisionHelp)
                        .accessibilityLabel("Model revision")
                    Button("Set Revision") { commit() }
                        .disabled(revisionText == storedRevision)
                        .help(
                            revisionText == storedRevision
                                ? "the field already matches the pinned "
                                    + "revision — nothing to write"
                                : Self.revisionHelp)
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
        // The revision can move under this field from another surface (a pack
        // apply, an import, a re-read). Without this the field kept the old
        // text while "Set" compared against the new value, so pressing it
        // would have written the stale string back (audit 2026-09-06).
        .onChange(of: manifest.modelRevision) { revisionText = storedRevision }
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
                + "a draft (frozen studies are read-only). Details: "
                + Self.describe(error)
        }
    }

    /// The store's refusal as written; a Foundation error's `description` is
    /// an `Error Domain=…` dump, so it goes through `localizedDescription`.
    private static func describe(_ error: Error) -> String {
        if let experiment = error as? ExperimentError { return experiment.reason }
        return error.localizedDescription
    }
}

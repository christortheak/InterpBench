import ExperimentKit
import SwiftUI

/// Study Setup row for the manifest's fixed seed list.
///
/// Audit 2026-08-01: `seeds` was the last true orphan — the seed *policy*
/// had a picker while the list it indexes into could only ever be the
/// default `[20260610]` or a confirm-draft carry. Renders only under the
/// fixed-list policy (the derived policy never reads it), and the help
/// text carries the greedy rule: at temperature 0 both engines decode
/// greedily, so every seed gives the same response and a study lists one
/// (`StudySampling.greedySeedListProblem`; verify refuses more, public issue
/// #9). The parser is the one `set-sampling --seeds` uses.
///
/// Pattern: `OrdinalScaleInstrumentControls` — the store's draft-edit gate,
/// never a parallel save path.
struct SeedsListControls: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @State private var seedsText: String = ""
    @State private var errorText: String?
    /// "Set" greying out was the only sign the write landed; a study author
    /// changing seeds on purpose deserves to be told it happened.
    @State private var confirmationText: String?

    private var isDraft: Bool { manifest.status == .draft }

    private static let seedsHelp: String =
        "the fixed seed list a stochastic run indexes into on either engine "
        + "(sample k of a record uses seed k). Comma-separated whole numbers, "
        + "order preserved, duplicates refused. At temperature 0 decoding is "
        + "greedy and every seed gives the same response, so list one seed: "
        + "a study at temperature 0 with several is refused at verify, "
        + "because each seed would write an identical record per item"

    var body: some View {
        Group {
            HStack(spacing: 8) {
                TextField(
                    "seeds (comma-separated, e.g. 20260610, 20260611)",
                    text: $seedsText)
                    .onSubmit { commit() }
                    .help(Self.seedsHelp)
                Button("Set") { commit() }
                    .disabled(!isDraft || seedsText == storedSeedsText)
                    .help(setButtonHelp)
            }
            .font(.caption)
            .disabled(!isDraft)
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let confirmationText {
                Label(confirmationText, systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear { seedsText = storedSeedsText }
        .onChange(of: manifest.name) { seedsText = storedSeedsText }
        // The field used to seed on appear and on a NAME change only, so a
        // seed list written by another path on the same draft left this row
        // showing the old value. Adopt the new stored list whenever the field
        // is not carrying an unsaved edit.
        .onChange(of: storedSeedsText) { previous, current in
            if seedsText == previous { seedsText = current }
        }
        .onChange(of: seedsText) {
            errorText = nil
            confirmationText = nil
        }
    }

    private var storedSeedsText: String {
        manifest.seeds.map(String.init).joined(separator: ", ")
    }

    private var setButtonHelp: String {
        isDraft
            ? "write this list into the draft's manifest — " + Self.seedsHelp
            : "the study is frozen; its seed list is pinned data — duplicate "
                + "the study to change it"
    }

    private func commit() {
        let seeds: [UInt64]
        do {
            seeds = try ExperimentStore.parseSeedList(
                seedsText, experimentName: manifest.name)
        } catch {
            errorText = (error as? ExperimentError)?.reason
                ?? error.localizedDescription
            confirmationText = nil
            return
        }
        do {
            _ = try ExperimentStore.setSeeds(
                seeds, experimentName: manifest.name)
            errorText = nil
            confirmationText =
                "seed list set — \(seeds.count) seed"
                + (seeds.count == 1 ? "" : "s")
            panel.refresh()
        } catch {
            confirmationText = nil
            errorText =
                "Couldn't set the seed list — the study must still be a "
                + "draft, the list non-empty, and every seed distinct. "
                + "Details: \(error.localizedDescription)"
        }
    }
}

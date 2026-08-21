import ExperimentKit
import SwiftUI

/// Study Setup row for the manifest's fixed seed list.
///
/// Audit 2026-08-01: `seeds` was the last true orphan — the seed *policy*
/// had a picker while the list it indexes into could only ever be the
/// default `[20260610]` or a confirm-draft carry. Renders only under the
/// fixed-list policy (the derived policy never reads it), and the help
/// text carries the local-engine caveat: local measured runs are greedy
/// and stamp `seedInert`, so these seeds are causally meaningful only on
/// the server substrate.
///
/// Pattern: `OrdinalScaleInstrumentControls` — the store's draft-edit gate,
/// never a parallel save path.
struct SeedsListControls: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @State private var seedsText: String = ""
    @State private var errorText: String?

    private var isDraft: Bool { manifest.status == .draft }

    private static let seedsHelp: String =
        "the fixed seed list stochastic server runs index into (sample k of "
        + "a record uses seed k). Comma-separated whole numbers, order "
        + "preserved, duplicates refused. Local Mac runs are greedy-only "
        + "and stamp seedInert — these seeds carry no causal meaning there"

    var body: some View {
        Group {
            HStack(spacing: 8) {
                TextField(
                    "seeds (comma-separated, e.g. 20260610, 20260611)",
                    text: $seedsText)
                    .onSubmit { commit() }
                Button("Set") { commit() }
                    .disabled(!isDraft || seedsText == storedSeedsText)
                    .help(Self.seedsHelp)
            }
            .font(.caption)
            .disabled(!isDraft)
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        .onAppear { seedsText = storedSeedsText }
        .onChange(of: manifest.name) { seedsText = storedSeedsText }
    }

    private var storedSeedsText: String {
        manifest.seeds.map(String.init).joined(separator: ", ")
    }

    private func commit() {
        let parts = seedsText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        var seeds: [UInt64] = []
        for part in parts {
            guard let seed = UInt64(part) else {
                errorText = "'\(part)' is not a whole number seed"
                return
            }
            seeds.append(seed)
        }
        do {
            _ = try ExperimentStore.setSeeds(
                seeds, experimentName: manifest.name)
            errorText = nil
            panel.refresh()
        } catch {
            errorText =
                "Couldn't set the seed list — the study must still be a "
                + "draft, the list non-empty, and every seed distinct. "
                + "Details: \(error)"
        }
    }
}

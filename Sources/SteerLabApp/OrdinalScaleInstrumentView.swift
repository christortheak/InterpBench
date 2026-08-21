import ExperimentKit
import SwiftUI

/// Minimal Evaluation-section controls for the ordinal-scale outcome
/// instrument (`outcomeInstruments: ["ordinalScale"]` + the REQUIRED
/// `ordinalAggregation` declaration).
///
/// `ordinalScale` is an AUXILIARY instrument (the `repeReaderScore`
/// pattern): the Outcome Mode picker does not own it, so once declared it
/// already renders as its own Evaluation row (description + Remove) via
/// `panel.auxiliaryOutcomeInstruments`. This view adds the two things that
/// row cannot do: declare the instrument in the first place, and declare
/// the aggregation — the instrument-design choice the manifest must state
/// explicitly (verify refuses `ordinalScale` without it; nothing is
/// silently defaulted).
///
/// Wiring: one line in `ExperimentsPanelView`'s Evaluation section (after
/// the auxiliary-instrument rows):
///
///     OrdinalScaleInstrumentControls(manifest: manifest, panel: panel)
struct OrdinalScaleInstrumentControls: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @State private var errorText: String?

    /// Sentinel tag for "no aggregation declared yet" (never written back).
    private static let undeclared = ""

    private var declared: Bool {
        (manifest.outcomeInstruments ?? []).contains("ordinalScale")
    }

    var body: some View {
        Group {
            if declared {
                aggregationPicker
            } else if manifest.status == .draft {
                Button("Add ordinal-scale instrument (ordinalScale)") {
                    write {
                        try ExperimentStore.setOutcomeInstruments(
                            (manifest.outcomeInstruments ?? []) + ["ordinalScale"],
                            experimentName: manifest.name)
                    }
                }
                .font(.caption)
                .help(
                    "declares ordinalScale alongside the current mode — the "
                        + "deterministic ordinal readout over each item's ordered "
                        + "options (its ladder, for example [\"1\"…\"7\"]). You "
                        + "must also choose how the ladder distribution becomes "
                        + "one number (expected value or most likely step); "
                        + "verify refuses the instrument until that choice is "
                        + "declared")
            }
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
    }

    @ViewBuilder private var aggregationPicker: some View {
        Picker(
            "Ordinal aggregation",
            selection: Binding(
                get: { manifest.ordinalAggregation ?? Self.undeclared },
                set: { newValue in
                    write {
                        try ExperimentStore.setOrdinalAggregation(
                            newValue == Self.undeclared ? nil : newValue,
                            experimentName: manifest.name)
                    }
                })
        ) {
            if manifest.ordinalAggregation == nil {
                Text("not declared (required)").tag(Self.undeclared)
            }
            Text("Expected value (probability-weighted mean step)")
                .tag("expectedValue")
            Text("Most likely step (argmax)").tag("argmax")
        }
        .disabled(manifest.status != .draft)
        .help(
            "how the probability spread over the item's ordered options "
                + "becomes one ladder position: expected value multiplies each "
                + "step (1…K) by its probability and sums — sensitive to the "
                + "whole distribution; most likely step reports the single "
                + "highest-probability option's position. This is an "
                + "instrument-design choice, so it is written into the "
                + "manifest — the study will not verify until it is declared")
        if manifest.ordinalAggregation == nil {
            Label(
                "Choose an aggregation — verify refuses ordinalScale until "
                    + "the choice is declared in the manifest.",
                systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
        }
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
                "Couldn't update the ordinal-scale instrument — the study "
                + "must still be a draft (frozen studies are read-only). "
                + "Details: \(error)"
        }
    }
}

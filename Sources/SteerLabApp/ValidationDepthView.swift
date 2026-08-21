import ExperimentKit
import SwiftUI

/// The declared validation read depth(s) (`validationLayer[s]` /
/// `validationLayerFraction[s]`, D4), rendered with the Validate button so
/// the depth choice sits exactly where validation is launched.
///
/// Reworked after review (2026-08-01, P1): the first version derived the
/// picker's selection from the persisted manifest, which made it unusable
/// from its default state — choosing "Layer index" wrote nothing, so the
/// manifest stayed Default and the value field never appeared (and a
/// mode-guessing heuristic reinterpreted values across modes). The pending
/// mode is now its own `@State`, the value field follows the PENDING mode,
/// nothing is reinterpreted on a mode switch (the field clears instead),
/// and a fraction displays at full precision (Double's own description
/// round-trips; the old 4-decimal formatting could silently rewrite a more
/// precise declaration).
///
/// Depth LISTS (2026-08-01, the validate-at-the-sweep-layers policy): the
/// value field accepts comma-separated values — "0.5, 0.6, 0.7, 0.8" — and
/// ONE validate run measures every declared depth (activations are captured
/// once for all layers, so extra depths are near-free). A single value
/// stores the scalar field, so old manifests and old habits are unchanged;
/// several store the list field. Each report entry carries a per-depth
/// `layerResolution` block.
///
/// Pattern: `OrdinalScaleInstrumentControls` (the store's draft-edit gate,
/// never a parallel save path).
struct ValidationDepthControls: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @State private var mode: Mode = .defaultRule
    @State private var valueText: String = ""
    @State private var errorText: String?

    enum Mode: String, CaseIterable {
        case defaultRule
        case layerIndex
        case depthFraction
    }

    private var isDraft: Bool { manifest.status == .draft }

    /// The mode the MANIFEST currently declares (the pending `mode` state
    /// may differ while the researcher is still typing a value).
    private var storedMode: Mode {
        if manifest.validationLayer != nil || manifest.validationLayers != nil {
            return .layerIndex
        }
        if manifest.validationLayerFraction != nil
            || manifest.validationLayerFractions != nil
        {
            return .depthFraction
        }
        return .defaultRule
    }

    private var storedValueText: String {
        if let layer = manifest.validationLayer { return String(layer) }
        if let layers = manifest.validationLayers {
            return layers.map(String.init).joined(separator: ", ")
        }
        // Full precision: Double's description round-trips the value.
        if let fraction = manifest.validationLayerFraction {
            return "\(fraction)"
        }
        if let fractions = manifest.validationLayerFractions {
            return fractions.map { "\($0)" }.joined(separator: ", ")
        }
        return ""
    }

    private static let depthHelp: String =
        "which layer(s) convergent validity reads at — a measurement "
        + "decision, written into the manifest rather than inherited from "
        + "the steering conditions. A layer index names one depth on one "
        + "model; a fraction reads the same relative depth across model "
        + "sizes (resolved with the sweep's truncating rule). Comma-separate "
        + "several values to measure a depth profile in ONE validate run — "
        + "e.g. the sweep's layer band, so the reading certificate covers "
        + "every layer the sweep may promote. Each report entry records the "
        + "resolved layer and which rule chose it"

    var body: some View {
        Group {
            modePicker
            if mode != .defaultRule {
                valueRow
            }
            statusLine
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        .onAppear { syncFromManifest() }
        .onChange(of: manifest.name) { syncFromManifest() }
    }

    private func syncFromManifest() {
        mode = storedMode
        valueText = storedValueText
        errorText = nil
    }

    @ViewBuilder private var modePicker: some View {
        Picker("Validation read depth", selection: $mode) {
            Text("Default (steering-condition layer, else mid-network)")
                .tag(Mode.defaultRule)
            Text("Layer index(es)").tag(Mode.layerIndex)
            Text("Depth fraction(s) (0–1)").tag(Mode.depthFraction)
        }
        .disabled(!isDraft)
        .help(Self.depthHelp)
        .onChange(of: mode) { previous, selected in
            guard previous != selected else { return }
            errorText = nil
            switch selected {
            case .defaultRule:
                // Default IS the value — clear immediately.
                write {
                    try ExperimentStore.setValidationReadDepth(
                        experimentName: manifest.name)
                }
                valueText = ""
            case .layerIndex, .depthFraction:
                // Never reinterpret a value across modes (21 is not the
                // fraction 21): keep the stored value only when the stored
                // declaration is already in this mode, else start empty
                // and wait for Set.
                valueText = selected == storedMode ? storedValueText : ""
            }
        }
    }

    @ViewBuilder private var valueRow: some View {
        let placeholder: String =
            mode == .layerIndex
            ? "e.g. 21 — or a list: 31, 37, 43, 49"
            : "e.g. 0.65 — or a list: 0.5, 0.6, 0.7, 0.8"
        let trimmedEmpty: Bool =
            valueText.trimmingCharacters(in: .whitespaces).isEmpty
        HStack {
            TextField(placeholder, text: $valueText)
                .frame(maxWidth: 260)
                .onSubmit { commit() }
            Button("Set") { commit() }
                .disabled(!isDraft || trimmedEmpty)
        }
        .disabled(!isDraft)
    }

    /// What the MANIFEST says right now — so a pending, uncommitted mode
    /// switch is visibly not yet a declaration.
    @ViewBuilder private var statusLine: some View {
        Text(statusText)
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    private var statusText: String {
        if let layer = manifest.validationLayer {
            return "declared: layer \(layer)"
        }
        if let layers = manifest.validationLayers {
            return "declared: layers "
                + layers.map(String.init).joined(separator: ", ")
                + " — one validate run measures every depth"
        }
        if let fraction = manifest.validationLayerFraction {
            return "declared: fraction \(fraction)"
        }
        if let fractions = manifest.validationLayerFractions {
            return "declared: fractions "
                + fractions.map { "\($0)" }.joined(separator: ", ")
                + " — one validate run measures every depth"
        }
        return mode == .defaultRule
            ? "declared: default rule"
            : "not yet declared — enter a value and press Set"
    }

    /// The comma-separated values in the field, split and trimmed.
    private var enteredValues: [String] {
        valueText.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// Commit uses the PENDING mode explicitly. No shape-guessing: an
    /// integer typed under Depth fraction is the fraction 21 and refuses in
    /// the store (out of [0, 1]), not a silently different declaration. The
    /// store maps a one-element list onto the scalar field, so a single
    /// value produces exactly the manifests the scalar era did.
    private func commit() {
        let values = enteredValues
        guard !values.isEmpty else { return }
        switch mode {
        case .layerIndex:
            var layers: [Int] = []
            for value in values {
                guard let layer = Int(value) else {
                    errorText = "'\(value)' is not a whole layer index"
                    return
                }
                layers.append(layer)
            }
            write {
                try ExperimentStore.setValidationReadDepth(
                    layers: layers, experimentName: manifest.name)
            }
        case .depthFraction:
            var fractions: [Double] = []
            for value in values {
                guard let fraction = Double(value) else {
                    errorText = "'\(value)' is not a number"
                    return
                }
                fractions.append(fraction)
            }
            write {
                try ExperimentStore.setValidationReadDepth(
                    fractions: fractions, experimentName: manifest.name)
            }
        case .defaultRule:
            break
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
                "Couldn't set the validation read depth — the study must "
                + "still be a draft, and exactly one declaration shape "
                + "(index / fraction / a list of one kind) may be set. "
                + "Details: \(error)"
        }
    }
}

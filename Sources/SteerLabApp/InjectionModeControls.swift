import ExperimentKit
import SteeringKit
import SwiftUI

/// The Steer / Ablate picker and the controls each mode needs — shared by the
/// Playground, New Agent, and Studies so the three surfaces cannot drift into
/// describing the same intervention differently.
///
/// The two modes take genuinely different inputs, so the picker swaps the
/// controls rather than greying half of them out:
///
/// - **Steer** `h + α·v` — a layer and an α.
/// - **Ablate** `h − λ·(h·v̂)v̂` — a λ and no layer, because ablation covers
///   the whole network (a single-layer removal is usually rewritten by the
///   layers above it) and because λ needs no residual-norm denominator: α's
///   whole job is comparability, and ablation removes exactly what is
///   present, so it is already self-scaling.
///
/// Every mode-specific rule that could otherwise surprise someone is stated
/// on screen, not in a tooltip: what λ = 1 and λ = 2 mean, that layers are
/// not chosen, and that ablation applies across the whole prompt.
struct InjectionModeControls: View {
    @Binding var mode: InterventionPlan.Mode
    @Binding var layer: Int
    /// α when steering, λ when ablating.
    @Binding var strength: Double
    /// Highest selectable layer, when the caller knows the model's depth.
    var layerCount: Int?
    /// Rendered under the controls. Callers pass the vector's name so the
    /// sentence reads concretely ("removes fear …").
    var conceptLabel: String?
    /// Compact form for dense rows (the Playground's slot list).
    var isCompact = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("Mode", selection: $mode) {
                Text("Steer").tag(InterventionPlan.Mode.add)
                Text("Ablate").tag(InterventionPlan.Mode.ablate)
            }
            .pickerStyle(.segmented)
            .frame(width: 160)
            .onChange(of: mode) { _, new in
                // Carrying a steering α over as λ would be a silent surprise:
                // α is typically 1–3, and λ = 2 is already a REFLECTION, not
                // an ablation. Reset to each mode's sensible default instead.
                strength = new == .ablate
                    ? ChatService.SteerSlot.defaultAblationStrength : 2
            }
            .help(InjectionModeCopy.pickerHelp)

            switch mode {
            case .add:
                steeringControls
            case .ablate:
                ablationControls
            }

            if !isCompact {
                Text(InjectionModeCopy.explanation(
                    mode: mode, concept: conceptLabel, strength: strength))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    @ViewBuilder
    private var steeringControls: some View {
        HStack(spacing: 8) {
            if let layerCount, layerCount > 0 {
                Stepper(
                    "Layer \(layer)", value: $layer,
                    in: 0...(layerCount - 1))
                    .frame(width: 140)
            } else {
                LabeledContent("Layer") {
                    TextField(
                        "Layer", value: $layer,
                        format: .number.grouping(.never))
                        .frame(width: 72)
                }
            }
            LabeledContent("Alpha") {
                TextField("Alpha", value: $strength, format: .number)
                    .frame(width: 92)
            }
        }
        .help(InjectionModeCopy.alphaHelp)
    }

    @ViewBuilder
    private var ablationControls: some View {
        HStack(spacing: 8) {
            LabeledContent("Strength λ") {
                TextField("λ", value: $strength, format: .number)
                    .frame(width: 92)
            }
            Text(InjectionModeCopy.lambdaLabel(strength))
                .font(.caption)
                .foregroundStyle(
                    InjectionModeCopy.lambdaIsUnusual(strength)
                        ? .orange : .secondary)
        }
        .help(InjectionModeCopy.lambdaHelp)
        if !isCompact {
            Label(
                "All layers — ablation is not aimed at one layer",
                systemImage: "square.stack.3d.up")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

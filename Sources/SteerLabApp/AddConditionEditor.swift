import ExperimentKit
import SteeringKit
import SwiftUI

/// The Injection Conditions & Controls add-condition editor, restructured
/// after the 2026-07-20 researcher round: the old single HStack scrunched
/// its TextFields until labels hyphenated ("lay-er") and everything
/// truncated. Now one labeled Form row per input — the pipeline gate-row
/// pattern: name leading, compact field trailing at a fixed sane width,
/// plain-language caption underneath. Lives in its own file because
/// `ExperimentsPanelView` sits at the type-checker's limits.
struct AddConditionEditor: View {
    @Bindable var panel: ExperimentPanel

    var body: some View {
        inputRow(label: "Condition name") {
            TextField("optional", text: $panel.conditionName)
                .textFieldStyle(.roundedBorder)
                .frame(width: 200)
        }
        inputRow(
            label: "Concept",
            caption: "the attached concept whose direction this condition "
                + "injects"
        ) {
            Picker("", selection: $panel.conditionConcept) {
                ForEach(panel.conditionConceptOptions, id: \.self) { concept in
                    Text(concept).tag(concept)
                }
            }
            .labelsHidden()
            .frame(width: 200)
        }
        inputRow(
            label: "What it does",
            caption: panel.conditionMode == .ablate
                ? "Ablate REMOVES whatever of the concept the model is "
                    + "representing, and adds nothing"
                : "Steer ADDS the concept whether or not the model was "
                    + "already representing it"
        ) {
            Picker("", selection: $panel.conditionMode) {
                Text("Steer").tag(InterventionPlan.Mode.add)
                Text("Ablate").tag(InterventionPlan.Mode.ablate)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 160)
            .help(InjectionModeCopy.pickerHelp)
        }
        // The two modes take different inputs, so the rows are SWAPPED rather
        // than disabled: a greyed-out Layer box next to an ablation would
        // suggest the layer means something here.
        if panel.conditionMode == .add {
            inputRow(
                label: "Layer",
                caption: "which transformer layer the steering applies at — the "
                    + "middle third of the network is the usual sweet spot"
            ) {
                TextField("e.g. 14", text: $panel.conditionLayerText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 72)
                    .multilineTextAlignment(.trailing)
            }
            inputRow(
                label: "Strength (α)",
                caption: "how hard to push along the concept's direction — "
                    + "negative pushes the opposite way (a direction control, "
                    + "not an error)"
            ) {
                TextField("e.g. 0.5", text: $panel.conditionAlphaText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 90)
                    .multilineTextAlignment(.trailing)
            }
            inputRow(
                label: "Relative strength",
                caption: "on: α is measured relative to the model's own activity "
                    + "at that layer (residual-norm units, comparable across "
                    + "concepts); off: raw activation units"
            ) {
                HStack(spacing: 6) {
                    Toggle("", isOn: $panel.conditionAlphaInNormUnits)
                        .toggleStyle(.checkbox)
                        .labelsHidden()
                    InfoButton(text: StudyInfo.strengthLayerNorm)
                }
            }
        } else {
            inputRow(
                label: "Strength (λ)",
                caption: InjectionModeCopy.lambdaLabel(
                    Double(panel.conditionAlphaText) ?? 1)
            ) {
                TextField("1", text: $panel.conditionAlphaText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 90)
                    .multilineTextAlignment(.trailing)
                    .help(InjectionModeCopy.lambdaHelp)
            }
            inputRow(
                label: "Layers",
                caption: "ablation is not aimed at one layer — a removal at a "
                    + "single layer is usually rewritten by the layers above "
                    + "it, so every layer is covered. It also applies to every "
                    + "token, including the whole prompt"
            ) {
                Text("all")
                    .foregroundStyle(.secondary)
            }
        }
        HStack(spacing: 8) {
            Button("Add Condition") { panel.addVectorCondition() }
                .help(
                    panel.conditionMode == .ablate
                        ? "adds a single-slot ablation condition from the "
                            + "fields above. For a dose-response, add further "
                            + "conditions at other λ (0.5 partial, 1 full, "
                            + "2 reflection)"
                        : "adds a single-slot vector condition from the fields "
                            + "above — a negative α is legal (direction control)")
        }
        // Editing any input the refusal named clears it — a stale refusal
        // sitting under a field the researcher has already corrected is its
        // own paper cut.
        .onChange(of: panel.conditionConcept) { panel.clearFormError(.addCondition) }
        .onChange(of: panel.conditionLayerText) { panel.clearFormError(.addCondition) }
        .onChange(of: panel.conditionAlphaText) { panel.clearFormError(.addCondition) }
        // Finding 11a: a refusal here used to appear ONLY in the panel-top
        // notice area, several hundred points above these fields — an α = 0
        // refusal read as "the button does nothing" (observed 2026-07-26).
        // The notice feed still records it; this renders it where the
        // researcher is looking.
        if let refusal = panel.formErrors[.addCondition] {
            Label(refusal, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        HStack(spacing: 8) {
            Button("Add Baseline") { panel.addBaselineCondition() }
                .help(
                    "explicit no-steer baseline condition (added "
                        + "automatically at run time when absent)")
            Button("Scaffold Control Matrix") { panel.scaffoldControlMatrix() }
                .help(
                    "for EVERY treatment condition, adds its missing "
                        + "counterparts — the −α direction-flip control and "
                        + "the random-direction control at the same strength "
                        + "— plus an explicit baseline. Idempotent, and it "
                        + "names what still needs authoring (ⓘ for details)")
            InfoButton(text: StudyInfo.controls)
        }
        ForEach(panel.lastControlMatrixNotes, id: \.self) { note in
            Label(note, systemImage: "hand.point.right")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// One input as a labeled Form row (the pipeline gate-row pattern):
    /// full-width label that never wraps mid-word, compact control
    /// trailing, caption underneath.
    @ViewBuilder
    private func inputRow(
        label: String, caption: String? = nil,
        @ViewBuilder content: () -> some View
    ) -> some View {
        LabeledContent(label) { content() }
            .font(.caption)
        if let caption {
            Text(caption)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

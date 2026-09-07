import ExperimentKit
import SteeringKit
import SwiftUI

/// Native steering conditions and controls, written through existing admitted commands.
struct StudyInjectionConditionsSection: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel

    /// Conditions/controls authoring in Studies: list + remove, single-slot
    /// vector conditions (negative α legal), one-click sign control and
    /// matched-norm random control per condition, explicit baseline, and the
    /// Step-5 control-matrix scaffold. Draft-only; every write goes through
    /// `ExperimentStore` helpers via the panel.
    @ViewBuilder
    var body: some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
        Section {
            if manifest.conditions.isEmpty {
                Text(
                    "No steering conditions yet. Baseline is implied at run "
                        + "time; a defensible matrix adds treatments plus "
                        + "direction (−α) and matched-norm random controls."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            ForEach(manifest.conditions, id: \.name) { condition in
                conditionRow(condition, manifest: manifest, panel: panel)
            }

            if manifest.status == .draft {
                if panel.conditionConceptOptions.isEmpty {
                    Text(
                        "attach a concept first — conditions reference pinned "
                            + "concepts (Attach Concept… in the Concepts section "
                            + "above, Agents → New Agent, or capture from "
                            + "Playground)"
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                } else {
                    // One labeled row per input (2026-07-20 researcher
                    // round: the old single HStack scrunched and
                    // hyphenated its labels) — see AddConditionEditor.
                    AddConditionEditor(panel: panel)
                }
            }
        } header: {
            InfoSectionHeader(
                title: "Injection Conditions & Controls",
                text: StudyInfo.injectionConditions)
        }
    }

    @ViewBuilder
    private func conditionRow(
        _ condition: ExperimentManifest.Condition,
        manifest: ExperimentManifest,
        panel: ExperimentPanel
    ) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(condition.name)
                        .font(.callout)
                    if condition.controlType == "randomMatchedNorm" {
                        Text("random-direction control")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(.orange.opacity(0.15), in: Capsule())
                            .help(
                                "matched-norm random: same layers and strength, "
                                    + "deterministic random direction — separates "
                                    + "the concept's direction from mere "
                                    + "perturbation energy")
                    } else if condition.slots.contains(where: { $0.alpha < 0 }) {
                        Text("direction control")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(.blue.opacity(0.12), in: Capsule())
                            .help(
                                "every strength is negative: the SAME direction "
                                    + "pushed the other way — separates the "
                                    + "concept's sign from any effect of "
                                    + "steering at all")
                    }
                }
                Text(conditionSummary(condition))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if manifest.status == .draft {
                if !condition.slots.isEmpty, condition.controlType == nil {
                    Button("Add sign control") { panel.addSignControl(for: condition.name) }
                        .controlSize(.small)
                        .help("adds '\(condition.name)-neg' with every α negated")
                    Button("Add random control") {
                        panel.addMatchedNormRandomControl(for: condition.name)
                    }
                    .controlSize(.small)
                    .help(
                        "adds '\(condition.name)-random' — a random-direction "
                            + "control at the same strength ('matched-norm "
                            + "random', controlType randomMatchedNorm): same "
                            + "layers/α, deterministic random direction of "
                            + "matched norm")
                }
                Button {
                    panel.removeCondition(condition.name)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .help("remove '\(condition.name)' from the draft")
                .accessibilityLabel("Remove condition \(condition.name)")
            }
        }
    }

    private func conditionSummary(_ condition: ExperimentManifest.Condition) -> String {
        guard !condition.slots.isEmpty else {
            return "no steering · baseline"
        }
        return condition.slots.map {
            "\($0.concept) L\($0.layer) α\($0.alpha)"
        }.joined(separator: " + ")
            + (condition.alphaInNormUnits ? " (norm units)" : "")
            + " · band \(condition.bandWidth)"
            + (condition.neutralPCBasisLabel.map { " · neutral-removed: \($0)" } ?? "")
    }
}

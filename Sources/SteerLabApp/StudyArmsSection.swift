import ExperimentKit
import SteeringKit
import SwiftUI

/// Comparison arms, confirmation policy and supplied agent evidence. Performs no evidence scans.
struct StudyArmsSection: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    let robustnessEvidence: [AgentEvidence.RobustnessEvidence]
    let currentSubstrate: String
    let availableVariants: [ModelVariantRecord]

    /// ONE Conditions section, content by study type: the ARMS of the
    /// study. Agents for a comparison; the scenario for multi-agent; the
    /// perturbation policy (and the conditions it expands into) for a
    /// confirmation. Concept studies see their arms here too — the
    /// concept-derivation machinery follows in its own sections.
    @ViewBuilder
    var body: some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
        let isDraft = manifest.status == .draft
        Section {
            if panel.studyKind == .multiAgent {
                // The picker itself now lives in Study Setup, beside Save —
                // pinning happens on save, and having the two in different
                // sections meant selecting a panel here looked like it had
                // taken effect when nothing had been written yet.
                Text(
                    panel.selectedMultiAgentScenarioID == nil
                        ? "No scenario selected. Choose one in Study Setup, cast "
                            + "its seats below, then save to pin it."
                        : "Scenario selected in Study Setup; its seats are cast in "
                            + "Seats below. Arms:"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                Text(
                    panel.multiAgentIncludeBaseline
                        ? "Two arms: the configured panel, and a baseline of the "
                            + "same panel with every intervention stripped."
                        : "One arm only: the configured panel. No baseline to "
                            + "compare against."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            } else {
                if isDraft {
                    HStack {
                        Picker("Add agent", selection: $draft.selectedVariantToAddID) {
                            Text("select…").tag(String?.none)
                            ForEach(panel.availableVariantsForStudy) { variant in
                                Text(variant.artifact.name).tag(String?.some(variant.id))
                            }
                        }
                        Button("Add") { panel.addVariantCondition() }
                            .disabled(panel.selectedVariantToAddID == nil)
                    }
                    .help("add a saved agent that uses the selected baseline model")
                    if !manifest.concepts.isEmpty {
                        Menu("Add sweep-created agent…") {
                            ForEach(manifest.concepts, id: \.name) { concept in
                                Button(concept.name) {
                                    panel.addForwardReferencedCondition(
                                        concept: concept.name)
                                }
                            }
                        }
                        .help(
                            "an agent that does not exist YET: this study's "
                                + "own sweep creates it at run time — the "
                                + "sweep selects the concept's best "
                                + "layer×strength cell, promote mints the "
                                + "agent, and the chain runs it as this arm, "
                                + "pinning the resolution as run evidence "
                                + "(forward-resolutions.json). Freezable "
                                + "before the agent exists")
                    }
                }
                // Confirmation authoring is the CONCEPT study's confirm
                // phase (2026-07-19 fold-in): the policy editor appears
                // when the funnel phase says confirm — or a policy is
                // already attached.
                if isDraft, panel.studyFocus == .conceptStudy,
                    panel.phaseField == "confirm"
                        || manifest.perturbationPolicy != nil
                {
                    confirmationControls(panel: panel)
                }
                if let policy = manifest.perturbationPolicy {
                    perturbationPolicySummary(policy, panel: panel)
                }
                if panel.studyFocus == .agentComparison,
                    manifest.concepts.isEmpty
                {
                    // Type parity note: comparisons consume EXISTING
                    // agents; deriving new ones (sweep-created agents,
                    // forward references) is Concept-study machinery.
                    Text(
                        "Comparisons run agents that already exist (built "
                            + "in Agents, or by a Concept study's sweep). To "
                            + "DERIVE new agents from concept data, use a "
                            + "Concept study."
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
                if manifest.variantConditions.isEmpty {
                    Text(
                        "Baseline only. Add agents to compare conditions "
                            + "(adding saves immediately)."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                } else {
                    ForEach(manifest.variantConditions, id: \.name) { variant in
                        attachedAgentRow(variant, isDraft: isDraft, panel: panel)
                    }
                }
            }
        } header: {
            InfoSectionHeader(
                title: "Conditions", text: StudyInfo.conditionsArms)
        }
    }

    /// Declared perturbation-policy inputs for a confirmation study —
    /// agent picker (promoted first), α deltas, control toggle. The
    /// expansion and every refusal live in `ConfirmationStudy.attach`.
    private func confirmationControls(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Picker("Confirm agent", selection: $draft.confirmAgentID) {
                    Text("select…").tag(ModelVariantRecord.ID?.none)
                    ForEach(panel.confirmableAgents) { record in
                        Text(confirmAgentLabel(record))
                            .tag(ModelVariantRecord.ID?.some(record.id))
                    }
                }
                TextField("strength deltas (α)", text: $draft.confirmDeltasText)
                    .frame(width: 120)
                    .help(
                        "comma-separated positive strength offsets around the "
                            + "anchor α, e.g. 0.2, 0.5")
                Toggle(
                    "random-direction control",
                    isOn: $draft.confirmIncludeControl
                )
                .help(
                    "adds a matched-norm random control at the anchor "
                        + "strength — same norm, deterministic random "
                        + "direction")
                Button("Attach Policy") { panel.attachPerturbations() }
                    .disabled(panel.confirmAgentID == nil)
            }
            // Non-blocking: confirmation of a hand-created agent stays legal,
            // but the evidence path runs through sweep-promoted agents.
            if let record = panel.confirmableAgents.first(where: {
                $0.id == panel.confirmAgentID
            }), record.artifact.promotion == nil {
                Text(
                    "hand-created agent: confirmation of an undeclared "
                        + "selection is exploratory, not evidence-grade — "
                        + "promote from a sweep for the evidence path"
                )
                .font(.caption2)
                .foregroundStyle(.orange)
            }
            Text(
                "declares the perturbation policy — anchor, α ± δ, control — "
                    + "and expands it into ordinary hashed conditions in this "
                    + "draft (visible below; the firewall pins them like any "
                    + "other condition)"
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }

    private func confirmAgentLabel(_ record: ModelVariantRecord) -> String {
        record.artifact.promotion != nil
            ? "\(record.artifact.name) · sweep-promoted"
            : "\(record.artifact.name) · hand-created"
    }

    /// The attached policy plus the conditions it generated — the agent
    /// vocabulary never hides the condition machinery underneath.
    private func perturbationPolicySummary(
        _ policy: ExperimentManifest.PerturbationPolicy, panel: ExperimentPanel
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Label(policySummaryLine(policy), systemImage: "scope")
                .font(.caption)
            ForEach(panel.generatedConfirmationConditions, id: \.name) { condition in
                Text(generatedConditionLine(condition))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func policySummaryLine(
        _ policy: ExperimentManifest.PerturbationPolicy
    ) -> String {
        let deltas = policy.alphaDeltas.map { "\($0)" }.joined(separator: ", ")
        let origin = policy.sourceAgent.promoted ? "sweep-promoted" : "hand-created"
        return "confirmation policy: '\(policy.sourceAgent.name)' (\(origin)) — "
            + "anchor L\(policy.cell.layer) α\(policy.cell.alpha), δ [\(deltas)]"
            + (policy.includeMatchedNormControl ? ", matched-norm control" : "")
    }

    private func generatedConditionLine(
        _ condition: ExperimentManifest.Condition
    ) -> String {
        guard let slot = condition.slots.first else { return condition.name }
        let control = condition.controlType == "randomMatchedNorm" ? " (control)" : ""
        return "\(condition.name) · L\(slot.layer) α\(slot.alpha)\(control)"
    }

    private func variantSummary(_ artifact: ModelVariantArtifact) -> String {
        "\(artifact.adapters.count) adapter\(artifact.adapters.count == 1 ? "" : "s")"
            + " · \(artifact.injections.count) injection\(artifact.injections.count == 1 ? "" : "s")"
            + " · \(artifact.promptMode)"
            + (artifact.systemPrompt == nil ? "" : " · system prompt")
    }

    /// One attached agent condition: name, composition summary, promotion
    /// provenance when present, and the non-blocking evidence notes
    /// (`AgentEvidence` — honesty chips, never a gate on attach/freeze/run).
    private func attachedAgentRow(
        _ variant: ExperimentManifest.VariantCondition,
        isDraft: Bool,
        panel: ExperimentPanel
    ) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(variant.name)
                    .font(.callout)
                if let forward = variant.fromPromotion {
                    // Stage-4 declaration: no artifact exists yet — say
                    // what the arm MEANS instead of summarizing nothing.
                    Text(
                        "forward-referenced: the agent this study's sweep "
                            + "promotes for '\(forward.concept)' — resolved and "
                            + "pinned at run time on the server"
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                } else {
                    Text(variantSummary(variant.artifact))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if let promotion = variant.artifact.promotion {
                        Text(AgentEvidence.provenanceLine(for: promotion))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    ForEach(conditionEvidenceNotes(variant)) { note in
                        evidenceNoteCaption(note)
                    }
                }
            }
            Spacer()
            if isDraft {
                Button {
                    panel.removeVariantCondition(variant.name)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func evidenceNoteCaption(_ note: AgentEvidence.Note) -> some View {
        Text(note.displayLine)
            .font(.caption2)
            .foregroundStyle(note.severity == .caution ? Color.orange : Color.secondary)
            .textSelection(.enabled)
    }

    /// Evidence notes for one attached condition, computed against the
    /// already-scanned robustness reports plus a library-resolution check.
    private func conditionEvidenceNotes(
        _ variant: ExperimentManifest.VariantCondition
    ) -> [AgentEvidence.Note] {
        let latest = AgentEvidence.latestRobustness(
            in: robustnessEvidence,
            variantName: variant.artifact.name,
            artifactHash: variant.artifactHash)
        var notes = AgentEvidence.notes(
            for: variant.artifact,
            currentSubstrate: currentSubstrate,
            latestRobustness: latest?.report)
        if !libraryResolves(variant) {
            notes.append(AgentEvidence.artifactNotFoundNote)
        }
        return notes
    }

    /// Whether the condition's variant reference still resolves in this
    /// workspace's agent library (by recorded relative path, then by name).
    private func libraryResolves(
        _ variant: ExperimentManifest.VariantCondition
    ) -> Bool {
        availableVariants.contains { record in
            ModelVariantStore.relativePath(for: record) == variant.artifactPath
                || record.artifact.name == variant.artifact.name
        }
    }
}

import ExperimentKit
import SteeringKit
import SwiftUI

/// Sampling declarations shared by model-output and multi-agent studies.
struct StudySamplingControls: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel

    /// Sampling policy lives WITH the other generation settings — samples
    /// per item and seed policy change how the run generates, exactly like
    /// temperature (they were never inert "science notes").
    @ViewBuilder
    var body: some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
        let isDraft = manifest.status == .draft
        let isPanel = panel.draft.studyKind == .multiAgent
        LabeledContent(isPanel ? "Play-throughs (transcripts)" : "Samples per item") {
            HStack(spacing: 6) {
                TextField(
                    "", value: $draft.samplesPerItemField,
                    format: .number.grouping(.never)
                )
                .frame(width: 56)
                .multilineTextAlignment(.leading)
                .disabled(!isDraft)
                InfoButton(text: StudyInfo.sampling)
            }
        }
        .help(
            isPanel
                ? "independent play-throughs of the panel per condition. This is the "
                    + "unit the statistics treat as one observation, and the unit a "
                    + "sharded submission splits across GPUs. Needs a temperature "
                    + "above 0 — at 0 every play-through is identical."
                : "stochastic samples per (condition, prompt); 1 = deterministic "
                    + "greedy")
        // Two real choices (2026-07-21, issue 1): absent and an explicit
        // 'manifestSeeds' behave identically on both engines (each
        // resolves absent to the fixed list), so the picker offers TWO
        // options. The explicit legacy tag renders only when a manifest
        // already declares it — silently normalizing it on load would
        // make an unchanged Save rewrite manifest bytes (a hash change
        // that trips the epoch guard on existing runs), exactly the kind
        // of quiet drift the firewall exists to prevent.
        HStack(spacing: 6) {
            Picker("Seed policy", selection: $draft.seedPolicyField) {
                Text("Fixed seed list (default)").tag("")
                if panel.draft.seedPolicyField == "manifestSeeds" {
                    Text("Fixed seed list (declared — same behavior)")
                        .tag("manifestSeeds")
                }
                Text("Derived per record — recommended for sampled runs")
                    .tag("derivedSHA256")
            }
            .disabled(!isDraft)
            .help(
                "how stochastic server runs seed each generated record — "
                    + "the local Mac engine is always greedy and ignores "
                    + "this. The ⓘ explains both policies")
            InfoButton(text: StudyInfo.seedPolicy)
        }
        // The list the fixed-list policy indexes into (audit 2026-08-01:
        // the policy had a picker; the list had no editor). Hidden under
        // the derived policy, which never reads it.
        if panel.draft.seedPolicyField != "derivedSHA256" {
            SeedsListControls(manifest: manifest, panel: panel)
        }
        // Gentle advisory (never a blocker): a stochastic design with the
        // fixed list loses per-record reproducibility; samples > 1
        // additionally requires the derived policy at verify.
        if panel.draft.runTemperature > 0 || panel.draft.samplesPerItemField > 1,
            panel.draft.seedPolicyField != "derivedSHA256"
        {
            Label(
                "this design is stochastic (temperature > 0 or several "
                    + "samples per item) — 'Derived per record' gives every "
                    + "(condition, prompt, sample) its own reproducible "
                    + "seed; the fixed list suits single-sample runs",
                systemImage: "info.circle"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        // The server-only-stochastic rule, surfaced exactly like the
        // temperature rule: local target + samplesPerItem > 1 explains
        // itself inline instead of failing later.
        if panel.draft.samplesPerItemField > 1, !panel.isServerWorkspace {
            Label(
                "samplesPerItem > 1 is a stochastic design — it runs on the "
                    + "Python server, which seeds PyTorch per record; the "
                    + "local MLX generator has no per-run sampling seed, so "
                    + "local runs stay greedy (temperature 0, 1 sample)",
                systemImage: "die.face.5"
            )
            .font(.caption)
            .foregroundStyle(.orange)
        }
        // Study-owned sampling (2026-07-21): with saved agents in the
        // design, say explicitly that these knobs govern every condition.
        if !manifest.variantConditions.isEmpty {
            Label(
                "the study's sampling policy governs the baseline AND every "
                    + "saved agent — an agent's Playground temperature is "
                    + "not used in measured runs",
                systemImage: "info.circle"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

}

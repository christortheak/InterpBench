import ExperimentKit
import SteeringKit
import SwiftUI

/// Study question, model and generation settings. Saves use the admitted protocol command.
struct StudySetupSection: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    let substrateLabel: String

    var body: some View {
        @Bindable var draft = panel.draft
        Section("Study Setup") {
            TextField(
                "question or purpose",
                text: $draft.protocolDescription,
                axis: .vertical
            )
            .lineLimit(1...3)
            .disabled(manifest.status != .draft)
            .help("what this study asks; not tied to any domain")

            TextField(
                "task: what the model or agents will do",
                text: $draft.taskDescription,
                axis: .vertical
            )
            .lineLimit(2...5)
            .disabled(manifest.status != .draft)
            .help(
                "examples: write an opinion, play a game, answer advice "
                    + "prompts, allocate resources, classify scenarios")

            TextField(
                "outcome measures",
                text: $draft.outcomeMeasures,
                axis: .vertical
            )
            .lineLimit(2...5)
            .disabled(manifest.status != .draft)
            .help(
                "what you will measure: flips, scores, cooperation rate, "
                    + "style markers, capability battery, degeneration")

            if panel.studyKind == .modelOutput {
                // Workspace-scoped model choice (same strict rule as
                // the chat's WorkspaceModelPicker): a server target
                // offers ONLY that server's installed models; a
                // current selection outside the inventory stays
                // rendered — "(not installed)" — but is never
                // pickable anew. The chosen server id flows into the
                // manifest exactly as local ids do.
                Picker(
                    panel.studyKind == .multiAgent
                        ? "Default model for seats"
                        : "Baseline model",
                    selection: $draft.studyBaseModelID
                ) {
                    if panel.studyBaseModelID.isEmpty {
                        Text("select model…").tag("")
                    }
                    ForEach(panel.modelOptions, id: \.self) { model in
                        Text(model).tag(model)
                    }
                    if WorkspaceScoping.selectionOutsideInventory(
                        panel.studyBaseModelID, inventory: panel.modelOptions)
                    {
                        Text(
                            panel.isServerWorkspace
                                ? "\(panel.studyBaseModelID) (not installed)"
                                : panel.studyBaseModelID
                        )
                        .tag(panel.studyBaseModelID)
                        .selectionDisabled()
                    }
                }
                .disabled(manifest.status != .draft)
                .help(
                    panel.studyKind == .multiAgent
                        ? "used only by panel seats that name no base model of "
                            + "their own. Seats may each carry a different "
                            + "model; every turn records the one it ran on, and "
                            + "this value is not a claim about the run."
                        : panel.isServerWorkspace
                            ? "the unmodified baseline model and required base for "
                                + "added agents — models installed on "
                                + "\(substrateLabel) (the active "
                                + "compute workspace)"
                            : "the unmodified baseline model and required base for added agents")
                if panel.isServerWorkspace, panel.modelOptions.isEmpty {
                    Text(
                        "no models installed on \(substrateLabel) — "
                            + "use Install model… (Compute menu) to prefetch one"
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
                // Draft-editable revision pin (audit 2026-08-01:
                // create-time only before — changing it required
                // duplicate-or-paste-JSON).
                ModelRevisionControls(manifest: manifest, panel: panel)
                studyDtypePicker(
                    manifest: manifest,
                    selection: $draft.studyDtypeField)

                HStack(spacing: 6) {
                    Picker("Baseline prompt mode", selection: $draft.promptMode) {
                        ForEach(ExperimentManifest.PromptMode.allCases, id: \.self) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .disabled(manifest.status != .draft)
                    .help(
                        "baseline chat assistant uses the model chat template with "
                            + "user/assistant roles; raw completion sends literal text. "
                            + "Saved agents use their own prompt mode")
                    InfoButton(text: StudyInfo.promptMode)
                }

                // Gemma has NO system role — the same text is
                // prepended to the first user turn instead, so the
                // label says what actually happens for the chosen
                // model family.
                TextField(
                    manifest.modelID.lowercased().contains("gemma")
                        ? "Baseline instruction (Gemma: prepended to "
                            + "the first user turn — no system role)"
                        : "Baseline system prompt",
                    text: $draft.systemPrompt, axis: .vertical
                )
                .lineLimit(1...4)
                .disabled(manifest.status != .draft)
                .help(
                    "optional standing instruction for the baseline "
                        + "condition; saved agents use their own. "
                        + "Qwen sends it as a true system message; "
                        + "Gemma's chat template has no system role, "
                        + "so the SAME text is prepended to the first "
                        + "user turn (this affects prompt-set "
                        + "hashing and prompt design)")

                Picker("Reasoning effort", selection: $draft.reasoningEffort) {
                    ForEach(ReasoningEffort.vocabulary, id: \.self) { effort in
                        Text(effort).tag(effort)
                    }
                }
                .disabled(
                    manifest.status != .draft
                        || !PromptRendering.hasThinkingMode(
                            manifest.modelID,
                            capabilities: ExperimentStore.modelCapabilities(for: manifest))
                )
                .help(
                    "The reasoning effort the chat template is rendered with "
                        + "(off = no thinking block; on = thinking at the template's "
                        + "default effort). A non-off effort needs a reasoning token "
                        + "budget, and a LEVEL only where the model's capability "
                        + "record — probed from its chat template, shown in the "
                        + "Compute section — says the template accepts it.")
                if panel.qwenThinkingEnabled {
                    TextField(
                        "Reasoning max tokens",
                        value: $draft.reasoningMaxTokens,
                        format: .number
                    )
                    .disabled(manifest.status != .draft)
                    .help(
                        "The reasoning block's own token cap (up to </think>); "
                            + "Max tokens is then the answer budget. Required.")
                }
            }

            TemperatureRow(value: $draft.runTemperature)
                .disabled(manifest.status != .draft)

            LabeledContent("Max tokens") {
                TextField(
                    "", value: $draft.runMaxTokens,
                    format: .number.grouping(.never)
                )
                .frame(width: 72)
                .multilineTextAlignment(.leading)
            }
            .disabled(manifest.status != .draft)
            .help("study-wide per-response token cap for baseline and agents")

            if panel.studyKind == .multiAgent {
                Picker(
                    "Scenario",
                    selection: $draft.selectedMultiAgentScenarioID
                ) {
                    Text("select…").tag(String?.none)
                    ForEach(panel.multiAgentScenarioOptions) { scenario in
                        Text(Self.scenarioMenuLabel(scenario))
                            .tag(String?.some(scenario.id))
                    }
                }
                .disabled(manifest.status != .draft)
                .help(
                    "the saved scenario this study runs: its roles, "
                        + "turns and materials. Selecting it does NOT "
                        + "pin it — Save study setup compiles the seat "
                        + "casting below and writes the pin, which is "
                        + "what Data & Prompts checks.")
                if panel.selectedMultiAgentScenarioID != nil,
                    manifest.multiAgentScenarioPath == nil
                {
                    Text(
                        "selected but not pinned — Save study setup to "
                            + "pin it"
                    )
                    .font(.caption2)
                    .foregroundStyle(.orange)
                }
                Toggle(
                    "Include stripped baseline arm",
                    isOn: $draft.multiAgentIncludeBaseline
                )
                .disabled(manifest.status != .draft)
                .help(
                    "runs the SAME panel a second time with every "
                        + "intervention removed — adapters and steering "
                        + "vectors stripped, base models unchanged. This "
                        + "is the control the measurement subtracts.")
                if !panel.multiAgentIncludeBaseline {
                    // Not a style preference: without the control arm
                    // there is nothing to difference against, so the
                    // whole analysis layer goes quiet.
                    Text(
                        "without the baseline arm this study produces no "
                            + "effect sizes and no panel-effect decomposition "
                            + "— there is nothing to compare the panel against"
                    )
                    .font(.caption2)
                    .foregroundStyle(.orange)
                }
            }

            // Both study kinds need this. For a panel, "samples per
            // item" IS the number of independent play-throughs
            // (transcripts) — the unit the statistics treat as one
            // observation, and the unit sharding splits over. It was
            // gated to model-output studies, so panel replicates were
            // implemented on both engines and unreachable from here.
            StudySamplingControls(manifest: manifest, panel: panel)

            // The funnel phase has two REAL effects (it is not a
            // note): analyze picks its multiple-comparison
            // correction from it, and phase=confirm activates the
            // held-out-pool verification rule. Never sent to any
            // model. CONCEPT studies only (2026-07-21, issue 2):
            // screen/confirm is concept-funnel vocabulary — the
            // picker is hidden, not removed, for other types, and
            // an existing manifest's value survives untouched
            // (saveProtocol always writes phaseField back).
            if panel.studyFocus == .conceptStudy {
                HStack(spacing: 6) {
                    Picker("Funnel phase", selection: $draft.phaseField) {
                        Text("not declared").tag("")
                        ForEach(ExperimentStore.knownPhases, id: \.self) { phase in
                            Text(phase).tag(phase)
                        }
                    }
                    .disabled(manifest.status != .draft)
                    .help(
                        "where this study sits in the screen→confirm→"
                            + "triangulate→panel funnel. Two real effects: "
                            + "analyze picks the correction from it (screens "
                            + "get BH-FDR, confirms get the stricter Holm), "
                            + "and 'confirm' additionally requires this "
                            + "study's prompts to be HELD OUT from the "
                            + "screen pool it references (a verify rule). "
                            + "Changes no prompt; never sent to any model")
                    InfoButton(text: StudyInfo.funnelPhase)
                }
            }

            if manifest.status == .draft {
                Button("Save Study Setup") { panel.saveProtocol() }
                    .help(
                        "save the study question, baseline model, "
                            + "generation & sampling settings (and, "
                            + "for concept studies, the funnel phase)")
            }
        }
    }

    /// The study's precision pin. Shown beside the baseline model because it
    /// qualifies that model: same repo at two precisions is two instruments.
    @ViewBuilder
    private func studyDtypePicker(
        manifest: ExperimentManifest, selection: Binding<String>
    ) -> some View {
        let vocabulary = ExperimentStore.judgeDtypeVocabulary
        let current = selection.wrappedValue
        Picker("Precision", selection: selection) {
            Text("device default").tag("")
            ForEach(vocabulary, id: \.self) { name in
                Text(name).tag(name)
            }
            // A pin outside the vocabulary (hand-edited JSON) stays visible
            // but unpickable — the same rule the model pickers use, so a
            // manifest's real state is never silently rewritten by opening
            // it in the editor.
            if !current.isEmpty, !vocabulary.contains(current) {
                Text("\(current) (not loadable)")
                    .tag(current)
                    .selectionDisabled()
            }
        }
        .disabled(manifest.status != .draft)
        .help(StudyControlCopy.studyDtypeHelp)
    }

    /// Scenarios that carry their own seat bindings are marked in the picker,
    /// matching the Panels editor: only one of two same-named entries can be
    /// cast from a study.
    private static func scenarioMenuLabel(_ record: MultiAgentScenarioRecord) -> String {
        PanelAuthoring.carriesBindings(record.scenario)
            ? "\(record.label) — bound (legacy)"
            : record.label
    }
}

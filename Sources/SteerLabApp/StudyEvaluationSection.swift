import ExperimentKit
import SteeringKit
import SwiftUI

/// Evaluation declarations and save controls, with judging UI supplied by the composition boundary.
struct StudyEvaluationSection<JudgingEditor: View>: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel
    @ViewBuilder var judgingEditor: () -> JudgingEditor

    var body: some View {
        Section("Evaluation") {
            // Instrument activation is a MODEL-OUTPUT concern (the
            // outcome instruments score per-prompt generations);
            // multi-agent studies judge transcripts — their
            // Evaluation pane shows judging only (2026-07-19 type
            // parity cleanup).
            if panel.draft.studyKind == .modelOutput {
                instrumentActivationControls(
                    manifest: manifest, panel: panel)
            }
            judgingControls(manifest: manifest, panel: panel)
            // Discriminant controls + instrument scope are
            // evaluation declarations — relocated here from the
            // actions section (2026-08-01), where nobody thought
            // to look for them. (The validation read DEPTH lives
            // with the Validate button instead — review feedback:
            // the setting belongs where validation is launched.)
            if !manifest.concepts.isEmpty {
                DiscriminantControlsSection(
                    manifest: manifest, panel: panel)
            }
            // The instrument scope is about task prompts, not
            // concepts — a concept-less Compare agents study with
            // mixed response formats NEEDS it to declare an
            // answer-token instrument (2026-08-03: the run refusal
            // named a field the UI hid).
            InstrumentScopeSection(manifest: manifest, panel: panel)
        }
    }

    /// Split out of the Evaluation section body (type parity cleanup —
    /// the outcome instruments score per-prompt generations, so they
    /// render for model-output studies only; also relieves the
    /// type-checker).
    @ViewBuilder
    private func instrumentActivationControls(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
        Group {
            // P1 — explicit instrument activation: what the DATA
            // supports (detected), what is ENABLED (declared in the
            // manifest, provenance), and the honest warning when the
            // two disagree. Never auto-enabled.
            if let detected = panel.detectedCapabilitiesLine {
                Label(detected, systemImage: "list.bullet.rectangle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help(
                        "items in the pinned prompt set carrying per-item "
                            + "`options` — a capability of the data, not an "
                            + "enabled measurement")
            }
            HStack(spacing: 6) {
                Picker(
                    "Outcome mode",
                    selection: Binding(
                        get: { panel.outcomeMode },
                        set: { panel.setOutcomeMode($0) })
                ) {
                    if panel.outcomeMode == .notDeclared {
                        Text(InstrumentActivation.OutcomeMode.notDeclared.label)
                            .tag(InstrumentActivation.OutcomeMode.notDeclared)
                    }
                    Text(InstrumentActivation.OutcomeMode.generatedChoice.label)
                        .tag(InstrumentActivation.OutcomeMode.generatedChoice)
                    Text(InstrumentActivation.OutcomeMode.answerTokenProbability.label)
                        .tag(InstrumentActivation.OutcomeMode.answerTokenProbability)
                    Text(InstrumentActivation.OutcomeMode.both.label)
                        .tag(InstrumentActivation.OutcomeMode.both)
                }
                .disabled(manifest.status != .draft)
                .help(StudyControlCopy.outcomeModeHelp)
                InfoButton(text: StudyInfo.evaluationOutcome)
            }
            if let warning = panel.instrumentActivationWarning {
                Label(warning, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }

            // F3 — auxiliary instruments (not owned by the picker)
            // get their own rows with the sampling implication
            // stated, plus a real remove affordance (draft-only,
            // written through the store like every instrument
            // edit).
            ForEach(panel.auxiliaryOutcomeInstruments, id: \.self) { id in
                HStack(alignment: .firstTextBaseline) {
                    Label(
                        InstrumentActivation.auxiliaryDescription(id),
                        systemImage: "waveform.and.magnifyingglass"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    Spacer()
                    Button("Remove") {
                        panel.removeAuxiliaryInstrument(id)
                    }
                    .font(.caption)
                    .disabled(manifest.status != .draft)
                    .help(
                        "removes '\(id)' from outcomeInstruments "
                            + "(draft-only). With the reader removed, "
                            + "Answer-token mode runs genuinely "
                            + "logprob-only — no sampled generation")
                }
            }
            // Ordinal-scale instrument (general option-ladder
            // endpoint) — controls live in their own file; this is
            // the one wiring line.
            OrdinalScaleInstrumentControls(manifest: manifest, panel: panel)
            if let recordKinds = panel.effectiveRecordKindsNote {
                Label(recordKinds, systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
            // Pin/unpin fitted reader artifacts — before 2026-08-01
            // the button below gated on a list nothing in the app
            // could populate.
            ReaderPinControls(manifest: manifest, panel: panel)
            if panel.canAddReaderInstrument {
                Button("Add Reader Instrument (repeReaderScore)") {
                    panel.addReaderInstrument()
                }
                .font(.caption)
                .help(
                    "declares repeReaderScore alongside the current "
                        + "mode — the pinned readers score every "
                        + "sampled response (sampled generation runs "
                        + "even in Answer-token mode)")
            }
        }
    }

    /// Judging controls — the unified judging section (its structure and
    /// copy live in `JudgingSectionControls`, JudgingSectionView.swift)
    /// plus the analysis settings and the save action. Every study type
    /// judges; multi-agent studies judge transcripts.
    @ViewBuilder
    private func judgingControls(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        Group {
            judgingEditor()

            analysisSettings(manifest: manifest, panel: panel)

            if manifest.status == .draft {
                Button("Save Evaluation Settings") { panel.saveProtocol() }
                    .help(
                        "save judges, rubric, structured output "
                            + "instructions, case family, and the "
                            + "option-length acknowledgment")
            }
        }
    }

    /// Analysis-shaping declarations live WITH the evaluation settings:
    /// the case family selects the endpoint parser and the option-length
    /// acknowledgment gates the answer-token instrument — neither is an
    /// inert note.
    @ViewBuilder
    private func analysisSettings(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
        let isDraft = manifest.status == .draft
        // Editable with suggestions (the manifest accepts any string —
        // `setCaseFamily` does not validate against the known list) plus
        // the honest ⓘ: only 'sentencing' has a parser today.
        CaseFamilyField(manifest: manifest, panel: panel)

        // Declared numeric parser + exclusion rules — controls live in
        // their own files (NumericParserPickerView / ExclusionRulesEditor-
        // View); these are the two wiring lines.
        NumericParserControls(manifest: manifest, panel: panel)
        ExclusionRulesEditor(manifest: manifest, panel: panel)

        // Answer-token instrument concern — model-output studies only
        // (multi-agent studies have no option scoring).
        if manifest.studyKind == .modelOutput {
            HStack(spacing: 6) {
                Toggle(
                    "Acknowledge unequal option lengths",
                    isOn: $draft.acknowledgeUnequalOptionLengthsField
                )
                .disabled(!isDraft)
                .help(
                    "opt-in: scored answer options that tokenize to unequal lengths "
                        + "bias joint logprobs toward shorter options — the run loop "
                        + "refuses unequal option sets unless this is acknowledged")
                InfoButton(text: StudyInfo.optionLengths)
            }
        }
    }
}

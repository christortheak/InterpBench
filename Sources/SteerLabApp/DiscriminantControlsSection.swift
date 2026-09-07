import ExperimentKit
import SteeringKit
import SwiftUI

/// Declaring discriminant-validity controls and the outcome-instrument scope.
///
/// Both settings shipped as manifest fields with refusal messages telling the
/// researcher to declare them — and no way to do so. C2's advisory said "add
/// them to validationControls (each with its own stimulus hash and extraction
/// options)"; A2's said "declare outcomeInstrumentScope". Following either
/// meant hand-editing JSON and computing a SHA-256 by hand, which made both
/// features strictly worse than what they replaced for anyone who hit them.
///
/// The hashes are computed by the store from the files on disk. The
/// researcher picks a concept and a method, or picks response formats — never
/// a digest.
struct DiscriminantControlsSection: View {
    let manifest: ExperimentManifest
    @Bindable var panel: ExperimentPanel

    private var isDraft: Bool { manifest.status == .draft }

    /// The control whose "Remove" was clicked, awaiting confirmation.
    @State private var controlPendingRemoval: String?

    var body: some View {
        Section("Discriminant controls") {
            explanation
            declaredControls
            if isDraft { controlEditor }
            if let refusal = panel.draft.formErrors[.validationControl] {
                Label(refusal, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
        }
        .confirmationDialog(
            "Remove this discriminant control?",
            isPresented: Binding(
                get: { controlPendingRemoval != nil },
                set: { if !$0 { controlPendingRemoval = nil } }),
            titleVisibility: .visible,
            presenting: controlPendingRemoval
        ) { concept in
            Button("Remove \(concept)", role: .destructive) {
                panel.removeValidationControl(concept)
                controlPendingRemoval = nil
            }
            Button("Cancel", role: .cancel) { controlPendingRemoval = nil }
        } message: { concept in
            Text("The draft stops bounding its concepts against \(concept), and "
                + "the stimulus hash pinned for it is dropped. You can declare "
                + "it again while the study is a draft.")
        }
    }

    @ViewBuilder
    private var explanation: some View {
        Text("Directions your concepts must NOT collapse into. Every cosine "
            + "is measured at one layer, and each control carries its own "
            + "extraction recipe — never a study concept's.")
            .font(.caption2)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private var declaredControls: some View {
        let controls = manifest.validationControls ?? []
        if controls.isEmpty {
            // Absence is a fact about the evidence, not a blank list.
            Label(
                "none declared — the cosine matrix covers this study's own "
                    + "concepts only, so nothing external bounds their "
                    + "distinctness",
                systemImage: "exclamationmark.shield")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            ForEach(controls, id: \.concept) { control in
                LabeledContent(control.concept) {
                    HStack(spacing: 8) {
                        Text(controlDetail(control))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        if isDraft {
                            Button("Remove") {
                                controlPendingRemoval = control.concept
                            }
                            .buttonStyle(.link)
                            .font(.caption2)
                            .help(
                                "undeclare \(control.concept) as a control — "
                                    + "asks first; the study then measures its "
                                    + "concepts against nothing external at "
                                    + "this position")
                        }
                    }
                }
            }
        }
    }

    private func controlDetail(_ control: ExperimentManifest.ValidationControl) -> String {
        // The researcher-facing name, never the artifact-compatibility raw
        // value ("lat" reads as RepE's LAT and is not one).
        var parts = [control.options.method.label]
        parts.append("stimuli \(control.stimulusSetHash.prefix(8))…")
        if let revision = control.modelRevision {
            parts.append("rev \(revision.prefix(8))…")
        }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private var controlEditor: some View {
        @Bindable var draft = panel.draft
        let candidates = panel.validationControlCandidates
        if candidates.isEmpty {
            Text("no other concepts in this workspace to use as controls")
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else {
            Picker("Concept", selection: $draft.controlConcept) {
                Text("select…").tag("")
                ForEach(candidates, id: \.self) { Text($0).tag($0) }
            }
            .help(
                "the direction this study's concepts must NOT collapse into — "
                    + "a workspace concept that is not one of this study's own")
            Picker("Extraction method", selection: $draft.controlMethod) {
                // Recipe methods only: a control re-derives its vector, so
                // pinnedArtifact/optvec (bytes, not recipes) can't be one.
                ForEach(
                    ExtractionMethod.allCases.filter(\.isRecipeMethod),
                    id: \.self
                ) { method in
                    // `.label`, never `rawValue`: the raw values are
                    // artifact-compatibility constants ("lat"), not names.
                    Text(methodTitle(method)).tag(method)
                }
            }
            .help(
                "the control's OWN recipe — a control authored for grand-mean "
                    + "extraction read by a paired method is measured at a "
                    + "position it was never authored for")
            // The default is a choice with consequences, so it is named rather
            // than left as whatever the picker happened to open on.
            Text("defaults to \(ExtractionMethod.meanDifference.label) — set it "
                + "to the recipe this control's own stimuli were authored for, "
                + "which need not be this study's recipe")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Button("Declare control") { panel.addValidationControl() }
                .disabled(panel.draft.controlConcept.isEmpty)
                .help(
                    "record the picked concept as a discriminant control on "
                        + "this draft, pinning its stimulus hash — declarable "
                        + "while the study is a draft")
            Text("the stimulus hash is read from the concept's files and "
                + "pinned automatically")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// Picker item text: the method's name, with the picker's own default
    /// marked so it cannot be accepted by inattention.
    private func methodTitle(_ method: ExtractionMethod) -> String {
        method == .meanDifference ? "\(method.label) (default)" : method.label
    }
}

/// The outcome-instrument scope declaration, its OWN section (2026-08-03
/// field report): it lived inside DiscriminantControlsSection, which the
/// panel gates on the study having CONCEPTS — so a Compare agents study
/// (concept-less by design) saw the run refusal "declare
/// outcomeInstrumentScope" with the only declaring affordance hidden.
/// Scope is an evaluation fact about task prompts + instruments, not about
/// concepts; it renders for every study type.
struct InstrumentScopeSection: View {
    let manifest: ExperimentManifest
    @Bindable var panel: ExperimentPanel

    private var isDraft: Bool { manifest.status == .draft }

    var body: some View {
        Section("Outcome-instrument scope") {
            scopeRows
        }
    }

    @ViewBuilder
    private var scopeRows: some View {
        let formats = panel.availableResponseFormats
        if let scope = manifest.outcomeInstrumentScope {
            Text("answer-token instruments read "
                + scope.responseFormats.joined(separator: ", ")
                + " rows only — \(scope.itemCount) items pinned")
                .font(.caption)
                .fixedSize(horizontal: false, vertical: true)
            if isDraft {
                Button("Clear scope") { panel.declareOutcomeInstrumentScope([]) }
                    .buttonStyle(.link)
                    .font(.caption2)
                    .help(
                        "undeclare the scope — answer-token instruments go "
                            + "back to reading every response format in the "
                            + "file, and the pinned row set is dropped")
            }
        } else if formats.count > 1 {
            // Only worth offering when the file is genuinely mixed.
            Text("this file mixes response formats: "
                + formats.map { "\($0.format) (\($0.count))" }
                    .joined(separator: ", ")
                // Plain quotes: this string is `+`-concatenated, so SwiftUI
                // renders it verbatim and backticks would show as backticks.
                + ". An answer-token instrument can only read 'label' rows.")
                .font(.caption)
                .fixedSize(horizontal: false, vertical: true)
            if isDraft {
                Button("Scope instruments to label rows") {
                    panel.declareOutcomeInstrumentScope(["label"])
                }
                .help(
                    "records which rows the instrument reads, and pins that "
                        + "row set — which rows were measured is a "
                        + "result-bearing fact")
            }
        } else {
            Text("no scope needed — every option-carrying row shares one "
                + "response format")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

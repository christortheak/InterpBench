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

    var body: some View {
        Section("Discriminant controls") {
            explanation
            declaredControls
            if isDraft { controlEditor }
            if let refusal = panel.formErrors[.validationControl] {
                Label(refusal, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
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
                                panel.removeValidationControl(control.concept)
                            }
                            .buttonStyle(.link)
                            .font(.caption2)
                        }
                    }
                }
            }
        }
    }

    private func controlDetail(_ control: ExperimentManifest.ValidationControl) -> String {
        var parts = [control.options.method.rawValue]
        parts.append("stimuli \(control.stimulusSetHash.prefix(8))…")
        if let revision = control.modelRevision {
            parts.append("rev \(revision.prefix(8))…")
        }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private var controlEditor: some View {
        let candidates = panel.validationControlCandidates
        if candidates.isEmpty {
            Text("no other concepts in this workspace to use as controls")
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else {
            Picker("Concept", selection: $panel.controlConcept) {
                Text("select…").tag("")
                ForEach(candidates, id: \.self) { Text($0).tag($0) }
            }
            Picker("Extraction method", selection: $panel.controlMethod) {
                // Recipe methods only: a control re-derives its vector, so
                // pinnedArtifact/optvec (bytes, not recipes) can't be one.
                ForEach(
                    ExtractionMethod.allCases.filter(\.isRecipeMethod),
                    id: \.self
                ) { method in
                    Text(method.rawValue).tag(method)
                }
            }
            .help(
                "the control's OWN recipe — a control authored for grand-mean "
                    + "extraction read by a paired method is measured at a "
                    + "position it was never authored for")
            Button("Declare control") { panel.addValidationControl() }
                .disabled(panel.controlConcept.isEmpty)
            Text("the stimulus hash is read from the concept's files and "
                + "pinned automatically")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
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
            }
        } else if formats.count > 1 {
            // Only worth offering when the file is genuinely mixed.
            Text("this file mixes response formats: "
                + formats.map { "\($0.format) (\($0.count))" }
                    .joined(separator: ", ")
                + ". An answer-token instrument can only read `label` rows.")
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

import ExperimentKit
import SwiftUI

/// Design lineage and the two explicit save destinations for a study's settings.
struct StudyDesignActionsView: View {
    let manifest: ExperimentManifest
    let management: StudyManagementController
    var openTemplates: () -> Void
    @State private var confirmSaveBackToDesign = false
    @State private var saveReview: SaveReview?
    private struct SaveReview {
        let source: StudyDesignSourceReview
        let design: StudyDesignSnapshot
    }

    var body: some View {
        templateLineageRows(manifest: manifest, panel: management)
        designWriteBackRow(manifest: manifest, panel: management)
            .onChange(of: manifest.name) { _, _ in
                confirmSaveBackToDesign = false
                saveReview = nil
            }
    }

    /// Template lineage on the study detail — one subtle line, plus the
    /// batch's other studies by DISPLAY name (the batch is read by a human,
    /// and canonical casting names are unreadable by design).
    @ViewBuilder
    private func templateLineageRows(
        manifest: ExperimentManifest, panel: StudyManagementController
    ) -> some View {
        if let lineage = panel.designs.templateLineage(manifest, experiments: panel.experiments) {
            Text(lineage)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            let siblings = panel.batchSiblings(manifest)
            if !siblings.isEmpty {
                DisclosureGroup(
                    "Minted with \(siblings.count) sibling "
                        + "\(siblings.count == 1 ? "study" : "studies")"
                ) {
                    ForEach(siblings, id: \.self) { sibling in
                        Text(sibling)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption2)
                .help(
                    "the other studies minted from this design in the same "
                        + "batch, by their display names")
            }
        }
    }

    /// The two ways this study's settings become a design — both visible, both
    /// worded for what they do, neither a default.
    ///
    /// "Save back to design" is the return leg of Templates' "Edit design…": it
    /// OVERWRITES the design the lineage line names. "Save as new design" is the
    /// existing mint, which adds an entry. The difference matters enough to be
    /// two buttons rather than one button with a mode: one of them grows the
    /// library and the other does not.
    @ViewBuilder
    private func designWriteBackRow(
        manifest: ExperimentManifest, panel: StudyManagementController
    ) -> some View {
        let target = panel.designs.saveBackToDesignTarget(for: manifest)
        let refusal = panel.designs.saveBackToDesignRefusal(for: manifest)
        HStack(spacing: 8) {
            if let target {
                Button("Save back to design '\(target)'") {
                    do {
                        saveReview = SaveReview(source: try panel.reviewEditorDesignSource(),
                            design: try panel.designs.reviewedDesign(named: target))
                        confirmSaveBackToDesign = true
                    } catch { panel.draft.formErrors[.template] = error.localizedDescription }
                }
                .disabled(refusal != nil)
                .help(refusal ?? StudyControlCopy.saveBackHelp)
                .confirmationDialog(
                    "Update reviewed design?",
                    isPresented: $confirmSaveBackToDesign,
                    titleVisibility: .visible,
                    presenting: saveReview
                ) { review in
                    Button("Update '\(review.design.template.name)' in place") {
                        panel.updateDesign(reviewedSource: review.source, reviewedDesign: review.design)
                    }
                } message: { review in
                    Text(Self.saveBackConfirmation(design: review.design.template.name, source: review.source.study.manifest.name)
                        + (review.source.panel?.warnings.isEmpty == false ? "\n\n" + (review.source.panel?.warnings.joined(separator: "\n") ?? "") : ""))
                }
            }
            Button("Save as new design") {
                do { panel.newDesignFromStudy(reviewedSource: try panel.reviewEditorDesignSource()) }
                catch { panel.draft.formErrors[.template] = error.localizedDescription }
            }
            // `saveAsNewDesignHelp` ends a sentence, so the appended clause
            // has to start one rather than trail a lowercase fragment.
            .help(
                StudyControlCopy.saveAsNewDesignHelp
                    + " Uses the SAVED study settings — save Study Setup first "
                    + "to include unsaved edits.")
            if target == nil {
                Button("Open Templates") { openTemplates() }
                    .buttonStyle(.link)
                    .font(.caption2)
                    .help(
                        "opens the Templates section, where designs are listed, "
                            + "edited and instantiated into studies")
            }
        }
        if let refusal, target != nil {
            Text(refusal)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        if let error = panel.draft.formErrors[.template] {
            Label(error, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }

    /// Stated plainly, because it is the one thing about the round trip a
    /// researcher could be surprised by afterwards.
    private static func saveBackConfirmation(design: String, source: String) -> String {
        "Uses the reviewed saved settings of study '\(source)'. Save Study Setup first to include unsaved edits. Keeps every generation and "
            + "measurement setting, no agents — and updates design '\(design)' "
            + "in place. Its content hash changes. Studies already minted from "
            + "it keep their original lineage stamps, so their divergence "
            + "display goes on reporting what they were minted from. The "
            + "design's name, description and creation date are unchanged."
    }
}

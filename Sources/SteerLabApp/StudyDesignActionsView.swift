import ExperimentKit
import SwiftUI

/// Design lineage and the two explicit save destinations for a study's settings.
struct StudyDesignActionsView: View {
    let manifest: ExperimentManifest
    let management: StudyManagementController
    var openTemplates: () -> Void
    @State private var confirmSaveBackToDesign = false

    var body: some View {
        templateLineageRows(manifest: manifest, panel: management)
        designWriteBackRow(manifest: manifest, panel: management)
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
                DisclosureGroup("Minted with \(siblings.count) sibling study(s)") {
                    ForEach(siblings, id: \.self) { sibling in
                        Text(sibling)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption2)
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
                    confirmSaveBackToDesign = true
                }
                .disabled(refusal != nil)
                .help(refusal ?? StudyControlCopy.saveBackHelp)
                .confirmationDialog(
                    "Update design '\(target)'?",
                    isPresented: $confirmSaveBackToDesign,
                    titleVisibility: .visible
                ) {
                    Button("Update '\(target)' in place") {
                        panel.saveSelectedStudyBackToDesign()
                    }
                } message: {
                    Text(Self.saveBackConfirmation(design: target))
                }
            }
            Button("Save as new design") {
                panel.newDesignFromStudy(named: manifest.name)
            }
            .help(StudyControlCopy.saveAsNewDesignHelp)
            if target == nil {
                Button("Open Templates") { openTemplates() }
                    .buttonStyle(.link)
                    .font(.caption2)
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
    private static func saveBackConfirmation(design: String) -> String {
        "Strips this study to its design form — every generation and "
            + "measurement setting, no agents — and updates design '\(design)' "
            + "in place. Its content hash changes. Studies already minted from "
            + "it keep their original lineage stamps, so their divergence "
            + "display goes on reporting what they were minted from. The "
            + "design's name, description and creation date are unchanged."
    }
}

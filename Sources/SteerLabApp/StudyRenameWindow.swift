import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct RenameStudySheet: Identifiable {
    let id = UUID()
    let reviewed: DraftAuthoringSnapshot
    let name: String
    let status: ExperimentManifest.Status
    let label: String
    /// Run directories already stamping this study's canonical name — the
    /// number a draft rename would strand (runs are immutable and are never
    /// rewritten).
    let runsStamped: Int

    var isDraft: Bool { status == .draft }
}

/// The ONE rename affordance. A draft offers both effects — canonical name
/// and display label — in a single action; a frozen or completed study
/// offers only the label, because its canonical name is hashed into the
/// manifest and stamped into every run's provenance.
struct RenameStudyWindow: View {
    let sheet: RenameStudySheet
    let panel: ExperimentPanel

    @Environment(\.dismiss) private var dismiss
    @State private var canonicalName = ""
    @State private var label = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Rename Study")
                    .font(.title2.weight(.semibold))
                Text(sheet.name)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            Divider()

            Form {
                if sheet.isDraft {
                    Section("Name") {
                        TextField("study name", text: $canonicalName)
                            .font(.body.monospaced())
                            .help(StudyControlCopy.canonicalNameHelp)
                        Text(StudyControlCopy.canonicalNameHelp)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        if sheet.runsStamped > 0 {
                            Label(strandedRunsNote, systemImage: "exclamationmark.triangle")
                                .font(.caption)
                                .foregroundStyle(.orange)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }

                Section("Display label") {
                    TextField("display label (optional)", text: $label)
                        .help(labelHelp)
                    Text(labelHelp)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .formStyle(.grouped)

            // The refusal (a name collision, a manifest that moved, a
            // workspace switch) renders HERE. It used to land in the form
            // behind this sheet, so a collision looked like a dead button
            // (UI audit 2026-09-06, headline 9).
            if let refusal = panel.draft.formErrors[.rename] {
                Label(refusal, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close without renaming; nothing is written")
                Button(primaryTitle) {
                    if panel.management.rename(reviewed: sheet.reviewed,
                        canonicalName: sheet.isDraft ? canonicalName : nil,
                        label: label) { dismiss() }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!hasChange)
                .help(primaryHelp)
            }
        }
        .padding(18)
        .frame(minWidth: 480, minHeight: sheet.isDraft ? 440 : 320)
        .onAppear {
            canonicalName = sheet.name
            label = sheet.label
        }
    }

    private var hasChange: Bool {
        (sheet.isDraft && canonicalName != sheet.name) || label != sheet.label
    }

    /// A frozen study's only available effect is the display label, so the
    /// button says what it will actually do (UI audit 2026-09-06).
    private var primaryTitle: String {
        sheet.isDraft ? "Rename" : "Set Label"
    }

    private var primaryHelp: String {
        guard hasChange else {
            return sheet.isDraft
                ? "nothing to apply yet — change the name or the display label "
                    + "above"
                : "nothing to apply yet — change the display label above"
        }
        return sheet.isDraft
            ? "applies both effects in one action: the canonical rename moves "
                + "experiments/<name>/ and rewrites the manifest, then the "
                + "display label is written beside it"
            : "writes the display label beside the manifest — no hash moves and "
                + "no run is rewritten"
    }

    // Long strings live outside the body — interpolating them inline blows
    // the SwiftUI type-checker budget.

    private var strandedRunsNote: String {
        let plural = sheet.runsStamped == 1 ? "run" : "runs"
        return "\(sheet.runsStamped) existing \(plural) stamp '\(sheet.name)'. Runs "
            + "are immutable and a rename never rewrites them — they will no longer "
            + "list under this study."
    }

    private var labelHelp: String {
        guard !sheet.isDraft else {
            return "shown first in study lists; the name above stays the identity "
                + "everywhere else."
        }
        return "shown first in study lists. This study is \(sheet.status.rawValue): "
            + "its name is hashed into the frozen manifest and stamped into every "
            + "run, so the canonical id stays '\(sheet.name)'. A label is stored "
            + "beside the manifest and moves no hash."
    }
}

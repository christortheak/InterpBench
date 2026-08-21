import ExperimentKit
import SwiftUI

/// ONE place for what's wrong with the selected study (2026-07-19 second
/// pass) — replacing the "complex mess of warnings" scattered across
/// sections. Three severities, in order:
///
/// - verify VIOLATIONS (red): pinned inputs no longer match reality —
///   blocking; a frozen study with violations must not run.
/// - data BLOCKERS (red): files the run/freeze will refuse without —
///   the compact echo of Data & Prompts, which has the fix affordances.
/// - ADVISORIES (orange): non-blocking evidence-grade cautions (from the
///   cached freeze-readiness computation — never recomputed per render)
///   plus the gate-less pipeline note.
///
/// The per-control context stays where it is (a disabled Freeze button
/// still explains itself); this box is the one-glance answer to "is this
/// study sound, and if not, why".
struct StudyIssuesSection: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel

    /// Readiness blockers, recomputed when the manifest changes (the scan
    /// reads files — never per-render).
    @State private var dataBlockers: [DataRequirement] = []

    private var advisories: [String] {
        var lines = panel.freezeReadiness?.advisories ?? []
        if let pipeline = PipelineDraft.parse(manifest.pipeline),
            pipeline.isGateless
        {
            lines.append(
                "pipeline declares no gates — the chain runs every stage to "
                    + "completion with no scientific stop conditions "
                    + "(fine for exploration)")
        }
        return lines
    }

    private var isClean: Bool {
        panel.violations.isEmpty && dataBlockers.isEmpty && advisories.isEmpty
    }

    var body: some View {
        Section {
            if isClean {
                Label(
                    "nothing wrong — pins verify, required data is present, "
                        + "no notes",
                    systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.green)
            }
            if !panel.violations.isEmpty || !dataBlockers.isEmpty {
                Text("Problems found — blocking")
                    .font(.caption.bold())
                    .foregroundStyle(.red)
            }
            ForEach(panel.violations, id: \.self) { violation in
                Label(violation, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .help(
                        "a problem found by verification (a 'verify "
                            + "violation'): a pinned input no longer matches "
                            + "what was recorded — a frozen study in this "
                            + "state must not be run; re-pin on a duplicate "
                            + "draft if the change was deliberate")
            }
            ForEach(dataBlockers) { blocker in
                Label(
                    "\(blocker.title): \(blocker.status.rawValue) — see "
                        + "Data & Prompts above to fix",
                    systemImage: "doc.badge.ellipsis")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .help(blocker.detail)
            }
            if !advisories.isEmpty {
                Text("Notes — non-blocking")
                    .font(.caption.bold())
                    .foregroundStyle(.orange)
            }
            ForEach(advisories, id: \.self) { advisory in
                Label(advisory, systemImage: "info.circle")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .help("a note ('advisory' in reports) — non-blocking; "
                        + "affects the evidence grade the study can claim, "
                        + "not whether it can run")
            }
        } header: {
            InfoSectionHeader(title: "Issues", text: StudyInfo.issues)
        }
        .onAppear { recompute() }
        .onChange(of: manifest) { recompute() }
    }

    private func recompute() {
        dataBlockers = StudyDataReadiness.requirements(
            for: manifest, workspaceRoot: VectorCatalog.projectRoot
        ).filter { $0.status == .missing || $0.status == .invalid }
    }
}

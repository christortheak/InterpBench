import ExperimentKit
import SwiftUI

struct StudyRecentJobsView: View {
    @Bindable var jobs: StudyRemoteJobController
    var resume: @MainActor (String) async -> Void
    var importEvidence: @MainActor (String) async -> Void
    var refresh: @MainActor () async -> Void
    var body: some View { recentServerJobsGroup() }
    @ViewBuilder
    private func recentServerJobsGroup() -> some View {
        Group {
            Text("Recent server jobs")
                .font(.caption.bold())
                .padding(.top, 4)
            ForEach(jobs.recentServerJobs) { job in
                HStack(spacing: 8) {
                    Text(job.id)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                    // Jobs DISCOVERED by a refresh carry no study name, so
                    // the row used to read "run · — · running" (audit 10):
                    // drop the segment rather than print a dash.
                    Text(Self.jobSummary(job))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    // Checkpointed (resumable) jobs get the Resume button
                    // right on the row — the 2026-07-22 incident was this
                    // exact dead end.
                    if RemoteJobStatusClass.offersResume(status: job.state) {
                        Button("Resume") {
                            Task { await resume(job.id) }
                        }
                        .controlSize(.small)
                        .help(
                            "re-submit this checkpointed job's own sbatch "
                                + "script — the run continues from its "
                                + "checkpoint; the status line reports the "
                                + "new Slurm job id")
                    }
                    // Completed run-verb jobs can carry an evidence bundle:
                    // the same import the Run-on-Server disclosure offers,
                    // right where the finished job is listed.
                    if ExperimentPanel.jobOffersEvidenceImport(
                        verb: job.verb, state: job.state)
                    {
                        Button("Import Evidence") {
                            Task { await importEvidence(job.id) }
                        }
                        .controlSize(.small)
                        .help(
                            "download this job's evidence bundle, verify its "
                                + "hashes, and land it under this workspace's "
                                + "runs/ — the status line names the imported "
                                + "run directory")
                    }
                }
                .padding(.vertical, 1)
            }
            if let imported = jobs.remoteImportedRunDirectory {
                LabeledContent("Imported run", value: imported)
                    .font(.caption2.monospaced())
                    .textSelection(.enabled)
                    .help(
                        "already visible in Results — imports land as immutable runs/ directories")
            }
            Button("Refresh Job States") {
                Task { await refresh() }
            }
            .help("re-query the active server's job list for current states")
            Text(StudyControlCopy.recentJobsCaption)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// verb · study · state, with the study segment omitted when the job was
    /// discovered rather than submitted from here (its record has no name).
    private static func jobSummary(_ job: StudyRemoteJobController.RecentServerJob) -> String {
        let study = job.study.trimmingCharacters(in: .whitespaces)
        let named = !study.isEmpty && study != "—" && study != "-"
        return named
            ? "\(job.verb) · \(study) · \(job.state)"
            : "\(job.verb) · \(job.state)"
    }
}

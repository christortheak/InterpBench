import AppKit
import ExperimentKit
import SwiftUI

/// "Export Results…" on a study's results: tables that open in R, Stata, SPSS,
/// or a spreadsheet, transcripts for coding by hand, a methods summary, and a
/// codebook, written to a new folder and revealed in Finder.
///
/// The export is the Python client's, reached through `ResultsExport`: the same
/// files `results export` writes on either command line. This view only asks
/// for it, shows what came back, and reveals the folder. Whether the button is
/// available is decided by `ResultsExport.canExport`, which is unit-tested.
struct ResultsExportButton: View {
    /// The selected study, or nil when none is selected.
    let studyName: String?
    let workspaceRoot: URL
    @Bindable var results: StudyResultsState

    @State private var isConfirming = false
    @State private var isExporting = false
    @State private var outcome: ResultsExport.Outcome?
    @State private var refusal: ExportProblem?

    /// What went wrong, and what to do about it.
    private struct ExportProblem: Equatable {
        let reason: String
        let repair: String
    }

    private var selectedRun: StudyRunListItem? {
        ResultsExport.runToExport(selected: results.selectedResult?.item)
    }

    private var isAvailable: Bool {
        studyName != nil
            && ResultsExport.canExport(runs: results.resultRuns, isExporting: isExporting)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Button("Export Results…") {
                    isConfirming = true
                }
                .disabled(!isAvailable)
                .help(
                    ResultsExport.unavailableReason(
                        runs: results.resultRuns, isExporting: isExporting)
                        ?? "writes this study's results to a new folder: tables "
                        + "for R, Stata, SPSS, or a spreadsheet, transcripts, a "
                        + "methods summary, and a codebook — nothing in runs/ "
                        + "is changed")
                if isExporting {
                    ProgressView().controlSize(.small)
                    Text("Exporting…").font(.caption).foregroundStyle(.secondary)
                }
            }
            if let outcome {
                exported(outcome)
            }
            if let refusal {
                Text(refusal.reason)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
                if !refusal.repair.isEmpty {
                    Text(refusal.repair)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
        .confirmationDialog(
            "Export results for \(studyName ?? "this study")?",
            isPresented: $isConfirming, titleVisibility: .visible
        ) {
            Button("Export") {
                Task { await export() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                (selectedRun.map { "The selected run, \($0.directoryName), is exported" }
                    ?? "The newest completed run is exported")
                    + ", with its newest analysis and evaluation. Tables, "
                    + "transcripts, a methods summary, and a codebook are written "
                    + "to a new folder under exports/ in this workspace. Nothing "
                    + "in runs/ is changed, and no statistic is recalculated.")
        }
    }

    @ViewBuilder
    private func exported(_ outcome: ResultsExport.Outcome) -> some View {
        Text("Exported to \(outcome.directory.path)")
            .font(.caption2.monospaced())
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
            .help(outcome.directory.path)
        ForEach(outcome.notAvailable, id: \.what) { missing in
            Text("Not available: \(missing.what) — \(missing.why).")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        if outcome.freezeForced {
            Text("This study was frozen with force. The methods summary says so.")
                .font(.caption)
                .foregroundStyle(.orange)
        }
        Button("Reveal Export in Finder") {
            NSWorkspace.shared.activateFileViewerSelecting([outcome.directory])
        }
        .font(.caption)
        .help("opens the exported folder in Finder")
    }

    private func export() async {
        guard let studyName else { return }
        isExporting = true
        outcome = nil
        refusal = nil
        defer { isExporting = false }
        do {
            let exported = try await ResultsExport.export(
                workspaceRoot: workspaceRoot, study: studyName,
                run: selectedRun?.directoryName, client: .app)
            outcome = exported
            NSWorkspace.shared.activateFileViewerSelecting([exported.directory])
        } catch let declined as ResultsExport.Refusal {
            refusal = ExportProblem(reason: declined.reason, repair: declined.repairAction)
        } catch let error as ExperimentError {
            // The local Python client could not run at all: the bridge's own
            // error names the repair (Research Setup).
            refusal = ExportProblem(
                reason: error.reason,
                repair: error.malformedInvocation?.repairAction ?? "")
        } catch {
            refusal = ExportProblem(reason: "\(error)", repair: "")
        }
    }
}

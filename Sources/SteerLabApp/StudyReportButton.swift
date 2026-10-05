import ExperimentKit
import SwiftUI

/// "Open Report" on a study's results: the study's stored results as one
/// readable page, shown in the same sheet as a scientific report.
///
/// The app draws nothing itself. It asks the shared Python client for the
/// page through `ResultsReport` (the same file `results report` writes on
/// either command line), then shows that file, which a browser opens and a
/// colleague can be sent. The page goes under `reports/` in the workspace;
/// nothing under `runs/` is written. Whether the button is available is
/// decided by `ResultsReport.canOpen`, which is unit-tested.
struct StudyReportButton: View {
    /// The selected study, or nil when none is selected.
    let studyName: String?
    let workspaceRoot: URL
    @Bindable var results: StudyResultsState

    @State private var page: ScienceReport.Page?
    @State private var problem: Problem?
    @State private var isOpening = false

    /// What went wrong, and what to do about it.
    private struct Problem: Equatable {
        let reason: String
        let repair: String
    }

    private var selectedRun: StudyRunListItem? {
        ResultsReport.runToShow(selected: results.selectedResult?.item)
    }

    var body: some View {
        Button {
            Task { await open() }
        } label: {
            Label("Open Report", systemImage: "doc.richtext")
        }
        .disabled(studyName == nil || !ResultsReport.canOpen(runs: results.resultRuns, isOpening: isOpening))
        .help(
            ResultsReport.unavailableReason(runs: results.resultRuns, isOpening: isOpening)
                ?? ((selectedRun.map { "shows the selected run, \($0.directoryName)," }
                    ?? "shows the newest completed run,")
                    + " as one readable page: the headline outcome, every effect with its "
                    + "interval, the judges, exclusions, and how the study was frozen — the "
                    + "same HTML file you can open in a browser or send to a colleague; "
                    + "nothing in runs/ is changed"))
        .sheet(item: $page) { page in
            ScienceReportSheet(
                page: page,
                origin: "Drawn from the run’s stored results, outside the run folder")
        }
        .alert(
            "The results page could not be opened",
            isPresented: Binding(get: { problem != nil }, set: { if !$0 { problem = nil } })
        ) {
            Button("OK") { problem = nil }
        } message: {
            Text([problem?.reason, problem?.repair].compactMap { $0 }.filter { !$0.isEmpty }
                .joined(separator: "\n\n"))
        }
    }

    private func open() async {
        guard let studyName else { return }
        isOpening = true
        defer { isOpening = false }
        do {
            page = try await ResultsReport.write(
                workspaceRoot: workspaceRoot, study: studyName,
                run: selectedRun?.directoryName, client: .app
            ).sheetPage
        } catch let declined as ResultsExport.Refusal {
            problem = Problem(reason: declined.reason, repair: declined.repairAction)
        } catch let error as ExperimentError {
            // The local Python client could not run at all: the bridge's own
            // error names the repair (Research Setup).
            problem = Problem(
                reason: error.reason, repair: error.malformedInvocation?.repairAction ?? "")
        } catch {
            problem = Problem(reason: "\(error)", repair: "")
        }
    }
}

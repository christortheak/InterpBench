import Foundation

/// A study's stored results as one readable page: the headline outcome with
/// its rule, every stored effect with its interval, each condition, controls,
/// the judges and their agreement, exclusions, and how the study was frozen.
///
/// The page is drawn in exactly one place, the shared Python client
/// (`steerlab_server.client.results_report`, which draws from the same reading
/// `results export` makes). This type only carries a request across the local
/// bridge and reads the answer back, so the Mac command line, the app, and the
/// Python client all write the same file. Nothing here reads a run directory
/// or draws anything, and nothing under `runs/` is ever written.
public enum ResultsReport {

    /// The bridge action. Python twin: `results_commands.REPORT_BRIDGE_ACTION`.
    public static let action = "results-report"
    /// The page's file name, under `reports/<study>/<run>/` by default.
    public static let pageName = "report.html"
    public static let purpose =
        "Write a completed run's results as one readable page (HTML) outside runs/: the "
        + "headline outcome with its interval, every stored effect, each condition, controls, "
        + "the judges' agreement, exclusions, and how the study was frozen. The page goes under "
        + "reports/ in the workspace unless --out names the file. Runs no model, and "
        + "recalculates nothing."

    /// Where the page was written, and what it was drawn from.
    public struct Written: Sendable, Equatable {
        /// The page itself, outside any run directory.
        public let page: URL
        /// The run the page shows, relative to the workspace.
        public let run: String
        /// The analysis whose effects the page shows, when one exists.
        public let analysis: String?
        /// False when the same page was already there, byte for byte.
        public let changed: Bool
        /// What the run did not store, for the asking surface to repeat.
        public let notAvailable: [ResultsExport.Unavailable]
        /// The study was frozen with force. The page says so first.
        public let freezeForced: Bool
        /// The Python client's whole answer, for the command line's envelope.
        public let response: [String: JSONValue]

        /// One sentence for a person: where the page is and what is missing.
        public var message: String {
            var text =
                (changed
                    ? "The results page was written to "
                    : "The results page was already up to date at ") + page.path + "."
            if !notAvailable.isEmpty {
                text += " Not available: " + notAvailable.map(\.what).joined(separator: ", ") + "."
            }
            return text + " Nothing under runs/ was changed, and no statistic was recalculated."
        }

        /// The same page as `ScienceReport` shows it in the app's report sheet.
        public var sheetPage: ScienceReport.Page { .init(url: page, inRunDirectory: false) }
    }

    /// The request as it crosses the bridge. `run` and `out` are left out when
    /// absent: the Python client then shows the newest completed run, and
    /// writes the page under the workspace's `reports/`.
    public static func payload(
        workspaceRoot: URL, study: String, run: String? = nil, out: String? = nil,
        client: ResultsExport.Client
    ) -> [String: JSONValue] {
        ResultsExport.payload(
            workspaceRoot: workspaceRoot, study: study, run: run, out: out, client: client)
    }

    /// Write the page. Throws `ResultsExport.Refusal` when the Python client
    /// declined, and the bridge's own error when it cannot run at all.
    public static func write(
        workspaceRoot: URL, study: String, run: String? = nil, out: String? = nil,
        client: ResultsExport.Client, python: URL? = nil, source: URL? = nil
    ) async throws -> Written {
        let answer = try await DiagnosticWorkspace.perform(
            action,
            payload: payload(
                workspaceRoot: workspaceRoot, study: study, run: run, out: out, client: client),
            python: python, source: source)
        return try written(from: answer)
    }

    /// Read the Python client's answer: a refusal arrives as data, with its
    /// code and state, in the same shape `results export` uses.
    static func written(from answer: JSONValue) throws -> Written {
        guard case .object(let fields) = answer else {
            throw ExperimentError(reason: "The results page request returned no answer.")
        }
        if case .object(let refusal) = fields["refusal"] {
            throw ResultsExport.Refusal(
                code: string(refusal["code"]) ?? "refused",
                reason: string(refusal["reason"]) ?? "The results page was refused.",
                repairAction: string(refusal["repairAction"]) ?? "",
                state: string(refusal["state"]).flatMap(SteerLabCLIState.init(rawValue:))
                    ?? .refused)
        }
        guard let path = string(fields["htmlPath"]), let run = string(fields["run"]) else {
            throw ExperimentError(reason: "The results page request answered without naming the page.")
        }
        var notAvailable: [ResultsExport.Unavailable] = []
        if case .array(let entries) = fields["notAvailable"] {
            for case .object(let entry) in entries {
                guard let what = string(entry["what"]) else { continue }
                notAvailable.append(
                    .init(what: what, why: string(entry["why"]) ?? "", repair: string(entry["repair"])))
            }
        }
        return Written(
            page: URL(filePath: path), run: run, analysis: string(fields["analysis"]),
            changed: fields["changed"] == .bool(true), notAvailable: notAvailable,
            freezeForced: fields["freezeForced"] == .bool(true), response: fields)
    }

    private static func string(_ value: JSONValue?) -> String? {
        if case .string(let text) = value { return text }
        return nil
    }

    // MARK: The app's button

    /// The run "Open Report" would show for this selection: the selected run
    /// when it is a completed study run, otherwise nil, which asks the Python
    /// client for the newest completed run. The same rule as Export Results.
    public static func runToShow(selected: StudyRunListItem?) -> StudyRunListItem? {
        ResultsExport.runToExport(selected: selected)
    }

    /// Whether there is a page to show: the study has a completed run, and no
    /// page is already being written.
    public static func canOpen(runs: [StudyRunListItem], isOpening: Bool) -> Bool {
        !isOpening && runs.contains(where: ResultsExport.isCompletedRun)
    }

    /// Why the button is unavailable, in a sentence a researcher can act on,
    /// or nil when it is available.
    public static func unavailableReason(runs: [StudyRunListItem], isOpening: Bool) -> String? {
        if isOpening { return "The results page is being written." }
        if runs.contains(where: ResultsExport.isCompletedRun) { return nil }
        return "This study has no completed run yet. Run it first; the results page can "
            + "be opened once a run has finished."
    }
}

/// `steerlab-cli results report <study> [--run <run-dir>] [--out <file>]`.
enum ResultsReportCLI {

    static let usage = "steerlab-cli results report <study> [--run <run-dir>] [--out <file>]"

    static func run(
        _ invocation: ExperimentCLIInvocation, sink: ExperimentCLISink,
        workspaceRoot: URL, python: URL? = nil, source: URL? = nil
    ) async throws -> ExperimentCLIResult {
        var positionals: [String] = []
        var flags: [String: String] = [:]
        var index = 1
        while index < invocation.args.count {
            let word = invocation.args[index]
            if word == "--run" || word == "--out", index + 1 < invocation.args.count {
                flags[word] = invocation.args[index + 1]
                index += 2
            } else {
                positionals.append(word)
                index += 1
            }
        }
        guard positionals.count == 1, let study = positionals.first else {
            throw ExperimentError.malformed(
                "Name one study to make a results page for.",
                repair: usage
                    + "  (steerlab-cli experiment list shows the studies in this workspace)")
        }
        // A file typed relative to where the command was run, not to the
        // workspace: the Python client runs in a directory of its own.
        let destination = flags["--out"].map {
            URL(
                filePath: ($0 as NSString).expandingTildeInPath,
                relativeTo: URL(filePath: FileManager.default.currentDirectoryPath)
            ).standardizedFileURL.path
        }
        do {
            let written = try await ResultsReport.write(
                workspaceRoot: workspaceRoot, study: study, run: flags["--run"],
                out: destination, client: .commandLine, python: python, source: source)
            sink.out("Results page for \(study): \(written.page.path)")
            sink.out("  from \(written.run)" + (written.analysis.map { ", analysis \($0)" } ?? ""))
            for missing in written.notAvailable {
                sink.out("  not available: \(missing.what) (\(missing.why))")
            }
            if written.freezeForced {
                sink.out("  note: this study was frozen with force; the page says so first.")
            }
            sink.out("  Open the file in a web browser. It is one self-contained page you can send as it is.")
            return ExperimentCLIResult(
                message: written.message, changed: written.changed, payload: written.response)
        } catch let refusal as ResultsExport.Refusal {
            sink.err(
                "steerlab-cli results report: \(refusal.reason)\n"
                    + (refusal.repairAction.isEmpty ? "" : "  \(refusal.repairAction)\n"))
            throw ExperimentCLIStop(
                exitCode: 1, state: refusal.state, code: refusal.code,
                reason: refusal.reason, repairAction: refusal.repairAction)
        }
    }
}

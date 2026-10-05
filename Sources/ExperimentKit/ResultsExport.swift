import Foundation

/// Taking a study's stored results out of the workspace: tables that open in
/// R, Stata, SPSS, or a spreadsheet, transcripts for coding by hand, a methods
/// summary, and a codebook.
///
/// The export itself is written once, in the Python client
/// (`steerlab_server.client.results_export`), and runs without a model or a
/// GPU. This type only carries a request across the local bridge and reads the
/// answer back, so the Mac command line and the app export exactly what the
/// Python client exports. Nothing here reads a run directory or builds a
/// table, and nothing under `runs/` is ever written.
public enum ResultsExport {

    /// The bridge action. Python twin: `results_commands.BRIDGE_ACTION`.
    public static let action = "results-export"

    /// Which surface is asking. A refusal's repair is written for it: commands
    /// for the command line, plain steps for the app.
    public enum Client: String, Sendable {
        case commandLine = "steerlab-cli"
        case app
    }

    /// One file the export wrote. `rows` is present for a table.
    public struct WrittenFile: Sendable, Equatable {
        public let file: String
        public let rows: Int?
    }

    /// Something the run did not store, so the export could not include it.
    public struct Unavailable: Sendable, Equatable {
        public let what: String
        public let why: String
        /// What to do about it on the surface that asked, when there is
        /// something to do.
        public let repair: String?
    }

    public struct Outcome: Sendable, Equatable {
        /// The new folder the export was written into.
        public let directory: URL
        /// The run that was exported, relative to the workspace.
        public let run: String
        /// The analysis whose effect table was copied, when one exists.
        public let analysis: String?
        public let files: [WrittenFile]
        public let notAvailable: [Unavailable]
        /// The study was frozen with force. `methods.md` says so; a caller
        /// should say so too.
        public let freezeForced: Bool
        /// The Python client's whole answer, for the command line's envelope.
        public let response: [String: JSONValue]

        /// One sentence for a person: where the files are and what is missing.
        public var message: String {
            var text = "Results exported to \(directory.path)."
            if !notAvailable.isEmpty {
                text +=
                    " Not available: "
                    + notAvailable.map(\.what).joined(separator: ", ") + "."
            }
            return text
                + " Nothing under runs/ was changed, and no statistic was recalculated."
        }

        /// What a person at a terminal reads after an export.
        public var summaryLines: [String] {
            var lines = [
                "Exported to \(directory.path)",
                "  from \(run)" + (analysis.map { ", analysis \($0)" } ?? ""),
            ]
            lines += files.map { entry in
                "  \(entry.file)" + (entry.rows.map { "  (\($0) rows)" } ?? "")
            }
            lines += notAvailable.map { "  not available: \($0.what) (\($0.why))" }
            if freezeForced {
                lines.append("  note: this study was frozen with force; methods.md says so.")
            }
            return lines
        }
    }

    /// An export the Python client declined, with its own code, state, and a
    /// repair the asking surface can carry out.
    public struct Refusal: Error, Sendable, Equatable, CustomStringConvertible {
        public let code: String
        public let reason: String
        public let repairAction: String
        public let state: SteerLabCLIState

        public var description: String { reason }
    }

    /// The request as it crosses the bridge. `run` and `out` are left out when
    /// absent: the Python client then exports the newest completed run into a
    /// new folder under the workspace's `exports/`.
    public static func payload(
        workspaceRoot: URL, study: String, run: String? = nil, out: String? = nil,
        client: Client
    ) -> [String: JSONValue] {
        var payload: [String: JSONValue] = [
            "workspaceRoot": .string(workspaceRoot.path),
            "study": .string(study),
            "client": .string(client.rawValue),
        ]
        if let run, !run.isEmpty { payload["run"] = .string(run) }
        if let out, !out.isEmpty { payload["out"] = .string(out) }
        return payload
    }

    /// Run the export. Throws `Refusal` when the Python client declined, and
    /// the bridge's own error when the local Python client cannot run at all.
    public static func export(
        workspaceRoot: URL, study: String, run: String? = nil, out: String? = nil,
        client: Client, python: URL? = nil, source: URL? = nil
    ) async throws -> Outcome {
        let answer = try await DiagnosticWorkspace.perform(
            action,
            payload: payload(
                workspaceRoot: workspaceRoot, study: study, run: run, out: out,
                client: client),
            python: python, source: source)
        return try outcome(from: answer)
    }

    /// Read the Python client's answer. A refusal arrives as data, with its
    /// code and state, so it can be answered here as the Python client
    /// answers it rather than as one flat failure.
    static func outcome(from answer: JSONValue) throws -> Outcome {
        guard case .object(let fields) = answer else {
            throw ExperimentError(reason: "The results export returned no answer.")
        }
        if case .object(let refusal) = fields["refusal"] {
            throw Refusal(
                code: string(refusal["code"]) ?? "refused",
                reason: string(refusal["reason"]) ?? "The export was refused.",
                repairAction: string(refusal["repairAction"]) ?? "",
                state: string(refusal["state"]).flatMap(SteerLabCLIState.init(rawValue:))
                    ?? .refused)
        }
        guard let directory = string(fields["exportDirectory"]),
            let run = string(fields["run"])
        else {
            throw ExperimentError(
                reason: "The results export answered without naming its folder.")
        }
        var files: [WrittenFile] = []
        if case .array(let entries) = fields["files"] {
            for case .object(let entry) in entries {
                guard let file = string(entry["file"]) else { continue }
                var rows: Int?
                if case .number(let count) = entry["rows"] { rows = Int(count) }
                files.append(WrittenFile(file: file, rows: rows))
            }
        }
        var notAvailable: [Unavailable] = []
        if case .array(let entries) = fields["notAvailable"] {
            for case .object(let entry) in entries {
                guard let what = string(entry["what"]) else { continue }
                notAvailable.append(
                    Unavailable(
                        what: what, why: string(entry["why"]) ?? "",
                        repair: string(entry["repair"])))
            }
        }
        return Outcome(
            directory: URL(filePath: directory), run: run,
            analysis: string(fields["analysis"]), files: files,
            notAvailable: notAvailable,
            freezeForced: fields["freezeForced"] == .bool(true), response: fields)
    }

    private static func string(_ value: JSONValue?) -> String? {
        if case .string(let text) = value { return text }
        return nil
    }

    // MARK: The app's button

    /// The run "Export Results…" would export for this selection: the selected
    /// run when it is a completed study run, otherwise nil, which asks the
    /// Python client for the newest completed run.
    public static func runToExport(selected: StudyRunListItem?) -> StudyRunListItem? {
        guard let selected, isCompletedRun(selected) else { return nil }
        return selected
    }

    /// Whether there is anything to export: the study has at least one
    /// completed run, and no export is already in progress.
    public static func canExport(runs: [StudyRunListItem], isExporting: Bool) -> Bool {
        !isExporting && runs.contains(where: isCompletedRun)
    }

    /// Why the button is unavailable, in a sentence a researcher can act on,
    /// or nil when it is available.
    public static func unavailableReason(
        runs: [StudyRunListItem], isExporting: Bool
    ) -> String? {
        if isExporting { return "An export is in progress." }
        if runs.contains(where: isCompletedRun) { return nil }
        return "This study has no completed run yet. Run it first; results can "
            + "be exported once a run has finished."
    }

    /// A study run that finished: it holds responses and wrote its report.
    static func isCompletedRun(_ item: StudyRunListItem) -> Bool {
        item.kind == .run && item.hasReport && item.generationCount > 0
    }
}

/// `steerlab-cli results export <study> [--run <run-dir>] [--out <dir>]`.
enum ResultsExportCLI {

    static let usage =
        "steerlab-cli results export <study> [--run <run-dir>] [--out <dir>]"

    static func run(
        _ invocation: ExperimentCLIInvocation, sink: ExperimentCLISink,
        workspaceRoot: URL, python: URL? = nil, source: URL? = nil
    ) async throws -> ExperimentCLIResult {
        guard invocation.verb == "export" else {
            // "verbs:" is how every family names its roster, and what the
            // runner classifies as an unknown verb rather than a failure.
            throw ExperimentError(reason: "verbs: export  (\(usage))")
        }
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
                "Name one study to export.",
                repair: usage
                    + "  (steerlab-cli experiment list shows the studies in this workspace)")
        }
        // A destination typed relative to where the command was run, not to
        // the workspace: the Python client runs in a directory of its own.
        let destination = flags["--out"].map {
            URL(
                filePath: ($0 as NSString).expandingTildeInPath,
                relativeTo: URL(filePath: FileManager.default.currentDirectoryPath)
            ).standardizedFileURL.path
        }
        do {
            let outcome = try await ResultsExport.export(
                workspaceRoot: workspaceRoot, study: study, run: flags["--run"],
                out: destination, client: .commandLine, python: python, source: source)
            for line in outcome.summaryLines { sink.out(line) }
            return ExperimentCLIResult(
                message: outcome.message, changed: true, payload: outcome.response)
        } catch let refusal as ResultsExport.Refusal {
            // A mid-verb stop prints nothing further, so the reason and the
            // repair are said here, once, for a person at a terminal.
            sink.err(
                "steerlab-cli results export: \(refusal.reason)\n"
                    + (refusal.repairAction.isEmpty ? "" : "  \(refusal.repairAction)\n"))
            throw ExperimentCLIStop(
                exitCode: 1, state: refusal.state, code: refusal.code,
                reason: refusal.reason, repairAction: refusal.repairAction)
        }
    }
}

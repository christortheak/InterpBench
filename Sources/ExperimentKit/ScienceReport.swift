import Foundation

/// The readable page for a stored scientific report.
///
/// The page is drawn in exactly one place: the shared Python client
/// (`steerlab_server/client/reports`). This type finds a page the engine
/// already wrote, or asks that owner for one through the existing bridge.
/// There is no second renderer here, so the app, a browser, and a colleague
/// all see the same file.
public enum ScienceReport {
    /// The bridge action and the `science` verb.
    public static let action = "report"
    /// The stored report a run directory holds, and the page a new job's
    /// engine writes beside it before the run is marked complete.
    public static let reportName = "assessment-report.json"
    public static let pageName = "assessment-report.html"
    /// Python twin: `science_commands.REPORT_PURPOSE`.
    public static let purpose = "Turn a stored J-lens assessment report into one self-contained HTML page a person can read. "
        + "Give the run folder or its assessment-report.json. A run folder is never written to: the page "
        + "goes to reports/ in the workspace, or to --out (read from the workspace unless absolute)."

    public struct Page: Sendable, Equatable, Identifiable {
        public let url: URL
        /// True when the engine wrote the page with the run, so it travelled
        /// with the evidence. False when it was drawn on request, outside the
        /// run directory.
        public let inRunDirectory: Bool
        public var id: String { url.path }
    }

    /// Whether a run directory holds a report a page can be drawn from.
    public static func hasReport(in runDirectory: URL) -> Bool {
        isFile(runDirectory.appending(component: reportName))
    }

    /// The page the engine wrote with the run, when there is one. Opening it
    /// needs no Python runtime at all.
    public static func storedPage(in runDirectory: URL) -> URL? {
        let page = runDirectory.appending(component: pageName)
        return hasReport(in: runDirectory) && isFile(page) ? page : nil
    }

    /// The page to show for a run: the one stored with it, or one drawn now.
    /// A run directory is never written to; a drawn page goes to `reports/`
    /// in the workspace.
    public static func page(for runDirectory: URL, root: URL,
                            python: URL? = nil, source: URL? = nil) async throws -> Page {
        if let stored = storedPage(in: runDirectory) { return Page(url: stored, inRunDirectory: true) }
        guard hasReport(in: runDirectory) else {
            throw ExperimentError.malformed(
                "This run folder holds no report that can be drawn as a page.",
                repair: "Choose a run folder that holds " + reportName + ".")
        }
        let result = try await render(path: runDirectory.path, root: root, python: python, source: source)
        guard case .object(let object) = result, case .string(let path) = object["htmlPath"] else {
            throw ExperimentError.malformed(
                "The report page was not returned.", repair: ScientificPythonRuntime.setupHint)
        }
        return Page(url: URL(filePath: path), inRunDirectory: object["inRunDirectory"] == .bool(true))
    }

    /// One bridge call. `path` is a run folder or a report file, read from the
    /// workspace unless absolute; `out`, when given, is where the page goes.
    public static func render(path: String, out: String? = nil, root: URL,
                              python: URL? = nil, source: URL? = nil) async throws -> JSONValue {
        var payload: [String: JSONValue] = ["workspaceRoot": .string(root.path), "path": .string(path)]
        if let out { payload["out"] = .string(out) }
        return try await DiagnosticWorkspace.perform(action, payload: payload, python: python, source: source)
    }

    static func run(_ arguments: DiagnosticArguments, root: URL, sink: ExperimentCLISink) async throws -> ExperimentCLIResult {
        guard let path = arguments.positional else { throw ExperimentError(reason: "Name a run folder or a report file.") }
        let result = try await render(path: path, out: arguments.flags["--out"], root: root)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        sink.out(String(decoding: try encoder.encode(result), as: UTF8.self))
        var changed = false
        var message = "The report page is ready."
        if case .object(let object) = result {
            if case .bool(let value) = object["changed"] { changed = value }
            if case .string(let text) = object["message"], case .string(let page) = object["htmlPath"] {
                message = text + " Open " + page + " in a browser."
            }
        }
        return .init(message: message, changed: changed, payload: ["response": result])
    }

    private static func isFile(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && !directory.boolValue
    }
}

extension JLensFitProvenance {
    /// One line for each text a mixed lens was fitted on, so the lens is
    /// visibly mixed wherever its fitting text is shown. Empty for a lens
    /// fitted on one text. Counts are shown as recorded; a missing count says so.
    public var corpusContributionLines: [String] {
        (corpora ?? []).map { entry in
            let fitted = entry.promptsFitted.map { $0.formatted() } ?? "an unrecorded number of"
            let considered = entry.rowsConsidered.map { $0.formatted() } ?? "an unrecorded number"
            return "\(entry.corpusSHA256 ?? "text with no recorded hash"): \(fitted) rows fitted of \(considered) considered"
        }
    }

    /// What to show as the lens's fitting text. For a mixed lens the recorded
    /// `corpus` is a combined digest, not a text, and the wording must say so.
    /// Nil when nothing is recorded, so each view keeps its own placeholder.
    public var corpusSummary: String? {
        guard let corpora, !corpora.isEmpty else { return corpus }
        return "mixed: \(corpora.count) texts (combined digest \(corpus ?? "not recorded"))"
    }
}

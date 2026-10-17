import Foundation

/// How a Mac-engine judge report, and each of its judgment rows, names the run
/// it judged under `sourceRunDirectory`.
///
/// The name is `runs/<run name>`: relative to the workspace, spelled as the
/// pipeline ledger spells the runs it names
/// (`steerlab_server/experiment/pipeline_ledger.py`), whatever the runs
/// folder is called on disk or however it is reached. Reports written before
/// this stored the run's absolute path, which carried a home folder into
/// every copy of the workspace and stopped matching once the workspace moved.
///
/// Every reader links a report to its run by the run's directory name (the
/// last path component), so the absolute form, this form, and the Python
/// engine's bare `sourceRun` name all read the same way.
enum JudgeSourceRunReference {
    /// The first component of a stored reference.
    static let runsFolder = "runs"

    /// What to store for `sourceRun`, judged into `evaluationDirectory`.
    ///
    /// `runs/<name>` when the source run sits in the folder that holds the
    /// evaluation directory, which is always the case for runs this workspace
    /// made. A source run anywhere else has no workspace-relative name, so its
    /// path is stored unchanged, as the pipeline ledger does.
    static func stored(_ sourceRun: URL, evaluationDirectory: URL) -> String {
        let folder = evaluationDirectory.standardizedFileURL.deletingLastPathComponent()
        let source = sourceRun.standardizedFileURL
        // As written first, then resolved: a runs folder reached through a
        // link must still find its runs.
        for base in [folder, folder.resolvingSymlinksInPath()] {
            for candidate in [source, source.resolvingSymlinksInPath()] {
                if let tail = relativePath(of: candidate.path, under: base.path) {
                    return runsFolder + "/" + tail
                }
            }
        }
        return sourceRun.path
    }

    private static func relativePath(of path: String, under base: String) -> String? {
        let prefix = base.hasSuffix("/") ? base : base + "/"
        guard path.hasPrefix(prefix) else { return nil }
        let tail = String(path.dropFirst(prefix.count))
        let parts = tail.split(separator: "/", omittingEmptySubsequences: false)
        guard !parts.isEmpty, !parts.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." })
        else { return nil }
        return tail
    }
}

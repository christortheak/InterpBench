import Foundation

/// Reader AND writer for the cross-engine `run-status.json` a stage writes
/// into its own run directory (Python twin:
/// `steerlab_server/experiment/run_status.py`).
///
/// The Swift writer is `Tracker` (below), wired around the local `run` and
/// `evaluate` tasks — the two local verbs that produce evidence-bearing
/// directories a gate could later be asked to trust. Remaining deliberate
/// asymmetry: local extract/validate/sweep/analyze write no status file;
/// their completion artifacts govern (exactly the legacy-directory rule),
/// and a local failure surfaces as a thrown error in the researcher's own
/// session rather than as an orphaned directory coming home from a cluster.
///
/// Retention (2026-07-24, external review
/// `docs/CLUSTER-SHARDING-JUDGING-REVIEW-2026-07-23.md`): failed cluster work
/// now comes home instead of being stranded under `/scratch`. That makes a
/// new kind of directory possible in `runs/` — one holding REAL data from a
/// stage that did not finish. The retention principle has two halves, and
/// this type is the second one:
///
///     Failure must prevent an evidentiary success claim, but it must not
///     prevent the researcher from retrieving the data and diagnostic
///     record that were actually produced.
///
/// Most gates are already safe by construction, because they require a
/// completion artifact a partial run never has (`validation-evidence.json`,
/// `judge-report.json`). `isPartial` is the explicit belt to that
/// suspenders: a gate should not have to be *accidentally* correct about
/// partial evidence.
public enum RunStatusFile {

    /// Cross-engine filename. Both engines write and read exactly this.
    public static let filename = "run-status.json"

    /// The human-readable failure note beside it.
    public static let failureNoteFilename = "FAILED.md"

    /// Malformed judge responses, one JSON object per line (Python twin:
    /// `run_status.INVALID_RESPONSES_FILENAME`).
    public static let invalidResponsesFilename = "judge-failures.jsonl"

    public struct Status: Codable, Sendable, Equatable {
        public var stage: String?
        public var status: String?
        public var error: String?
        public var errorType: String?
        /// What this stage produces one of ("judgment", "record", "cell").
        /// Stages differ in what they emit but not in what a reader needs
        /// to know — how much survived — so the count key is shared and
        /// only the label varies.
        public var itemLabel: String?
        public var itemsWritten: Int?
        public var invalidResponses: Int?
        public var pendingUnits: [String]?
        public var completedUnits: [String]?
        public var evidenceComplete: Bool?
    }

    /// What a run directory's status file says — distinguishing "there
    /// isn't one" from "there is one and it is broken", because the two
    /// mean opposite things about whether the run may be trusted.
    public enum Reading: Sendable, Equatable {
        /// No status file. A LEGACY run: unannotated, not incomplete.
        case absent
        /// Present but malformed. Something wrote it and did not finish.
        case unreadable
        case present(Status)
    }

    /// Full three-state reading. Collapsing `unreadable` into `absent` was
    /// a fail-OPEN (external review 2026-07-24, finding 4): a process
    /// killed mid-write left a torn file, which read as "no status", which
    /// read as "legacy", which read as citable — so the crash that made a
    /// run partial made it look complete.
    public static func reading(at runDirectory: URL) -> Reading {
        let url = runDirectory.appending(component: filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .absent
        }
        guard let data = try? Data(contentsOf: url),
            let status = try? JSONDecoder().decode(Status.self, from: data)
        else { return .unreadable }
        return .present(status)
    }

    /// This run's status record, or nil when the directory carries none
    /// OR carries one that cannot be parsed. Callers deciding TRUST must
    /// use `reading` or `isPartial` instead — those two cases differ.
    public static func read(at runDirectory: URL) -> Status? {
        if case .present(let status) = reading(at: runDirectory) {
            return status
        }
        return nil
    }

    /// Whether this directory is a KNOWN-incomplete record.
    ///
    /// An ABSENT status file is not partial: every run written before this
    /// contract existed carries none, and treating those as incomplete
    /// would retroactively invalidate real results.
    ///
    /// An UNREADABLE one IS partial — it fails closed. A torn status file
    /// is evidence that a writer died, which is exactly the situation the
    /// marker exists to record, and the safe reading of "I cannot tell" is
    /// "not citable".
    ///
    /// A readable status that does not say `completed` is partial,
    /// including `inProgress` — what a process killed mid-stage leaves
    /// behind, never having reached `failed`.
    public static func isPartial(at runDirectory: URL) -> Bool {
        switch reading(at: runDirectory) {
        case .absent: return false
        case .unreadable: return true
        case .present(let status): return (status.status ?? "") != "completed"
        }
    }

    /// Whether this run stopped at a CHECKPOINT rather than failing.
    ///
    /// A checkpointed run is partial — it is not complete and no gate may
    /// take it — but it is the reliability path working as designed, and
    /// surfaces differently: resumable, not broken.
    public static func isCheckpointed(at runDirectory: URL) -> Bool {
        (read(at: runDirectory)?.status ?? "") == "checkpointed"
    }

    /// One-line description of a partial run for status lines and rows,
    /// or nil when the run is not partial.
    public static func partialSummary(at runDirectory: URL) -> String? {
        if case .unreadable = reading(at: runDirectory) {
            return "run-status.json is present but unreadable — a writer "
                + "died mid-update, so this run is treated as incomplete"
        }
        guard let status = read(at: runDirectory),
            (status.status ?? "") != "completed"
        else { return nil }
        let stage = status.stage ?? "stage"
        var parts = [
            status.status == "checkpointed"
                ? "\(stage) checkpointed — resumable"
                : "\(stage) did not complete"
        ]
        if let error = status.error, !error.isEmpty {
            parts.append(error)
        }
        if let written = status.itemsWritten, written > 0 {
            parts.append("\(written) \(status.itemLabel ?? "item")(s) kept")
        }
        if let pending = status.pendingUnits, !pending.isEmpty {
            parts.append("missing: " + pending.joined(separator: ", "))
        }
        return parts.joined(separator: " — ")
    }

    // MARK: - writer

    /// The Swift half of the writer contract (2026-07-27; the header above
    /// claimed "both engines write and read exactly this" while no Swift
    /// writer existed — a failed LOCAL run/evaluate left a directory with no
    /// status, which `isPartial` read as legacy and therefore trusted).
    ///
    /// The task wrappers in `ExperimentTasks` create one Tracker per
    /// run/evaluate invocation, `begin` it when the task reports its run
    /// directory, and `finish`/`fail` it around the task body — the same
    /// lifecycle points as the server's `RunStatus` (in-progress before
    /// work, terminal status + `FAILED.md` on the way out). Field names are
    /// the Python writer's exactly; the Swift floor omits the optional
    /// per-unit detail (`expectedUnits`/`completedUnits`) and the traceback
    /// section of `FAILED.md`, which Swift does not have. Everything else
    /// is tracked for real: `itemsWritten` is refreshed per item
    /// (`noteItem`) rather than only at the terminal write, and
    /// `invalidResponses` counts actual malformed judge verdicts
    /// (`noteInvalidResponse`) rather than being stamped zero — a false
    /// measurement is worse than a declared omission, and this comment is
    /// the place the asymmetry is supposed to be legible (2026-07-27).
    ///
    /// An actor: `begin` is called from the task's `@Sendable` progress
    /// handler while `finish`/`fail` run on the task itself.
    public actor Tracker {
        private let stage: String
        private let experiment: String
        private let sourceRun: String?
        /// What this stage produces one of, singular ("record", "judgment").
        private let itemLabel: String
        /// The JSONL artifact whose non-blank line count is `itemsWritten`
        /// (the server floor's `_generation_count` rule).
        private let itemsFile: String
        private let startedAt = Date()
        private var directory: URL?
        private var finishedAt: Date?
        /// Malformed judge responses seen so far — the real count, matching
        /// the Python writer's `invalid_count`. It used to be written as a
        /// hardcoded `0`, which is not an omission but a false measurement:
        /// a reader could not distinguish "none occurred" from "not tracked
        /// on this engine" (2026-07-27).
        private var invalidResponses = 0

        public init(
            stage: String, experiment: String, sourceRun: String? = nil,
            itemLabel: String, itemsFile: String
        ) {
            self.stage = stage
            self.experiment = experiment
            self.sourceRun = sourceRun
            self.itemLabel = itemLabel
            self.itemsFile = itemsFile
        }

        /// The task created its run directory: from here on there is
        /// something on disk that must never be mistakable for a result.
        /// Idempotent; only the first directory is tracked (a task creates
        /// exactly one).
        public func begin(directoryPath: String) {
            guard directory == nil else { return }
            directory = URL(filePath: directoryPath, directoryHint: .isDirectory)
            write(status: "inProgress", error: nil, errorType: nil)
        }

        /// One item finished. Refreshes the status file so `itemsWritten` is
        /// current ON DISK, mirroring the Python writer's `note_item`.
        ///
        /// Without this the last write before a HARD stop (preemption, power
        /// loss, force-quit) was `begin`'s `inProgress, itemsWritten: 0`,
        /// beside a directory holding hundreds of real records. `isPartial`
        /// was still right — anything but `completed` is partial — but the
        /// count a reader consults to learn *how much survived* said zero in
        /// the one situation this file exists for (2026-07-27).
        public func noteItem() {
            guard directory != nil else { return }
            write(status: "inProgress", error: nil, errorType: nil)
        }

        /// One malformed judge response, recorded verbatim beside the status.
        ///
        /// Appended, never rewritten — the Python writer's rule: a retry that
        /// later succeeds still leaves its failed attempt on disk, because
        /// "the judge needed two tries" is exactly the fact a quiet retry
        /// erases. Same filename (`judge-failures.jsonl`) and same record
        /// keys, so either engine's file reads the same way.
        public func noteInvalidResponse(_ record: [String: String]) {
            guard let directory else { return }
            invalidResponses += 1
            var payload = record
            payload["at"] = String(Date().timeIntervalSince1970)
            if let data = try? JSONSerialization.data(
                withJSONObject: payload, options: [.sortedKeys]),
                var line = String(data: data, encoding: .utf8)
            {
                line += "\n"
                let url = directory.appending(
                    component: RunStatusFile.invalidResponsesFilename)
                if let handle = try? FileHandle(forWritingTo: url) {
                    // Append; best effort by design — losing the diagnostic
                    // record is bad, but letting the recorder's own failure
                    // replace the judge's real error would be worse.
                    try? handle.seekToEnd()
                    try? handle.write(contentsOf: Data(line.utf8))
                    try? handle.close()
                } else {
                    try? Data(line.utf8).write(to: url)
                }
            }
            write(status: "inProgress", error: nil, errorType: nil)
        }

        /// The task body returned. Ordinarily that means `completed` — but a
        /// COOPERATIVE CANCELLATION also returns normally (with a
        /// cancellation note in the directory and no completion artifact),
        /// and stamping that `completed` would make the partial it left look
        /// citable. A cancelled task is stamped `failed`/`Cancelled` with NO
        /// `FAILED.md`: `cancelled.txt` already is the human-readable
        /// account, and calling a deliberate stop a failure in prose would
        /// train the researcher to ignore the marker that matters (the
        /// server's checkpoint rule).
        public func finish() {
            guard let directory else { return }
            finishedAt = Date()
            let note = directory.appending(
                component: ExperimentTasks.cancellationNoteFileName)
            if FileManager.default.fileExists(atPath: note.path) {
                let text = (try? String(contentsOf: note, encoding: .utf8))?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                write(
                    status: "failed",
                    error: text ?? "cancelled by user",
                    errorType: "Cancelled")
            } else {
                write(status: "completed", error: nil, errorType: nil)
            }
        }

        /// The task body threw. Writes the terminal status AND the
        /// human-readable `FAILED.md` before the caller rethrows — the
        /// directory must carry its own account even when the error message
        /// scrolls away. No-op when no directory was ever created (a
        /// preflight refusal left nothing on disk to annotate).
        public func fail(_ error: Error) {
            guard directory != nil else { return }
            finishedAt = Date()
            let message = (error as? ExperimentError)?.reason
                ?? String(describing: error)
            let errorType = String(describing: type(of: error))
            write(status: "failed", error: message, errorType: errorType)
            writeFailureNote(error: message, errorType: errorType)
        }

        /// Non-blank line count of the stage's output artifact — the
        /// server floor's rule (`bundles._generation_count`). 0 when the
        /// file does not exist yet.
        private func itemCount() -> Int {
            guard let directory,
                let data = try? Data(
                    contentsOf: directory.appending(component: itemsFile))
            else { return 0 }
            return String(decoding: data, as: UTF8.self)
                .split(separator: "\n", omittingEmptySubsequences: true)
                .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
                .count
        }

        /// Best-effort atomic replace, mirroring the Python writer: a
        /// status write must never take down a run that is otherwise fine,
        /// and must never replace a real error with an I/O error.
        private func write(
            status statusValue: String, error: String?, errorType: String?
        ) {
            guard let directory else { return }
            var payload: [String: Any] = [
                "schemaVersion": 1,
                "stage": stage,
                "status": statusValue,
                "startedAt": startedAt.timeIntervalSince1970,
                // Only a completed stage claims complete evidence.
                "evidenceComplete": statusValue == "completed",
                "itemLabel": itemLabel,
                "itemsWritten": itemCount(),
                "invalidResponses": invalidResponses,
                "experiment": experiment,
            ]
            if let sourceRun { payload["sourceRun"] = sourceRun }
            if let finishedAt {
                payload["finishedAt"] = finishedAt.timeIntervalSince1970
            }
            if let error {
                payload["error"] = error
                payload["errorType"] = errorType ?? "Error"
            }
            guard
                let data = try? JSONSerialization.data(
                    withJSONObject: payload,
                    options: [.prettyPrinted, .sortedKeys])
            else { return }
            try? data.write(
                to: directory.appending(component: RunStatusFile.filename),
                options: .atomic)
        }

        /// The Python `_write_failure_note` wording, minus the traceback
        /// section (Swift has none to give).
        private func writeFailureNote(error: String, errorType: String) {
            guard let directory else { return }
            var lines = [
                "# \(stage) FAILED",
                "",
                "This run directory is a **failure record**, not a result. "
                    + "The data below is real and was produced before the "
                    + "failure; it is preserved so it can be inspected and "
                    + "retried, and it must not be cited as a completed "
                    + "\(stage).",
                "",
                "- **Stage:** \(stage)",
                "- **Experiment:** \(experiment)",
            ]
            if let sourceRun {
                lines.append("- **Source run:** \(sourceRun)")
            }
            lines.append("- **Error:** `\(errorType): \(error)`")
            lines.append(
                "- **\(itemLabel.capitalized)s written before the "
                    + "failure:** \(itemCount())")
            lines.append("")
            try? (lines.joined(separator: "\n") + "\n").write(
                to: directory.appending(
                    component: RunStatusFile.failureNoteFilename),
                atomically: true, encoding: .utf8)
        }
    }
}

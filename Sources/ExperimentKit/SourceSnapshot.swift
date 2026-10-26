import CryptoKit
import Foundation

/// Which records of its source run an analysis read.
///
/// A run's `generations.jsonl` is append-only, and a run parked at a
/// checkpoint gains records when it is continued. An analysis names its source
/// run (`source-run.txt`, `analysis.json`'s `sourceRun`), but a name alone
/// cannot tell an analysis of the finished run from one made while it was
/// paused: after the continuation both name the same directory. Review finding
/// (October 23, 2026): a checkpoint's analysis (6 over two pairs) was exported
/// as the completed run's (3 over four pairs), with nothing to say so.
///
/// So an analysis records the snapshot it read (`source-snapshot.json`): the
/// source's status at that moment, how many records it held, and the size and
/// SHA-256 of the exact `generations.jsonl` bytes it parsed. A reader compares
/// that with the run as it stands (`compare`) and says when they differ instead
/// of presenting the analysis as the run's. An analysis made before the stamp
/// existed is `notRecorded`: never assumed to match, and flagged only when the
/// run itself shows it was paused and continued.
///
/// Server twin: `source_snapshot` (Server/steerlab_server/experiment/
/// source_snapshot.py), the same keys, statuses, verdicts, and sentences,
/// held to one contract by `Tests/Fixtures/cross-engine/source-snapshot.json`.
/// Nothing here changes a statistic: the analysis reads the same records.
public enum SourceSnapshot {
    public static let stampFile = "source-snapshot.json"
    public static let schemaVersion = 1

    public static let completed = "completed"
    public static let noCompletionReport = "noCompletionReport"
    public static let unreadable = "unreadable"

    public struct Stamp: Codable, Sendable, Equatable {
        public var schemaVersion: Int
        public var sourceRun: String
        public var status: String
        public var records: Int
        public var generationsBytes: Int
        public var generationsSHA256: String
    }

    /// Records, size, and SHA-256 of a `generations.jsonl`'s bytes.
    public struct Description: Sendable, Equatable {
        public var records: Int
        public var generationsBytes: Int
        public var generationsSHA256: String
    }

    // MARK: - The source

    /// The run's state now: `completed`, the status its `run-status.json`
    /// records (`checkpointed`, `inProgress`, `failed`, …), `unreadable` for a
    /// torn status file, or `noCompletionReport`. Read as the server reads it:
    /// the status file as plain JSON, so both engines agree on any file.
    ///
    /// `completed` means the records are final: the completion report is
    /// written and no status says the run failed or cannot be read. A park
    /// left beside a report (`checkpointed` or `inProgress`, never settled
    /// because the run was continued before 0.9.16 settled parks) is
    /// completed too: a run writes its report only once every record is in.
    public static func status(of runDirectory: URL) -> String {
        let reported = FileManager.default.fileExists(
            atPath: runDirectory.appending(component: "report.json").path)
        switch statusObject(runDirectory) {
        case .unreadable:
            return unreadable
        case .present(let object):
            if let value = object["status"] as? String, !value.isEmpty, value != completed,
                !(reported && parkedStatuses.contains(value))
            {
                return value
            }
        case .absent:
            break
        }
        return reported ? completed : noCompletionReport
    }

    /// The statuses a parked attempt leaves (server twin:
    /// `run_status.PARKED_STATUSES`).
    static let parkedStatuses: Set<String> = ["checkpointed", "inProgress"]

    /// Whether the run was parked and then continued: its status records a
    /// settled park (`resumedFrom`), or a park was left beside the completion
    /// report (continued before parks were settled).
    public static func wasContinued(_ runDirectory: URL) -> Bool {
        guard case .present(let object) = statusObject(runDirectory) else { return false }
        if let parked = object["resumedFrom"] as? [String: Any], !parked.isEmpty { return true }
        guard let value = object["status"] as? String, parkedStatuses.contains(value) else {
            return false
        }
        return FileManager.default.fileExists(
            atPath: runDirectory.appending(component: "report.json").path)
    }

    private enum StatusObject {
        case absent, unreadable
        case present([String: Any])
    }

    private static func statusObject(_ runDirectory: URL) -> StatusObject {
        let url = runDirectory.appending(component: RunStatusFile.filename)
        guard FileManager.default.fileExists(atPath: url.path) else { return .absent }
        guard let data = try? Data(contentsOf: url),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return .unreadable }
        return .present(object)
    }

    /// Non-blank lines: the record count a snapshot states (no JSON decode,
    /// so a reader can count any run). Universal newlines, as the server.
    public static func describe(_ data: Data) -> Description {
        let text = String(decoding: data, as: UTF8.self)
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let records = text.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .count
        return Description(
            records: records, generationsBytes: data.count,
            generationsSHA256: SHA256.hash(data: data).map { String(format: "%02x", $0) }
                .joined())
    }

    /// The stamp for the exact bytes an analysis parsed. `status` is read
    /// BEFORE the bytes (`status(of:)`): a source that completes between the
    /// two reads is then stamped unfinished, never completed over records
    /// that were not all there.
    public static func stamp(runDirectory: URL, status: String, generations: Data) -> Stamp {
        let described = describe(generations)
        return Stamp(
            schemaVersion: schemaVersion,
            sourceRun: runDirectory.standardizedFileURL.lastPathComponent,
            status: status, records: described.records,
            generationsBytes: described.generationsBytes,
            generationsSHA256: described.generationsSHA256)
    }

    /// An analysis directory's stamp, or nil when it records none.
    public static func read(analysisDirectory: URL) -> Stamp? {
        (try? Data(contentsOf: analysisDirectory.appending(component: stampFile)))
            .flatMap(decode)
    }

    public static func decode(_ data: Data) -> Stamp? {
        try? JSONDecoder().decode(Stamp.self, from: data)
    }

    /// The run's `generations.jsonl` as it stands, described; nil when it
    /// cannot be read.
    public static func current(runDirectory: URL) -> Description? {
        (try? Data(contentsOf: runDirectory.appending(component: "generations.jsonl")))
            .map(describe)
    }

    // MARK: - The comparison

    public enum State: String, Sendable, Equatable {
        case matches, notRecorded, unfinishedSource, changed, notRecordedRunContinued
    }

    public struct Verdict: Sendable, Equatable {
        public var state: State
        public var analyzedRecords: Int?
        public var analyzedStatus: String?
        public var runRecords: Int?
    }

    /// The comparison of an analysis's stamp with its run as it stands. Pure,
    /// so both engines answer the shared fixture's cases from the same inputs.
    public static func verdict(stamp: Stamp?, now: Description?, continued: Bool) -> Verdict {
        guard let stamp else {
            return Verdict(state: continued ? .notRecordedRunContinued : .notRecorded)
        }
        var found = Verdict(
            state: .matches, analyzedRecords: stamp.records, analyzedStatus: stamp.status,
            runRecords: now?.records)
        if stamp.status != completed {
            found.state = .unfinishedSource
        } else if let now, now.generationsSHA256 != stamp.generationsSHA256 {
            found.state = .changed
        }
        return found
    }

    /// `verdict` for an analysis directory and the run it names.
    public static func compare(analysisDirectory: URL, runDirectory: URL) -> Verdict {
        verdict(
            stamp: read(analysisDirectory: analysisDirectory),
            now: current(runDirectory: runDirectory),
            continued: wasContinued(runDirectory))
    }

    /// Whether an analysis reads its run as it stands, by the cheapest
    /// evidence, for a page that looks at every analysis on each visit (the
    /// first-study checklist): the stamp's status, and the size of the run's
    /// records now. They only grow, so a size that moved means records were
    /// added; nothing is hashed. With no stamp, only a run that was paused and
    /// continued is in doubt (`notRecordedRunContinued`).
    public static func coversTheRun(analysisDirectory: URL, runDirectory: URL) -> Bool {
        guard let stamp = read(analysisDirectory: analysisDirectory) else {
            return !wasContinued(runDirectory)
        }
        guard stamp.status == completed else { return false }
        let generations = runDirectory.appending(component: "generations.jsonl").path
        guard
            let size = (try? FileManager.default.attributesOfItem(atPath: generations))?[.size]
                as? NSNumber
        else { return true }
        return size.intValue == stamp.generationsBytes
    }

    static func statusWords(_ state: String) -> String {
        switch state {
        case "checkpointed": return "paused at a checkpoint"
        case "inProgress": return "still running"
        case "failed": return "marked failed"
        case unreadable: return "in an unknown state (its status file could not be read)"
        case noCompletionReport: return "without a completion report"
        default: return "not complete (status: \(state))"
        }
    }

    /// What a reader is told, or nil when there is nothing to say.
    public static func sentence(_ found: Verdict, run: String) -> String? {
        let analyzed = found.analyzedRecords.map(String.init) ?? "None"
        switch found.state {
        case .unfinishedSource:
            var held = ""
            if let now = found.runRecords, now != found.analyzedRecords {
                held = ", and the run now holds \(now)"
            }
            return "This analysis read run \(run) while it was "
                + "\(statusWords(found.analyzedStatus ?? "None")), with \(analyzed) "
                + "records\(held). Re-analyze the run for effects over all its records."
        case .changed:
            return "This analysis read \(analyzed) records of run \(run), and the run's "
                + "records have changed since (it now holds "
                + "\(found.runRecords.map(String.init) ?? "None")). Re-analyze the run "
                + "for effects over its current records."
        case .notRecordedRunContinued:
            return "This analysis does not record which of run \(run)'s records it read, "
                + "and the run was paused and continued, so it may predate the "
                + "continuation. Re-analyze the run to be sure its effects cover every "
                + "record."
        case .matches, .notRecorded:
            return nil
        }
    }
}

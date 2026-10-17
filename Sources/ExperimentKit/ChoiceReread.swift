import CryptoKit
import Foundation

/// Re-reading a run's stored choices with the current choice parser.
///
/// A choice is read once, when a run writes its records (`parsedChoice` in
/// `generations.jsonl`, through `ExperimentTasks.judicialParses`). Runs are
/// immutable, so a run read by an older parser keeps that parser's mistakes,
/// and every analysis of it repeats them. `analyze --reread-choices` reads
/// every record that carries `parsedChoice` again from the reply the record
/// stored, in memory, with `Judicial.parseChoice` as this build has it
/// (`Judicial.choiceParserVersion`), and the analysis uses those readings and
/// writes `choice-reread.json` saying so. Nothing is written into a run
/// directory.
///
/// **Where each item's options come from.** Sampled records carry
/// `promptID`, not the options the item offered, so the options are read
/// from the task prompts file the run's own `experiment.json` names
/// (`taskPromptsFile` + `taskPromptsHash`): the workspace file when its
/// SHA-256 matches, otherwise a matching copy under
/// `experiments/<study>/pinned/`. A file that does not match is never used.
///
/// **Cut-off replies.** `truncated` is the record's own `finishReason`.
/// Records without one come from engines older than that field; they are read
/// as complete replies and counted separately.
///
/// Server twin: `steerlab_server/experiment/choice_reread.py`, which also
/// answers `results check-choices` for both clients. The hand-written fixture
/// `Tests/Fixtures/cross-engine/choice-reread.json` holds the two to the same
/// counts and the same effect rows.
public enum ChoiceReread {

    /// The analysis stamp's file name. Python twin: `STAMP_FILE`.
    public static let stampFile = "choice-reread.json"

    /// Records shown per kind of change. Python twin: `EXAMPLES_PER_KIND`.
    static let examplesPerKind = 5

    /// The three ways a re-read can differ from the stored reading, in the
    /// order every count and sentence lists them.
    public enum Kind: String, CaseIterable, Sendable {
        case changedToOtherOption
        case changedToNoAnswer
        case gainedAnswer
    }

    /// The analysis stamp's note. Python twin: `NOTE` (byte-identical).
    static let note =
        "Every record that carries parsedChoice was read again from its "
        + "stored reply with the choice parser this engine has now. The "
        + "re-read choices replace the stored ones in this analysis only: "
        + "every choice outcome, exclusion, and table here uses them. The run "
        + "directory, including generations.jsonl, was not changed. Records "
        + "without finishReason come from engines that did not record whether "
        + "a reply was cut off by the length limit; they were read as complete "
        + "replies."

    // MARK: - The stamp (cross-engine shape)

    /// Per-condition (and total) counts. `storedChoices` is every record that
    /// carries `parsedChoice`, a stored parse failure included; `changed` is
    /// the sum of the three kinds.
    public struct Counts: Codable, Equatable, Sendable {
        public var storedChoices = 0
        public var unchanged = 0
        public var changedToOtherOption = 0
        public var changedToNoAnswer = 0
        public var gainedAnswer = 0
        public var changed = 0
        public var noFinishReason = 0
        public var noFinishReasonChanged = 0

        public init() {}

        mutating func observe(_ kind: Kind?, finishReasonRecorded: Bool) {
            storedChoices += 1
            switch kind {
            case nil: unchanged += 1
            case .changedToOtherOption?: changedToOtherOption += 1
            case .changedToNoAnswer?: changedToNoAnswer += 1
            case .gainedAnswer?: gainedAnswer += 1
            }
            if kind != nil { changed += 1 }
            if !finishReasonRecorded {
                noFinishReason += 1
                if kind != nil { noFinishReasonChanged += 1 }
            }
        }
    }

    /// One record to look at. `sampleIndex` is omitted when the record has
    /// none; `stored` and `reread` are written as null for no answer.
    public struct Example: Codable, Equatable, Sendable {
        public let line: Int
        public let condition: String
        public let promptID: String
        public let sampleIndex: Int?
        public let stored: String?
        public let reread: String?

        enum CodingKeys: String, CodingKey {
            case line, condition, promptID, sampleIndex, stored, reread
        }

        public init(
            line: Int, condition: String, promptID: String, sampleIndex: Int?,
            stored: String?, reread: String?
        ) {
            self.line = line
            self.condition = condition
            self.promptID = promptID
            self.sampleIndex = sampleIndex
            self.stored = stored
            self.reread = reread
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(line, forKey: .line)
            try container.encode(condition, forKey: .condition)
            try container.encode(promptID, forKey: .promptID)
            try container.encodeIfPresent(sampleIndex, forKey: .sampleIndex)
            try container.encode(stored, forKey: .stored)
            try container.encode(reread, forKey: .reread)
        }
    }

    /// Where the options were read from.
    public struct TaskPrompts: Codable, Equatable, Sendable {
        public let file: String
        public let sha256: String
        public let readFrom: String
        /// `workspace` or `pinned`.
        public let location: String
    }

    /// `choice-reread.json`. Python twin: `choice_reread.stamp`.
    public struct Stamp: Codable, Equatable, Sendable {
        public let parserVersion: String
        public let sourceRun: String
        public let runUnchanged: Bool
        public let taskPrompts: TaskPrompts
        public let total: Counts
        public let conditions: [String: Counts]
        public let examples: [String: [Example]]
        public let note: String
    }

    // MARK: - Lines

    /// The non-blank lines of `generations.jsonl`, numbered from 1 — the
    /// numbering both engines use, and the one `StudyAnalysisCalculator`
    /// walks so a re-read reaches the record it was read from.
    static func lines(of text: String) -> [Substring] {
        text.split(separator: "\n").filter {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    // MARK: - Finding the options

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func relative(_ url: URL, to root: URL) -> String {
        let path = url.standardizedFileURL.path
        let base = root.standardizedFileURL.path
        if path.hasPrefix(base + "/") { return String(path.dropFirst(base.count + 1)) }
        return path
    }

    /// Likely copies first (the Python engine's mirror of the workspace path,
    /// and this engine's `task-prompts-<name>`), then every other JSONL file
    /// in the studies' `pinned/` folders.
    static func pinnedCandidates(
        workspaceRoot: URL, studies: [String], relative file: String
    ) -> [URL] {
        var ordered: [URL] = []
        let fm = FileManager.default
        for study in studies {
            let pinned = workspaceRoot.appending(components: "experiments", study, "pinned")
            guard fm.fileExists(atPath: pinned.path) else { continue }
            ordered.append(pinned.appending(path: file))
            ordered.append(
                pinned.appending(component: "task-prompts-" + (file as NSString).lastPathComponent))
            if let walker = fm.enumerator(at: pinned, includingPropertiesForKeys: nil) {
                let found = walker.compactMap { $0 as? URL }
                    .filter { $0.pathExtension == "jsonl" }
                    .sorted { $0.path < $1.path }
                ordered += found
            }
        }
        var seen = Set<String>()
        return ordered.filter { seen.insert($0.resolvingSymlinksInPath().path).inserted }
    }

    /// `(options by item id, where they were read from)` for one run, or a
    /// typed refusal with a repair when no file with the run's pinned hash
    /// can be found. Python twin: `task_prompt_options`.
    static func taskPromptOptions(
        runDirectory: URL, workspaceRoot: URL, study: String?
    ) throws -> (options: [String: [String]], source: TaskPrompts) {
        let runName = runDirectory.lastPathComponent
        guard
            let data = try? Data(contentsOf: runDirectory.appending(component: "experiment.json")),
            let settings = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "Run '\(runName)' has no readable experiment.json, so the task "
                    + "prompts it was made from, and the answer options each item "
                    + "offered, cannot be established. Its stored choices cannot be "
                    + "re-read.",
                repair: "Choose a run that holds its own experiment.json (every "
                    + "complete run does); a run copied in without it cannot be "
                    + "re-read.")
        }
        guard let file = settings["taskPromptsFile"] as? String, !file.isEmpty,
            let digest = settings["taskPromptsHash"] as? String, !digest.isEmpty
        else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "Run '\(runName)' names no pinned task prompts file (taskPromptsFile "
                    + "and taskPromptsHash in its experiment.json), so the answer "
                    + "options each item offered cannot be verified. Its stored "
                    + "choices cannot be re-read.",
                repair: "Only a run made from pinned task prompts can be re-read. "
                    + "Pin the study's task prompts before its next run.")
        }
        var studies: [String] = []
        for name in [settings["name"] as? String, study] {
            if let name, !name.isEmpty, !studies.contains(name), !name.contains("/") {
                studies.append(name)
            }
        }
        let workspaceFile =
            file.hasPrefix("/") ? URL(filePath: file) : workspaceRoot.appending(path: file)
        let candidates =
            [(workspaceFile, "workspace")]
            + pinnedCandidates(workspaceRoot: workspaceRoot, studies: studies, relative: file)
            .map { ($0, "pinned") }
        for (url, location) in candidates {
            guard let bytes = try? Data(contentsOf: url), sha256(bytes) == digest else {
                continue
            }
            var options: [String: [String]] = [:]
            for prompt in try StudyPromptParsing.parseTaskPrompts(bytes) {
                if let itemOptions = prompt.options, !itemOptions.isEmpty {
                    options[prompt.id] = itemOptions
                }
            }
            return (
                options,
                TaskPrompts(
                    file: file, sha256: digest,
                    readFrom: relative(url, to: workspaceRoot), location: location)
            )
        }
        let hint = "experiments/\(studies.first ?? "<study>")/pinned/\(file)"
        throw ExperimentError.refusing(
            .pinDrift,
            "No copy of the task prompts file run '\(runName)' was made from could "
                + "be found: \(file) with SHA-256 \(digest.prefix(12))…. The file in the "
                + "workspace is missing or has changed since the run, and no matching "
                + "copy is in the study's pinned folder. A file with other bytes might "
                + "offer other options, so the stored choices are not re-read with it.",
            repair: "Restore \(file) to the bytes the run used (for example from "
                + "version control), or put a copy with those bytes at \(hint); then "
                + "run this command again.")
    }

    // MARK: - Re-reading

    /// Which kind of change `stored` → `fresh` is, or nil when the readings
    /// agree. Python twin: `kind_of`.
    static func kind(stored: String?, fresh: String?) -> Kind? {
        if stored == fresh { return nil }
        if stored == nil { return .gainedAnswer }
        if fresh == nil { return .changedToNoAnswer }
        return .changedToOtherOption
    }

    /// The outcome of a re-read: each re-read line's new reading, and the
    /// counts and examples.
    struct Result {
        var fresh: [Int: String?] = [:]
        var total = Counts()
        var conditions: [String: Counts] = [:]
        var examples: [String: [Example]] = [:]
    }

    /// Re-read every line of `text` that carries `parsedChoice`. Refuses when
    /// a record names an item the verified file offers no options for, or
    /// carries its own `taskPromptsHash` naming another file. Python twin:
    /// `reread`.
    static func reread(
        generations text: String, options: [String: [String]], promptsHash: String?
    ) throws -> Result {
        var result = Result()
        var candidates: [Kind: [String: [Example]]] = [:]
        var unmatched: [String] = []
        for (offset, raw) in lines(of: text).enumerated() {
            let line = offset + 1
            guard
                let object = try? JSONSerialization.jsonObject(with: Data(raw.utf8)),
                let record = object as? [String: Any],
                let storedValue = record["parsedChoice"]
            else { continue }
            let promptID = (record["promptID"] as? String) ?? ""
            let ownHash = record["taskPromptsHash"] as? String
            guard let itemOptions = options[promptID],
                !(promptsHash != nil && ownHash != nil && ownHash != "" && ownHash != promptsHash)
            else {
                if !unmatched.contains(promptID) { unmatched.append(promptID) }
                continue
            }
            let reason = record["finishReason"] as? String
            let output = (record["output"] as? String) ?? ""
            let fresh = Judicial.parseChoice(
                output, options: itemOptions,
                truncated: reason.map(ExperimentTasks.FinishReason.isCutOff) ?? false)
            let stored = storedValue as? String
            result.fresh[line] = .some(fresh)
            let condition = (record["condition"] as? String) ?? ""
            let change = kind(stored: stored, fresh: fresh)
            let recorded = record["finishReason"] != nil && !(record["finishReason"] is NSNull)
            result.total.observe(change, finishReasonRecorded: recorded)
            result.conditions[condition, default: Counts()]
                .observe(change, finishReasonRecorded: recorded)
            if let change {
                var shown = candidates[change, default: [:]][condition, default: []]
                if shown.count < examplesPerKind {
                    var sampleIndex: Int?
                    if let number = record["sampleIndex"] as? NSNumber,
                        CFGetTypeID(number) != CFBooleanGetTypeID()
                    {
                        sampleIndex = number.intValue
                    }
                    shown.append(
                        Example(
                            line: line, condition: condition, promptID: promptID,
                            sampleIndex: sampleIndex, stored: stored, reread: fresh))
                    candidates[change, default: [:]][condition] = shown
                }
            }
        }
        if !unmatched.isEmpty {
            let listed = unmatched.prefix(3).map { "'\($0)'" }.joined(separator: ", ")
            let more = unmatched.count - 3
            throw ExperimentError.refusing(
                .pinDrift,
                "\(unmatched.count) item(s) with stored choices in this run (\(listed)"
                    + (more > 0 ? ", and \(more) more" : "")
                    + ") are not items with answer options in the task prompts file "
                    + "the run's settings name, or their records name a different "
                    + "prompts file. The run was probably made from a different "
                    + "prompts file, so the options those items offered cannot be "
                    + "known and nothing was re-read.",
                repair: "Re-read only runs made from the study's own pinned task "
                    + "prompts. A run made with a different prompts file has to be "
                    + "made again to be read with the current parser.")
        }
        for kind in Kind.allCases {
            let byCondition = candidates[kind] ?? [:]
            var picked: [Example] = []
            var round = 0
            while picked.count < examplesPerKind {
                var added = false
                for condition in byCondition.keys.sorted() {
                    let shown = byCondition[condition] ?? []
                    if round < shown.count, picked.count < examplesPerKind {
                        picked.append(shown[round])
                        added = true
                    }
                }
                if !added { break }
                round += 1
            }
            result.examples[kind.rawValue] = picked
        }
        return result
    }

    /// `analyze --reread-choices`: the line → new reading map the calculator
    /// substitutes, and the stamp. Python twin: `for_analysis`.
    static func forAnalysis(
        runDirectory: URL, generations: String, workspaceRoot: URL, study: String?
    ) throws -> (fresh: [Int: String?], stamp: Stamp) {
        let (options, source) = try taskPromptOptions(
            runDirectory: runDirectory, workspaceRoot: workspaceRoot, study: study)
        let result = try reread(
            generations: generations, options: options, promptsHash: source.sha256)
        return (
            result.fresh,
            Stamp(
                parserVersion: Judicial.choiceParserVersion,
                sourceRun: runDirectory.lastPathComponent, runUnchanged: true,
                taskPrompts: source, total: result.total, conditions: result.conditions,
                examples: result.examples, note: note)
        )
    }

    // MARK: - Words

    private static func plural(_ count: Int, _ noun: String) -> String {
        "\(count) \(noun)\(count == 1 ? "" : "s")"
    }

    /// One plain sentence from a stamp's own numbers, and a second when some
    /// readings rest on records that never said whether a reply was cut off.
    /// Python twin: `choice_reread.sentence` (identical words).
    public static func sentence(_ stamp: Stamp) -> String {
        sentence(parserVersion: stamp.parserVersion, total: stamp.total)
    }

    public static func sentence(parserVersion: String, total: Counts) -> String {
        var text =
            "Choices were re-read from the stored replies with choice parser "
            + "version \(parserVersion); \(total.changed) of "
            + "\(plural(total.storedChoices, "stored choice")) changed, and the run "
            + "itself is unchanged."
        if total.noFinishReason > 0 {
            text +=
                " \(total.noFinishReason) of those replies come from an older engine "
                + "that did not record whether a reply was cut off by the length "
                + "limit, so they were read as complete replies "
                + "(\(total.noFinishReasonChanged) of them changed)."
        }
        return text
    }

    /// The discoverability count for an ordinary analysis: how many stored
    /// choices in the run the current parser would read differently. Never
    /// throws; nil when the options cannot be verified or nothing differs.
    public static func advisoryCount(
        runDirectory: URL, workspaceRoot: URL, study: String?
    ) -> (changed: Int, storedChoices: Int)? {
        guard
            let text = try? String(
                contentsOf: runDirectory.appending(component: "generations.jsonl"),
                encoding: .utf8),
            let (options, source) = try? taskPromptOptions(
                runDirectory: runDirectory, workspaceRoot: workspaceRoot, study: study),
            let result = try? reread(
                generations: text, options: options, promptsHash: source.sha256),
            result.total.changed > 0
        else { return nil }
        return (result.total.changed, result.total.storedChoices)
    }

    /// The advisory's sentence. Python twin: `advisory_detail` (identical
    /// words around the asking command line's spellings).
    public static func advisoryDetail(
        changed: Int, storedChoices: Int, run: String,
        checkCommand: String, analyzeCommand: String
    ) -> String {
        "\(changed) of \(plural(storedChoices, "stored choice")) in run '\(run)' would "
            + "be read differently by the current choice parser (version "
            + "\(Judicial.choiceParserVersion)). This analysis used the choices as "
            + "the run stored them. To see which changed: \(checkCommand). To "
            + "analyze with the current reading: \(analyzeCommand). The run itself "
            + "is never changed."
    }

    /// The stamp an analysis directory holds, or nil when it re-read nothing.
    /// For readers that show it (the app's results): they read the stored
    /// stamp and recompute nothing.
    public static func stamp(inRunAt directory: URL) -> Stamp? {
        guard let data = try? Data(contentsOf: directory.appending(component: stampFile))
        else { return nil }
        return try? JSONDecoder().decode(Stamp.self, from: data)
    }
}

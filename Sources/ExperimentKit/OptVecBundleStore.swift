import Crypto
import Foundation

/// OptVec dataset bundles — the WORKSPACE-truth side of the app's OptVec
/// surface (docs/OPTVEC-OPTIMIZED-INJECTION-VECTORS-PLAN.md; bundle
/// architecture landed 2026-08-10).
///
/// An OptVec campaign is FILE-SHAPED: one folder under `prompts/optvec/`
/// holding nine hashed dataset files, `bundle.json` (the table of contents —
/// research directives + per-file SHA-256 pins), and `REPORT.md` (the human
/// QC record). Training, eval, and interpretation are Python-server-only
/// (hard requirement: no MLX path); their outputs land as immutable
/// `runs/<stamp>-optvec-*` directories that `OptVecRunStore` reads. This
/// store therefore reads the workspace (Mac = source of truth) and NEVER
/// treats server-side campaign state as canonical — live scheduler state is
/// a JOIN the panel layers on top, not a fact stored here.
///
/// ## Version contract (design the additions, not a rewrite)
///
/// - **v1 (this file): read-only.** `list()` + `check(directory:)` — the
///   Swift mirror of `steerlab-server data check optvec`
///   (`data_readiness.check_optvec_bundle`; same requirement names, status
///   vocabulary, and blocker rule, so the researcher reads both checks the
///   same way). The ONE action in the v1 surface — attaching a trained
///   OptVec artifact to a study — lives on `ExperimentStore.attachArtifact`,
///   not here: bundles are data, attach is a study-lifecycle verb.
/// - **v2 (authoring): additions only.** `save(_:)`/`update(_:)` writing
///   `bundle.json` atomically with re-computed hashes, dataset-file editors
///   writing the nine files, and a `submit` path that drives the EXISTING
///   server verbs (`optvec train/campaign` via bundle submit) — no new
///   submit machinery here. The Codable model round-trips losslessly today
///   (encode is the inverse of decode) precisely so v2 can edit-in-place
///   without a schema migration.
/// - **v3 (lifecycle): duplicate-and-adjust, never edit-after-submit.**
///   A `duplicate(name:as:)` that copies the folder and re-pins hashes;
///   REPORT.md's own rule ("any future correction must be treated as a new
///   bundle version") becomes mechanical. Submitted bundles get the same
///   treatment frozen manifests get: the store refuses in-place writes once
///   any training run pins the bundle's composite hash.
///
/// Follows the `StudyTemplateStore` arc: Codable artifact struct, stateless
/// store enum over `ExperimentStore.workspaceRoot` (so the test root
/// override reaches bundles as it reaches everything else), observable
/// state one layer up (`OptVecPanel`), views that decide nothing.
public struct OptVecDatasetBundle: Codable, Sendable, Equatable {
    /// One pinned file in the table of contents. The checker is LENIENT
    /// about the `files` map's KEYS (spec uses camelCase role keys, the
    /// server fixture uses filename keys — both validate); identity is the
    /// basename of `path`, never the dictionary key. `reader` is the
    /// three-reader firewall role ("optimizer" | "selector" | "examiner"),
    /// optional and unvalidated — surfaced in UI, never enforced here.
    public struct FileEntry: Codable, Sendable, Equatable {
        public var path: String
        public var sha256: String
        public var reader: String?

        public init(path: String, sha256: String, reader: String? = nil) {
            self.path = path
            self.sha256 = sha256
            self.reader = reader
        }
    }

    /// A research directive: the authoring spec allows a plain string or a
    /// list of strings; both decode to `lines` (and re-encode in the shape
    /// they arrived, so v2 editing never rewrites an untouched field).
    public struct Directive: Codable, Sendable, Equatable {
        public var lines: [String]
        /// True when the JSON carried a single string, not a list.
        public var wasScalar: Bool

        public init(lines: [String], wasScalar: Bool = false) {
            self.lines = lines
            self.wasScalar = wasScalar
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let scalar = try? container.decode(String.self) {
                lines = [scalar]
                wasScalar = true
            } else {
                lines = try container.decode([String].self)
                wasScalar = false
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            if wasScalar, let only = lines.first, lines.count == 1 {
                try container.encode(only)
            } else {
                try container.encode(lines)
            }
        }

        /// The server checker's emptiness rule: a non-empty string or a
        /// non-empty list passes.
        public var isEmpty: Bool {
            lines.isEmpty
                || (wasScalar
                    && lines[0].trimmingCharacters(in: .whitespaces).isEmpty)
        }
    }

    public struct Authoring: Codable, Sendable, Equatable {
        public var spec: String?
        public var qcReport: String?
    }

    public var bundle: String
    public var targetIssue: Directive?
    public var shiftDirection: Directive?
    public var caseFamilies: Directive?
    public var anchorIssues: Directive?
    public var files: [String: FileEntry]?
    /// Researcher-declared shared inputs (e.g. `neutralCorpus` naming the
    /// workspace's pinned neutral corpus path).
    public var shared: [String: String]?
    public var authoring: Authoring?

    /// Pinned entries keyed by the basename of their `path` — the identity
    /// rule the server checker uses (`_check_bundle_json`).
    public var filesByBasename: [String: FileEntry] {
        var out: [String: FileEntry] = [:]
        for entry in (files ?? [:]).values {
            let basename = (entry.path as NSString).lastPathComponent
            out[basename] = entry
        }
        return out
    }
}

public enum OptVecBundleStore {

    /// Where the bundle-authoring contract lives (named in refusal details;
    /// same constant as the server's `data_readiness.AUTHORING_SPEC`).
    public static let authoringSpec =
        "prompts/generation/COWORK-JOB-optvec-datasets.md"

    /// The eight strict choice-row files, in role order (cross-engine
    /// constant: `data_readiness.CHOICE_FILES`).
    public static let choiceFiles = [
        "target-train.jsonl", "target-val.jsonl", "target-test.jsonl",
        "anchor-train.jsonl", "anchor-val.jsonl", "anchor-test.jsonl",
        "capability-train.jsonl", "capability-eval.jsonl",
    ]

    public static let neutralFile = "neutral-fluency.jsonl"

    /// All nine bundle DATA files. `bundle.json` and `REPORT.md` are checked
    /// separately.
    public static var bundleFiles: [String] { choiceFiles + [neutralFile] }

    public static let bundleJSON = "bundle.json"
    public static let reportFile = "REPORT.md"

    /// The three research directives (four keys — issue and direction are
    /// two fields of directive 1) that must travel with the data.
    static let directiveKeys = [
        "targetIssue", "shiftDirection", "caseFamilies", "anchorIssues",
    ]

    /// Per-file A/B target-balance window (share of rows targeting the
    /// FIRST option), so a pure position preference scores zero shift.
    static let balanceLow = 0.45
    static let balanceHigh = 0.55

    /// The cross-file id-uniqueness requirement's display name.
    public static let bundleIDsRequirement = "bundle ids"

    // MARK: - Location

    /// `prompts/optvec/` in the workspace. Resolved through
    /// `ExperimentStore.workspaceRoot`, so the test root override reaches
    /// bundles as it reaches everything else.
    public static var directory: URL {
        ExperimentStore.workspaceRoot
            .appending(components: "prompts", "optvec")
    }

    // MARK: - Listing

    /// One bundle folder as found on disk. A folder with an unreadable
    /// `bundle.json` still LISTS (with `decodeFailure` set) — the check
    /// surface is where malformation is reported, and hiding the folder
    /// would make the report unreachable.
    public struct Entry: Sendable, Equatable, Identifiable {
        public var name: String
        public var directory: URL
        public var bundle: OptVecDatasetBundle?
        public var decodeFailure: String?

        public var id: String { name }
    }

    /// Every subdirectory of `prompts/optvec/` that carries (or should
    /// carry) a `bundle.json`, sorted by name. A missing `prompts/optvec/`
    /// is an empty list, not an error.
    public static func list() -> [Entry] {
        let fm = FileManager.default
        guard
            let names = try? fm.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        return names
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?
                .isDirectory == true }
            .map { url in
                let jsonURL = url.appending(component: bundleJSON)
                guard let data = try? Data(contentsOf: jsonURL) else {
                    return Entry(
                        name: url.lastPathComponent, directory: url,
                        bundle: nil, decodeFailure: nil)
                }
                do {
                    let bundle = try JSONDecoder().decode(
                        OptVecDatasetBundle.self, from: data)
                    return Entry(
                        name: url.lastPathComponent, directory: url,
                        bundle: bundle, decodeFailure: nil)
                } catch {
                    return Entry(
                        name: url.lastPathComponent, directory: url,
                        bundle: nil, decodeFailure: "\(error)")
                }
            }
            .sorted { $0.name < $1.name }
    }

    // MARK: - Data check (mirror of `steerlab-server data check optvec`)

    /// One readiness line — the `data check` vocabulary (`invalid` and
    /// `missing` block; `partial` and `present` do not). Cross-engine twin:
    /// `data_readiness.Requirement`.
    public struct Requirement: Sendable, Equatable, Identifiable {
        public enum Status: String, Sendable, Comparable {
            case invalid, missing, partial, present

            var order: Int {
                switch self {
                case .invalid: 0
                case .missing: 1
                case .partial: 2
                case .present: 3
                }
            }

            public static func < (lhs: Status, rhs: Status) -> Bool {
                lhs.order < rhs.order
            }
        }

        public var status: Status
        public var name: String
        public var detail: String
        /// Only for files that PASSED every check — this layer never hands
        /// out a paste-able pin for bytes the engine would refuse.
        public var sha256: String?
        public var rows: Int?

        public var id: String { name }
        public var blocker: Bool { status == .invalid || status == .missing }
    }

    /// Cross-engine twin: `data_readiness.BundleReport`.
    public struct Report: Sendable, Equatable {
        public var directory: URL
        public var requirements: [Requirement]

        public var blockers: [Requirement] {
            requirements.filter(\.blocker)
        }

        public var ready: Bool { blockers.isEmpty }
    }

    /// Run the OptVec dataset-bundle readiness template over `directory`.
    /// Purely local file checks — no model, no tokenizer, no network.
    /// Requirements come back blockers-first (invalid, missing, partial,
    /// present), stable within each status — the server's exact ordering.
    public static func check(directory: URL) -> Report {
        var bundleIDs: [String: String] = [:]
        var duplicates: [(id: String, first: String, second: String)] = []
        var results: [Requirement] = []
        for name in choiceFiles {
            let (result, ids) = checkChoiceFile(directory: directory, name: name)
            // The strict loader already refused WITHIN-file duplicates; any
            // id already claimed by an earlier file is a cross-file
            // duplicate.
            for rowID in ids {
                if let first = bundleIDs[rowID] {
                    duplicates.append((rowID, first, name))
                } else {
                    bundleIDs[rowID] = name
                }
            }
            results.append(result)
        }
        results.append(checkNeutralFile(directory: directory))
        results.append(checkBundleJSON(directory: directory))
        results.append(checkReportFile(directory: directory))
        results.append(checkBundleIDs(results, duplicates: duplicates))
        let ordered = results.enumerated()
            .sorted {
                ($0.element.status.order, $0.offset)
                    < ($1.element.status.order, $1.offset)
            }
            .map(\.element)
        return Report(directory: directory, requirements: ordered)
    }

    /// One choice file through the strict cross-engine loader
    /// (`SweepSelectionRule.loadChoiceRows` — the same rules the sweep's
    /// logprobShift instrument refuses on) plus the bundle-spec checks.
    /// Returns the requirement and, when the file PASSED, its item ids so
    /// the caller can check uniqueness across the whole bundle.
    static func checkChoiceFile(
        directory: URL, name: String
    ) -> (Requirement, [String]) {
        let url = directory.appending(component: name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return (
                Requirement(
                    status: .missing, name: name,
                    detail: "not found — author it per \(authoringSpec)"),
                [])
        }
        let rows: [SweepSelectionRule.ChoiceRow]
        let digest: String
        do {
            let loaded = try SweepSelectionRule.loadChoiceRows(
                file: url.path, root: directory)
            rows = loaded.rows
            digest = loaded.hash
        } catch {
            return (
                Requirement(
                    status: .invalid, name: name,
                    detail: (error as? ExperimentError)?.reason ?? "\(error)"),
                [])
        }
        let multiChar = rows.flatMap { row in
            row.options.filter { $0.count != 1 }.map { (row.id, $0) }
        }
        if let (rowID, option) = multiChar.first {
            return (
                Requirement(
                    status: .invalid, name: name,
                    detail: "row '\(rowID)' option '\(option)' is not a "
                        + "single character (\(multiChar.count) such "
                        + "option(s) in the file) — training refuses any "
                        + "option that tokenizes to more than one token; use "
                        + "bare letters, never words or \"(A)\""),
                [])
        }
        let firstOptionTargets = rows.filter { $0.target == $0.options[0] }.count
        let share = Double(firstOptionTargets) / Double(rows.count)
        guard share >= balanceLow, share <= balanceHigh else {
            return (
                Requirement(
                    status: .invalid, name: name,
                    detail: "first-option (A) target share is "
                        + "\(percent(share)) of \(rows.count) row(s) — "
                        + "outside the 45%–55% balance window; a skewed file "
                        + "lets a pure position preference read as a shift"),
                [])
        }
        return (
            Requirement(
                status: .present, name: name,
                detail: "\(rows.count) row(s), first-option targets "
                    + percent(share),
                sha256: digest, rows: rows.count),
            rows.map(\.id))
    }

    /// `neutral-fluency.jsonl` strictly to the authoring spec: one JSON
    /// object per line with a non-empty string `text`. Stricter than the
    /// eval loader on purpose (a bundle that needs the eval loader's
    /// permissiveness was not authored to spec).
    static func checkNeutralFile(directory: URL) -> Requirement {
        let url = directory.appending(component: neutralFile)
        guard let data = try? Data(contentsOf: url) else {
            return Requirement(
                status: .missing, name: neutralFile,
                detail: "not found — author it per \(authoringSpec)")
        }
        var texts = 0
        let lines = String(decoding: data, as: UTF8.self)
            .components(separatedBy: "\n")
        for (index, raw) in lines.enumerated() {
            let line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            guard
                let object = try? JSONSerialization.jsonObject(
                    with: Data(line.utf8))
            else {
                return Requirement(
                    status: .invalid, name: neutralFile,
                    detail: "line \(index + 1) is not valid JSON — the "
                        + "bundle spec pins one {\"text\": …} object per line")
            }
            guard let dict = object as? [String: Any],
                let text = dict["text"] as? String,
                !text.trimmingCharacters(in: .whitespaces).isEmpty
            else {
                return Requirement(
                    status: .invalid, name: neutralFile,
                    detail: "line \(index + 1) is not a {\"text\": …} object "
                        + "with a non-empty string text")
            }
            texts += 1
        }
        guard texts > 0 else {
            return Requirement(
                status: .invalid, name: neutralFile,
                detail: "parsed to zero texts — the fluency guard needs real "
                    + "passages")
        }
        return Requirement(
            status: .present, name: neutralFile, detail: "\(texts) text(s)",
            sha256: sha256Hex(data), rows: texts)
    }

    /// The table of contents: required, complete, and hash-true. A pinned
    /// hash that disagrees with the file's actual bytes is a stale table of
    /// contents — the precise drift `bundle.json` exists to prevent.
    static func checkBundleJSON(directory: URL) -> Requirement {
        let url = directory.appending(component: bundleJSON)
        guard let data = try? Data(contentsOf: url) else {
            return Requirement(
                status: .missing, name: bundleJSON,
                detail: "not found — the bundle's table of contents "
                    + "(directives + file pins); author it per \(authoringSpec)")
        }
        guard let object = try? JSONSerialization.jsonObject(with: data) else {
            return Requirement(
                status: .invalid, name: bundleJSON, detail: "not valid JSON")
        }
        guard let payload = object as? [String: Any] else {
            return Requirement(
                status: .invalid, name: bundleJSON,
                detail: "must be a JSON object")
        }
        // The server's emptiness rule verbatim: a directive passes as a
        // non-empty string OR a non-empty list.
        let empty = directiveKeys.filter { key in
            if let text = payload[key] as? String {
                return text.trimmingCharacters(in: .whitespaces).isEmpty
            }
            if let list = payload[key] as? [Any] {
                return list.isEmpty
            }
            return true
        }
        if !empty.isEmpty {
            return Requirement(
                status: .invalid, name: bundleJSON,
                detail: "research directive(s) absent or empty: "
                    + empty.joined(separator: ", ")
                    + " — the three directives are the bundle's science and "
                    + "must travel with the data (spec: REQUIRED INPUTS)")
        }
        guard let files = payload["files"] as? [String: Any] else {
            return Requirement(
                status: .invalid, name: bundleJSON,
                detail: "'files' must be an object pinning the nine bundle "
                    + "files")
        }
        var byBasename: [String: String] = [:]
        for entry in files.values {
            guard let dict = entry as? [String: Any],
                let path = dict["path"] as? String,
                let sha = dict["sha256"] as? String
            else { continue }
            byBasename[(path as NSString).lastPathComponent] =
                sha.trimmingCharacters(in: .whitespaces).lowercased()
        }
        let omitted = bundleFiles.filter { byBasename[$0] == nil }
        if !omitted.isEmpty {
            return Requirement(
                status: .invalid, name: bundleJSON,
                detail: "files table omits: "
                    + omitted.joined(separator: ", ")
                    + " — every bundle file is pinned in the table of contents")
        }
        var stale: [String] = []
        for name in bundleFiles {
            // A missing file's own requirement line already blocks.
            guard
                let fileData = try? Data(
                    contentsOf: directory.appending(component: name))
            else { continue }
            if sha256Hex(fileData) != byBasename[name] {
                stale.append(name)
            }
        }
        if !stale.isEmpty {
            return Requirement(
                status: .invalid, name: bundleJSON,
                detail: "pinned hash disagrees with the file's bytes for: "
                    + stale.joined(separator: ", ")
                    + " — re-hash after the last edit; a stale table of "
                    + "contents is the drift bundle.json exists to prevent")
        }
        return Requirement(
            status: .present, name: bundleJSON,
            detail: "\(bundleFiles.count) file(s) pinned, hashes agree; "
                + "directives present")
    }

    /// `REPORT.md` presence — deliberately NON-blocking either way: the
    /// report is the human half of acceptance (leakage QC, domain quality),
    /// and a mechanical check can neither validate its content nor stand in
    /// for it.
    static func checkReportFile(directory: URL) -> Requirement {
        let url = directory.appending(component: reportFile)
        guard
            let size = try? FileManager.default
                .attributesOfItem(atPath: url.path)[.size] as? Int
        else {
            return Requirement(
                status: .partial, name: reportFile,
                detail: "not found — the human half of acceptance "
                    + "(cross-split leakage QC, domain quality) is not yet "
                    + "recorded")
        }
        return Requirement(
            status: .present, name: reportFile,
            detail: "\(size ?? 0) byte(s) — content is the researcher's "
                + "read, not this check's")
    }

    /// The cross-file requirement: ids unique across the WHOLE bundle (the
    /// engine keys baselines by id and refuses cross-split duplicates).
    static func checkBundleIDs(
        _ fileResults: [Requirement],
        duplicates: [(id: String, first: String, second: String)]
    ) -> Requirement {
        let parsed = fileResults.filter {
            choiceFiles.contains($0.name) && $0.status == .present
        }
        if !duplicates.isEmpty {
            let shown = duplicates.prefix(5)
                .map { "'\($0.id)' (\($0.first) and \($0.second))" }
                .joined(separator: ", ")
            let more =
                duplicates.count > 5 ? " (+\(duplicates.count - 5) more)" : ""
            return Requirement(
                status: .invalid, name: bundleIDsRequirement,
                detail: "\(duplicates.count) id(s) duplicated across files: "
                    + shown + more + " — the engine keys baselines by id and "
                    + "refuses cross-split duplicates")
        }
        if parsed.count < choiceFiles.count {
            return Requirement(
                status: .partial, name: bundleIDsRequirement,
                detail: "uniqueness verified over \(parsed.count) of "
                    + "\(choiceFiles.count) choice files — resolve the "
                    + "blockers above to check the full bundle")
        }
        let total = parsed.compactMap(\.rows).reduce(0, +)
        return Requirement(
            status: .present, name: bundleIDsRequirement,
            detail: "\(total) id(s) unique across all \(choiceFiles.count) "
                + "choice files")
    }

    // MARK: - Helpers

    /// Python's `f"{share:.1%}"` shape ("50.0%"), so the two engines' detail
    /// lines read identically.
    static func percent(_ share: Double) -> String {
        String(format: "%.1f%%", share * 100)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

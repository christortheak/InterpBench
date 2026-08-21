import Foundation

// =============================================================================
// `catalog/` — the generated navigation overlay over the flat `runs/`
// (open-issues §20). Ported from the workspace-local `scripts/build_catalog.py`
// so ONE implementation exists: this one. The workspace copy should be retired
// by the runner once this ships.
//
// The canonical layer is `runs/`: flat, immutable, cluster-mirroring, and
// REFERENCED BY PATH from manifests and artifacts — nothing is ever re-homed.
// This derives a human-navigable view (symlinks + INDEX.md) and nothing else.
// Deterministic and idempotent: the tree is wiped and rebuilt on every
// invocation, and deleting it loses nothing.
//
// NAVIGATION ONLY. Every leaf is a symlink or the generated INDEX; no real file
// is ever written under `catalog/`, and nothing paper-relevant may live there —
// the same rule the UI layers follow. `catalog/` is gitignored (tightening 5),
// so a freeze auto-commit can never snapshot a symlink forest.
//
// WHERE THIS DIVERGES FROM THE SCRIPT (deliberate, and reported):
//
//   1. The script's `wave_of` is a ladder of literal study labels — wave
//      prefixes, family names, an era bucket. The engine is concept-agnostic,
//      so grouping keys come from each run's OWN metadata: `config.json`'s
//      `experiment`, whose leading hyphen-token is the wave. A run with no
//      readable config falls back to its directory stem. The shape of the
//      output (by-wave/<wave>/<study>/…) is unchanged; only the source of the
//      key moved from constants to data.
//   2. The script finds adapters by testing whether the study name starts with
//      a particular job-name prefix. Here adapters are found by SHAPE — any
//      submit receipt containing `run/<name>/adapter_model.safetensors` —
//      which is the property that actually matters and needs no vocabulary.
//   3. The kind/verb vocabulary is not re-listed: classification is
//      `WorkspaceImportPolicy.classify`, so the catalog and the import policy
//      can never disagree about what a directory is.
// =============================================================================

public enum WorkspaceRunCatalog {

    public static let directoryName = "catalog"
    public static let indexFileName = "INDEX.md"

    /// The `.gitignore` line the workspace must carry so a generated symlink
    /// forest never reaches a commit.
    public static let gitignoreLine = "catalog/"

    // MARK: - Rows

    /// One catalogued run directory.
    public struct Row: Sendable, Equatable {
        public var name: String
        public var kind: String
        public var wave: String
        public var study: String
        public var stamp: String
        public var fileCount: Int

        public init(
            name: String, kind: String, wave: String, study: String,
            stamp: String, fileCount: Int
        ) {
            self.name = name
            self.kind = kind
            self.wave = wave
            self.study = study
            self.stamp = stamp
            self.fileCount = fileCount
        }
    }

    public struct BuildReport: Sendable, Equatable {
        public var rows: [Row]
        public var linkCount: Int
        public var adapterCount: Int
        public var libraryCount: Int
        /// True when the rebuild appended `catalog/` to the workspace's
        /// `.gitignore` (tightening 5).
        public var gitignoreUpdated: Bool

        public init(
            rows: [Row], linkCount: Int, adapterCount: Int, libraryCount: Int,
            gitignoreUpdated: Bool
        ) {
            self.rows = rows
            self.linkCount = linkCount
            self.adapterCount = adapterCount
            self.libraryCount = libraryCount
            self.gitignoreUpdated = gitignoreUpdated
        }

        public var summary: String {
            "catalog rebuilt: \(rows.count) run director\(rows.count == 1 ? "y" : "ies"), "
                + "\(linkCount) link\(linkCount == 1 ? "" : "s"), "
                + "\(adapterCount) adapter\(adapterCount == 1 ? "" : "s"), "
                + "\(libraryCount) librar\(libraryCount == 1 ? "y" : "ies")"
        }
    }

    // MARK: - Grouping keys (data-driven — divergence 1)

    /// The study key a run is filed under: the experiment name its own
    /// `config.json` records, falling back to the directory stem when the run
    /// has no readable config (submit receipts, vector campaigns, legacy
    /// directories).
    public static func study(
        classification: WorkspaceImportPolicy.Classification,
        experimentName: String?
    ) -> String {
        guard let experimentName, !experimentName.isEmpty else {
            return classification.stem
        }
        return experimentName
    }

    /// The wave a study belongs to: the leading hyphen-token of the study key.
    /// Purely structural — whatever the researcher's naming convention is, its
    /// first segment is the coarse grouping, and the engine never has to know
    /// what the segment MEANS.
    public static func wave(forStudy study: String, kind: WorkspaceImportPolicy.DirectoryKind)
        -> String
    {
        switch kind {
        case .vectorArtifact: return "vectors"
        case .lensSupport: return "lenses"
        case .session: return "sessions"
        default: break
        }
        let head = study.split(separator: "-").first.map(String.init) ?? study
        return head.isEmpty ? "other" : head
    }

    /// The `vectors/<subkind>/` bucket a vector-artifact directory files
    /// under, from its own name prefix.
    public static func vectorSubkind(stem: String) -> String {
        if stem.hasPrefix("optvec-") { return "optvec" }
        if stem.hasPrefix("sae-feature-") { return "sae-feature" }
        if stem.hasPrefix("derived-") { return "derived" }
        return "other"
    }

    // MARK: - Build

    /// Rebuild `catalog/` under `workspaceRoot`. Idempotent: the tree is
    /// removed and regenerated, so two consecutive builds over an unchanged
    /// `runs/` produce byte-identical output.
    @discardableResult
    public static func rebuild(workspaceRoot: URL) throws -> BuildReport {
        let fm = FileManager.default
        let runsRoot = workspaceRoot.appending(component: "runs")
        var isDirectory: ObjCBool = false
        guard
            fm.fileExists(atPath: runsRoot.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw ExperimentError(
                reason: "no runs/ directory at \(runsRoot.path) — the catalog "
                    + "is an overlay over runs/, and there is nothing to "
                    + "overlay")
        }
        let catalogRoot = workspaceRoot.appending(component: directoryName)
        if fm.fileExists(atPath: catalogRoot.path) {
            try fm.removeItem(at: catalogRoot)
        }
        try fm.createDirectory(at: catalogRoot, withIntermediateDirectories: true)

        var rows: [Row] = []
        var linkCount = 0
        var adapterCount = 0

        let names = (try? fm.contentsOfDirectory(atPath: runsRoot.path))?.sorted() ?? []
        for name in names {
            let full = runsRoot.appending(component: name)
            var directory: ObjCBool = false
            guard
                fm.fileExists(atPath: full.path, isDirectory: &directory),
                directory.boolValue,
                !WorkspaceImportPolicy.librarySubtrees.contains(name)
            else { continue }
            let classification = WorkspaceImportPolicy.classify(
                directoryName: name,
                containsShardStamp: fm.fileExists(
                    atPath: full.appending(
                        component: WorkspaceImportPolicy.shardStampFileName).path))
            guard let stamp = classification.stamp else { continue }

            let studyKey = study(
                classification: classification,
                experimentName: experimentName(inRun: full))
            let waveKey = wave(forStudy: studyKey, kind: classification.kind)
            let kindKey = catalogKind(classification)

            try link(
                at: catalogRoot.appending(components: "by-kind", kindKey, name),
                to: full)
            linkCount += 1
            try link(
                at: catalogRoot.appending(
                    components: "by-wave", waveKey, studyKey, "\(kindKey)-\(stamp)"),
                to: full)
            linkCount += 1

            if classification.kind == .vectorArtifact {
                try link(
                    at: catalogRoot.appending(
                        components: "vectors",
                        vectorSubkind(stem: classification.stem), name),
                    to: full)
                linkCount += 1
            }

            // Divergence 2: adapters are found by SHAPE, not by job name.
            if classification.kind == .submit {
                for adapter in adapterDirectories(inSubmit: full) {
                    try link(
                        at: catalogRoot.appending(
                            components: "adapters", adapter.lastPathComponent),
                        to: adapter)
                    linkCount += 1
                    adapterCount += 1
                }
            }

            rows.append(
                Row(
                    name: name, kind: kindKey, wave: waveKey, study: studyKey,
                    stamp: stamp, fileCount: fileCount(under: full)))
        }

        var libraryCount = 0
        for library in WorkspaceImportPolicy.librarySubtrees.sorted() {
            let libraryURL = runsRoot.appending(component: library)
            var directory: ObjCBool = false
            guard
                fm.fileExists(atPath: libraryURL.path, isDirectory: &directory),
                directory.boolValue
            else { continue }
            try link(
                at: catalogRoot.appending(components: "libraries", library),
                to: libraryURL)
            linkCount += 1
            libraryCount += 1
        }

        try indexText(rows: rows).write(
            to: catalogRoot.appending(component: indexFileName), atomically: true,
            encoding: .utf8)

        let gitignoreUpdated = ensureGitignored(workspaceRoot: workspaceRoot)
        return BuildReport(
            rows: rows, linkCount: linkCount, adapterCount: adapterCount,
            libraryCount: libraryCount, gitignoreUpdated: gitignoreUpdated)
    }

    /// The kind label used in path components: the policy's own vocabulary,
    /// hyphenated for the filesystem.
    public static func catalogKind(
        _ classification: WorkspaceImportPolicy.Classification
    ) -> String {
        switch classification.kind {
        case .vectorArtifact: "vector-" + vectorSubkind(stem: classification.stem)
        case .shardPartial: "shard-partial"
        case .lensSupport: "lens-support"
        case .notARunDirectory: "other"
        case .unknown: "unknown"
        default: classification.kind.rawValue
        }
    }

    // MARK: - INDEX.md

    public static func indexText(rows: [Row]) -> String {
        var lines = [
            "# runs/ catalog (generated — do not edit)",
            "",
            "Rebuilt by the engine after every import "
                + "(`steerlab-cli cluster import`, and the app's import hook). "
                + "Navigation only: every entry below is a symlink into the "
                + "flat, immutable `runs/`, whose paths are load-bearing. "
                + "Deleting `catalog/` loses nothing.",
            "",
            "\(rows.count) run director\(rows.count == 1 ? "y" : "ies").",
            "",
            "| directory | kind | wave | study | files |",
            "|---|---|---|---|---|",
        ]
        for row in rows {
            lines.append(
                "| \(row.name) | \(row.kind) | \(row.wave) | \(row.study) | "
                    + "\(row.fileCount) |")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    // MARK: - Gitignore maintenance (tightening 5)

    /// Ensure the workspace's `.gitignore` covers `catalog/`. Appends when
    /// missing; never rewrites or reorders what is already there (a workspace
    /// is data, not a managed install — the same rule
    /// `WorkspaceStore.ensureAgentContract` follows). Returns whether it wrote.
    @discardableResult
    public static func ensureGitignored(workspaceRoot: URL) -> Bool {
        let url = workspaceRoot.appending(component: ".gitignore")
        let existing = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        let covered = existing
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .contains { $0 == gitignoreLine || $0 == "catalog" || $0 == "/catalog/" }
        guard !covered else { return false }
        var text = existing
        if !text.isEmpty, !text.hasSuffix("\n") { text += "\n" }
        text += gitignoreLine + "\n"
        do {
            try text.write(to: url, atomically: true, encoding: .utf8)
            return true
        } catch {
            return false
        }
    }

    // MARK: - Filesystem helpers

    /// A RELATIVE symlink, so the whole workspace stays movable. Never
    /// overwrites an existing entry (the script's `lexists` guard).
    static func link(at path: URL, to target: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard
            (try? fm.destinationOfSymbolicLink(atPath: path.path)) == nil,
            !fm.fileExists(atPath: path.path)
        else { return }
        try fm.createSymbolicLink(
            atPath: path.path,
            withDestinationPath: relativePath(from: path.deletingLastPathComponent(),
                                              to: target))
    }

    /// `../../runs/<name>` — a path from one directory to a target, computed
    /// on standardized components so the link survives the workspace moving.
    static func relativePath(from base: URL, to target: URL) -> String {
        let baseParts = base.standardizedFileURL.pathComponents
        let targetParts = target.standardizedFileURL.pathComponents
        var shared = 0
        while shared < baseParts.count, shared < targetParts.count,
            baseParts[shared] == targetParts[shared]
        {
            shared += 1
        }
        let up = Array(repeating: "..", count: baseParts.count - shared)
        let down = Array(targetParts[shared...])
        let parts = up + down
        return parts.isEmpty ? "." : parts.joined(separator: "/")
    }

    /// Recursive file count, symlinks not followed.
    static func fileCount(under directory: URL) -> Int {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
        else { return 0 }
        var count = 0
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            if values?.isRegularFile == true { count += 1 }
        }
        return count
    }

    /// Submit-receipt subdirectories holding FINAL adapter weights.
    static func adapterDirectories(inSubmit submit: URL) -> [URL] {
        let fm = FileManager.default
        let runRoot = submit.appending(component: "run")
        guard let children = try? fm.contentsOfDirectory(atPath: runRoot.path) else {
            return []
        }
        return children.sorted().compactMap { child in
            let candidate = runRoot.appending(component: child)
            let weights = candidate.appending(
                component: WorkspaceImportPolicy.adapterWeightFileName)
            return fm.fileExists(atPath: weights.path) ? candidate : nil
        }
    }

    /// The experiment name a run's canonical `config.json` records.
    static func experimentName(inRun directory: URL) -> String? {
        guard
            let data = try? Data(
                contentsOf: directory.appending(component: RunMetadata.fileName)),
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any],
            let name = dictionary["experiment"] as? String,
            !name.isEmpty
        else { return nil }
        return name
    }
}

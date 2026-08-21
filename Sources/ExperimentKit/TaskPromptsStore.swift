import Foundation

/// Which conventional prompt-set root a file sits in — and, for each, which
/// loader counts it, which pin hashes it, and the one sentence the inventory
/// row carries.
///
/// Lives beside `TaskPromptsStore` (rather than inside `DatasetInventory`,
/// where phase 3 first put it) because the store's public listing names it:
/// the difference between the two roots is a fact about the LAYOUT, not about
/// the table that renders it.
public enum PromptSetFamily: String, Sendable, CaseIterable, Identifiable, Codable {
    case task
    case dev

    public var id: String { rawValue }

    public var label: String { rawValue }

    /// The root this family owns, from the layout authority.
    public func directory(root: URL) -> URL {
        switch self {
        case .task: VectorCatalog.taskPromptsDirectory(root: root)
        case .dev: VectorCatalog.devPromptsDirectory(root: root)
        }
    }

    public var note: String {
        switch self {
        case .task:
            "measured task prompts, counted by the parser the pin uses "
                + "(ExperimentTasks.parseTaskPrompts: {\"id\", \"prompt\"} "
                + "rows, legacy {\"text\"} accepted, ids unique). Its "
                + "sha256 is what a study pins as taskPromptsHash."
        case .dev:
            "sweep dev split / robustness coherence prompts, counted by "
                + "the loader the sweep uses (StimulusSet.loadTexts: "
                + "{\"text\"} rows only). Its sha256 is what a frozen "
                + "sweep pins as devPromptsHash. A dev file is also the "
                + "engine's DEFAULT task-prompt file, so one of these may "
                + "be a study's measured task set — when a study pins one, "
                + "the row says which."
        }
    }
}

/// Every prompt-set file this workspace can honestly claim to know about
/// (WP-Data phase 4, closing phase 3's disclosed residual).
///
/// Phase 3 listed the two conventional roots and said, on every row, that a
/// manifest pins its task prompts by workspace-relative PATH — so the listing
/// was not exhaustive and the researcher had to go read the study to find the
/// file it actually measures. That disclosure was honest but unhelpful: the
/// missing rows are knowable, because the manifests are right there.
///
/// **The listing rule, exactly:**
///
/// 1. every `*.jsonl` directly inside `prompts/tasks/` (family `.task`);
/// 2. every `*.jsonl` directly inside `prompts/dev/` (family `.dev`);
/// 3. every path an `experiments/<name>/experiment.json` in this workspace
///    names as its `taskPromptsFile` — resolved by the manifest's own rule
///    (`ExperimentStore.resolveProjectPath`), wherever it lives — provided
///    the file EXISTS as a regular file;
/// 4. a file reached by both (1)/(2) and (3) is ONE record, carrying its
///    conventional family AND the pins; a file reached only by (3) is an
///    OUTLIER record with no family;
/// 5. nothing else. No directory is walked recursively, no sibling is
///    inferred, and a pinned path that is missing on disk is not listed —
///    the inventory reports what exists (a pin whose bytes moved is a
///    `verify()` violation, which is where it is meant to surface).
///
/// `pinnedBy` names the experiments, sorted, so a row can say *which* study
/// measures it rather than only that some study might.
public enum TaskPromptsStore {

    /// One prompt-set file the workspace holds or pins.
    public struct Record: Sendable, Equatable, Identifiable {
        public let url: URL
        /// Workspace-relative where the file is inside the root; the absolute
        /// path otherwise (a manifest may pin an absolute path).
        public let relativePath: String
        /// The conventional root it sits in, or nil for a manifest-named
        /// outlier that lives outside both.
        public let family: PromptSetFamily?
        /// Experiments whose manifest pins this file as `taskPromptsFile`,
        /// sorted. Empty is the common case.
        public let pinnedBy: [String]

        public var id: String {
            url.standardizedFileURL.resolvingSymlinksInPath().path
        }

        public init(
            url: URL, relativePath: String, family: PromptSetFamily?,
            pinnedBy: [String]
        ) {
            self.url = url
            self.relativePath = relativePath
            self.family = family
            self.pinnedBy = pinnedBy
        }

        /// The FAMILY whose loader should read this file. An outlier is a
        /// study's measured task prompts by definition — that is the only way
        /// it got here — so it is read as a task set.
        public var readingFamily: PromptSetFamily { family ?? .task }

        /// The sub-family label the inventory shows.
        public var familyLabel: String { family?.label ?? "pinned" }

        /// The per-row sentence: the family's own note, plus who pins it.
        public var note: String {
            var parts = [readingFamily.note]
            if let pins = pinsSentence { parts.append(pins) }
            if family == nil {
                parts.append(
                    "Filed outside prompts/tasks/ and prompts/dev/ — listed "
                        + "because a manifest names it, not because the "
                        + "directory is scanned.")
            }
            return parts.joined(separator: " ")
        }

        /// "Pinned as the task prompts of …" — nil when nothing pins it.
        public var pinsSentence: String? {
            guard !pinnedBy.isEmpty else { return nil }
            return "Pinned as the task prompts of "
                + pinnedBy.joined(separator: ", ") + "."
        }
    }

    /// The workspace's prompt-set files, by the rule in the type comment.
    /// Pure and synchronous; sorted by relative path so a scan is stable.
    public static func list(root: URL = VectorCatalog.projectRoot) -> [Record] {
        let pins = taskPromptPins(root: root)

        // Insertion order is the two conventional roots first; outliers are
        // appended and the whole list is sorted at the end anyway.
        var byResolvedPath: [String: Record] = [:]
        var order: [String] = []

        func add(url: URL, family: PromptSetFamily?) {
            let key = url.standardizedFileURL.resolvingSymlinksInPath().path
            guard byResolvedPath[key] == nil else { return }
            order.append(key)
            byResolvedPath[key] = Record(
                url: url,
                relativePath: DatasetCreationPlanner.relativePath(of: url, root: root),
                family: family,
                pinnedBy: pins[key] ?? [])
        }

        for family in PromptSetFamily.allCases {
            for url in jsonlFiles(in: family.directory(root: root)) {
                add(url: url, family: family)
            }
        }
        for key in pins.keys.sorted() {
            guard byResolvedPath[key] == nil else { continue }
            let url = URL(filePath: key)
            guard isRegularFile(url) else { continue }
            add(url: url, family: nil)
        }

        return order.compactMap { byResolvedPath[$0] }
            .sorted { $0.relativePath < $1.relativePath }
    }

    /// Resolved-path → the experiments pinning it as `taskPromptsFile`.
    ///
    /// Manifests are read with a MINIMAL decode rather than
    /// `ExperimentStore.load`: a legacy or partially-hand-edited manifest
    /// that fails a full decode would silently drop its pinned file from the
    /// listing, which is exactly the invisibility this store exists to end.
    /// The key it reads is the manifest's own (`taskPromptsFile`).
    static func taskPromptPins(root: URL) -> [String: [String]] {
        struct Peek: Decodable {
            let name: String?
            let taskPromptsFile: String?
        }
        let fm = FileManager.default
        let experiments = root.appending(component: "experiments")
        guard
            let directories = try? fm.contentsOfDirectory(
                at: experiments, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [:] }

        var pins: [String: [String]] = [:]
        for directory in directories {
            guard
                (try? directory.resourceValues(forKeys: [.isDirectoryKey]).isDirectory)
                    == true,
                let data = try? Data(
                    contentsOf: directory.appending(component: "experiment.json")),
                let peek = try? JSONDecoder().decode(Peek.self, from: data),
                let file = peek.taskPromptsFile?.trimmingCharacters(
                    in: .whitespacesAndNewlines),
                !file.isEmpty
            else { continue }
            let url = ExperimentStore.resolveProjectPath(file, root: root)
            let key = url.standardizedFileURL.resolvingSymlinksInPath().path
            let experiment = peek.name ?? directory.lastPathComponent
            pins[key, default: []].append(experiment)
        }
        return pins.mapValues { Array(Set($0)).sorted() }
    }

    /// Sorted `*.jsonl` files directly inside one directory. Non-recursive on
    /// purpose: these roots are flat by convention, and descending into
    /// subdirectories would be inventing a layout.
    static func jsonlFiles(in directory: URL) -> [URL] {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])) ?? []
        return contents
            .filter { $0.pathExtension.lowercased() == "jsonl" }
            .filter(isRegularFile)
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func isRegularFile(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
    }
}

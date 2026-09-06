import Foundation

/// The exact source bytes shown by a prompt editor, with an explicit workspace.
/// This review is session metadata, never part of a study's scientific identity.
public struct TaskPromptsFileReview: Sendable {
    public let workspaceRoot: URL
    public let path: String
    public let file: ManifestFileSnapshot

    public init(path: String, workspaceRoot: URL) throws {
        self.workspaceRoot = workspaceRoot.standardizedFileURL
        self.path = path
        file = try ManifestFileTransaction.snapshot(
            at: TaskPromptsAuthoring.resolve(path, workspaceRoot: workspaceRoot))
        _ = try TaskPromptsDocument.load(file.data)
    }

    fileprivate init(path: String, workspaceRoot: URL, data: Data) {
        self.workspaceRoot = workspaceRoot.standardizedFileURL
        self.path = path
        file = ManifestFileSnapshot(data: data)
    }
}

/// Publishes edited full records as an immutable content-addressed input, then
/// pins those exact bytes in the reviewed draft. Existing input files are never
/// replaced: another study or run may still depend on them. A failed publication
/// can leave an unreferenced prepared input; it cannot change the old input or pin.
public enum TaskPromptsAuthoring {
    public struct Result: Sendable {
        public let study: DraftAuthoringSnapshot
        public let prompts: TaskPromptsFileReview
        public let changed: Bool
    }

    public static func save(
        reviewed: DraftAuthoringSnapshot, path: String,
        source: TaskPromptsFileReview?, editorText: String
    ) throws -> Result {
        let root = reviewed.workspaceRoot
        let storage = ExperimentRepository(workspaceRoot: root)
        return try ManifestFileTransaction.withLock(
            manifestURL: storage.manifestURL(reviewed.manifest.name), workspaceRoot: root
        ) {
            try ManifestFileTransaction.requireCurrent(
                .sha256(reviewed.file.sha256), at: storage.manifestURL(reviewed.manifest.name))
            try ManifestMutationPolicy.admitDraftEdit(reviewed.manifest)
            let url = try resolve(path, workspaceRoot: root)
            let data = try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: root) {
                let document: TaskPromptsDocument
                if let source {
                    guard source.path == path,
                        try ManifestFileTransaction.canonicalPath(source.workspaceRoot)
                            == ManifestFileTransaction.canonicalPath(root)
                    else { throw inputChanged() }
                    guard let current = try? Data(contentsOf: url), current == source.file.data
                    else { throw inputChanged() }
                    document = try TaskPromptsDocument.load(source.file.data)
                } else {
                    // A textarea with no loaded review cannot borrow fresh file
                    // metadata at save time and thereby authorize an old edit.
                    guard !FileManager.default.fileExists(atPath: url.path) else {
                        throw inputChanged()
                    }
                    document = TaskPromptsDocument.fromTexts([])
                }
                let blocks = TaskPromptsDocument.editorBlocks(editorText)
                guard !blocks.isEmpty else {
                    throw ExperimentError.refusing(.missingPrerequisite, "Add at least one prompt.",
                        repair: "Enter prompt text, separating prompts with a line containing only ---.")
                }
                return document.applyingEditedTexts(blocks).serialized()
            }
            return try publish(data, reviewed: reviewed)
        }
    }

    /// Import full JSONL records without replacing any previously pinned input.
    /// A new content-addressed version is prepared and pinned in the reviewed draft.
    public static func importJSONL(_ text: String, reviewed: DraftAuthoringSnapshot) throws -> Result {
        let text = TaskPromptsImport.normalizedLineEndings(text)
        switch TaskPromptsImport.preview(text) {
        case .empty: throw ExperimentError(reason: "nothing to import — no non-empty lines")
        case .failure(let line, let reason):
            throw ExperimentError(reason: "refusing to import: line \(line) — \(reason)")
        case .preview: break
        }
        let document = try TaskPromptsDocument.load(Data(text.utf8))
        return try publish(document.serialized(), reviewed: reviewed)
    }

    private static func publish(_ data: Data, reviewed: DraftAuthoringSnapshot) throws -> Result {
        let root = reviewed.workspaceRoot
        let storage = ExperimentRepository(workspaceRoot: root)
        return try ManifestFileTransaction.withLock(
            manifestURL: storage.manifestURL(reviewed.manifest.name), workspaceRoot: root
        ) {
            try ManifestFileTransaction.requireCurrent(.sha256(reviewed.file.sha256),
                at: storage.manifestURL(reviewed.manifest.name))
            try ManifestMutationPolicy.admitDraftEdit(reviewed.manifest)
            let hash = ManifestFileTransaction.digest(data)
            let outputPath = "prompts/tasks/versions/\(hash).jsonl"
            let outputURL = try resolve(outputPath, workspaceRoot: root)
            var manifest = reviewed.manifest
            try ExperimentStore.pinTaskPrompts(outputPath, data: data, into: &manifest)
            return try ManifestFileTransaction.withLock(manifestURL: outputURL, workspaceRoot: root) {
                let existed = FileManager.default.fileExists(atPath: outputURL.path)
                if existed {
                    guard try Data(contentsOf: outputURL) == data else { throw inputChanged() }
                } else {
                    try FileManager.default.createDirectory(
                        at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try data.write(to: outputURL, options: .atomic)
                }
                let saved = try DraftAuthoringTransaction.replace(manifest, reviewed: reviewed)
                return Result(study: saved,
                    prompts: TaskPromptsFileReview(path: outputPath, workspaceRoot: root, data: data),
                    changed: !existed || saved.file.data != reviewed.file.data)
            }
        }
    }

    /// Resolve symlinks as well as lexical traversal; publication never follows
    /// a workspace directory symlink into another workspace or runs directory.
    static func resolve(_ path: String, workspaceRoot: URL) throws -> URL {
        guard !path.isEmpty, !(path as NSString).isAbsolutePath else {
            throw VectorCatalog.PathError.outsideProjectRoot(path)
        }
        let root = try ManifestFileTransaction.canonicalPath(workspaceRoot)
        let candidate = try ManifestFileTransaction.canonicalPath(workspaceRoot.appending(path: path))
        guard candidate.hasPrefix(root + "/") else {
            throw VectorCatalog.PathError.outsideProjectRoot(path)
        }
        let relative = String(candidate.dropFirst(root.count + 1))
        guard relative != "runs", !relative.hasPrefix("runs/") else {
            throw ExperimentError.refusing(.missingPrerequisite, "Run artifacts are immutable.",
                repair: "Choose a prompt source outside runs; save edits as a new draft input.")
        }
        return URL(fileURLWithPath: candidate)
    }

    private static func inputChanged() -> ExperimentError {
        ExperimentError.refusing(.staleManifest,
            "The prompt source is unreviewed or changed after it was loaded; no edit was published.",
            repair: "Use Load Prompts to review the current source, then apply the edit again.")
    }
}

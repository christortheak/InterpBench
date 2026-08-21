import Foundation

/// Assembles the inline document text a server LoRA training job uploads,
/// from an adapter's STORED training-data path. Two rules make this honest:
///
/// 1. **Stored paths resolve through `FineTuneStore.absoluteURL`, never the
///    process working directory.** Adapter sidecars store workspace-relative
///    paths for anything inside the workspace; reading them with
///    `URL(filePath:)` silently resolves against the CWD (the code checkout
///    when launched from Xcode) — the bug that made a correctly saved
///    workspace folder read as "empty".
/// 2. **The folder model is first-class.** Researchers select training
///    FOLDERS; a path that resolves to a directory gathers its text-like
///    files (recursively, hidden files skipped, sorted by relative path —
///    the same enumeration and ordering conventions as
///    `FineTuneStore.hashFileOrDirectory`, so the bytes sent and the folder
///    hash pinned describe the same file set in the same order) and
///    concatenates them with per-document separators. A single file keeps
///    working. Failures name the RESOLVED path and what was wrong.
///
/// File semantics match the local trainer (`FineTuneTrainer.loadDataset`):
/// `.json`/`.jsonl` rows are parsed to their text fields (raw JSON syntax is
/// never sent as training text), `.txt`/`.md` are read whole, `.pdf` text is
/// extracted, and README-prefixed files are skipped — so a folder trains the
/// same corpus locally and on the server. Unsupported extensions are
/// excluded from the text but still covered by the folder hash, which pins
/// the whole directory.
///
/// **Two payloads, one selection model (2026-08-12,
/// `docs/CLUSTER-LORA-READINESS.md` §2.1).** `inlineText` is the LEGACY,
/// lossy payload described above — it is what servers predating explicit
/// splits accept, and it stays the honest route for mixed-document corpora.
/// `structuredPayload` is the evidence path: it walks the training AND
/// validation folders in the same order, and uploads each `.jsonl` file
/// separately with its raw bytes and their SHA-256 — no concatenation, no
/// document headers, no row flattening, so example boundaries and the frozen
/// split survive the wire.
public enum FineTuneTrainingData {

    /// The inline text payload plus the provenance the submission note
    /// reports (how many files, from where).
    public struct Payload: Sendable, Equatable {
        public var text: String
        /// Number of files that contributed text (1 for a single file).
        public var fileCount: Int
        /// The resolved absolute path the text came from.
        public var resolvedPath: String
        /// Contributing files, folder-relative, in the order concatenated.
        public var sources: [String]
    }

    /// Plain-language refusal: names the resolved path and the remedy.
    ///
    /// `kind` exists so a CALLER can tell "this selection is not a structured
    /// train/validation JSONL dataset at all" (fall back to the legacy inline
    /// corpus, which is what mixed documents are for) from "it is meant to be
    /// one and it is broken" (refuse loudly — never silently downgrade an
    /// evidence-shaped dataset to the lossy path).
    public struct Problem: Error, CustomStringConvertible, Equatable, Sendable {
        public enum Kind: String, Sendable, Equatable {
            /// Nothing chosen, missing on disk, empty, or unreadable.
            case unusable
            /// Readable, but not a JSONL-only dataset — mixed documents.
            case notStructured
            /// Training side is fine; the held-out validation side is not.
            case missingValidation
        }

        public var message: String
        public var kind: Kind

        public init(_ message: String, kind: Kind = .unusable) {
            self.message = message
            self.kind = kind
        }
        public var description: String { message }
    }

    /// Extensions treated as training text sources — the same set the local
    /// trainer loads (`FineTuneTrainer.isSupportedTrainingSource`).
    public static let textLikeExtensions: Set<String> = [
        "txt", "md", "json", "jsonl", "pdf",
    ]

    /// Separator line written before each document in a folder payload.
    public static func documentHeader(_ relativeName: String) -> String {
        "===== document: \(relativeName) ====="
    }

    /// Resolve the stored training-data path and build the inline payload.
    ///
    /// - A stored path ending in `defaultFilename` (the legacy picker
    ///   convention that recorded `<folder>/train.jsonl`) gathers from the
    ///   CONTAINING folder — mirroring `FineTuneTrainer.loadDataset`, which
    ///   reads the primary file plus its siblings.
    /// - A directory gathers its text-like files; an empty or unreadable
    ///   selection refuses with the resolved path in the message.
    public static func inlineText(
        storedPath: String,
        defaultFilename: String = "train.jsonl"
    ) throws -> Payload {
        let trimmed = storedPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw Problem(
                "no training data is set for this adapter — choose a training "
                    + "data folder (or file) in the Training data row first")
        }
        let resolved = resolveSelection(
            storedPath: trimmed, defaultFilename: defaultFilename)
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(
            atPath: resolved.path, isDirectory: &isDirectory)
        else {
            throw Problem(
                "the training data path \(resolved.path) does not exist — "
                    + "re-choose the training data folder for this adapter")
        }
        if isDirectory.boolValue {
            return try folderPayload(resolved)
        }
        guard let text = fileText(resolved) ?? (try? String(contentsOf: resolved, encoding: .utf8))
        else {
            throw Problem(
                "the training data file \(resolved.path) could not be read as text")
        }
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else {
            throw Problem(
                "the training data file \(resolved.path) is empty — add training "
                    + "text before sending it to the server")
        }
        return Payload(
            text: clean,
            fileCount: 1,
            resolvedPath: resolved.path,
            sources: [resolved.lastPathComponent])
    }

    private static func folderPayload(_ folder: URL) throws -> Payload {
        let files = orderedTextLikeFiles(in: folder)
        guard !files.isEmpty else {
            throw Problem(
                "the training folder \(folder.path) is empty — add .txt, .md, "
                    + ".json, .jsonl, or .pdf files (drop them onto the "
                    + "Training data row) before training")
        }
        var sections: [(relative: String, text: String)] = []
        for file in files {
            guard
                let text = fileText(file.url)?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                !text.isEmpty
            else { continue }
            sections.append((file.relative, text))
        }
        guard !sections.isEmpty else {
            throw Problem(
                "the training folder \(folder.path) has "
                    + "\(files.count) file\(files.count == 1 ? "" : "s") but none "
                    + "contained readable training text")
        }
        let text = sections
            .map { "\(documentHeader($0.relative))\n\n\($0.text)" }
            .joined(separator: "\n\n")
        return Payload(
            text: text,
            fileCount: sections.count,
            resolvedPath: folder.path,
            sources: sections.map(\.relative))
    }

    /// Resolve a STORED training/validation path to the selection on disk.
    /// A path ending in `defaultFilename` whose parent is a directory
    /// resolves to that FOLDER — the legacy picker convention recorded
    /// `<folder>/train.jsonl` while the folder was the real selection, and
    /// the folder is also what `FineTuneStore.hashFileOrDirectory` pins.
    static func resolveSelection(storedPath: String, defaultFilename: String) -> URL {
        let resolved = FineTuneStore.absoluteURL(storedPath).standardizedFileURL
        guard resolved.lastPathComponent == defaultFilename else { return resolved }
        let parent = resolved.deletingLastPathComponent()
        var parentIsDirectory = ObjCBool(false)
        if FileManager.default.fileExists(
            atPath: parent.path, isDirectory: &parentIsDirectory),
            parentIsDirectory.boolValue
        {
            return parent
        }
        return resolved
    }

    /// Recursive, hidden-skipping, regular-file enumeration sorted by
    /// relative path — deliberately the same conventions as
    /// `FineTuneStore.hashFileOrDirectory` (so the bytes sent and the folder
    /// hash pinned describe the same file set in the same order), minus
    /// README-prefixed files (which the local trainer also skips).
    static func orderedFiles(in folder: URL) -> [(relative: String, url: URL)] {
        guard
            let enumerator = FileManager.default.enumerator(
                at: folder,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        var entries: [(String, URL)] = []
        // Symlink-resolve both sides of the prefix strip: directory
        // enumerators hand back real paths (/private/var/…) even when the
        // root was given in symlinked form (/var/…).
        let base = folder.resolvingSymlinksInPath().path
        for case let url as URL in enumerator {
            guard
                (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile)
                    == true,
                !url.lastPathComponent.lowercased().hasPrefix("readme")
            else { continue }
            let filePath = url.resolvingSymlinksInPath().path
            let relative = filePath.hasPrefix(base + "/")
                ? String(filePath.dropFirst(base.count + 1))
                : url.lastPathComponent
            entries.append((relative, url))
        }
        return entries.sorted { $0.0 < $1.0 }.map { (relative: $0.0, url: $0.1) }
    }

    /// `orderedFiles` restricted to the extensions the local trainer loads
    /// as training text — the legacy inline path's file set.
    static func orderedTextLikeFiles(in folder: URL) -> [(relative: String, url: URL)] {
        orderedFiles(in: folder).filter {
            textLikeExtensions.contains($0.url.pathExtension.lowercased())
        }
    }

    /// Text for one file, by extension — the local trainer's semantics
    /// (`FineTuneTrainer.loadExamples`): structured rows parsed to their
    /// text fields, plain text read whole, PDF extracted. nil = unreadable
    /// or unsupported.
    static func fileText(_ url: URL) -> String? {
        switch url.pathExtension.lowercased() {
        case "json", "jsonl":
            guard let content = try? String(contentsOf: url, encoding: .utf8) else {
                return nil
            }
            let rows = FineTuneTrainer.parseStructuredExamples(content)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            return rows.isEmpty ? "" : rows.joined(separator: "\n\n")
        case "txt", "md":
            return try? String(contentsOf: url, encoding: .utf8)
        case "pdf":
            return FineTuneTrainer.extractPDFText(url)
        default:
            return nil
        }
    }

    // MARK: - Structured upload (explicit train/validation splits)

    /// Which half of the frozen split a file belongs to. The raw values are
    /// the wire spelling (`dataset.files[].role`) — see the LoRA readiness
    /// contract §6.
    public enum SplitRole: String, Sendable, Equatable, Codable {
        case train
        case validation
    }

    /// One uploaded dataset file, byte-faithful. `content` is the file's
    /// bytes decoded as UTF-8 and NOT otherwise touched — no trimming, no
    /// row parsing, no document headers — and `sha256` is the digest of
    /// those same raw bytes (`FineTuneStore.hashFile`), so the server can
    /// verify what it received against what the workspace pinned.
    public struct StructuredFile: Sendable, Equatable {
        public var role: SplitRole
        /// Workspace-relative where the file lives inside the workspace,
        /// absolute otherwise (`FineTuneStore.relativePath`).
        public var path: String
        public var content: String
        public var sha256: String

        public init(role: SplitRole, path: String, content: String, sha256: String) {
            self.role = role
            self.path = path
            self.content = content
            self.sha256 = sha256
        }
    }

    /// Both selected folders, walked in `orderedFiles` order (train first,
    /// then validation) with example boundaries intact. This is the payload
    /// the v2 fine-tune request carries; the legacy `inlineText` payload —
    /// which concatenates everything into one text stream and flattens JSONL
    /// rows to their text fields — remains for mixed-document corpora and
    /// for servers that predate explicit splits.
    public struct StructuredPayload: Sendable, Equatable {
        public var files: [StructuredFile]
        public var trainingResolvedPath: String
        public var validationResolvedPath: String

        public var trainFiles: [StructuredFile] { files.filter { $0.role == .train } }
        public var validationFiles: [StructuredFile] {
            files.filter { $0.role == .validation }
        }
        public var totalBytes: Int {
            files.reduce(0) { $0 + $1.content.utf8.count }
        }
    }

    /// Extensions eligible for structured upload. Deliberately only JSONL:
    /// a structured dataset is rows, and a `.txt`/`.pdf` in the folder means
    /// the selection is a document corpus, not a frozen split.
    public static let structuredExtensions: Set<String> = ["jsonl"]

    /// Build the structured (unflattened) upload payload for BOTH selected
    /// folders.
    ///
    /// Refusals name the resolved path and carry a `Problem.Kind` so the
    /// caller can distinguish a mixed-document selection (`.notStructured` —
    /// legacy inline is the honest route for it) from a structured dataset
    /// that is broken or missing its held-out half (`.missingValidation` /
    /// `.unusable` — those must never be silently downgraded).
    public static func structuredPayload(
        trainingPath: String,
        validationPath: String,
        trainingDefaultFilename: String = "train.jsonl",
        validationDefaultFilename: String = "validation.jsonl"
    ) throws -> StructuredPayload {
        let train = try structuredSplit(
            storedPath: trainingPath,
            role: .train,
            defaultFilename: trainingDefaultFilename)
        let validation = try structuredSplit(
            storedPath: validationPath,
            role: .validation,
            defaultFilename: validationDefaultFilename)
        guard train.resolvedPath != validation.resolvedPath else {
            throw Problem(
                "the training and validation selections both resolve to "
                    + "\(train.resolvedPath) — held-out validation must be a "
                    + "separate folder, or the split measures nothing",
                kind: .missingValidation)
        }
        return StructuredPayload(
            files: train.files + validation.files,
            trainingResolvedPath: train.resolvedPath,
            validationResolvedPath: validation.resolvedPath)
    }

    private static func structuredSplit(
        storedPath: String,
        role: SplitRole,
        defaultFilename: String
    ) throws -> (files: [StructuredFile], resolvedPath: String) {
        let label = role == .train ? "training" : "validation"
        let brokenKind: Problem.Kind = role == .train ? .unusable : .missingValidation
        let trimmed = storedPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw Problem(
                "no \(label) data is set for this adapter — an explicit "
                    + "\(label) folder is required for split-honest training "
                    + "(the server no longer invents a validation split); "
                    + "choose one in the \(label.capitalized) data row",
                kind: brokenKind)
        }
        let resolved = resolveSelection(
            storedPath: trimmed, defaultFilename: defaultFilename)
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(
            atPath: resolved.path, isDirectory: &isDirectory)
        else {
            throw Problem(
                "the \(label) data path \(resolved.path) does not exist — "
                    + "re-choose the \(label) data folder for this adapter",
                kind: brokenKind)
        }

        let candidates: [(relative: String, url: URL)]
        if isDirectory.boolValue {
            candidates = orderedFiles(in: resolved)
            guard !candidates.isEmpty else {
                throw Problem(
                    "the \(label) folder \(resolved.path) is empty — add "
                        + ".jsonl rows before training",
                    kind: brokenKind)
            }
        } else {
            candidates = [(resolved.lastPathComponent, resolved)]
        }

        let offenders = candidates
            .filter { !structuredExtensions.contains($0.url.pathExtension.lowercased()) }
            .map(\.relative)
        guard offenders.isEmpty else {
            let shown = offenders.prefix(5).joined(separator: ", ")
            let more = offenders.count > 5 ? " (and \(offenders.count - 5) more)" : ""
            throw Problem(
                "the \(label) selection \(resolved.path) contains "
                    + "non-.jsonl file\(offenders.count == 1 ? "" : "s"): "
                    + "\(shown)\(more) — structured upload sends JSONL rows "
                    + "unchanged, so a mixed-document corpus must use the "
                    + "legacy inline route instead",
                kind: .notStructured)
        }

        // Workspace-relative paths are built from the SELECTION's own URL
        // plus the enumerator's relative key — never from the enumerated
        // absolute path. Enumerators hand back real paths (/private/var/…)
        // while the workspace root may be held in symlinked form (/var/…),
        // and hashing that mismatch into `path` would silently send absolute
        // host paths where the contract wants workspace-relative ones.
        let selectionPath = FineTuneStore.relativePath(for: resolved)
        var files: [StructuredFile] = []
        for candidate in candidates {
            guard let data = try? Data(contentsOf: candidate.url) else {
                throw Problem(
                    "the \(label) file \(candidate.url.path) could not be read",
                    kind: brokenKind)
            }
            guard let content = String(data: data, encoding: .utf8) else {
                throw Problem(
                    "the \(label) file \(candidate.url.path) is not valid "
                        + "UTF-8 — JSONL rows must be UTF-8 text",
                    kind: brokenKind)
            }
            guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw Problem(
                    "the \(label) file \(candidate.url.path) is empty — "
                        + "remove it or add rows before training",
                    kind: brokenKind)
            }
            guard let sha256 = FineTuneStore.hashFile(candidate.url) else {
                throw Problem(
                    "the \(label) file \(candidate.url.path) could not be "
                        + "hashed — the server verifies uploaded bytes "
                        + "against this digest",
                    kind: brokenKind)
            }
            files.append(
                StructuredFile(
                    role: role,
                    path: isDirectory.boolValue
                        ? "\(selectionPath)/\(candidate.relative)"
                        : selectionPath,
                    content: content,
                    sha256: sha256))
        }
        return (files, resolved.path)
    }
}

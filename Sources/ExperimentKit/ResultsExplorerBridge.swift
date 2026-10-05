import Foundation

/// The native Results Explorer bridge — the pure, testable logic behind the
/// app's `WKURLSchemeHandler` that hosts the embedded explorer SPA
/// (`results-explorer/`, built to `web/results-explorer/` by
/// `npm run build:embed`) and answers its `/api/tree` and `/api/file`
/// requests read-only from the active workspace's `runs/` directory.
///
/// Containment is the point: every path the PAGE supplies is a relative
/// POSIX path that must resolve inside the served root — the same
/// plain-name discipline the promotion gates apply to run names. The
/// explorer is a READING surface (CLAUDE.md's thin-view rule): it renders
/// what the engines wrote; paper numbers never originate in it.
///
/// One narrow write exists, for exports and downloads: `save` writes a
/// single file to a place the reader picked in a save panel, and refuses
/// any place inside `runs/`. Nothing under the served root is ever written.
public enum ResultsExplorerBridge {

    /// One directory entry, shaped for the page's fetch adapter
    /// (`app/embedded-workspace.ts` — name/kind/size/modified are the only
    /// fields discovery touches; `modified` is milliseconds since epoch to
    /// match JavaScript's `File.lastModified`).
    public struct TreeEntry: Codable, Equatable, Sendable {
        public let name: String
        public let kind: String  // "file" | "directory"
        public let size: Int
        public let modified: Int
    }

    /// Resolve a page-supplied relative path inside `root`, or nil when the
    /// path tries to escape: absolute paths, `.`/`..` components, empty
    /// components, backslashes, and NULs all refuse. An empty path is the
    /// root itself.
    public static func containedURL(path: String, under root: URL) -> URL? {
        if path.isEmpty { return root }
        if path.hasPrefix("/") || path.contains("\\") || path.contains("\0") {
            return nil
        }
        var url = root
        for component in path.split(separator: "/", omittingEmptySubsequences: false) {
            if component.isEmpty || component == "." || component == ".." {
                return nil
            }
            url.append(component: String(component))
        }
        // Belt over suspenders: the standardized result must stay under the
        // standardized root even if a component smuggled something exotic.
        let rootPath = root.standardizedFileURL.path
        guard url.standardizedFileURL.path.hasPrefix(rootPath) else {
            return nil
        }
        // Symlink escapes (review 2026-08-03, P1): textual containment is
        // not enough — a symlink beneath the root can point anywhere. The
        // candidate itself must not be a symlink, and its RESOLVED path
        // must stay under the RESOLVED root (resolution leaves nonexistent
        // tails untouched, so missing files still contain correctly and
        // fail later with a plain read error).
        if (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]))?
            .isSymbolicLink == true
        {
            return nil
        }
        let resolvedRoot = root.resolvingSymlinksInPath()
            .standardizedFileURL.path
        let resolved = url.resolvingSymlinksInPath().standardizedFileURL.path
        let rootPrefix =
            resolvedRoot.hasSuffix("/") ? resolvedRoot : resolvedRoot + "/"
        guard resolved == resolvedRoot || resolved.hasPrefix(rootPrefix)
        else {
            return nil
        }
        return url
    }

    /// List a contained directory, shaped for the fetch adapter. Hidden
    /// entries are skipped (the explorer skips dot-names client-side too;
    /// not sending them saves the round trip). Throws when the path escapes
    /// or is not a directory.
    public static func tree(path: String, under root: URL) throws -> [TreeEntry] {
        guard let directory = containedURL(path: path, under: root) else {
            throw ExperimentError(
                reason: "results-explorer bridge: path '\(path)' is not a "
                    + "plain relative path — refusing to read outside the "
                    + "served root")
        }
        let manager = FileManager.default
        var isDirectory: ObjCBool = false
        guard manager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw ExperimentError(
                reason: "results-explorer bridge: no directory at '\(path)'")
        }
        let children = try manager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [
                .isDirectoryKey, .fileSizeKey, .contentModificationDateKey,
                .isSymbolicLinkKey,
            ],
            options: [.skipsHiddenFiles])
        return children.compactMap { child -> TreeEntry? in
            guard
                let values = try? child.resourceValues(forKeys: [
                    .isDirectoryKey, .fileSizeKey, .contentModificationDateKey,
                    .isSymbolicLinkKey,
                ]),
                // Symlink entries are refused wholesale (review 2026-08-03,
                // P1) — not listed, so never addressable.
                values.isSymbolicLink != true
            else { return nil }
            let modified = Int(
                (values.contentModificationDate ?? .distantPast)
                    .timeIntervalSince1970 * 1000)
            return TreeEntry(
                name: child.lastPathComponent,
                kind: (values.isDirectory ?? false) ? "directory" : "file",
                size: values.fileSize ?? 0,
                modified: modified)
        }
        .sorted { $0.name < $1.name }
    }

    /// Read a contained file's bytes — bounded when the caller asks
    /// (review 2026-08-03, P2: the page's "bounded preview" sliced
    /// client-side while the bridge loaded the ENTIRE file; a
    /// multi-gigabyte generations.jsonl must never be materialized whole).
    /// Throws when the path escapes or the file is unreadable.
    public static func fileData(
        path: String, under root: URL,
        offset: Int? = nil, length: Int? = nil
    ) throws -> Data {
        guard let url = containedURL(path: path, under: root) else {
            throw ExperimentError(
                reason: "results-explorer bridge: path '\(path)' is not a "
                    + "plain relative path — refusing to read outside the "
                    + "served root")
        }
        guard offset != nil || length != nil else {
            return try Data(contentsOf: url)
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        if let offset, offset > 0 {
            try handle.seek(toOffset: UInt64(offset))
        }
        if let length {
            guard length >= 0 else {
                throw ExperimentError(
                    reason: "results-explorer bridge: negative read length")
            }
            return try handle.read(upToCount: length) ?? Data()
        }
        return try handle.readToEnd() ?? Data()
    }

    // MARK: - Saving one file the reader chose a place for

    /// The one thing the page may ask the host to write: a file the reader
    /// then places with a save panel. The page supplies WHAT to save and a
    /// suggested name; it never supplies WHERE. Everything else on this
    /// bridge stays read-only.
    public enum SaveRequest: Equatable, Sendable {
        /// Text the page built itself, such as a table export.
        case text(String)
        /// One file under the served `runs/` root, named by a contained
        /// relative path. The host copies the bytes; they never pass
        /// through the page, so a multi-gigabyte `generations.jsonl` is
        /// not loaded into the web view to be saved.
        case runFile(path: String)
    }

    /// Read the page's message. Only two shapes are understood, and
    /// anything else is nil:
    /// `{"kind": "text", "filename": …, "text": …}` and
    /// `{"kind": "runFile", "filename": …, "path": …}`.
    public static func saveRequest(
        fromMessage body: Any?
    ) -> (request: SaveRequest, suggestedName: String)? {
        guard let fields = body as? [String: Any],
            let kind = fields["kind"] as? String,
            let filename = fields["filename"] as? String
        else { return nil }
        switch kind {
        case "text":
            guard let text = fields["text"] as? String else { return nil }
            return (.text(text), suggestedFilename(filename))
        case "runFile":
            guard let path = fields["path"] as? String, !path.isEmpty else {
                return nil
            }
            return (.runFile(path: path), suggestedFilename(filename))
        default:
            return nil
        }
    }

    /// A page-supplied name reduced to a plain file name for the save
    /// panel's name field: the last path component, with control characters
    /// and leading dots removed. It is a suggestion only; the reader can
    /// change it, and the destination always comes from the panel.
    public static func suggestedFilename(
        _ raw: String, fallback: String = "export"
    ) -> String {
        let last =
            raw.split(whereSeparator: { $0 == "/" || $0 == "\\" || $0 == ":" })
            .last.map(String.init) ?? ""
        let cleaned = String(
            last.unicodeScalars.filter {
                !CharacterSet.controlCharacters.contains($0)
            }.map(Character.init)
        )
        .trimmingCharacters(in: .whitespaces)
        let visible = String(cleaned.drop(while: { $0 == "." }))
        return visible.isEmpty ? fallback : String(visible.prefix(200))
    }

    /// Write the requested file to `destination`, which must be the place
    /// the reader picked in a save panel. Returns the number of bytes
    /// written.
    ///
    /// Refusals, each in plain words:
    /// - a destination inside the served `runs/` root, however it is
    ///   reached. A run's folder is never changed after the run, because
    ///   custody checks re-read its files.
    /// - a destination that is a symbolic link or a folder.
    /// - a run-file path that is not contained under the root, or is a
    ///   symbolic link (the same refusals every read on this bridge makes).
    @discardableResult
    public static func save(
        _ request: SaveRequest, to destination: URL, runsRoot: URL
    ) throws -> Int {
        let name = destination.lastPathComponent
        guard destination.isFileURL, !name.isEmpty else {
            throw ExperimentError(
                reason: "The explorer can only save to a file on this Mac.")
        }
        let values = try? destination.resourceValues(
            forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
        if values?.isSymbolicLink == true {
            throw ExperimentError(
                reason: "'\(name)' is a link to another file, so the "
                    + "explorer will not write through it. Choose a "
                    + "different name or location.")
        }
        if values?.isDirectory == true {
            throw ExperimentError(
                reason: "'\(name)' is a folder. Choose a file name instead.")
        }
        // The folder the file will sit in, with every link resolved, so a
        // linked folder that leads into runs/ is refused like runs/ itself.
        let resolvedRoot = runsRoot.resolvingSymlinksInPath()
            .standardizedFileURL.path
        let resolvedDestination = destination.deletingLastPathComponent()
            .resolvingSymlinksInPath().standardizedFileURL
            .appending(component: name).path
        let rootPrefix =
            resolvedRoot.hasSuffix("/") ? resolvedRoot : resolvedRoot + "/"
        if resolvedDestination == resolvedRoot
            || resolvedDestination.hasPrefix(rootPrefix)
        {
            throw ExperimentError(
                reason: "The explorer does not save into the workspace's "
                    + "runs folder, because a run's folder is never changed "
                    + "after the run. Choose a location outside it, such as "
                    + "Documents or the Desktop.")
        }
        switch request {
        case .text(let text):
            let data = Data(text.utf8)
            do {
                try data.write(to: destination, options: .atomic)
            } catch {
                throw ExperimentError(
                    reason: "Could not save '\(name)': "
                        + error.localizedDescription)
            }
            return data.count
        case .runFile(let path):
            guard let source = containedURL(path: path, under: runsRoot) else {
                throw ExperimentError(
                    reason: "results-explorer bridge: path '\(path)' is not "
                        + "a plain relative path — refusing to read outside "
                        + "the served root")
            }
            let manager = FileManager.default
            var isDirectory: ObjCBool = false
            guard
                manager.fileExists(
                    atPath: source.path, isDirectory: &isDirectory),
                !isDirectory.boolValue
            else {
                throw ExperimentError(
                    reason: "The file '\(path)' is not in the workspace's "
                        + "runs folder any more, so there is nothing to "
                        + "save. Rescan the workspace and try again.")
            }
            // Copy beside the destination first, then move into place, so
            // a copy that fails part-way never leaves a short file under
            // the name the reader chose.
            let partial = destination.deletingLastPathComponent()
                .appending(
                    component: ".\(name).partial-\(UUID().uuidString)")
            do {
                try manager.copyItem(at: source, to: partial)
                if manager.fileExists(atPath: destination.path) {
                    _ = try manager.replaceItemAt(
                        destination, withItemAt: partial)
                } else {
                    try manager.moveItem(at: partial, to: destination)
                }
            } catch {
                try? manager.removeItem(at: partial)
                throw ExperimentError(
                    reason: "Could not save '\(name)': "
                        + error.localizedDescription)
            }
            let size =
                (try? destination.resourceValues(forKeys: [.fileSizeKey]))?
                .fileSize
            return size ?? 0
        }
    }

    /// Content type by extension — the handful the embedded page actually
    /// serves (SPA assets) plus the artifact types it fetches.
    public static func contentType(for name: String) -> String {
        switch (name as NSString).pathExtension.lowercased() {
        case "html": "text/html; charset=utf-8"
        case "js", "mjs": "text/javascript; charset=utf-8"
        case "css": "text/css; charset=utf-8"
        case "json": "application/json; charset=utf-8"
        case "jsonl": "application/x-ndjson; charset=utf-8"
        case "csv": "text/csv; charset=utf-8"
        case "svg": "image/svg+xml"
        case "png": "image/png"
        case "safetensors": "application/octet-stream"
        default: "application/octet-stream"
        }
    }
}

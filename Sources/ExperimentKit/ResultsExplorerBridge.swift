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

import CryptoKit
import Foundation

/// Pure resolution of the file paths experiment manifests and panels carry
/// (judge rubrics, task prompts, capability batteries, human baselines,
/// robustness files): project-relative or absolute strings → a display name
/// plus a resolved URL, with the same containment rule as
/// `VectorCatalog.projectFile` — a relative path may never escape the
/// project root. UI affordances (view sheet, Reveal in Finder) live in
/// `FileReferenceRow` (SteerLabApp); this type is UI-free and unit-tested.
public struct FileReference: Sendable, Equatable {
    /// The path string exactly as the manifest/panel carries it.
    public let originalPath: String
    /// Resolved on-disk location; nil for empty paths or a relative path
    /// that traverses outside the project root.
    public let url: URL?
    /// Last path component, for compact row display.
    public let displayName: String

    public var exists: Bool {
        guard let url else { return false }
        return FileManager.default.fileExists(atPath: url.path)
    }

    /// Resolves `path` against `root` (defaults to the project root).
    /// Absolute paths pass through standardized; relative paths must stay
    /// under `root` or resolve to nil (the row then renders the name but
    /// offers no view/reveal affordances — never a traversal).
    public static func resolve(
        _ path: String,
        root: URL = VectorCatalog.projectRoot
    ) -> FileReference {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return FileReference(originalPath: path, url: nil, displayName: "")
        }
        let name = (trimmed as NSString).lastPathComponent
        if trimmed.hasPrefix("/") {
            return FileReference(
                originalPath: trimmed,
                url: URL(filePath: trimmed).standardizedFileURL,
                displayName: name)
        }
        let base = root.standardizedFileURL
        let candidate = base.appending(path: trimmed).standardizedFileURL
        guard candidate.path == base.path || candidate.path.hasPrefix(base.path + "/") else {
            return FileReference(originalPath: trimmed, url: nil, displayName: name)
        }
        return FileReference(originalPath: trimmed, url: candidate, displayName: name)
    }

    /// SHA-256 of the file's current bytes (the same digest the pin
    /// machinery stamps), or nil when unreadable — so a viewer can show
    /// "pinned @ …" next to "current @ …" and make drift visible.
    public static func currentSHA256(of url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private init(originalPath: String, url: URL?, displayName: String) {
        self.originalPath = originalPath
        self.url = url
        self.displayName = displayName
    }
}

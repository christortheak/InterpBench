import CryptoKit
import Foundation

/// The packaging resource manifest (`resource-manifest.json` inside the app
/// bundle's Resources/, resolved via `CodeResources.buildManifest()`):
/// which files a payload ships and the SHA-256 of each, so an installed
/// payload can be verified complete and untampered before use
/// (docs/MAC-DISTRIBUTION-AND-MANAGED-SERVER-PROPOSAL.md §3, §6.3).
///
/// Written at packaging time by `generate(over:…)` (a future packaging
/// script calls the same entry point the tests call); read and checked by
/// `verify(_:against:)`. JSON is canonical in the engines' house style —
/// sorted keys — so identical inputs produce identical bytes.
public struct ResourceManifest: Codable, Equatable, Sendable {
    /// Bump when the manifest's own shape changes.
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    /// The Swift app version this payload was packaged with
    /// (`SteerLabVersion.version` convention).
    public var appVersion: String
    /// The Python server payload version this bundle carries.
    public var serverVersion: String
    /// The app↔server protocol version (proposal §8).
    public var protocolVersion: Int
    /// Short SHA of the source revision the payload was built from —
    /// stamped by the packaging environment when it has one, omitted (nil)
    /// otherwise. Optional and omit-when-nil, so schema stays 1; readers
    /// treat absence as "unknown", never an error.
    public var sourceRevision: String?
    /// Relative path (always "/"-separated, relative to the payload root)
    /// → lowercase-hex SHA-256 of the file's bytes.
    public var files: [String: String]

    public init(
        schemaVersion: Int = ResourceManifest.currentSchemaVersion,
        appVersion: String,
        serverVersion: String,
        protocolVersion: Int,
        sourceRevision: String? = nil,
        files: [String: String]
    ) {
        self.schemaVersion = schemaVersion
        self.appVersion = appVersion
        self.serverVersion = serverVersion
        self.protocolVersion = protocolVersion
        self.sourceRevision = sourceRevision
        self.files = files
    }

    // MARK: Serialization (canonical: sorted keys, house style)

    public func canonicalJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    public func write(to url: URL) throws {
        try canonicalJSON().write(to: url, options: .atomic)
    }

    public static func load(from url: URL) throws -> ResourceManifest {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ExperimentError(
                reason: "cannot read resource manifest at \(url.path): \(error)")
        }
        do {
            return try JSONDecoder().decode(ResourceManifest.self, from: data)
        } catch {
            throw ExperimentError(
                reason:
                    "resource manifest at \(url.path) is not valid: \(error)")
        }
    }

    // MARK: Generation (deterministic walk)

    /// Walk `root` and produce the manifest: every regular file, as a sorted
    /// "/"-relative path map to its SHA-256. Deterministic — paths are
    /// sorted and dot-prefixed entries (`.DS_Store`, `.git`, `.venv.nosync`,
    /// …) are skipped entirely, so the same tree always yields the same
    /// manifest (and `canonicalJSON()` the same bytes). Callable from a
    /// future packaging script and from tests alike.
    public static func generate(
        over root: URL,
        appVersion: String = SteerLabVersion.version,
        serverVersion: String,
        protocolVersion: Int,
        sourceRevision: String? = nil
    ) throws -> ResourceManifest {
        let fm = FileManager.default
        let standardRoot = root.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard
            fm.fileExists(atPath: standardRoot.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw ExperimentError(
                reason: "cannot generate a resource manifest: \(standardRoot.path) "
                    + "is not a directory")
        }
        guard
            let enumerator = fm.enumerator(
                at: standardRoot,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [])
        else {
            throw ExperimentError(
                reason: "cannot enumerate \(standardRoot.path) for manifest generation")
        }
        let rootPrefix = standardRoot.path + "/"
        var files: [String: String] = [:]
        for case let entry as URL in enumerator {
            let name = entry.lastPathComponent
            if name.hasPrefix(".") {
                // Skip hidden entries and never descend into hidden dirs.
                var entryIsDirectory: ObjCBool = false
                if fm.fileExists(
                    atPath: entry.path, isDirectory: &entryIsDirectory),
                    entryIsDirectory.boolValue
                {
                    enumerator.skipDescendants()
                }
                continue
            }
            let values = try? entry.resourceValues(forKeys: [.isRegularFileKey])
            guard values?.isRegularFile == true else { continue }
            let fullPath = entry.standardizedFileURL.path
            guard fullPath.hasPrefix(rootPrefix) else { continue }
            let relative = String(fullPath.dropFirst(rootPrefix.count))
            guard let data = try? Data(contentsOf: entry) else {
                throw ExperimentError(
                    reason: "cannot read \(relative) while generating the "
                        + "resource manifest")
            }
            files[relative] =
                SHA256.hash(data: data)
                .map { String(format: "%02x", $0) }.joined()
        }
        return ResourceManifest(
            appVersion: appVersion,
            serverVersion: serverVersion,
            protocolVersion: protocolVersion,
            sourceRevision: sourceRevision,
            files: files)
    }

    // MARK: Verification

    /// One plain-language problem found by `verify(_:against:)`.
    public struct Problem: Sendable, Equatable, CustomStringConvertible {
        public enum Kind: String, Sendable {
            case missing
            case unreadable
            case mismatch
        }
        public let kind: Kind
        /// The manifest-relative path of the offending file.
        public let path: String
        public let message: String
        public var description: String { message }
    }

    /// Check the payload at `root` against this manifest: completeness
    /// (every listed file exists) and integrity (every hash matches).
    /// Returns plain-language problems, sorted by path; empty means the
    /// payload verifies clean. Files on disk that the manifest does not
    /// list are not an error (bundles legitimately carry siblings).
    public func verify(against root: URL) -> [Problem] {
        let fm = FileManager.default
        let standardRoot = root.standardizedFileURL
        var problems: [Problem] = []
        for (path, expectedHash) in files.sorted(by: { $0.key < $1.key }) {
            let url = standardRoot.appending(path: path)
            guard fm.fileExists(atPath: url.path) else {
                problems.append(
                    Problem(
                        kind: .missing, path: path,
                        message: "\(path) is missing from the payload"))
                continue
            }
            guard let actualHash = FileReference.currentSHA256(of: url) else {
                problems.append(
                    Problem(
                        kind: .unreadable, path: path,
                        message: "\(path) cannot be read — the payload is "
                            + "corrupt or tampered"))
                continue
            }
            if actualHash != expectedHash.lowercased() {
                problems.append(
                    Problem(
                        kind: .mismatch, path: path,
                        message: "\(path) does not match the manifest — the "
                            + "payload is corrupt or tampered"))
            }
        }
        return problems
    }
}

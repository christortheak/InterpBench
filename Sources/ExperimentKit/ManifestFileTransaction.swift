import CryptoKit
import Darwin
import Foundation

/// Authoring metadata outside the manifest's encoded document and scientific identity.
public enum ManifestFilePrecondition: Sendable, Equatable {
    case absent
    case sha256(String)
}

/// Exact bytes and their concurrency tag. Decoding a scientific manifest is a
/// separate operation, so a concurrency tag can never enter its CodingKeys.
public struct ManifestFileSnapshot: Sendable {
    public let data: Data
    public let sha256: String

    public init(data: Data) {
        self.data = data
        self.sha256 = ManifestFileTransaction.digest(data)
    }
}

public enum ManifestFileTransaction {
    private static let registryLock = NSLock()
    // Accessed only under registryLock; individual entries serialize their path.
    nonisolated(unsafe) private static var locks: [String: NSRecursiveLock] = [:]

    public static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func snapshot(at url: URL) throws -> ManifestFileSnapshot {
        ManifestFileSnapshot(data: try Data(contentsOf: url))
    }

    /// Foundation's standardized file URLs can preserve platform aliases
    /// such as /tmp. Hash the POSIX real path, as Python os.path.realpath does,
    /// including when creation has not yet made the final path components.
    static func canonicalPath(_ url: URL) throws -> String {
        var path = url.path
        var missing: [String] = []
        while true {
            if let resolved = Darwin.realpath(path, nil) {
                defer { free(resolved) }
                return missing.reversed().reduce(String(cString: resolved)) {
                    ($0 as NSString).appendingPathComponent($1)
                }
            }
            guard errno == ENOENT else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            let parent = (path as NSString).deletingLastPathComponent
            guard parent != path, !parent.isEmpty else { throw POSIXError(.ENOENT) }
            missing.append((path as NSString).lastPathComponent)
            path = parent
        }
    }

    /// Python's manifest_files.transaction uses the same canonical-path key,
    /// directory and flock primitive. Keep the sidecar inode stable across writes.
    public static func withLock<T>(
        manifestURL: URL, workspaceRoot: URL, _ operation: () throws -> T
    ) throws -> T {
        let canonical = try canonicalPath(manifestURL)
        let root = URL(fileURLWithPath: try canonicalPath(workspaceRoot))
        let key = digest(Data(canonical.utf8))
        let directory = root.appending(components: ".steerlab", "manifest-locks")
        let lockPath = directory.appending(component: key + ".lock").path
        registryLock.lock()
        let lock = locks[lockPath] ?? NSRecursiveLock()
        locks[lockPath] = lock
        registryLock.unlock()
        lock.lock()
        defer { lock.unlock() }
        let heldKey = "SteerLabManifestLock:" + lockPath
        if Thread.current.threadDictionary[heldKey] != nil {
            return try operation()
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // Synchronization metadata is local runtime state, not freeze provenance.
        let ignore = Darwin.open(directory.appending(component: ".gitignore").path,
                                 O_CREAT | O_EXCL | O_WRONLY, mode_t(0o600))
        if ignore >= 0 {
            defer { Darwin.close(ignore) }
            let rule = Data("*\n".utf8)
            _ = rule.withUnsafeBytes { Darwin.write(ignore, $0.baseAddress, $0.count) }
        } else if errno != EEXIST {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        let descriptor = Darwin.open(lockPath, O_CREAT | O_RDWR, mode_t(0o600))
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(descriptor) }
        guard flock(descriptor, LOCK_EX) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        Thread.current.threadDictionary[heldKey] = true
        defer { Thread.current.threadDictionary.removeObject(forKey: heldKey) }
        return try operation()
    }

    /// Must be called under withLock, with publication inside the same closure.
    public static func requireCurrent(_ expected: ManifestFilePrecondition, at url: URL) throws {
        let current: String?
        do {
            current = try snapshot(at: url).sha256
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            current = nil
        }
        let matches: Bool
        switch expected {
        case .absent:
            matches = current == nil
        case .sha256(let digest):
            matches = digest.count == 64
                && digest.allSatisfy { "0123456789abcdef".contains($0) }
                && current == digest
        }
        guard matches else {
            throw ExperimentError.refusing(
                .staleManifest,
                "The manifest changed after it was read; no edit was published.",
                repair: "steerlab-cli experiment manifest \(url.deletingLastPathComponent().lastPathComponent) --json; review the intervening changes and submit with manifestFileSHA256.")
        }
    }
}

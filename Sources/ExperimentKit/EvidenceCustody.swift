import CryptoKit
import Foundation

/// Local custody is a verifiable record of bytes held here, independent of
/// whether those bytes describe a successful study or a failure record.
public struct EvidenceCustodyReceipt: Codable, Sendable, Equatable {
    public struct FileDigest: Codable, Sendable, Equatable {
        public let path: String
        public let sha256: String
    }

    public let schemaVersion: Int
    public let createdAt: String
    public let workspaceRoot: URL
    public let origin: EvidenceImportOrigin?
    public let archivePath: String
    public let archiveSHA256: String
    public let metadataSHA256: String
    public let runID: String
    public let importedRunIDs: [String]
    public let files: [FileDigest]
    public let evidenceComplete: Bool?
    public let failureRecorded: Bool
}

public struct EvidenceImportResult: Sendable {
    public let runDirectory: URL
    public let receiptURL: URL
    public let receiptSHA256: String
    public let receipt: EvidenceCustodyReceipt
}

struct CapturedEvidenceArchive: Sendable {
    let url: URL
    let sha256: String
}

/// Discovery verifies receipt identity only. Archive/file verification is an
/// explicit operation so listing remains cheap and does not imply current custody.
public struct EvidenceCustodyInventory: Codable, Sendable {
    public struct Entry: Codable, Sendable, Identifiable {
        public var id: String { receiptSHA256 }
        public let receiptSHA256: String
        public let createdAt: String
        public let archiveSHA256: String
        public let origin: EvidenceImportOrigin?
        public let evidenceComplete: Bool?
        public let failureRecorded: Bool
    }
    public let runID: String
    public let entries: [Entry]
    public let issues: [String]
}

public enum EvidenceCustodyStore {
    private static let archiveDirectory = ".steerlab/evidence-archives"
    private static let receiptDirectory = ".steerlab/evidence-custody"

    public static func inventory(runID: String, workspaceRoot: URL) throws -> EvidenceCustodyInventory {
        guard EvidenceBundleImporter.isSafeComponent(runID) else {
            throw ChatServiceError(reason: "invalid run ID for custody discovery")
        }
        let root = try canonicalRoot(workspaceRoot)
        let directory = root.appending(path: receiptDirectory)
        let fm = FileManager.default
        // Absence is an empty inventory, not a reason to create workspace state.
        var componentURL = root
        for component in receiptDirectory.split(separator: "/") {
            componentURL.append(component: String(component))
            do {
                guard try fm.attributesOfItem(atPath: componentURL.path)[.type] as? FileAttributeType == .typeDirectory else {
                    throw ChatServiceError(reason: "custody discovery requires ordinary workspace directories")
                }
            } catch CocoaError.fileReadNoSuchFile {
                return .init(runID: runID, entries: [], issues: [])
            }
        }
        var entries: [EvidenceCustodyInventory.Entry] = []
        var issues: [String] = []
        for name in try fm.contentsOfDirectory(atPath: directory.path).sorted() where name.hasSuffix(".json") {
            let digest = String(name.dropLast(5))
            do {
                let receipt = try loadReceipt(receiptSHA256: digest, root: root)
                guard receipt.importedRunIDs.contains(runID) else { continue }
                entries.append(.init(receiptSHA256: digest, createdAt: receipt.createdAt,
                    archiveSHA256: receipt.archiveSHA256, origin: receipt.origin,
                    evidenceComplete: receipt.evidenceComplete, failureRecorded: receipt.failureRecorded))
            } catch {
                issues.append("Could not inspect receipt \(name): \(error)")
            }
        }
        return .init(runID: runID, entries: entries.sorted {
            $0.createdAt == $1.createdAt ? $0.id < $1.id : $0.createdAt > $1.createdAt
        }, issues: issues)
    }

    /// Capture ordinary file bytes before extraction. Content-addressed copies
    /// retain every archive member, including metadata and unexpanded logs.
    /// A failed import may leave an unreferenced archive, never a custody receipt.
    static func captureArchive(
        _ source: URL, workspaceRoot: URL, expectedSHA256: String?
    ) throws -> CapturedEvidenceArchive {
        let root = try canonicalRoot(workspaceRoot)
        let directory = try createOwnedDirectory(archiveDirectory, root: root)
        let temporary = directory.appending(component: ".capture-\(UUID().uuidString)")
        let fm = FileManager.default
        guard try fm.attributesOfItem(atPath: source.path)[.type] as? FileAttributeType == .typeRegular else {
            throw ChatServiceError(reason: "evidence archive must be an ordinary file")
        }
        try fm.copyItem(at: source, to: temporary)
        defer { try? fm.removeItem(at: temporary) }
        guard try fm.attributesOfItem(atPath: temporary.path)[.type] as? FileAttributeType == .typeRegular else {
            throw ChatServiceError(reason: "captured archive must contain ordinary file bytes")
        }
        let digest = try hashFile(temporary)
        if let expectedSHA256, !expectedSHA256.isEmpty, expectedSHA256 != digest {
            throw ChatServiceError(reason: "evidence bundle hash mismatch before import")
        }
        let target = directory.appending(component: digest + ".tar.gz")
        try ManifestFileTransaction.withLock(manifestURL: target, workspaceRoot: root) {
            if fm.fileExists(atPath: target.path) {
                guard try localFileHash(archiveDirectory + "/" + digest + ".tar.gz", root: root) == digest else {
                    throw ChatServiceError(reason: "retained evidence archive differs from its content address")
                }
            } else {
                try fm.moveItem(at: temporary, to: target)
            }
        }
        return CapturedEvidenceArchive(url: target, sha256: digest)
    }

    /// Called only after the shared importer has verified the archive and
    /// published its declared runs. This rechecks the actual local files before
    /// issuing a receipt; the caller rolls back newly published runs on failure.
    static func record(
        archive: CapturedEvidenceArchive, metadata: [String: CodableValue], metadataData: Data,
        runID: String, importedRunIDs: Set<String>, workspaceRoot: URL,
        origin: EvidenceImportOrigin?, portableData: Data?
    ) throws -> EvidenceImportResult {
        let root = try canonicalRoot(workspaceRoot)
        guard case .array(let entries)? = metadata["entries"] else {
            throw ChatServiceError(reason: "evidence receipt requires a verified entry manifest")
        }
        var files: [String: String] = [:]
        for entry in entries {
            guard case .object(let item) = entry,
                case .string(let path)? = item["path"],
                case .string(let digest)? = item["sha256"] else { continue }
            let components = path.split(separator: "/", omittingEmptySubsequences: false)
            guard components.count > 2, components[0] == "runs",
                importedRunIDs.contains(String(components[1])) else { continue }
            if let prior = files[path], prior != digest {
                throw ChatServiceError(reason: "conflicting evidence digests for \(path)")
            }
            files[path] = digest
        }
        if let portableData {
            let path = "runs/\(runID)/pipeline-portable.json"
            let digest = ManifestFileTransaction.digest(portableData)
            if let prior = files[path], prior != digest {
                throw ChatServiceError(reason: "portable evidence conflicts with the run's declared ledger")
            }
            files[path] = digest
        }
        let complete: Bool? = if case .bool(let value)? = metadata["evidenceComplete"] { value } else { nil }
        let receipt = EvidenceCustodyReceipt(
            schemaVersion: 1, createdAt: HousekeepingDates.format(Date()), workspaceRoot: root, origin: origin,
            archivePath: archiveDirectory + "/" + archive.sha256 + ".tar.gz", archiveSHA256: archive.sha256,
            metadataSHA256: ManifestFileTransaction.digest(metadataData), runID: runID,
            importedRunIDs: importedRunIDs.sorted(),
            files: files.sorted { $0.key < $1.key }.map { .init(path: $0.key, sha256: $0.value) },
            evidenceComplete: complete, failureRecorded: metadata["failure"] != nil)
        try verify(receipt, workspaceRoot: root)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(receipt)
        let digest = ManifestFileTransaction.digest(data)
        let directory = try createOwnedDirectory(receiptDirectory, root: root)
        let url = directory.appending(component: digest + ".json")
        try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: root) {
            if FileManager.default.fileExists(atPath: url.path) {
                guard try localFileHash(receiptDirectory + "/" + digest + ".json", root: root) == digest else {
                    throw ChatServiceError(reason: "evidence custody receipt differs from its content address")
                }
            } else {
                try data.write(to: url, options: .atomic)
            }
        }
        return EvidenceImportResult(runDirectory: workspaceRoot.appending(components: "runs", runID),
                                    receiptURL: url, receiptSHA256: digest, receipt: receipt)
    }

    /// Re-read a receipt by its content address, then verify retained archive
    /// bytes and every expanded evidence file. This never repairs or rewrites.
    public static func loadVerified(
        receiptSHA256: String, workspaceRoot: URL
    ) throws -> EvidenceCustodyReceipt {
        let root = try canonicalRoot(workspaceRoot)
        let receipt = try loadReceipt(receiptSHA256: receiptSHA256, root: root)
        try verify(receipt, workspaceRoot: root)
        return receipt
    }

    private static func loadReceipt(receiptSHA256: String, root: URL) throws -> EvidenceCustodyReceipt {
        guard isDigest(receiptSHA256) else { throw ChatServiceError(reason: "invalid custody receipt digest") }
        let relative = receiptDirectory + "/" + receiptSHA256 + ".json"
        let bytes = try Data(contentsOf: localFileURL(relative, root: root))
        guard ManifestFileTransaction.digest(bytes) == receiptSHA256 else {
            throw ChatServiceError(reason: "custody receipt content hash mismatch")
        }
        let receipt = try JSONDecoder().decode(EvidenceCustodyReceipt.self, from: bytes)
        guard receipt.schemaVersion == 1, receipt.workspaceRoot.path == root.path else {
            throw ChatServiceError(reason: "custody receipt identity does not match this workspace")
        }
        return receipt
    }

    public static func verify(_ receipt: EvidenceCustodyReceipt, workspaceRoot: URL) throws {
        let root = try canonicalRoot(workspaceRoot)
        let originRoot = try receipt.origin.map { try canonicalRoot($0.workspaceRoot).path }
        guard receipt.schemaVersion == 1, receipt.workspaceRoot.path == root.path,
            originRoot == nil || originRoot == root.path,
            isDigest(receipt.archiveSHA256), isDigest(receipt.metadataSHA256),
            receipt.archivePath == archiveDirectory + "/" + receipt.archiveSHA256 + ".tar.gz",
            receipt.importedRunIDs.contains(receipt.runID),
            receipt.importedRunIDs.allSatisfy(EvidenceBundleImporter.isSafeComponent) else {
            throw ChatServiceError(reason: "custody receipt identity does not match this workspace")
        }
        guard try localFileHash(receipt.archivePath, root: root) == receipt.archiveSHA256 else {
            throw ChatServiceError(reason: "retained evidence archive failed custody verification")
        }
        for file in receipt.files {
            let components = file.path.split(separator: "/", omittingEmptySubsequences: false)
            guard components.count > 2, components[0] == "runs",
                receipt.importedRunIDs.contains(String(components[1])), isDigest(file.sha256),
                try localFileHash(file.path, root: root) == file.sha256 else {
                throw ChatServiceError(reason: "local evidence failed custody verification: \(file.path)")
            }
        }
    }

    private static func canonicalRoot(_ root: URL) throws -> URL {
        URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(root))
    }

    private static func isDigest(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    private static func createOwnedDirectory(_ path: String, root: URL) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        var url = root
        for component in path.split(separator: "/") {
            url.append(component: String(component))
            if !fm.fileExists(atPath: url.path) {
                do { try fm.createDirectory(at: url, withIntermediateDirectories: false) }
                catch CocoaError.fileWriteFileExists { /* A concurrent creator must still pass the directory check below. */ }
            }
            guard try fm.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType == .typeDirectory else {
                throw ChatServiceError(reason: "evidence custody storage must use ordinary workspace directories")
            }
        }
        return url
    }

    private static func localFileHash(_ path: String, root: URL) throws -> String {
        try hashFile(localFileURL(path, root: root))
    }

    private static func localFileURL(_ path: String, root: URL) throws -> URL {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.isEmpty, components.allSatisfy({ EvidenceBundleImporter.isSafeComponent(String($0)) }) else {
            throw ChatServiceError(reason: "unsafe custody file path")
        }
        var url = root
        for (index, component) in components.enumerated() {
            url.append(component: String(component))
            let kind: FileAttributeType = index == components.count - 1 ? .typeRegular : .typeDirectory
            guard try FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType == kind else {
                throw ChatServiceError(reason: "custody requires ordinary local files: \(path)")
            }
        }
        return url
    }

    /// Archives can be large; hashing must not load the whole bundle into RAM.
    private static func hashFile(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while let bytes = try file.read(upToCount: 1024 * 1024), !bytes.isEmpty { hash.update(data: bytes) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

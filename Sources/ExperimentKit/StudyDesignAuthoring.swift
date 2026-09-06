import Foundation

public struct StudyDesignAuthoringError: Error, LocalizedError, Sendable {
    public let code: String
    public let reason: String
    public let repairAction: String
    public var errorDescription: String? { reason + " " + repairAction }
}

/// A design and the exact external file version that supplied its editor.
/// Neither the template encoding nor its scientific content hash changes.
public struct StudyDesignSnapshot: Sendable {
    public let workspaceRoot: URL
    public let template: StudyTemplate
    public let file: ManifestFileSnapshot

    public init(workspaceRoot: URL, name: String) throws {
        let repository = StudyDesignRepository(workspaceRoot: workspaceRoot)
        let url = try repository.existingFile(name: name)
        let file = try ManifestFileTransaction.snapshot(at: url)
        let template = try JSONDecoder().decode(StudyTemplate.self, from: file.data)
        guard template.name == name else {
            throw StudyDesignAuthoringError(code: "designIdentityMismatch", reason: "The design file names another destination.",
                repairAction: "Check the named design and its directory before editing.")
        }
        self.workspaceRoot = workspaceRoot.standardizedFileURL
        self.template = template
        self.file = file
    }
}

public struct StudyDesignDocument: Encodable, Sendable {
    public let name: String
    public let workspaceRoot: String
    public let designFileSHA256: String
    public let contentHash: String
    public let portableContentHash: String
    public let portableHashAlgorithm = PortableDesignIdentity.algorithm
    public let document: JSONValue
    public let seatIDs: [String]?
    public let advisories: [String]

    public init(_ snapshot: StudyDesignSnapshot) throws {
        name = snapshot.template.name
        workspaceRoot = snapshot.workspaceRoot.path
        designFileSHA256 = snapshot.file.sha256
        contentHash = StudyTemplateStore.hash(snapshot.template)
        portableContentHash = try PortableDesignIdentity.hash(snapshot.template)
        document = try JSONDecoder().decode(JSONValue.self, from: snapshot.file.data)
        if let ref = snapshot.template.semanticScenario {
            do {
                seatIDs = PanelComposition.seatIDs(try StudyTemplateStore.loadSemanticPanel(ref, workspaceRoot: snapshot.workspaceRoot))
                advisories = []
            } catch {
                seatIDs = nil
                advisories = ["The design was read, but its panel cannot currently be cast: " + error.localizedDescription]
            }
        } else {
            seatIDs = nil
            advisories = []
        }
    }
}

public struct StudyDesignCatalog: Encodable, Sendable {
    public struct Entry: Encodable, Sendable {
        public let name: String
        public let description: String
        public let designFileSHA256: String
        public let contentHash: String
    }
    public let entries: [Entry]
    public let issues: [String]
}

/// Storage boundary for reviewed edits of existing designs. Creation, rename and
/// instantiation are separate operations and cannot borrow this update authority.
struct StudyDesignRepository {
    let workspaceRoot: URL

    func existingFile(name: String) throws -> URL {
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/"),
            !name.contains("\\"), !name.contains("\0") else {
            throw StudyDesignAuthoringError(code: "invalidDesignName", reason: "A design name must be one path component.",
                repairAction: "Use a name from the design library.")
        }
        var url = URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(workspaceRoot))
        for (index, component) in ["templates", name, "template.json"].enumerated() {
            url.append(component: component)
            let expected: FileAttributeType = index == 2 ? .typeRegular : .typeDirectory
            guard try FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType == expected else {
                throw StudyDesignAuthoringError(code: "unsafeDesignPath", reason: "Design edits require ordinary workspace directories and files.",
                    repairAction: "Inspect the design path; do not redirect an edit through links into another artifact tree.")
            }
        }
        return url
    }
}

public enum StudyDesignAuthoring {
    public static func list(workspaceRoot: URL) throws -> StudyDesignCatalog {
        let root = URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(workspaceRoot))
        let directory = root.appending(component: "templates")
        do {
            guard try FileManager.default.attributesOfItem(atPath: directory.path)[.type] as? FileAttributeType == .typeDirectory else {
                throw StudyDesignAuthoringError(code: "unsafeDesignPath", reason: "The design library must be an ordinary workspace directory.",
                    repairAction: "Inspect the library path before loading designs.")
            }
        } catch CocoaError.fileReadNoSuchFile { return .init(entries: [], issues: []) }
        var entries: [StudyDesignCatalog.Entry] = []
        var issues: [String] = []
        for name in try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
            where name != ".DS_Store" && !name.hasPrefix("._") {
            do {
                let read = try StudyDesignSnapshot(workspaceRoot: workspaceRoot, name: name)
                entries.append(.init(name: name, description: read.template.templateDescription,
                    designFileSHA256: read.file.sha256, contentHash: StudyTemplateStore.hash(read.template)))
            } catch { issues.append("Could not inspect design \(name): \(error.localizedDescription)") }
        }
        return .init(entries: entries, issues: issues)
    }

    /// Transport admission: the caller's digest must identify the document it
    /// reviewed. Reading a fresh file here cannot silently retag stale text.
    public static func review(name: String, workspaceRoot: URL, expectedFileSHA256: String) throws -> StudyDesignSnapshot {
        guard expectedFileSHA256.count == 64,
            expectedFileSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw StudyDesignAuthoringError(code: "invalidDesignPrecondition", reason: "The design file precondition must be a lowercase SHA-256 digest.",
                repairAction: "Use designFileSHA256 from the design inspection result.")
        }
        let reviewed = try StudyDesignSnapshot(workspaceRoot: workspaceRoot, name: name)
        guard reviewed.file.sha256 == expectedFileSHA256 else {
            throw StudyDesignAuthoringError(code: "designChanged", reason: "The design changed after the supplied file version was reviewed.",
                repairAction: "Inspect the named design again and review the changes before reconstructing the intended edit.")
        }
        return reviewed
    }

    /// Holds the shared design lock throughout admission and publication by the
    /// supplied operation. The external file version never becomes study data.
    static func withReviewedDesign<T>(_ reviewed: StudyDesignSnapshot, _ body: (URL) throws -> T) throws -> T {
        let repository = StudyDesignRepository(workspaceRoot: reviewed.workspaceRoot)
        let url = try repository.existingFile(name: reviewed.template.name)
        return try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: reviewed.workspaceRoot) {
            _ = try repository.existingFile(name: reviewed.template.name)
            do { try ManifestFileTransaction.requireCurrent(.sha256(reviewed.file.sha256), at: url) }
            catch let error as ExperimentError where error.lifecycleRefusal?.gate == .staleManifest {
                throw StudyDesignAuthoringError(code: "designChanged", reason: "The design changed after it was reviewed; this operation published no study or design edit.",
                    repairAction: "Discard the old design review and reload it, review the intervening changes, then reconstruct the intended operation.")
            }
            return try body(url)
        }
    }

    @discardableResult
    public static func updateDescription(_ description: String, reviewed: StudyDesignSnapshot) throws -> StudyDesignSnapshot {
        try withReviewedDesign(reviewed) { url in
            if reviewed.template.templateDescription == description { return reviewed }
            var updated = reviewed.template
            updated.templateDescription = description
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(updated).write(to: url, options: .atomic)
            return try StudyDesignSnapshot(workspaceRoot: reviewed.workspaceRoot, name: updated.name)
        }
    }
}

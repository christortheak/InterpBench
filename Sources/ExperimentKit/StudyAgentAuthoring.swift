import Foundation

/// An exact, named agent artifact read in its workspace. The digest is transport
/// metadata; it is not inserted into an artifact or study schema.
public struct AgentArtifactSnapshot: Sendable {
    public let workspaceRoot: URL
    public let path: String
    public let file: ManifestFileSnapshot
    public let record: ModelVariantRecord

    public init(workspaceRoot: URL, path: String) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard components.count >= 2, components.first == "runs",
            components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\\") && !$0.contains("\0") }) else {
            throw ExperimentError.malformed("An agent artifact must name a workspace-relative file under runs/.",
                repair: "Use an artifact path returned by agent list --json.")
        }
        var url = URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(workspaceRoot))
        for (index, component) in components.enumerated() {
            url.append(component: component)
            let kind: FileAttributeType = index == components.count - 1 ? .typeRegular : .typeDirectory
            guard try FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType == kind else {
                throw ExperimentError.refusing(.artifactPin, "The agent path must use ordinary workspace files and directories.",
                    repair: "Inspect the named artifact; do not redirect attachment through a link into another workspace.")
            }
        }
        let snapshot = try ManifestFileTransaction.snapshot(at: url)
        let artifact: ModelVariantArtifact
        do { artifact = try JSONDecoder().decode(ModelVariantArtifact.self, from: snapshot.data) }
        catch {
            throw ExperimentError.refusing(.artifactPin, "The selected file is not a readable agent artifact.",
                repair: "Inspect the named file and restore its verified source before attaching it.")
        }
        self.workspaceRoot = workspaceRoot.standardizedFileURL
        self.path = path
        file = snapshot
        record = ModelVariantRecord(url: url, artifact: artifact)
    }

    public init(workspaceRoot: URL, reviewedRecord: ModelVariantRecord) throws {
        let prefix = try ManifestFileTransaction.canonicalPath(workspaceRoot) + "/"
        let path = try ManifestFileTransaction.canonicalPath(reviewedRecord.url)
        guard path.hasPrefix(prefix) else {
            throw ExperimentError.refusing(.artifactPin, "The selected agent is outside the study workspace.",
                repair: "Import the agent's evidence into this workspace, then select it again.")
        }
        try self.init(workspaceRoot: workspaceRoot, path: String(path.dropFirst(prefix.count)))
        guard record.artifact == reviewedRecord.artifact else {
            throw ExperimentError.refusing(.artifactPin, "The agent changed after it was displayed.",
                repair: "Reload the agent library and review the changed artifact before selecting it again.")
        }
    }

}

public struct AgentArtifactDocument: Encodable, Sendable {
    public let path: String
    public let artifactFileSHA256: String
    public let artifact: ModelVariantArtifact
    public let document: JSONValue

    public init(_ snapshot: AgentArtifactSnapshot) throws {
        path = snapshot.path
        artifactFileSHA256 = snapshot.file.sha256
        artifact = snapshot.record.artifact
        document = try JSONDecoder().decode(JSONValue.self, from: snapshot.file.data)
    }
}

public enum StudyAgentAuthoring {
    public struct Catalog: Encodable, Sendable {
        public let agents: [AgentArtifactDocument]
        public let issues: [String]
    }

    public static func list(workspaceRoot: URL) throws -> Catalog {
        let root = URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(workspaceRoot))
        let runs = root.appending(component: "runs")
        guard FileManager.default.fileExists(atPath: runs.path) else { return .init(agents: [], issues: []) }
        guard try FileManager.default.attributesOfItem(atPath: runs.path)[.type] as? FileAttributeType == .typeDirectory else {
            throw ExperimentError.refusing(.artifactPin, "The agent library's runs path is not an ordinary directory.",
                repair: "Inspect the workspace before listing its agent artifacts.")
        }
        let records = ModelVariantStore.scan(directory: runs.appending(component: "model-variants"), importedRoot: runs)
        var agents: [AgentArtifactDocument] = []
        var issues: [String] = []
        for record in records {
            do {
                let prefix = try ManifestFileTransaction.canonicalPath(root) + "/"
                let canonical = try ManifestFileTransaction.canonicalPath(record.url)
                guard canonical.hasPrefix(prefix) else { continue }
                let path = String(canonical.dropFirst(prefix.count))
                agents.append(try AgentArtifactDocument(AgentArtifactSnapshot(workspaceRoot: root, path: path)))
            } catch { issues.append("Could not inspect \(record.url.lastPathComponent): \(error)") }
        }
        return .init(agents: agents, issues: issues)
    }

    public static func reviewArtifact(path: String, workspaceRoot: URL, expectedFileSHA256: String) throws -> AgentArtifactSnapshot {
        try requireDigest(expectedFileSHA256)
        let artifact = try AgentArtifactSnapshot(workspaceRoot: workspaceRoot, path: path)
        guard artifact.file.sha256 == expectedFileSHA256 else {
            throw ExperimentError.refusing(.artifactPin, "The agent artifact changed after inspection.",
                repair: "Inspect the artifact and review the changes before reconstructing the attachment request.")
        }
        return artifact
    }

    @discardableResult
    public static func attach(_ agent: AgentArtifactSnapshot, reviewed: DraftAuthoringSnapshot,
                              baseModelChoice: String? = nil) throws -> DraftAuthoringSnapshot {
        guard try ManifestFileTransaction.canonicalPath(agent.workspaceRoot)
            == ManifestFileTransaction.canonicalPath(reviewed.workspaceRoot) else {
            throw ExperimentError.refusing(.artifactPin, "The selected agent belongs to another workspace.",
                repair: "Inspect an agent in the study's workspace before attaching it.")
        }
        let storage = ExperimentRepository(workspaceRoot: reviewed.workspaceRoot)
        return try ManifestFileTransaction.withLock(manifestURL: storage.manifestURL(reviewed.manifest.name), workspaceRoot: reviewed.workspaceRoot) {
            try ManifestFileTransaction.requireCurrent(.sha256(reviewed.file.sha256), at: storage.manifestURL(reviewed.manifest.name))
            try ManifestMutationPolicy.admitDraftEdit(reviewed.manifest)
            var manifest = reviewed.manifest
            if let choice = baseModelChoice { StudyProtocolAuthoring.applyBaseModelChoice(choice, to: &manifest) }
            try ExperimentStore.attachAgent(agent.record, into: &manifest, workspaceRoot: reviewed.workspaceRoot,
                expectedArtifactFileSHA256: agent.file.sha256)
            return try DraftAuthoringTransaction.replace(manifest, reviewed: reviewed)
        }
    }

    private static func requireDigest(_ value: String) throws {
        guard value.count == 64, value.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw ExperimentError.malformed("A reviewed file precondition must be a lowercase SHA-256 digest.",
                repair: "Use the exact file digest from the corresponding inspection result.")
        }
    }
}

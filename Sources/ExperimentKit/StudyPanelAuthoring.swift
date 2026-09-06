import Foundation

/// Semantic panel inputs are immutable versions. Casting edits only a reviewed
/// draft and delegates scientific compilation to the existing seat owner.
public enum StudyPanelAuthoring {
    public struct Publication: Sendable {
        public let record: MultiAgentScenarioRecord
        public let changed: Bool
    }
    public struct Document: Encodable, Sendable {
        public let path: String
        public let fileSHA256: String
        public let document: MultiAgentScenario
        public let semantic: Bool
        public let seatIDs: [String]
    }

    public static func ordinary(_ path: String, root: URL) throws -> URL {
        let components = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard components.count >= 3, components.prefix(2) == ["prompts", "panels"],
            components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\\") && !$0.contains("\0") }) else {
            throw malformed("Use an ordinary workspace-relative panel path under prompts/panels/.")
        }
        var url = URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(root))
        for (index, component) in components.enumerated() {
            url.append(component: component)
            do {
                let type = try FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType
                guard type == (index == components.count - 1 ? .typeRegular : .typeDirectory) else {
                    throw malformed("Panel paths must not follow links or nonordinary files.")
                }
            } catch CocoaError.fileReadNoSuchFile { continue }
        }
        return url
    }

    public static func inspect(path: String, root: URL) throws -> Document {
        let data = try Data(contentsOf: ordinary(path, root: root))
        let panel = try JSONDecoder().decode(MultiAgentScenario.self, from: data)
        return Document(path: path, fileSHA256: MultiAgentScenarioStore.hash(data), document: panel,
                        semantic: !PanelAuthoring.carriesBindings(panel), seatIDs: panel.agents.map(\.id))
    }

    public static func validate(_ panel: MultiAgentScenario) throws {
        guard !PanelAuthoring.carriesBindings(panel), !panel.agents.isEmpty,
            Set(panel.agents.map(\.id)).count == panel.agents.count else {
            throw malformed("Use a semantic panel with unique seats; model settings and agents belong to a study.")
        }
        let bound = try PanelAuthoring.rehearsalScenario(panel, modelID: "validation/model", temperature: 0, maxTokens: 2048)
        try MultiAgentRunner.validate(bound)
    }

    public static func publish(_ panel: MultiAgentScenario, root: URL) throws -> Publication {
        try validate(panel)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(panel)
        let path = "prompts/panels/semantic-\(MultiAgentScenarioStore.hash(data)).json"
        let url = try ordinary(path, root: root)
        return try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: root) {
            _ = try ordinary(path, root: root)
            let existed = FileManager.default.fileExists(atPath: url.path)
            if existed {
                guard try Data(contentsOf: url) == data else { throw malformed("The existing panel does not contain its named bytes.") }
            } else {
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                let temporary = url.deletingLastPathComponent().appending(component: ".panel-\(UUID().uuidString).tmp")
                defer { try? FileManager.default.removeItem(at: temporary) }
                try data.write(to: temporary, options: .withoutOverwriting)
                try FileManager.default.linkItem(at: temporary, to: url)
            }
            return Publication(record: MultiAgentScenarioRecord(url: url, scenario: panel), changed: !existed)
        }
    }

    public static func compile(path: String, expectedPanel: String, casting: Data,
                               reviewed: DraftAuthoringSnapshot) throws -> DraftAuthoringSnapshot {
        let root = reviewed.workspaceRoot
        let storage = ExperimentRepository(workspaceRoot: root)
        return try ManifestFileTransaction.withLock(manifestURL: storage.manifestURL(reviewed.manifest.name), workspaceRoot: root) {
            try ManifestFileTransaction.requireCurrent(.sha256(reviewed.file.sha256), at: storage.manifestURL(reviewed.manifest.name))
            try ManifestMutationPolicy.admitDraftEdit(reviewed.manifest)
            let panel = try inspect(path: path, root: root)
            guard panel.fileSHA256 == expectedPanel else { throw ExperimentError.refusing(.artifactPin, "The panel changed after inspection.", repair: "Inspect and review the panel again before casting.") }
            try validate(panel.document)
            guard let object = try JSONSerialization.jsonObject(with: casting) as? [String: Any], Set(object.keys) == ["seats"],
                let seats = object["seats"] as? [String: Any], Set(seats.keys) == Set(panel.seatIDs) else {
                throw malformed("Cast every seat explicitly; null selects baseline, and extra or missing seats refuse.")
            }
            var occupants: [String: SeatOccupant] = [:]
            for (id, value) in seats {
                if value is NSNull { occupants[id] = .baseline; continue }
                guard let reference = value as? [String: String], Set(reference.keys) == ["artifactPath", "artifactFileSHA256"] else {
                    throw malformed("Every agent must name artifactPath and artifactFileSHA256 from inspection.")
                }
                let agent = try StudyAgentAuthoring.reviewArtifact(path: reference["artifactPath"]!, workspaceRoot: root, expectedFileSHA256: reference["artifactFileSHA256"]!)
                guard agent.record.artifact.baseModelID == reviewed.manifest.modelID else { throw malformed("A seat agent uses another base model.") }
                occupants[id] = .agent(name: agent.record.artifact.name, artifactPath: agent.path, artifactHash: agent.file.sha256)
            }
            return try saveAssignment(.init(seatIDs: panel.seatIDs, occupants: occupants), semantic: panel.document,
                semanticPath: path, reviewed: reviewed, expectedPanel: panel.fileSHA256)
        }
    }

    public static func saveAssignment(_ assignment: SeatAssignment, semantic: MultiAgentScenario,
        semanticPath: String?, reviewed: DraftAuthoringSnapshot, expectedPanel: String? = nil,
        modelID: String? = nil, temperature: Double? = nil, maxTokens: Int? = nil,
        fileSlug: String? = nil) throws -> DraftAuthoringSnapshot {
        let root = reviewed.workspaceRoot
        let storage = ExperimentRepository(workspaceRoot: root)
        return try ManifestFileTransaction.withLock(manifestURL: storage.manifestURL(reviewed.manifest.name), workspaceRoot: root) {
            try ManifestFileTransaction.requireCurrent(.sha256(reviewed.file.sha256), at: storage.manifestURL(reviewed.manifest.name))
            try ManifestMutationPolicy.admitDraftEdit(reviewed.manifest)
            try validate(semantic)
            let seats = semantic.agents.map(\.id)
            guard Set(assignment.seatIDs) == Set(seats), assignment.seatIDs.count == seats.count,
                Set(assignment.occupants.keys) == Set(seats) else { throw malformed("Cast every seat exactly once.") }
            var draft = reviewed.manifest
            if let modelID { StudyProtocolAuthoring.applyBaseModelChoice(modelID, to: &draft) }
            if let temperature { draft.temperature = temperature }
            if let maxTokens { draft.maxTokens = maxTokens }
            try ManifestDraftEdits.setSamplingProtocol(temperature: draft.temperature, maxTokens: draft.maxTokens,
                experimentName: draft.name, manifest: &draft)
            for occupant in assignment.occupants.values {
                if case .agent(_, let path, let hash) = occupant {
                    let agent = try StudyAgentAuthoring.reviewArtifact(path: path, workspaceRoot: root, expectedFileSHA256: hash)
                    guard agent.record.artifact.baseModelID == draft.modelID else { throw malformed("A seat agent uses another base model.") }
                }
            }
            let source: Document
            if let semanticPath {
                source = try inspect(path: semanticPath, root: root)
                guard source.document == semantic, expectedPanel == nil || source.fileSHA256 == expectedPanel else {
                    throw ExperimentError.refusing(.artifactPin, "The semantic panel changed after selection.", repair: "Reload and review the panel before saving its casting.")
                }
            } else {
                let record = try publish(semantic, root: root).record
                let prefix = try ManifestFileTransaction.canonicalPath(root) + "/"
                source = try inspect(path: String(record.url.path.dropFirst(prefix.count)), root: root)
            }
            draft.studyKind = .multiAgent
            draft.studyType = StudyIntent.multiAgent.rawValue
            _ = try ordinary("prompts/panels/compiled/admission.json", root: root)
            try SeatCasting.compile(assignment, semantic: semantic, semanticPath: source.path,
                into: &draft, fileSlug: fileSlug, workspaceRoot: root)
            draft.multiAgentSemanticScenarioHash = source.fileSHA256
            return try DraftAuthoringTransaction.replace(draft, reviewed: reviewed)
        }
    }

    static func malformed(_ reason: String) -> ExperimentError {
        .malformed(reason, repair: "Inspect the study, panel and agent files; supply their reviewed digests and explicit casting.")
    }
}

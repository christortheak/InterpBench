import Foundation

/// Publishes a draft from a retained design review and an explicit casting.
/// Scientific derivation stays in the same scope, arm and seat owners as the
/// study editor. This command owns admission, workspace identity and publication.
public enum StudyDesignInstantiation {
    public enum Casting: Sendable {
        case agents([AgentArtifactSnapshot])
        case seating(SeatAssignment)

        var cell: StudyTemplateStore.Cell {
            switch self {
            case .agents(let agents): .agents(agents.map(\.record))
            case .seating(let assignment): .seating(assignment)
            }
        }
    }

    public static func mintBatch(reviewed: StudyDesignSnapshot, castings: [Casting],
                                 names: [String?] = [], batchID: String? = nil,
                                 onRow: ((Int, StudyTemplateStore.RowMint) -> Void)? = nil) -> StudyTemplateStore.BatchMint {
        StudyTemplateStore.mintRows(count: castings.count, names: names, batchID: batchID, onRow: onRow) { index, name, batch in
            try instantiate(reviewed: reviewed, casting: castings[index], studyName: name, batchGroup: batch).manifest
        }
    }

    @discardableResult
    public static func instantiate(reviewed: StudyDesignSnapshot, casting: Casting,
                                   studyName: String? = nil, batchGroup: String? = nil) throws -> DraftAuthoringSnapshot {
        try StudyDesignAuthoring.withReviewedDesign(reviewed) { _ in
            let root = reviewed.workspaceRoot
            let template = reviewed.template
            let storage = ExperimentRepository(workspaceRoot: root)
            try requireOrdinaryDirectories(["experiments"], root: root)
            var draft = template.study
            draft.createdAt = ISO8601DateFormatter().string(from: Date())
            draft.status = .draft
            draft.templateProvenance = .init(template: template.name,
                templateHash: StudyTemplateStore.hash(template), batchGroup: batchGroup)
            let base = ExperimentStore.sanitizedExperimentName(studyName ?? "\(template.name)-\(casting.cell.descriptor)")
            guard !base.isEmpty else {
                throw ExperimentError.malformed("The requested casting needs a usable study name.", repair: "Supply a nonempty study name.")
            }
            var name = base
            var suffix = 1
            while try entryExists(storage.directory.appending(component: name)) {
                suffix += 1
                name = "\(base)-\(suffix)"
            }
            draft.name = name
            return try ManifestFileTransaction.withLock(manifestURL: storage.manifestURL(name), workspaceRoot: root) {
                try requireOrdinaryDirectories(["experiments"], root: root)
                guard try !entryExists(storage.directory.appending(component: name)) else {
                    throw ExperimentError.refusing(.staleManifest, "Another operation occupied the new study's directory.",
                        repair: "Inspect the workspace and choose a fresh study name before retrying creation.")
                }
                try ManifestFileTransaction.requireCurrent(.absent, at: storage.manifestURL(name))
                let prompts = try StudyTemplateStore.reviewedTaskPrompts(draft, workspaceRoot: root)
                if let scope = draft.outcomeInstrumentScope {
                    try OutcomeInstrumentScopeAuthoring.apply(responseFormats: scope.responseFormats,
                        into: &draft, workspaceRoot: root, reviewedPrompts: prompts)
                }
                switch casting {
                case .agents(let agents):
                    guard template.intent != .multiAgent else {
                        throw ExperimentError.malformed("A panel design requires a seat assignment.",
                            repair: "Fill each design seat, using baseline explicitly for an unsteered seat.")
                    }
                    for agent in agents {
                        guard try ManifestFileTransaction.canonicalPath(agent.workspaceRoot) == ManifestFileTransaction.canonicalPath(root) else {
                            throw ExperimentError.refusing(.artifactPin, "The reviewed agent belongs to another workspace.",
                                repair: "Import and inspect that agent in the design's workspace before casting it.")
                        }
                        try ExperimentStore.attachAgent(agent.record, into: &draft, workspaceRoot: root,
                            expectedArtifactFileSHA256: agent.file.sha256)
                    }
                case .seating(let assignment):
                    guard let ref = template.semanticScenario else {
                        throw ExperimentError.malformed("The design declares no semantic panel for this seat assignment.",
                            repair: "Inspect the intended panel design, or use an agent comparison casting.")
                    }
                    let semantic = try StudyTemplateStore.loadSemanticPanel(ref, workspaceRoot: root)
                    let seats = PanelComposition.seatIDs(semantic)
                    guard Set(assignment.seatIDs).count == assignment.seatIDs.count,
                        Set(assignment.seatIDs) == Set(seats), Set(assignment.occupants.keys) == Set(seats) else {
                        throw ExperimentError.malformed("The casting must name every design seat exactly once, with no extra occupants.",
                            repair: "Inspect the design's semantic panel and fill its exact seat IDs.")
                    }
                    var pinnedAssignment = assignment
                    for (seat, occupant) in assignment.occupants {
                        if case .agent(_, let path, let hash) = occupant {
                            let relativePath = try workspaceAgentPath(path, root: root)
                            let agent = try StudyAgentAuthoring.reviewArtifact(path: relativePath, workspaceRoot: root, expectedFileSHA256: hash)
                            guard agent.record.artifact.baseModelID == draft.modelID else {
                                throw ExperimentError.refusing(.artifactPin, "A seat agent uses a different base model from this design.",
                                    repair: "Choose a reviewed agent built on the design's base model.")
                            }
                            pinnedAssignment.occupants[seat] = .agent(name: agent.record.artifact.name,
                                artifactPath: agent.path, artifactHash: agent.file.sha256)
                        }
                    }
                    try requireOrdinaryDirectories(["prompts", "panels", "compiled"], root: root)
                    try SeatCasting.compile(pinnedAssignment, semantic: semantic, semanticPath: ref.path,
                        into: &draft, workspaceRoot: root)
                    // The source pin describes the same reviewed bytes that
                    // supplied compilation, even if an external writer races it.
                    draft.multiAgentSemanticScenarioHash = ref.hash
                }
                try ExperimentStore.save(draft, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
                return try DraftAuthoringSnapshot(workspaceRoot: root, name: name)
            }
        }
    }
    /// Existing study castings can contain absolute pins (including macOS path
    /// aliases). Normalize only paths that belong to this captured workspace;
    /// the ordinary runs-file and reviewed-hash checks still follow below.
    static func workspaceAgentPath(_ path: String, root: URL) throws -> String {
        guard path.hasPrefix("/") else { return path }
        let prefix = try ManifestFileTransaction.canonicalPath(root) + "/"
        let canonical = try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: path))
        guard canonical.hasPrefix(prefix) else {
            throw ExperimentError.refusing(.artifactPin, "The seat's agent artifact is outside the design workspace.",
                repair: "Import the agent evidence into this workspace and review its local artifact before creating sibling studies.")
        }
        return String(canonical.dropFirst(prefix.count))
    }

    private static func entryExists(_ url: URL) throws -> Bool {
        do { _ = try FileManager.default.attributesOfItem(atPath: url.path); return true }
        catch CocoaError.fileReadNoSuchFile { return false }
    }

    private static func requireOrdinaryDirectories(_ components: [String], root: URL) throws {
        var path = URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(root))
        for component in components {
            path.append(component: component)
            do {
                guard try FileManager.default.attributesOfItem(atPath: path.path)[.type] as? FileAttributeType == .typeDirectory else {
                    throw ExperimentError.malformed("Study creation requires ordinary workspace directories.",
                        repair: "Inspect the study and compiled-panel destinations; do not redirect authoring into immutable evidence through links.")
                }
            } catch CocoaError.fileReadNoSuchFile { return }
        }
    }

}

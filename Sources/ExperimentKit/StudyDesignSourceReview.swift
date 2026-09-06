import Foundation

/// A source study and the reviewed scenario bytes from which a reusable design
/// will be derived. Saving a design never mutates this source, including when it
/// is frozen. The file versions remain outside both scientific schemas.
public struct StudyDesignSourceReview: Sendable {
    public let study: DraftAuthoringSnapshot
    public let panel: PanelComposition.LegacyHoist?
    private let scenarioURL: URL?
    private let scenarioFile: ManifestFileSnapshot?

    public init(study: DraftAuthoringSnapshot) throws {
        self.study = study
        if let path = study.manifest.multiAgentScenarioPath, !path.isEmpty {
            let url = ExperimentStore.resolveProjectPath(path, root: study.workspaceRoot)
            try Self.requireWorkspaceInput(url, root: study.workspaceRoot)
            let file = try ManifestFileTransaction.snapshot(at: url)
            if let pin = study.manifest.multiAgentScenarioHash, pin != file.sha256 {
                throw ExperimentError.refusing(.artifactPin, "The source study's panel no longer matches its pinned bytes.",
                    repair: "Restore the verified panel input, or deliberately revise and review a draft before deriving a design.")
            }
            let scenario = try JSONDecoder().decode(MultiAgentScenario.self, from: file.data)
            panel = try PanelComposition.hoistLegacyScenario(scenario, workspaceRoot: study.workspaceRoot)
            scenarioURL = url
            scenarioFile = file
        } else {
            panel = nil
            scenarioURL = nil
            scenarioFile = nil
        }
    }

    /// Serialize the reviewed source against participating study writers. The
    /// caller publishes a separate design; frozen-study edit admission is not
    /// applicable to this read-only source authority.
    func withCurrentSource<T>(_ body: () throws -> T) throws -> T {
        let root = study.workspaceRoot
        let url = ExperimentRepository(workspaceRoot: root).manifestURL(study.manifest.name)
        return try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: root) {
            try ManifestFileTransaction.requireCurrent(.sha256(study.file.sha256), at: url)
            if let scenarioURL, let scenarioFile {
                try Self.requireWorkspaceInput(scenarioURL, root: root)
                guard try ManifestFileTransaction.snapshot(at: scenarioURL).sha256 == scenarioFile.sha256 else {
                    throw ExperimentError.refusing(.artifactPin, "The source panel changed after the design source was reviewed.",
                        repair: "Inspect the study and its panel again, then reconstruct the intended design save.")
                }
            }
            return try body()
        }
    }

    private static func requireWorkspaceInput(_ url: URL, root: URL) throws {
        let prefix = try ManifestFileTransaction.canonicalPath(root) + "/"
        guard try ManifestFileTransaction.canonicalPath(url).hasPrefix(prefix) else {
            throw ExperimentError.refusing(.artifactPin, "The source study's panel is outside its workspace.",
                repair: "Import and pin the intended panel in the study workspace before saving a design.")
        }
    }
}

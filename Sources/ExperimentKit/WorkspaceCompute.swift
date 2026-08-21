import Foundation

/// What a workspace COMPUTES ON — declared once, not re-derived per verb.
///
/// The intended working model has two shapes, and only two:
///
/// - **Cluster workspace.** All computation runs on the Python/PyTorch
///   engine; the Mac manages data. Studies, manifests, and imported evidence
///   live locally, the Mac freezes and analyzes, and MLX is never used. This
///   is the shape real studies take.
/// - **Local MLX workspace.** The Mac's own engine computes. Toy models,
///   pipeline shakedowns, tests.
///
/// Before this type, the app never held that fact. Each verb re-derived
/// intent from the pairing heuristic (`isKnownUnpairedServerWorkspace`), and
/// they disagreed: freeze learned to match evidence against the SERVER's
/// substrate in Mac-authority mode (`freezeEvidenceRunSubstrate`, 2026-07-21),
/// while promotion, the vector-artifact matcher, and the epoch guard kept
/// keying on `RepEReader.substrate` — a Swift compile-time constant. In a
/// cluster workspace that made the workspace's own vectors "foreign" and
/// refused a promotion that was entirely legitimate (observed 2026-07-26).
///
/// The binding is a DECLARATION about the workspace, not a guess about the
/// current connection: it survives the server being offline, unpaired, or
/// reachable at a different address, none of which change what the workspace
/// is for.
public enum WorkspaceCompute: String, Sendable, Codable, CaseIterable {
    /// Computation happens on the Python/PyTorch engine (cluster or a local
    /// server process); this Mac manages data and evidence.
    case cluster
    /// Computation happens in-process on MLX.
    case localMLX = "local-mlx"

    /// The substrate whose artifacts and evidence this workspace treats as
    /// NATIVE — what `substrate` fields in its runs and vector sidecars
    /// should read.
    public var substrate: String {
        switch self {
        case .cluster: WorkspaceScoping.serverSubstrate
        case .localMLX: ExperimentStore.evidenceSubstrate
        }
    }

    /// Whether MLX execution should be offered at all. A cluster workspace
    /// hides it rather than refusing it later: an option that is always wrong
    /// is worse than an absent one.
    public var allowsLocalExecution: Bool { self == .localMLX }

    public var label: String {
        switch self {
        case .cluster: "Cluster (Python/PyTorch)"
        case .localMLX: "Local (MLX)"
        }
    }

    // MARK: Persistence

    /// Lives beside the other workspace-local state (`bundles/`, `imports/`,
    /// `downloads/`), not in `WORKSPACE.md` — it is machine-read config, and
    /// the marker is prose for humans.
    static let configPath = [".steerlab", "workspace.json"]

    private struct Config: Codable {
        var computeSubstrate: WorkspaceCompute
    }

    private static func configURL(root: URL) -> URL {
        configPath.reduce(root) { $0.appending(component: $1) }
    }

    /// The workspace's DECLARED binding, or nil when it has never declared
    /// one (every workspace made before this type existed).
    public static func declared(root: URL) -> WorkspaceCompute? {
        guard let data = try? Data(contentsOf: configURL(root: root)) else {
            return nil
        }
        return try? JSONDecoder().decode(Config.self, from: data).computeSubstrate
    }

    /// Record the binding. Idempotent; creates `.steerlab/` as needed.
    public static func declare(_ compute: WorkspaceCompute, root: URL) throws {
        let url = configURL(root: root)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(Config(computeSubstrate: compute)).write(to: url)
    }

    // MARK: Inference for workspaces that predate the declaration

    /// What the workspace's OWN RUNS say it computes on.
    ///
    /// Deliberately evidence-based rather than a default: an existing
    /// cluster workspace holding fifty server runs should not be told it is
    /// a local MLX workspace because a config file is missing. Ties and
    /// empty trees return nil — the caller falls back, and the researcher
    /// can declare.
    ///
    /// Not persisted. A written-down guess is indistinguishable from a
    /// decision the researcher made, and this one should stay visibly
    /// provisional until they confirm it.
    public static func inferred(root: URL) -> WorkspaceCompute? {
        let census = substrateCensus(root: root)
        let server = census[WorkspaceScoping.serverSubstrate] ?? 0
        let local = census[ExperimentStore.evidenceSubstrate] ?? 0
        if server > local { return .cluster }
        if local > server { return .localMLX }
        return nil
    }

    /// Count run directories by the substrate their `config.json` records.
    static func substrateCensus(root: URL) -> [String: Int] {
        let runs = root.appending(component: "runs")
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: runs, includingPropertiesForKeys: nil)
        else { return [:] }
        var census: [String: Int] = [:]
        for entry in entries {
            guard
                let data = try? Data(
                    contentsOf: entry.appending(component: "config.json")),
                let object = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
                let substrate = object["substrate"] as? String
            else { continue }
            census[substrate, default: 0] += 1
        }
        return census
    }

    /// The binding in force: declared, else inferred from the workspace's own
    /// runs, else local MLX (a fresh workspace with no evidence either way).
    public static func resolved(root: URL) -> WorkspaceCompute {
        declared(root: root) ?? inferred(root: root) ?? .localMLX
    }

    /// Whether the binding in force was actually declared, or is standing in
    /// for one. Surfaces in the UI so an inferred binding reads as a
    /// suggestion rather than a setting the researcher chose.
    public static func isDeclared(root: URL) -> Bool {
        declared(root: root) != nil
    }
}

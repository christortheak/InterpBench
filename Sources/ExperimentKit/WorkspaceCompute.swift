import Foundation

/// What a workspace COMPUTES ON — declared once, not re-derived per verb.
///
/// The BINDING has two values, and only two, because it answers one
/// question — whose artifacts and evidence are native to this workspace:
///
/// - **`cluster`: the Python engine.** All computation runs on the
///   Python/PyTorch engine; the Mac manages data. Studies, manifests, and
///   imported evidence live locally, the Mac freezes and analyzes, and MLX
///   is never used. The engine may be on another machine OR on this Mac's
///   own GPU — both are this binding, and both run every method.
/// - **`local-mlx`: the engine built into the app.** The Mac's own MLX
///   engine computes: core steering studies on small models, with nothing
///   to install beyond a model.
///
/// Working on a laptop alone is a supported way to do real studies
/// (maintainer's ruling, 2026-10-04). What a researcher picks between is
/// therefore three plainly named choices, `ComputeChoice`; the binding below
/// is what two of them share, and its file format is unchanged.
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

    /// The binding in plain words. The three choices a researcher actually
    /// picks between are `ComputeChoice`; this names the two engines for the
    /// places that speak about the binding itself.
    public var label: String {
        switch self {
        case .cluster: "The Python engine"
        case .localMLX: ComputeChoice.macQuickStart.title
        }
    }

    // MARK: Persistence

    /// Lives beside the other workspace-local state (`bundles/`, `imports/`,
    /// `downloads/`), not in `WORKSPACE.md` — it is machine-read config, and
    /// the marker is prose for humans.
    static let configPath = [".steerlab", "workspace.json"]

    /// The binding's key — the file's whole content before `Location`.
    static let substrateKey = "computeSubstrate"
    /// The ONE optional key added beside it (W1-D). Additive: the binding
    /// above is unchanged, a file without this key loads exactly as before,
    /// and readers that do not know the key ignore it.
    static let locationKey = "computeLocation"

    /// WHERE the Python engine runs, for a `cluster` binding: on this Mac's
    /// own GPU, or on another machine. A convenience for the interface — it
    /// lets the app show which of its three choices the researcher picked —
    /// and nothing more: the lifecycle reads only the binding, and both
    /// locations name the same engine and the same native artifacts.
    public enum Location: String, Sendable, Codable, CaseIterable {
        case thisMac = "this-mac"
        case anotherMachine = "another-machine"
    }

    private struct Config: Codable {
        var computeSubstrate: WorkspaceCompute
    }

    private static func configURL(root: URL) -> URL {
        configPath.reduce(root) { $0.appending(component: $1) }
    }

    /// The config file as a plain object, or empty when it is absent or
    /// unreadable. Read this way so a write can keep keys this build does
    /// not know about.
    private static func configObject(root: URL) -> [String: Any] {
        guard let data = try? Data(contentsOf: configURL(root: root)),
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else { return [:] }
        return object
    }

    /// The workspace's DECLARED binding, or nil when it has never declared
    /// one (every workspace made before this type existed).
    public static func declared(root: URL) -> WorkspaceCompute? {
        guard let data = try? Data(contentsOf: configURL(root: root)) else {
            return nil
        }
        return try? JSONDecoder().decode(Config.self, from: data).computeSubstrate
    }

    /// The recorded location, when the binding is `cluster` and the key is
    /// present with a value this build knows. Nil otherwise — including for
    /// every workspace declared before the key existed, and for a value a
    /// later build wrote that this one does not recognise. Never an error:
    /// the binding still loads.
    public static func declaredLocation(root: URL) -> Location? {
        guard declared(root: root) == .cluster,
            let raw = configObject(root: root)[locationKey] as? String
        else { return nil }
        return Location(rawValue: raw)
    }

    /// Record the binding. Idempotent; creates `.steerlab/` as needed.
    ///
    /// A recorded location is kept while the binding stays `cluster` and
    /// dropped when it becomes local, where it has no meaning.
    public static func declare(_ compute: WorkspaceCompute, root: URL) throws {
        try write(
            compute,
            location: compute == .cluster ? declaredLocation(root: root) : nil,
            root: root)
    }

    /// Record one of the three choices: its binding, and — for the two that
    /// share the `cluster` binding — which of them it was.
    public static func declare(_ choice: ComputeChoice, root: URL) throws {
        try write(choice.binding, location: choice.location, root: root)
    }

    /// The one writer. Keeps every key it does not own, sets the binding,
    /// and sets or removes the location. With no location and no foreign
    /// keys the bytes are exactly what this file has always held.
    private static func write(
        _ compute: WorkspaceCompute, location: Location?, root: URL
    ) throws {
        let url = configURL(root: root)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var object = configObject(root: root)
        object[substrateKey] = compute.rawValue
        if let location {
            object[locationKey] = location.rawValue
        } else {
            object.removeValue(forKey: locationKey)
        }
        try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys]
        ).write(to: url)
    }

    /// The choice this workspace's declaration names, or nil when it has
    /// never declared one. `activeEngineIsThisMac` settles a `cluster`
    /// binding that recorded no location (see `ComputeChoice.init`).
    public static func declaredChoice(
        root: URL, activeEngineIsThisMac: Bool = false
    ) -> ComputeChoice? {
        guard let binding = declared(root: root) else { return nil }
        return ComputeChoice(
            binding: binding, location: declaredLocation(root: root),
            activeEngineIsThisMac: activeEngineIsThisMac)
    }

    /// The choice in force: declared, else read from the binding the
    /// workspace's own runs imply, else the quick start.
    public static func resolvedChoice(
        root: URL, activeEngineIsThisMac: Bool = false
    ) -> ComputeChoice {
        declaredChoice(root: root, activeEngineIsThisMac: activeEngineIsThisMac)
            ?? ComputeChoice(
                binding: resolved(root: root), location: nil,
                activeEngineIsThisMac: activeEngineIsThisMac)
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

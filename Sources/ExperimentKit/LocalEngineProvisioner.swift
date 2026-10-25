import CryptoKit
import Foundation
import Observation

// =============================================================================
// WP3 — the Local Engine flow: "bootstrap, but localhost".
//
// The cluster path has had a real provisioner for a year: `bootstrap.sh` runs
// ON the far machine, creates an environment from a pinned lock, stamps what it
// did, and reports a step ledger the wizard renders. The LOCAL Python engine
// had none of that. It had a shell script that assumed a code checkout, a
// system `python3`, and a researcher who would sit through `pip install torch`
// with no idea it was coming.
//
// This file is the same shape as the cluster bootstrap, aimed at 127.0.0.1:
//
//   1. ENGINE SOURCE   a checkout if there is one; otherwise MATERIALIZE the
//                      bundled ServerPayload/ to ~/SteerLab/Engine.
//   2. INTERPRETER     `uv` (pinned, hash-verified) → a managed CPython.
//   3. ENVIRONMENT     `uv venv` + the committed platform lock.
//   4. SERVE           the same posture-respecting `serve` invocation the
//                      one-click script uses.
//   5. ACCEPTANCE      /api/info, then `site qualify`, rendered row by row.
//
// Three properties are load-bearing, and each is a way this could otherwise do
// harm:
//
//   * RESUMABLE AND IDEMPOTENT. Every step observes reality and decides
//     `satisfied` / `act` / `blocked` BEFORE doing anything. Re-running after a
//     failure (or a quit) continues; re-running when everything is already
//     there does nothing but re-check and re-qualify. The decision layer
//     (`LocalEngineDecisions`) is pure over a `LocalEngineObservation`, so all
//     of it is testable without a network, a venv, or a GPU.
//   * HONEST ABOUT DOWNLOADS. The plan names every byte it intends to fetch
//     BEFORE the first one moves — uv (~18 MB, exact), CPython (~50 MB),
//     wheels including torch (~2 GB). A researcher on a hotel connection gets
//     to say no.
//   * NEVER WRITES INTO THE BUNDLE. Materialization COPIES out of
//     `Contents/Resources/ServerPayload`; the first write into a signed bundle
//     breaks its seal. The destination is `CodeResources.EngineRoot`, which is
//     a checkout's poor cousin BY CONSTRUCTION (no `Package.swift`) and can
//     never be mistaken for one.
//
// What this file does NOT do: change the existing developer path. When
// `Server/.venv.nosync` already exists, step 3 is `satisfied` and
// `LocalServerController.start` keeps running `scripts/start-local-server.sh`
// exactly as before.
// =============================================================================

// MARK: - Pinned tooling

/// The `uv` build this flow will download when the machine has none.
///
/// WHY uv AT ALL: on a clean Mac `/usr/bin/python3` is a stub that triggers the
/// Command Line Tools installer — a modal system dialog, a multi-GB download,
/// and an Xcode licence prompt, in the middle of what the researcher thinks is
/// "start the local server". uv ships a standalone CPython, so the flow never
/// touches the system Python at all.
///
/// PROVENANCE of the constants below (recorded 2026-08-20):
///   * version 0.12.5 is the latest release AND the floor of this project's own
///     dev pin, `Server/pyproject.toml` `[project.optional-dependencies] dev =
///     ["uv>=0.12.5,<0.13"]` — the same resolver identity that generated the
///     committed locks (`Server/scripts/update-locks.sh`). Keeping the two in
///     the same range means the tool that INSTALLS a lock is the tool that
///     could have PRODUCED it.
///   * the sha256 was taken from the artifact itself, downloaded from the URL
///     below on 2026-08-20, and cross-checked against the release's published
///     `uv-aarch64-apple-darwin.tar.gz.sha256` sidecar. Both agree:
///         5bb0e5fe008a773c3dbcb97ff79cd89e1241464fe9d2f986d52ad8f1b037bd62
///   * byteCount 18518284 is the GitHub release asset's recorded size, used
///     only for the "this will download about N" preamble.
///
/// TO BUMP: download the new artifact, verify it against the release's own
/// `.sha256` sidecar, paste BOTH numbers here, and say in the commit message
/// which release you took them from. Never paste a hash you did not compute.
public enum PinnedUV: Sendable {
    public static let version = "0.12.5"
    /// The minimum a SYSTEM uv must report before this flow will use it
    /// instead of downloading. Same value as the pin: older uv releases have
    /// changed `uv python install` behavior, and a flow that silently drives
    /// an unknown-older tool is not a pinned flow.
    public static let minimumUsableVersion = "0.12.5"
    public static let platformSlug = "aarch64-apple-darwin"
    public static let archiveName = "uv-\(platformSlug).tar.gz"
    public static let sha256 =
        "5bb0e5fe008a773c3dbcb97ff79cd89e1241464fe9d2f986d52ad8f1b037bd62"
    public static let byteCount = 18_518_284
    public static var downloadURL: URL {
        URL(string: "https://github.com/astral-sh/uv/releases/download/"
            + "\(version)/\(archiveName)")!
    }
    /// Path of the `uv` executable INSIDE the extracted archive.
    public static let archiveExecutableSubpath = "uv-\(platformSlug)/uv"
}

/// The CPython minor uv installs and the venv is built from.
///
/// 3.12 is not a preference — it is what the committed macOS lock was RESOLVED
/// for (`Server/requirements-macos-arm64.lock`: "CPython 3.12", produced by
/// `update-locks.sh --python-version 3.12`) and what `bootstrap.sh` creates on
/// a cluster node. Building the venv from any other minor would install a set
/// of wheels the lock never resolved.
public enum PinnedCPython: Sendable {
    public static let minor = "3.12"
    /// Rough download size for the honesty preamble (a managed CPython
    /// tarball is ~50 MB; the exact figure is uv's business, not a pin).
    public static let approximateByteCount = 52_000_000
}

/// Where the flow's own tools and the managed engine live.
///
/// Both are created ON DEMAND, and neither is part of `HomeLayout`'s minimal
/// layout: `steerlab-cli init` materializes what a RESEARCHER needs
/// (`Workspaces/`, `Sites/`), and an empty `tools/` folder promising a
/// downloader that has not run would be layout fiction.
public enum LocalEngineLayout: Sendable {

    /// `~/SteerLab/tools` (test-overridable via the engine-root seam's parent).
    public static var toolsDirectory: URL {
        CodeResources.defaultEngineRoot
            .deletingLastPathComponent().appending(component: "tools")
    }

    /// `~/SteerLab/tools/uv`.
    public static var managedUV: URL { toolsDirectory.appending(component: "uv") }

    /// The platform lock this Mac installs from.
    public static let lockFileName = "requirements-macos-arm64.lock"

    /// The stamp written INSIDE the venv recording which lock produced it, so
    /// "does this environment match the committed lock?" is a file read rather
    /// than a re-resolution. Inside the venv deliberately: delete the venv and
    /// the claim goes with it.
    public static let lockStampFileName = "steerlab-lock-stamp.json"

    /// Lowercase hex sha256 of a file's bytes; nil when it cannot be read.
    public static func sha256Hex(ofFileAt url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// sha256 of the lock THIS app build ships in its engine payload — the
    /// lock a Refresh Engine would copy — or nil when the build ships no
    /// payload or no lock. Read without materializing or starting anything,
    /// so the first launch after an update can ask "will the packages need
    /// reinstalling?" before the researcher has touched the engine. Both
    /// payload spellings are accepted, as `materialize` accepts them.
    public static func payloadLockSHA256() -> String? {
        guard let payload = try? CodeResources.serverPayload() else { return nil }
        for candidate in [
            payload.appending(component: lockFileName),
            payload.appending(components: "Server", lockFileName),
        ] {
            if let sha = sha256Hex(ofFileAt: candidate) { return sha }
        }
        return nil
    }
}

/// What a venv was built from. Absent on venvs created before WP3 (and on any
/// created by `start-local-server.sh`, which uses pip and floors) — absence is
/// reported as UNVERIFIED, never as drift, because forcing a multi-GB reinstall
/// on a working developer environment would be the worst possible reading of
/// "no evidence".
public struct LocalEngineLockStamp: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var lockFile: String
    public var lockSHA256: String
    public var installedAt: String
    public var uvVersion: String?
    public var pythonVersion: String?
    /// Additive (2026-10-25): true while Reinstall Packages is replacing the
    /// packages, and left true when it stops partway. `uv pip sync` removes
    /// the engine package before the editable install puts it back, so an
    /// interrupted reinstall can leave an environment that cannot import the
    /// engine; the environment step refuses to serve it and names the repair.
    /// Absent on every stamp written before this field existed.
    public var incomplete: Bool?

    public init(
        schemaVersion: Int = LocalEngineLockStamp.currentSchemaVersion,
        lockFile: String, lockSHA256: String, installedAt: String,
        uvVersion: String? = nil, pythonVersion: String? = nil,
        incomplete: Bool? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.lockFile = lockFile
        self.lockSHA256 = lockSHA256
        self.installedAt = installedAt
        self.uvVersion = uvVersion
        self.pythonVersion = pythonVersion
        self.incomplete = incomplete
    }

    /// The stamp inside `venv`, when there is one that decodes.
    public static func read(venv: URL) -> LocalEngineLockStamp? {
        let url = venv.appending(component: LocalEngineLayout.lockStampFileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(LocalEngineLockStamp.self, from: data)
    }
}

/// One located `uv` binary.
public struct LocalEngineTool: Sendable, Equatable {
    public let path: String
    /// As reported by `uv --version`, e.g. "0.12.5"; nil when it did not
    /// answer (a broken or half-extracted binary).
    public let version: String?

    public init(path: String, version: String?) {
        self.path = path
        self.version = version
    }
}

// MARK: - Steps, actions, and the observation they decide over

/// The five steps, in order. Each one CHECKS before it acts, which is what
/// makes the whole sequence resumable.
public enum LocalEngineStep: String, CaseIterable, Sendable, Equatable {
    case engineSource
    case interpreter
    case environment
    case serve
    case acceptance

    public var title: String {
        switch self {
        case .engineSource: return "Engine source"
        case .interpreter: return "Python interpreter"
        case .environment: return "Dependency environment"
        case .serve: return "Local server"
        case .acceptance: return "Acceptance checks"
        }
    }

    /// The precondition this step tests, in one sentence — rendered beside the
    /// step so "skipped" never looks like "not done".
    public var precondition: String {
        switch self {
        case .engineSource:
            return "a writable engine tree exists (a code checkout, or a "
                + "materialized engine at ~/SteerLab/Engine)"
        case .interpreter:
            return "a usable uv is present and a managed CPython "
                + "\(PinnedCPython.minor) is installed"
        case .environment:
            return "Server/.venv.nosync exists and matches the committed lock"
        case .serve:
            return "a SteerLab server is answering on the chosen port"
        case .acceptance:
            return "/api/info identifies the engine and site qualify passes"
        }
    }
}

/// One unit of work a step may perform. Every case names what it downloads (0
/// for the ones that download nothing), because the preamble is assembled from
/// these rather than from a hand-maintained sentence that can drift.
public enum LocalEngineAction: Sendable, Equatable {
    case materializeEngine(destination: String)
    case refreshEngine(destination: String, from: String?, to: String?)
    case downloadUV(destination: String)
    case installCPython(minor: String)
    case createVenv(path: String)
    case installLock(lock: String, editableTarget: String)
    /// Reinstall Packages: bring an EXISTING, stamped venv in line with the
    /// lock the tree now carries. Offered, never planned (see
    /// `LocalEngineDecisions.environmentSyncOffer`).
    case syncEnvironment(
        lock: String, editableTarget: String, installedSHA256: String, lockSHA256: String)
    case startServer(port: Int)
    case runQualification

    /// The wheel set (torch, transformers, accelerate, sae-lens, …): the big
    /// download, and the one people are surprised by. One figure for the
    /// first install and for a reinstall, so the two can never disagree.
    public static let lockInstallEstimatedBytes = 2_000_000_000

    public var title: String {
        switch self {
        case .materializeEngine(let destination):
            return "Copy the bundled Python engine to \(destination)"
        case .refreshEngine(let destination, let from, let to):
            return "Refresh the engine at \(destination) "
                + "(\(from ?? "unstamped") → \(to ?? "this build"))"
        case .downloadUV(let destination):
            return "Download uv \(PinnedUV.version) to \(destination) "
                + "(sha256-verified)"
        case .installCPython(let minor):
            return "Install a managed CPython \(minor)"
        case .createVenv(let path):
            return "Create the virtual environment at \(path)"
        case .installLock(let lock, _):
            return "Install the pinned dependencies from \(lock)"
        case .syncEnvironment(let lock, let editableTarget, let installed, let current):
            return "Reinstall the packages in \(editableTarget)/.venv.nosync to match "
                + "\(URL(filePath: lock).lastPathComponent) "
                + "(sha256 \(String(installed.prefix(8))) → \(String(current.prefix(8))))"
        case .startServer(let port):
            return "Start the local server on 127.0.0.1:\(port)"
        case .runQualification:
            return "Run site qualify against the local engine"
        }
    }

    /// Bytes this action is expected to pull over the network.
    public var estimatedDownloadBytes: Int {
        switch self {
        case .downloadUV: return PinnedUV.byteCount
        case .installCPython: return PinnedCPython.approximateByteCount
        case .installLock, .syncEnvironment: return Self.lockInstallEstimatedBytes
        case .materializeEngine, .refreshEngine, .createVenv, .startServer,
            .runQualification:
            return 0
        }
    }

    /// True when the estimate is a rough figure rather than a pinned one.
    public var downloadEstimateIsApproximate: Bool {
        switch self {
        case .downloadUV: return false
        default: return estimatedDownloadBytes > 0
        }
    }

    /// True when the estimate is a CEILING: a reinstall fetches only the
    /// packages that changed and uv reuses what its cache already holds, so
    /// the whole wheel set is the most it can download, not what it will.
    public var downloadEstimateIsUpperBound: Bool {
        if case .syncEnvironment = self { return true }
        return false
    }
}

/// What a step decided, having looked.
public enum LocalEngineStepState: Sendable, Equatable {
    /// The precondition already holds; the step does nothing. The string says
    /// WHY it was skipped, so a skipped step reads as verified.
    case satisfied(String)
    /// Satisfied, but with something the researcher should know (an unstamped
    /// venv, an engine tree older than the running app).
    case satisfiedWithAdvisory(String, advisory: String)
    /// Work to do, in order.
    case act([LocalEngineAction])
    /// Cannot proceed, and no amount of retrying will change it.
    case blocked(String)

    public var actions: [LocalEngineAction] {
        if case .act(let actions) = self { return actions }
        return []
    }

    public var isBlocked: Bool {
        if case .blocked = self { return true }
        return false
    }

    public var advisory: String? {
        if case .satisfiedWithAdvisory(_, let advisory) = self { return advisory }
        return nil
    }

    /// One line for the UI.
    public var summary: String {
        switch self {
        case .satisfied(let reason): return reason
        case .satisfiedWithAdvisory(let reason, let advisory):
            return reason + " — " + advisory
        case .act(let actions):
            return actions.map(\.title).joined(separator: "; ")
        case .blocked(let reason): return reason
        }
    }
}

/// Everything the decision layer is allowed to look at. Assembled by
/// `LocalEngineEnvironment` (live) or handed over whole (tests).
public struct LocalEngineObservation: Sendable, Equatable {

    /// What tier the engine source resolved to.
    public enum SourceState: Sendable, Equatable {
        /// A real code checkout — richest tier, and untouched by WP3.
        case checkout(path: String)
        /// A materialized engine root, with the revision its stamp records
        /// (nil for a tree with no stamp).
        case engineRoot(path: String, stampRevision: String?)
        /// No tree yet, but this build ships a `ServerPayload` we may copy.
        case materializable(destination: String, payload: String)
        /// Neither, and the reason is a LAYOUT fact, not a missing file.
        case unavailable(reason: String)
    }

    public var source: SourceState
    /// The running app's own build revision (`SLSourceRevision`), for skew.
    public var appRevision: String?
    /// `uv` found on PATH, if any.
    public var systemUV: LocalEngineTool?
    /// `~/SteerLab/tools/uv`, if it is there.
    public var managedUV: LocalEngineTool?
    /// Whether the chosen uv already has a managed CPython of the pinned minor.
    public var managedPythonInstalled: Bool
    /// Whether `<serverDirectory>/.venv.nosync/bin/python` is executable.
    public var venvPythonExists: Bool
    /// The lock stamp recorded inside the venv (nil = unstamped/legacy).
    public var venvLockStamp: LocalEngineLockStamp?
    /// sha256 of the platform lock file on disk; nil when the engine tree
    /// carries no lock (a payload built before locks existed).
    public var lockFileSHA256: String?
    public var lockFilePath: String?
    /// True when something on `port` answers AND identifies as SteerLab.
    public var serverAnswering: Bool
    public var port: Int

    public init(
        source: SourceState, appRevision: String? = nil,
        systemUV: LocalEngineTool? = nil, managedUV: LocalEngineTool? = nil,
        managedPythonInstalled: Bool = false, venvPythonExists: Bool = false,
        venvLockStamp: LocalEngineLockStamp? = nil,
        lockFileSHA256: String? = nil, lockFilePath: String? = nil,
        serverAnswering: Bool = false, port: Int = 8080
    ) {
        self.source = source
        self.appRevision = appRevision
        self.systemUV = systemUV
        self.managedUV = managedUV
        self.managedPythonInstalled = managedPythonInstalled
        self.venvPythonExists = venvPythonExists
        self.venvLockStamp = venvLockStamp
        self.lockFileSHA256 = lockFileSHA256
        self.lockFilePath = lockFilePath
        self.serverAnswering = serverAnswering
        self.port = port
    }
}

// MARK: - The decision layer (pure)

/// Every step's precondition, as a function of the observation and nothing
/// else. No file system, no network, no clock — so the whole state machine is
/// unit-testable against fakes, which is the only way the cold-start path can
/// be exercised on a machine that is already warm.
public enum LocalEngineDecisions {

    // MARK: Version comparison

    /// Numeric dotted-version comparison: -1 / 0 / 1. Non-numeric trailing
    /// junk ("0.12.5+build") is ignored rather than making the whole
    /// comparison fail, and a component that will not parse counts as 0.
    public static func compareVersions(_ lhs: String, _ rhs: String) -> Int {
        func components(_ text: String) -> [Int] {
            text.split(whereSeparator: { $0 == "." || $0 == "-" || $0 == "+" })
                .map { part -> Int in
                    Int(part.prefix { $0.isNumber }) ?? 0
                }
        }
        let a = components(lhs)
        let b = components(rhs)
        for index in 0..<max(a.count, b.count) {
            let left = index < a.count ? a[index] : 0
            let right = index < b.count ? b[index] : 0
            if left != right { return left < right ? -1 : 1 }
        }
        return 0
    }

    /// A located uv is usable when it answered `--version` at all and is not
    /// older than the pin. An unanswering binary is never usable — a
    /// half-extracted file must not be driven.
    public static func isUsable(_ tool: LocalEngineTool?) -> Bool {
        guard let tool, let version = tool.version else { return false }
        return compareVersions(version, PinnedUV.minimumUsableVersion) >= 0
    }

    /// Which uv the flow will drive, given both candidates. The MANAGED one
    /// wins when both are usable: it is the one this flow pinned, verified,
    /// and can reason about, and preferring it means a researcher who upgrades
    /// their system uv does not silently change how the engine is built.
    public static func chosenUV(_ observation: LocalEngineObservation)
        -> LocalEngineTool?
    {
        if isUsable(observation.managedUV) { return observation.managedUV }
        if isUsable(observation.systemUV) { return observation.systemUV }
        return nil
    }

    // MARK: Per-step preconditions

    public static func engineSourceState(_ observation: LocalEngineObservation)
        -> LocalEngineStepState
    {
        switch observation.source {
        case .checkout(let path):
            return .satisfied(
                "running from the code checkout at \(path) — the richest tier; "
                    + "nothing to materialize")
        case .engineRoot(let path, let stampRevision):
            // Skew is STATED and non-blocking: an engine one build behind the
            // app usually still works, and stopping the researcher because two
            // short shas differ would be theatre. Offering the refresh (never
            // performing it silently) is the honest middle.
            guard let appRevision = observation.appRevision else {
                return .satisfied("managed engine at \(path)")
            }
            guard let stampRevision else {
                return .satisfiedWithAdvisory(
                    "managed engine at \(path)",
                    advisory: "this tree carries no engine stamp, so its "
                        + "revision cannot be compared with the app's "
                        + "(\(appRevision)) — choose Refresh Engine to make "
                        + "the two knowable (installed packages are kept)")
            }
            if stampRevision == appRevision {
                return .satisfied(
                    "managed engine at \(path), stamped \(stampRevision) — "
                        + "same revision as this app build")
            }
            return .satisfiedWithAdvisory(
                "managed engine at \(path)",
                advisory: "engine stamped \(stampRevision), app built "
                    + "\(appRevision) — the engine was copied by a "
                    + "different build (an update does this); choose "
                    + "Refresh Engine when convenient (installed packages "
                    + "are kept)")
        case .materializable(let destination, _):
            return .act([.materializeEngine(destination: destination)])
        case .unavailable(let reason):
            return .blocked(reason)
        }
    }

    /// Whether the materialized engine on this Mac was copied by a different
    /// app build, read from its stamp without starting or probing anything —
    /// what the first launch after an update checks before saying "refresh
    /// the engine". False for a checkout, for no engine yet, and for a build
    /// with no revision of its own.
    public static func materializedEngineIsFromAnotherBuild(
        appRevision: String? = CodeResources.bundleSourceRevision
    ) -> Bool {
        guard let appRevision,
            case .engineRoot(let engine)? = CodeResources.localEngineSource()
        else { return false }
        return engine.stamp?.sourceRevision != appRevision
    }

    /// Refresh Engine — OFFERED, never part of the plan and never performed
    /// unasked: the materialized engine was copied by a different app build
    /// (after an update, always) or carries no stamp to compare. Nil for a
    /// checkout, a matching stamp, or no tree yet.
    public static func refreshOffer(_ observation: LocalEngineObservation)
        -> LocalEngineAction?
    {
        guard case .engineRoot(let path, let stampRevision) = observation.source,
            let appRevision = observation.appRevision, stampRevision != appRevision
        else { return nil }
        return .refreshEngine(destination: path, from: stampRevision, to: appRevision)
    }

    /// Whether a venv was installed from a different lock than `lockSHA256`.
    /// FALSE with no stamp (the developer case: absence of evidence is never
    /// drift) and with no lock to compare against.
    public static func lockMovedSinceInstall(
        stamp: LocalEngineLockStamp?, lockSHA256: String?
    ) -> Bool {
        guard let stamp, let lockSHA256 else { return false }
        return stamp.lockSHA256 != lockSHA256
    }

    /// Reinstall Packages — OFFERED, never part of the plan and never
    /// performed unasked: the venv this flow installed is stamped with a
    /// different lock from the one the tree now carries (after Refresh Engine
    /// copied a version whose lock moved, always). Nil when the stamp
    /// matches, when there is no venv yet (Set Up installs one), and — the
    /// developer case — when the venv carries no stamp at all: a venv made by
    /// `scripts/start-local-server.sh` keeps its advisory and is never
    /// offered a reinstall it did not ask for. Offered ONLY for SteerLab's own
    /// materialized engine root (2026-10-25, review): a checkout's
    /// `Server/.venv.nosync` can carry a stamp too (Set Up stamps whatever
    /// tier it installed), but it is the developer's environment, holding
    /// extras the lock does not list (`Server[jlens]`), and `uv pip sync`
    /// would remove them. A checkout keeps the advisory. Also offered when a
    /// previous reinstall stopped partway (`incomplete`). Offered whether or
    /// not the engine is answering; `refusalWhileEngineAnswers` is the gate on
    /// performing it.
    public static func environmentSyncOffer(_ observation: LocalEngineObservation)
        -> LocalEngineAction?
    {
        guard case .engineRoot = observation.source,
            observation.venvPythonExists,
            let lockPath = observation.lockFilePath,
            let lockSHA = observation.lockFileSHA256,
            let stamp = observation.venvLockStamp,
            stamp.incomplete == true
                || lockMovedSinceInstall(stamp: stamp, lockSHA256: lockSHA)
        else { return nil }
        return .syncEnvironment(
            lock: lockPath, editableTarget: observation.serverDirectoryForDisplay,
            installedSHA256: stamp.lockSHA256, lockSHA256: lockSHA)
    }

    /// Why an offered action that replaces what a running engine has loaded
    /// (Refresh Engine's files, Reinstall Packages' packages) must wait, or
    /// nil when nothing is answering. Replacing either under a running engine
    /// would mix two versions in one process.
    public static func refusalWhileEngineAnswers(
        _ observation: LocalEngineObservation, actionName: String, replacing: String
    ) -> String? {
        guard observation.serverAnswering else { return nil }
        return "stop the local engine first (Stop in the Compute menu), then choose "
            + "\(actionName) again — \(replacing) cannot be replaced while it is running"
    }

    /// Whether the packages installed for the materialized engine on this Mac
    /// were made from a different lock than the one THIS app build ships —
    /// read from the venv's stamp and the bundled lock, without starting or
    /// probing anything, so the first launch after an update can say that
    /// Reinstall Packages will follow Refresh Engine. False for a checkout,
    /// for no engine or no venv yet, for an unstamped venv, and for a build
    /// with no revision of its own (it offers no Refresh Engine, so the
    /// shipped lock would never reach the tree for the setup to offer the
    /// reinstall against).
    public static func materializedEnvironmentLockMoved(
        appRevision: String? = CodeResources.bundleSourceRevision,
        payloadLockSHA256: @autoclosure () -> String? = LocalEngineLayout.payloadLockSHA256()
    ) -> Bool {
        guard appRevision != nil,
            case .engineRoot(let engine)? = CodeResources.localEngineSource()
        else { return false }
        let source = CodeResources.LocalEngineSource.engineRoot(engine)
        guard FileManager.default.isExecutableFile(atPath: source.venvPython.path),
            let stamp = LocalEngineLockStamp.read(
                venv: source.serverDirectory.appending(component: ".venv.nosync"))
        else { return false }
        return lockMovedSinceInstall(stamp: stamp, lockSHA256: payloadLockSHA256())
    }

    public static func interpreterState(_ observation: LocalEngineObservation)
        -> LocalEngineStepState
    {
        var actions: [LocalEngineAction] = []
        let uv = chosenUV(observation)
        if uv == nil {
            actions.append(
                .downloadUV(destination: LocalEngineLayout.managedUV.path))
        }
        // A CPython check is only meaningful against a uv that exists. With no
        // uv yet, the install is always part of the plan — it cannot have run.
        if uv == nil || !observation.managedPythonInstalled {
            actions.append(.installCPython(minor: PinnedCPython.minor))
        }
        if actions.isEmpty, let uv, let version = uv.version {
            let origin = uv.path == LocalEngineLayout.managedUV.path
                ? "managed" : "system"
            return .satisfied(
                "\(origin) uv \(version) at \(uv.path), with a managed CPython "
                    + "\(PinnedCPython.minor) already installed")
        }
        return .act(actions)
    }

    public static func environmentState(_ observation: LocalEngineObservation)
        -> LocalEngineStepState
    {
        guard let lockPath = observation.lockFilePath,
            let lockSHA = observation.lockFileSHA256
        else {
            // No lock in the tree: the flow refuses rather than resolving from
            // pyproject FLOORS behind the researcher's back. An unpinned local
            // environment is exactly the drift the committed locks exist to
            // stop, and this path is not urgent enough to earn an exception.
            return .blocked(
                "this engine tree carries no \(LocalEngineLayout.lockFileName) "
                    + "— the pinned dependency set is what makes a local "
                    + "environment comparable with the cluster's, so the flow "
                    + "will not resolve from pyproject floors instead")
        }
        guard observation.venvPythonExists else {
            return .act([
                .createVenv(path: observation.venvPathForDisplay),
                .installLock(
                    lock: lockPath, editableTarget: observation.serverDirectoryForDisplay),
            ])
        }
        guard let stamp = observation.venvLockStamp else {
            // THE DEVELOPER CASE, and the reason this is an advisory and not
            // an action: every venv that exists today was made by
            // `start-local-server.sh` with pip and floors, and has no stamp.
            // Reinstalling it unasked would replace a working environment with
            // a 2 GB download the researcher never requested.
            return .satisfiedWithAdvisory(
                "\(observation.venvPathForDisplay) already exists",
                advisory: "it carries no lock stamp, so whether it matches "
                    + "\(LocalEngineLayout.lockFileName) is unknown — it was "
                    + "most likely created by scripts/start-local-server.sh, "
                    + "which installs from pyproject floors. Reinstall from "
                    + "the lock when you want the pinned set")
        }
        if stamp.incomplete == true {
            return .blocked(
                "a Reinstall Packages run on \(observation.venvPathForDisplay) did "
                    + "not finish, so it may be missing packages, the engine's own "
                    + "included — choose Reinstall Packages to finish it before "
                    + "starting the engine")
        }
        if stamp.lockSHA256 == lockSHA {
            return .satisfied(
                "\(observation.venvPathForDisplay) matches \(stamp.lockFile) "
                    + "(sha256 \(String(stamp.lockSHA256.prefix(8))))")
        }
        // Still an advisory, never an action: the researcher decides when
        // gigabytes move, through the offered Reinstall Packages — offered
        // only for the app's own engine (see environmentSyncOffer), so a
        // checkout's advisory names the developer's own repair instead.
        let moved = "it was installed from \(stamp.lockFile) sha256 "
            + "\(String(stamp.lockSHA256.prefix(8))) but the tree now "
            + "carries sha256 \(String(lockSHA.prefix(8))) — the lock "
            + "moved under it. "
        let repair: String
        if case .engineRoot = observation.source {
            repair = "Choose Reinstall Packages to match it (downloads up to about "
                + "\(describeBytes(LocalEngineAction.lockInstallEstimatedBytes)); "
                + "nothing is fetched until you confirm)"
        } else {
            repair = "This environment belongs to the code checkout, so the app "
                + "leaves it as it is; reinstall it from the lock yourself when "
                + "you want the pinned set (Server/README.md, \"Dependency locks\")"
        }
        return .satisfiedWithAdvisory(
            "\(observation.venvPathForDisplay) already exists", advisory: moved + repair)
    }

    public static func serveState(_ observation: LocalEngineObservation)
        -> LocalEngineStepState
    {
        if observation.serverAnswering {
            return .satisfied(
                "a SteerLab server is already answering on "
                    + "127.0.0.1:\(observation.port)")
        }
        return .act([.startServer(port: observation.port)])
    }

    public static func acceptanceState(_: LocalEngineObservation)
        -> LocalEngineStepState
    {
        // Acceptance is never "already satisfied": it is the evidence, and
        // evidence from a previous run is not evidence about this one.
        .act([.runQualification])
    }

    // MARK: The whole plan

    /// Every step's decision, in order. Steps after a `blocked` one are still
    /// computed (a plan that stops at the first problem hides the rest of the
    /// work), but the driver refuses to execute past the block.
    public static func plan(_ observation: LocalEngineObservation)
        -> [LocalEnginePlanEntry]
    {
        [
            LocalEnginePlanEntry(step: .engineSource, state: engineSourceState(observation)),
            LocalEnginePlanEntry(step: .interpreter, state: interpreterState(observation)),
            LocalEnginePlanEntry(step: .environment, state: environmentState(observation)),
            LocalEnginePlanEntry(step: .serve, state: serveState(observation)),
            LocalEnginePlanEntry(step: .acceptance, state: acceptanceState(observation)),
        ]
    }

    /// True when the flow has nothing to do but re-verify — the idempotent
    /// re-run, which should be quick and quiet.
    public static func isAlreadyProvisioned(_ plan: [LocalEnginePlanEntry]) -> Bool {
        plan.allSatisfy { entry in
            switch entry.step {
            case .acceptance: return true  // always re-run; never counted
            default: return entry.state.actions.isEmpty && !entry.state.isBlocked
            }
        }
    }

    // MARK: The honesty preamble

    /// Human byte size, deliberately coarse: "about 2 GB" is the number a
    /// researcher decides on, and a precise 1.93 GiB would be false precision
    /// for an estimate.
    public static func describeBytes(_ bytes: Int) -> String {
        if bytes >= 1_000_000_000 {
            return String(format: "%.1f GB", Double(bytes) / 1_000_000_000)
        }
        if bytes >= 1_000_000 {
            return "\(Int((Double(bytes) / 1_000_000).rounded())) MB"
        }
        return "\(bytes) bytes"
    }

    /// What this plan will DOWNLOAD, stated before anything is fetched. Empty
    /// when the plan downloads nothing — and callers must render that as "no
    /// downloads", not as silence.
    public static func downloadPreamble(_ plan: [LocalEnginePlanEntry]) -> [String] {
        downloadPreamble(for: plan.flatMap(\.state.actions))
    }

    /// The same statement for actions outside the plan — an offered Reinstall
    /// Packages states its download through here before it runs.
    public static func downloadPreamble(for actions: [LocalEngineAction]) -> [String] {
        var lines: [String] = []
        var total = 0
        var anyCeiling = false
        for action in actions where action.estimatedDownloadBytes > 0 {
            let size = describeBytes(action.estimatedDownloadBytes)
            let qualifier = action.downloadEstimateIsUpperBound
                ? "up to about " : action.downloadEstimateIsApproximate ? "about " : ""
            lines.append("\(action.title) — \(qualifier)\(size)")
            total += action.estimatedDownloadBytes
            anyCeiling = anyCeiling || action.downloadEstimateIsUpperBound
        }
        guard !lines.isEmpty else { return [] }
        lines.append(
            "Total: \(anyCeiling ? "up to " : "")about \(describeBytes(total)) "
                + "over the network.")
        return lines
    }
}

/// One row of the plan: a step and what it decided. A named type rather than a
/// tuple so the sheet can `ForEach` it and tests can compare whole plans.
public struct LocalEnginePlanEntry: Sendable, Equatable, Identifiable {
    public let step: LocalEngineStep
    public let state: LocalEngineStepState
    public var id: String { step.rawValue }

    public init(step: LocalEngineStep, state: LocalEngineStepState) {
        self.step = step
        self.state = state
    }
}

extension LocalEngineObservation {
    /// `<serverDirectory>` as a display string, derived from whichever source
    /// tier answered.
    public var serverDirectoryForDisplay: String {
        switch source {
        case .checkout(let path): return path + "/Server"
        case .engineRoot(let path, _): return path + "/Server"
        case .materializable(let destination, _): return destination + "/Server"
        case .unavailable: return "(no engine tree)"
        }
    }

    public var venvPathForDisplay: String {
        serverDirectoryForDisplay + "/.venv.nosync"
    }
}

// MARK: - The observation seam

/// Everything the observation is built from. One protocol so the decision
/// layer can be exercised against fakes covering the states this machine will
/// otherwise only ever meet on somebody else's Mac: no uv, a stale uv, a
/// missing venv, a drifted venv, a bundle with no checkout.
public protocol LocalEngineEnvironment: Sendable {
    /// The resolved engine source, or nil when there is none yet.
    func engineSource() -> CodeResources.LocalEngineSource?
    /// Where a materialization would copy FROM and TO, when this build ships a
    /// payload; nil when it does not (then there is nothing to materialize and
    /// the step is blocked with a layout reason).
    func materializationPlan() -> (destination: URL, payload: URL)?
    /// Why there is no engine source, in layout terms.
    func unavailabilityReason() -> String
    /// The running app's build revision.
    var appRevision: String? { get }
    func locateSystemUV() async -> LocalEngineTool?
    func locateManagedUV() async -> LocalEngineTool?
    func managedPythonIsInstalled(uv: LocalEngineTool) async -> Bool
    func isExecutableFile(_ url: URL) -> Bool
    func sha256OfFile(_ url: URL) -> String?
    func readLockStamp(venv: URL) -> LocalEngineLockStamp?
    func serverAnswers(port: Int) -> Bool
}

extension LocalEngineEnvironment {

    /// Assemble the observation. Ordered so nothing expensive runs that a
    /// cheap answer made irrelevant.
    public func observe(port: Int) async -> LocalEngineObservation {
        let source = engineSource()
        let sourceState: LocalEngineObservation.SourceState
        switch source {
        case .checkout(let checkout):
            sourceState = .checkout(path: checkout.root.path)
        case .engineRoot(let engine):
            sourceState = .engineRoot(
                path: engine.root.path, stampRevision: engine.stamp?.sourceRevision)
        case nil:
            if let plan = materializationPlan() {
                sourceState = .materializable(
                    destination: plan.destination.path, payload: plan.payload.path)
            } else {
                sourceState = .unavailable(reason: unavailabilityReason())
            }
        }

        let system = await locateSystemUV()
        let managed = await locateManagedUV()
        var pythonInstalled = false
        if let uv = LocalEngineDecisions.chosenUV(
            LocalEngineObservation(
                source: sourceState, systemUV: system, managedUV: managed))
        {
            pythonInstalled = await managedPythonIsInstalled(uv: uv)
        }

        var venvExists = false
        var stamp: LocalEngineLockStamp?
        var lockSHA: String?
        var lockPath: String?
        if let source {
            let venv = source.serverDirectory.appending(component: ".venv.nosync")
            venvExists = isExecutableFile(source.venvPython)
            if venvExists { stamp = readLockStamp(venv: venv) }
            let lock = source.serverDirectory
                .appending(component: LocalEngineLayout.lockFileName)
            if let sha = sha256OfFile(lock) {
                lockSHA = sha
                lockPath = lock.path
            }
        }

        return LocalEngineObservation(
            source: sourceState, appRevision: appRevision, systemUV: system,
            managedUV: managed, managedPythonInstalled: pythonInstalled,
            venvPythonExists: venvExists, venvLockStamp: stamp,
            lockFileSHA256: lockSHA, lockFilePath: lockPath,
            serverAnswering: serverAnswers(port: port), port: port)
    }
}

// MARK: - The live environment

/// The real one: `CodeResources` for tiers, `/usr/bin/env` for PATH lookups,
/// `LocalPythonRuntime` for the endpoint probe.
public struct SystemLocalEngineEnvironment: LocalEngineEnvironment {
    private let shell: any ClusterShellRunner

    public init(shell: any ClusterShellRunner = ProvisionShellRunner()) {
        self.shell = shell
    }

    public func engineSource() -> CodeResources.LocalEngineSource? {
        CodeResources.localEngineSource()
    }

    public func materializationPlan() -> (destination: URL, payload: URL)? {
        guard let payload = try? CodeResources.serverPayload() else { return nil }
        return (destination: CodeResources.defaultEngineRoot, payload: payload)
    }

    public func unavailabilityReason() -> String {
        "this build ships no Python engine payload and no code checkout was "
            + "found beside it — reinstall the app, or clone the code "
            + "repository into \(HomeLayout.defaultHome.path)"
    }

    public var appRevision: String? { CodeResources.bundleSourceRevision }

    public func locateSystemUV() async -> LocalEngineTool? {
        let found = await shell.run(["/usr/bin/env", "which", "uv"])
        guard found.succeeded else { return nil }
        let path = found.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty, path.hasPrefix("/") else { return nil }
        return LocalEngineTool(path: path, version: await uvVersion(at: path))
    }

    public func locateManagedUV() async -> LocalEngineTool? {
        let path = LocalEngineLayout.managedUV.path
        guard FileManager.default.isExecutableFile(atPath: path) else { return nil }
        return LocalEngineTool(path: path, version: await uvVersion(at: path))
    }

    /// `uv --version` → "uv 0.12.5 (abcdef 2026-08-14)" → "0.12.5".
    private func uvVersion(at path: String) async -> String? {
        let result = await shell.run([path, "--version"])
        guard result.succeeded else { return nil }
        let fields = result.text.split(whereSeparator: \.isWhitespace)
        guard fields.count >= 2, fields[0] == "uv" else { return nil }
        return String(fields[1])
    }

    public func managedPythonIsInstalled(uv: LocalEngineTool) async -> Bool {
        // `--managed-python` is LOAD-BEARING, not tidiness. Measured
        // 2026-08-20: a bare `uv python find 3.12` on this Mac succeeded
        // BEFORE any managed CPython existed, because it happily matches a
        // system interpreter on PATH. Without the flag the flow would report
        // "already installed", skip the download, and then build the venv from
        // whatever system Python it found — on a clean Mac, the
        // `/usr/bin/python3` stub whose first use pops the Command Line Tools
        // installer, which is the exact trap uv is here to avoid.
        // `uv python find` downloads nothing either way.
        let result = await shell.run(
            [uv.path, "python", "find", "--managed-python", PinnedCPython.minor])
        return result.succeeded
    }

    public func isExecutableFile(_ url: URL) -> Bool {
        FileManager.default.isExecutableFile(atPath: url.path)
    }

    public func sha256OfFile(_ url: URL) -> String? {
        LocalEngineLayout.sha256Hex(ofFileAt: url)
    }

    public func readLockStamp(venv: URL) -> LocalEngineLockStamp? {
        LocalEngineLockStamp.read(venv: venv)
    }

    public func serverAnswers(port: Int) -> Bool {
        LocalServerPidfile.endpointIsSteerLab(port: port)
    }
}

// MARK: - Typed failures

public enum LocalEngineError: Error, Sendable, Equatable, CustomStringConvertible {
    case blocked(step: LocalEngineStep, reason: String)
    case cancelled(afterStep: LocalEngineStep?)
    case downloadFailed(url: String, reason: String)
    /// The refusal that has to be loud: the bytes are not the pinned bytes.
    case hashMismatch(url: String, expected: String, observed: String)
    case commandFailed(
        step: LocalEngineStep, command: String, exitCode: Int32, lastLines: [String])
    case fileOperationFailed(step: LocalEngineStep, reason: String)

    public var description: String {
        switch self {
        case .blocked(let step, let reason):
            return "\(step.title): \(reason)"
        case .cancelled(let step):
            return step.map { "cancelled after \($0.title)" } ?? "cancelled"
        case .downloadFailed(let url, let reason):
            return "could not download \(url): \(reason)"
        case .hashMismatch(let url, let expected, let observed):
            return "REFUSED: \(url) did not match its pinned sha256 "
                + "(expected \(expected), got \(observed)). Nothing was "
                + "installed. This is either a corrupted download or a "
                + "substituted artifact — retry once; if it repeats, do not "
                + "work around it."
        case .commandFailed(let step, let command, let code, let lines):
            var text = "\(step.title): `\(command)` exited \(code)"
            if !lines.isEmpty {
                text += " — " + lines.suffix(3).joined(separator: " / ")
            }
            return text
        case .fileOperationFailed(let step, let reason):
            return "\(step.title): \(reason)"
        }
    }
}

// MARK: - The provisioner

/// Drives the five steps, streaming into the Activity feed, cancellable
/// between steps, and safe to re-run.
///
/// It deliberately does NOT own the running server's lifecycle: once step 4
/// has a server up, `LocalServerController` is the thing that stops it, adopts
/// it across relaunches, and reports it in the menu. This type provisions and
/// hands over.
@MainActor
@Observable
public final class LocalEngineProvisioner {

    public enum Phase: Sendable, Equatable {
        case unknown
        case planning
        case ready(String)
        case needsSetup(actionCount: Int)
        case running(LocalEngineStep)
        case failed(String)
        case cancelled

        public var isRunning: Bool {
            if case .running = self { return true }
            if case .planning = self { return true }
            return false
        }
    }

    public private(set) var phase: Phase = .unknown
    /// The last computed plan, in step order — what the sheet renders.
    public private(set) var plan: [LocalEnginePlanEntry] = []
    /// Per-step outcome text while the flow runs ("done", "skipped", a failure).
    public private(set) var progress: [LocalEngineStep: String] = [:]
    /// What the plan intends to download, stated before anything is fetched.
    public private(set) var downloadPreamble: [String] = []
    /// `site qualify`'s rows, once acceptance has run.
    public private(set) var qualification: SiteQualificationReport?
    /// The engine's own version string from `/api/info`, once it answered.
    public private(set) var engineVersion: String?
    /// Stated, labelled, non-blocking.
    public private(set) var skewAdvisory: String?
    /// Refresh Engine, when the last look found something to refresh (see
    /// `LocalEngineDecisions.refreshOffer`). The setup sheet offers it.
    public private(set) var refreshOffer: LocalEngineAction?
    /// Reinstall Packages, when the last look found the installed packages
    /// stamped with a different lock (see
    /// `LocalEngineDecisions.environmentSyncOffer`). The setup sheet offers it.
    public private(set) var environmentSyncOffer: LocalEngineAction?
    /// What Reinstall Packages would download, stated before it runs. Empty
    /// when nothing is offered.
    public var environmentSyncPreamble: [String] {
        environmentSyncOffer.map { LocalEngineDecisions.downloadPreamble(for: [$0]) } ?? []
    }
    /// "Refresh Engine" or "Reinstall Packages" while one is in flight, set
    /// BEFORE its first await so a second click, Set Up, or an engine start
    /// cannot slip in while it is still looking. Neither can be stopped
    /// partway (a reinstall stopped between its two commands would leave the
    /// engine package uninstalled), so `cancel()` says so instead.
    public private(set) var offeredActionInFlight: String?
    /// True while setup or an offered action is running — what every control
    /// that would start, stop, or replace the engine is gated on.
    public var isBusy: Bool { phase.isRunning || offeredActionInFlight != nil }
    public private(set) var statusLine = "not checked yet"
    public var port = 8080

    private let environment: any LocalEngineEnvironment
    private let shell: any ClusterShellRunner
    private let downloader: any LocalEngineDownloader
    private var cancelRequested = false
    private var work: Task<Void, Never>?
    /// The Activity-feed sink for the run in flight. Held on the object rather
    /// than threaded through every method: the step functions are already long
    /// enough without an `emit:` parameter on each one, and this way the
    /// streaming and the state live in the same place.
    private var logLines: [String] = []
    private var logID: UUID?
    private weak var logHost: ChatService?
    private let logTitle = "Local engine setup"

    public init(
        environment: (any LocalEngineEnvironment)? = nil,
        shell: (any ClusterShellRunner)? = nil,
        downloader: any LocalEngineDownloader = URLSessionEngineDownloader()
    ) {
        let resolvedShell = shell ?? ProvisionShellRunner()
        self.shell = resolvedShell
        self.environment = environment
            ?? SystemLocalEngineEnvironment(shell: resolvedShell)
        self.downloader = downloader
    }

    // MARK: Planning

    /// Observe and decide, touching nothing. Safe to call on every sheet open.
    public func refreshPlan() async {
        phase = .planning
        statusLine = "checking what is already set up…"
        let observation = await environment.observe(port: port)
        let plan = LocalEngineDecisions.plan(observation)
        self.plan = plan
        refreshOffer = LocalEngineDecisions.refreshOffer(observation)
        environmentSyncOffer = LocalEngineDecisions.environmentSyncOffer(observation)
        downloadPreamble = LocalEngineDecisions.downloadPreamble(plan)
        if let blocked = plan.first(where: { $0.state.isBlocked }) {
            phase = .failed(blocked.state.summary)
            statusLine = blocked.state.summary
            return
        }
        let actionCount = plan.reduce(0) { $0 + $1.state.actions.count }
        if LocalEngineDecisions.isAlreadyProvisioned(plan) {
            phase = .ready("everything is already set up")
            // Set up, but "everything is set up" alone would hide the one
            // thing left to do: the packages predate the lock in the tree.
            // "Ready" means the engine is answering, and a reinstall waits
            // until it is stopped, so the hint says both.
            statusLine = environmentSyncOffer == nil
                ? "ready — everything is set up; re-running only "
                    + "re-verifies (\(plan.count) steps)"
                : "set up, but the installed packages were made from an earlier "
                    + "lock — stop the engine, then choose Reinstall Packages to "
                    + "match this engine"
        } else {
            phase = .needsSetup(actionCount: actionCount)
            statusLine = "\(actionCount) step\(actionCount == 1 ? "" : "s") to "
                + "run" + (downloadPreamble.isEmpty
                    ? " (nothing to download)"
                    : " — see what will be downloaded before starting")
            // The usual shape after Refresh Engine: the engine is stopped, so
            // serving is a step to run, and the packages still need matching.
            // Worded to hold for an unfinished reinstall too, whose stamp may
            // already name this lock.
            if environmentSyncOffer != nil {
                statusLine += "; the installed packages do not match this engine's "
                    + "lock — choose Reinstall Packages to match it"
            }
        }
    }

    // MARK: Running

    public func cancel() {
        if let action = offeredActionInFlight {
            statusLine = "\(action) cannot be stopped partway — it finishes on its "
                + "own, and nothing else starts until it does"
            return
        }
        cancelRequested = true
        statusLine = "cancelling after the current step…"
    }

    /// Append one line to the Activity feed (and remember it).
    private func emit(_ line: String) {
        logLines.append(line)
        if logLines.count > 400 { logLines.removeFirst(logLines.count - 400) }
        if let logID {
            logHost?.updateLiveLog(id: logID, title: logTitle, lines: logLines)
        }
    }

    /// Run the plan, detached, streaming into `host`'s Activity pane. Every
    /// step re-checks its own precondition first, so this is the same call
    /// whether it is the first attempt or the fifth.
    public func run(host: ChatService) {
        guard work == nil, offeredActionInFlight == nil else { return }
        work = Task { [weak self] in
            await self?.runAwaitingCompletion(host: host)
            self?.work = nil
        }
    }

    /// The same sequence, awaited rather than detached, with the live-log host
    /// optional. This is what `run(host:)` calls, and what a test drives — so
    /// the tested path and the button's path are one body, not two.
    public func runAwaitingCompletion(host: ChatService? = nil) async {
        // The engine's last steps serve and check the current workspace.
        // With none chosen, stop before anything is installed or started.
        guard WorkspaceRoot.hasWorkspace else {
            let reason = "create or open a workspace first — the engine on "
                + "this Mac works inside one"
            phase = .failed(reason)
            statusLine = reason
            return
        }
        cancelRequested = false
        let first = "checking what is already set up…"
        logHost = host
        logLines = [first]
        logID = host?.startLiveLog(title: logTitle, initialLine: first)
        do {
            try await execute()
            phase = .ready(statusLine)
        } catch let error as LocalEngineError {
            if case .cancelled = error {
                phase = .cancelled
                statusLine = "cancelled — nothing further was changed; "
                    + "re-running continues from here"
            } else {
                phase = .failed(error.description)
                statusLine = error.description
            }
            emit(statusLine)
        } catch {
            phase = .failed(error.localizedDescription)
            statusLine = error.localizedDescription
            emit(statusLine)
        }
    }

    /// Refresh Engine: copy THIS app build's engine over the materialized one,
    /// keeping its installed packages (`Server/.venv.nosync`), so an engine
    /// left behind by an update catches up without a 2 GB reinstall. Refused,
    /// with the reason, while the engine is answering: replacing files under
    /// a running engine would mix two versions in one process.
    ///
    /// Returns true when the copy was made. The fresh plan it ends with
    /// carries `environmentSyncOffer` when the copied lock differs from the
    /// one the packages were installed from, so the caller can offer
    /// Reinstall Packages at once instead of leaving only an advisory.
    @discardableResult
    public func refreshEngine(host: ChatService? = nil) async -> Bool {
        guard work == nil, !isBusy else { return false }
        offeredActionInFlight = "Refresh Engine"
        defer { offeredActionInFlight = nil }
        let observation = await environment.observe(port: port)
        guard let offer = LocalEngineDecisions.refreshOffer(observation),
            case .refreshEngine(let destination, _, _) = offer
        else {
            refreshOffer = nil
            statusLine = "the engine already matches this app build — nothing to refresh"
            return false
        }
        if let refusal = LocalEngineDecisions.refusalWhileEngineAnswers(
            observation, actionName: "Refresh Engine", replacing: "its files")
        {
            statusLine = refusal
            return false
        }
        logHost = host
        logLines = [offer.title]
        logID = host?.startLiveLog(title: logTitle, initialLine: offer.title)
        phase = .running(.engineSource)
        do {
            try await materialize(
                to: URL(filePath: destination), step: .engineSource,
                preservingEnvironment: true)
            emit("Refresh Engine: done — the installed packages were kept")
        } catch let error as LocalEngineError {
            phase = .failed(error.description)
            statusLine = error.description
            emit(statusLine)
            return false
        } catch {
            phase = .failed(error.localizedDescription)
            statusLine = error.localizedDescription
            emit(statusLine)
            return false
        }
        await refreshPlan()
        if let sync = environmentSyncOffer {
            emit(
                "Refresh Engine: this version pins different packages — "
                    + "Reinstall Packages is offered: " + sync.title)
        }
        return true
    }

    /// Reinstall Packages: bring the installed packages in line with the lock
    /// the engine tree now carries, INTO the existing venv, then re-stamp it.
    /// `uv pip sync` installs what changed, removes what the lock no longer
    /// lists, and leaves the rest; the engine package is then reinstalled
    /// editable, exactly as the first install does. Offered only (never
    /// planned), stated before it runs (the download preamble goes to the
    /// Activity feed first), and refused while the engine is answering, for
    /// the same reason as Refresh Engine. A failure leaves the OLD stamp in
    /// place, so the offer stands until a reinstall completes.
    public func syncEnvironment(host: ChatService? = nil) async {
        guard work == nil, !isBusy else { return }
        offeredActionInFlight = "Reinstall Packages"
        defer { offeredActionInFlight = nil }
        let observation = await environment.observe(port: port)
        guard let offer = LocalEngineDecisions.environmentSyncOffer(observation) else {
            environmentSyncOffer = nil
            statusLine = "nothing to reinstall — the installed packages are not "
                + "recorded against a different lock"
            return
        }
        if let refusal = LocalEngineDecisions.refusalWhileEngineAnswers(
            observation, actionName: "Reinstall Packages", replacing: "its packages")
        {
            statusLine = refusal
            return
        }
        logHost = host
        logLines = [offer.title]
        logID = host?.startLiveLog(title: logTitle, initialLine: offer.title)
        for line in LocalEngineDecisions.downloadPreamble(for: [offer]) {
            emit("will download: " + line)
        }
        phase = .running(.environment)
        // A stop partway leaves the stamp marked incomplete (see
        // installDependencies), so the engine will not start on it until a
        // reinstall finishes; the failure says how to finish. The mark is read
        // back from disk: a failure before it was written changed nothing.
        func unfinished() -> String {
            guard case .syncEnvironment(_, let target, _, _) = offer,
                LocalEngineLockStamp.read(
                    venv: URL(filePath: target).appending(component: ".venv.nosync"))?
                    .incomplete == true
            else { return "" }
            return " — the reinstall did not finish; choose Reinstall Packages again "
                + "to complete it before starting the engine"
        }
        do {
            try await perform(offer, step: .environment, observation: observation)
            emit("Reinstall Packages: done — the packages match the engine's lock")
        } catch let error as LocalEngineError {
            phase = .failed(error.description)
            statusLine = error.description + unfinished()
            emit(statusLine)
            return
        } catch {
            phase = .failed(error.localizedDescription)
            statusLine = error.localizedDescription + unfinished()
            emit(statusLine)
            return
        }
        await refreshPlan()
    }

    private func execute() async throws {
        var observation = await environment.observe(port: port)
        var plan = LocalEngineDecisions.plan(observation)
        self.plan = plan
        downloadPreamble = LocalEngineDecisions.downloadPreamble(plan)
        for line in downloadPreamble { emit("will download: " + line) }

        var lastCompleted: LocalEngineStep?
        for step in LocalEngineStep.allCases {
            if cancelRequested { throw LocalEngineError.cancelled(afterStep: lastCompleted) }
            // Re-decide from a FRESH observation at every step: the previous
            // step just changed the world, and a plan computed before it ran
            // would be describing a machine that no longer exists.
            observation = await environment.observe(port: port)
            plan = LocalEngineDecisions.plan(observation)
            self.plan = plan
            let state = plan.first { $0.step == step }?.state
                ?? .blocked("no decision for \(step.rawValue)")
            phase = .running(step)
            switch state {
            case .blocked(let reason):
                progress[step] = "blocked"
                throw LocalEngineError.blocked(step: step, reason: reason)
            case .satisfied(let reason):
                progress[step] = "already done"
                emit("\(step.title): skipped — \(reason)")
            case .satisfiedWithAdvisory(let reason, let advisory):
                progress[step] = "already done (advisory)"
                emit("\(step.title): skipped — \(reason)")
                emit("\(step.title): advisory — \(advisory)")
            case .act(let actions):
                for action in actions {
                    if cancelRequested {
                        throw LocalEngineError.cancelled(afterStep: lastCompleted)
                    }
                    emit("\(step.title): \(action.title)")
                    try await perform(action, step: step, observation: observation)
                }
                progress[step] = "done"
            }
            lastCompleted = step
        }
        statusLine = qualification.map { report in
            "ready — \(report.summaryLine)"
        } ?? "ready"
        emit(statusLine)
    }

    // MARK: Actions

    private func perform(
        _ action: LocalEngineAction, step: LocalEngineStep,
        observation: LocalEngineObservation
    ) async throws {
        switch action {
        case .materializeEngine(let destination), .refreshEngine(let destination, _, _):
            try await materialize(to: URL(filePath: destination), step: step)
        case .downloadUV:
            try await installPinnedUV(step: step)
        case .installCPython(let minor):
            // The uv may have been installed by the action just before this
            // one, so the observation predates it — re-locate rather than
            // trusting a snapshot the previous action invalidated.
            var located = LocalEngineDecisions.chosenUV(observation)
            if located == nil { located = await environment.locateManagedUV() }
            guard let uv = located else {
                throw LocalEngineError.blocked(
                    step: step,
                    reason: "no usable uv after the download step — nothing to "
                        + "install a CPython with")
            }
            try await runOrThrow([uv.path, "python", "install", minor], step: step)
        case .createVenv(let path):
            guard let uv = LocalEngineDecisions.chosenUV(observation) else {
                throw LocalEngineError.blocked(
                    step: step, reason: "no usable uv to create the venv with")
            }
            // `--managed-python` for the same reason as the probe above: the
            // venv must be built from the interpreter this flow installed,
            // never from a system Python that merely happens to satisfy 3.12.
            try await runOrThrow(
                [uv.path, "venv", "--managed-python", "--python",
                 PinnedCPython.minor, path],
                step: step)
        case .installLock(let lock, let editableTarget):
            try await installDependencies(
                lock: lock, serverDirectory: editableTarget,
                observation: observation, step: step)
        case .syncEnvironment(let lock, let editableTarget, _, _):
            try await installDependencies(
                lock: lock, serverDirectory: editableTarget,
                observation: observation, step: step, syncing: true)
        case .startServer:
            try await startServer(observation: observation, step: step)
        case .runQualification:
            try await runAcceptance(step: step)
        }
    }

    /// Step 1 — COPY out of the bundle. `cp -R` of the payload's contents into
    /// `<destination>/Server`, so the materialized tree is laid out exactly
    /// like a checkout's `Server/` and one path expression covers both tiers.
    private func materialize(
        to destination: URL, step: LocalEngineStep, preservingEnvironment: Bool = false
    ) async throws {
        guard let payload = try? CodeResources.serverPayload() else {
            throw LocalEngineError.blocked(
                step: step, reason: environment.unavailabilityReason())
        }
        let fm = FileManager.default
        let serverDirectory = destination.appending(component: "Server")
        // The payload root is "the Server tree" in BOTH layouts a build can
        // present (`<checkout>/Server` in dev, `Resources/ServerPayload` in a
        // bundle), which is why the package probe accepts either spelling —
        // the same two-candidate rule `ClusterProvisioningOperations
        // .packageSubpath` applies to a cluster payload.
        let packageRoot: URL
        if fm.fileExists(atPath: payload.appending(path: "steerlab_server").path) {
            packageRoot = payload
        } else if fm.fileExists(
            atPath: payload.appending(path: "Server/steerlab_server").path)
        {
            packageRoot = payload.appending(component: "Server")
        } else {
            throw LocalEngineError.fileOperationFailed(
                step: step,
                reason: "the bundled payload at \(payload.path) holds no "
                    + "steerlab_server package — this build was assembled "
                    + "without its Python engine; reinstall the app")
        }
        let count: Int
        do {
            if preservingEnvironment, fm.fileExists(atPath: serverDirectory.path) {
                count = try LocalEngineTreeRefresh.replaceSources(
                    in: serverDirectory, from: packageRoot)
            } else {
                // Replace wholesale rather than merging: a half-updated engine
                // tree is the failure mode nobody can diagnose later.
                if fm.fileExists(atPath: serverDirectory.path) {
                    try fm.removeItem(at: serverDirectory)
                }
                try fm.createDirectory(at: destination, withIntermediateDirectories: true)
                try fm.copyItem(at: packageRoot, to: serverDirectory)
                count = fileCount(under: serverDirectory)
            }
        } catch {
            throw LocalEngineError.fileOperationFailed(
                step: step,
                reason: "could not copy the engine to \(serverDirectory.path): "
                    + error.localizedDescription)
        }
        let stamp = CodeResources.EngineStamp(
            sourceRevision: environment.appRevision,
            appVersion: CodeResources.bundleFullVersion ?? SteerLabVersion.version,
            materializedAt: ISO8601DateFormatter().string(from: Date()),
            fileCount: count)
        if let data = try? JSONEncoder().encode(stamp) {
            try? data.write(
                to: destination.appending(component: CodeResources.EngineStamp.fileName))
        }
        emit(
            "\(step.title): copied \(count) files to \(serverDirectory.path), "
                + "stamped \(environment.appRevision ?? "unstamped build")")
    }

    private func fileCount(under root: URL) -> Int {
        guard
            let enumerator = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.isRegularFileKey])
        else { return 0 }
        var count = 0
        for case let url as URL in enumerator {
            if (try? url.resourceValues(forKeys: [.isRegularFileKey]))?
                .isRegularFile == true
            {
                count += 1
            }
        }
        return count
    }

    /// Step 2 — download the PINNED uv, verify its sha256, and only then put it
    /// anywhere. No `curl | sh`: the bytes are hashed in this process before a
    /// single one is executed, and a mismatch is a refusal, not a warning.
    private func installPinnedUV(step: LocalEngineStep) async throws {
        let url = PinnedUV.downloadURL
        emit(
            "\(step.title): fetching \(url.absoluteString) "
                + "(\(LocalEngineDecisions.describeBytes(PinnedUV.byteCount)))")
        let data: Data
        do {
            data = try await downloader.download(from: url)
        } catch {
            throw LocalEngineError.downloadFailed(
                url: url.absoluteString, reason: error.localizedDescription)
        }
        let digest = SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }.joined()
        guard digest == PinnedUV.sha256 else {
            throw LocalEngineError.hashMismatch(
                url: url.absoluteString, expected: PinnedUV.sha256,
                observed: digest)
        }
        emit("\(step.title): sha256 verified (\(String(digest.prefix(12)))…)")

        let fm = FileManager.default
        let staging = fm.temporaryDirectory
            .appending(component: "steerlab-uv-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: staging) }
        do {
            try fm.createDirectory(at: staging, withIntermediateDirectories: true)
            try data.write(to: staging.appending(component: PinnedUV.archiveName))
        } catch {
            throw LocalEngineError.fileOperationFailed(
                step: step,
                reason: "could not stage the download: \(error.localizedDescription)")
        }
        try await runOrThrow(
            ["/usr/bin/tar", "-xzf",
             staging.appending(component: PinnedUV.archiveName).path,
             "-C", staging.path],
            step: step)
        let extracted = staging.appending(path: PinnedUV.archiveExecutableSubpath)
        guard fm.fileExists(atPath: extracted.path) else {
            throw LocalEngineError.fileOperationFailed(
                step: step,
                reason: "the verified archive did not contain "
                    + PinnedUV.archiveExecutableSubpath)
        }
        do {
            try fm.createDirectory(
                at: LocalEngineLayout.toolsDirectory,
                withIntermediateDirectories: true)
            let destination = LocalEngineLayout.managedUV
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.copyItem(at: extracted, to: destination)
            try fm.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: destination.path)
            emit("\(step.title): uv \(PinnedUV.version) installed at \(destination.path)")
        } catch {
            throw LocalEngineError.fileOperationFailed(
                step: step,
                reason: "could not install uv: \(error.localizedDescription)")
        }
    }

    /// Step 3 — the committed platform lock, then the engine package itself.
    ///
    /// DIVERGENCE FROM THE CLUSTER, deliberately: `Server/scripts/bootstrap.sh`
    /// filters `torch|triton|nvidia-` OUT of the lock and installs torch from
    /// the site's `--torch-index`, because WHICH CUDA build a node wants is a
    /// site fact. On this Mac there is no such fact — the macOS lock's
    /// `torch==2.13.0` IS the Apple-silicon wheel we want — so the lock is
    /// installed whole. Copying the cluster's carve-out here would leave the
    /// venv with no torch at all.
    ///
    /// `syncing` is Reinstall Packages: `uv pip sync` instead of `install`,
    /// so the existing venv ends up holding exactly the lock — a package the
    /// lock dropped is removed rather than left behind under a stamp that
    /// would then claim a match it does not have. Sync also removes the
    /// editable engine package (the lock does not list it), which the second
    /// command below puts back.
    private func installDependencies(
        lock: String, serverDirectory: String, observation: LocalEngineObservation,
        step: LocalEngineStep, syncing: Bool = false
    ) async throws {
        guard let uv = LocalEngineDecisions.chosenUV(observation) else {
            throw LocalEngineError.blocked(
                step: step, reason: "no usable uv to install dependencies with")
        }
        let venv = URL(filePath: serverDirectory).appending(component: ".venv.nosync")
        let interpreter = venv.appending(components: "bin", "python").path
        // The macOS lock was resolved with MACOSX_DEPLOYMENT_TARGET=14.0
        // (torch publishes its Apple-silicon wheels as macosx_14_0_arm64);
        // installing under a different target is how a "pinned" install
        // quietly resolves a different torch.
        let overrides = ["MACOSX_DEPLOYMENT_TARGET": "14.0"]
        let stampURL = venv.appending(component: LocalEngineLayout.lockStampFileName)
        if syncing, var marked = observation.venvLockStamp {
            // From here until the new stamp is written, the environment may
            // lack packages (sync removes the editable engine first). The mark
            // is what keeps a stop partway from reading as an old, working
            // install: the environment step refuses to serve it.
            marked.incomplete = true
            guard let data = try? JSONEncoder().encode(marked),
                (try? data.write(to: stampURL)) != nil
            else {
                throw LocalEngineError.fileOperationFailed(
                    step: step,
                    reason: "could not mark \(stampURL.path) before reinstalling — "
                        + "nothing was changed")
            }
        }
        try await runOrThrow(
            syncing
                ? [uv.path, "pip", "sync", "--python", interpreter, lock]
                : [uv.path, "pip", "install", "--python", interpreter, "-r", lock],
            step: step, extraEnvironment: overrides)
        // The lock pins third-party versions only; the engine package itself
        // is installed editable afterwards, exactly as bootstrap.sh does.
        try await runOrThrow(
            [uv.path, "pip", "install", "--python", interpreter, "--no-deps",
             "-e", serverDirectory],
            step: step, extraEnvironment: overrides)

        if let sha = observation.lockFileSHA256 {
            let stamp = LocalEngineLockStamp(
                lockFile: URL(filePath: lock).lastPathComponent,
                lockSHA256: sha,
                installedAt: ISO8601DateFormatter().string(from: Date()),
                uvVersion: uv.version, pythonVersion: PinnedCPython.minor)
            if let data = try? JSONEncoder().encode(stamp) {
                try? data.write(to: stampURL)
            }
            emit(
                "\(step.title): stamped the venv against "
                    + "\(URL(filePath: lock).lastPathComponent) "
                    + "(sha256 \(String(sha.prefix(8))))")
        }
    }

    /// Step 4 — serve, through the SAME posture resolution as the one-click
    /// script.
    ///
    /// Two source tiers, one posture:
    ///   * a CHECKOUT drives `scripts/start-local-server.sh` unchanged. That
    ///     script is the tested path, it writes the adoption pidfile, and its
    ///     argv carries `--dev-open-loopback` explicitly. Byte-compatible with
    ///     today.
    ///   * a MATERIALIZED ENGINE has no `scripts/` (the payload is the
    ///     `Server/` tree only), so the same argv is issued directly:
    ///     `python -m steerlab_server.cli serve --port P --root R
    ///     --dev-open-loopback`.
    ///
    /// Both go through `serve`, which is where WP-S resolves posture in
    /// `cli._serve` and EXPORTS it into the environment. Neither embeds the
    /// FastAPI app object — that is the documented bypass, and it would take
    /// the default `auth_mode = none` without any of serve's refusals (a
    /// non-loopback bind or a Slurm executor must exit 64, and only `serve`
    /// enforces that).
    private func startServer(
        observation: LocalEngineObservation, step: LocalEngineStep
    ) async throws {
        guard let source = environment.engineSource() else {
            throw LocalEngineError.blocked(
                step: step, reason: environment.unavailabilityReason())
        }
        let workspaceRoot = VectorCatalog.projectRoot
        let argv = Self.serveArgv(
            source: source, workspaceRoot: workspaceRoot, port: port)
        emit("\(step.title): " + argv.joined(separator: " "))

        let process = Process()
        process.executableURL = URL(filePath: argv[0])
        process.arguments = Array(argv.dropFirst())
        process.currentDirectoryURL = source.workingDirectory
        var environmentVariables = ProcessInfo.processInfo.environment
        environmentVariables["PYTHONUNBUFFERED"] = "1"
        process.environment = environmentVariables
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            throw LocalEngineError.commandFailed(
                step: step, command: argv.joined(separator: " "), exitCode: 127,
                lastLines: ["could not launch: \(error.localizedDescription)"])
        }
        // The materialized tier has no start script to write the adoption
        // pidfile, so write it here in the SAME format and with the same
        // binding contract: this process IS the server (no shell wrapper), and
        // the argv above names --port/--root explicitly, so
        // `LocalServerPidfile.commandLineMatchesSteerLabServer` can bind it
        // and the menu's Stop stays safe.
        if case .engineRoot = source {
            let pidfile = LocalServerPidfile.url(workspaceRoot: workspaceRoot)
            try? "\(process.processIdentifier) \(port)\n"
                .write(to: pidfile, atomically: true, encoding: .utf8)
        }
        // Drain output into the Activity feed for as long as the server runs.
        let handle = pipe.fileHandleForReading
        Task { @MainActor [weak self] in
            do {
                for try await line in handle.bytes.lines { self?.emit(line) }
            } catch {
                self?.emit("server log stream ended: \(error.localizedDescription)")
            }
        }
        // Wait for the endpoint to identify, rather than for a log line: the
        // endpoint is the thing acceptance is about to use.
        for _ in 0..<600 {
            if cancelRequested {
                process.terminate()
                throw LocalEngineError.cancelled(afterStep: .environment)
            }
            if environment.serverAnswers(port: port) { return }
            if !process.isRunning {
                throw LocalEngineError.commandFailed(
                    step: step, command: argv.joined(separator: " "),
                    exitCode: process.terminationStatus,
                    lastLines: ["the server exited before it answered on "
                        + "127.0.0.1:\(port) — see the Activity pane"])
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
        throw LocalEngineError.commandFailed(
            step: step, command: argv.joined(separator: " "), exitCode: 0,
            lastLines: ["the server did not answer on 127.0.0.1:\(port) within "
                + "60 seconds"])
    }

    /// The serve argv for a source tier — pure, so the posture flag and the
    /// pidfile-binding flags are asserted in tests rather than trusted.
    public nonisolated static func serveArgv(
        source: CodeResources.LocalEngineSource, workspaceRoot: URL, port: Int
    ) -> [String] {
        switch source {
        case .checkout(let checkout):
            // The tested one-click path, unchanged: the script itself passes
            // --dev-open-loopback and writes the pidfile.
            return [
                "/bin/zsh",
                checkout.root.appending(components: "scripts", "start-local-server.sh")
                    .path,
                "--root", workspaceRoot.path, "--port", "\(port)",
            ]
        case .engineRoot(let engine):
            return [
                engine.serverDirectory
                    .appending(components: ".venv.nosync", "bin", "python").path,
                "-m", "steerlab_server.cli", "serve",
                "--port", "\(port)", "--root", workspaceRoot.path,
                "--dev-open-loopback",
            ]
        }
    }

    /// Step 5 — `/api/info`, then `site qualify`, rows and all.
    private func runAcceptance(step: LocalEngineStep) async throws {
        guard let source = environment.engineSource() else {
            throw LocalEngineError.blocked(
                step: step, reason: environment.unavailabilityReason())
        }
        if let response = LocalServerPidfile.fetchInfoResponse(port: port),
            let version = Self.engineVersion(fromInfoResponse: response)
        {
            engineVersion = version
            emit("\(step.title): /api/info → engineVersion \(version)")
            skewAdvisory = Self.skewAdvisory(
                engineVersion: version, appRevision: environment.appRevision)
            if let skewAdvisory { emit("\(step.title): advisory — \(skewAdvisory)") }
        } else {
            emit(
                "\(step.title): /api/info answered without an engineVersion — "
                    + "the qualification below still ran")
        }

        let reportFile = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-site-qualify-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: reportFile) }
        let argv = Self.qualifyArgv(
            source: source, workspaceRoot: VectorCatalog.projectRoot,
            reportFile: reportFile)
        let result = await shell.run(argv)
        for line in result.lines.suffix(40) { emit(line) }
        guard let data = try? Data(contentsOf: reportFile),
            let report = try? SiteQualificationReport.decode(data)
        else {
            throw LocalEngineError.commandFailed(
                step: step, command: "site qualify", exitCode: result.exitCode,
                lastLines: result.lines.suffix(3).map { $0 })
        }
        qualification = report
        emit("\(step.title): \(report.summaryLine)")
        for row in report.checks {
            emit("  \(row.status.uppercased()) \(row.id): \(row.observed)")
        }
        // A FAILING check is not a flow failure — the engine is provisioned
        // and running, and the report is the product. It is surfaced as the
        // report's own verdict so the researcher reads the rows.
        if !report.failing.isEmpty {
            statusLine = "running, but site qualify failed: "
                + report.failing.joined(separator: ", ")
        }
    }

    /// The `site qualify` argv — pure, so the root pin is asserted in tests
    /// rather than trusted.
    ///
    /// `--root` is LOAD-BEARING (field incident 2026-08-29): the qualify
    /// subprocess inherits the app's environment and working directory —
    /// no `STEERLAB_ROOT`, cwd `/` when launched from Finder — so without
    /// the pin its profile check derived the artifact root from `getcwd()`
    /// and probed a metadataRoot at the filesystem root (`/.steerlab does
    /// not exist`), a verdict about a deployment nobody was running. The
    /// pin names the SAME workspace root the serve step passes, so the
    /// check describes the server it just started.
    public nonisolated static func qualifyArgv(
        source: CodeResources.LocalEngineSource, workspaceRoot: URL,
        reportFile: URL
    ) -> [String] {
        [
            source.venvPython.path, "-u", "-m", "steerlab_server.cli",
            "site", "qualify", "--root", workspaceRoot.path,
            "--json", reportFile.path,
        ]
    }

    /// `{"service": "steerlab-server", "engineVersion": "steerlab-server 0.1.0+ab12cd34", …}`
    /// → the version string. Pure for tests.
    public nonisolated static func engineVersion(fromInfoResponse response: String)
        -> String?
    {
        guard let start = response.range(of: "\"engineVersion\"") else { return nil }
        var rest = response[start.upperBound...].drop { $0 != ":" }
        guard !rest.isEmpty else { return nil }
        rest = rest.dropFirst().drop(while: { $0 == " " })
        guard rest.first == "\"" else { return nil }
        let value = rest.dropFirst().prefix { $0 != "\"" }
        return value.isEmpty ? nil : String(value)
    }

    /// Skew between what the app was built from and what the engine reports.
    /// Stated, labelled, and non-blocking — the two are separately installable
    /// by design, and a mismatch is information, not a fault. Pure for tests.
    public nonisolated static func skewAdvisory(
        engineVersion: String?, appRevision: String?
    ) -> String? {
        guard let engineVersion, let appRevision, !appRevision.isEmpty
        else { return nil }
        // engine_version() is "steerlab-server <version>+<sha8>" (or bare).
        guard let plus = engineVersion.lastIndex(of: "+") else {
            return "the engine reports \(engineVersion), which carries no build "
                + "revision, so it cannot be compared with this app build "
                + "(\(appRevision))"
        }
        let engineRevision = String(engineVersion[engineVersion.index(after: plus)...])
        guard engineRevision != appRevision else { return nil }
        return "engine build \(engineRevision), app build \(appRevision) — the "
            + "two were assembled from different revisions. Not a fault (they "
            + "install separately), but worth refreshing the engine before a "
            + "measured run"
    }

    // MARK: Command plumbing

    private func runOrThrow(
        _ argv: [String], step: LocalEngineStep,
        extraEnvironment: [String: String] = [:]
    ) async throws {
        let command = argv.joined(separator: " ")
        let result: ClusterShellResult
        if extraEnvironment.isEmpty {
            result = await shell.run(argv)
        } else {
            // `/usr/bin/env K=V …` rather than a second runner protocol: the
            // shell seam takes an argv and nothing else, and keeping it that
            // way is what makes every command in this file assertable as a
            // list of strings.
            let prefixed = ["/usr/bin/env"]
                + extraEnvironment.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }
                + argv
            result = await shell.run(prefixed)
        }
        for line in result.lines.suffix(20) { emit(line) }
        guard result.succeeded else {
            throw LocalEngineError.commandFailed(
                step: step, command: command, exitCode: result.exitCode,
                lastLines: result.lines.suffix(3).map { $0 })
        }
    }
}

// MARK: - Download seam

public protocol LocalEngineDownloader: Sendable {
    /// Fetch the whole artifact. Returning `Data` rather than a file path is
    /// deliberate: the hash check then happens on the bytes this process
    /// actually holds, with nothing on disk in between for anything else to
    /// swap.
    func download(from url: URL) async throws -> Data
}

public struct URLSessionEngineDownloader: LocalEngineDownloader {
    public init() {}

    public func download(from url: URL) async throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = 120
        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw LocalEngineError.downloadFailed(
                url: url.absoluteString, reason: "HTTP \(http.statusCode)")
        }
        return data
    }
}

// MARK: - site qualify's report, as rows

/// The `site qualify` document (`Server/steerlab_server/site_qualify.py`),
/// decoded far enough to RENDER THE ROWS. The Mac baseline is 7 pass / 1 warn
/// / 1 skip, and a summary that only said "passed" would hide the skip —
/// which is the one number that says how much was actually verified.
public struct SiteQualificationReport: Sendable, Equatable {

    public struct Check: Sendable, Equatable, Identifiable {
        public let id: String
        public let title: String
        /// "pass" / "warn" / "fail" / "skip".
        public let status: String
        public let expected: String
        public let observed: String
        public let detail: String
    }

    public let generatedBy: String
    public let platform: String
    public let checks: [Check]
    public let passed: Int
    public let warnings: Int
    public let failed: Int
    public let skipped: Int
    public let summaryLine: String

    public var failing: [String] {
        checks.filter { $0.status == "fail" }.map(\.id)
    }

    public var warningIDs: [String] {
        checks.filter { $0.status == "warn" }.map(\.id)
    }

    public static func decode(_ data: Data) throws -> SiteQualificationReport {
        struct Wire: Decodable {
            struct Row: Decodable {
                let id: String
                let title: String
                let status: String
                let expected: String
                let observed: String
                let detail: String
            }
            struct Summary: Decodable {
                let passed: Int
                let warnings: Int
                let failed: Int
                let skipped: Int
                let line: String
            }
            let generatedBy: String
            let platform: String
            let checks: [Row]
            let summary: Summary
        }
        let wire = try JSONDecoder().decode(Wire.self, from: data)
        return SiteQualificationReport(
            generatedBy: wire.generatedBy, platform: wire.platform,
            checks: wire.checks.map {
                Check(
                    id: $0.id, title: $0.title, status: $0.status,
                    expected: $0.expected, observed: $0.observed, detail: $0.detail)
            },
            passed: wire.summary.passed, warnings: wire.summary.warnings,
            failed: wire.summary.failed, skipped: wire.summary.skipped,
            summaryLine: wire.summary.line)
    }
}

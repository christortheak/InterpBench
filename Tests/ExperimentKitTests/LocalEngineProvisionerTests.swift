import Foundation
import Testing

@testable import ExperimentKit

// =============================================================================
// WP3 — the local-engine flow, tested where it can actually be tested.
//
// The flow's whole value is the states it handles on SOMEBODY ELSE'S Mac: no
// uv, a uv too old to drive, no venv, a venv that drifted from the lock, an app
// with no checkout beside it. This machine has none of those — it has a warm
// developer checkout with a working venv. So the decision layer is pure over a
// `LocalEngineObservation` and every one of those states is a fixture here,
// while the live proof (below, in the report) is limited to what a warm machine
// can honestly demonstrate: the idempotent path, and the argv/pidfile contract.
// =============================================================================

@Suite struct LocalEngineDecisionTests {

    // MARK: Fixtures

    private func observation(
        source: LocalEngineObservation.SourceState = .checkout(path: "/code"),
        systemUV: LocalEngineTool? = nil,
        managedUV: LocalEngineTool? = nil,
        pythonInstalled: Bool = false,
        venvExists: Bool = false,
        stamp: LocalEngineLockStamp? = nil,
        lockSHA: String? = "aaaa1111",
        serverAnswering: Bool = false
    ) -> LocalEngineObservation {
        LocalEngineObservation(
            source: source, appRevision: "app12345", systemUV: systemUV,
            managedUV: managedUV, managedPythonInstalled: pythonInstalled,
            venvPythonExists: venvExists, venvLockStamp: stamp,
            lockFileSHA256: lockSHA,
            lockFilePath: lockSHA == nil
                ? nil : "/code/Server/" + LocalEngineLayout.lockFileName,
            serverAnswering: serverAnswering, port: 8080)
    }

    private func stamp(sha: String) -> LocalEngineLockStamp {
        LocalEngineLockStamp(
            lockFile: LocalEngineLayout.lockFileName, lockSHA256: sha,
            installedAt: "2026-08-20T00:00:00Z")
    }

    // MARK: Version comparison (the gate on "may I drive this uv?")

    @Test func versionComparisonOrdersNumerically() {
        #expect(LocalEngineDecisions.compareVersions("0.12.5", "0.12.5") == 0)
        #expect(LocalEngineDecisions.compareVersions("0.11.33", "0.12.5") == -1)
        // The trap a lexical compare falls into: 33 > 5.
        #expect(LocalEngineDecisions.compareVersions("0.12.33", "0.12.5") == 1)
        #expect(LocalEngineDecisions.compareVersions("1.0", "0.99.99") == 1)
        #expect(LocalEngineDecisions.compareVersions("0.12.5+abc", "0.12.5") == 0)
    }

    @Test func aToolThatDidNotAnswerIsNeverUsable() {
        // A half-extracted binary answers nothing; driving it would be worse
        // than downloading again.
        #expect(!LocalEngineDecisions.isUsable(LocalEngineTool(path: "/x/uv", version: nil)))
        #expect(!LocalEngineDecisions.isUsable(nil))
        #expect(LocalEngineDecisions.isUsable(LocalEngineTool(path: "/x/uv", version: "0.12.5")))
        #expect(LocalEngineDecisions.isUsable(LocalEngineTool(path: "/x/uv", version: "0.12.9")))
    }

    // MARK: Step 2 — interpreter

    @Test func noUVPlansADownloadAndACPythonInstall() {
        let state = LocalEngineDecisions.interpreterState(observation())
        #expect(
            state.actions == [
                .downloadUV(destination: LocalEngineLayout.managedUV.path),
                .installCPython(minor: PinnedCPython.minor),
            ])
    }

    @Test func aStaleSystemUVIsReplacedRatherThanDriven() {
        let state = LocalEngineDecisions.interpreterState(
            observation(systemUV: LocalEngineTool(path: "/opt/bin/uv", version: "0.11.33")))
        #expect(state.actions.contains(.downloadUV(destination: LocalEngineLayout.managedUV.path)))
    }

    @Test func aUsableSystemUVSkipsTheDownload() {
        let state = LocalEngineDecisions.interpreterState(
            observation(
                systemUV: LocalEngineTool(path: "/opt/bin/uv", version: "0.12.7"),
                pythonInstalled: false))
        #expect(!state.actions.contains(.downloadUV(destination: LocalEngineLayout.managedUV.path)))
        #expect(state.actions == [.installCPython(minor: PinnedCPython.minor)])
    }

    @Test func everythingPresentMakesTheStepASkipThatSaysWhy() {
        let state = LocalEngineDecisions.interpreterState(
            observation(
                systemUV: LocalEngineTool(path: "/opt/bin/uv", version: "0.12.7"),
                pythonInstalled: true))
        #expect(state.actions.isEmpty)
        #expect(state.summary.contains("0.12.7"))
        #expect(state.summary.contains(PinnedCPython.minor))
    }

    @Test func theManagedUVWinsOverAUsableSystemOne() {
        let chosen = LocalEngineDecisions.chosenUV(
            observation(
                systemUV: LocalEngineTool(path: "/opt/bin/uv", version: "0.12.9"),
                managedUV: LocalEngineTool(
                    path: LocalEngineLayout.managedUV.path, version: "0.12.5")))
        #expect(chosen?.path == LocalEngineLayout.managedUV.path)
    }

    // MARK: Step 3 — environment

    @Test func aMissingVenvPlansCreationThenTheLockInstall() {
        let state = LocalEngineDecisions.environmentState(observation(venvExists: false))
        #expect(state.actions.count == 2)
        guard case .createVenv(let path) = state.actions[0] else {
            Issue.record("first action was not createVenv")
            return
        }
        #expect(path.hasSuffix("/Server/.venv.nosync"))
        guard case .installLock(let lock, let editable) = state.actions[1] else {
            Issue.record("second action was not installLock")
            return
        }
        #expect(lock.hasSuffix(LocalEngineLayout.lockFileName))
        #expect(editable.hasSuffix("/Server"))
    }

    /// THE developer-protection property: every venv that exists today was
    /// made by `start-local-server.sh` with pip and floors, so it has no lock
    /// stamp. That must never trigger a multi-gigabyte reinstall nobody asked
    /// for — it is an advisory with ZERO actions.
    @Test func anUnstampedVenvIsAnAdvisoryAndNeverAReinstall() {
        let state = LocalEngineDecisions.environmentState(
            observation(venvExists: true, stamp: nil))
        #expect(state.actions.isEmpty)
        #expect(!state.isBlocked)
        let advisory = state.advisory ?? ""
        #expect(advisory.contains("no lock stamp"))
        #expect(advisory.contains("start-local-server.sh"))
    }

    @Test func aMatchingStampIsAQuietSkip() {
        let state = LocalEngineDecisions.environmentState(
            observation(venvExists: true, stamp: stamp(sha: "aaaa1111"), lockSHA: "aaaa1111"))
        #expect(state.actions.isEmpty)
        #expect(state.advisory == nil)
        #expect(state.summary.contains("matches"))
    }

    @Test func aDriftedStampSaysBothHashes() {
        let state = LocalEngineDecisions.environmentState(
            observation(venvExists: true, stamp: stamp(sha: "bbbb2222"), lockSHA: "cccc3333"))
        let advisory = try? #require(state.advisory)
        #expect(advisory?.contains("bbbb2222") == true)
        #expect(advisory?.contains("cccc3333") == true)
        // Still no forced action: the researcher decides when 2 GB moves.
        #expect(state.actions.isEmpty)
    }

    /// No lock in the tree is a REFUSAL, not a fallback to pyproject floors.
    /// An unpinned local environment is exactly the drift the committed locks
    /// exist to stop.
    @Test func aTreeWithNoLockIsBlockedRatherThanResolvedFromFloors() {
        let state = LocalEngineDecisions.environmentState(observation(lockSHA: nil))
        #expect(state.isBlocked)
        #expect(state.summary.contains("pyproject floors"))
    }

    // MARK: Steps 1, 4, 5

    @Test func aCheckoutNeedsNoMaterialization() {
        let state = LocalEngineDecisions.engineSourceState(
            observation(source: .checkout(path: "/code")))
        #expect(state.actions.isEmpty)
        #expect(state.summary.contains("/code"))
    }

    @Test func aBundleWithNoTreePlansAMaterialization() {
        let state = LocalEngineDecisions.engineSourceState(
            observation(
                source: .materializable(destination: "/fixture/Engine", payload: "/fixture/App/ServerPayload")))
        #expect(state.actions == [.materializeEngine(destination: "/fixture/Engine")])
    }

    @Test func engineSkewIsStatedAndNonBlocking() {
        let same = LocalEngineDecisions.engineSourceState(
            observation(source: .engineRoot(path: "/fixture/Engine", stampRevision: "app12345")))
        #expect(same.advisory == nil)
        #expect(same.actions.isEmpty)

        let different = LocalEngineDecisions.engineSourceState(
            observation(source: .engineRoot(path: "/fixture/Engine", stampRevision: "old99999")))
        #expect(different.actions.isEmpty)  // non-blocking: never a forced refresh
        #expect(different.advisory?.contains("old99999") == true)
        #expect(different.advisory?.contains("app12345") == true)

        let unstamped = LocalEngineDecisions.engineSourceState(
            observation(source: .engineRoot(path: "/fixture/Engine", stampRevision: nil)))
        #expect(unstamped.advisory?.contains("no engine stamp") == true)
    }

    @Test func noSourceAtAllIsBlockedWithALayoutReason() {
        let state = LocalEngineDecisions.engineSourceState(
            observation(source: .unavailable(reason: "no payload and no checkout")))
        #expect(state.isBlocked)
    }

    @Test func aLiveServerMakesTheServeStepASkip() {
        #expect(
            LocalEngineDecisions.serveState(observation(serverAnswering: true)).actions.isEmpty)
        #expect(
            LocalEngineDecisions.serveState(observation(serverAnswering: false)).actions
                == [.startServer(port: 8080)])
    }

    @Test func acceptanceAlwaysRuns() {
        // Evidence from a previous run is not evidence about this one.
        #expect(
            LocalEngineDecisions.acceptanceState(observation(serverAnswering: true)).actions
                == [.runQualification])
    }

    // MARK: The whole plan

    @Test func aFullyProvisionedMachineHasNothingToDoButReVerify() {
        let warm = observation(
            managedUV: LocalEngineTool(path: LocalEngineLayout.managedUV.path, version: "0.12.5"),
            pythonInstalled: true, venvExists: true,
            stamp: stamp(sha: "aaaa1111"), lockSHA: "aaaa1111", serverAnswering: true)
        let plan = LocalEngineDecisions.plan(warm)
        #expect(LocalEngineDecisions.isAlreadyProvisioned(plan))
        #expect(LocalEngineDecisions.downloadPreamble(plan).isEmpty)
    }

    @Test func aColdMachinePlansEveryStepAndNamesTheDownloads() {
        let cold = observation(
            source: .materializable(destination: "/fixture/Engine", payload: "/fixture/App/ServerPayload"))
        let plan = LocalEngineDecisions.plan(cold)
        #expect(!LocalEngineDecisions.isAlreadyProvisioned(plan))
        let preamble = LocalEngineDecisions.downloadPreamble(plan)
        // uv is pinned, so it is stated exactly; CPython and the wheels are
        // estimates and must say "about".
        #expect(preamble.contains { $0.contains("uv \(PinnedUV.version)") && !$0.contains("about") })
        #expect(preamble.contains { $0.contains("CPython") && $0.contains("about") })
        #expect(preamble.contains { $0.contains("about 2.0 GB") })
        #expect(preamble.last?.hasPrefix("Total: about") == true)
    }

    // MARK: The serve argv (posture + pidfile binding)

    @Test func theCheckoutTierDrivesTheOneClickScriptUnchanged() {
        let checkout = CodeResources.ExecutableCheckout(
            root: URL(filePath: "/code"), origin: .compiledCheckout)
        let argv = LocalEngineProvisioner.serveArgv(
            source: .checkout(checkout), workspaceRoot: URL(filePath: "/data/ws"), port: 8080)
        #expect(argv[0] == "/bin/zsh")
        #expect(argv[1] == "/code/scripts/start-local-server.sh")
        #expect(argv.contains("--root"))
        #expect(argv.contains("/data/ws"))
        #expect(argv.contains("--port"))
        #expect(argv.contains("8080"))
    }

    /// The materialized tier has no `scripts/`, so it issues the same argv
    /// directly — and the two things that argv has to carry are the WP-S
    /// posture flag and the pidfile binding.
    @Test func theEngineTierIssuesTheSamePostureRespectingArgv() {
        let engine = CodeResources.EngineRoot(root: URL(filePath: "/fixture/Engine"), stamp: nil)
        let argv = LocalEngineProvisioner.serveArgv(
            source: .engineRoot(engine), workspaceRoot: URL(filePath: "/data/ws"), port: 8123)
        #expect(argv[0] == "/fixture/Engine/Server/.venv.nosync/bin/python")
        #expect(argv.contains("-m"))
        #expect(argv.contains("steerlab_server.cli"))
        #expect(argv.contains("serve"))
        // The posture flag WP-S requires on the argv: `cli._serve` resolves
        // and EXPORTS the posture, and it refuses on a non-loopback bind or a
        // Slurm executor. Embedding the app object would bypass all of that.
        #expect(argv.contains("--dev-open-loopback"))
        #expect(!argv.contains { $0.contains("uvicorn") })

        // …and the adoption contract: the pidfile's Stop button is only safe
        // when the recorded pid's argv BINDS this port and root.
        let command = argv.joined(separator: " ")
        #expect(
            LocalServerPidfile.commandLineMatchesSteerLabServer(
                command, port: 8123, workspaceRoot: "/data/ws"))
        #expect(
            !LocalServerPidfile.commandLineMatchesSteerLabServer(
                command, port: 9999, workspaceRoot: "/data/ws"))
    }

    // MARK: Acceptance parsing

    @Test func engineVersionIsReadFromTheInfoResponse() {
        let body = "HTTP/1.1 200 OK\r\n\r\n"
            + #"{"service": "steerlab-server", "engineVersion": "steerlab-server 0.1.0+ab12cd34", "root": "/x"}"#
        #expect(
            LocalEngineProvisioner.engineVersion(fromInfoResponse: body)
                == "steerlab-server 0.1.0+ab12cd34")
        #expect(LocalEngineProvisioner.engineVersion(fromInfoResponse: "{}") == nil)
    }

    @Test func skewIsStatedLabelledAndNonBlocking() {
        #expect(
            LocalEngineProvisioner.skewAdvisory(
                engineVersion: "steerlab-server 0.1.0+ab12cd34", appRevision: "ab12cd34") == nil)
        let advisory = LocalEngineProvisioner.skewAdvisory(
            engineVersion: "steerlab-server 0.1.0+ab12cd34", appRevision: "ff00ee11")
        #expect(advisory?.contains("ab12cd34") == true)
        #expect(advisory?.contains("ff00ee11") == true)
        #expect(advisory?.contains("Not a fault") == true)
        // No app revision (a dev build) → nothing to compare, so nothing said.
        #expect(
            LocalEngineProvisioner.skewAdvisory(
                engineVersion: "steerlab-server 0.1.0+ab12cd34", appRevision: nil) == nil)
    }

    // MARK: The hash refusal's wording

    @Test func aHashMismatchRefusesLoudlyAndNamesBothHashes() {
        let error = LocalEngineError.hashMismatch(
            url: PinnedUV.downloadURL.absoluteString, expected: PinnedUV.sha256,
            observed: "deadbeef")
        #expect(error.description.contains("REFUSED"))
        #expect(error.description.contains(PinnedUV.sha256))
        #expect(error.description.contains("deadbeef"))
        #expect(error.description.contains("Nothing was installed"))
        #expect(error.description.contains("do not work around it"))
    }

    /// The pin's provenance, asserted so a careless bump cannot leave a
    /// placeholder behind: 64 lowercase hex, and inside the range this
    /// project's own `Server[dev]` extra declares for the lock resolver.
    @Test func thePinnedUVConstantsAreWellFormed() {
        #expect(PinnedUV.sha256.count == 64)
        #expect(PinnedUV.sha256.allSatisfy { $0.isHexDigit && !$0.isUppercase })
        #expect(PinnedUV.byteCount > 1_000_000)
        #expect(LocalEngineDecisions.compareVersions(PinnedUV.version, "0.12.5") >= 0)
        #expect(LocalEngineDecisions.compareVersions(PinnedUV.version, "0.13.0") < 0)
        #expect(PinnedUV.downloadURL.absoluteString.contains(PinnedUV.version))
        #expect(PinnedUV.downloadURL.host() == "github.com")
    }
}

// =============================================================================
// Resolution tiers: a checkout is preferred, an engine root is used only when
// there is none, and the two are never conflated.
// =============================================================================

@Suite(.serialized) struct LocalEngineResolutionTests {

    private func makeEngineRoot() throws -> URL {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appending(component: "steerlab-engine-\(UUID().uuidString)")
        try fm.createDirectory(
            at: root.appending(path: "Server/steerlab_server"),
            withIntermediateDirectories: true)
        try "[project]\n".write(
            to: root.appending(path: "Server/pyproject.toml"),
            atomically: true, encoding: .utf8)
        return root
    }

    private func makeHome(withCheckout: Bool) throws -> URL {
        let fm = FileManager.default
        let home = fm.temporaryDirectory
            .appending(component: "steerlab-home-\(UUID().uuidString)")
        try fm.createDirectory(at: home, withIntermediateDirectories: true)
        if withCheckout {
            let root = home.appending(component: "code")
            try fm.createDirectory(
                at: root.appending(path: "Server/steerlab_server"),
                withIntermediateDirectories: true)
            try "// swift-tools-version: 6.2\n".write(
                to: root.appending(component: "Package.swift"),
                atomically: true, encoding: .utf8)
        }
        return home
    }

    @Test func theCheckoutTierWinsOnThisDeveloperMachine() throws {
        // No overrides: this test process runs from the checkout, so the local
        // engine resolves exactly where the pre-WP3 code did.
        let source = try #require(CodeResources.localEngineSource())
        guard case .checkout(let checkout) = source else {
            Issue.record("resolved \(source) rather than the developer checkout")
            return
        }
        #expect(checkout.origin == .compiledCheckout)
        #expect(source.supportsDeveloperFeatures)
        // …and the venv spelling is the SAME one LocalPythonRuntime resolves,
        // which is what keeps LocalServerController's path untouched.
        #expect(source.venvPython.path == LocalPythonRuntime.venvPython?.path)
    }

    @Test func theEngineRootIsUsedOnlyWhenThereIsNoCheckout() throws {
        let fm = FileManager.default
        let engine = try makeEngineRoot()
        let emptyHome = try makeHome(withCheckout: false)
        let homeWithCheckout = try makeHome(withCheckout: true)
        defer {
            try? fm.removeItem(at: engine)
            try? fm.removeItem(at: emptyHome)
            try? fm.removeItem(at: homeWithCheckout)
        }

        ExperimentRootOverrideLock.acquire()
        CodeResources.modeOverrideForTesting = .release
        CodeResources.engineRootOverrideForTesting = engine
        defer {
            CodeResources.modeOverrideForTesting = nil
            CodeResources.engineRootOverrideForTesting = nil
            CodeResources.executableHomesOverrideForTesting = nil
            ExperimentRootOverrideLock.release()
        }

        // A checkout beside the app beats the engine root, every time.
        CodeResources.executableHomesOverrideForTesting = [homeWithCheckout]
        guard case .checkout = try #require(CodeResources.localEngineSource()) else {
            Issue.record("the engine root outranked a real checkout")
            return
        }

        // With no checkout anywhere, the engine root answers.
        CodeResources.executableHomesOverrideForTesting = [emptyHome]
        let source = try #require(CodeResources.localEngineSource())
        guard case .engineRoot(let root) = source else {
            Issue.record("resolved \(source) rather than the engine root")
            return
        }
        #expect(root.root == engine.standardizedFileURL)
        #expect(root.serverDirectory.lastPathComponent == "Server")
        // It is NOT a checkout, and says so — no dev feature may run from it.
        #expect(!source.supportsDeveloperFeatures)
    }

    /// The conflation guard, by content: an engine root has no
    /// `Package.swift`, so the checkout resolver can never find one, and
    /// `isEngineRoot` refuses a tree that IS a checkout.
    @Test func anEngineRootIsNeverMistakenForACheckout() throws {
        let fm = FileManager.default
        let engine = try makeEngineRoot()
        let emptyHome = try makeHome(withCheckout: false)
        defer {
            try? fm.removeItem(at: engine)
            try? fm.removeItem(at: emptyHome)
        }
        #expect(!HomeLayout.isCheckout(engine))
        #expect(CodeResources.isEngineRoot(engine))

        // Add the SwiftPM manifest and it becomes a checkout, at which point
        // the engine-root probe refuses it: exactly one tier may claim a tree.
        try "// swift-tools-version: 6.2\n".write(
            to: engine.appending(component: "Package.swift"),
            atomically: true, encoding: .utf8)
        #expect(HomeLayout.isCheckout(engine))
        #expect(!CodeResources.isEngineRoot(engine))
    }

    @Test func anEngineRootBesideAHomeIsNotReturnedByTheCheckoutResolver() throws {
        let fm = FileManager.default
        let home = try makeHome(withCheckout: false)
        // Put the engine tree INSIDE the home, where the sibling search looks.
        let engine = home.appending(component: "Engine")
        try fm.createDirectory(
            at: engine.appending(path: "Server/steerlab_server"),
            withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: home) }

        ExperimentRootOverrideLock.acquire()
        CodeResources.modeOverrideForTesting = .release
        CodeResources.executableHomesOverrideForTesting = [home]
        defer {
            CodeResources.modeOverrideForTesting = nil
            CodeResources.executableHomesOverrideForTesting = nil
            ExperimentRootOverrideLock.release()
        }
        switch CodeResources.executableCheckout() {
        case .success(let found):
            Issue.record("the engine root was returned as a checkout: \(found.root.path)")
        case .failure(let absence):
            #expect(absence.searched == [home])
        }
        // And the bucket-B consumers stay nil — the engine root must never
        // become the gemmascope venv's home or the start script's tree.
        #expect(LocalPythonRuntime.repoRoot == nil)
        #expect(LocalPythonRuntime.startServerScript == nil)
    }

    /// LIVE, and deliberately conditional: this observes the REAL machine
    /// through the real environment, so it asserts only what is true of any
    /// developer checkout rather than of this one Mac. Its job is to catch the
    /// observation layer wiring itself to the wrong paths — a plan whose steps
    /// disagree with what is actually on disk.
    @MainActor
    @Test func theLiveObservationAgreesWithWhatIsOnDisk() async throws {
        let environment = SystemLocalEngineEnvironment()
        let observation = await environment.observe(port: 8080)

        guard case .checkout(let path) = observation.source else {
            // A packaged build with no checkout is a legitimate state; there
            // is simply nothing to assert about a checkout here.
            return
        }
        let server = URL(filePath: path).appending(component: "Server")
        // The environment step must agree with the file system, in BOTH
        // directions — this is the assertion that would have caught a venv
        // probe pointed at the bundle or at the wrong tier.
        let venvOnDisk = FileManager.default.isExecutableFile(
            atPath: server.appending(path: ".venv.nosync/bin/python").path)
        #expect(observation.venvPythonExists == venvOnDisk)
        let state = LocalEngineDecisions.environmentState(observation)
        if venvOnDisk {
            #expect(state.actions.isEmpty, "an existing venv must never be reinstalled unasked")
        } else {
            #expect(state.actions.count == 2)
        }
        // The committed macOS lock is checkout content, so it must be found
        // and hashed; a nil here means the flow would refuse a real checkout.
        #expect(observation.lockFilePath?.hasSuffix(LocalEngineLayout.lockFileName) == true)
        #expect(observation.lockFileSHA256?.count == 64)
        // Step 1 is always a skip in a checkout — nothing is ever materialized
        // over a developer's own tree.
        #expect(LocalEngineDecisions.engineSourceState(observation).actions.isEmpty)
    }

    @Test func theEngineStampRoundTrips() throws {
        let fm = FileManager.default
        let engine = try makeEngineRoot()
        defer { try? fm.removeItem(at: engine) }
        #expect(CodeResources.engineStamp(at: engine) == nil)  // honestly absent

        let stamp = CodeResources.EngineStamp(
            sourceRevision: "ab12cd34", appVersion: "0.9.0-dev",
            materializedAt: "2026-08-20T00:00:00Z", fileCount: 231)
        try JSONEncoder().encode(stamp).write(
            to: engine.appending(component: CodeResources.EngineStamp.fileName))
        #expect(CodeResources.engineStamp(at: engine) == stamp)
    }
}

// =============================================================================
// The driver: cancellation and the hash refusal, against fakes.
// =============================================================================

/// A canned environment. Every field is a fixture, so the states this Mac will
/// never be in are all reachable.
private struct FakeEngineEnvironment: LocalEngineEnvironment {
    var source: CodeResources.LocalEngineSource?
    var materialization: (destination: URL, payload: URL)?
    var reason = "no payload and no checkout"
    var appRevision: String? = "app12345"
    var systemTool: LocalEngineTool?
    var managedTool: LocalEngineTool?
    var pythonInstalled = false
    var venvExists = false
    var lockSHA: String? = "aaaa1111"
    var stamp: LocalEngineLockStamp?
    var answering = false

    func engineSource() -> CodeResources.LocalEngineSource? { source }
    func materializationPlan() -> (destination: URL, payload: URL)? { materialization }
    func unavailabilityReason() -> String { reason }
    func locateSystemUV() async -> LocalEngineTool? { systemTool }
    func locateManagedUV() async -> LocalEngineTool? { managedTool }
    func managedPythonIsInstalled(uv: LocalEngineTool) async -> Bool { pythonInstalled }
    func isExecutableFile(_ url: URL) -> Bool { venvExists }
    func sha256OfFile(_ url: URL) -> String? { lockSHA }
    func readLockStamp(venv: URL) -> LocalEngineLockStamp? { stamp }
    func serverAnswers(port: Int) -> Bool { answering }
}

/// A downloader that hands back bytes that are NOT the pinned artifact.
private struct SubstitutedArtifactDownloader: LocalEngineDownloader {
    func download(from url: URL) async throws -> Data {
        Data("not the pinned uv".utf8)
    }
}

private struct NeverRunShell: ClusterShellRunner {
    func run(_ argv: [String]) async -> ClusterShellResult {
        ClusterShellResult(exitCode: 0, lines: ["fake: " + argv.joined(separator: " ")])
    }
}

/// A shell that PARKS inside the first command until the test releases it, so
/// "cancel arrived while a step was in flight" is a deterministic fact rather
/// than a race the test hopes to win.
private final class GatedShell: ClusterShellRunner, @unchecked Sendable {
    private let lock = NSLock()
    private var entered = false
    private var released = false

    var hasEntered: Bool { lock.withLock { entered } }
    func release() { lock.withLock { released = true } }

    func run(_ argv: [String]) async -> ClusterShellResult {
        lock.withLock { entered = true }
        while !lock.withLock({ released }) { await Task.yield() }
        return ClusterShellResult(exitCode: 0, lines: ["gated: " + argv.joined(separator: " ")])
    }
}

@Suite struct LocalEngineDriverTests {

    private func warmEnvironment() -> FakeEngineEnvironment {
        FakeEngineEnvironment(
            source: .checkout(
                CodeResources.ExecutableCheckout(
                    root: URL(filePath: "/code"), origin: .compiledCheckout)),
            managedTool: LocalEngineTool(
                path: LocalEngineLayout.managedUV.path, version: "0.12.5"),
            pythonInstalled: true, venvExists: true,
            stamp: LocalEngineLockStamp(
                lockFile: LocalEngineLayout.lockFileName, lockSHA256: "aaaa1111",
                installedAt: "2026-08-20T00:00:00Z"),
            answering: true)
    }

    @MainActor
    @Test func planningAWarmMachineReportsReadyAndPlansNoDownloads() async {
        let provisioner = LocalEngineProvisioner(
            environment: warmEnvironment(), shell: NeverRunShell(),
            downloader: SubstitutedArtifactDownloader())
        await provisioner.refreshPlan()
        guard case .ready = provisioner.phase else {
            Issue.record("warm machine reported \(provisioner.phase)")
            return
        }
        #expect(provisioner.downloadPreamble.isEmpty)
        #expect(provisioner.plan.count == LocalEngineStep.allCases.count)
    }

    /// Cancellation is checked BETWEEN steps: a cancel that arrives while step
    /// 2 is running lets step 2 finish and stops before step 3. A cancelled
    /// run says so rather than reading as a failure, and nothing is undone —
    /// which is why the message promises that re-running continues.
    @MainActor
    @Test func cancellationStopsBetweenStepsAndSaysSo() async {
        var environment = warmEnvironment()
        // A usable system uv with no managed CPython gives step 2 exactly one
        // shell command — the one the gate parks in.
        environment.managedTool = nil
        environment.systemTool = LocalEngineTool(path: "/opt/bin/uv", version: "0.12.7")
        environment.pythonInstalled = false
        let shell = GatedShell()
        let provisioner = LocalEngineProvisioner(
            environment: environment, shell: shell,
            downloader: SubstitutedArtifactDownloader())

        let run = Task { await provisioner.runAwaitingCompletion() }
        while !shell.hasEntered { await Task.yield() }
        provisioner.cancel()
        #expect(provisioner.statusLine.contains("cancelling"))
        shell.release()
        await run.value

        #expect(provisioner.phase == .cancelled)
        #expect(provisioner.statusLine.contains("re-running continues"))
        // Step 2 completed (the command it was in the middle of was allowed to
        // finish); step 3 never started.
        #expect(provisioner.progress[.interpreter] == "done")
        #expect(provisioner.progress[.environment] == nil)
    }

    /// A substituted or corrupted uv download is a REFUSAL: the flow fails,
    /// names both hashes, and nothing is installed.
    @MainActor
    @Test func aSubstitutedDownloadIsRefusedAndInstallsNothing() async {
        var environment = warmEnvironment()
        environment.managedTool = nil  // forces the download action
        environment.pythonInstalled = false
        // "Installed nothing" is asserted as UNCHANGED rather than as absent:
        // on a machine that has already run the flow, a real uv legitimately
        // lives at that path, and a test that demanded its absence would fail
        // for the researcher and pass only for us.
        let before = try? Data(
            contentsOf: LocalEngineLayout.managedUV, options: .alwaysMapped)
        let provisioner = LocalEngineProvisioner(
            environment: environment, shell: NeverRunShell(),
            downloader: SubstitutedArtifactDownloader())
        await provisioner.runAwaitingCompletion()
        guard case .failed(let reason) = provisioner.phase else {
            Issue.record("a substituted artifact did not fail the flow: \(provisioner.phase)")
            return
        }
        #expect(reason.contains("REFUSED"))
        #expect(reason.contains(PinnedUV.sha256))
        let after = try? Data(
            contentsOf: LocalEngineLayout.managedUV, options: .alwaysMapped)
        #expect(before == after, "the refused download still wrote something")
    }

    /// A blocked step stops the run at that step, with the blocking reason as
    /// the status — never a silent skip onwards.
    @MainActor
    @Test func aBlockedStepStopsTheRunWithItsReason() async {
        var environment = warmEnvironment()
        environment.lockSHA = nil  // no platform lock in the tree
        let provisioner = LocalEngineProvisioner(
            environment: environment, shell: NeverRunShell(),
            downloader: SubstitutedArtifactDownloader())
        await provisioner.runAwaitingCompletion()
        guard case .failed(let reason) = provisioner.phase else {
            Issue.record("a blocked step did not stop the run: \(provisioner.phase)")
            return
        }
        #expect(reason.contains("pyproject floors"))
        #expect(provisioner.progress[.environment] == "blocked")
        // The steps BEFORE it still ran and reported.
        #expect(provisioner.progress[.engineSource] != nil)
    }
}

import Foundation
import Testing

@testable import ExperimentKit

// =============================================================================
// WP2 — the baked-path census, as a test that cannot be bypassed by forgetting.
//
// SteerLab.app now ships as a signed, self-contained bundle, and the home
// layout it lives in has three tiers: app alone, app + a code checkout beside
// it, and a developer running from the checkout itself. Every place in
// `Sources/` that assumes it can reach a code checkout has to declare which
// KIND of assumption it is making, because the three kinds have different
// correct answers:
//
//   A — SHIPPED READ-ONLY BYTES (seed, templates, web assets, payloads-as-
//       source). Must resolve through a typed `CodeResources` accessor, which
//       probes the bundle before the checkout. Correct in all three tiers.
//   B — MUTABLE / EXECUTABLE TREES (the Python venv, scripts that write into
//       their own tree). Must resolve through `CodeResources.executableCheckout()`
//       and refuse with a TYPED layout reason when there is no checkout — a
//       signed bundle must never be written into.
//   C — DEV-ONLY OBSERVATIONS (git stamps, the committed `docs/` tree, the
//       workspace/checkout conflation guards). Must degrade with a stated
//       reason; they are allowed to name the compiled-in checkout path.
//
// The mechanism below: walk `Sources/` for the tokens that can only appear in
// a checkout assumption, and require every (file, token) pair to appear in
// `CheckoutDependency.census` with a declared bucket. A new feature that
// reaches for a checkout therefore cannot compile-and-ship without declaring
// its bucket — the test names the offending file and refuses.
// =============================================================================

/// One declared checkout assumption in production code.
struct CheckoutDependency: Sendable, Equatable {

    enum Bucket: String, Sendable {
        /// Shipped read-only bytes; resolves through a typed family accessor.
        case shippedResource = "A"
        /// A tree this process writes into or executes; needs a real checkout.
        case executableTree = "B"
        /// Developer-only observation; degrades with a stated reason.
        case developerOnly = "C"
    }

    /// Path relative to the repository root.
    let file: String
    /// The token that makes this file a checkout consumer.
    let token: String
    /// What it resolves.
    let resolves: String
    let bucket: Bucket
    /// How it behaves with no checkout available.
    let withoutACheckout: String

    var key: String { "\(file)#\(token)" }
}

extension CheckoutDependency {

    /// Tokens that mark a file as reaching for a code checkout. Deliberately
    /// broad: a false positive costs one census line, a false negative costs a
    /// silently broken packaged build.
    static let tokens = [
        "compiledCheckoutPath",
        "developerCheckoutRoot",
        "developerCheckoutAvailable",
        "releaseModeAsserted",
        "bundledSeedRoot",
        "executableCheckout",
        ".venv.nosync",
        "start-local-server.sh",
        "#filePath",
        // WP3 — the SECOND writable tier. `localEngineSource()` may return a
        // materialized ENGINE ROOT rather than a checkout, and an engine root
        // cannot run dev features (no Package.swift, no scripts/, no git
        // identity). Anything that reaches for one is therefore making a
        // checkout-shaped assumption too, and has to declare its bucket the
        // same way — otherwise the tier would be exactly the silent
        // conflation WP2 exists to prevent.
        "localEngineSource",
        "defaultEngineRoot",
    ]

    /// THE CENSUS. Every (file, token) pair in `Sources/` must be here.
    static let census: [CheckoutDependency] = [

        // ── The locator itself ────────────────────────────────────────────
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "#filePath",
            resolves: "the compiled-in checkout root — the ONE sanctioned derivation",
            bucket: .developerOnly,
            withoutACheckout: "`developerCheckoutRoot` is nil and `mode` becomes .release"),
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "compiledCheckoutPath",
            resolves: "same — the derivation and its existence-checked wrapper",
            bucket: .developerOnly,
            withoutACheckout: "nil; every family then resolves from the bundle only"),
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "developerCheckoutRoot",
            resolves: "the compiled checkout when it still carries Package.swift",
            bucket: .developerOnly,
            withoutACheckout: "nil"),
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "developerCheckoutAvailable",
            resolves: "whether dev fallback is possible at all",
            bucket: .developerOnly,
            withoutACheckout: "false"),
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "executableCheckout",
            resolves: "bucket B's resolver: compiled checkout → app sibling → ~/SteerLab",
            bucket: .executableTree,
            withoutACheckout: "typed `ExecutableCheckoutUnavailable` naming the home layout"),
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: ".venv.nosync",
            resolves: "documentation only — names the venv bucket B protects",
            bucket: .executableTree,
            withoutACheckout: "n/a (a comment)"),
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "start-local-server.sh",
            resolves: "documentation only — the script whose self-writing tree defines bucket B",
            bucket: .executableTree,
            withoutACheckout: "n/a (a comment)"),
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "bundledSeedRoot",
            resolves: "documentation only — names the compatibility alias in VectorCatalog",
            bucket: .developerOnly,
            withoutACheckout: "n/a (a comment)"),
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "releaseModeAsserted",
            resolves: "the override that forces release mode regardless of what is on disk",
            bucket: .developerOnly,
            withoutACheckout: "n/a — it is the switch, not a consumer"),
        .init(
            file: "Sources/SteerLabApp/SteerLabApp.swift",
            token: "releaseModeAsserted",
            resolves:
                "asserts release mode when running from an .app, so a packaged "
                + "build never falls back to the BUILD machine's checkout",
            bucket: .developerOnly,
            withoutACheckout:
                "unset for the SwiftPM dev executable — developer mode, unchanged"),
        .init(
            file: "Sources/steerlab-cli/main.swift",
            token: "releaseModeAsserted",
            resolves:
                "asserts release mode when the CLI runs from inside a bundle "
                + "(Contents/Helpers), mirroring SteerLabApp.init — a bundled "
                + "helper never falls back to the BUILD machine's checkout",
            bucket: .developerOnly,
            withoutACheckout:
                "unset for the SwiftPM dev executable — developer mode, unchanged"),

        // ── Bucket A: shipped read-only bytes ─────────────────────────────
        .init(
            file: "Sources/ExperimentKit/ResourceManifest.swift",
            token: ".venv.nosync",
            resolves: "documentation of the packaging walk's dot-prefixed skips",
            bucket: .shippedResource,
            withoutACheckout: "n/a (a comment); the walk is over a staged tree, not a checkout"),
        .init(
            file: "Sources/ExperimentKit/VectorCatalog.swift",
            token: "bundledSeedRoot",
            resolves: "compatibility alias for compiledCheckoutPath, used by the guards below",
            bucket: .developerOnly,
            withoutACheckout: "still a path; callers are guards that compare, never read"),
        .init(
            file: "Sources/ExperimentKit/VectorCatalog.swift",
            token: "compiledCheckoutPath",
            resolves: "the alias's body",
            bucket: .developerOnly,
            withoutACheckout: "as above"),

        // ── Bucket B: mutable / executable trees ──────────────────────────
        .init(
            file: "Sources/ExperimentKit/LocalPythonTools.swift",
            token: "executableCheckout",
            resolves: "the checkout the local Python engine runs from",
            bucket: .executableTree,
            withoutACheckout: "every accessor nil; `unavailableHint` names the home layout"),
        .init(
            file: "Sources/ExperimentKit/LocalPythonTools.swift",
            token: ".venv.nosync",
            resolves: "<checkout>/Server/.venv.nosync/bin/python — created by the start script",
            bucket: .executableTree,
            withoutACheckout: "nil; never the bundle's ServerPayload/"),
        .init(
            file: "Sources/ExperimentKit/LocalPythonTools.swift",
            token: "start-local-server.sh",
            resolves: "<checkout>/scripts/start-local-server.sh — writes the venv into its own tree",
            bucket: .executableTree,
            withoutACheckout: "nil; the Start button says why"),
        // WP3 — the local-engine tier. `localEngineSource()` is bucket B's
        // SECOND answer: a checkout when there is one, otherwise a
        // materialized engine root, which is writable and executable but is
        // NOT a checkout and must never be used as one.
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "localEngineSource",
            resolves:
                "the local Python engine's tier: checkout first, then the "
                + "materialized engine root at ~/SteerLab/Engine",
            bucket: .executableTree,
            withoutACheckout:
                "nil when no engine root has been materialized either — the "
                + "provisioner turns that into \"materialize\", never an error"),
        .init(
            file: "Sources/ExperimentKit/CodeResources.swift",
            token: "defaultEngineRoot",
            resolves: "~/SteerLab/Engine — created on demand, never by `init`",
            bucket: .executableTree,
            withoutACheckout: "still a path; `isEngineRoot` decides whether anything is there"),
        .init(
            file: "Sources/ExperimentKit/LocalEngineProvisioner.swift",
            token: "localEngineSource",
            resolves: "the tier every provisioning step acts on",
            bucket: .executableTree,
            withoutACheckout:
                "step 1 materializes the bundled ServerPayload; with no payload "
                + "either, the step is blocked with a layout reason"),
        .init(
            file: "Sources/ExperimentKit/LocalEngineProvisioner.swift",
            token: "defaultEngineRoot",
            resolves: "the materialization destination and the tools/ sibling",
            bucket: .executableTree,
            withoutACheckout: "n/a — it is where the flow CREATES the tree"),
        .init(
            file: "Sources/ExperimentKit/LocalEngineProvisioner.swift",
            token: ".venv.nosync",
            resolves:
                "<engine tree>/Server/.venv.nosync — the venv this flow creates "
                + "from the committed platform lock, in a checkout or in the "
                + "materialized engine root, never in the bundle",
            bucket: .executableTree,
            withoutACheckout: "the environment step is blocked with the layout reason"),
        .init(
            file: "Sources/ExperimentKit/LocalEngineProvisioner.swift",
            token: "start-local-server.sh",
            resolves:
                "the CHECKOUT tier's serve invocation — driven unchanged so the "
                + "one-click path stays byte-compatible; the materialized tier "
                + "issues the same argv directly",
            bucket: .executableTree,
            withoutACheckout: "the materialized tier never reaches for it"),
        .init(
            file: "Sources/SteerLabApp/LocalEngineSetupSheet.swift",
            token: ".venv.nosync",
            resolves: "report text naming the venv a step decided about",
            bucket: .executableTree,
            withoutACheckout: "the step renders its blocked reason instead"),

        .init(
            file: "Sources/ExperimentKit/GemmaScopeAnalysis.swift",
            token: "executableCheckout",
            resolves: "the checkout holding the gemmascope-py312 venv",
            bucket: .executableTree,
            withoutACheckout: "throws .engineUnavailable carrying the layout sentence"),
        .init(
            file: "Sources/ExperimentKit/GemmaScopeAnalysis.swift",
            token: ".venv.nosync",
            resolves: "<checkout>/.venv.nosync/gemmascope-py312/bin/python",
            bucket: .executableTree,
            withoutACheckout: "as above (a checkout WITHOUT the venv falls back to python3)"),
        .init(
            file: "Sources/SteerLabApp/ClusterConnectionDot.swift",
            token: ".venv.nosync",
            resolves: "help text naming what the Start button creates",
            bucket: .executableTree,
            withoutACheckout: "the status line carries `unavailableHint` instead"),
        .init(
            file: "Sources/SteerLabApp/SteerLabApp.swift",
            token: "executableCheckout",
            resolves:
                "the startup self-check's bucket-B line, written to stderr — "
                + "build-app.sh's launch verification prints it, which is how "
                + "the PACKAGED tier proves its own resolution",
            bucket: .executableTree,
            withoutACheckout: "prints \"local engine: unavailable — <layout reason>\""),
        .init(
            file: "Sources/SteerLabApp/ClusterConnectionDot.swift",
            token: "start-local-server.sh",
            resolves: "help text naming the script the Start button runs",
            bucket: .executableTree,
            withoutACheckout: "as above — the button's status line says why it cannot run"),

        // ── Bucket C: developer-only observations ─────────────────────────
        .init(
            file: "Sources/ExperimentKit/SteerLabVersion.swift",
            token: "developerCheckoutRoot",
            resolves: "git rev-parse HEAD of the code checkout, for the engine stamp",
            bucket: .developerOnly,
            withoutACheckout:
                "the packaged manifest wins; then Info.plist SLSourceRevision "
                + "(labelled the APP's build revision); then a bare version"),
        .init(
            file: "Sources/ExperimentKit/UpdateCheck.swift",
            token: "executableCheckout",
            resolves:
                "READ-ONLY: where to run `git ls-remote` so the update signpost can "
                + "see the repository's tags. It uses bucket B's resolver (an "
                + "app-only install with a checkout beside it must still be able to "
                + "ask) but never writes into the tree, so the ASSUMPTION is a "
                + "developer-style observation, not a mutable-tree claim",
            bucket: .developerOnly,
            withoutACheckout:
                "the git half is skipped entirely; the check falls back to the "
                + "public release feed, and reports \"unknown\" with a reason when "
                + "that is unavailable too — never \"up to date\""),
        .init(
            file: "Sources/ExperimentKit/InstallProvenance.swift",
            token: "developerCheckoutRoot",
            resolves: "reporting only — `install version` names the checkout when there is one",
            bucket: .developerOnly,
            withoutACheckout: "the report says \"release mode — bundled resources only\""),
        .init(
            file: "Sources/ExperimentKit/CLIReferenceDocument.swift",
            token: "compiledCheckoutPath",
            resolves: "<checkout>/docs/CLI-REFERENCE.md, regenerated by a dev verb",
            bucket: .developerOnly,
            withoutACheckout: "nil — an installed CLI ships no docs/, and says so"),
        .init(
            file: "Sources/ExperimentKit/HomeLayout.swift",
            token: "compiledCheckoutPath",
            resolves: "the running binary's own checkout, reported by `init` when it is outside the home",
            bucket: .developerOnly,
            withoutACheckout: "nil — `externalCheckout` is simply absent from the plan"),
        .init(
            file: "Sources/ExperimentKit/WorkspaceStore.swift",
            token: "bundledSeedRoot",
            resolves: "the legacy workspace-root fallback + the isLegacyRepoRoot guard",
            bucket: .developerOnly,
            withoutACheckout:
                "the fallback path simply does not exist, so no workspace resolves to it"),
        .init(
            file: "Sources/ExperimentKit/WorkspaceStore.swift",
            token: "compiledCheckoutPath",
            resolves: "doc comment naming the alias",
            bucket: .developerOnly,
            withoutACheckout: "n/a (a comment)"),
        .init(
            file: "Sources/ExperimentKit/ExperimentStore.swift",
            token: "bundledSeedRoot",
            resolves: "freeze auto-commit refuses when the WORKSPACE is the code checkout",
            bucket: .developerOnly,
            withoutACheckout: "the comparison cannot match, so auto-commit is decided by the other gates"),
    ]

    static var byKey: [String: CheckoutDependency] {
        Dictionary(uniqueKeysWithValues: census.map { ($0.key, $0) })
    }
}

@Suite(.serialized) struct CheckoutDependencyTests {

    // MARK: Repo walking

    private static var repoRoot: URL {
        URL(filePath: #filePath)  // …/Tests/ExperimentKitTests/…
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .standardizedFileURL
    }

    /// Every (file, token) pair actually present in `Sources/`.
    private static func observedDependencies() throws -> Set<String> {
        let sources = repoRoot.appending(component: "Sources")
        let fm = FileManager.default
        let enumerator = try #require(
            fm.enumerator(at: sources, includingPropertiesForKeys: nil))
        var observed: Set<String> = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            guard let contents = try? String(contentsOf: url, encoding: .utf8)
            else { continue }
            let relative = String(
                url.standardizedFileURL.path
                    .dropFirst(repoRoot.path.count + 1))
            for token in CheckoutDependency.tokens where contents.contains(token) {
                observed.insert("\(relative)#\(token)")
            }
        }
        return observed
    }

    // MARK: 1 — the coherence test

    /// The property that has to hold: an unclassified checkout consumer fails
    /// a test. Both directions — a new consumer is a failure until it is
    /// declared, and a census entry for a consumer that no longer exists is a
    /// failure too (so the table cannot rot into fiction).
    @Test func everyCheckoutConsumerIsClassified() throws {
        let observed = try Self.observedDependencies()
        let declared = Set(CheckoutDependency.census.map(\.key))

        let undeclared = observed.subtracting(declared).sorted()
        #expect(
            undeclared.isEmpty,
            """
            Unclassified checkout assumption(s): \(undeclared.joined(separator: ", ")).
            SteerLab.app ships as a signed bundle that may have no code checkout \
            beside it, so every checkout-shaped path has to declare its bucket: \
            A (shipped read-only bytes — resolve through a typed CodeResources \
            family accessor), B (a tree this process writes into or executes — \
            resolve through CodeResources.executableCheckout() and refuse with a \
            typed layout reason when absent), or C (a developer-only observation \
            that degrades with a stated reason). Add an entry to \
            CheckoutDependency.census in this file.
            """)

        let stale = declared.subtracting(observed).sorted()
        #expect(
            stale.isEmpty,
            """
            Census entr(ies) for consumers that no longer exist: \
            \(stale.joined(separator: ", ")). Remove them — a census that \
            describes code that is gone stops being evidence.
            """)
    }

    /// Bucket B is the one with a hard prohibition attached, so it gets a
    /// second, sharper assertion: no bucket-B file may reach a checkout by any
    /// route EXCEPT the one resolver. In particular none of them may call
    /// `developerCheckoutAvailable`/`developerCheckoutRoot` directly (that was
    /// the pre-WP2 shape, which found only the COMPILED checkout and so
    /// reported "unavailable" for a checkout sitting right beside the app).
    @Test func bucketBFilesResolveOnlyThroughTheOneResolver() throws {
        let bucketBFiles = Set(
            CheckoutDependency.census
                .filter { $0.bucket == .executableTree }
                .map(\.file)
        ).subtracting(["Sources/ExperimentKit/CodeResources.swift"])

        for file in bucketBFiles.sorted() {
            let contents = try String(
                contentsOf: Self.repoRoot.appending(path: file), encoding: .utf8)
            for forbidden in [
                "developerCheckoutRoot", "developerCheckoutAvailable",
                "compiledCheckoutPath", "bundledSeedRoot",
            ] {
                #expect(
                    !contents.contains(forbidden),
                    """
                    \(file) is a bucket-B (writable/executable) consumer but \
                    reaches a checkout via `\(forbidden)`. Bucket B must go \
                    through CodeResources.executableCheckout(), which also \
                    finds a checkout beside a packaged app; the compiled-in \
                    path only ever names the machine the binary was BUILT on.
                    """)
            }
        }
    }

    // MARK: 2 — the bucket-B resolver's rules

    /// A home layout with an optional checkout in it. `Package.swift` +
    /// `Server/steerlab_server/` is the content rule (`HomeLayout.isCheckout`),
    /// so the name of the folder is deliberately NOT "SteerLab"-shaped.
    private func makeHome(
        checkoutNamed name: String?, extraDecoys: [String] = []
    ) throws -> URL {
        let fm = FileManager.default
        let home = fm.temporaryDirectory
            .appending(component: "steerlab-home-\(UUID().uuidString)")
        for layout in HomeLayout.directoryNames {
            try fm.createDirectory(
                at: home.appending(component: layout),
                withIntermediateDirectories: true)
        }
        for decoy in extraDecoys {
            // A folder with SOME of the markers is not a checkout.
            let root = home.appending(component: decoy)
            try fm.createDirectory(at: root, withIntermediateDirectories: true)
            try "// swift-tools-version: 6.2\n".write(
                to: root.appending(component: "Package.swift"),
                atomically: true, encoding: .utf8)
        }
        if let name {
            let root = home.appending(component: name)
            try fm.createDirectory(
                at: root.appending(path: "Server/steerlab_server"),
                withIntermediateDirectories: true)
            try "// swift-tools-version: 6.2\n".write(
                to: root.appending(component: "Package.swift"),
                atomically: true, encoding: .utf8)
        }
        return home
    }

    /// Runs `body` with the compiled checkout suppressed (release mode) and
    /// `homes` standing in for the app's home candidates.
    private func withHomes(_ homes: [URL], _ body: () throws -> Void) rethrows {
        ExperimentRootOverrideLock.acquire()
        CodeResources.modeOverrideForTesting = .release
        CodeResources.executableHomesOverrideForTesting = homes
        defer {
            CodeResources.modeOverrideForTesting = nil
            CodeResources.executableHomesOverrideForTesting = nil
            ExperimentRootOverrideLock.release()
        }
        try body()
    }

    @Test func developerCheckoutIsHonoredFirst() throws {
        // No overrides: this test process runs from the checkout, so bucket B
        // resolves exactly where it did before WP2 — the dev workflow is
        // byte-identical, which is the whole constraint.
        let expected = try #require(CodeResources.developerCheckoutRoot)
        let resolved = try CodeResources.executableCheckout().get()
        #expect(resolved.root == expected)
        #expect(resolved.origin == .compiledCheckout)
        #expect(LocalPythonRuntime.repoRoot == expected)
        #expect(
            LocalPythonRuntime.venvPython?.path
                == expected.appending(path: "Server/.venv.nosync/bin/python").path)
        #expect(
            LocalPythonRuntime.startServerScript?.path
                == expected.appending(path: "scripts/start-local-server.sh").path)
    }

    @Test func siblingCheckoutIsFoundByContentNotByName() throws {
        let fm = FileManager.default
        // Named nothing like the release repo, and sitting beside a decoy that
        // carries only ONE of the two markers.
        let home = try makeHome(checkoutNamed: "zz-renamed-clone", extraDecoys: ["aaa-not-a-checkout"])
        defer { try? fm.removeItem(at: home) }

        try withHomes([home]) {
            let resolved = try CodeResources.executableCheckout().get()
            #expect(resolved.root == home.appending(component: "zz-renamed-clone")
                .standardizedFileURL)
            #expect(resolved.origin == .homeFolder(home))
            // …and the bucket-B consumers land inside it, never in a bundle.
            #expect(
                LocalPythonRuntime.venvPython?.path
                    == resolved.root.appending(path: "Server/.venv.nosync/bin/python").path)
            #expect(
                LocalPythonRuntime.startServerScript?.path
                    == resolved.root.appending(path: "scripts/start-local-server.sh").path)
            // The decoy (Package.swift but no Server/steerlab_server) is not it,
            // even though it sorts first.
            #expect(resolved.root.lastPathComponent != "aaa-not-a-checkout")
        }
    }

    @Test func homesAreSearchedInOrder() throws {
        let fm = FileManager.default
        let first = try makeHome(checkoutNamed: nil)
        let second = try makeHome(checkoutNamed: "code")
        defer {
            try? fm.removeItem(at: first)
            try? fm.removeItem(at: second)
        }
        try withHomes([first, second]) {
            let resolved = try CodeResources.executableCheckout().get()
            #expect(resolved.root == second.appending(component: "code").standardizedFileURL)
            #expect(resolved.origin == .homeFolder(second))
        }
    }

    @Test func absenceIsATypedLayoutReasonNotAMissingFile() throws {
        let fm = FileManager.default
        let home = try makeHome(checkoutNamed: nil)
        defer { try? fm.removeItem(at: home) }

        try withHomes([home]) {
            switch CodeResources.executableCheckout() {
            case .success(let found):
                Issue.record("resolved a checkout that does not exist: \(found.root.path)")
            case .failure(let absence):
                #expect(absence.searched == [home])
                #expect(absence.message.contains("no SteerLab code checkout"))
                #expect(absence.message.contains(HomeLayout.defaultHome.path))
                #expect(absence.message.contains("beside SteerLab.app"))
                // Never a file-not-found: no venv/script path appears.
                #expect(!absence.message.contains(".venv.nosync"))
                #expect(!absence.message.contains("start-local-server.sh"))
            }

            // The consumers degrade to nil + the layout sentence.
            #expect(LocalPythonRuntime.repoRoot == nil)
            #expect(LocalPythonRuntime.venvPython == nil)
            #expect(LocalPythonRuntime.startServerScript == nil)
            #expect(!LocalPythonRuntime.venvExists)
            let hint = LocalPythonRuntime.unavailableHint
            #expect(hint.contains("local Python engine needs the code checkout"))
            #expect(hint.contains(HomeLayout.defaultHome.path))
        }
    }

    /// The bundle must never become the venv's home: with a staged bundle
    /// that DOES carry a ServerPayload and no checkout anywhere, the Python
    /// paths stay nil rather than pointing into read-only bytes.
    @Test func bundledServerPayloadIsNeverTheVenvHome() throws {
        let fm = FileManager.default
        let home = try makeHome(checkoutNamed: nil)
        let staged = fm.temporaryDirectory
            .appending(component: "steerlab-payload-\(UUID().uuidString)")
        try fm.createDirectory(
            at: staged.appending(path: "ServerPayload/steerlab_server"),
            withIntermediateDirectories: true)
        defer {
            try? fm.removeItem(at: home)
            try? fm.removeItem(at: staged)
        }
        ExperimentRootOverrideLock.acquire()
        CodeResources.bundleOverrideForTesting = staged
        CodeResources.modeOverrideForTesting = .release
        CodeResources.executableHomesOverrideForTesting = [home]
        defer {
            CodeResources.bundleOverrideForTesting = nil
            CodeResources.modeOverrideForTesting = nil
            CodeResources.executableHomesOverrideForTesting = nil
            ExperimentRootOverrideLock.release()
        }
        // The family resolves (it is shipped, bucket A)…
        #expect(try CodeResources.serverPayload().path
            == staged.appending(component: "ServerPayload").standardizedFileURL.path)
        // …and the venv still refuses, because writing there breaks the seal.
        #expect(LocalPythonRuntime.venvPython == nil)
        #expect(LocalPythonRuntime.repoRoot == nil)
    }

    // MARK: 3 — bucket A resolves in a packaged layout, family by family

    /// Per-family resolution against a synthetic bundle laid out exactly like
    /// `scripts/build-app.sh` writes `Contents/Resources/`, with the packaging
    /// manifest generated the same way. `Bundle.main` cannot be faked in a
    /// unit test, which is why `bundleCandidates` is an injectable list — and
    /// why the app additionally runs `CodeResources.selfCheck()` for real at
    /// launch (build-app.sh's launch verification prints it).
    private func stagePackagedResources() throws -> URL {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appending(component: "steerlab-packaged-\(UUID().uuidString)")
        func stage(_ relative: String, _ contents: String) throws {
            let url = root.appending(path: relative)
            try fm.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try contents.write(to: url, atomically: true, encoding: .utf8)
        }
        try stage("WorkspaceSeed/prompts/batteries/basic.jsonl", "{}\n")
        try stage("ServerPayload/pyproject.toml", "[project]\n")
        try stage("ClusterPayload/Server/scripts/bootstrap.sh", "#!/bin/sh\n")
        try stage("AnalysisTools/gemmascope_analyze.py", "# stub\n")
        try stage("web/index.html", "<!doctype html>\n")
        let manifest = try ResourceManifest.generate(
            over: root, serverVersion: "0.0-test", protocolVersion: 1)
        try manifest.write(to: root.appending(component: "resource-manifest.json"))
        return root
    }

    @Test func everyFamilyResolvesInAPackagedLayout() throws {
        let fm = FileManager.default
        let staged = try stagePackagedResources()
        let home = try makeHome(checkoutNamed: nil)
        defer {
            try? fm.removeItem(at: staged)
            try? fm.removeItem(at: home)
        }
        ExperimentRootOverrideLock.acquire()
        CodeResources.bundleOverrideForTesting = staged
        CodeResources.modeOverrideForTesting = .release
        CodeResources.executableHomesOverrideForTesting = [home]
        defer {
            CodeResources.bundleOverrideForTesting = nil
            CodeResources.modeOverrideForTesting = nil
            CodeResources.executableHomesOverrideForTesting = nil
            ExperimentRootOverrideLock.release()
        }

        let check = CodeResources.selfCheck()
        #expect(check.mode == .release)
        let unresolved = check.problems
            .map { "\($0.family.rawValue): \($0.problem ?? "")" }
            .joined(separator: "; ")
        #expect(
            check.problems.isEmpty,
            "families unresolved in a packaged layout: \(unresolved)")
        #expect(check.resolved.count == CodeResources.Family.allCases.count)

        // Resolution is verified AGAINST the packaging manifest: every
        // directory family must own at least one manifest entry, which is what
        // catches "packaging forgot to stage this family" as opposed to
        // "some path happens to exist".
        for row in check.rows where row.family != .buildManifest {
            let entries = try #require(
                row.manifestEntries,
                "\(row.family.rawValue) had no manifest coverage count")
            #expect(entries > 0, "\(row.family.rawValue) is absent from the manifest")
            let path = try #require(row.resolvedPath)
            #expect(path.hasPrefix(staged.standardizedFileURL.path + "/"))
        }
        // Bucket B is honestly absent in this layout, and the report says so
        // in the language the UI shows.
        #expect(check.executableCheckout == nil)
        #expect(
            try #require(check.executableCheckoutProblem)
                .contains("no SteerLab code checkout"))
        #expect(check.summaryLine.contains("no problems"))
    }

    /// A family packaging forgot becomes a NAMED problem, not a silent
    /// fallback to the developer tree.
    @Test func aFamilyMissingFromPackagingIsNamed() throws {
        let fm = FileManager.default
        let staged = try stagePackagedResources()
        let home = try makeHome(checkoutNamed: nil)
        defer {
            try? fm.removeItem(at: staged)
            try? fm.removeItem(at: home)
        }
        try fm.removeItem(at: staged.appending(component: "AnalysisTools"))

        ExperimentRootOverrideLock.acquire()
        CodeResources.bundleOverrideForTesting = staged
        CodeResources.modeOverrideForTesting = .release
        CodeResources.executableHomesOverrideForTesting = [home]
        defer {
            CodeResources.bundleOverrideForTesting = nil
            CodeResources.modeOverrideForTesting = nil
            CodeResources.executableHomesOverrideForTesting = nil
            ExperimentRootOverrideLock.release()
        }

        let check = CodeResources.selfCheck()
        let problem = try #require(
            check.problems.first(where: { $0.family == .analysisTools }))
        #expect(try #require(problem.problem).contains("reinstall the app"))
        #expect(check.summaryLine.contains("PROBLEMS"))
        #expect(check.summaryLine.contains("analysis tools"))
        // Per-family, not all-or-nothing: the intact families still resolve.
        #expect(check.resolved.contains { $0.family == .webAssets })
    }

    /// Developer mode: the self-check is quiet, the missing packaging manifest
    /// is honest absence rather than a problem, and bucket B is the checkout.
    @Test func selfCheckIsQuietInADeveloperCheckout() throws {
        let check = CodeResources.selfCheck()
        #expect(check.mode == .developer)
        let manifestRow = try #require(
            check.rows.first(where: { $0.family == .buildManifest }))
        #expect(manifestRow.problem == nil)
        #expect(manifestRow.resolvedPath == nil)
        #expect(try #require(check.executableCheckout).origin == .compiledCheckout)
        // `web/` is a committed BUILD OUTPUT (results-explorer → npm run
        // build:embed), so a cold clone legitimately lacks it; every other
        // family is checkout content and must resolve.
        let expectedProblems: Set<CodeResources.Family> =
            ResearchTreeFixtures.hasBuiltWebAssets ? [] : [.webAssets]
        #expect(Set(check.problems.map(\.family)) == expectedProblems)
    }
}

import Foundation

// MARK: - CodeResources — the one authority for immutable program resources
//
// Phase A of docs/MAC-DISTRIBUTION-AND-MANAGED-SERVER-PROPOSAL.md (§3–§4,
// §12). Everything the app ships as CODE — workspace seed data, the Python
// server payload, cluster bootstrap/Slurm templates, analysis scripts, web
// assets, and the packaging resource manifest — resolves through this type
// and nothing else. Wave 2 migrated the non-cluster call sites (workspace
// seeding, analysis tools, the local Python engine, web assets, version
// stamping); wave 3 migrated the last one — the cluster payload
// (`ClusterProvisioner.defaultLocalRepoPath` resolves through
// `clusterPayload()`), so the ReleaseModeResourceTests regression-guard
// allowlist is now this file alone.
// `VectorCatalog.bundledSeedRoot` is now a compatibility alias for
// `compiledCheckoutPath` below.
//
// WP2 (2026-08-20) added the SECOND question this type answers. The families
// above are BUCKET A — shipped read-only bytes, which a signed app bundle is
// the ideal home for. `executableCheckout()`, far below, is BUCKET B: the one
// code tree this process may WRITE INTO or EXECUTE FROM, which a signed
// bundle can never be (the first write breaks the seal). Every consumer of a
// checkout-shaped path is classified into one of those two, or into BUCKET C
// (dev-only observations that degrade with a stated reason — git stamps, the
// committed `docs/`, the workspace/checkout conflation guards); the census is
// enforced by `CheckoutDependencyTests`, which fails on any unclassified
// consumer.
//
// Resolution order, per the proposal's §4.2:
//
//   1. test override (`bundleOverrideForTesting`) — a staged directory laid
//      out like the app bundle's Resources/ stands in for the bundle, so the
//      release path is testable before Phase B produces a real .app;
//   2. the release bundle — `Bundle.main.resourceURL` today; Phase B's app
//      target extends `bundleCandidates` (ExperimentKit declares no SwiftPM
//      resources, so `Bundle.module` does not exist yet);
//   3. the development checkout, derived from this file's `#filePath` —
//      available ONLY in developer mode.
//
// WHY MODE IS A RUNTIME SIGNAL, NOT `#if DEBUG`: an Xcode *Release* build of
// the SwiftPM executables is still a developer running from a checkout — the
// optimization level says nothing about distribution. Until Phase B ships a
// real app target, the honest signal is: does this build's compiled-in
// checkout root still exist on disk AND carry the `Package.swift` marker?
// If yes, developer mode; if not (the binary was copied away, the repo
// deleted — exactly what release simulates), release mode, which FAILS
// CLOSED: a missing family throws a plain-language error naming the family
// and the remedy, and never silently rediscovers a checkout. Phase B gets
// `releaseModeAsserted` to force release mode explicitly from the app
// target regardless of what exists on the build machine.
public enum CodeResources {

    /// One immutable resource family the program ships. Raw value = the
    /// directory (or file) name inside the app bundle's Resources/, matching
    /// the proposal §3.1 layout.
    public enum Family: String, CaseIterable, Sendable {
        /// Generic batteries, rubrics, templates, the neutral corpus — the
        /// curated seed tree `WorkspaceStore.create` copies into a fresh
        /// workspace, file by file, through `WorkspaceStore.seedManifest`.
        /// `WorkspaceSeed/` in BOTH modes since WP1: the dev fallback used
        /// to be the checkout root itself, which made the research tree's
        /// own `prompts/` the seed source and leaked study material into
        /// every new workspace. The resolved URL is still "a root that
        /// contains `prompts/…`" in both modes.
        case workspaceSeed = "WorkspaceSeed"
        /// The versioned Python server source (`Server/` in the checkout).
        case serverPayload = "ServerPayload"
        /// The cluster deployment payload — the ROOT-LAYOUT tree the
        /// provisioner rsyncs to a cluster: a filtered `Server/` (which
        /// contains `Server/scripts/` — bootstrap.sh,
        /// submit-bootstrap-job.sh, controller-job.sbatch.template, …) plus
        /// the `prompts/fixtures/` parity fixtures, matching
        /// `ClusterProvisioner.pushFilterArguments`. Dev fallback is the
        /// checkout root itself (the rsync source root today); a bundle's
        /// `ClusterPayload/` mirrors that root layout and additionally
        /// carries `deployment-manifest.json` (proposal §10). Wave 1 guessed
        /// this family was just the `Server/scripts/` template directory —
        /// wave 3 reconciled it with what the push actually ships.
        case clusterPayload = "ClusterPayload"
        /// App-owned analysis scripts (`scripts/` in the checkout, e.g.
        /// `gemmascope_analyze.py`).
        case analysisTools = "AnalysisTools"
        /// The Swift web server's browser client (`web/` in both layouts).
        case webAssets = "web"
        /// The packaging resource manifest (`resource-manifest.json`),
        /// generated at packaging time (Phase B). A file, not a directory;
        /// honestly absent in developer mode — see `buildManifest()`.
        case buildManifest = "resource-manifest.json"

        /// Plain-language name used in error messages.
        public var displayName: String {
            switch self {
            case .workspaceSeed: return "workspace seed data"
            case .serverPayload: return "Python server payload"
            case .clusterPayload: return "cluster deployment payload"
            case .analysisTools: return "analysis tools"
            case .webAssets: return "web assets"
            case .buildManifest: return "resource manifest"
            }
        }

        /// Where the family lives relative to the DEVELOPMENT checkout root;
        /// nil when the family simply does not exist in a checkout (the
        /// build manifest is minted by packaging, not by development).
        var developmentSubpath: String? {
            switch self {
            case .workspaceSeed: return "WorkspaceSeed"
            case .serverPayload: return "Server"
            case .clusterPayload: return "."
            case .analysisTools: return "scripts"
            case .webAssets: return "web"
            case .buildManifest: return nil
            }
        }
    }

    /// How this process resolves resources. Decided in exactly one place
    /// (`mode`) from a runtime signal — see the file header for why this is
    /// not `#if DEBUG`.
    public enum Mode: String, Sendable {
        /// Running from a live source checkout: bundle lookups still win
        /// when they hit, but missing families fall back to the checkout.
        case developer
        /// No checkout available (or Phase B asserted release): bundled
        /// resources are the only source, and a missing family fails closed.
        case release
    }

    // MARK: Test / Phase B seams

    /// Resolution step 1: a staged directory that stands in for the app
    /// bundle's Resources/ root. Test-only. `nonisolated(unsafe)` is
    /// justified the same way as `ExperimentStore.rootOverride`: mutated
    /// only by tests inside a serialized override window, never raced
    /// against readers by construction.
    nonisolated(unsafe) public static var bundleOverrideForTesting: URL?

    /// Forces `mode`, so tests can exercise release-mode fail-closed
    /// behavior on a machine where the checkout exists. Test-only; same
    /// safety justification as `bundleOverrideForTesting`.
    nonisolated(unsafe) public static var modeOverrideForTesting: Mode?

    /// RESERVED FOR PHASE B: the real app target sets this true at startup
    /// so a distributed build is release-mode even on a machine that
    /// happens to carry a checkout at the compiled-in path. Written once
    /// before any resolution; never toggled mid-process.
    nonisolated(unsafe) public static var releaseModeAsserted = false

    // MARK: Mode decision (the single site)

    /// The compiled-in checkout PATH, with no existence check. This
    /// `#filePath` derivation is THE sanctioned dev-fallback site
    /// (regression-guarded); `VectorCatalog.bundledSeedRoot` aliases it for
    /// compatibility. The workspace/checkout conflation guards
    /// (`WorkspaceStore.isLegacyRepoRoot`,
    /// `ExperimentStore.freezeAutoCommitIsEnabled`) compare the WORKSPACE
    /// root against this path — a workspace that IS the checkout must
    /// refuse freeze auto-commit even when the checkout has lost its
    /// `Package.swift` marker, so this deliberately does NOT gate on
    /// existence; `developerCheckoutRoot` layers that check on top.
    public static var compiledCheckoutPath: URL {
        // …/Sources/ExperimentKit/CodeResources.swift → repo root.
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// The compiled-in checkout root when it still exists on disk and
    /// carries the `Package.swift` marker; nil otherwise.
    public static var developerCheckoutRoot: URL? {
        let root = compiledCheckoutPath
        let marker = root.appending(component: "Package.swift")
        guard FileManager.default.fileExists(atPath: marker.path) else { return nil }
        return root.standardizedFileURL
    }

    /// The one place resolution mode is decided.
    public static var mode: Mode {
        if let forced = modeOverrideForTesting { return forced }
        if releaseModeAsserted { return .release }
        return developerCheckoutRoot != nil ? .developer : .release
    }

    /// True only when dev-fallback resolution is actually possible — the
    /// gate callers use for honest refusals of checkout-dependent features
    /// (the editable local Python engine stays developer-only until Phase C).
    public static var developerCheckoutAvailable: Bool {
        mode == .developer && developerCheckoutRoot != nil
    }

    // MARK: Bundle lookup

    /// The directory the running executable ACTUALLY lives in, symlinks
    /// resolved.
    ///
    /// `Bundle.main` is derived from the path the process was launched WITH,
    /// not from the file it ended up executing, so a symlink on `PATH` — the
    /// documented way to reach the CLI inside the app bundle — makes
    /// `Bundle.main.bundleURL` the SYMLINK'S directory. Measured on this
    /// machine, not assumed: `~/.local/bin/steerlab-cli ->
    /// SteerLab.app/Contents/Helpers/steerlab-cli` reports `~/.local/bin`.
    /// (MLX's own colocated-shader lookup is unaffected — it asks `dladdr`,
    /// which reports the resolved path — which is why the symlink is a
    /// supported shape at all.)
    ///
    /// Every layout question below is therefore asked of this, with
    /// `Bundle.main` kept as the fallback for a process that has no
    /// executable URL at all.
    static var executableDirectory: URL? {
        guard let executable = Bundle.main.executableURL?.resolvingSymlinksInPath()
        else { return nil }
        return executable.deletingLastPathComponent().standardizedFileURL
    }

    /// The `.app` this process is running out of, when there is one — whether
    /// it is the app's own executable or a HELPER nested beside it.
    ///
    /// CFBundle recognizes exactly one shape: an executable in
    /// `Contents/MacOS/` makes the enclosing `.app` the main bundle. A helper
    /// tool at `SteerLab.app/Contents/Helpers/steerlab-cli` matches nothing,
    /// so `Bundle.main` becomes THE HELPER'S OWN DIRECTORY and
    /// `Bundle.main.resourceURL` is that same directory — the app's real
    /// `Contents/Resources` is not reachable from it at all. Measured, not
    /// assumed: one binary run from `Contents/MacOS/` reports the `.app`; the
    /// same binary run from `Contents/Helpers/` reports `Contents/Helpers`.
    ///
    /// So the enclosing bundle is derived from the layout instead, and the two
    /// consumers below (`bundleCandidates`, `executableHomeCandidates`) ask
    /// this rather than testing `pathExtension == "app"` themselves.
    public static var enclosingAppBundle: URL? {
        var probes: [URL] = []
        if let directory = executableDirectory { probes.append(directory) }
        probes.append(Bundle.main.bundleURL.standardizedFileURL)
        for probe in probes {
            // The app's own executable: `Bundle.main.bundleURL` IS the .app.
            if probe.pathExtension == "app" { return probe }
            // A helper: …/SteerLab.app/Contents/<helper dir> → …/SteerLab.app
            let contents = probe.deletingLastPathComponent()
            guard contents.lastPathComponent == "Contents" else { continue }
            let app = contents.deletingLastPathComponent()
            if app.pathExtension == "app" { return app.standardizedFileURL }
        }
        return nil
    }

    /// Resource roots a release build may carry, probed in order: the main
    /// bundle, then the real executable's own directory (which differs only
    /// when the binary was reached through a symlink), then the enclosing
    /// `.app`'s `Contents/Resources` when this process is a bundled helper. A
    /// future SwiftPM `resources:` declaration providing `Bundle.module`
    /// extends this again.
    static var bundleCandidates: [URL] {
        var candidates: [URL] = []
        if let override = bundleOverrideForTesting {
            // The override REPLACES the real bundle (it stands in for it) —
            // a staged bundle missing a family must behave exactly like a
            // damaged install, not fall through to this build's real bundle.
            candidates.append(override)
            return candidates
        }
        func offer(_ url: URL?) {
            guard let url = url?.standardizedFileURL, !candidates.contains(url)
            else { return }
            candidates.append(url)
        }
        offer(Bundle.main.resourceURL)
        // A symlinked install: `resourceURL` above is the symlink's directory,
        // so this is the one that finds anything colocated with the binary —
        // an install-cli.sh tree's `resource-manifest.json` included.
        offer(executableDirectory)
        // The bundled `steerlab-cli` (Contents/Helpers/) — see
        // `enclosingAppBundle`. LAST, so anything genuinely colocated with the
        // helper still wins.
        offer(enclosingAppBundle?.appending(path: "Contents/Resources"))
        return candidates
    }

    // MARK: Resolution

    /// Resolve a directory family. Throws `CodeResourceError` (plain
    /// language, names the family and the remedy) when the family cannot be
    /// found under the active mode — never silently falls through.
    public static func url(for family: Family) throws -> URL {
        let fm = FileManager.default
        for candidate in bundleCandidates {
            let resolved = candidate
                .appending(path: family.rawValue).standardizedFileURL
            if fm.fileExists(atPath: resolved.path) { return resolved }
        }
        if mode == .developer,
            let root = developerCheckoutRoot,
            let subpath = family.developmentSubpath
        {
            let resolved =
                subpath == "."
                ? root : root.appending(path: subpath).standardizedFileURL
            if fm.fileExists(atPath: resolved.path) { return resolved }
            throw CodeResourceError(
                family: family,
                reason:
                    "the developer checkout at \(root.path) is missing \(subpath)/ "
                    + "(\(family.displayName)) — the checkout is incomplete; "
                    + "restore it or reinstall the app")
        }
        throw CodeResourceError(
            family: family,
            reason:
                "this build is missing its bundled \(family.displayName) "
                + "(\(family.rawValue)) — reinstall the app")
    }

    // MARK: Typed accessors

    /// The seed root workspace creation copies from — a root containing
    /// `prompts/…` in both modes (`<checkout>/WorkspaceSeed/` in dev,
    /// `WorkspaceSeed/` in a bundle). `WorkspaceStore.create` resolves
    /// through this and copies only `seedManifest` files; the
    /// end-to-end proof is `ReleaseModeResourceTests.createWorkspaceFromStagedBundle`.
    public static func workspaceSeed() throws -> URL { try url(for: .workspaceSeed) }

    /// The Python server payload directory (`Server/` in dev).
    public static func serverPayload() throws -> URL { try url(for: .serverPayload) }

    /// The root-layout cluster deployment payload (`Server/` +
    /// `prompts/fixtures/`): the checkout root in dev, `ClusterPayload/` in
    /// a bundle. Bootstrap/Slurm templates live at
    /// `<clusterPayload>/Server/scripts/` in both layouts.
    public static func clusterPayload() throws -> URL { try url(for: .clusterPayload) }

    /// App-owned analysis scripts directory (`scripts/` in dev).
    public static func analysisTools() throws -> URL { try url(for: .analysisTools) }

    /// Browser-client assets directory (`web/` in both layouts).
    public static func webAssets() throws -> URL { try url(for: .webAssets) }

    /// The packaging resource manifest. Release mode: required — missing is
    /// the same fail-closed error as any family. Developer mode: honestly
    /// absent (nil) — a checkout has no packaging manifest and synthesizing
    /// a fake one would defeat integrity checking. A staged/real bundle that
    /// carries one wins in either mode.
    public static func buildManifest() throws -> URL? {
        let fm = FileManager.default
        for candidate in bundleCandidates {
            let resolved = candidate
                .appending(path: Family.buildManifest.rawValue).standardizedFileURL
            if fm.fileExists(atPath: resolved.path) { return resolved }
        }
        if mode == .developer { return nil }
        throw CodeResourceError(
            family: .buildManifest,
            reason:
                "this build is missing its bundled resource manifest "
                + "(resource-manifest.json) — reinstall the app")
    }

    // MARK: - Bucket B: the EXECUTABLE checkout
    //
    // WP2. Everything above resolves SHIPPED READ-ONLY BYTES, and a signed
    // app bundle is a perfectly good home for those. This section answers a
    // different question, and conflating the two is what WP2 exists to stop:
    //
    //   *where may this process WRITE INTO, or EXECUTE, a code tree?*
    //
    // Never the bundle. The first write into a signed bundle breaks its
    // seal, and `scripts/start-local-server.sh` — the one-click local Python
    // engine — creates `<root>/Server/.venv.nosync` and `pip install -e`s
    // into it, i.e. it writes into its OWN tree by design (it derives that
    // tree from its own location, `${0:A:h}:h`). The same is true of the
    // Gemma Scope analysis venv. So those callers need a real, writable
    // checkout or an honest refusal — never a bundle path, and never a
    // file-not-found.
    //
    // The resolution rules, in order, and why each exists:
    //
    //   1. THE COMPILED CHECKOUT, when `developerCheckoutAvailable` (which
    //      honors `modeOverrideForTesting` / `releaseModeAsserted`). This is
    //      the developer workflow and it is byte-identical to the pre-WP2
    //      behavior — running from a checkout, the checkout IS the writable
    //      home for every bucket.
    //   2. A CHECKOUT BESIDE THE RUNNING APP — `SteerLab.app`'s parent
    //      folder, which in the shipped home layout is `~/SteerLab/`.
    //   3. A CHECKOUT IN THE DEFAULT HOME (`~/SteerLab`), for an app that
    //      was launched from somewhere else (e.g. /Applications).
    //
    // 2 and 3 detect BY CONTENT via `HomeLayout.isCheckout` (Package.swift +
    // Server/steerlab_server) — never by folder name, since the release
    // clone, this research tree, and a renamed clone all differ.

    /// Where an executable/writable checkout was found.
    public enum ExecutableCheckoutOrigin: Sendable, Equatable {
        /// This binary was compiled inside that checkout and it is still
        /// there — the developer workflow.
        case compiledCheckout
        /// Found by content inside a home folder (the app's own parent, or
        /// `~/SteerLab`), which is the payload.
        case homeFolder(URL)

        /// One clause naming how it was found, for status lines.
        public var displayName: String {
            switch self {
            case .compiledCheckout: return "this build's own checkout"
            case .homeFolder(let home): return "beside the app in \(home.path)"
            }
        }
    }

    /// A code checkout this process may write into and execute from.
    public struct ExecutableCheckout: Sendable, Equatable {
        public let root: URL
        public let origin: ExecutableCheckoutOrigin

        public init(root: URL, origin: ExecutableCheckoutOrigin) {
            self.root = root
            self.origin = origin
        }
    }

    /// No writable checkout — a typed answer about the LAYOUT, so the UI can
    /// say what to do instead of surfacing a path that does not exist.
    public struct ExecutableCheckoutUnavailable:
        Error, Sendable, Equatable, CustomStringConvertible
    {
        /// The home folders that were searched, in order.
        public let searched: [URL]

        public init(searched: [URL]) { self.searched = searched }

        public var description: String { message }

        /// The plain sentence every caller prefixes with its own feature
        /// name ("the local Python engine needs …").
        public var message: String {
            let places = searched.isEmpty
                ? ""
                : " (looked in " + searched.map(\.path).joined(separator: ", ") + ")"
            return "this build has no SteerLab code checkout beside it\(places)"
                + " — clone the code repository into \(HomeLayout.defaultHome.path),"
                + " beside SteerLab.app and Workspaces/, then reopen the app"
        }
    }

    /// Test seam for rules 2–3: stands in for the derived home folders, so a
    /// temp-directory home layout can be exercised without a real `.app`.
    /// Same safety justification as `bundleOverrideForTesting`.
    nonisolated(unsafe) public static var executableHomesOverrideForTesting: [URL]?

    /// The home folders rules 2–3 search, in order and de-duplicated: the
    /// running `.app`'s parent directory, then `~/SteerLab`. A bare SwiftPM
    /// executable (the CLI, the dev app) contributes no bundle candidate —
    /// rule 1 already covers it. The bundled helper CLI resolves the SAME
    /// parent the app does, through `enclosingAppBundle`.
    static var executableHomeCandidates: [URL] {
        if let override = executableHomesOverrideForTesting { return override }
        var candidates: [URL] = []
        if let app = enclosingAppBundle {
            candidates.append(app.deletingLastPathComponent())
        }
        let home = HomeLayout.defaultHome
        if !candidates.contains(home) { candidates.append(home) }
        return candidates
    }

    /// THE bucket-B resolver: the one checkout this process may write into
    /// or execute from, or a typed reason there is none.
    public static func executableCheckout()
        -> Result<ExecutableCheckout, ExecutableCheckoutUnavailable>
    {
        if developerCheckoutAvailable, let root = developerCheckoutRoot {
            return .success(
                ExecutableCheckout(root: root, origin: .compiledCheckout))
        }
        let homes = executableHomeCandidates
        for home in homes {
            if let found = HomeLayout.checkouts(in: home).first {
                return .success(
                    ExecutableCheckout(
                        root: found.standardizedFileURL, origin: .homeFolder(home)))
            }
        }
        return .failure(ExecutableCheckoutUnavailable(searched: homes))
    }

    /// Convenience for the many `guard let root = …` call sites.
    public static var executableCheckoutRoot: URL? {
        try? executableCheckout().get()
            .root
    }

    // MARK: - Bucket B, second tier: the MATERIALIZED ENGINE ROOT (WP3)
    //
    // `executableCheckout()` above answers "where may I write into or execute
    // a CODE CHECKOUT?" — and for an app-only install the honest answer is
    // "nowhere". That is correct for every dev feature (the CLI reference
    // regenerator, the gemmascope venv, `start-local-server.sh`), and it is
    // too strict for exactly ONE thing: the local Python engine, whose source
    // the app already ships as read-only bytes in `ServerPayload/`.
    //
    // So WP3 adds a second, NARROWER writable tier — the ENGINE ROOT — and
    // keeps it explicitly distinct from a checkout:
    //
    //   * a checkout can run every developer feature; an engine root cannot.
    //     It has no `Package.swift`, no `scripts/`, no git identity. Its only
    //     capability is "hold the Python engine and its venv".
    //   * `HomeLayout.isCheckout` is FALSE for it by construction (no
    //     `Package.swift`), so it can never be discovered as a checkout by
    //     the sibling/home search, and `executableCheckout()` never returns
    //     one. The conflation is closed by content, not by convention.
    //   * it is MATERIALIZED (copied) out of the bundle. Nothing ever writes
    //     into `Contents/Resources/ServerPayload`; the first write there
    //     would break the app's signature.
    //
    // Resolution order for the local engine, therefore:
    //   1. a real checkout (`executableCheckout()`) — richer, and the
    //      developer workflow, byte-identical to pre-WP3;
    //   2. the materialized engine root, when one has been created;
    //   (3. neither — the provisioner's step 1 materializes, if this build
    //       carries a `ServerPayload` at all.)

    /// The stamp a materialization writes at the engine root, so SKEW between
    /// the running app and the engine tree beside it is knowable rather than
    /// guessed. Written from the bundle's `SLSourceRevision`.
    public struct EngineStamp: Codable, Sendable, Equatable {
        public static let currentSchemaVersion = 1
        public static let fileName = "engine-stamp.json"

        public var schemaVersion: Int
        /// The app build revision this tree was copied out of; nil when the
        /// materializing build carried none (a dev bundle).
        public var sourceRevision: String?
        /// `SteerLabVersion`-style full version of the materializing app.
        public var appVersion: String?
        /// ISO-8601 instant of the copy.
        public var materializedAt: String
        /// How many regular files were copied — a cheap "is this tree
        /// plausibly intact" number for the report line.
        public var fileCount: Int

        public init(
            schemaVersion: Int = EngineStamp.currentSchemaVersion,
            sourceRevision: String?, appVersion: String?,
            materializedAt: String, fileCount: Int
        ) {
            self.schemaVersion = schemaVersion
            self.sourceRevision = sourceRevision
            self.appVersion = appVersion
            self.materializedAt = materializedAt
            self.fileCount = fileCount
        }
    }

    /// A materialized Python-engine tree: writable, executable, and NOT a
    /// checkout.
    public struct EngineRoot: Sendable, Equatable {
        /// `~/SteerLab/Engine` by default.
        public let root: URL
        /// The stamp found at the root, when there is one (a hand-copied tree
        /// has none — reported, never invented).
        public let stamp: EngineStamp?

        public init(root: URL, stamp: EngineStamp?) {
            self.root = root.standardizedFileURL
            self.stamp = stamp
        }

        /// `<root>/Server` — deliberately the same spelling a checkout uses,
        /// so `Server/.venv.nosync/bin/python` is ONE path expression for
        /// both tiers instead of two.
        public var serverDirectory: URL { root.appending(component: "Server") }
    }

    /// Where the LOCAL PYTHON ENGINE runs from. Two tiers, never conflated.
    public enum LocalEngineSource: Sendable, Equatable {
        case checkout(ExecutableCheckout)
        case engineRoot(EngineRoot)

        /// The directory holding `steerlab_server/`, `pyproject.toml`, and
        /// the platform locks.
        public var serverDirectory: URL {
            switch self {
            case .checkout(let checkout):
                return checkout.root.appending(component: "Server")
            case .engineRoot(let engine):
                return engine.serverDirectory
            }
        }

        /// The venv interpreter for this source — the same relative spelling
        /// `LocalPythonRuntime.venvPython` resolves in the checkout case.
        public var venvPython: URL {
            serverDirectory.appending(components: ".venv.nosync", "bin", "python")
        }

        /// The working directory a serve/tool invocation runs in.
        public var workingDirectory: URL {
            switch self {
            case .checkout(let checkout): return checkout.root
            case .engineRoot(let engine): return engine.root
            }
        }

        /// FALSE for a materialized engine root: it carries the Python engine
        /// and nothing else, so no caller may treat it as a checkout and go
        /// looking for `Package.swift`, `scripts/`, `docs/`, or a git identity.
        public var supportsDeveloperFeatures: Bool {
            if case .checkout = self { return true }
            return false
        }

        /// One clause naming the tier, for status lines.
        public var displayName: String {
            switch self {
            case .checkout(let checkout):
                return "code checkout at \(checkout.root.path) "
                    + "(\(checkout.origin.displayName))"
            case .engineRoot(let engine):
                return "managed engine at \(engine.root.path)"
            }
        }
    }

    /// Test seam for the engine root's location.
    /// Same safety justification as `bundleOverrideForTesting`.
    nonisolated(unsafe) public static var engineRootOverrideForTesting: URL?

    /// `~/SteerLab/Engine`. Created ON DEMAND by the local-engine provisioner
    /// — deliberately NOT part of `HomeLayout.directoryNames`, because
    /// `steerlab-cli init` materializes the layout a RESEARCHER needs
    /// (`Workspaces/`, `Sites/`) and an empty `Engine/` folder would be a
    /// promise the layout command cannot keep.
    public static var defaultEngineRoot: URL {
        engineRootOverrideForTesting?.standardizedFileURL
            ?? HomeLayout.defaultHome.appending(component: "Engine")
            .standardizedFileURL
    }

    /// Does this directory hold a materialized Python engine? By CONTENT, the
    /// same discipline `HomeLayout.isCheckout` uses: the package directory
    /// under `Server/`. A tree that also carried `Package.swift` would be a
    /// checkout, not an engine root, and is refused here so the two tiers can
    /// never answer for each other.
    public static func isEngineRoot(_ url: URL) -> Bool {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard
            fm.fileExists(
                atPath: url.appending(path: "Server/steerlab_server").path,
                isDirectory: &isDirectory), isDirectory.boolValue
        else { return false }
        // A checkout is resolved by the checkout resolver, never here.
        return !fm.fileExists(atPath: url.appending(path: "Package.swift").path)
    }

    /// Read the engine stamp at `root`, when there is one.
    public static func engineStamp(at root: URL) -> EngineStamp? {
        let url = root.appending(component: EngineStamp.fileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(EngineStamp.self, from: data)
    }

    /// THE local-engine resolver: a checkout if there is one, else the
    /// materialized engine root if it exists, else nil (which the provisioner
    /// turns into "materialize", not into an error).
    public static func localEngineSource() -> LocalEngineSource? {
        if case .success(let checkout) = executableCheckout() {
            return .checkout(checkout)
        }
        let root = defaultEngineRoot
        guard isEngineRoot(root) else { return nil }
        return .engineRoot(EngineRoot(root: root, stamp: engineStamp(at: root)))
    }

    // MARK: - The app's own build identity (never the checkout's state)

    /// `SLSourceRevision` from the packaged app's Info.plist: the short SHA
    /// of the checkout `scripts/build-app.sh` ASSEMBLED THIS BUILD FROM.
    ///
    /// It says nothing about any checkout on this machine now, and callers
    /// must label it as the app's build revision rather than letting it read
    /// as a live git observation. Empty/absent → nil.
    public static var bundleSourceRevision: String? {
        nonEmptyBundleString("SLSourceRevision")
    }

    /// `SLFullVersionString` — the engine stamp including any pre-release
    /// suffix, which `CFBundleShortVersionString` cannot carry.
    public static var bundleFullVersion: String? {
        nonEmptyBundleString("SLFullVersionString")
    }

    private static func nonEmptyBundleString(_ key: String) -> String? {
        guard let raw = Bundle.main.object(forInfoDictionaryKey: key) as? String
        else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    // MARK: - Self-check (the packaged tier's own proof)

    /// A resolution self-check over every shipped family plus the bucket-B
    /// resolver. Cheap by construction — it resolves paths and reads the
    /// manifest's file LIST; it never hashes (that is `install verify`), so
    /// the app can run it at startup.
    public struct SelfCheck: Sendable {

        /// One family's outcome.
        public struct Row: Sendable, Equatable {
            public let family: Family
            /// Where it resolved, when it did.
            public let resolvedPath: String?
            /// A plain-language problem, when there is one. `nil` with a
            /// `nil` path means honestly absent (the build manifest in a
            /// developer checkout) — not a failure.
            public let problem: String?
            /// How many `resource-manifest.json` entries live under this
            /// family, when a manifest was found.
            public let manifestEntries: Int?
        }

        public let mode: Mode
        public let rows: [Row]
        /// The app bundle's build revision (Info.plist `SLSourceRevision`),
        /// when this is a packaged build.
        public let buildRevision: String?
        /// Where bucket B resolved, and how.
        public let executableCheckout: ExecutableCheckout?
        /// Why it did not, when it did not.
        public let executableCheckoutProblem: String?

        /// Families that are genuinely broken (honest absence excluded).
        public var problems: [Row] { rows.filter { $0.problem != nil } }

        /// Families that resolved.
        public var resolved: [Row] { rows.filter { $0.resolvedPath != nil } }

        /// One line, suitable for a launch log.
        public var summaryLine: String { "CodeResources: " + summaryDetail }

        /// The same summary without the `CodeResources: ` prefix, for
        /// contexts (like `install version`) that already label the row.
        public var summaryDetail: String {
            var line = "\(mode.rawValue) mode — "
                + "\(resolved.count)/\(rows.count) resource families resolved"
            if let revision = buildRevision { line += "; app build \(revision)" }
            if problems.isEmpty {
                line += "; no problems"
            } else {
                line += "; PROBLEMS: "
                    + problems.map(\.family.displayName).joined(separator: ", ")
            }
            return line
        }

        /// The multi-line report `install version` renders.
        public var report: [String] {
            var lines = [summaryLine]
            for row in rows {
                if let problem = row.problem {
                    lines.append("    \(row.family.rawValue): \(problem)")
                } else if let path = row.resolvedPath {
                    var detail = path
                    if let entries = row.manifestEntries {
                        detail += " (\(entries) manifest entr\(entries == 1 ? "y" : "ies"))"
                    }
                    lines.append("    \(row.family.rawValue): \(detail)")
                } else {
                    lines.append(
                        "    \(row.family.rawValue): absent — a checkout ships no "
                            + "packaging manifest")
                }
            }
            if let checkout = executableCheckout {
                lines.append(
                    "    executable checkout: \(checkout.root.path) "
                        + "(\(checkout.origin.displayName))")
            } else {
                lines.append(
                    "    executable checkout: none — "
                        + (executableCheckoutProblem ?? "unavailable"))
            }
            return lines
        }
    }

    /// Run the self-check. Never throws: a diagnostic that fails because
    /// something is missing has answered the wrong question.
    public static func selfCheck() -> SelfCheck {
        let activeMode = mode

        // The manifest's file list, when a manifest is there — used only to
        // count entries per family, which is what ties "this family resolved"
        // to "packaging actually shipped it".
        var manifestPaths: [String] = []
        var manifestRow: SelfCheck.Row
        do {
            if let manifestURL = try buildManifest() {
                let manifest = try ResourceManifest.load(from: manifestURL)
                manifestPaths = Array(manifest.files.keys)
                manifestRow = SelfCheck.Row(
                    family: .buildManifest,
                    resolvedPath: manifestURL.path, problem: nil,
                    manifestEntries: manifest.files.count)
            } else {
                // Developer mode: honestly absent, and that is not a failure.
                manifestRow = SelfCheck.Row(
                    family: .buildManifest, resolvedPath: nil, problem: nil,
                    manifestEntries: nil)
            }
        } catch {
            manifestRow = SelfCheck.Row(
                family: .buildManifest, resolvedPath: nil,
                problem: String(describing: error), manifestEntries: nil)
        }

        var rows: [SelfCheck.Row] = []
        for family in Family.allCases where family != .buildManifest {
            do {
                let url = try self.url(for: family)
                var entries: Int?
                if !manifestPaths.isEmpty {
                    // `clusterPayload`'s dev fallback is the checkout ROOT, so
                    // a prefix count is meaningless there; only count when the
                    // family lives at its own name (any bundle layout).
                    let prefix = family.rawValue + "/"
                    entries = manifestPaths.count { $0.hasPrefix(prefix) }
                }
                var problem: String?
                if activeMode == .release, let entries, entries == 0 {
                    problem = "resolved at \(url.path) but the resource manifest "
                        + "lists no files under \(family.rawValue)/ — this build "
                        + "was packaged without its \(family.displayName)"
                }
                rows.append(
                    SelfCheck.Row(
                        family: family, resolvedPath: problem == nil ? url.path : nil,
                        problem: problem, manifestEntries: entries))
            } catch let error as CodeResourceError {
                rows.append(
                    SelfCheck.Row(
                        family: family, resolvedPath: nil, problem: error.reason,
                        manifestEntries: nil))
            } catch {
                rows.append(
                    SelfCheck.Row(
                        family: family, resolvedPath: nil,
                        problem: String(describing: error), manifestEntries: nil))
            }
        }
        rows.append(manifestRow)

        switch executableCheckout() {
        case .success(let checkout):
            return SelfCheck(
                mode: activeMode, rows: rows, buildRevision: bundleSourceRevision,
                executableCheckout: checkout, executableCheckoutProblem: nil)
        case .failure(let absence):
            return SelfCheck(
                mode: activeMode, rows: rows, buildRevision: bundleSourceRevision,
                executableCheckout: nil, executableCheckoutProblem: absence.message)
        }
    }
}

/// Typed, plain-language failure: which family is unresolvable and what the
/// user should do about it.
public struct CodeResourceError: Error, CustomStringConvertible, Sendable, Equatable {
    public let family: CodeResources.Family
    public let reason: String
    public var description: String { reason }

    public init(family: CodeResources.Family, reason: String) {
        self.family = family
        self.reason = reason
    }
}

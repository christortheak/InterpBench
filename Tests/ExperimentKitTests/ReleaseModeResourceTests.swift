import Foundation
import Testing

@testable import ExperimentKit

/// Phase A exit criterion (docs/MAC-DISTRIBUTION-AND-MANAGED-SERVER-PROPOSAL.md
/// §12): SteerLab can operate with the source checkout unavailable. These
/// tests stage a release-like resource bundle in a REAL temp directory —
/// never under the checkout, which lives in iCloud Drive where an
/// "unavailable checkout" simulation would lie — and drive `CodeResources`
/// through it.
///
/// The suite is `.serialized` and every test restores the `CodeResources`
/// globals it touches. Since wave 2, production code across the target
/// (workspace seeding, seed-or-workspace template resolution, web assets,
/// version stamping) READS these globals through `CodeResources`, so every
/// override window here also holds the target-wide
/// `ExperimentRootOverrideLock` — suites creating workspaces in parallel
/// must never observe a staged bundle they didn't stage.
@Suite(.serialized) struct ReleaseModeResourceTests {

    // MARK: Staging

    /// Creates a fresh staged bundle laid out like the app's Resources/
    /// (§3.1): a minimal but real member of every family, plus a generated
    /// `resource-manifest.json`. Caller removes it.
    private func stageBundle() throws -> URL {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appending(component: "steerlab-staged-bundle-\(UUID().uuidString)")
        func stage(_ relative: String, _ contents: String) throws {
            let url = root.appending(path: relative)
            try fm.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try contents.write(to: url, atomically: true, encoding: .utf8)
        }
        // Workspace seeds: the tree WorkspaceStore.create copies from, at
        // paths `WorkspaceStore.seedManifest` actually names (WP1 — the
        // seed is an explicit file allowlist, so unlisted staged files are
        // simply not copied).
        try stage(
            "WorkspaceSeed/prompts/templates/unnamed-scenario-v1.json",
            "{\"id\": \"unnamed-scenario-v1\", \"text\": \"staged\"}\n")
        try stage(
            "WorkspaceSeed/prompts/rubrics/default-paired-v1.md",
            "STAGED-BUNDLE rubric\n")
        try stage(
            "WorkspaceSeed/prompts/batteries/basic.jsonl",
            "{\"prompt\": \"staged?\", \"answer\": \"yes\", \"grading\": \"yes_no\"}\n")
        // The OpenRouter provider-identity fixture (2026-07-24). Staged
        // because it is a SHIPPED resource on the measurement path: without
        // it, canonicalization degrades to lowercasing and a pinned
        // `google-vertex` stops matching OpenRouter's `Google`. It rides in
        // the CLUSTER PAYLOAD (the root-layout tree that carries
        // `prompts/fixtures/` for parity) — WP1 narrowed `WorkspaceSeed/`
        // to curated seed DATA, which carries no fixtures.
        try stage(
            "ClusterPayload/" + OpenRouterProviderIdentity.fixtureRelativePath,
            #"""
            {"schemaVersion": 1, "source": "…/api/v1/providers",
             "fetchedOn": "2026-07-24", "providerCount": 2,
             "providers": [{"name": "Google", "slug": "google-vertex"},
                           {"name": "DeepInfra", "slug": "deepinfra"}]}
            """#)
        // Server payload subset.
        try stage("ServerPayload/pyproject.toml", "[project]\nname = \"steerlab-server\"\n")
        try stage("ServerPayload/steerlab_server/cli.py", "print('stub')\n")
        // Cluster payload: ROOT layout, mirroring what the provisioner
        // pushes from a checkout (Server/ incl. its scripts/, plus parity
        // fixtures), with its own deployment manifest (proposal §10).
        try stage(
            "ClusterPayload/Server/scripts/bootstrap.sh", "#!/bin/sh\necho bootstrap\n")
        try stage(
            "ClusterPayload/Server/scripts/controller-job.sbatch.template",
            "#SBATCH --job-name=steerlab-controller\n")
        try stage(
            "ClusterPayload/prompts/fixtures/parity-prompts.json", "{\"prompts\": []}\n")
        let deployment = try ResourceManifest.generate(
            over: root.appending(component: "ClusterPayload"),
            serverVersion: "0.0-test", protocolVersion: 1)
        try deployment.write(
            to: root.appending(
                components: "ClusterPayload",
                ClusterProvisioner.deploymentManifestFileName))
        // Analysis tools.
        try stage("AnalysisTools/gemmascope_analyze.py", "# fake analysis script\n")
        // Web assets.
        try stage("web/index.html", "<!doctype html><title>SteerLab</title>\n")
        // Manifest over everything staged so far (written after generation,
        // so it never lists itself — verify() ignores unlisted siblings).
        let manifest = try ResourceManifest.generate(
            over: root, serverVersion: "0.0-test", protocolVersion: 1)
        try manifest.write(to: root.appending(component: "resource-manifest.json"))
        return root
    }

    /// Runs `body` with the staged bundle standing in for the app bundle,
    /// restoring all CodeResources globals afterwards. Holds the shared
    /// override lock: production code in other suites resolves seeds and
    /// assets through these globals now.
    private func withStagedBundle(
        mode: CodeResources.Mode? = nil,
        _ body: (URL) throws -> Void
    ) throws {
        ExperimentRootOverrideLock.acquire()
        defer { ExperimentRootOverrideLock.release() }
        let staged = try stageBundle()
        CodeResources.bundleOverrideForTesting = staged
        CodeResources.modeOverrideForTesting = mode
        defer {
            CodeResources.bundleOverrideForTesting = nil
            CodeResources.modeOverrideForTesting = nil
            try? FileManager.default.removeItem(at: staged)
        }
        try body(staged)
    }

    // MARK: Provider-identity fixture (2026-07-24)

    @Test func providerFixtureResolvesInsideAStagedBundle() throws {
        // The regression this closes: the fixture used to resolve from the
        // compiled-in source location, so a distributed .app with no
        // checkout would find nothing, degrade to plain lowercasing, and
        // silently reinstate the bug the fixture exists to fix — a pinned
        // `google-vertex` no longer matching OpenRouter's `Google`.
        let checkout = try #require(CodeResources.developerCheckoutRoot)
        try withStagedBundle(mode: .release) { staged in
            let url = try #require(
                OpenRouterProviderIdentity.resolvedFixtureURL(),
                "the provider fixture must resolve in RELEASE mode")
            #expect(url.path.hasPrefix(staged.standardizedFileURL.path + "/"))
            #expect(!url.path.hasPrefix(checkout.path + "/"))

            // ... and it is genuinely readable there, mapping OpenRouter's
            // display name onto the routing slug.
            let table = OpenRouterProviderIdentity.loadAliases(from: url)
            #expect(table["google"] == "google-vertex")
            #expect(table["deepinfra"] == "deepinfra")
        }
    }

    @Test func providerFixtureIsShippedInTheRealCheckout() throws {
        // Packaging must carry it: a bundle whose ClusterPayload lacks the
        // fixture is a silently-degraded build, so assert the source of
        // truth exists where the family looks for it.
        let payloadRoot = try CodeResources.clusterPayload()
        let url = payloadRoot.appending(
            path: OpenRouterProviderIdentity.fixtureRelativePath)
        #expect(FileManager.default.fileExists(atPath: url.path))
        #expect(OpenRouterProviderIdentity.knownSpellingCount >= 50)
    }

    // MARK: 1 — every family resolves inside the staged bundle

    @Test func familiesResolveInsideStagedBundleNeverTheCheckout() throws {
        let checkout = try #require(CodeResources.developerCheckoutRoot)
        try withStagedBundle { staged in
            let stagedPrefix = staged.standardizedFileURL.path + "/"
            let checkoutPrefix = checkout.path + "/"
            let resolved: [URL] = [
                try CodeResources.workspaceSeed(),
                try CodeResources.serverPayload(),
                try CodeResources.clusterPayload(),
                try CodeResources.analysisTools(),
                try CodeResources.webAssets(),
                try #require(try CodeResources.buildManifest()),
            ]
            for url in resolved {
                #expect(url.path.hasPrefix(stagedPrefix) || url.path == staged.standardizedFileURL.path)
                #expect(!url.path.hasPrefix(checkoutPrefix))
                #expect(url.path != checkout.path)
            }
            // The seed root really is "a root containing prompts/…".
            let seeded = try CodeResources.workspaceSeed()
                .appending(path: "prompts/rubrics/default-paired-v1.md")
            #expect(FileManager.default.fileExists(atPath: seeded.path))
        }
    }

    // MARK: 2 — staged manifest verifies clean; damage is named plainly

    @Test func manifestVerifiesCleanThenFlagsCorruptionAndDeletion() throws {
        try withStagedBundle { staged in
            let manifestURL = try #require(try CodeResources.buildManifest())
            let manifest = try ResourceManifest.load(from: manifestURL)
            #expect(manifest.schemaVersion == ResourceManifest.currentSchemaVersion)
            #expect(manifest.verify(against: staged).isEmpty)

            // Corrupt one file in place.
            let corrupted = staged.appending(
                path: "ClusterPayload/Server/scripts/bootstrap.sh")
            try "#!/bin/sh\necho tampered\n".write(
                to: corrupted, atomically: true, encoding: .utf8)
            // Delete another outright.
            let deleted = staged.appending(path: "web/index.html")
            try FileManager.default.removeItem(at: deleted)

            let problems = manifest.verify(against: staged)
            #expect(problems.count == 2)
            let mismatch = try #require(
                problems.first(where: { $0.kind == .mismatch }))
            #expect(mismatch.path == "ClusterPayload/Server/scripts/bootstrap.sh")
            #expect(
                mismatch.message == "ClusterPayload/Server/scripts/bootstrap.sh "
                    + "does not match the manifest — the payload is corrupt or tampered")
            let missing = try #require(
                problems.first(where: { $0.kind == .missing }))
            #expect(missing.path == "web/index.html")
            #expect(missing.message == "web/index.html is missing from the payload")
        }
    }

    // MARK: 3 — release mode fails closed on a missing family

    @Test func missingFamilyFailsClosedInReleaseMode() throws {
        try withStagedBundle(mode: .release) { staged in
            try FileManager.default.removeItem(
                at: staged.appending(component: "AnalysisTools"))
            #expect(!CodeResources.developerCheckoutAvailable)
            do {
                _ = try CodeResources.analysisTools()
                Issue.record("release mode resolved a missing family instead of failing closed")
            } catch let error as CodeResourceError {
                #expect(error.family == .analysisTools)
                #expect(error.description.contains("analysis tools"))
                #expect(error.description.contains("reinstall the app"))
            }
            // The intact families still resolve — fail-closed is per family.
            #expect(try CodeResources.webAssets().path.hasPrefix(
                staged.standardizedFileURL.path))
        }
    }

    // MARK: 3b — developer-only tools refuse honestly without a checkout

    @Test func localPythonEngineIsUnavailableWithoutADeveloperCheckout() throws {
        // WP2 sharpened what "without a checkout" means. Release mode alone no
        // longer implies it: a packaged app whose home folder holds a checkout
        // beside it CAN run the local Python engine, which is the whole point
        // of the app+checkout tier (`CheckoutDependencyTests`
        // .siblingCheckoutIsFoundByContentNotByName pins that direction). What
        // stays true, and is asserted here, is the app-ALONE tier: no compiled
        // checkout and no checkout in any home folder means every accessor is
        // nil — never a guessed path, and never the bundle's own ServerPayload,
        // which the staged bundle deliberately carries.
        let fm = FileManager.default
        let emptyHome = fm.temporaryDirectory
            .appending(component: "steerlab-no-checkout-\(UUID().uuidString)")
        try fm.createDirectory(at: emptyHome, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: emptyHome) }

        try withStagedBundle(mode: .release) { staged in
            CodeResources.executableHomesOverrideForTesting = [emptyHome]
            defer { CodeResources.executableHomesOverrideForTesting = nil }
            #expect(!CodeResources.developerCheckoutAvailable)
            let payloadPath = try CodeResources.serverPayload().path
            #expect(payloadPath.hasPrefix(staged.standardizedFileURL.path))
            #expect(LocalPythonRuntime.repoRoot == nil)
            #expect(LocalPythonRuntime.venvPython == nil)
            #expect(LocalPythonRuntime.startServerScript == nil)
            #expect(!LocalPythonRuntime.venvExists)
            // The refusal names the LAYOUT, not a missing file.
            #expect(
                LocalPythonRuntime.unavailableHint.contains(
                    "needs the code checkout beside the app"))
        }
    }

    // MARK: 3c — the cluster payload: bundle-resolved, manifest-gated

    @Test func clusterPayloadResolvesFromBundleAndFailsClosedWithout() throws {
        try withStagedBundle(mode: .release) { staged in
            // The provisioner's default local payload path IS the bundled
            // ClusterPayload tree…
            let payload = staged.appending(component: "ClusterPayload")
                .standardizedFileURL.path
            #expect(ClusterProvisioner.defaultLocalRepoPath() == payload)
            // …whose shipped deployment manifest verifies clean…
            #expect(
                ClusterProvisioner.deploymentManifestFailure(atPayloadRoot: payload)
                    == nil)
            // …and drift is a plain-language refusal naming file and remedy.
            try "#!/bin/sh\necho tampered\n".write(
                to: staged.appending(
                    path: "ClusterPayload/Server/scripts/bootstrap.sh"),
                atomically: true, encoding: .utf8)
            let failure = try #require(
                ClusterProvisioner.deploymentManifestFailure(atPayloadRoot: payload))
            #expect(failure.contains("does not match its manifest"))
            #expect(failure.contains("Server/scripts/bootstrap.sh"))
            #expect(failure.contains("make-server-payload.sh"))

            // A release build MISSING the family resolves to no path at all
            // (empty — the push step fails closed with the locator's error),
            // never a guessed directory like the CWD.
            try FileManager.default.removeItem(
                at: staged.appending(component: "ClusterPayload"))
            #expect(ClusterProvisioner.defaultLocalRepoPath() == "")
            #expect(throws: CodeResourceError.self) {
                try CodeResources.clusterPayload()
            }
        }
    }

    /// Byte parity with the pre-wave-3 `#filePath` derivation: in a dev
    /// checkout the default local payload path is still the repo root.
    @Test func provisionerDefaultPathIsTheCheckoutInDeveloperMode() throws {
        let expected = URL(filePath: #filePath)  // …/Tests/ExperimentKitTests/…
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .path
        #expect(ClusterProvisioner.defaultLocalRepoPath() == expected)
        // A checkout ships no deployment manifest — the push verification
        // is honestly inert in developer mode.
        #expect(
            ClusterProvisioner.deploymentManifestFailure(atPayloadRoot: expected) == nil)
    }

    @Test func missingManifestFailsClosedInReleaseModeOnly() throws {
        try withStagedBundle(mode: .developer) { staged in
            try FileManager.default.removeItem(
                at: staged.appending(component: "resource-manifest.json"))
            // Developer mode: honestly absent, not an error and not a fake.
            #expect(try CodeResources.buildManifest() == nil)
            CodeResources.modeOverrideForTesting = .release
            #expect(throws: CodeResourceError.self) {
                try CodeResources.buildManifest()
            }
        }
    }

    // MARK: 4 — generator round-trip

    @Test func generatorRoundTripDetectsSingleByteMutation() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory
            .appending(component: "steerlab-manifest-roundtrip-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        let nested = root.appending(components: "payload", "nested")
        try fm.createDirectory(at: nested, withIntermediateDirectories: true)
        let target = nested.appending(component: "asset.bin")
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: target)
        try "hello\n".write(
            to: root.appending(component: "readme.txt"),
            atomically: true, encoding: .utf8)
        // Hidden entries never enter the manifest (determinism contract).
        try "junk".write(
            to: root.appending(component: ".DS_Store"),
            atomically: true, encoding: .utf8)

        let manifest = try ResourceManifest.generate(
            over: root, serverVersion: "0.0-test", protocolVersion: 1)
        #expect(
            manifest.files.keys.sorted() == ["payload/nested/asset.bin", "readme.txt"])
        #expect(manifest.verify(against: root).isEmpty)

        // Determinism: regenerating yields identical canonical bytes.
        let again = try ResourceManifest.generate(
            over: root, serverVersion: "0.0-test", protocolVersion: 1)
        #expect(try manifest.canonicalJSON() == again.canonicalJSON())

        // Mutate a single byte.
        try Data([0x00, 0x01, 0x02, 0xFF]).write(to: target)
        let problems = manifest.verify(against: root)
        #expect(problems.count == 1)
        #expect(problems.first?.kind == .mismatch)
        #expect(problems.first?.path == "payload/nested/asset.bin")

        // sourceRevision (schema 1, optional): omitted when nil, carried
        // through the canonical JSON round trip when set.
        #expect(manifest.sourceRevision == nil)
        #expect(!String(
            decoding: try manifest.canonicalJSON(), as: UTF8.self
        ).contains("sourceRevision"))
        let stamped = try ResourceManifest.generate(
            over: root, serverVersion: "0.0-test", protocolVersion: 1,
            sourceRevision: "abc12345")
        let reloaded = try JSONDecoder().decode(
            ResourceManifest.self, from: stamped.canonicalJSON())
        #expect(reloaded.sourceRevision == "abc12345")
        #expect(reloaded.schemaVersion == ResourceManifest.currentSchemaVersion)
    }

    // MARK: 5 — end to end: a workspace is born from the staged bundle

    /// The Phase A exit criterion as behavior, not just resolution:
    /// `WorkspaceStore.create` seeds a fully functional workspace from the
    /// staged bundle in release mode — the checkout is never consulted.
    @Test func createWorkspaceFromStagedBundle() throws {
        let fm = FileManager.default
        try withStagedBundle(mode: .release) { staged in
            // Enrich the minimal staged seed with more MANIFEST entries, so
            // the copy is exercised across several destination directories.
            // Bytes are distinctive and unlike the checkout's, so
            // provenance is unambiguous. A staged file the manifest does
            // not name is planted deliberately: it must NOT be copied.
            func stage(_ relative: String, _ contents: String) throws {
                let url = staged.appending(path: relative)
                try fm.createDirectory(
                    at: url.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                try contents.write(to: url, atomically: true, encoding: .utf8)
            }
            try stage(
                "WorkspaceSeed/prompts/dev/dev-prompts.jsonl",
                "{\"text\": \"staged dev prompt\"}\n")
            try stage(
                "WorkspaceSeed/prompts/parsers/parser-registry.json",
                "{\"schemaVersion\": 1, \"parsers\": {}}\n")
            try stage(
                "WorkspaceSeed/prompts/tasks/example-task-prompts.jsonl",
                "{\"id\": \"t1\", \"text\": \"staged task\"}\n")
            // Unlisted: the allowlist's whole point.
            try stage(
                "WorkspaceSeed/prompts/concepts/staged-concept/positive.jsonl",
                "{\"text\": \"staged positive\"}\n")
            try stage(
                "WorkspaceSeed/prompts/rubrics/staged-rubric.md",
                "STAGED-BUNDLE unlisted rubric\n")

            let workspace = fm.temporaryDirectory.appending(
                component: "staged-workspace-\(UUID().uuidString)")
            defer { try? fm.removeItem(at: workspace) }
            #expect(!CodeResources.developerCheckoutAvailable)
            let created = try WorkspaceStore.create(at: workspace)

            // Functional workspace: marker, full skeleton, gitignore.
            #expect(WorkspaceStore.isWorkspace(url: created))
            #expect(fm.fileExists(
                atPath: created.appending(
                    component: WorkspaceStore.markerFileName).path))
            for sub in WorkspaceStore.promptSubdirectories {
                var isDirectory: ObjCBool = false
                #expect(
                    fm.fileExists(
                        atPath: created.appending(components: "prompts", sub).path,
                        isDirectory: &isDirectory) && isDirectory.boolValue,
                    "missing prompts/\(sub)")
            }
            for top in ["experiments", "runs", "adapters"] {
                #expect(fm.fileExists(atPath: created.appending(component: top).path))
            }
            #expect(
                try String(
                    contentsOf: created.appending(component: ".gitignore"),
                    encoding: .utf8
                ).contains("runs/"))

            // Seeded bytes came from the STAGED bundle…
            func stagedBytesLanded(_ seedRelative: String, at workspaceRelative: String) throws {
                let expected = try Data(
                    contentsOf: staged.appending(path: "WorkspaceSeed/" + seedRelative))
                let actual = try? Data(
                    contentsOf: created.appending(path: workspaceRelative))
                #expect(actual == expected, "staged seed missing: \(workspaceRelative)")
            }
            for relative in [
                "prompts/templates/unnamed-scenario-v1.json",
                "prompts/rubrics/default-paired-v1.md",
                "prompts/batteries/basic.jsonl",
                "prompts/dev/dev-prompts.jsonl",
                "prompts/parsers/parser-registry.json",
                "prompts/tasks/example-task-prompts.jsonl",
            ] {
                try stagedBytesLanded(relative, at: relative)
            }

            // …the unlisted staged files did NOT: an explicit allowlist, not
            // a sweep of whatever the bundle happens to carry.
            for unlisted in [
                "prompts/concepts/staged-concept/positive.jsonl",
                "prompts/rubrics/staged-rubric.md",
            ] {
                #expect(
                    !fm.fileExists(atPath: created.appending(path: unlisted).path),
                    "unlisted seed file was copied: \(unlisted)")
            }

            // Negative proof the checkout was not the source: manifest
            // entries the staged bundle does NOT carry are simply absent,
            // rather than silently re-resolved from the developer tree.
            #expect(!fm.fileExists(
                atPath: created.appending(
                    path: "prompts/neutral/corpus.jsonl").path))
            #expect(!fm.fileExists(
                atPath: created.appending(
                    path: "prompts/templates/validation/validation-template.jsonl").path))
        }
    }

    // MARK: 6 — the workspace/checkout conflation guards survive migration

    /// A workspace that IS the checkout must refuse freeze auto-commit
    /// (auto-`git add -A` on the developer's repo would commit unrelated
    /// work), while a managed workspace with its own repo allows it. The
    /// guard compares against `CodeResources.compiledCheckoutPath` via the
    /// `bundledSeedRoot` alias — this pins the semantics across the wave-2
    /// migration.
    @Test func freezeAutoCommitStillRefusesTheDeveloperCheckout() throws {
        ExperimentRootOverrideLock.acquire()
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
        }
        // The guard consults the live workspace root only when the test
        // root override is clear (it is, outside any withTempRoot window).
        WorkspaceRoot.programmaticOverride = VectorCatalog.bundledSeedRoot
        #expect(!ExperimentStore.freezeAutoCommitIsEnabled())

        // Control: a managed workspace (own git repo) IS auto-committable.
        let temp = FileManager.default.temporaryDirectory.appending(
            component: "autocommit-guard-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        let created = try WorkspaceStore.create(at: temp)
        WorkspaceRoot.programmaticOverride = created
        #expect(ExperimentStore.freezeAutoCommitIsEnabled())
    }

    /// The alias itself: `bundledSeedRoot` and the compiled checkout path
    /// are the same location (the other conflation guards —
    /// `WorkspaceStore.isLegacyRepoRoot`, `WorkspaceRoot.current`'s
    /// fallback — compare against the alias, so this equality carries them).
    @Test func bundledSeedRootAliasesTheCompiledCheckoutPath() throws {
        #expect(
            VectorCatalog.bundledSeedRoot.standardizedFileURL.path
                == CodeResources.compiledCheckoutPath.standardizedFileURL.path)
    }

    // MARK: Mode detection + dev fallback map to today's real locations

    @Test func modeIsDeveloperInAWorkingCheckout() throws {
        // No overrides: this test process runs from the checkout, so the
        // runtime signal (root exists + Package.swift marker) must say
        // developer — the exact reason mode is not an #if DEBUG decision.
        let root = try #require(CodeResources.developerCheckoutRoot)
        #expect(CodeResources.mode == .developer)
        #expect(CodeResources.developerCheckoutAvailable)
        #expect(
            FileManager.default.fileExists(
                atPath: root.appending(component: "Package.swift").path))
        // Phase B's assertion flips it without touching the filesystem.
        // (An override window — held under the shared lock like the rest.)
        ExperimentRootOverrideLock.acquire()
        CodeResources.releaseModeAsserted = true
        defer {
            CodeResources.releaseModeAsserted = false
            ExperimentRootOverrideLock.release()
        }
        #expect(CodeResources.mode == .release)
        #expect(!CodeResources.developerCheckoutAvailable)
    }

    @Test func devFallbackResolvesTodaysCheckoutLocations() throws {
        // No staged bundle: every family lands on the location the
        // grandfathered call sites use today, so later-wave migrations are
        // path-for-path behavior-preserving.
        let root = try #require(CodeResources.developerCheckoutRoot)
        // WP1: the seed family is the curated `WorkspaceSeed/` tree in BOTH
        // modes. It used to be the checkout root itself, which made the
        // research tree's own `prompts/` the seed source.
        #expect(
            try CodeResources.workspaceSeed().path
                == root.appending(path: "WorkspaceSeed").path)
        #expect(try CodeResources.workspaceSeed() != root)
        #expect(try CodeResources.serverPayload().path == root.appending(path: "Server").path)
        // The cluster payload is the ROOT-LAYOUT push tree — the checkout
        // root itself in dev (wave 3 reconciled wave 1's Server/scripts
        // guess with what the provisioner actually rsyncs).
        #expect(try CodeResources.clusterPayload() == root)
        #expect(try CodeResources.analysisTools().path == root.appending(path: "scripts").path)
        // `web/` is a committed BUILD OUTPUT here and is built from source
        // (`results-explorer/`, `npm run build:embed`) in the released tree,
        // so a cold clone has no `web/` until that build runs — and
        // `webAssets()` fails closed, by design, when a family is missing.
        // Asserting it unconditionally made this test a proxy for "has
        // someone run npm yet". Every other family is checkout content and
        // stays unconditional.
        if ResearchTreeFixtures.hasBuiltWebAssets {
            #expect(try CodeResources.webAssets().path == root.appending(path: "web").path)
        }
        // The checkout ships no packaging manifest — honest absence.
        #expect(try CodeResources.buildManifest() == nil)
    }

    // MARK: Regression guard — #filePath resource discovery is fenced

    /// Files under Sources/ allowed to mention `#filePath` today. Wave 3
    /// migrated the last grandfather (`ClusterProvisioner`), so the only
    /// sanctioned site is the locator itself.
    private static let filePathAllowlist: Set<String> = [
        // The sanctioned dev-fallback derivation (compiledCheckoutPath).
        "Sources/ExperimentKit/CodeResources.swift"
    ]

    @Test func productionSourcesRouteResourceDiscoveryThroughCodeResources() throws {
        // Tests may use #filePath freely; production resource discovery may
        // not. Walk Sources/ (never Tests/) for offenders.
        let repoRoot = URL(filePath: #filePath)  // …/Tests/ExperimentKitTests/…
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourcesRoot = repoRoot.appending(component: "Sources")
        let fm = FileManager.default
        let enumerator = try #require(
            fm.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil))
        var offenders: [String] = []
        for case let url as URL in enumerator
        where url.pathExtension == "swift" {
            guard
                let contents = try? String(contentsOf: url, encoding: .utf8),
                contents.contains("#filePath")
            else { continue }
            let relative = String(
                url.standardizedFileURL.path.dropFirst(repoRoot.standardizedFileURL.path.count + 1))
            if !Self.filePathAllowlist.contains(relative) {
                offenders.append(relative)
            }
        }
        #expect(
            offenders.isEmpty,
            """
            #filePath-based resource discovery found outside the sanctioned \
            sites: \(offenders.sorted().joined(separator: ", ")). Immutable \
            program resources must resolve through CodeResources (Sources/\
            ExperimentKit/CodeResources.swift) — release builds have no \
            source checkout, and a compiled-in path silently breaks them. \
            If this file genuinely belongs on the allowlist, add it to \
            ReleaseModeResourceTests.filePathAllowlist with a justification.
            """)
    }
}

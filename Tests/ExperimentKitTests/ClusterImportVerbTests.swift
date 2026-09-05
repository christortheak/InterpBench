import Foundation
import Testing

@testable import ExperimentKit

// =============================================================================
// `steerlab-cli cluster import` (open-issues §20) — the WP0 contract surfaces:
// strict argv, `--help`, one envelope, typed refusals, and the generated
// reference region.
//
// The verb drives the SAME `WorkspaceRunImport` code path the app's hook does;
// what is asserted here is the command-line contract around it, with the import
// engine injected so no ssh, no rsync, and no cluster are involved.
// =============================================================================

// MARK: - Parsing

struct ClusterImportParsingTests {

    @Test func theVerbParsesWithItsOwnFlags() throws {
        let invocation = try ClusterCLIParser.parse(
            ["import", "--site", "s", "--since", "2026-08-01", "--dry-run", "--json"])
        #expect(invocation.verb == .importRuns)
        #expect(invocation.siteReference == "s")
        #expect(invocation.since == "20260801T000000000")
        #expect(invocation.dryRun)
        #expect(invocation.json)
    }

    @Test func theVerbRequiresASite() {
        #expect(throws: ClusterCLIError.missingSite(.importRuns)) {
            _ = try ClusterCLIParser.parse(["import"])
        }
    }

    /// A `--since` no date grammar accepts REFUSES. Silently ignoring it would
    /// read as "nothing new to import" — the worst possible failure for a verb
    /// whose whole job is not losing evidence.
    @Test func anUnparseableSinceRefuses() {
        #expect(throws: ClusterCLIError.invalidSince("last tuesday")) {
            _ = try ClusterCLIParser.parse(
                ["import", "--site", "s", "--since", "last tuesday"])
        }
        let error = ClusterCLIError.invalidSince("nope")
        #expect(error.exitCode == 64)
        #expect(error.code == "invalidSince")
        #expect(error.repairAction.contains("2026-08-01"))
    }

    @Test func sinceRequiresAValue() {
        #expect(throws: ClusterCLIError.missingFlagValue("--since")) {
            _ = try ClusterCLIParser.parse(["import", "--site", "s", "--since"])
        }
    }

    /// Strictness: the verb rejects flags it cannot honor rather than ignoring
    /// them. The provisioning overrides are bootstrap inputs and mean nothing
    /// here.
    @Test func theVerbRejectsFlagsItCannotHonor() {
        #expect(
            throws: ClusterCLIError.unknownFlag(flag: "--target", verb: .importRuns)
        ) {
            _ = try ClusterCLIParser.parse(
                ["import", "--site", "s", "--target", "connected"])
        }
        #expect(
            throws: ClusterCLIError.unknownFlag(flag: "--remote-repo", verb: .importRuns)
        ) {
            _ = try ClusterCLIParser.parse(
                ["import", "--site", "s", "--remote-repo", "/tmp"])
        }
    }

    /// `--help` answers before `--site`: a caller asking what a verb takes must
    /// not have to supply it first.
    @Test func helpAnswersWithoutASite() throws {
        let invocation = try ClusterCLIParser.parse(["import", "--help"])
        #expect(invocation.help)
        #expect(invocation.verb == .importRuns)
    }

    @Test func theHelpPageNamesEveryFlagAndThePurpose() {
        let text = ClusterCLIVerb.importRuns.helpText
        #expect(text.hasPrefix("usage: steerlab-cli cluster import"))
        for flag in ClusterCLIVerb.importRuns.declaredFlags {
            #expect(text.contains(flag), "--help omits \(flag)")
        }
        #expect(text.contains("--since <date>"))
        // The generic `--dry-run` gloss is about submission; this verb submits
        // nothing, so it carries a verb-qualified one.
        #expect(text.contains("purge-eligibility"))
        #expect(!text.contains("submit nothing"))
        #expect(text.contains(ExperimentCLIHelp.exitCodeLine))
    }

    @Test func theVerbAppearsInTheGeneratedClusterRegion() {
        let body = CLIReferenceDocument.clusterBody()
        #expect(body.contains("`cluster import`"))
        #expect(body.contains(ClusterCLIVerb.importRuns.purpose))
        #expect(ClusterCLIVerb.usageText.contains("import"))
    }
}

// MARK: - Dispatch

struct ClusterImportRunnerTests {

    private struct Harness {
        var runner: ClusterCLIRunner
        var root: URL
    }

    private func profile() -> ClusterSiteProfile {
        var profile = ClusterSiteProfile.exampleCluster
        profile.name = "Import Test Site"
        profile.constraints.storageRoots = ["workspace": "/scratch/import-test"]
        return profile
    }

    private func harness(
        _ label: String,
        engine: @escaping @Sendable (ClusterSiteRecord, @escaping @Sendable (String) -> Void)
            throws -> WorkspaceRunImport.Engine
    ) throws -> Harness {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-import-verb-\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let repository = ClusterSiteRepository(
            directory: root.appending(component: "cluster-sites"),
            legacyRegistryData: { nil })
        _ = try repository.upsert(profile: profile())
        let runner = ClusterCLIRunner(
            repository: repository,
            operationStore: ClusterOperationStore(
                rootDirectory: root.appending(component: "operations")),
            shell: NeverRunImportShell(),
            now: { Date(timeIntervalSince1970: 1_000) },
            workspaceImportEngine: engine)
        return Harness(runner: runner, root: root)
    }

    private func siteID(_ harness: Harness) throws -> String {
        let repository = ClusterSiteRepository(
            directory: harness.root.appending(component: "cluster-sites"),
            legacyRegistryData: { nil })
        return try #require(repository.sites().first).id
    }

    /// The happy path: one envelope, `ready`, with the report as data.
    @Test func aCleanImportReportsReadyWithTheSummaryAsData() async throws {
        let stamp = "20260819T101500123"
        let run = "\(stamp)-exp-alpha-run"
        let harness = try harness("clean") { _, _ in
            WorkspaceRunImport.Engine(
                listRemoteDirectories: { [run] },
                remoteInventory: { _ in
                    [run: [WorkspaceImportPolicy.FileStat(relativePath: "c.json", size: 8)]]
                },
                transfer: { _, _ in },
                localExists: { _ in false },
                localInventory: { _ in
                    [WorkspaceImportPolicy.FileStat(relativePath: "c.json", size: 8)]
                },
                rebuildCatalog: {
                    WorkspaceRunCatalog.BuildReport(
                        rows: [
                            WorkspaceRunCatalog.Row(
                                name: run, kind: "run", wave: "alpha", study: "alpha",
                                stamp: stamp, fileCount: 1)
                        ],
                        linkCount: 2, adapterCount: 0, libraryCount: 0,
                        gitignoreUpdated: true)
                })
        }
        defer { try? FileManager.default.removeItem(at: harness.root) }

        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .importRuns, siteReference: try siteID(harness)))
        #expect(outcome.exitCode == 0)
        #expect(outcome.envelope.state == "ready")
        #expect(outcome.envelope.changed)
        let summary = try #require(outcome.envelope.importSummary)
        #expect(summary.imported == [run])
        #expect(summary.violations.isEmpty)
        #expect(summary.catalogRuns == 1)
        #expect(!summary.dryRun)
        // One JSON document, and it encodes.
        #expect(try outcome.envelope.jsonText().contains("importSummary"))
    }

    /// `--dry-run` prints classification, what would transfer, and the
    /// purge-eligibility report — and touches nothing.
    @Test func dryRunTransfersNothingAndSaysSo() async throws {
        let stamp = "20260819T101500123"
        let run = "\(stamp)-exp-alpha-run"
        let transfers = TransferLog()
        let harness = try harness("dry") { _, _ in
            WorkspaceRunImport.Engine(
                listRemoteDirectories: { [run] },
                remoteInventory: { _ in
                    [run: [WorkspaceImportPolicy.FileStat(relativePath: "c.json", size: 8)]]
                },
                transfer: { name, _ in transfers.record(name) },
                localExists: { _ in false },
                localInventory: { _ in [] },
                rebuildCatalog: {
                    Issue.record("a dry run must not rebuild the catalog")
                    return WorkspaceRunCatalog.BuildReport(
                        rows: [], linkCount: 0, adapterCount: 0, libraryCount: 0,
                        gitignoreUpdated: false)
                })
        }
        defer { try? FileManager.default.removeItem(at: harness.root) }

        let outcome = await harness.runner.run(
            ClusterCLIInvocation(
                verb: .importRuns, siteReference: try siteID(harness), dryRun: true))
        #expect(outcome.exitCode == 0)
        #expect(!outcome.envelope.changed)
        #expect(transfers.names.isEmpty)
        let summary = try #require(outcome.envelope.importSummary)
        #expect(summary.dryRun)
        #expect(summary.imported == [run])
        #expect(outcome.envelope.message.contains("DRY RUN"))
    }

    /// An orphaned shard family degrades the verb and asks for a human — it is
    /// a finding, not a crash, and never a silent line in a long report.
    @Test func loudPurgeFindingsDegradeTheVerbAndNameTheNextAction() async throws {
        let stamp = "20260819T101500123"
        let partials = (0..<2).map { "\(stamp)-exp-gamma-run-shard\($0)of2" }
        let harness = try harness("orphan") { _, _ in
            WorkspaceRunImport.Engine(
                listRemoteDirectories: { partials },
                remoteInventory: { _ in [:] },
                transfer: { _, _ in Issue.record("partials must never transfer") },
                localExists: { _ in false },
                localInventory: { _ in [] })
        }
        defer { try? FileManager.default.removeItem(at: harness.root) }

        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .importRuns, siteReference: try siteID(harness)))
        #expect(outcome.envelope.state == "degraded")
        #expect(outcome.exitCode == 13)
        let summary = try #require(outcome.envelope.importSummary)
        #expect(summary.purgeEligible.isEmpty)
        #expect(summary.purgeBlocked.count == 1)
        #expect(summary.purgeBlocked.first?.contains("ORPHANED") == true)
        #expect(outcome.envelope.nextAction?.requiresHuman == true)
    }

    /// Authoring divergence (§8 residual (a)) degrades the verb exactly like a
    /// loud purge finding: the import stays runs-only, so a study whose live
    /// manifest holds fewer arms than its run evidence needs a human decision,
    /// never a silent line.
    @Test func authoringDivergenceDegradesTheVerbAndNamesTheNextAction() async throws {
        let stamp = "20260819T101500123"
        let run = "\(stamp)-exp-alpha-run"
        let harness = try harness("diverged") { _, _ in
            WorkspaceRunImport.Engine(
                listRemoteDirectories: { [run] },
                remoteInventory: { _ in
                    [run: [WorkspaceImportPolicy.FileStat(relativePath: "experiment.json", size: 512)]]
                },
                transfer: { _, _ in },
                localExists: { _ in false },
                localInventory: { _ in
                    [WorkspaceImportPolicy.FileStat(relativePath: "experiment.json", size: 512)]
                },
                localRunManifestArms: { _ in
                    WorkspaceImportPolicy.ManifestArms(
                        studyName: "alpha", concepts: 16, conditions: 16)
                },
                liveExperimentArms: { _ in
                    WorkspaceImportPolicy.ManifestArms(
                        studyName: "alpha", concepts: 0, conditions: 0,
                        status: "draft")
                })
        }
        defer { try? FileManager.default.removeItem(at: harness.root) }

        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .importRuns, siteReference: try siteID(harness)))
        #expect(outcome.envelope.state == "degraded")
        #expect(outcome.exitCode == 13)
        let summary = try #require(outcome.envelope.importSummary)
        #expect(summary.authoringDivergences.count == 1)
        #expect(summary.authoringDivergences.first?.contains("AUTHORING DIVERGENCE") == true)
        #expect(summary.violations.isEmpty)
        #expect(outcome.envelope.nextAction?.requiresHuman == true)
        #expect(
            outcome.envelope.nextAction?.detail?.contains("never writes") == true)
        #expect(outcome.envelope.message.contains("AUTHORING DIVERGENCE"))
    }

    /// A run the cluster has not finished — records and no report.json, the
    /// shape of a shard merge the controller died under (2026-09-05) —
    /// degrades the verb and names the next action. Its bytes come home (the
    /// envelope says something changed), but the summary never calls them
    /// imported or already complete.
    @Test func anUnfinishedRunDegradesTheVerbAndIsNeverCertified() async throws {
        let stamp = "20260819T101500123"
        let run = "\(stamp)-exp-alpha-run"
        let files = [
            WorkspaceImportPolicy.FileStat(relativePath: "config.json", size: 120),
            WorkspaceImportPolicy.FileStat(relativePath: "generations.jsonl", size: 4096),
        ]
        let harness = try harness("unfinished") { _, _ in
            WorkspaceRunImport.Engine(
                listRemoteDirectories: { [run] },
                remoteInventory: { _ in [run: files] },
                transfer: { _, _ in },
                localExists: { _ in false },
                localInventory: { _ in files })
        }
        defer { try? FileManager.default.removeItem(at: harness.root) }

        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .importRuns, siteReference: try siteID(harness)))
        #expect(outcome.envelope.state == "degraded")
        #expect(outcome.exitCode == 13)
        #expect(outcome.envelope.changed)
        let summary = try #require(outcome.envelope.importSummary)
        #expect(summary.imported.isEmpty)
        #expect(summary.alreadyComplete.isEmpty)
        #expect(summary.incompleteRuns == [run])
        #expect(summary.violations.isEmpty)
        #expect(outcome.envelope.nextAction?.requiresHuman == true)
        #expect(outcome.envelope.nextAction?.detail?.contains("report.json") == true)
        #expect(outcome.envelope.message.contains("INCOMPLETE RUNS"))
        #expect(try outcome.envelope.jsonText().contains("incompleteRuns"))
    }

    /// Byte drift is a typed failure with a stable code, and its repair action
    /// explicitly forbids re-running the verb.
    @Test func byteDriftFailsWithAStableCodeAndAHumanRepair() async throws {
        let stamp = "20260819T101500123"
        let run = "\(stamp)-exp-alpha-run"
        let harness = try harness("drift") { _, _ in
            WorkspaceRunImport.Engine(
                listRemoteDirectories: { [run] },
                remoteInventory: { _ in
                    [run: [WorkspaceImportPolicy.FileStat(relativePath: "g.jsonl", size: 4096)]]
                },
                transfer: { _, _ in Issue.record("a drifted directory must not transfer") },
                localExists: { _ in true },
                localInventory: { _ in
                    [WorkspaceImportPolicy.FileStat(relativePath: "g.jsonl", size: 2048)]
                })
        }
        defer { try? FileManager.default.removeItem(at: harness.root) }

        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .importRuns, siteReference: try siteID(harness)))
        #expect(outcome.envelope.state == "failed")
        #expect(outcome.exitCode == 70)
        #expect(outcome.envelope.error?.code == "importIncomplete")
        #expect(
            outcome.envelope.error?.repairAction.contains("never a re-run of this verb") == true)
    }

    /// A setup refusal (no ssh transport, no declared run root, a workspace
    /// that is not a cluster workspace) is a typed 64, with the repair.
    @Test func setupRefusalsSurfaceAsTypedBlockedEnvelopes() async throws {
        let harness = try harness("setup") { record, _ in
            throw WorkspaceRunImport.SetupError.noRemoteRunRoot(siteID: record.id)
        }
        defer { try? FileManager.default.removeItem(at: harness.root) }

        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .importRuns, siteReference: try siteID(harness)))
        #expect(outcome.envelope.state == "blocked")
        #expect(outcome.exitCode == 64)
        #expect(outcome.envelope.error?.code == "noRemoteRunRoot")
        #expect(outcome.envelope.error?.repairAction.contains("storage roots") == true)
    }

    /// An unknown site is the pre-existing typed refusal, unchanged.
    @Test func anUnknownSiteRefuses() async throws {
        let harness = try harness("unknown-site") { _, _ in
            Issue.record("the engine must not be built for an unknown site")
            return WorkspaceRunImport.Engine(
                listRemoteDirectories: { [] }, remoteInventory: { _ in [:] },
                transfer: { _, _ in }, localExists: { _ in false },
                localInventory: { _ in [] })
        }
        defer { try? FileManager.default.removeItem(at: harness.root) }

        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .importRuns, siteReference: "no-such-site"))
        #expect(outcome.exitCode != 0)
        #expect(outcome.envelope.error?.code == "unknownSite")
    }
}

/// Records what the fake engine was asked to transfer.
private final class TransferLog: @unchecked Sendable {
    // @unchecked Sendable: appended only from the single operation under test,
    // read after it completes; never escapes the test.
    private(set) var names: [String] = []
    func record(_ name: String) { names.append(name) }
}

/// The verb must never reach the process seam when its engine is injected.
private struct NeverRunImportShell: ClusterShellRunner {
    func run(_ argv: [String]) async -> ClusterShellResult {
        Issue.record("no command may run: \(argv.joined(separator: " "))")
        return ClusterShellResult(exitCode: 1)
    }
}

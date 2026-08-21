import Foundation
import Testing

@testable import ExperimentKit

/// `remote import-chain` — the headless chain importer. Resolution follows
/// the ledger rule (completed disposition only, never name/timestamp among
/// siblings), directories import skip-if-present so a second invocation is
/// a no-op, the server's structured failure-record skip is a note rather
/// than an error, and revision adoption runs for every imported directory.
/// Everything runs against the injected Engine seams — no networking, no
/// real bundles.
struct EvidenceChainImportTests {

    // MARK: Fixtures

    private func row(
        run: String, experiment: String?, disposition: String?,
        updatedAt: String? = nil,
        parked: ClusterClient.PipelineRunSummary.Parked? = nil,
        stageRuns: [(stage: String, status: String, runID: String?)] = []
    ) -> ClusterClient.PipelineRunSummary {
        ClusterClient.PipelineRunSummary(
            run: run, schema: 2, experiment: experiment,
            disposition: disposition, parked: parked,
            epochDriftAtContinuation: nil, manifestStatus: "frozen",
            experimentHash: nil, updatedAt: updatedAt,
            stages: stageRuns.map {
                .init(stage: $0.stage, status: $0.status, runID: $0.runID)
            },
            abort: nil, promotedAgents: nil)
    }

    /// A recording fake engine over an in-memory "local runs tree".
    /// `embeds[runID]` lists sibling directories that materialize when that
    /// bundle imports — the embedded-stage behavior observed live 2026-08-12.
    private final class Harness: @unchecked Sendable {
        private let lock = NSLock()
        var present: Set<String> = []
        var packaged: [String] = []
        var downloaded: [String] = []
        var adopted: [String] = []
        var skipRecords: [String: String] = [:]
        var embeds: [String: [String]] = [:]
        var adoptionOutcome: EvidenceRevisionAdoption.Outcome = .noEvidenceRevision
        /// A dead endpoint (open-issues §3): the pre-flight probe throws this.
        var probeError: Error?
        /// A transport failure raised by the download of these run ids.
        var downloadErrors: [String: Error] = [:]
        var probes = 0

        init(present: Set<String> = []) { self.present = present }

        private func withLock<T>(_ body: () -> T) -> T {
            lock.lock()
            defer { lock.unlock() }
            return body()
        }

        func engine() -> EvidenceChainImport.Engine {
            EvidenceChainImport.Engine(
                packageEvidence: { runID in
                    let skip: String? = self.withLock {
                        self.packaged.append(runID)
                        return self.skipRecords[runID]
                    }
                    if let skip {
                        return ClusterClient.EvidencePackageReceipt(
                            bundlePath: nil, bundleSha256: nil, runID: runID,
                            evidenceComplete: nil, missingEvidence: nil,
                            skipped: true, reason: skip)
                    }
                    return ClusterClient.EvidencePackageReceipt(
                        bundlePath: "/srv/runs/\(runID)/\(runID).evidence-bundle.tar.gz",
                        bundleSha256: "sha-\(runID)", runID: runID,
                        evidenceComplete: true, missingEvidence: nil)
                },
                downloadAndImport: { bundlePath, _ in
                    let runID = URL(filePath: bundlePath)
                        .deletingLastPathComponent().lastPathComponent
                    if let failure = self.withLock({
                        self.downloadErrors[runID]
                    }) {
                        throw failure
                    }
                    self.withLock {
                        self.downloaded.append(runID)
                        self.present.insert(runID)
                        for embedded in self.embeds[runID] ?? [] {
                            self.present.insert(embedded)
                        }
                    }
                    return URL(filePath: "/local/runs/\(runID)")
                },
                localRunExists: { runID in
                    self.withLock { self.present.contains(runID) }
                },
                adoptRevision: { imported in
                    self.withLock {
                        self.adopted.append(imported.lastPathComponent)
                        return self.adoptionOutcome
                    }
                },
                probeEndpoint: {
                    let failure: Error? = self.withLock {
                        self.probes += 1
                        return self.probeError
                    }
                    if let failure { throw failure }
                })
        }
    }

    // MARK: Transport failure is not allowed to touch the workspace (§3)

    /// The 2026-08-15 incident: a VPN drop leaves the SSH forward configured
    /// with nothing listening behind it. The verb used to walk the whole chain
    /// against that dead endpoint — one failed attempt per directory, each a
    /// chance to leave a 0-file run directory that passes exists-checks. It
    /// now refuses ONCE, before anything is packaged or downloaded.
    @Test func anUnreachableEndpointRefusesBeforeTouchingAnything() async {
        let harness = Harness()
        harness.probeError = URLError(.cannotConnectToHost)
        let chain = EvidenceChainImport.ResolvedChain(
            pipelineRunID: "p1", stageRunIDs: ["s1", "s2"])
        let reports = await EvidenceChainImport.importChain(
            chain, engine: harness.engine())

        #expect(reports.count == 1)
        #expect(reports[0].runID == "p1")
        #expect(reports[0].isFailure)
        guard case .endpointUnreachable(let message) = reports[0].outcome else {
            Issue.record("expected an endpointUnreachable outcome")
            return
        }
        // The refusal must name the remedy the operations rule names.
        #expect(message.contains("cluster tunnel open"))
        #expect(message.contains("Nothing was written"))
        // Nothing was asked of the server, and nothing entered the local tree.
        #expect(harness.packaged.isEmpty)
        #expect(harness.downloaded.isEmpty)
        #expect(harness.present.isEmpty)
        #expect(harness.probes == 1)
    }

    @Test func theProbeRunsOnceForTheWholeChain() async {
        let harness = Harness()
        let chain = EvidenceChainImport.ResolvedChain(
            pipelineRunID: "p1", stageRunIDs: ["s1", "s2"])
        _ = await EvidenceChainImport.importChain(chain, engine: harness.engine())
        #expect(harness.probes == 1)
        #expect(harness.downloaded == ["p1", "s1", "s2"])
    }

    /// A transport failure that happens AFTER the probe (the endpoint died
    /// mid-chain) must not mark the directory present, and must not stop the
    /// remaining stages from being attempted — re-running the verb is the
    /// documented recovery, and it can only work if the failed directory is
    /// genuinely absent.
    @Test func aMidChainTransportFailureLeavesThatDirectoryAbsent() async {
        let harness = Harness()
        harness.downloadErrors["s1"] = URLError(.networkConnectionLost)
        let chain = EvidenceChainImport.ResolvedChain(
            pipelineRunID: "p1", stageRunIDs: ["s1", "s2"])
        let reports = await EvidenceChainImport.importChain(
            chain, engine: harness.engine())

        #expect(reports.count == 3)
        #expect(reports[1].runID == "s1")
        #expect(reports[1].isFailure)
        #expect(!harness.present.contains("s1"))
        // The directories either side still came home.
        #expect(harness.present.contains("p1"))
        #expect(harness.present.contains("s2"))
    }

    @Test func theRefusalRendersAsARefusalNotAFailure() {
        let lines = EvidenceChainImport.summaryLines([
            .init(runID: "p1", isPipelineDirectory: true,
                  outcome: .endpointUnreachable(message: "the server did not answer")),
        ])
        #expect(lines.contains { $0.contains("REFUSED") })
        #expect(lines.last == "FAILED 1")
    }

    // MARK: Resolution — the ledger rule

    @Test func experimentNameResolvesTheNewestCompletedChainOnly() throws {
        let rows = [
            // The newest SIBLING by name and timestamp is unfinished — the
            // ledger rule must never pick it.
            row(run: "20260812-c20-pipeline", experiment: "c20",
                disposition: nil, updatedAt: "2026-08-12T10:00:00Z",
                stageRuns: [("run", "started", "20260812-c20-run")]),
            row(run: "20260810-c20-pipeline", experiment: "c20",
                disposition: "completed", updatedAt: "2026-08-10T09:00:00Z",
                stageRuns: [
                    ("run", "completed", "20260810-c20-run"),
                    ("evaluate", "completed", "20260810-c20-evaluate"),
                    // Duplicate + self references collapse out of the plan.
                    ("analyze", "completed", "20260810-c20-run"),
                    ("promote", "completed", "20260810-c20-pipeline"),
                ]),
            row(run: "20260808-c20-pipeline", experiment: "c20",
                disposition: "completed", updatedAt: "2026-08-08T09:00:00Z",
                stageRuns: [("run", "completed", "20260808-c20-run")]),
            row(run: "20260811-other-pipeline", experiment: "other",
                disposition: "completed", updatedAt: "2026-08-11T09:00:00Z"),
        ]
        let chain = try EvidenceChainImport.resolveChain(
            argument: "c20", rows: rows)
        // Newest COMPLETED — not the newer unfinished sibling, not another
        // experiment's chain.
        #expect(chain.pipelineRunID == "20260810-c20-pipeline")
        #expect(chain.stageRunIDs == ["20260810-c20-run", "20260810-c20-evaluate"])
    }

    @Test func anExplicitRunIDResolvesEvenWhenNotCompleted() throws {
        let rows = [
            row(run: "20260812-c20-pipeline", experiment: "c20",
                disposition: nil,
                parked: .init(at: "2026-08-12T10:00:00Z", by: "reconcile",
                              reason: "orphaned", completedStages: ["run"],
                              remainingStages: ["analyze"]),
                stageRuns: [("run", "completed", "20260812-c20-run")])
        ]
        // The researcher named THAT chain — its completed stages import.
        let chain = try EvidenceChainImport.resolveChain(
            argument: "20260812-c20-pipeline", rows: rows)
        #expect(chain.pipelineRunID == "20260812-c20-pipeline")
        #expect(chain.stageRunIDs == ["20260812-c20-run"])
    }

    @Test func refusesNamingEveryCandidateWhenNoSiblingIsCompleted() {
        let rows = [
            row(run: "20260812-c20-pipeline", experiment: "c20",
                disposition: nil,
                parked: .init(at: nil, by: nil, reason: "orphaned",
                              completedStages: nil, remainingStages: nil)),
            row(run: "20260811-c20-pipeline", experiment: "c20",
                disposition: "aborted"),
        ]
        do {
            _ = try EvidenceChainImport.resolveChain(argument: "c20", rows: rows)
            Issue.record("expected the ledger-rule refusal")
        } catch {
            let message = String(describing: error)
            #expect(message.contains("COMPLETED"))
            // The refusal names each candidate and its state.
            #expect(message.contains("20260812-c20-pipeline"))
            #expect(message.contains("parked"))
            #expect(message.contains("20260811-c20-pipeline"))
            #expect(message.contains("aborted (gate)"))
        }
    }

    @Test func refusesClearlyWhenNothingMatchesTheArgument() {
        do {
            _ = try EvidenceChainImport.resolveChain(argument: "nope", rows: [])
            Issue.record("expected a refusal")
        } catch {
            #expect(String(describing: error).contains("no pipeline run"))
        }
    }

    // MARK: Import — embedded-stage dedupe and idempotency

    @Test func pipelineFirstEmbeddedStagesAreNeverDownloadedTwice() async {
        let harness = Harness()
        // The pipeline bundle EMBEDS its stage dirs (observed live
        // 2026-08-12): importing it materializes run + analyze too.
        harness.embeds["P"] = ["R", "A"]
        let chain = EvidenceChainImport.ResolvedChain(
            pipelineRunID: "P", stageRunIDs: ["R", "A"])

        let first = await EvidenceChainImport.importChain(
            chain, engine: harness.engine())
        #expect(first.map(\.outcome) == [
            .imported(adoptionNotice: nil), .alreadyPresent, .alreadyPresent,
        ])
        #expect(first.map(\.isPipelineDirectory) == [true, false, false])
        // One packaging round-trip, one download — the stages came along.
        #expect(harness.packaged == ["P"])
        #expect(harness.downloaded == ["P"])

        // Second invocation: pure no-op — every row "already present",
        // no further server calls. This is what the old refuse-and-abort
        // collision behavior made impossible.
        let second = await EvidenceChainImport.importChain(
            chain, engine: harness.engine())
        #expect(second.allSatisfy { $0.outcome == .alreadyPresent })
        #expect(harness.packaged == ["P"])
        #expect(harness.downloaded == ["P"])
    }

    @Test func aPartialImportFinishesOnRerunWithoutOverwriting() async {
        // The pipeline dir came home earlier; one stage did not (a
        // non-embedding bundle, or an interrupted first pass).
        let harness = Harness(present: ["P", "R"])
        let chain = EvidenceChainImport.ResolvedChain(
            pipelineRunID: "P", stageRunIDs: ["R", "A"])
        let reports = await EvidenceChainImport.importChain(
            chain, engine: harness.engine())
        #expect(reports.map(\.outcome) == [
            .alreadyPresent, .alreadyPresent, .imported(adoptionNotice: nil),
        ])
        // Only the missing directory touched the server.
        #expect(harness.packaged == ["A"])
        #expect(harness.downloaded == ["A"])
    }

    // MARK: Failure records are notes, not errors

    @Test func aFailureRecordSkipSurfacesAsANote() async {
        let harness = Harness()
        harness.skipRecords["A"] =
            "ledger-only failure record from a refused continuation"
        let chain = EvidenceChainImport.ResolvedChain(
            pipelineRunID: "P", stageRunIDs: ["A"])
        let reports = await EvidenceChainImport.importChain(
            chain, engine: harness.engine())
        #expect(reports[0].outcome == .imported(adoptionNotice: nil))
        #expect(reports[1].outcome == .skippedFailureRecord(
            note: "ledger-only failure record from a refused continuation"))
        // A note is never a failure — the verb must not exit non-zero on it.
        #expect(reports.allSatisfy { !$0.isFailure })
        let summary = EvidenceChainImport.summaryLines(reports).joined(separator: "\n")
        #expect(summary.contains("failure record"))
        #expect(summary.contains("refused continuation"))
        #expect(!summary.contains("FAILED"))
    }

    // MARK: Adoption runs per imported directory

    @Test func revisionAdoptionRunsForEveryImportedDirectoryOnly() async {
        let harness = Harness(present: ["R"])
        harness.skipRecords["S"] = "nothing to bundle"
        let chain = EvidenceChainImport.ResolvedChain(
            pipelineRunID: "P", stageRunIDs: ["R", "S", "A"])
        let reports = await EvidenceChainImport.importChain(
            chain, engine: harness.engine())
        // Imported: P and A. Already present (R) and skipped (S) never
        // reach adoption — there is no fresh snapshot to reconcile.
        #expect(harness.adopted == ["P", "A"])
        #expect(reports.count == 4)
    }

    @Test func adoptionNoticesRideTheImportRow() async {
        let harness = Harness()
        harness.adoptionOutcome = .adopted(
            experiment: "c20", revision: "abcdef123456")
        let chain = EvidenceChainImport.ResolvedChain(
            pipelineRunID: "P", stageRunIDs: [])
        let reports = await EvidenceChainImport.importChain(
            chain, engine: harness.engine())
        guard case .imported(let notice) = reports[0].outcome else {
            Issue.record("expected an import")
            return
        }
        #expect(notice?.contains("abcdef123456") == true)
        #expect(EvidenceChainImport.summaryLines(reports)
            .joined(separator: "\n").contains("abcdef123456"))
    }

    // MARK: Output redaction — --site resolution stays unprintable

    @Test func summaryLinesCarryRunOutcomesOnlyNeverEndpointOrToken() async {
        // The engine's inputs deliberately carry endpoint- and token-shaped
        // strings; the summary is derived from run ids and outcomes only.
        let harness = Harness()
        let chain = EvidenceChainImport.ResolvedChain(
            pipelineRunID: "P", stageRunIDs: ["R"])
        let reports = await EvidenceChainImport.importChain(
            chain, engine: harness.engine())
        let summary = EvidenceChainImport.summaryLines(reports).joined(separator: "\n")
        #expect(!summary.contains("/srv/runs"))
        #expect(!summary.contains("sha-"))
        #expect(!summary.contains("127.0.0.1"))

        // The verb reaches the server through the SAME shared site
        // resolution as every other `remote` verb; the only printable
        // artifact of that resolution is presence-and-provenance.
        let resolution = ClusterRemoteSiteResolution(
            siteID: "example-hpc", siteName: "Example HPC",
            baseURL: URL(string: "http://127.0.0.1:8718")!,
            token: "sk-CHAIN-SECRET", tokenSource: "keychain")
        #expect(!resolution.redactedSummary.contains("sk-CHAIN-SECRET"))
        #expect(resolution.redactedSummary.contains("keychain"))
        // --site and --url stay mutually exclusive on this verb too.
        #expect(throws: ClusterCLIError.self) {
            _ = try ClusterRemoteSiteResolver.choose(site: "a", url: "b")
        }
    }
}

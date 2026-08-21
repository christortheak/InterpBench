import Foundation

// =============================================================================
// `remote import-chain` (2026-08-12) — bring a whole pipeline chain's evidence
// home, headlessly and RE-RUNNABLY.
//
// The app's dead-pipeline affordance (`EvidenceAutoImportService
// .importPipeline`) already packages a chain ON the server and imports the
// bundle; this is its CLI twin, extended with what a terminal session needs:
//
//  - resolution by EXPERIMENT NAME under the ledger rule (the study
//    workspace's RUNS-GUIDE.md): among sibling pipeline runs, the COMPLETED
//    disposition chooses — never the newest name or timestamp among
//    unfinished siblings. When no sibling is completed, the refusal names
//    every candidate and its state instead of guessing.
//  - SKIP-IF-PRESENT semantics per run directory: a run already in the local
//    workspace reports "already present" and is never overwritten. The
//    importer's refuse-and-abort collision behavior made re-running a verb
//    after a partial import impossible; here, re-running IS the recovery.
//  - the structured failure-record skip from `POST /api/bundles/evidence`
//    (2026-08-11: a refused continuation's ledger-only pipeline dir has
//    nothing to bundle) surfaces as a NOTE, never an error.
//  - `EvidenceRevisionAdoption` runs for every imported directory — the
//    reconciliation every import path must perform (2026-08-06, a
//    replication-run incident).
//
// Pipeline evidence bundles may EMBED their stage directories (observed live
// 2026-08-12: importing the pipeline bundle materialized run + analyze too),
// so the chain imports pipeline-first and re-checks local presence before
// each stage — an embedded stage reports "already present" without a second
// packaging round-trip or download.
// =============================================================================

public enum EvidenceChainImport {

    // MARK: - Chain resolution (pure, unit-tested)

    /// The run directories one chain import will visit: the pipeline ledger
    /// directory first, then its stage run directories in ledger order.
    public struct ResolvedChain: Sendable, Equatable {
        public var pipelineRunID: String
        /// Stage run ids from the ledger's `stageResults`, deduped in order,
        /// never containing the pipeline directory itself.
        public var stageRunIDs: [String]

        public init(pipelineRunID: String, stageRunIDs: [String]) {
            self.pipelineRunID = pipelineRunID
            self.stageRunIDs = stageRunIDs
        }
    }

    /// Resolve the argument against the server's pipeline listing. An exact
    /// pipeline-run-id match wins regardless of disposition (the researcher
    /// named THAT chain — a parked chain's completed stages are importable).
    /// An experiment name resolves to its NEWEST COMPLETED chain by the
    /// ledger's `updatedAt` stamp; anything less than completed refuses,
    /// naming the candidates.
    public static func resolveChain(
        argument: String, rows: [ClusterClient.PipelineRunSummary]
    ) throws -> ResolvedChain {
        if let row = rows.first(where: { $0.run == argument }) {
            return chain(from: row)
        }
        let siblings = rows.filter { $0.experiment == argument }
        guard !siblings.isEmpty else {
            throw ExperimentError(
                reason: "no pipeline run named '\(argument)' and no pipeline "
                    + "runs for an experiment of that name on the server — "
                    + "pass a pipeline run id or an experiment with "
                    + "server-side pipeline runs")
        }
        let completed = siblings.filter { $0.disposition == "completed" }
        guard !completed.isEmpty else {
            let candidates = siblings
                .map { "  \($0.run) — \($0.stateLabel)" }
                .joined(separator: "\n")
            throw ExperimentError(
                reason: "no COMPLETED pipeline run for experiment "
                    + "'\(argument)' — the ledger rule selects by completed "
                    + "disposition only, never by name or timestamp among "
                    + "siblings. Candidates:\n\(candidates)\n"
                    + "Resume the chain to completion, or pass a run id "
                    + "explicitly to import a partial chain's completed "
                    + "stages.")
        }
        // Newest among the COMPLETED — `updatedAt` is the ledger's last
        // write (UTC ISO, so lexicographic order is chronological); the run
        // id breaks ties. Name/timestamp never arbitrates between a
        // completed chain and an unfinished one.
        let newest = completed.max {
            ($0.updatedAt ?? "", $0.run) < ($1.updatedAt ?? "", $1.run)
        }!
        return chain(from: newest)
    }

    private static func chain(
        from row: ClusterClient.PipelineRunSummary
    ) -> ResolvedChain {
        var seen: Set<String> = [row.run]
        var stageIDs: [String] = []
        for stage in row.stages {
            guard let runID = stage.runID, seen.insert(runID).inserted else {
                continue
            }
            stageIDs.append(runID)
        }
        return ResolvedChain(pipelineRunID: row.run, stageRunIDs: stageIDs)
    }

    // MARK: - Per-directory outcomes

    public enum Outcome: Sendable, Equatable {
        /// Downloaded, hash-verified, imported; revision adoption ran and
        /// its user-visible notice (if any) rides along.
        case imported(adoptionNotice: String?)
        /// The run directory already exists locally (an earlier import, a
        /// paired tree, or a stage the pipeline bundle embedded) — nothing
        /// downloaded, nothing overwritten.
        case alreadyPresent
        /// The server's structured skip: a ledger-only failure record with
        /// nothing to bundle. A note, never an error.
        case skippedFailureRecord(note: String)
        /// The endpoint did not answer the pre-flight probe, so the chain was
        /// REFUSED before anything was created locally (open-issues §3). A
        /// stale tunnel — the forward still configured, nothing listening
        /// behind it after a VPN drop — is the case this exists for.
        case endpointUnreachable(message: String)
        case failed(message: String)
    }

    public struct DirectoryReport: Sendable, Equatable {
        public var runID: String
        /// True for the pipeline ledger directory itself.
        public var isPipelineDirectory: Bool
        public var outcome: Outcome

        public var isFailure: Bool {
            switch outcome {
            case .failed, .endpointUnreachable: return true
            default: return false
            }
        }

        public init(
            runID: String, isPipelineDirectory: Bool, outcome: Outcome
        ) {
            self.runID = runID
            self.isPipelineDirectory = isPipelineDirectory
            self.outcome = outcome
        }
    }

    // MARK: - Import engine (seams injected so tests reach the logic)

    public struct Engine: Sendable {
        public var packageEvidence:
            @Sendable (String) async throws -> ClusterClient.EvidencePackageReceipt
        /// (bundlePath, serverStampedSHA256) → imported local run directory.
        public var downloadAndImport: @Sendable (String, String?) async throws -> URL
        public var localRunExists: @Sendable (String) -> Bool
        public var adoptRevision:
            @Sendable (URL) -> EvidenceRevisionAdoption.Outcome
        /// One cheap read against the endpoint, run BEFORE the chain touches
        /// anything local (open-issues §3). Throwing here refuses the whole
        /// import; the default no-op keeps hand-built engines source-compatible.
        public var probeEndpoint: @Sendable () async throws -> Void

        public init(
            packageEvidence: @escaping @Sendable (String) async throws
                -> ClusterClient.EvidencePackageReceipt,
            downloadAndImport: @escaping @Sendable (String, String?) async throws -> URL,
            localRunExists: @escaping @Sendable (String) -> Bool,
            adoptRevision: @escaping @Sendable (URL)
                -> EvidenceRevisionAdoption.Outcome,
            probeEndpoint: @escaping @Sendable () async throws -> Void = {}
        ) {
            self.packageEvidence = packageEvidence
            self.downloadAndImport = downloadAndImport
            self.localRunExists = localRunExists
            self.adoptRevision = adoptRevision
            self.probeEndpoint = probeEndpoint
        }
    }

    /// The refusal text for an endpoint that did not answer. Names the remedy
    /// the operations rule names — `cluster tunnel open` — because a stale
    /// tunnel after a VPN drop is the observed cause (2026-08-15). Twin of
    /// `ClusterConnectionStore.friendlyConnectionFailure`, kept here so the
    /// headless path needs no main-actor state.
    public static func unreachableMessage(_ error: Error) -> String {
        let cause: String
        if let urlError = error as? URLError {
            switch urlError.code {
            case .cannotConnectToHost, .cannotFindHost:
                cause = "nothing is listening behind the endpoint"
            case .timedOut:
                cause = "the endpoint timed out"
            case .notConnectedToInternet, .networkConnectionLost:
                cause = "the network dropped"
            default:
                cause = urlError.localizedDescription
            }
        } else {
            cause = String(describing: error)
        }
        return "the server did not answer — \(cause). Nothing was written: no "
            + "run directory was created, so no empty shell can block a "
            + "re-import. If the VPN reconnected, the SSH forward is stale even "
            + "though it looks configured — run `steerlab-cli cluster tunnel "
            + "open` and import again."
    }

    /// The live engine: the same verified path the auto-import uses —
    /// server-side packaging, download, `EvidenceBundleImporter` (per-file
    /// hash verification), `EvidenceRevisionAdoption`.
    public static func liveEngine(
        client: ClusterClient, workspaceRoot: URL
    ) -> Engine {
        Engine(
            packageEvidence: { runID in
                try await client.packageEvidence(runDirectory: runID)
            },
            downloadAndImport: { bundlePath, sha256 in
                let downloads = workspaceRoot.appending(
                    components: ".steerlab", "downloads")
                let local = try await client.downloadArtifact(
                    path: bundlePath, to: downloads)
                return try EvidenceBundleImporter.importEvidenceBundle(
                    local, expectedSHA256: sha256)
            },
            localRunExists: { runID in
                var isDirectory: ObjCBool = false
                let url = ExperimentStore.runsDirectory.appending(component: runID)
                return FileManager.default.fileExists(
                    atPath: url.path, isDirectory: &isDirectory)
                    && isDirectory.boolValue
            },
            adoptRevision: { imported in
                EvidenceRevisionAdoption.adoptModelRevision(
                    fromImportedRun: imported)
            },
            probeEndpoint: {
                // The cheapest authenticated read the server offers. It runs
                // once, before the chain touches the workspace, so a stale
                // tunnel refuses instead of erroring N times with N chances to
                // leave a partial directory behind (open-issues §3).
                _ = try await client.capabilities()
            })
    }

    /// Import the chain: pipeline directory first (its bundle may embed the
    /// stage directories), then each stage — every directory re-checked for
    /// local presence at its turn, so nothing is downloaded twice and a
    /// second invocation over a fully imported chain is a pure no-op.
    public static func importChain(
        _ chain: ResolvedChain, engine: Engine
    ) async -> [DirectoryReport] {
        // Refuse LOUDLY, creating nothing, when the endpoint is unreachable
        // (open-issues §3). Without this the verb walked the whole chain
        // against a dead tunnel, erroring once per directory — N chances for a
        // download to half-land, and N failure rows hiding one cause.
        do {
            try await engine.probeEndpoint()
        } catch {
            return [DirectoryReport(
                runID: chain.pipelineRunID, isPipelineDirectory: true,
                outcome: .endpointUnreachable(
                    message: unreachableMessage(error)))]
        }
        var reports: [DirectoryReport] = []
        reports.append(await importDirectory(
            chain.pipelineRunID, isPipeline: true, engine: engine))
        for stageID in chain.stageRunIDs {
            reports.append(await importDirectory(
                stageID, isPipeline: false, engine: engine))
        }
        return reports
    }

    private static func importDirectory(
        _ runID: String, isPipeline: Bool, engine: Engine
    ) async -> DirectoryReport {
        func report(_ outcome: Outcome) -> DirectoryReport {
            DirectoryReport(
                runID: runID, isPipelineDirectory: isPipeline, outcome: outcome)
        }
        guard !engine.localRunExists(runID) else {
            return report(.alreadyPresent)
        }
        do {
            let receipt = try await engine.packageEvidence(runID)
            if receipt.skipped == true {
                return report(.skippedFailureRecord(
                    note: receipt.reason
                        ?? "the server reported a failure record with "
                        + "nothing to bundle"))
            }
            guard let bundlePath = receipt.bundlePath else {
                return report(.failed(
                    message: "the server packaged '\(runID)' but returned "
                        + "no bundle path"))
            }
            let imported = try await engine.downloadAndImport(
                bundlePath, receipt.bundleSha256)
            let adoption = engine.adoptRevision(imported)
            let notice = EvidenceRevisionAdoption.notice(for: adoption).map {
                ($0.isWarning ? "warning: " : "") + $0.message
            }
            return report(.imported(adoptionNotice: notice))
        } catch {
            let message = String(describing: error)
            if message.contains("refusing to overwrite existing run") {
                // The verified importer's collision refusal: the run became
                // durable locally between our presence check and the import
                // (e.g. embedded in a bundle imported this same pass) —
                // that is "already present", not a failure.
                return report(.alreadyPresent)
            }
            return report(.failed(message: message))
        }
    }

    // MARK: - Summary rendering (pure, unit-tested)

    /// Per-directory rows plus a totals line — derived from run ids and
    /// outcomes ONLY, so no endpoint, token, or server path can leak into
    /// the verb's stdout.
    public static func summaryLines(_ reports: [DirectoryReport]) -> [String] {
        var lines: [String] = []
        var imported = 0
        var present = 0
        var skipped = 0
        var failed = 0
        for report in reports {
            let role = report.isPipelineDirectory ? "pipeline" : "stage   "
            switch report.outcome {
            case .imported(let adoptionNotice):
                imported += 1
                lines.append("\(role)  \(report.runID)  imported")
                if let adoptionNotice {
                    lines.append("          \(adoptionNotice)")
                }
            case .alreadyPresent:
                present += 1
                lines.append("\(role)  \(report.runID)  already present")
            case .skippedFailureRecord(let note):
                skipped += 1
                lines.append(
                    "\(role)  \(report.runID)  skipped — failure record: \(note)")
            case .endpointUnreachable(let message):
                failed += 1
                lines.append("\(role)  \(report.runID)  REFUSED — \(message)")
            case .failed(let message):
                failed += 1
                lines.append("\(role)  \(report.runID)  FAILED — \(message)")
            }
        }
        var totals: [String] = []
        if imported > 0 { totals.append("imported \(imported)") }
        if present > 0 { totals.append("already present \(present)") }
        if skipped > 0 {
            totals.append(
                "skipped \(skipped) failure record\(skipped == 1 ? "" : "s")")
        }
        if failed > 0 { totals.append("FAILED \(failed)") }
        lines.append(
            totals.isEmpty ? "nothing to import" : totals.joined(separator: " · "))
        return lines
    }
}

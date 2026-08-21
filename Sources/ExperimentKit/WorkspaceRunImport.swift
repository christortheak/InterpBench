import CryptoKit
import Foundation

// =============================================================================
// The workspace import operation (open-issues §20): bring a paired cluster
// workspace's run directories home, per `WorkspaceImportPolicy`, verify them by
// CONTENT, rebuild the catalog, and report what scratch may now drop.
//
// Design commitments, each one a researcher tightening of 2026-08-20:
//
//   1. The purge report's shard-partial gate reads merge EVIDENCE — the
//      `sharded` completeness stamp a merged run's `report.json` carries —
//      never the existence of a merged-looking directory. An orphan is loud
//      and is never eligible.
//   2. Verification is stat-level on BOTH sides (file counts PLUS per-file
//      sizes), with hash checks wherever a pin already exists.
//   3. IMPORT NEVER PURGES. This verb emits an eligibility REPORT; deletion is
//      a separate deliberate act that nothing here performs.
//   4. Re-import is idempotent: gaps are filled, and remote bytes that DIFFER
//      from already-imported local bytes are an immutability violation that
//      REFUSES the directory. Nothing is ever overwritten.
//   6. Transfer rides the existing ssh/rsync machinery (`ClusterShellRunner`
//      plus `ClusterProvisioner`'s argv builders), not the HTTP API: run
//      directories are GB-scale, and the API path exists for single evidence
//      bundles.
//
// Structure mirrors `EvidenceChainImport`: pure rules plus an `Engine` of
// injected seams, so the whole operation is exercised in tests with no ssh, no
// network, and no cluster.
// =============================================================================

public enum WorkspaceRunImport {

    // MARK: - Per-directory outcome

    public enum Outcome: Sendable, Equatable {
        /// Transferred (in full, or filling gaps in a partial earlier import),
        /// then verified.
        case imported(files: Int, bytes: Int64)
        /// Already present and verified complete — nothing transferred.
        case alreadyComplete(files: Int)
        /// A shard partial: never imported. Its family's purge eligibility is
        /// decided by the evidence gate, in the purge report.
        case skippedShardPartial(shardIndex: Int?, shardCount: Int?)
        /// Not a run directory (library subtree, stray entry).
        case notApplicable(reason: String)
        /// Filtered out by `--since`.
        case outsideWindow(stamp: String?)
        /// Tightening 4: the remote bytes differ from local bytes already
        /// imported. Refused, nothing overwritten.
        case refusedByteDrift(message: String)
        /// Transferred, but the post-transfer verification still disagrees —
        /// reported, never silently accepted.
        case verificationFailed(findings: [String])
        case failed(message: String)

        public var isFailure: Bool {
            switch self {
            case .refusedByteDrift, .verificationFailed, .failed: true
            default: false
            }
        }
    }

    public struct DirectoryReport: Sendable, Equatable {
        public var name: String
        public var kind: WorkspaceImportPolicy.DirectoryKind
        public var outcome: Outcome
        /// Paths the policy's NEVER rules kept on the far side.
        public var excludedByPolicy: [String]

        public init(
            name: String, kind: WorkspaceImportPolicy.DirectoryKind,
            outcome: Outcome, excludedByPolicy: [String] = []
        ) {
            self.name = name
            self.kind = kind
            self.outcome = outcome
            self.excludedByPolicy = excludedByPolicy
        }
    }

    /// One family's purge verdict, ready to print.
    public struct PurgeRow: Sendable, Equatable {
        public var family: WorkspaceImportPolicy.ShardFamily
        public var verdict: WorkspaceImportPolicy.ShardFamilyVerdict
        public var message: String

        public init(
            family: WorkspaceImportPolicy.ShardFamily,
            verdict: WorkspaceImportPolicy.ShardFamilyVerdict
        ) {
            self.family = family
            self.verdict = verdict
            self.message = WorkspaceImportPolicy.message(
                family: family, verdict: verdict)
        }
    }

    /// Non-shard bytes the policy declined to import and that scratch may
    /// therefore drop.
    public struct PurgeablePathRow: Sendable, Equatable {
        public var directory: String
        public var rule: WorkspaceImportPolicy.ExclusionRule
        public var paths: [String]
        public var bytes: Int64

        public init(
            directory: String, rule: WorkspaceImportPolicy.ExclusionRule,
            paths: [String], bytes: Int64
        ) {
            self.directory = directory
            self.rule = rule
            self.paths = paths
            self.bytes = bytes
        }
    }

    public struct Report: Sendable, Equatable {
        public var directories: [DirectoryReport]
        /// Shape the policy did not recognize — imported anyway, and named
        /// here so the policy can learn them.
        public var unknowns: [String]
        public var purgeFamilies: [PurgeRow]
        public var purgeablePaths: [PurgeablePathRow]
        /// Studies whose live workspace manifest holds fewer arms than their
        /// own imported run evidence (open-issues §8 residual (a)): cluster
        /// authoring that never came home. A loud report — this operation
        /// never writes `experiments/`.
        public var authoringDivergences: [WorkspaceImportPolicy.AuthoringDivergence]
        /// Immutability violations and verification failures, in full.
        public var violations: [String]
        public var catalog: WorkspaceRunCatalog.BuildReport?
        /// True when nothing was written anywhere (a `--dry-run`).
        public var dryRun: Bool

        public init(
            directories: [DirectoryReport] = [], unknowns: [String] = [],
            purgeFamilies: [PurgeRow] = [], purgeablePaths: [PurgeablePathRow] = [],
            authoringDivergences: [WorkspaceImportPolicy.AuthoringDivergence] = [],
            violations: [String] = [],
            catalog: WorkspaceRunCatalog.BuildReport? = nil, dryRun: Bool = false
        ) {
            self.directories = directories
            self.unknowns = unknowns
            self.purgeFamilies = purgeFamilies
            self.purgeablePaths = purgeablePaths
            self.authoringDivergences = authoringDivergences
            self.violations = violations
            self.catalog = catalog
            self.dryRun = dryRun
        }

        public var imported: [DirectoryReport] {
            directories.filter {
                if case .imported = $0.outcome { return true }
                return false
            }
        }

        public var skippedByPolicy: [DirectoryReport] {
            directories.filter {
                switch $0.outcome {
                case .skippedShardPartial, .notApplicable, .outsideWindow: true
                default: false
                }
            }
        }

        public var failures: [DirectoryReport] {
            directories.filter(\.outcome.isFailure)
        }

        /// Whether any family was surfaced loudly (orphan, unstamped merge,
        /// or a stamp that does not cover its partials).
        public var hasLoudPurgeFindings: Bool {
            purgeFamilies.contains { $0.verdict.isLoud }
        }

        /// Whether cluster-side authoring diverges from the live workspace —
        /// as loud as an unmerged fan-out, and equally a human's decision.
        public var hasAuthoringDivergences: Bool {
            !authoringDivergences.isEmpty
        }
    }

    // MARK: - Seams

    /// Everything that touches the far side, the filesystem, or the clock.
    /// Live implementations are built by `liveEngine`; tests inject fakes.
    public struct Engine: Sendable {
        /// One cheap read against the remote before anything local happens —
        /// the refuse-early rule `remote import-chain` learned (open-issues
        /// §3): a stale forward or a dropped VPN must refuse, not fail once
        /// per directory.
        public var probeRemote: @Sendable () async throws -> Void
        /// Top-level directory names under the remote run root.
        public var listRemoteDirectories: @Sendable () async throws -> [String]
        /// Directory names that carry a `shard.json` (corroborates the name
        /// suffix; a partial whose name lost its suffix is still a partial).
        public var remoteShardStamped: @Sendable () async throws -> Set<String>
        /// Per-file inventory (path relative to the run directory, size) for
        /// the named remote directories.
        public var remoteInventory: @Sendable ([String]) async throws
            -> [String: [WorkspaceImportPolicy.FileStat]]
        /// rsync one directory home under the policy's exclusions, filling
        /// gaps only — never overwriting an existing local file.
        public var transfer: @Sendable (String, [WorkspaceImportPolicy.ExclusionRule])
            async throws -> Void
        public var localExists: @Sendable (String) -> Bool
        public var localInventory: @Sendable (String) -> [WorkspaceImportPolicy.FileStat]
        public var pinnedHashes: @Sendable (String) -> [WorkspaceImportPolicy.PinnedHash]
        /// SHA-256 of a local file inside a run directory, or nil when it
        /// cannot be read.
        public var localFileHash: @Sendable (String, String) -> String?
        /// Merge evidence read from the WORKSPACE — stamped merges, and
        /// merged-looking runs that carry no stamp.
        public var workspaceMergeEvidence: @Sendable () -> (
            evidence: [WorkspaceImportPolicy.MergeEvidence],
            unstamped: [WorkspaceImportPolicy.UnstampedMergeCandidate]
        )
        /// The arm counts of a LOCAL run directory's `experiment.json`
        /// snapshot (nil when the directory carries none — submit receipts,
        /// sessions, a dry run's not-yet-imported directory).
        public var localRunManifestArms: @Sendable (String)
            -> WorkspaceImportPolicy.ManifestArms?
        /// The arm counts of the live workspace manifest for a study name
        /// (nil when `experiments/<name>/experiment.json` does not exist).
        public var liveExperimentArms: @Sendable (String)
            -> WorkspaceImportPolicy.ManifestArms?
        public var rebuildCatalog: @Sendable () throws -> WorkspaceRunCatalog.BuildReport

        public init(
            probeRemote: @escaping @Sendable () async throws -> Void = {},
            listRemoteDirectories: @escaping @Sendable () async throws -> [String],
            remoteShardStamped: @escaping @Sendable () async throws -> Set<String> = { [] },
            remoteInventory: @escaping @Sendable ([String]) async throws
                -> [String: [WorkspaceImportPolicy.FileStat]],
            transfer: @escaping @Sendable (String, [WorkspaceImportPolicy.ExclusionRule])
                async throws -> Void,
            localExists: @escaping @Sendable (String) -> Bool,
            localInventory: @escaping @Sendable (String)
                -> [WorkspaceImportPolicy.FileStat],
            pinnedHashes: @escaping @Sendable (String)
                -> [WorkspaceImportPolicy.PinnedHash] = { _ in [] },
            localFileHash: @escaping @Sendable (String, String) -> String? = { _, _ in nil },
            workspaceMergeEvidence: @escaping @Sendable () -> (
                evidence: [WorkspaceImportPolicy.MergeEvidence],
                unstamped: [WorkspaceImportPolicy.UnstampedMergeCandidate]
            ) = { ([], []) },
            localRunManifestArms: @escaping @Sendable (String)
                -> WorkspaceImportPolicy.ManifestArms? = { _ in nil },
            liveExperimentArms: @escaping @Sendable (String)
                -> WorkspaceImportPolicy.ManifestArms? = { _ in nil },
            rebuildCatalog: @escaping @Sendable () throws
                -> WorkspaceRunCatalog.BuildReport = {
                    WorkspaceRunCatalog.BuildReport(
                        rows: [], linkCount: 0, adapterCount: 0, libraryCount: 0,
                        gitignoreUpdated: false)
                }
        ) {
            self.probeRemote = probeRemote
            self.listRemoteDirectories = listRemoteDirectories
            self.remoteShardStamped = remoteShardStamped
            self.remoteInventory = remoteInventory
            self.transfer = transfer
            self.localExists = localExists
            self.localInventory = localInventory
            self.pinnedHashes = pinnedHashes
            self.localFileHash = localFileHash
            self.workspaceMergeEvidence = workspaceMergeEvidence
            self.localRunManifestArms = localRunManifestArms
            self.liveExperimentArms = liveExperimentArms
            self.rebuildCatalog = rebuildCatalog
        }
    }

    public struct Options: Sendable, Equatable {
        /// Normalized run stamp; directories older than it are skipped.
        public var since: String?
        /// Classify, plan, and report — write nothing, transfer nothing.
        public var dryRun: Bool

        public init(since: String? = nil, dryRun: Bool = false) {
            self.since = since
            self.dryRun = dryRun
        }
    }

    // MARK: - The operation

    /// Enumerate, classify, transfer, verify, rebuild the catalog, and report.
    /// `emit` receives progress lines for the Activity feed / the CLI's
    /// stderr; the returned report is the whole answer either way.
    public static func run(
        engine: Engine, options: Options = Options(),
        emit: @Sendable (String) -> Void = { _ in }
    ) async -> Report {
        var report = Report(dryRun: options.dryRun)
        do {
            try await engine.probeRemote()
        } catch {
            report.violations.append(
                "the cluster did not answer — \(String(describing: error)). "
                    + "Nothing was enumerated and nothing was written. If the "
                    + "VPN reconnected, the shared SSH session is stale even "
                    + "though it looks configured: run `steerlab-cli cluster "
                    + "auth open --site <id>` and import again.")
            return report
        }

        let names: [String]
        let shardStamped: Set<String>
        do {
            names = try await engine.listRemoteDirectories()
            shardStamped = try await engine.remoteShardStamped()
        } catch {
            report.violations.append(
                "could not list the remote run root: \(String(describing: error))")
            return report
        }

        // Classify everything first: the plan is a pure function of the names
        // plus the shard-stamp probe, so a dry run and a real run agree by
        // construction.
        let classifications = names.sorted().map { name in
            WorkspaceImportPolicy.classify(
                directoryName: name,
                containsShardStamp: shardStamped.contains(name))
        }
        emit("enumerated \(classifications.count) remote entries")

        var transferable: [WorkspaceImportPolicy.Classification] = []
        var deferredReports: [DirectoryReport] = []
        for classification in classifications {
            let decision = WorkspaceImportPolicy.decision(for: classification)
            guard decision.transfers else {
                let outcome: Outcome =
                    switch decision {
                    case .skipShardPartial(let index, let count):
                        .skippedShardPartial(shardIndex: index, shardCount: count)
                    case .notApplicable(let reason):
                        .notApplicable(reason: reason)
                    default:
                        .notApplicable(reason: "not transferable")
                    }
                deferredReports.append(
                    DirectoryReport(
                        name: classification.name, kind: classification.kind,
                        outcome: outcome))
                continue
            }
            guard
                WorkspaceImportPolicy.passesSince(classification, since: options.since)
            else {
                deferredReports.append(
                    DirectoryReport(
                        name: classification.name, kind: classification.kind,
                        outcome: .outsideWindow(stamp: classification.stamp)))
                continue
            }
            if case .importConservatively = decision {
                report.unknowns.append(classification.name)
            }
            transferable.append(classification)
        }

        let inventories: [String: [WorkspaceImportPolicy.FileStat]]
        do {
            inventories = try await engine.remoteInventory(transferable.map(\.name))
        } catch {
            report.violations.append(
                "could not inventory the remote run directories: "
                    + "\(String(describing: error))")
            report.directories = deferredReports
            return report
        }

        for classification in transferable {
            let rules = WorkspaceImportPolicy.exclusions(for: classification.kind)
            let remote = inventories[classification.name] ?? []
            let excluded = remote.filter {
                WorkspaceImportPolicy.isExcluded(relativePath: $0.relativePath, rules: rules)
            }
            for rule in rules {
                let matching = excluded.filter {
                    WorkspaceImportPolicy.isExcluded(
                        relativePath: $0.relativePath, rules: [rule])
                }
                guard !matching.isEmpty else { continue }
                report.purgeablePaths.append(
                    PurgeablePathRow(
                        directory: classification.name, rule: rule,
                        paths: matching.map(\.relativePath).sorted(),
                        bytes: matching.reduce(0) { $0 + $1.size }))
            }

            let outcome = await importOne(
                classification, remote: remote, rules: rules, engine: engine,
                options: options, emit: emit)
            if case .refusedByteDrift(let message) = outcome {
                report.violations.append(message)
            }
            if case .verificationFailed(let findings) = outcome {
                report.violations.append(
                    "'\(classification.name)' did not verify after transfer:\n  "
                        + findings.joined(separator: "\n  "))
            }
            report.directories.append(
                DirectoryReport(
                    name: classification.name, kind: classification.kind,
                    outcome: outcome,
                    excludedByPolicy: excluded.map(\.relativePath).sorted()))
        }
        report.directories.append(contentsOf: deferredReports)
        report.directories.sort { $0.name < $1.name }

        // Authoring-locus divergence (§8 residual (a)), read AFTER the
        // transfers so a snapshot that came home in this same pass counts as
        // evidence. Read-only on both sides: the seam answers nil for a
        // directory with no manifest snapshot (submit receipts, sessions, a
        // dry run's not-yet-imported directory), and nothing here ever writes
        // `experiments/`.
        let snapshots = transferable.compactMap { classification in
            engine.localRunManifestArms(classification.name)
                .map { (runName: classification.name, arms: $0) }
        }
        report.authoringDivergences = WorkspaceImportPolicy.authoringDivergences(
            snapshots: snapshots, liveArms: engine.liveExperimentArms)
        if report.hasAuthoringDivergences {
            let n = report.authoringDivergences.count
            emit(
                "\(n) stud\(n == 1 ? "y" : "ies") diverge\(n == 1 ? "s" : "") "
                    + "from \(n == 1 ? "its" : "their") own run evidence — see "
                    + "AUTHORING DIVERGENCE")
        }

        // Purge eligibility LAST, and from the workspace as it now stands: a
        // merged run that came home in this same pass is evidence, and a
        // family whose merge is only on the far side is not.
        let families = WorkspaceImportPolicy.shardFamilies(classifications)
        if !families.isEmpty {
            let (evidence, unstamped) = engine.workspaceMergeEvidence()
            report.purgeFamilies = families.map { family in
                PurgeRow(
                    family: family,
                    verdict: WorkspaceImportPolicy.verdict(
                        family: family, evidence: evidence,
                        unstampedCandidates: unstamped))
            }
        }

        if !options.dryRun {
            do {
                let catalog = try engine.rebuildCatalog()
                report.catalog = catalog
                emit(catalog.summary)
            } catch {
                report.violations.append(
                    "the catalog rebuild failed: \(String(describing: error))")
            }
        }
        return report
    }

    /// One directory: verify what is already here, refuse drift, fill gaps.
    static func importOne(
        _ classification: WorkspaceImportPolicy.Classification,
        remote: [WorkspaceImportPolicy.FileStat],
        rules: [WorkspaceImportPolicy.ExclusionRule],
        engine: Engine, options: Options,
        emit: @Sendable (String) -> Void
    ) async -> Outcome {
        let name = classification.name
        let kept = remote.filter {
            !WorkspaceImportPolicy.isExcluded(relativePath: $0.relativePath, rules: rules)
        }
        let bytes = kept.reduce(Int64(0)) { $0 + $1.size }

        guard engine.localExists(name) else {
            guard !options.dryRun else {
                return .imported(files: kept.count, bytes: bytes)
            }
            emit("importing \(name) — \(kept.count) files, \(formatted(bytes: bytes))")
            do {
                try await engine.transfer(name, rules)
            } catch {
                return .failed(message: String(describing: error))
            }
            let findings = verifyLanded(name, remote: remote, rules: rules, engine: engine)
            let problems = findings.filter { finding in
                if case .localOnly = finding { return false }
                return true
            }
            guard problems.isEmpty else {
                return .verificationFailed(findings: problems.map(describe))
            }
            return .imported(files: kept.count, bytes: bytes)
        }

        // Already here. Verify BEFORE anything transfers: tightening 4 makes a
        // size (or pinned-hash) disagreement an immutability violation, and a
        // violation must refuse rather than let rsync decide.
        let findings = verifyLanded(name, remote: remote, rules: rules, engine: engine)
        let violations = findings.filter(\.isViolation)
        guard violations.isEmpty else {
            return .refusedByteDrift(
                message: WorkspaceImportPolicy.immutabilityRefusal(
                    directory: name, violations: violations))
        }
        let gaps = findings.compactMap { finding -> WorkspaceImportPolicy.FileStat? in
            if case .gap(let path, let size) = finding {
                return WorkspaceImportPolicy.FileStat(relativePath: path, size: size)
            }
            return nil
        }
        guard !gaps.isEmpty else {
            return .alreadyComplete(files: kept.count)
        }
        guard !options.dryRun else {
            return .imported(
                files: gaps.count, bytes: gaps.reduce(0) { $0 + $1.size })
        }
        emit(
            "\(name) is partially present — filling \(gaps.count) gap"
                + "\(gaps.count == 1 ? "" : "s")")
        do {
            try await engine.transfer(name, rules)
        } catch {
            return .failed(message: String(describing: error))
        }
        let after = verifyLanded(name, remote: remote, rules: rules, engine: engine)
        let remaining = after.filter { finding in
            if case .localOnly = finding { return false }
            return true
        }
        guard remaining.isEmpty else {
            return .verificationFailed(findings: remaining.map(describe))
        }
        return .imported(files: gaps.count, bytes: gaps.reduce(0) { $0 + $1.size })
    }

    static func verifyLanded(
        _ name: String, remote: [WorkspaceImportPolicy.FileStat],
        rules: [WorkspaceImportPolicy.ExclusionRule], engine: Engine
    ) -> [WorkspaceImportPolicy.Finding] {
        WorkspaceImportPolicy.verify(
            remote: remote,
            local: engine.localInventory(name),
            exclusions: rules,
            pinnedHashes: engine.pinnedHashes(name),
            localHash: { engine.localFileHash(name, $0) })
    }

    // MARK: - Rendering

    public static func describe(_ finding: WorkspaceImportPolicy.Finding) -> String {
        switch finding {
        case .gap(let path, let size):
            "missing locally: \(path) (\(formatted(bytes: size)))"
        case .sizeDrift(let path, let remote, let local):
            "size drift: \(path) — remote \(remote) B, local \(local) B"
        case .hashDrift(let path, let pinned, let actual, let source):
            "hash drift: \(path) — pinned \(String(pinned.prefix(12)))… "
                + "(\(source)), local \(String(actual.prefix(12)))…"
        case .countMismatch(let remote, let local):
            "file-count mismatch: remote \(remote), local \(local)"
        case .localOnly(let path):
            "local only (not on the cluster): \(path)"
        }
    }

    public static func formatted(bytes: Int64) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var value = Double(bytes)
        var index = 0
        while value >= 1024, index < units.count - 1 {
            value /= 1024
            index += 1
        }
        return index == 0
            ? "\(bytes) B"
            : String(format: "%.1f %@", value, units[index])
    }

    /// The human report: per-directory rows, then the unknowns, then the
    /// purge-eligibility section, then violations. Derived from names and
    /// outcomes only — no host, no path on the far side, no token.
    public static func summaryLines(_ report: Report) -> [String] {
        var lines: [String] = []
        if report.dryRun {
            lines.append("DRY RUN — nothing was transferred and nothing was written.")
            lines.append("")
        }
        for directory in report.directories {
            let label = directory.kind.label
            switch directory.outcome {
            case .imported(let files, let bytes):
                lines.append(
                    "\(directory.name)  [\(label)]  "
                        + (report.dryRun ? "would import" : "imported")
                        + " \(files) file\(files == 1 ? "" : "s"), \(formatted(bytes: bytes))")
            case .alreadyComplete(let files):
                lines.append(
                    "\(directory.name)  [\(label)]  already complete "
                        + "(\(files) file\(files == 1 ? "" : "s") verified)")
            case .skippedShardPartial(let index, let count):
                let which = index.map { "\($0 + 1)" } ?? "?"
                let of = count.map { "/\($0)" } ?? ""
                lines.append(
                    "\(directory.name)  [\(label) \(which)\(of)]  skipped by policy "
                        + "— the merged run carries these records")
            case .notApplicable(let reason):
                lines.append("\(directory.name)  skipped — \(reason)")
            case .outsideWindow(let stamp):
                lines.append(
                    "\(directory.name)  skipped — older than --since"
                        + (stamp.map { " (\($0))" } ?? ""))
            case .refusedByteDrift:
                lines.append(
                    "\(directory.name)  [\(label)]  REFUSED — remote bytes differ "
                        + "from the local copy (see violations)")
            case .verificationFailed(let findings):
                lines.append(
                    "\(directory.name)  [\(label)]  DID NOT VERIFY — "
                        + "\(findings.count) finding\(findings.count == 1 ? "" : "s")")
            case .failed(let message):
                lines.append("\(directory.name)  [\(label)]  FAILED — \(message)")
            }
            if !directory.excludedByPolicy.isEmpty {
                let n = directory.excludedByPolicy.count
                lines.append(
                    "    left on the cluster by policy: \(n) path\(n == 1 ? "" : "s")")
            }
        }

        if !report.unknowns.isEmpty {
            lines.append("")
            lines.append(
                "UNKNOWN SHAPES (imported conservatively — the policy does not "
                    + "recognize these names):")
            for name in report.unknowns { lines.append("  \(name)") }
        }

        if report.hasAuthoringDivergences {
            lines.append("")
            lines.append(
                "AUTHORING DIVERGENCE (studies whose live manifest holds fewer "
                    + "arms than their own run evidence — this import is "
                    + "runs-only and never writes experiments/):")
            for divergence in report.authoringDivergences {
                lines.append(
                    "  ! " + WorkspaceImportPolicy.message(divergence: divergence))
            }
        }

        if !report.purgeFamilies.isEmpty || !report.purgeablePaths.isEmpty {
            lines.append("")
            lines.append(
                "PURGE ELIGIBILITY (a report — this verb deletes nothing, here "
                    + "or on the cluster):")
            for row in report.purgeFamilies {
                lines.append("  \(row.verdict.isPurgeEligible ? "·" : "!") \(row.message)")
            }
            for row in report.purgeablePaths {
                lines.append(
                    "  · \(row.directory): \(row.paths.count) path"
                        + "\(row.paths.count == 1 ? "" : "s") "
                        + "(\(formatted(bytes: row.bytes))) — \(row.rule.reason)")
            }
        }

        if !report.violations.isEmpty {
            lines.append("")
            lines.append("VIOLATIONS:")
            for violation in report.violations {
                lines.append(contentsOf: violation.split(separator: "\n").map { "  \($0)" })
            }
        }

        var totals: [String] = []
        let imported = report.imported.count
        if imported > 0 {
            totals.append((report.dryRun ? "would import " : "imported ") + "\(imported)")
        }
        let skipped = report.skippedByPolicy.count
        if skipped > 0 { totals.append("skipped \(skipped)") }
        if !report.unknowns.isEmpty { totals.append("unknown \(report.unknowns.count)") }
        if report.hasAuthoringDivergences {
            totals.append("DIVERGED \(report.authoringDivergences.count)")
        }
        let failed = report.failures.count
        if failed > 0 { totals.append("FAILED \(failed)") }
        lines.append("")
        lines.append(totals.isEmpty ? "nothing to import" : totals.joined(separator: " · "))
        return lines
    }
}

// =============================================================================
// MARK: - The live engine (ssh + rsync, tightening 6)
// =============================================================================

extension WorkspaceRunImport {

    /// Typed refusals this operation raises before any command runs.
    public enum SetupError: Error, LocalizedError, Equatable {
        case notAClusterWorkspace(root: String)
        case noSSHTransport(siteID: String)
        case noRemoteRunRoot(siteID: String)

        public var code: String {
            switch self {
            case .notAClusterWorkspace: "notAClusterWorkspace"
            case .noSSHTransport: "noSSHTransport"
            case .noRemoteRunRoot: "noRemoteRunRoot"
            }
        }

        public var reason: String {
            switch self {
            case .notAClusterWorkspace(let root):
                "the workspace at \(root) is not declared a CLUSTER workspace, "
                    + "so it has no paired cluster runs to import"
            case .noSSHTransport(let siteID):
                "site '\(siteID)' has no SSH transport — run directories are "
                    + "GB-scale and travel over rsync, not the HTTP API"
            case .noRemoteRunRoot(let siteID):
                "site '\(siteID)' declares no workspace or run storage root, so "
                    + "there is no remote runs/ to enumerate"
            }
        }

        public var repairAction: String {
            switch self {
            case .notAClusterWorkspace:
                "declare the workspace's compute substrate as cluster in the "
                    + "app's workspace settings, or point --workspace at the "
                    + "cluster workspace"
            case .noSSHTransport(let siteID):
                "give the site an ssh transport "
                    + "(`steerlab-cli cluster sites export --site \(siteID) "
                    + "--out site.json`, edit, `cluster sites import site.json`)"
            case .noRemoteRunRoot(let siteID):
                "set the site's storage roots (`workspace`, or `run`) and "
                    + "re-import it: `steerlab-cli cluster preview --site \(siteID)` "
                    + "shows what the site currently declares"
            }
        }

        public var errorDescription: String? { "\(reason) — \(repairAction)" }
    }

    /// Build the live engine for one site + one local workspace. Every remote
    /// read is one `find` through the shared SSH ControlMaster; every transfer
    /// is one `rsync` over the same master — the identical seam
    /// `ClusterProvisioningOperations` uses, so authentication state, proxy
    /// jumps, and the login guard are shared rather than re-implemented.
    public static func liveEngine(
        site: ClusterSiteProfile,
        siteID: String,
        workspaceRoot: URL,
        shell: any ClusterShellRunner,
        requireClusterWorkspace: Bool = true,
        emit: @escaping @Sendable (String) -> Void = { _ in }
    ) throws -> Engine {
        if requireClusterWorkspace {
            let declared = WorkspaceCompute.declared(root: workspaceRoot)
            guard declared == nil || declared == .cluster else {
                throw SetupError.notAClusterWorkspace(root: workspaceRoot.path)
            }
        }
        guard case .ssh = site.transport else {
            throw SetupError.noSSHTransport(siteID: siteID)
        }
        guard let runRoot = ClusterEnvironmentRenderer.resolvedRunRoot(site) else {
            throw SetupError.noRemoteRunRoot(siteID: siteID)
        }
        let localRuns = workspaceRoot.appending(component: "runs")

        return Engine(
            probeRemote: {
                let argv = ClusterProvisioner.sshRemoteArgv(
                    site: site, remoteWords: ["test", "-d", runRoot])
                let result = await shell.run(argv)
                guard result.succeeded else {
                    throw ExperimentError(
                        reason: "the remote run root is not reachable "
                            + "(exit \(result.exitCode)): \(result.text)")
                }
            },
            listRemoteDirectories: {
                let argv = ClusterProvisioner.sshRemoteArgv(
                    site: site,
                    remoteWords: [
                        "find", runRoot, "-mindepth", "1", "-maxdepth", "1",
                        "-type", "d", "-printf", "%f\\n",
                    ])
                let result = await shell.run(argv)
                guard result.succeeded else {
                    throw ExperimentError(
                        reason: "could not list \(runRoot) (exit "
                            + "\(result.exitCode)): \(result.text)")
                }
                return result.lines
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
            },
            remoteShardStamped: {
                let argv = ClusterProvisioner.sshRemoteArgv(
                    site: site,
                    remoteWords: [
                        "find", runRoot, "-mindepth", "2", "-maxdepth", "2",
                        "-name", WorkspaceImportPolicy.shardStampFileName,
                        "-printf", "%P\\n",
                    ])
                let result = await shell.run(argv)
                guard result.succeeded else { return [] }
                return Set(
                    result.lines.compactMap { line in
                        let trimmed = line.trimmingCharacters(in: .whitespaces)
                        guard let slash = trimmed.firstIndex(of: "/") else { return nil }
                        return String(trimmed[trimmed.startIndex..<slash])
                    })
            },
            remoteInventory: { names in
                var inventory: [String: [WorkspaceImportPolicy.FileStat]] = [:]
                // Batched so one very wide workspace cannot overflow argv.
                for batch in stride(from: 0, to: names.count, by: 40).map({ start in
                    Array(names[start..<min(start + 40, names.count)])
                }) {
                    let argv = ClusterProvisioner.sshRemoteArgv(
                        site: site,
                        remoteWords: ["find"] + batch.map { "\(runRoot)/\($0)" }
                            + ["-type", "f", "-printf", "%p\\t%s\\n"])
                    let result = await shell.run(argv)
                    guard result.succeeded else {
                        throw ExperimentError(
                            reason: "could not inventory remote run directories "
                                + "(exit \(result.exitCode)): \(result.text)")
                    }
                    for (name, stats) in parseInventory(
                        result.lines, runRoot: runRoot, names: batch)
                    {
                        inventory[name, default: []].append(contentsOf: stats)
                    }
                }
                for name in names where inventory[name] == nil { inventory[name] = [] }
                return inventory
            },
            transfer: { name, rules in
                let destination = localRuns.appending(component: name)
                try FileManager.default.createDirectory(
                    at: destination, withIntermediateDirectories: true)
                let argv = rsyncArgv(
                    site: site, remoteRunRoot: runRoot, name: name,
                    destination: destination, rules: rules)
                let result = await shell.run(argv)
                guard result.succeeded else {
                    throw ExperimentError(
                        reason: "rsync of '\(name)' failed (exit "
                            + "\(result.exitCode)): \(result.text)")
                }
                emit("transferred \(name)")
            },
            localExists: { name in
                var isDirectory: ObjCBool = false
                return FileManager.default.fileExists(
                    atPath: localRuns.appending(component: name).path,
                    isDirectory: &isDirectory) && isDirectory.boolValue
            },
            localInventory: { name in
                localInventory(at: localRuns.appending(component: name))
            },
            pinnedHashes: { name in
                pinnedHashes(inRun: localRuns.appending(component: name))
            },
            localFileHash: { name, relative in
                let url = localRuns.appending(component: name).appending(path: relative)
                guard let data = try? Data(contentsOf: url) else { return nil }
                return SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }
                    .joined()
            },
            workspaceMergeEvidence: { mergeEvidence(inWorkspaceRuns: localRuns) },
            localRunManifestArms: { name in
                manifestArms(
                    at: localRuns.appending(component: name)
                        .appending(component: "experiment.json"))
            },
            liveExperimentArms: { study in
                manifestArms(
                    at: workspaceRoot.appending(component: "experiments")
                        .appending(component: study)
                        .appending(component: "experiment.json"))
            },
            rebuildCatalog: {
                try WorkspaceRunCatalog.rebuild(workspaceRoot: workspaceRoot)
            })
    }

    /// `rsync -a --ignore-existing <policy filters> -e <ssh through the shared
    /// master> host:<runRoot>/<name>/ <workspace>/runs/<name>/`
    ///
    /// `--ignore-existing` is the immutability rule in argv form: a file that
    /// is already here is never rewritten, so a re-import can only ever FILL
    /// gaps. Drift is caught by the verification pass before this runs; the
    /// flag makes the transfer itself incapable of overwriting even if a
    /// future caller forgets to verify first.
    public static func rsyncArgv(
        site: ClusterSiteProfile, remoteRunRoot: String, name: String,
        destination: URL, rules: [WorkspaceImportPolicy.ExclusionRule]
    ) -> [String] {
        guard case .ssh(let host, let proxyJump, _, _) = site.transport else { return [] }
        var argv = [ClusterProvisioner.rsyncExecutablePath, "-a", "--ignore-existing"]
        for rule in rules { argv += rule.rsyncFilters }
        argv += ["-e", ClusterProvisioner.rsyncTransportCommand(proxyJump: proxyJump)]
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        argv += [
            "\(trimmedHost):\(remoteRunRoot)/\(name)/",
            destination.path.hasSuffix("/") ? destination.path : destination.path + "/",
        ]
        return argv
    }

    /// Parse `find … -printf '%p\t%s\n'` output into per-directory stats.
    /// Lines that do not sit under one of the requested directories are
    /// ignored rather than guessed at.
    public static func parseInventory(
        _ lines: [String], runRoot: String, names: [String]
    ) -> [String: [WorkspaceImportPolicy.FileStat]] {
        let prefix = runRoot.hasSuffix("/") ? runRoot : runRoot + "/"
        let wanted = Set(names)
        var out: [String: [WorkspaceImportPolicy.FileStat]] = [:]
        for line in lines {
            guard let tab = line.lastIndex(of: "\t") else { continue }
            let path = String(line[line.startIndex..<tab])
            guard let size = Int64(
                line[line.index(after: tab)...].trimmingCharacters(in: .whitespaces))
            else { continue }
            guard path.hasPrefix(prefix) else { continue }
            let relative = String(path.dropFirst(prefix.count))
            guard let slash = relative.firstIndex(of: "/") else { continue }
            let directory = String(relative[relative.startIndex..<slash])
            guard wanted.contains(directory) else { continue }
            out[directory, default: []].append(
                WorkspaceImportPolicy.FileStat(
                    relativePath: String(relative[relative.index(after: slash)...]),
                    size: size))
        }
        return out
    }

    // MARK: Local reads

    /// The arm counts of a manifest on disk, or nil when no manifest is there
    /// (which is the same answer as "no evidence" / "no live copy").
    static func manifestArms(at url: URL) -> WorkspaceImportPolicy.ManifestArms? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return WorkspaceImportPolicy.manifestArms(fromManifestJSON: data)
    }

    /// Stat-level inventory of a local run directory (tightening 2's local
    /// half): every regular file, its directory-relative path, and its size.
    public static func localInventory(at directory: URL) -> [WorkspaceImportPolicy.FileStat] {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: directory, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey])
        else { return [] }
        let base = directory.standardizedFileURL.path
        var stats: [WorkspaceImportPolicy.FileStat] = []
        for case let url as URL in enumerator {
            guard
                let values = try? url.resourceValues(forKeys: [
                    .isRegularFileKey, .fileSizeKey,
                ]), values.isRegularFile == true
            else { continue }
            let path = url.standardizedFileURL.path
            guard path.hasPrefix(base + "/") else { continue }
            stats.append(
                WorkspaceImportPolicy.FileStat(
                    relativePath: String(path.dropFirst(base.count + 1)),
                    size: Int64(values.fileSize ?? 0)))
        }
        return stats
    }

    /// Per-file hash pins a run directory ALREADY carries.
    ///
    /// Deliberately narrow: the only per-file pins that exist inside a run
    /// directory today are `ResourceManifest`-shaped `files` maps (relative
    /// path → lowercase-hex SHA-256). Vector sidecars pin their INPUTS
    /// (stimulus, recipe, neutral-corpus hashes), not the weight bytes, and
    /// the evidence-bundle manifest lives in the bundle rather than in the
    /// run. Where no pin exists — the common case — verification rests on
    /// counts plus per-file sizes, which is the tightening's floor; this
    /// function does not invent a pin so that a hash check can appear to run.
    public static func pinnedHashes(inRun directory: URL) -> [WorkspaceImportPolicy.PinnedHash] {
        let candidates = [
            "deployment-manifest.json", "resource-manifest.json",
            "artifact-manifest.json",
        ]
        var pins: [WorkspaceImportPolicy.PinnedHash] = []
        for candidate in candidates {
            let url = directory.appending(component: candidate)
            guard
                let data = try? Data(contentsOf: url),
                let object = try? JSONSerialization.jsonObject(with: data),
                let dictionary = object as? [String: Any],
                let files = dictionary["files"] as? [String: String]
            else { continue }
            for (path, hash) in files.sorted(by: { $0.key < $1.key }) {
                pins.append(
                    WorkspaceImportPolicy.PinnedHash(
                        relativePath: path, sha256: hash, source: candidate))
            }
        }
        return pins
    }

    /// Read every merged run's completeness stamp out of the workspace, plus
    /// the merged-LOOKING runs that carry none. The gate's only input
    /// (tightening 1).
    public static func mergeEvidence(inWorkspaceRuns runs: URL) -> (
        evidence: [WorkspaceImportPolicy.MergeEvidence],
        unstamped: [WorkspaceImportPolicy.UnstampedMergeCandidate]
    ) {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: runs.path) else {
            return ([], [])
        }
        var evidence: [WorkspaceImportPolicy.MergeEvidence] = []
        var unstamped: [WorkspaceImportPolicy.UnstampedMergeCandidate] = []
        for name in names.sorted() {
            let directory = runs.appending(component: name)
            var isDirectory: ObjCBool = false
            guard
                fm.fileExists(atPath: directory.path, isDirectory: &isDirectory),
                isDirectory.boolValue
            else { continue }
            let classification = WorkspaceImportPolicy.classify(directoryName: name)
            guard classification.kind == .run else { continue }
            if let stamp = mergeEvidence(inRun: directory, named: name) {
                evidence.append(stamp)
            } else {
                unstamped.append(
                    WorkspaceImportPolicy.UnstampedMergeCandidate(
                        runName: name, stem: classification.stem))
            }
        }
        return (evidence, unstamped)
    }

    /// The `sharded` block of one run's `report.json`, exactly as
    /// `merge_shard_runs` writes it. Nil when the report is absent,
    /// unreadable, or carries no block — all three of which mean "no proof",
    /// which is the same answer as far as the gate is concerned.
    public static func mergeEvidence(
        inRun directory: URL, named name: String
    ) -> WorkspaceImportPolicy.MergeEvidence? {
        guard
            let data = try? Data(
                contentsOf: directory.appending(
                    component: WorkspaceImportPolicy.reportFileName)),
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any],
            let sharded = dictionary["sharded"] as? [String: Any],
            let count = sharded["shardCount"] as? Int,
            let runs = sharded["shardRuns"] as? [String],
            !runs.isEmpty
        else { return nil }
        return WorkspaceImportPolicy.MergeEvidence(
            mergedRun: name, shardCount: count, shardRuns: runs)
    }
}

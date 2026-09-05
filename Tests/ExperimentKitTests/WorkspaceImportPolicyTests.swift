import Foundation
import Testing

@testable import ExperimentKit

// =============================================================================
// The workspace import policy (open-issues §20), as behavior.
//
// Every fixture name here is SYNTHETIC: neutral study stems (`alpha`, `beta`,
// `gamma`), a fictional site, and no concept, case family, or institution. The
// policy is concept-agnostic by design, so its tests must be able to prove that
// by containing no vocabulary the release scanner would flag.
//
// The suite is organized by the six tightenings the researcher attached to the
// policy on 2026-08-20, because those are the properties that must not regress:
// the merge-EVIDENCE gate, stat-level verification, report-never-purge,
// byte-drift refusal, gitignored catalog, and the ssh/rsync transfer seam.
// =============================================================================

// MARK: - Classification

struct WorkspaceImportPolicyClassificationTests {

    private let stamp = "20260819T101500123"

    private func classify(_ rest: String, shardStamp: Bool = false)
        -> WorkspaceImportPolicy.Classification
    {
        WorkspaceImportPolicy.classify(
            directoryName: "\(stamp)-\(rest)", containsShardStamp: shardStamp)
    }

    /// The ALWAYS list, shape by shape. Each of these is science, a receipt, or
    /// hash-pinned bytes, and each must come home.
    @Test func everyAlwaysImportedShapeIsRecognized() {
        let cases: [(String, WorkspaceImportPolicy.DirectoryKind)] = [
            ("exp-alpha-run", .run),
            ("exp-alpha-analyze", .analyze),
            ("exp-alpha-evaluate", .evaluate),
            ("exp-alpha-evaluate-judgment", .evaluate),
            ("exp-alpha-validate", .validate),
            ("exp-alpha-extract", .extract),
            ("exp-alpha-sweep", .sweep),
            ("exp-alpha-confirm", .confirm),
            ("exp-alpha-pipeline", .pipeline),
            ("submit-alpha-run", .submit),
            ("submit-trainer-alpha", .submit),
            ("optvec-alpha-l20", .vectorArtifact),
            ("sae-feature-alpha", .vectorArtifact),
            ("derived-alpha-gm", .vectorArtifact),
            ("jlens-support-alpha", .lensSupport),
            ("session-gpu", .session),
        ]
        for (name, expected) in cases {
            let classification = classify(name)
            #expect(
                classification.kind == expected,
                "\(name) classified \(classification.kind.rawValue), expected \(expected.rawValue)")
            #expect(classification.kind.isAlwaysImported, "\(name) must be always-imported")
            #expect(
                WorkspaceImportPolicy.decision(for: classification).transfers,
                "\(name) must transfer")
        }
    }

    /// A superseded attempt is a receipt like any other run of its kind: the
    /// `resume-` prefix is bookkeeping about the ATTEMPT, and must not change
    /// what the directory is or which family it belongs to.
    @Test func resumeAttemptsKeepTheirKindAndFamily() {
        let plain = classify("exp-alpha-run")
        let resumed = classify("resume-exp-alpha-run")
        let resumedAgain = classify("resume2-exp-alpha-run")
        #expect(resumed.kind == .run)
        #expect(resumedAgain.kind == .run)
        #expect(resumed.stem == plain.stem)
        #expect(resumedAgain.stem == plain.stem)
    }

    /// Shard partials never transfer — and are identified by the NAME suffix
    /// (which is what a merged run's stamp joins on) OR by a `shard.json` the
    /// enumeration found.
    @Test func shardPartialsAreNeverTransferred() {
        let byName = classify("exp-alpha-run-shard2of4")
        #expect(byName.kind == .shardPartial)
        #expect(byName.shardIndex == 2)
        #expect(byName.shardCount == 4)
        #expect(byName.stem == "alpha")
        #expect(!WorkspaceImportPolicy.decision(for: byName).transfers)

        // A partial whose name lost its suffix is still a partial.
        let byStamp = classify("exp-alpha-run", shardStamp: true)
        #expect(byStamp.kind == .shardPartial)
        #expect(!WorkspaceImportPolicy.decision(for: byStamp).transfers)
    }

    /// The conservative branch: an unrecognized shape is REPORTED and imported
    /// anyway. Silently skipping on a purging filesystem loses evidence.
    @Test func unknownShapesAreImportedConservatively() {
        let classification = classify("something-nobody-declared")
        #expect(classification.kind == .unknown)
        let decision = WorkspaceImportPolicy.decision(for: classification)
        #expect(decision.transfers)
        guard case .importConservatively(let reason) = decision else {
            Issue.record("expected a conservative import decision")
            return
        }
        #expect(reason.contains("unrecognized"))
    }

    /// The mutable library subtrees are not run directories, and an import
    /// never sweeps them: they have their own lifecycle.
    @Test func librarySubtreesAreNotRunDirectories() {
        for library in WorkspaceImportPolicy.librarySubtrees {
            let classification = WorkspaceImportPolicy.classify(directoryName: library)
            #expect(classification.kind == .notARunDirectory)
            let decision = WorkspaceImportPolicy.decision(for: classification)
            #expect(!decision.transfers)
            guard case .notApplicable(let reason) = decision else {
                Issue.record("\(library) must be notApplicable")
                return
            }
            #expect(reason.contains("library"))
        }
    }

    // MARK: Exclusions

    /// The two NEVER-import path rules, and the scope each applies at.
    @Test func exclusionRulesMatchThePolicysNeverList() {
        let submit = WorkspaceImportPolicy.exclusions(for: .submit)
        #expect(submit.contains(.evidenceBundleTarball))
        #expect(submit.contains(.trainingCheckpointTree))
        // Evidence bundles are written into submit dirs; a run directory has
        // none, and a blanket tarball exclusion elsewhere could drop a real
        // artifact that happens to be compressed.
        #expect(!WorkspaceImportPolicy.exclusions(for: .run).contains(.evidenceBundleTarball))

        #expect(
            WorkspaceImportPolicy.isExcluded(
                relativePath: "run-alpha.evidence-bundle.tar.gz", rules: submit))
        #expect(
            WorkspaceImportPolicy.isExcluded(
                relativePath: "run/adapter-alpha/checkpoints/step-500/optimizer.pt",
                rules: submit))
        // The FINAL adapter weights are kept — that is the whole point of
        // importing a finetune receipt.
        #expect(
            !WorkspaceImportPolicy.isExcluded(
                relativePath: "run/adapter-alpha/"
                    + WorkspaceImportPolicy.adapterWeightFileName,
                rules: submit))
        #expect(
            !WorkspaceImportPolicy.isExcluded(
                relativePath: "generations.jsonl", rules: submit))
    }

    // MARK: `--since`

    @Test func sinceAcceptsTheDateGrammarsAndRefusesTheRest() {
        #expect(WorkspaceImportPolicy.normalizedSince("2026-08-01") == "20260801T000000000")
        #expect(WorkspaceImportPolicy.normalizedSince("20260801") == "20260801T000000000")
        #expect(
            WorkspaceImportPolicy.normalizedSince("2026-08-01T09:30:00")
                == "20260801T093000000")
        #expect(WorkspaceImportPolicy.normalizedSince("last tuesday") == nil)
        #expect(WorkspaceImportPolicy.normalizedSince("") == nil)
    }

    @Test func sinceFiltersOnTheRunStampAndPassesTheUnstamped() {
        let since = WorkspaceImportPolicy.normalizedSince("2026-08-19")
        let older = WorkspaceImportPolicy.classify(directoryName: "20260701T090000000-exp-alpha-run")
        let newer = WorkspaceImportPolicy.classify(directoryName: "20260819T090000000-exp-alpha-run")
        #expect(!WorkspaceImportPolicy.passesSince(older, since: since))
        #expect(WorkspaceImportPolicy.passesSince(newer, since: since))
        // No stamp = no opinion = conservative pass.
        let stampless = WorkspaceImportPolicy.classify(directoryName: "model-variants")
        #expect(WorkspaceImportPolicy.passesSince(stampless, since: since))
    }
}

// MARK: - The merge-evidence gate (tightening 1)

struct WorkspaceImportMergeEvidenceTests {

    private let stamp = "20260819T101500123"

    private func family(_ stem: String, count: Int) -> WorkspaceImportPolicy.ShardFamily {
        let partials = (0..<count).map { "\(stamp)-exp-\(stem)-run-shard\($0)of\(count)" }
        return WorkspaceImportPolicy.ShardFamily(
            stem: stem, declaredCount: count, partials: partials)
    }

    /// The eligible case: a merged run is in the workspace AND its report
    /// carries the `sharded` completeness stamp naming every partial.
    @Test func anEvidencedMergeMakesTheFamilyPurgeEligible() {
        let family = family("alpha", count: 3)
        let evidence = WorkspaceImportPolicy.MergeEvidence(
            mergedRun: "20260819T120000000-exp-alpha-run",
            shardCount: 3, shardRuns: family.partials)
        let verdict = WorkspaceImportPolicy.verdict(
            family: family, evidence: [evidence], unstampedCandidates: [])
        #expect(verdict == .purgeEligible(mergedRun: evidence.mergedRun))
        #expect(verdict.isPurgeEligible)
        #expect(!verdict.isLoud)
        let message = WorkspaceImportPolicy.message(family: family, verdict: verdict)
        #expect(message.contains("merge completeness stamp"))
    }

    /// THE non-negotiable case. A merged-LOOKING directory is not a proof: the
    /// gate reads the stamp, and its absence is loud and never eligible.
    @Test func aMergedDirectoryWithoutTheStampIsNeverEligible() {
        let family = family("beta", count: 4)
        let candidate = WorkspaceImportPolicy.UnstampedMergeCandidate(
            runName: "20260819T120000000-exp-beta-run", stem: "beta")
        let verdict = WorkspaceImportPolicy.verdict(
            family: family, evidence: [], unstampedCandidates: [candidate])
        #expect(verdict == .mergedRunNotStamped(candidates: [candidate.runName]))
        #expect(!verdict.isPurgeEligible)
        #expect(verdict.isLoud)
        let message = WorkspaceImportPolicy.message(family: family, verdict: verdict)
        #expect(message.contains("NOT purge-eligible"))
        #expect(message.contains("no merge completeness stamp"))
        #expect(message.contains("A directory is not a proof"))
    }

    /// The orphan: no merged run anywhere. Loud, and never eligible.
    @Test func anOrphanedPartialIsLoudAndNeverEligible() {
        let family = family("gamma", count: 2)
        let verdict = WorkspaceImportPolicy.verdict(
            family: family, evidence: [], unstampedCandidates: [])
        #expect(verdict == .orphaned)
        #expect(!verdict.isPurgeEligible)
        let message = WorkspaceImportPolicy.message(family: family, verdict: verdict)
        #expect(message.contains("ORPHANED"))
        #expect(message.contains("Never purge"))
    }

    /// A stamp that names only SOME of the partials belongs to a different
    /// fan-out or attempt, and cannot license dropping the rest.
    @Test func aStampThatDoesNotCoverEveryPartialIsNotEligible() {
        let family = family("alpha", count: 3)
        let evidence = WorkspaceImportPolicy.MergeEvidence(
            mergedRun: "20260819T120000000-exp-alpha-run",
            shardCount: 3, shardRuns: Array(family.partials.dropLast()))
        let verdict = WorkspaceImportPolicy.verdict(
            family: family, evidence: [evidence], unstampedCandidates: [])
        guard case .stampDoesNotCoverPartials(let mergedRun, let missing) = verdict else {
            Issue.record("expected a coverage refusal, got \(verdict)")
            return
        }
        #expect(mergedRun == evidence.mergedRun)
        #expect(missing == [family.partials.last!])
        #expect(!verdict.isPurgeEligible)
    }

    /// A stamp whose shard COUNT disagrees is a different fan-out too.
    @Test func aStampWithADifferentShardCountIsNotEligible() {
        let family = family("alpha", count: 3)
        let evidence = WorkspaceImportPolicy.MergeEvidence(
            mergedRun: "20260819T120000000-exp-alpha-run",
            shardCount: 4, shardRuns: family.partials)
        let verdict = WorkspaceImportPolicy.verdict(
            family: family, evidence: [evidence], unstampedCandidates: [])
        #expect(!verdict.isPurgeEligible)
    }

    /// Grouping: partials of two different studies never merge into one family.
    @Test func partialsGroupByStem() {
        let names = [
            "\(stamp)-exp-alpha-run-shard0of2", "\(stamp)-exp-alpha-run-shard1of2",
            "\(stamp)-exp-beta-run-shard0of3",
            "\(stamp)-exp-alpha-run",
        ]
        let families = WorkspaceImportPolicy.shardFamilies(
            names.map { WorkspaceImportPolicy.classify(directoryName: $0) })
        #expect(families.map(\.stem) == ["alpha", "beta"])
        #expect(families[0].partials.count == 2)
        #expect(families[0].declaredCount == 2)
        #expect(families[1].partials.count == 1)
    }

    /// The stamp reader, against the exact JSON `merge_shard_runs` writes.
    @Test func theStampReaderReadsTheShardedBlockAndNothingElse() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-merge-stamp-\(UUID().uuidString)")
        let stamped = root.appending(component: "20260819T120000000-exp-alpha-run")
        let bare = root.appending(component: "20260819T130000000-exp-beta-run")
        for directory in [stamped, bare] {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
        }
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(
            """
            {"sharded": {"shardCount": 2,
                         "shardRuns": ["a-shard0of2", "a-shard1of2"],
                         "shardJobIDs": ["1", "2"]}}
            """.utf8
        ).write(to: stamped.appending(component: WorkspaceImportPolicy.reportFileName))
        // A report with no `sharded` block: a single-job run, or a merge that
        // never happened. Both mean "no proof".
        try Data(#"{"conditions": []}"#.utf8)
            .write(to: bare.appending(component: WorkspaceImportPolicy.reportFileName))

        let evidence = WorkspaceRunImport.mergeEvidence(
            inRun: stamped, named: stamped.lastPathComponent)
        #expect(evidence?.shardCount == 2)
        #expect(evidence?.shardRuns == ["a-shard0of2", "a-shard1of2"])
        #expect(
            WorkspaceRunImport.mergeEvidence(inRun: bare, named: bare.lastPathComponent) == nil)

        let (found, unstamped) = WorkspaceRunImport.mergeEvidence(inWorkspaceRuns: root)
        #expect(found.map(\.mergedRun) == [stamped.lastPathComponent])
        #expect(unstamped.map(\.runName) == [bare.lastPathComponent])
        #expect(unstamped.map(\.stem) == ["beta"])
    }
}

// MARK: - Run completeness (2026-09-05)

struct WorkspaceImportRunCompletenessTests {

    private func stat(_ path: String, _ size: Int64 = 1) -> WorkspaceImportPolicy.FileStat {
        WorkspaceImportPolicy.FileStat(relativePath: path, size: size)
    }

    /// Records with no report beside them is the shape a merge parent had on
    /// 2026-09-05 after the controller died mid-merge — and the shape of any
    /// run still executing. Neither is complete; the report makes it so.
    @Test func recordsWithoutAReportAreAnUnfinishedRun() {
        let unfinished = [stat("config.json"), stat("generations.jsonl", 4096)]
        #expect(WorkspaceImportPolicy.isIncomplete(kind: .run, remote: unfinished, exclusions: []))
        #expect(
            !WorkspaceImportPolicy.isIncomplete(
                kind: .run, remote: unfinished + [stat("report.json", 300)], exclusions: []))
    }

    /// No records, nothing to be unfinished about: an empty directory and a
    /// bare config are not judged (the empty-on-both-sides case stays
    /// "already complete", as decided on 2026-08-24).
    @Test func directoriesWithoutRecordsAreNotJudged() {
        #expect(!WorkspaceImportPolicy.isIncomplete(kind: .run, remote: [], exclusions: []))
        #expect(
            !WorkspaceImportPolicy.isIncomplete(
                kind: .run, remote: [stat("config.json")], exclusions: []))
    }

    /// The marker is the engines' own (`resume.completion_file_for`): a study
    /// run finishes with report.json; other shapes are not judged here.
    @Test func onlyRunsCarryTheCompletionMarker() {
        #expect(WorkspaceImportPolicy.completionMarker(for: .run) == "report.json")
        let unfinished = [stat("generations.jsonl", 4096)]
        for kind in WorkspaceImportPolicy.DirectoryKind.allCases where kind != .run {
            #expect(WorkspaceImportPolicy.completionMarker(for: kind) == nil, "\(kind)")
            #expect(
                !WorkspaceImportPolicy.isIncomplete(kind: kind, remote: unfinished, exclusions: []),
                "\(kind)")
        }
    }

    /// Junk beside the records changes nothing: a `.DS_Store` is not a report,
    /// and a report is a report whatever else is in the listing.
    @Test func inertFilesNeitherCompleteNorHideARun() {
        let withJunk = [stat("generations.jsonl", 4096), stat(".DS_Store", 6148)]
        #expect(WorkspaceImportPolicy.isIncomplete(kind: .run, remote: withJunk, exclusions: []))
        #expect(
            !WorkspaceImportPolicy.isIncomplete(
                kind: .run, remote: withJunk + [stat("report.json")], exclusions: []))
    }
}

// MARK: - Verification (tightening 2) and byte-drift refusal (tightening 4)

struct WorkspaceImportVerificationTests {

    private func stat(_ path: String, _ size: Int64) -> WorkspaceImportPolicy.FileStat {
        WorkspaceImportPolicy.FileStat(relativePath: path, size: size)
    }

    @Test func aCompleteDirectoryProducesNoFindings() {
        let files = [stat("config.json", 100), stat("generations.jsonl", 4096)]
        let findings = WorkspaceImportPolicy.verify(
            remote: files, local: files, exclusions: [])
        #expect(findings.isEmpty)
    }

    /// A missing file is a GAP an idempotent re-import fills — never a
    /// violation, and never a reason to refuse.
    @Test func aMissingFileIsAGapNotAViolation() {
        let findings = WorkspaceImportPolicy.verify(
            remote: [stat("a.json", 10), stat("b.jsonl", 20)],
            local: [stat("a.json", 10)], exclusions: [])
        #expect(findings.count == 1)
        #expect(findings.first == .gap(relativePath: "b.jsonl", size: 20))
        #expect(findings.allSatisfy { !$0.isViolation })
    }

    /// The truncation case counts alone would pass: same file count, different
    /// bytes. Per-file SIZE is why this refuses.
    @Test func aTruncatedFileIsAViolationEvenThoughCountsAgree() {
        let findings = WorkspaceImportPolicy.verify(
            remote: [stat("generations.jsonl", 4096)],
            local: [stat("generations.jsonl", 2048)], exclusions: [])
        #expect(
            findings == [
                .sizeDrift(relativePath: "generations.jsonl", remote: 4096, local: 2048)
            ])
        #expect(findings.contains { $0.isViolation })
    }

    /// A local-only file is REPORTED and never counted against the transfer.
    ///
    /// It used to also produce a `remote 1, local 2` count row, which the
    /// fresh-import path judged as a verification failure — 158 of them on
    /// 2026-08-24, over correctly landed directories. A file that is here and
    /// not on the cluster says nothing about whether the bytes that ARE on the
    /// cluster arrived.
    @Test func aLocalOnlyFileIsReportedButNeverACountMismatch() {
        let findings = WorkspaceImportPolicy.verify(
            remote: [stat("a.json", 10)],
            local: [stat("a.json", 10), stat("stray.json", 5)], exclusions: [])
        #expect(findings == [.localOnly(relativePath: "stray.json")])
        #expect(findings.allSatisfy { !$0.isViolation })
    }

    /// …and the count row still fires for totals the per-file walk cannot
    /// explain, which is the only thing counts were ever for.
    @Test func totalsThePerFileWalkCannotExplainStillMismatch() {
        // A duplicated path in an inventory: two local rows, one file.
        let findings = WorkspaceImportPolicy.verify(
            remote: [stat("a.json", 10)],
            local: [stat("a.json", 10), stat("a.json", 10)], exclusions: [])
        #expect(findings == [.countMismatch(remote: 1, local: 2)])
    }

    /// Junk and locally-generated artifacts are inert on BOTH sides: neither a
    /// local-only row nor a count. `.DS_Store` is the sharpest case — the
    /// import's own rules never carry it, so verification failing over it was
    /// failing over a file the transfer is designed never to move.
    @Test func junkAndLocallyGeneratedFilesAreInertOnBothSides() {
        let findings = WorkspaceImportPolicy.verify(
            remote: [stat("config.json", 10)],
            local: [
                stat("config.json", 10),
                stat(".DS_Store", 6148),
                stat("pipeline-portable.json", 900),
                stat("nested/.DS_Store", 6148),
            ],
            exclusions: [])
        #expect(findings.isEmpty)
        #expect(WorkspaceImportPolicy.isVerificationInert(relativePath: ".DS_Store"))
        #expect(WorkspaceImportPolicy.isVerificationInert(
            relativePath: "a/b/pipeline-portable.json"))
        #expect(!WorkspaceImportPolicy.isVerificationInert(
            relativePath: "generations.jsonl"))
    }

    /// A pinned hash that disagrees refuses, wherever the pin exists.
    @Test func aPinnedHashMismatchIsAViolation() {
        let pin = WorkspaceImportPolicy.PinnedHash(
            relativePath: "vector.safetensors", sha256: "aa" + String(repeating: "0", count: 62),
            source: "artifact-manifest.json")
        let findings = WorkspaceImportPolicy.verify(
            remote: [stat("vector.safetensors", 1024)],
            local: [stat("vector.safetensors", 1024)],
            exclusions: [], pinnedHashes: [pin],
            localHash: { _ in "bb" + String(repeating: "0", count: 62) })
        #expect(findings.count == 1)
        #expect(findings.first?.isViolation == true)
        guard case .hashDrift(let path, _, _, let source)? = findings.first else {
            Issue.record("expected a hash drift")
            return
        }
        #expect(path == "vector.safetensors")
        #expect(source == "artifact-manifest.json")
    }

    /// A matching pin (and case-insensitive hex) is silent.
    @Test func aMatchingPinnedHashIsSilent() {
        let hex = String(repeating: "AB", count: 32)
        let pin = WorkspaceImportPolicy.PinnedHash(
            relativePath: "vector.safetensors", sha256: hex,
            source: "artifact-manifest.json")
        let findings = WorkspaceImportPolicy.verify(
            remote: [stat("vector.safetensors", 1024)],
            local: [stat("vector.safetensors", 1024)],
            exclusions: [], pinnedHashes: [pin],
            localHash: { _ in hex.lowercased() })
        #expect(findings.isEmpty)
    }

    /// The policy's own NEVER-import rules must never read as missing files.
    @Test func excludedRemotePathsAreNotCountedAsGaps() {
        let findings = WorkspaceImportPolicy.verify(
            remote: [
                stat("plan.json", 10),
                stat("evidence.tar.gz", 1_000_000),
                stat("run/adapter/checkpoints/step-1/optimizer.pt", 2_000_000),
            ],
            local: [stat("plan.json", 10)],
            exclusions: WorkspaceImportPolicy.exclusions(for: .submit))
        #expect(findings.isEmpty)
    }

    /// …and the same rules apply to the LOCAL side, or a directory that
    /// already holds an excluded file fails its own count comparison for ever:
    /// rsync will never bring the remote twin over to make raw counts agree.
    @Test func excludedLocalPathsAreCountedOnNeitherSide() {
        let findings = WorkspaceImportPolicy.verify(
            remote: [stat("plan.json", 10), stat("manifest.json", 4)],
            local: [
                stat("plan.json", 10),
                stat("manifest.json", 4),
                stat("evidence.tar.gz", 1_000_000),
                stat("run/adapter/checkpoints/step-1/optimizer.pt", 2_000_000),
            ],
            exclusions: WorkspaceImportPolicy.exclusions(for: .submit))
        #expect(findings.isEmpty)
    }

    @Test func theImmutabilityRefusalNamesTheFileAndForbidsARetry() {
        let text = WorkspaceImportPolicy.immutabilityRefusal(
            directory: "20260819T101500123-exp-alpha-run",
            violations: [
                .sizeDrift(relativePath: "generations.jsonl", remote: 4096, local: 2048)
            ])
        #expect(text.contains("generations.jsonl"))
        #expect(text.contains("immutability violation"))
        #expect(text.contains("Nothing was overwritten"))
    }
}

// MARK: - The operation

/// A scripted remote: names, per-directory inventories, and a local tree that
/// the fake transfer actually materializes, so idempotency is a real property
/// rather than a mocked one.
private final class FakeImportRemote: @unchecked Sendable {
    // @unchecked Sendable: mutated only from the serialized test body and the
    // single operation under test; never escapes the test.
    var directories: [String] = []
    var shardStamped: Set<String> = []
    var inventories: [String: [WorkspaceImportPolicy.FileStat]] = [:]
    var localFiles: [String: [WorkspaceImportPolicy.FileStat]] = [:]
    var pinned: [String: [WorkspaceImportPolicy.PinnedHash]] = [:]
    var hashes: [String: String] = [:]
    var evidence: [WorkspaceImportPolicy.MergeEvidence] = []
    var unstamped: [WorkspaceImportPolicy.UnstampedMergeCandidate] = []
    var runManifestArms: [String: WorkspaceImportPolicy.ManifestArms] = [:]
    var liveArms: [String: WorkspaceImportPolicy.ManifestArms] = [:]
    private(set) var transferred: [String] = []
    private(set) var catalogRebuilds = 0
    /// Runs AFTER the modelled rsync — what the landing writes locally
    /// (`pipeline-portable.json`) and what the filesystem adds (`.DS_Store`).
    var transferHook: (@Sendable () -> Void)?
    /// Replaces the modelled rsync entirely, for the imperfect-transfer cases.
    var transferOverride: (@Sendable () -> Void)?

    func engine() -> WorkspaceRunImport.Engine {
        WorkspaceRunImport.Engine(
            listRemoteDirectories: { self.directories },
            remoteShardStamped: { self.shardStamped },
            remoteInventory: { names in
                var out: [String: [WorkspaceImportPolicy.FileStat]] = [:]
                for name in names { out[name] = self.inventories[name] ?? [] }
                return out
            },
            transfer: { name, rules in
                self.transferred.append(name)
                if let override = self.transferOverride {
                    override()
                    return
                }
                // The live transfer is `rsync --ignore-existing`: it can only
                // ever ADD files the policy keeps. The fake obeys the same
                // rule, so a re-import over a complete tree is a no-op here
                // exactly as it is there.
                var local = self.localFiles[name] ?? []
                let known = Set(local.map(\.relativePath))
                for file in self.inventories[name] ?? []
                where !known.contains(file.relativePath)
                    && !WorkspaceImportPolicy.isExcluded(
                        relativePath: file.relativePath, rules: rules)
                {
                    local.append(file)
                }
                self.localFiles[name] = local
                self.transferHook?()
            },
            localExists: { self.localFiles[$0] != nil },
            localInventory: { self.localFiles[$0] ?? [] },
            pinnedHashes: { self.pinned[$0] ?? [] },
            localFileHash: { name, path in self.hashes["\(name)/\(path)"] },
            workspaceMergeEvidence: { (self.evidence, self.unstamped) },
            localRunManifestArms: { self.runManifestArms[$0] },
            liveExperimentArms: { self.liveArms[$0] },
            rebuildCatalog: {
                self.catalogRebuilds += 1
                return WorkspaceRunCatalog.BuildReport(
                    rows: [], linkCount: 0, adapterCount: 0, libraryCount: 0,
                    gitignoreUpdated: false)
            })
    }
}

struct WorkspaceRunImportOperationTests {

    private let stamp = "20260819T101500123"

    private func remote(files: [(String, Int64)]) -> [WorkspaceImportPolicy.FileStat] {
        files.map { WorkspaceImportPolicy.FileStat(relativePath: $0.0, size: $0.1) }
    }

    @Test func aFreshImportTransfersVerifiesAndRebuildsTheCatalog() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [
            ("config.json", 120), ("generations.jsonl", 4096), ("report.json", 300),
        ])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred == [run])
        #expect(report.imported.map(\.name) == [run])
        #expect(report.violations.isEmpty)
        #expect(fake.catalogRebuilds == 1)
    }

    /// Re-running is the recovery: an already-complete directory transfers
    /// nothing and reports itself verified.
    @Test func reimportingACompleteDirectoryIsANoOp() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [("config.json", 120)])
        fake.localFiles[run] = remote(files: [("config.json", 120)])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty)
        #expect(report.imported.isEmpty)
        guard case .alreadyComplete = report.directories.first?.outcome else {
            Issue.record("expected alreadyComplete, got \(String(describing: report.directories.first))")
            return
        }
    }

    /// An EMPTY remote inventory can never certify a populated local
    /// directory. Before 2026-08-24 it certified 31 of them: `gaps` is derived
    /// entirely from the remote inventory, so "no gaps" over an empty one says
    /// nothing at all.
    @Test func anEmptyRemoteInventoryNeverCertifiesAPopulatedDirectory() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = []
        fake.localFiles[run] = remote(files: [
            ("config.json", 120), ("generations.jsonl", 4096),
        ])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty)
        guard
            case .refusedEmptyRemoteInventory(let localFiles)?
                = report.directories.first?.outcome
        else {
            Issue.record(
                "expected a refusal, got \(String(describing: report.directories.first))")
            return
        }
        #expect(localFiles == 2)
        let violation = report.violations.joined(separator: "\n")
        #expect(violation.contains(run))
        #expect(violation.contains("inventory failed"))
        #expect(violation.contains("remote run really is gone"))
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("REFUSED"))
        #expect(!text.contains("already complete"))
    }

    /// …and a directory that is empty on BOTH sides is not a refusal: there is
    /// nothing there to be wrong about.
    @Test func anEmptyDirectoryOnBothSidesIsStillComplete() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = []
        fake.localFiles[run] = []

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(report.violations.isEmpty)
        guard case .alreadyComplete = report.directories.first?.outcome else {
            Issue.record(
                "expected alreadyComplete, got \(String(describing: report.directories.first))")
            return
        }
    }

    /// The exclusion rules apply to BOTH sides of the count. A directory that
    /// already holds an excluded file — a tarball copied home before the rule
    /// existed — must not fail its own verification for ever, because rsync
    /// will never bring the remote twin over to make the raw counts agree.
    @Test func policyExcludedPathsAreDroppedFromBothSidesOfTheCount() async {
        let fake = FakeImportRemote()
        let submit = "\(stamp)-submit-alpha-run"
        let tarball = "alpha.evidence-bundle.tar.gz"
        fake.directories = [submit]
        fake.inventories[submit] = remote(files: [
            ("plan.json", 128), ("manifest.json", 64), (tarball, 30_000_000),
        ])
        fake.localFiles[submit] = remote(files: [("plan.json", 128), (tarball, 30_000_000)])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(report.violations.isEmpty, "the tarball is excluded, not missing")
        guard case .imported(let files, _)? = report.directories.first?.outcome else {
            Issue.record(
                "expected the one gap to be filled, got \(String(describing: report.directories.first))")
            return
        }
        #expect(files == 1)

        // The field report's second case, stated on its own: excluded remotely,
        // absent locally, and complete.
        let clean = FakeImportRemote()
        clean.directories = [submit]
        clean.inventories[submit] = remote(files: [
            ("plan.json", 128), ("manifest.json", 64), (tarball, 30_000_000),
        ])
        clean.localFiles[submit] = remote(files: [("plan.json", 128), ("manifest.json", 64)])
        let second = await WorkspaceRunImport.run(engine: clean.engine())
        #expect(clean.transferred.isEmpty)
        guard case .alreadyComplete = second.directories.first?.outcome else {
            Issue.record(
                "expected alreadyComplete, got \(String(describing: second.directories.first))")
            return
        }
    }

    /// §2.4, the fresh-import path: a file our OWN machinery writes into the
    /// landed directory is not a failed transfer.
    ///
    /// The repair pass of 2026-08-24 reported 175 violations where 17 were
    /// real. All 158 false ones were `remote N / local N+1` — the extra file a
    /// locally written `pipeline-portable.json` (151 directories) or a
    /// `.DS_Store` (7). The bytes had landed correctly in every one.
    @Test func aLocallyGeneratedFileNeverFailsAFreshImport() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-pipeline"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [
            ("pipeline.json", 512), ("stage-1/report.json", 128),
        ])
        // The transfer lands the remote files; the import machinery then
        // writes its portable ledger, and Finder leaves its droppings.
        fake.transferHook = {
            fake.localFiles[run] = (fake.localFiles[run] ?? []) + [
                WorkspaceImportPolicy.FileStat(
                    relativePath: "pipeline-portable.json", size: 900),
                WorkspaceImportPolicy.FileStat(relativePath: ".DS_Store", size: 6148),
            ]
        }

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred == [run])
        #expect(report.violations.isEmpty, "\(report.violations)")
        guard case .imported = report.directories.first?.outcome else {
            Issue.record(
                "expected a clean import, got \(String(describing: report.directories.first))")
            return
        }
    }

    /// …and a genuine remote-side gap after a transfer still FAILS. The filter
    /// is about the local side only; a file the cluster has and we do not is
    /// exactly what verification exists to catch.
    @Test func aRemoteGapAfterTransferStillFailsTheFreshImport() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [
            ("config.json", 120), ("generations.jsonl", 4096),
        ])
        // A transfer that drops one file, and writes a local artifact besides.
        fake.transferOverride = {
            fake.localFiles[run] = [
                WorkspaceImportPolicy.FileStat(relativePath: "config.json", size: 120),
                WorkspaceImportPolicy.FileStat(relativePath: ".DS_Store", size: 6148),
            ]
        }

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        guard case .verificationFailed(let findings)? = report.directories.first?.outcome
        else {
            Issue.record(
                "a missing remote file must fail, got \(String(describing: report.directories.first))")
            return
        }
        #expect(findings.contains { $0.contains("generations.jsonl") })
        #expect(!findings.contains { $0.contains(".DS_Store") })
    }

    /// A partial earlier import is FILLED, not refused.
    @Test func aPartialImportHasItsGapsFilled() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [
            ("config.json", 120), ("generations.jsonl", 4096), ("report.json", 300),
        ])
        fake.localFiles[run] = remote(files: [("config.json", 120)])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred == [run])
        #expect(report.imported.map(\.name) == [run])
        #expect(report.violations.isEmpty)
    }

    /// The 2026-09-05 directory, mirrored on both sides: records complete, no
    /// report.json. The merge never finished, so the directory is NOT "already
    /// complete" — and the report says so in words a reader cannot miss.
    @Test func aMergedRunWithoutItsReportIsNeverCertifiedComplete() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        let files = remote(files: [("config.json", 120), ("generations.jsonl", 4096)])
        fake.inventories[run] = files
        fake.localFiles[run] = files

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty)
        #expect(report.imported.isEmpty)
        #expect(report.violations.isEmpty)
        #expect(report.hasIncompleteRuns)
        #expect(report.incompleteRuns.map(\.name) == [run])
        #expect(!report.transferredAnything)
        guard case .incompleteRun(let count, let transferred)? = report.directories.first?.outcome
        else {
            Issue.record(
                "expected incompleteRun, got \(String(describing: report.directories.first))")
            return
        }
        #expect(count == 2)
        #expect(transferred == 0)
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("INCOMPLETE"))
        #expect(text.contains("no report.json"))
        #expect(text.contains("NOT certified complete"))
        #expect(!text.contains("already complete"))
        #expect(!text.contains("imported"))
    }

    /// A fresh import of an unfinished run brings its bytes home — evidence
    /// never stays behind on a purging filesystem — but reports them as
    /// incomplete, never imported. Once the controller's reconciler finishes
    /// the merge, the report is one more gap, and the re-import that fills it
    /// is the ordinary complete import.
    @Test func anUnfinishedRunComesHomeAsIncompleteAndCompletesOnReimport() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [
            ("config.json", 120), ("generations.jsonl", 4096),
        ])

        let first = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred == [run])
        #expect(first.imported.isEmpty)
        #expect(first.hasIncompleteRuns)
        #expect(first.transferredAnything)
        guard case .incompleteRun(let count, let transferred)? = first.directories.first?.outcome
        else {
            Issue.record(
                "expected incompleteRun, got \(String(describing: first.directories.first))")
            return
        }
        #expect(count == 2)
        #expect(transferred == 2)

        // The reconciler completed the merge on the cluster: report.json exists.
        fake.inventories[run] = remote(files: [
            ("config.json", 120), ("generations.jsonl", 4096), ("report.json", 300),
        ])
        let second = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred == [run, run])
        #expect(second.imported.map(\.name) == [run])
        #expect(!second.hasIncompleteRuns)
        guard case .imported(let filled, let bytes)? = second.directories.first?.outcome else {
            Issue.record(
                "expected the gap fill to import, got \(String(describing: second.directories.first))")
            return
        }
        #expect(filled == 1)
        #expect(bytes == 300)
        #expect(fake.localFiles[run]?.contains { $0.relativePath == "report.json" } == true)
    }

    /// Tightening 4: remote bytes that DIFFER from imported local bytes refuse,
    /// loudly, and nothing transfers.
    @Test func byteDriftRefusesAndTransfersNothing() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [("generations.jsonl", 4096)])
        fake.localFiles[run] = remote(files: [("generations.jsonl", 2048)])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty, "a drifted directory must not be rsynced")
        guard case .refusedByteDrift(let message)? = report.directories.first?.outcome else {
            Issue.record("expected a byte-drift refusal")
            return
        }
        #expect(message.contains("immutability violation"))
        #expect(report.violations.count == 1)
    }

    /// A pinned-hash mismatch refuses on the same footing as a size mismatch.
    @Test func pinnedHashDriftRefusesToo() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-optvec-alpha-l20"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [("vector.safetensors", 1024)])
        fake.localFiles[run] = remote(files: [("vector.safetensors", 1024)])
        fake.pinned[run] = [
            WorkspaceImportPolicy.PinnedHash(
                relativePath: "vector.safetensors",
                sha256: String(repeating: "a", count: 64),
                source: "artifact-manifest.json")
        ]
        fake.hashes["\(run)/vector.safetensors"] = String(repeating: "b", count: 64)

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty)
        #expect(report.failures.count == 1)
        #expect(report.violations.first?.contains("vector.safetensors") == true)
    }

    /// Tightening 3: the verb REPORTS purge eligibility and performs no
    /// deletion. The report is gated per tightening 1.
    @Test func thePurgeReportIsGatedByEvidenceAndDeletesNothing() async {
        let fake = FakeImportRemote()
        let merged = "\(stamp)-exp-alpha-run"
        let partials = (0..<2).map { "\(stamp)-exp-alpha-run-shard\($0)of2" }
        let orphans = (0..<2).map { "\(stamp)-exp-gamma-run-shard\($0)of2" }
        fake.directories = [merged] + partials + orphans
        fake.inventories[merged] = remote(files: [("report.json", 64)])
        fake.evidence = [
            WorkspaceImportPolicy.MergeEvidence(
                mergedRun: merged, shardCount: 2, shardRuns: partials)
        ]

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        // Partials never transfer.
        #expect(fake.transferred == [merged])
        let eligible = report.purgeFamilies.filter { $0.verdict.isPurgeEligible }
        let loud = report.purgeFamilies.filter { $0.verdict.isLoud }
        #expect(eligible.map(\.family.stem) == ["alpha"])
        #expect(loud.map(\.family.stem) == ["gamma"])
        #expect(report.hasLoudPurgeFindings)

        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("PURGE ELIGIBILITY"))
        #expect(text.contains("this verb deletes nothing"))
        #expect(text.contains("ORPHANED"))
    }

    /// Bytes the policy leaves on the cluster are named in the purge report —
    /// the other half of "what scratch may now drop".
    @Test func excludedBytesAreReportedAsPurgeablePaths() async {
        let fake = FakeImportRemote()
        let submit = "\(stamp)-submit-alpha-run"
        fake.directories = [submit]
        fake.inventories[submit] = remote(files: [
            ("plan.json", 128),
            ("alpha.evidence-bundle.tar.gz", 30_000_000),
            ("run/adapter-alpha/checkpoints/step-500/optimizer.pt", 40_000_000),
            ("run/adapter-alpha/\(WorkspaceImportPolicy.adapterWeightFileName)", 12_000),
        ])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(report.violations.isEmpty)
        let rules = Set(report.purgeablePaths.map(\.rule))
        #expect(rules == [.evidenceBundleTarball, .trainingCheckpointTree])
        // The final adapter weights came home; only the never-import paths did not.
        let local = fake.localFiles[submit]?.map(\.relativePath) ?? []
        #expect(local.contains("run/adapter-alpha/\(WorkspaceImportPolicy.adapterWeightFileName)"))
        #expect(!local.contains { $0.hasSuffix(".tar.gz") })
        #expect(!local.contains { $0.contains("/checkpoints/") })
    }

    /// A dry run classifies, plans, and reports — and writes nothing anywhere,
    /// including the catalog.
    @Test func aDryRunTransfersNothingAndRebuildsNothing() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [("config.json", 120)])

        let report = await WorkspaceRunImport.run(
            engine: fake.engine(), options: .init(dryRun: true))
        #expect(fake.transferred.isEmpty)
        #expect(fake.catalogRebuilds == 0)
        #expect(report.dryRun)
        #expect(report.imported.map(\.name) == [run])
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("DRY RUN"))
        #expect(text.contains("would import"))
    }

    /// An unreachable cluster refuses before anything is enumerated or written.
    @Test func anUnreachableClusterRefusesEarly() async {
        let fake = FakeImportRemote()
        fake.directories = ["\(stamp)-exp-alpha-run"]
        var engine = fake.engine()
        engine.probeRemote = { throw ExperimentError(reason: "no route to host") }

        let report = await WorkspaceRunImport.run(engine: engine)
        #expect(report.directories.isEmpty)
        #expect(fake.transferred.isEmpty)
        #expect(fake.catalogRebuilds == 0)
        #expect(report.violations.first?.contains("did not answer") == true)
    }

    @Test func unknownShapesAreCarriedIntoTheReport() async {
        let fake = FakeImportRemote()
        let odd = "\(stamp)-nobody-declared-this"
        fake.directories = [odd]
        fake.inventories[odd] = remote(files: [("thing.bin", 8)])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(report.unknowns == [odd])
        #expect(fake.transferred == [odd], "an unknown shape is imported, not skipped")
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("UNKNOWN SHAPES"))
    }

    @Test func sinceSkipsOlderRunsWithoutTouchingThem() async {
        let fake = FakeImportRemote()
        let old = "20260701T090000000-exp-alpha-run"
        let new = "20260819T090000000-exp-beta-run"
        fake.directories = [old, new]
        fake.inventories[old] = remote(files: [("config.json", 1)])
        fake.inventories[new] = remote(files: [("config.json", 1)])

        let report = await WorkspaceRunImport.run(
            engine: fake.engine(),
            options: .init(since: WorkspaceImportPolicy.normalizedSince("2026-08-01")))
        #expect(fake.transferred == [new])
        #expect(report.skippedByPolicy.map(\.name) == [old])
    }

    /// §8 residual (a), end to end: a study attached on the cluster keeps a
    /// shell on the Mac, and the import — which stays runs-only — must say so
    /// loudly instead of leaving the divergence invisible.
    @Test func clusterAuthoredArmsAreReportedAsAuthoringDivergence() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [("experiment.json", 512)])
        fake.runManifestArms[run] = WorkspaceImportPolicy.ManifestArms(
            studyName: "alpha", concepts: 16, conditions: 16)
        fake.liveArms["alpha"] = WorkspaceImportPolicy.ManifestArms(
            studyName: "alpha", concepts: 0, conditions: 0, status: "draft")

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(report.hasAuthoringDivergences)
        #expect(report.authoringDivergences.map(\.studyName) == ["alpha"])
        #expect(report.authoringDivergences.first?.evidencedBy == [run])
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("AUTHORING DIVERGENCE"))
        #expect(text.contains("DIVERGED 1"))
        // A report, never a repair: nothing was transferred beyond the run,
        // and violations stay empty — divergence is loud, not broken.
        #expect(report.violations.isEmpty)
    }

    /// A live manifest that already holds its evidence's arms is silent — the
    /// report exists for loss, not for agreement.
    @Test func aFullyArmedLiveManifestReportsNoDivergence() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [("experiment.json", 512)])
        fake.runManifestArms[run] = WorkspaceImportPolicy.ManifestArms(
            studyName: "alpha", concepts: 2, conditions: 3)
        fake.liveArms["alpha"] = WorkspaceImportPolicy.ManifestArms(
            studyName: "alpha", concepts: 2, conditions: 3, status: "frozen")

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(!report.hasAuthoringDivergences)
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(!text.contains("AUTHORING DIVERGENCE"))
    }
}

// MARK: - Authoring-locus divergence (open-issues §8 residual (a))

struct WorkspaceImportAuthoringDivergenceTests {

    private func arms(
        _ study: String, _ concepts: Int, _ conditions: Int, status: String? = nil
    ) -> WorkspaceImportPolicy.ManifestArms {
        WorkspaceImportPolicy.ManifestArms(
            studyName: study, concepts: concepts, conditions: conditions,
            status: status)
    }

    /// The parser reads the watched pair — and ONLY manifests: bytes with no
    /// decodable `name` are not evidence.
    @Test func manifestArmsParsesTheWatchedPairAndRefusesNonManifests() throws {
        let manifest = """
            {"name": "alpha", "status": "draft",
             "concepts": [{"a": 1}, {"b": 2}], "conditions": [{"c": 3}]}
            """
        let parsed = try #require(
            WorkspaceImportPolicy.manifestArms(
                fromManifestJSON: Data(manifest.utf8)))
        #expect(parsed == arms("alpha", 2, 1, status: "draft"))

        // Absent arrays count zero — the s4x shell's exact shape.
        let shell = WorkspaceImportPolicy.manifestArms(
            fromManifestJSON: Data(#"{"name": "alpha"}"#.utf8))
        #expect(shell == arms("alpha", 0, 0))

        #expect(
            WorkspaceImportPolicy.manifestArms(
                fromManifestJSON: Data(#"{"concepts": [1]}"#.utf8)) == nil,
            "no name, no manifest")
        #expect(
            WorkspaceImportPolicy.manifestArms(
                fromManifestJSON: Data("not json".utf8)) == nil)
    }

    /// The defining case: a live shell whose run evidence carries arms.
    @Test func aShellLiveManifestDivergesFromItsOwnRunEvidence() {
        let divergences = WorkspaceImportPolicy.authoringDivergences(
            snapshots: [("r1", arms("alpha", 16, 16))],
            liveArms: { _ in self.arms("alpha", 0, 0, status: "draft") })
        #expect(divergences.count == 1)
        let divergence = divergences[0]
        #expect(divergence.studyName == "alpha")
        #expect(divergence.evidenceConcepts == 16)
        #expect(divergence.evidenceConditions == 16)
        #expect(divergence.evidencedBy == ["r1"])
        let message = WorkspaceImportPolicy.message(divergence: divergence)
        #expect(message.contains("AUTHORING DIVERGENCE"))
        #expect(message.contains("16 concepts / 16 conditions"))
        #expect(message.contains("never writes experiments/"))
        #expect(message.contains("steerlab-cli experiment verify alpha"))
        #expect(message.contains("runs/r1/experiment.json"))
    }

    /// A study whose manifest never came home AT ALL is the same finding —
    /// but only when the evidence actually holds arms; a shell that stayed on
    /// the cluster holds nothing to lose.
    @Test func aMissingLiveManifestDivergesOnlyWhenEvidenceHoldsArms() {
        let armed = WorkspaceImportPolicy.authoringDivergences(
            snapshots: [("r1", arms("alpha", 4, 0))], liveArms: { _ in nil })
        #expect(armed.count == 1)
        #expect(armed[0].live == nil)
        #expect(
            WorkspaceImportPolicy.message(divergence: armed[0]).contains("absent"))

        let shellOnly = WorkspaceImportPolicy.authoringDivergences(
            snapshots: [("r1", arms("alpha", 0, 0))], liveArms: { _ in nil })
        #expect(shellOnly.isEmpty)
    }

    /// Counts are the comparison: a live copy at or above its evidence is
    /// silent (same-count content drift is the epoch guard's job, not this
    /// report's).
    @Test func aLiveManifestAtOrAboveItsEvidenceIsSilent() {
        let divergences = WorkspaceImportPolicy.authoringDivergences(
            snapshots: [
                ("r1", arms("alpha", 2, 2)), ("r2", arms("beta", 1, 1)),
            ],
            liveArms: { study in
                study == "alpha"
                    ? self.arms("alpha", 2, 2) : self.arms("beta", 3, 3)
            })
        #expect(divergences.isEmpty)
    }

    /// Evidence aggregates per study: the max on each axis, with only the
    /// exceeding runs named — a run whose own snapshot is a shell is not
    /// evidence for the arms.
    @Test func evidenceAggregatesPerStudyAndNamesOnlyExceedingRuns() {
        let divergences = WorkspaceImportPolicy.authoringDivergences(
            snapshots: [
                ("r-old", arms("alpha", 0, 0)),
                ("r-mid", arms("alpha", 8, 16)),
                ("r-new", arms("alpha", 16, 8)),
            ],
            liveArms: { _ in self.arms("alpha", 0, 0) })
        #expect(divergences.count == 1)
        #expect(divergences[0].evidenceConcepts == 16)
        #expect(divergences[0].evidenceConditions == 16)
        #expect(divergences[0].evidencedBy == ["r-mid", "r-new"])
    }
}

// MARK: - The transfer seam (tightening 6)

struct WorkspaceImportTransferSeamTests {

    /// Transfer rides rsync over the shared SSH ControlMaster — the same seam
    /// `cluster push` uses — and never the HTTP API.
    @Test func rsyncArgvRidesTheSharedSSHMaster() {
        var profile = ClusterSiteProfile.exampleCluster
        profile.transport = .ssh(
            host: "user@login.example.edu", proxyJump: nil, remotePort: 8080,
            vpnExpected: true)
        let argv = WorkspaceRunImport.rsyncArgv(
            site: profile, remoteRunRoot: "/scratch/work/runs",
            name: "20260819T101500123-exp-alpha-run",
            destination: URL(filePath: "/tmp/ws/runs/20260819T101500123-exp-alpha-run"),
            rules: WorkspaceImportPolicy.exclusions(for: .submit))
        #expect(argv.first == ClusterProvisioner.rsyncExecutablePath)
        #expect(argv.contains("--ignore-existing"), "the transfer must be incapable of overwriting")
        #expect(argv.contains("*.tar.gz"))
        #expect(argv.contains("checkpoints/"))
        let transportIndex = argv.firstIndex(of: "-e")
        #expect(transportIndex != nil)
        if let transportIndex {
            #expect(argv[transportIndex + 1].contains("ControlPath="))
        }
        #expect(
            argv.contains(
                "user@login.example.edu:/scratch/work/runs/20260819T101500123-exp-alpha-run/"))
        #expect(argv.last?.hasSuffix("/") == true)
    }

    /// The remote inventory parser turns `find -printf '%p\t%s\n'` into
    /// directory-relative stats, and ignores anything outside the requested set.
    @Test func inventoryParsingIsRelativeToTheRunDirectory() throws {
        let lines = [
            "/scratch/runs/20260819T101500123-exp-alpha-run/config.json\t120",
            "/scratch/runs/20260819T101500123-exp-alpha-run/sub/dir/file.jsonl\t4096",
            "/scratch/runs/somebody-elses-tree/file\t9",
            "not a find line",
        ]
        let parsed = try WorkspaceRunImport.parseInventory(
            lines, runRoot: "/scratch/runs",
            names: ["20260819T101500123-exp-alpha-run"])
        #expect(parsed.count == 1)
        let stats = parsed["20260819T101500123-exp-alpha-run"] ?? []
        #expect(
            stats.sorted { $0.relativePath < $1.relativePath }
                == [
                    WorkspaceImportPolicy.FileStat(relativePath: "config.json", size: 120),
                    WorkspaceImportPolicy.FileStat(
                        relativePath: "sub/dir/file.jsonl", size: 4096),
                ])
    }

    /// The setup refusals are typed, with a stable code and a concrete repair.
    @Test func setupRefusalsAreTypedAndActionable() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-import-setup-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var direct = ClusterSiteProfile.exampleCluster
        direct.transport = .direct(baseURL: URL(string: "http://127.0.0.1:8080")!)
        await #expect(
            throws: WorkspaceRunImport.SetupError.noSSHTransport(siteID: "site-a")
        ) {
            _ = try await WorkspaceRunImport.liveEngine(
                site: direct, siteID: "site-a", workspaceRoot: root,
                shell: NeverRunShell())
        }

        // ssh, but no declared storage roots: nothing to enumerate.
        let noRoots = ClusterSiteProfile.exampleCluster
        await #expect(
            throws: WorkspaceRunImport.SetupError.noRemoteRunRoot(siteID: "site-a")
        ) {
            _ = try await WorkspaceRunImport.liveEngine(
                site: noRoots, siteID: "site-a", workspaceRoot: root,
                shell: NeverRunShell())
        }
    }
}

// =============================================================================
// MARK: - The run root, resolved on the far side (2026-08-24 defect)
//
// A site profile may declare its run storage root as a shell EXPRESSION, which
// is the right shape for a site file shared between researchers. The remote
// shell expands it, so `find` prints EXPANDED paths; before this suite the
// client compared them against the UNEXPANDED template, matched nothing, and
// reported every directory empty — which then certified incomplete directories
// complete. The fixtures below are the shape no fixture had: a far side whose
// shell actually expands.
// =============================================================================

/// A far side that expands what its shell is handed. `find` answers with
/// EXPANDED paths, exactly as a real cluster does.
private final class ExpandingRemoteShell: ClusterShellRunner, @unchecked Sendable {
    // @unchecked Sendable: written only by the serialized test body and the
    // operation under test; never escapes the test.
    let expandedRoot: String
    var files: [(path: String, size: Int64)] = []
    /// Overrides the round trip's reply: nil answers normally.
    var expansionReply: ClusterShellResult?
    private(set) var commands: [String] = []

    init(expandedRoot: String) {
        self.expandedRoot = expandedRoot
    }

    /// Every remote command word the far side was handed, joined — the argv's
    /// last element is the whole remote command string.
    var findCommands: [String] {
        commands.filter { $0.hasPrefix("find") }
    }

    func run(_ argv: [String]) async -> ClusterShellResult {
        let command = argv.last ?? ""
        commands.append(command)
        if command.contains(WorkspaceRunImport.runRootAnswerMarker) {
            return expansionReply
                ?? ClusterShellResult(
                    exitCode: 0,
                    lines: ["\(WorkspaceRunImport.runRootAnswerMarker)\(expandedRoot)"])
        }
        if command.hasPrefix("find") {
            return ClusterShellResult(
                exitCode: 0,
                lines: files.map { "\(expandedRoot)/\($0.path)\t\($0.size)" })
        }
        return ClusterShellResult(exitCode: 0)
    }
}

struct WorkspaceRunImportRemoteRootTests {

    private let declaredRoot = "/scratch/${USER:-$(id -un)}/ws/runs"
    private let expandedRoot = "/scratch/someone/ws/runs"
    private let run = "20260819T101500123-exp-alpha-run"

    private func siteDeclaringAnExpression() -> ClusterSiteProfile {
        var profile = ClusterSiteProfile.exampleCluster
        profile.transport = .ssh(
            host: "user@login.example.edu", proxyJump: nil, remotePort: 8080,
            vpnExpected: false)
        profile.constraints.storageRoots["run"] = declaredRoot
        return profile
    }

    private func workspace() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-import-root-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    /// The regression the field report asked for: a declared shell expression
    /// plus a far side that expands it. The inventory must be NON-EMPTY, and
    /// the `find` must have been given the expanded root — the prefix the
    /// client compares is byte-derived from the same string.
    @Test func anExpandedRunRootIsUsedForBothTheFindAndThePrefix() async throws {
        let root = try workspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let shell = ExpandingRemoteShell(expandedRoot: expandedRoot)
        shell.files = [
            (path: "\(run)/config.json", size: 120),
            (path: "\(run)/generations.jsonl", size: 4096),
        ]

        let engine = try await WorkspaceRunImport.liveEngine(
            site: siteDeclaringAnExpression(), siteID: "site-a",
            workspaceRoot: root, shell: shell)
        let inventory = try await engine.remoteInventory([run])

        let stats = try #require(inventory[run])
        #expect(
            stats.sorted { $0.relativePath < $1.relativePath }
                == [
                    WorkspaceImportPolicy.FileStat(relativePath: "config.json", size: 120),
                    WorkspaceImportPolicy.FileStat(
                        relativePath: "generations.jsonl", size: 4096),
                ])
        let find = try #require(shell.findCommands.last)
        #expect(find.contains(expandedRoot))
        #expect(!find.contains(declaredRoot))
    }

    /// A round trip that fails, says nothing, or says several things REFUSES.
    /// Falling back to the declared template is the defect itself.
    @Test func anUnresolvableRunRootRefusesInsteadOfFallingBack() async throws {
        let root = try workspace()
        defer { try? FileManager.default.removeItem(at: root) }

        let replies: [ClusterShellResult] = [
            ClusterShellResult(exitCode: 255, lines: ["connection closed"]),
            ClusterShellResult(exitCode: 0, lines: ["motd: welcome"]),
            ClusterShellResult(
                exitCode: 0,
                lines: [
                    "\(WorkspaceRunImport.runRootAnswerMarker)/scratch/a/runs",
                    "\(WorkspaceRunImport.runRootAnswerMarker)/scratch/b/runs",
                ]),
        ]
        for reply in replies {
            let shell = ExpandingRemoteShell(expandedRoot: expandedRoot)
            shell.expansionReply = reply
            do {
                _ = try await WorkspaceRunImport.liveEngine(
                    site: siteDeclaringAnExpression(), siteID: "site-a",
                    workspaceRoot: root, shell: shell)
                Issue.record("an unresolvable run root must refuse")
            } catch let error as WorkspaceRunImport.SetupError {
                #expect(error.code == "runRootUnresolved")
                #expect(error.reason.contains(declaredRoot))
                #expect(error.repairAction.contains("site-a"))
            }
            #expect(shell.findCommands.isEmpty, "nothing may be enumerated after a refusal")
        }
    }

    /// The loud discard. `find` printed real lines and not one of them sat
    /// under the root we compared against: a defect, named, with the expected
    /// prefix and one observed path — never an empty directory.
    @Test func aListingUnderNoKnownRootIsATypedErrorNotAnEmptyResult() throws {
        let lines = [
            "\(expandedRoot)/\(run)/config.json\t120",
            "\(expandedRoot)/\(run)/report.json\t64",
        ]
        // The pre-fix comparison: expanded output against the declared template.
        do {
            _ = try WorkspaceRunImport.parseInventory(
                lines, runRoot: declaredRoot, names: [run])
            Issue.record("a listing under no known root must throw")
        } catch let error as WorkspaceRunImport.InventoryError {
            #expect(error.code == "inventoryPrefixMismatch")
            #expect(error.reason.contains(declaredRoot))
            #expect(error.reason.contains("\(expandedRoot)/\(run)/config.json"))
        }

        // A genuinely empty listing is still an empty result, not an error.
        #expect(try WorkspaceRunImport.parseInventory(
            [], runRoot: expandedRoot, names: [run]).isEmpty)
    }
}

/// A shell that must never be reached — every refusal it guards happens before
/// any command runs.
private struct NeverRunShell: ClusterShellRunner {
    func run(_ argv: [String]) async -> ClusterShellResult {
        Issue.record("no command may run: \(argv.joined(separator: " "))")
        return ClusterShellResult(exitCode: 1)
    }
}

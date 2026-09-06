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
            // A bundle submission's receipt is a receipt like any other:
            // the `submit-` origin decides, not the word after it.
            ("submit-bundle-alpha-run", .submit),
            ("submit-bundle-alpha-verify", .submit),
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

    /// An upload's staging directory is recognized, and skipped BY POLICY
    /// rather than reported as an unknown shape: it holds only the bundle a
    /// client sent, which that client's workspace already has.
    @Test func uploadStagingDirectoriesAreSkippedByPolicyNotUnknown() {
        let classification = classify("uploaded-bundle")
        #expect(classification.kind == .uploadStaging)
        #expect(!classification.kind.isAlwaysImported)
        let decision = WorkspaceImportPolicy.decision(for: classification)
        #expect(!decision.transfers)
        guard case .notApplicable(let reason) = decision else {
            Issue.record("expected a policy skip, got \(decision)")
            return
        }
        #expect(reason.contains("staging directory"))
        #expect(reason.contains("client"))
    }

    /// A `-reimport<N>` copy — what `--reimport-drifted` makes of a drifted
    /// directory — classifies exactly as its original: same kind, stamp, and
    /// family, so the catalog groups the pair and every rule sees the run it
    /// is. The ordinal is carried; the shape is not changed by it.
    @Test func reimportCopiesClassifyAsTheirOriginal() {
        let original = classify("submit-bundle-alpha-run")
        let copy = classify("submit-bundle-alpha-run-reimport")
        let second = classify("submit-bundle-alpha-run-reimport2")
        #expect(copy.kind == .submit)
        #expect(second.kind == .submit)
        #expect(copy.stem == original.stem)
        #expect(copy.stamp == original.stamp)
        #expect(original.reimportOrdinal == nil)
        #expect(copy.reimportOrdinal == 1)
        #expect(second.reimportOrdinal == 2)
        // The suffix is stripped BEFORE the verb is read, so a stage copy
        // keeps its stage instead of falling to the `.run` default.
        let evaluate = classify("exp-alpha-evaluate-reimport")
        #expect(evaluate.kind == .evaluate)
        #expect(evaluate.stem == "alpha")
        #expect(WorkspaceImportPolicy.reimportName("x", ordinal: 1) == "x-reimport")
        #expect(WorkspaceImportPolicy.reimportName("x", ordinal: 3) == "x-reimport3")
        // A study whose own name carries the word is not a copy: only a LAST
        // token, after the verb, is the suffix.
        let named = classify("exp-reimport-run")
        #expect(named.reimportOrdinal == nil)
        #expect(named.kind == .run)
        #expect(named.stem == "reimport")
    }

    // MARK: The in-progress gate

    /// The completion artifacts are the ENGINE's own: the study report for a
    /// run, the validation report for a validate, the coding or judge report
    /// for an evaluate, the recommendations file for a sweep. Every other
    /// shape declares none, and the gate never holds one back.
    @Test func completionArtifactsAreTheEnginesOwnAndOnlyForStagesThatWriteOne() {
        #expect(WorkspaceImportPolicy.DirectoryKind.run.completionArtifacts == ["report.json"])
        #expect(
            WorkspaceImportPolicy.DirectoryKind.validate.completionArtifacts
                == ["validation-report.json", "report.json"])
        #expect(
            WorkspaceImportPolicy.DirectoryKind.evaluate.completionArtifacts
                == ["coding-report.json", "judge-report.json"])
        #expect(
            WorkspaceImportPolicy.DirectoryKind.sweep.completionArtifacts
                == ["recommendations.json"])
        for kind in WorkspaceImportPolicy.DirectoryKind.allCases
        where ![.run, .validate, .evaluate, .sweep].contains(kind) {
            #expect(kind.completionArtifacts.isEmpty, "\(kind.rawValue) must not be gated")
            #expect(
                WorkspaceImportPolicy.awaitedCompletionArtifacts(for: kind, remote: []) == nil,
                "\(kind.rawValue) must never read as in progress")
        }
    }

    /// The artifact must sit at the directory's ROOT: a chain's
    /// `stage-1/report.json` belongs to that stage, and a run whose only
    /// report is nested has not finished.
    @Test func theGateReadsTheDirectoryRootAndAnyOneArtifactSuffices() {
        func stat(_ path: String) -> WorkspaceImportPolicy.FileStat {
            WorkspaceImportPolicy.FileStat(relativePath: path, size: 1)
        }
        #expect(
            WorkspaceImportPolicy.awaitedCompletionArtifacts(
                for: .evaluate, remote: [stat("config.json"), stat("codings.jsonl")])
                == ["coding-report.json", "judge-report.json"])
        #expect(
            WorkspaceImportPolicy.awaitedCompletionArtifacts(
                for: .evaluate, remote: [stat("codings.jsonl"), stat("coding-report.json")])
                == nil)
        #expect(
            WorkspaceImportPolicy.awaitedCompletionArtifacts(
                for: .evaluate, remote: [stat("judgments.jsonl"), stat("judge-report.json")])
                == nil)
        #expect(
            WorkspaceImportPolicy.awaitedCompletionArtifacts(
                for: .run, remote: [stat("stage-1/report.json")]) == ["report.json"])
        #expect(
            WorkspaceImportPolicy.awaitedCompletionArtifacts(
                for: .run, remote: [stat("report.json")]) == nil)
        // An empty inventory has no artifact in it either.
        #expect(
            WorkspaceImportPolicy.awaitedCompletionArtifacts(for: .run, remote: [])
                == ["report.json"])
    }

    /// The reason names the directory, what it is waiting for, both readings
    /// of an absent report, and — when an earlier import already brought
    /// part of it home — that the local partial will read as drift later.
    @Test func theInProgressReasonNamesTheArtifactAndBothReadings() {
        let plain = WorkspaceImportPolicy.inProgressReason(
            directory: "\(stamp)-exp-alpha-evaluate",
            awaiting: ["coding-report.json", "judge-report.json"], localFiles: 0)
        #expect(plain.contains("\(stamp)-exp-alpha-evaluate"))
        #expect(plain.contains("coding-report.json or judge-report.json"))
        #expect(plain.contains("never rewrites"))
        #expect(plain.contains("Import again once the job completes"))
        #expect(plain.contains("died before writing its report"))
        #expect(!plain.contains("earlier import"))

        let partial = WorkspaceImportPolicy.inProgressReason(
            directory: "\(stamp)-exp-alpha-evaluate",
            awaiting: ["coding-report.json", "judge-report.json"], localFiles: 3)
        #expect(partial.contains("3 files from an earlier import already sit here"))
        #expect(partial.contains("drifted"))
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

// MARK: - The receipt gate (the Slurm jobs a directory names)

struct WorkspaceImportPolicyReceiptGateTests {

    private func stat(_ path: String, _ size: Int64 = 1) -> WorkspaceImportPolicy.FileStat {
        WorkspaceImportPolicy.FileStat(relativePath: path, size: size)
    }

    private func job(
        _ id: String, in directory: String = "slurm", marked: Bool = false
    ) -> WorkspaceImportPolicy.NamedJob {
        WorkspaceImportPolicy.NamedJob(
            id: id, bundleDirectory: directory, hasEndMarker: marked,
            captures: ["\(directory)/slurm-\(id).out"])
    }

    /// A receipt names its jobs by the scheduler's own captures — one bundle
    /// directory per sbatch — and the engine's end marker beside a capture
    /// settles that job by content. Records, scripts, and manifests name
    /// nothing; neither does a marker with no capture beside it.
    @Test func namedJobsAreReadOffTheCaptureNames() {
        let jobs = WorkspaceImportPolicy.namedJobs(remote: [
            stat("records/0123456789ab.json", 400),
            stat("slurm/run.sbatch", 2_000), stat("slurm/bundle.json", 900),
            stat("slurm/slurm-47923657.out", 143), stat("slurm/slurm-47923657.err", 0),
            stat("slurm-shard-1/slurm-47923658.out", 14_336),
            stat("slurm-shard-1/slurm-47923658.exit", 2),
            stat("slurm-shard-0/slurm-47923659.out", 14_336),
            stat("slurm-judge-0-x/slurm-4.exit", 2),
            stat("slurm/slurm-notanid.out", 5),
            stat("slurm/slurm-.out", 5),
        ])
        #expect(jobs.map(\.id) == ["47923657", "47923659", "47923658"])
        #expect(jobs.map(\.bundleDirectory) == ["slurm", "slurm-shard-0", "slurm-shard-1"])
        #expect(jobs.map(\.hasEndMarker) == [false, false, true])
        #expect(jobs[0].captures == ["slurm/slurm-47923657.err", "slurm/slurm-47923657.out"])
        #expect(jobs[2].endMarkerPath == "slurm-shard-1/slurm-47923658.exit")
        // A stage directory carries no captures and names no jobs.
        #expect(
            WorkspaceImportPolicy.namedJobs(remote: [
                stat("config.json"), stat("generations.jsonl"), stat("report.json"),
            ]).isEmpty)
        // A capture at the directory root has an empty bundle directory.
        let root = WorkspaceImportPolicy.namedJobs(remote: [stat("slurm-7.out")])
        #expect(root.map(\.bundleDirectory) == [""])
        #expect(root.first?.endMarkerPath == "slurm-7.exit")
    }

    /// The scheduler's answers, read the way the engine's own poll reads
    /// them: `squeue` presence is live (whatever `sacct` says); a terminal
    /// `sacct` state is ended; a requeue-class state is still alive; an id
    /// neither knows is ended only when both queries answered.
    @Test func schedulerAnswersAreInterpretedLikeTheEngines() {
        let states = WorkspaceImportPolicy.schedulerStates(
            requested: ["1", "2", "3", "4", "5", "6"],
            squeueRows: ["1|RUNNING", "2|PENDING", "999|RUNNING", "garbage"],
            sacctRows: ["3|COMPLETED", "4|CANCELLED by 1234", "5|REQUEUED", "1|COMPLETED"],
            sacctAnswered: true)
        #expect(states["1"] == .live("RUNNING"))
        #expect(states["2"] == .live("PENDING"))
        #expect(states["3"] == .ended("COMPLETED"))
        #expect(states["4"] == .ended("CANCELLED"))
        #expect(states["5"] == .live("REQUEUED"))
        #expect(states["6"] == .ended("not known to squeue or sacct"))
        #expect(states["999"] == nil, "only requested ids are answered")

        let silent = WorkspaceImportPolicy.schedulerStates(
            requested: ["6"], squeueRows: [], sacctRows: [], sacctAnswered: false)
        #expect(silent["6"] == .unknown("sacct did not answer"))

        let terminal = ["TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "FAILED", "BOOT_FAIL", "DEADLINE"]
        for state in terminal {
            #expect(
                WorkspaceImportPolicy.schedulerStates(
                    requested: ["9"], squeueRows: [], sacctRows: ["9|\(state)"],
                    sacctAnswered: true)["9"] == .ended(state))
        }
        for state in ["PREEMPTED", "SUSPENDED", "RESIZING", "COMPLETING"] {
            #expect(
                WorkspaceImportPolicy.schedulerStates(
                    requested: ["9"], squeueRows: [], sacctRows: ["9|\(state)"],
                    sacctAnswered: true)["9"] == .live(state))
        }
    }

    /// The gate: a marked job is settled by content and never looked up; an
    /// unmarked one is held unless the scheduler said ENDED; an id the
    /// scheduler did not answer for — or a query that threw — is UNKNOWN,
    /// and unknown holds rather than guesses.
    @Test func markedJobsAreSettledByContentAndTheRestByTheScheduler() {
        let marked = job("1", marked: true)
        let running = job("2", in: "slurm-shard-1")
        let ended = job("3", in: "slurm-shard-0")
        let unasked = job("4")
        let held = WorkspaceImportPolicy.heldJobs(
            [marked, running, ended, unasked],
            states: ["2": .live("RUNNING"), "3": .ended("COMPLETED")])
        #expect(held.map(\.job.id) == ["2", "4"])
        #expect(held[0].state == "RUNNING")
        #expect(!held[0].isUnknown)
        #expect(held[1].isUnknown)
        #expect(held[1].state == "the scheduler was not asked")

        // A marker outranks a scheduler that still calls the job live.
        #expect(WorkspaceImportPolicy.heldJobs([marked], states: ["1": .live("RUNNING")]).isEmpty)
        // A failed query is carried into the reason, and still holds.
        let failed = WorkspaceImportPolicy.heldJobs(
            [unasked], states: [:], failure: "squeue exited 255")
        #expect(failed.first?.state == "squeue exited 255")
        #expect(failed.first?.isUnknown == true)
        let unknown = WorkspaceImportPolicy.heldJobs(
            [unasked], states: ["4": .unknown("sacct did not answer")])
        #expect(unknown.first?.state == "sacct did not answer")
        #expect(WorkspaceImportPolicy.heldJobs([], states: [:]).isEmpty)
    }

    /// The hold reason names each job, where its captures live, the
    /// scheduler's word, what ends the hold, and the frozen-partial caveat.
    @Test func theLiveJobsReasonNamesJobsStatesAndTheRepair() {
        let running = WorkspaceImportPolicy.HeldJob(
            job: job("47923657", in: "slurm-shard-1"), state: "RUNNING", isUnknown: false)
        let plain = WorkspaceImportPolicy.liveJobsReason(
            directory: "d", held: [running], localFiles: 0)
        #expect(plain.contains("'d' names 1 Slurm job that has not ended"))
        #expect(plain.contains("job 47923657 (slurm-shard-1/): RUNNING"))
        #expect(plain.contains("slurm-<jobid>.exit"))
        #expect(plain.contains("Import again once the job has ended"))
        #expect(!plain.contains("earlier import"))
        #expect(!plain.contains("auth open"))

        let partial = WorkspaceImportPolicy.liveJobsReason(
            directory: "d", held: [running], localFiles: 3)
        #expect(partial.contains("3 files from an earlier import already sit here"))
        #expect(partial.contains("--reimport-drifted"))

        let unknown = WorkspaceImportPolicy.HeldJob(
            job: job("5"), state: "sacct did not answer", isUnknown: true)
        let refused = WorkspaceImportPolicy.liveJobsReason(
            directory: "d", held: [unknown], localFiles: 0)
        #expect(refused.contains("whose state the scheduler could not report"))
        #expect(refused.contains("job 5 (slurm/): state unknown — sacct did not answer"))
        #expect(refused.contains("refusal to guess"))
        #expect(refused.contains("auth open"))

        let mixed = WorkspaceImportPolicy.liveJobsReason(
            directory: "d", held: [running, unknown], localFiles: 0)
        #expect(mixed.contains("names 2 Slurm jobs not yet ended, or whose state"))
    }

    /// The immutability refusal now ends with the repair: the copy name and
    /// the exact command. Earlier copies that drifted too are named, and the
    /// bare form (no copy offered) carries no repair sentence.
    @Test func theRefusalNamesTheReimportCopyAndTheCommand() {
        let drift: [WorkspaceImportPolicy.Finding] = [
            .sizeDrift(relativePath: "slurm/slurm-1.out", remote: 14_336, local: 143)
        ]
        let text = WorkspaceImportPolicy.immutabilityRefusal(
            directory: "d", violations: drift, reimportCopy: "d-reimport")
        #expect(text.contains("Nothing was overwritten"))
        #expect(text.contains("'d-reimport'"))
        #expect(text.contains("`steerlab-cli cluster import --site <id> --reimport-drifted`"))
        #expect(text.contains("resolved instead of as a violation"))

        let again = WorkspaceImportPolicy.immutabilityRefusal(
            directory: "d", violations: drift, reimportCopy: "d-reimport2",
            driftedCopies: ["d-reimport"])
        #expect(again.contains("d-reimport is here from an earlier --reimport-drifted"))
        #expect(again.contains("'d-reimport2'"))

        let bare = WorkspaceImportPolicy.immutabilityRefusal(directory: "d", violations: drift)
        #expect(!bare.contains("--reimport-drifted"))
        #expect(bare.contains("under a different name"))
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
    /// What the fake scheduler says per Slurm job id; an id absent here is
    /// unknown to it, exactly as a live query that returned no row.
    var schedulerStates: [String: WorkspaceImportPolicy.SchedulerJobState] = [:]
    /// When set, the scheduler seam throws this instead of answering.
    var schedulerFailure: ExperimentError?
    /// Every id list the operation asked the scheduler about.
    private(set) var schedulerAsked: [[String]] = []
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
            transfer: { remoteName, localName, rules in
                self.transferred.append(
                    remoteName == localName ? remoteName : "\(remoteName) -> \(localName)")
                if let override = self.transferOverride {
                    override()
                    return
                }
                // The live transfer is `rsync --ignore-existing`: it can only
                // ever ADD files the policy keeps. The fake obeys the same
                // rule, so a re-import over a complete tree is a no-op here
                // exactly as it is there. The destination may be a reimport
                // copy of the source, which is the one case the names differ.
                var local = self.localFiles[localName] ?? []
                let known = Set(local.map(\.relativePath))
                for file in self.inventories[remoteName] ?? []
                where !known.contains(file.relativePath)
                    && !WorkspaceImportPolicy.isExcluded(
                        relativePath: file.relativePath, rules: rules)
                {
                    local.append(file)
                }
                self.localFiles[localName] = local
                self.transferHook?()
            },
            remoteSchedulerStates: { ids in
                self.schedulerAsked.append(ids)
                if let failure = self.schedulerFailure { throw failure }
                return self.schedulerStates.filter { ids.contains($0.key) }
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
        fake.inventories[run] = remote(files: [("config.json", 120), ("report.json", 64)])
        fake.localFiles[run] = remote(files: [("config.json", 120), ("report.json", 64)])

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
            ("config.json", 120), ("generations.jsonl", 4096), ("report.json", 64),
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
        #expect(localFiles == 3)
        let violation = report.violations.joined(separator: "\n")
        #expect(violation.contains(run))
        #expect(violation.contains("inventory failed"))
        #expect(violation.contains("remote run really is gone"))
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("REFUSED"))
        #expect(!text.contains("already complete"))
    }

    /// …and a directory that is empty on BOTH sides is not a refusal: there is
    /// nothing there to be wrong about. (A receipt, which no completion
    /// artifact gates; an empty RUN directory is a stage that has not
    /// written its report, and is held back as in progress instead.)
    @Test func anEmptyDirectoryOnBothSidesIsStillComplete() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-alpha-run"
        fake.directories = [receipt]
        fake.inventories[receipt] = []
        fake.localFiles[receipt] = []

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(report.violations.isEmpty)
        guard case .alreadyComplete = report.directories.first?.outcome else {
            Issue.record(
                "expected alreadyComplete, got \(String(describing: report.directories.first))")
            return
        }
    }

    // MARK: The in-progress gate, end to end

    /// The field case (2026-09-02): an evaluate whose job is still running —
    /// its record file partly written, its report not yet — was classified
    /// as importable. Now it is held back, by CONTENT, and the report says
    /// why; nothing is transferred and the catalog still rebuilds.
    @Test func aStageWithoutItsCompletionArtifactIsSkippedAsInProgress() async {
        let fake = FakeImportRemote()
        let evaluate = "\(stamp)-exp-alpha-evaluate"
        fake.directories = [evaluate]
        fake.inventories[evaluate] = remote(files: [
            ("config.json", 120), ("codings.jsonl", 26_800),
        ])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty, "an unfinished stage must not be transferred")
        #expect(fake.localFiles[evaluate] == nil, "nothing may be created locally")
        #expect(report.imported.isEmpty)
        #expect(report.violations.isEmpty)
        #expect(report.skippedInProgress.map(\.name) == [evaluate])
        #expect(report.skippedByPolicy.isEmpty, "in progress is its own key, not a policy skip")
        guard
            case .skippedInProgress(let awaiting, let localFiles)?
                = report.directories.first?.outcome
        else {
            Issue.record("expected in progress, got \(String(describing: report.directories.first))")
            return
        }
        #expect(awaiting == ["coding-report.json", "judge-report.json"])
        #expect(localFiles == 0)
        #expect(fake.catalogRebuilds == 1)

        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("\(evaluate)  [evaluate]  skipped — in progress: no coding-report.json or judge-report.json on the cluster yet"))
        #expect(text.contains("IN PROGRESS"))
        #expect(text.contains("has not finished"))
        #expect(text.contains("in progress 1"))
        #expect(!text.contains("would import"))
        #expect(!text.contains("imported 1"))
    }

    /// Once the report lands, the same directory transfers on the next
    /// import — the gate defers, it never strands.
    @Test func theSameDirectoryTransfersOnceItsReportLands() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        // The records are landing; the report is not there yet.
        fake.inventories[run] = remote(files: [("config.json", 120), ("generations.jsonl", 4096)])

        let first = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(first.skippedInProgress.map(\.name) == [run])
        #expect(fake.transferred.isEmpty)

        fake.inventories[run]! += remote(files: [("report.json", 64)])
        let second = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(second.skippedInProgress.isEmpty)
        #expect(second.imported.map(\.name) == [run])
        #expect(fake.transferred == [run])
        #expect(second.violations.isEmpty)
    }

    /// A partial copy an EARLIER import froze here (the pre-gate damage) does
    /// not turn a still-running stage into a drift violation: the gate comes
    /// before the drift check, holds the directory back, and says that the
    /// partial is here. Drift is judged when the stage has finished.
    @Test func aFrozenPartialCopyOfARunningStageIsHeldBackNotRefused() async {
        let fake = FakeImportRemote()
        let evaluate = "\(stamp)-exp-alpha-evaluate"
        fake.directories = [evaluate]
        fake.inventories[evaluate] = remote(files: [
            ("config.json", 120), ("codings.jsonl", 120_000),
        ])
        fake.localFiles[evaluate] = remote(files: [
            ("config.json", 120), ("codings.jsonl", 26_800),
        ])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty)
        #expect(report.violations.isEmpty, "drift is not judged on an unfinished stage")
        #expect(report.failures.isEmpty)
        guard
            case .skippedInProgress(_, let localFiles)? = report.directories.first?.outcome
        else {
            Issue.record("expected in progress, got \(String(describing: report.directories.first))")
            return
        }
        #expect(localFiles == 2)
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("2 files from an earlier import already here"))
        #expect(text.contains("will refuse the directory as drifted"))
    }

    /// The empty-inventory refusal keeps precedence over the gate: a populated
    /// local directory the cluster reports no files for is the inventory in
    /// question, never a stage "still running".
    @Test func anEmptyRemoteInventoryStillRefusesRatherThanReadingAsInProgress() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = []
        fake.localFiles[run] = remote(files: [("report.json", 64)])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(report.skippedInProgress.isEmpty)
        guard case .refusedEmptyRemoteInventory = report.directories.first?.outcome else {
            Issue.record("expected a refusal, got \(String(describing: report.directories.first))")
            return
        }
    }

    /// A running job's excluded bytes (its checkpoint tree) are not listed as
    /// purgeable: the resume reads them, and nothing about the directory is
    /// settled until it finishes.
    @Test func anInProgressDirectoryContributesNoPurgeablePaths() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [
            ("config.json", 120), ("checkpoints/step-100/state.pt", 5_000_000),
        ])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(report.skippedInProgress.map(\.name) == [run])
        #expect(report.purgeablePaths.isEmpty)
    }

    /// The gate over a directory on disk, through the SAME inventory walker
    /// the live engine uses: a fixture that lacks its completion artifact
    /// reads as in progress on a dry run, and as importable once the report
    /// is written beside it.
    @Test func aFixtureDirectoryWithoutItsReportReadsAsInProgressOnADryRun() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-import-in-progress-\(UUID().uuidString)")
        let evaluate = "\(stamp)-exp-alpha-evaluate"
        let directory = root.appending(component: evaluate)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"stage": "evaluate"}"#.utf8)
            .write(to: directory.appending(component: "config.json"))
        // A record file mid-write: some rows, no report.
        try Data(
            (0..<3).map { #"{"row": \#($0)}"# }.joined(separator: "\n").utf8
        ).write(to: directory.appending(component: "codings.jsonl"))

        let fake = FakeImportRemote()
        fake.directories = [evaluate]
        fake.inventories[evaluate] = WorkspaceRunImport.localInventory(at: directory)
        #expect(fake.inventories[evaluate]?.count == 2)

        let dry = await WorkspaceRunImport.run(
            engine: fake.engine(), options: .init(dryRun: true))
        #expect(dry.dryRun)
        #expect(fake.transferred.isEmpty)
        #expect(fake.catalogRebuilds == 0)
        #expect(dry.skippedInProgress.map(\.name) == [evaluate])
        #expect(dry.imported.isEmpty)
        let text = WorkspaceRunImport.summaryLines(dry).joined(separator: "\n")
        #expect(text.contains("DRY RUN"))
        #expect(text.contains("\(evaluate)  [evaluate]  skipped — in progress"))
        #expect(text.contains("no coding-report.json or judge-report.json on the cluster yet"))
        #expect(!text.contains("would import"))

        // The job finishes: the report lands beside the records.
        try Data(#"{"codings": 3}"#.utf8)
            .write(to: directory.appending(component: WorkspaceImportPolicy.codingReportFileName))
        fake.inventories[evaluate] = WorkspaceRunImport.localInventory(at: directory)
        let again = await WorkspaceRunImport.run(
            engine: fake.engine(), options: .init(dryRun: true))
        #expect(again.skippedInProgress.isEmpty)
        #expect(again.imported.map(\.name) == [evaluate])
        #expect(
            WorkspaceRunImport.summaryLines(again).joined(separator: "\n")
                .contains("would import 3 files"))
    }

    /// An upload's staging directory is skipped by policy — never transferred,
    /// never an unknown shape — and says why.
    @Test func uploadStagingIsSkippedByPolicyEndToEnd() async {
        let fake = FakeImportRemote()
        let staging = "\(stamp)-uploaded-bundle"
        fake.directories = [staging]
        fake.inventories[staging] = remote(files: [("alpha.run-bundle.tar.gz", 4_000_000)])

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty)
        #expect(report.unknowns.isEmpty)
        #expect(report.skippedByPolicy.map(\.name) == [staging])
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("\(staging)  skipped — "))
        #expect(text.contains("staging directory"))
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
            ("config.json", 120), ("generations.jsonl", 4096), ("report.json", 64),
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
    /// complete" — it is held back as in progress (precedence decision,
    /// 2026-09-05: the in-progress gate runs before the incomplete-run
    /// classification), the reason names the partial copy already here, and
    /// nothing is transferred.
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
        #expect(!report.hasIncompleteRuns)
        #expect(report.skippedInProgress.map(\.name) == [run])
        #expect(!report.transferredAnything)
        guard case .skippedInProgress(let awaiting, let localFiles)? = report.directories.first?.outcome
        else {
            Issue.record(
                "expected in progress, got \(String(describing: report.directories.first))")
            return
        }
        #expect(awaiting == ["report.json"])
        #expect(localFiles == 2)
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("skipped — in progress"))
        #expect(text.contains("no report.json on the cluster yet"))
        #expect(text.contains("IN PROGRESS"))
        #expect(!text.contains("already complete"))
        #expect(!text.contains("INCOMPLETE"))
    }

    /// A fresh import of an unfinished run brings nothing home — a growing
    /// record stream copied mid-write would be frozen locally for good, so the
    /// in-progress gate holds it back (precedence decision, 2026-09-05). Once
    /// the controller's reconciler finishes the merge and the report exists,
    /// the next import is the ordinary complete import.
    @Test func anUnfinishedRunIsHeldBackAndImportsOnceItsReportExists() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [
            ("config.json", 120), ("generations.jsonl", 4096),
        ])

        let first = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty)
        #expect(fake.localFiles[run] == nil, "nothing may be created locally")
        #expect(first.imported.isEmpty)
        #expect(!first.hasIncompleteRuns)
        #expect(first.skippedInProgress.map(\.name) == [run])
        #expect(!first.transferredAnything)
        guard case .skippedInProgress(let awaiting, let localFiles)? = first.directories.first?.outcome
        else {
            Issue.record(
                "expected in progress, got \(String(describing: first.directories.first))")
            return
        }
        #expect(awaiting == ["report.json"])
        #expect(localFiles == 0)

        // The reconciler completed the merge on the cluster: report.json exists.
        fake.inventories[run] = remote(files: [
            ("config.json", 120), ("generations.jsonl", 4096), ("report.json", 300),
        ])
        let second = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred == [run])
        #expect(second.imported.map(\.name) == [run])
        #expect(second.skippedInProgress.isEmpty)
        #expect(!second.hasIncompleteRuns)
        guard case .imported(let files, let bytes)? = second.directories.first?.outcome else {
            Issue.record(
                "expected the finished run to import, got \(String(describing: second.directories.first))")
            return
        }
        #expect(files == 3)
        #expect(bytes == 120 + 4096 + 300)
        #expect(fake.localFiles[run]?.contains { $0.relativePath == "report.json" } == true)
    }

    // MARK: The receipt gate, end to end

    /// A sharded receipt as the engine writes it: one bundle directory per
    /// shard, the scheduler's captures inside, a child record per finished
    /// shard, and — for a shard whose script ran its EXIT trap — the end
    /// marker beside its capture.
    private func receiptInventory(
        runningCaptureBytes: Int64 = 143, finishedMarked: Bool = true
    ) -> [WorkspaceImportPolicy.FileStat] {
        var files: [(String, Int64)] = [
            ("slurm-shard-0/run.sbatch", 2_000), ("slurm-shard-0/bundle.json", 900),
            ("slurm-shard-0/slurm-47923657.out", runningCaptureBytes),
            ("slurm-shard-0/slurm-47923657.err", 0),
            ("slurm-shard-1/run.sbatch", 2_000), ("slurm-shard-1/bundle.json", 900),
            ("slurm-shard-1/slurm-47923658.out", 14_336),
            ("slurm-shard-1/slurm-47923658.err", 0),
            ("records/0123456789ab.json", 400),
        ]
        if finishedMarked { files.append(("slurm-shard-1/slurm-47923658.exit", 2)) }
        return remote(files: files)
    }

    /// The field case (2026-09-04): a sharded receipt imported while its
    /// shards ran froze 143-byte captures that were 14 KB on the cluster by
    /// the next import — an immutability violation no re-import could clear.
    /// Now a receipt naming a job the scheduler still lists is held back
    /// whole, nothing is created locally, and the marked shard is never
    /// asked about.
    @Test func aReceiptWhoseJobIsStillRunningIsSkippedAsInProgress() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        fake.directories = [receipt]
        fake.inventories[receipt] = receiptInventory()
        fake.schedulerStates = ["47923657": .live("RUNNING")]

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty, "a receipt with a live job must not be transferred")
        #expect(fake.localFiles[receipt] == nil, "nothing may be created locally")
        #expect(fake.schedulerAsked == [["47923657"]], "only the unmarked job is asked about")
        #expect(report.skippedInProgress.map(\.name) == [receipt])
        #expect(report.imported.isEmpty)
        #expect(report.violations.isEmpty)
        #expect(report.failures.isEmpty)
        #expect(report.skippedByPolicy.isEmpty)
        guard case .skippedJobsLive(let held, let localFiles)? = report.directories.first?.outcome
        else {
            Issue.record("expected a live-job hold, got \(String(describing: report.directories.first))")
            return
        }
        #expect(held.map(\.job.id) == ["47923657"])
        #expect(held.first?.state == "RUNNING")
        #expect(held.first?.isUnknown == false)
        #expect(localFiles == 0)
        #expect(fake.catalogRebuilds == 1)
        #expect(report.purgeablePaths.isEmpty, "a held receipt names nothing as purgeable")

        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("\(receipt)  [submit receipt]  skipped — in progress: Slurm job 47923657 RUNNING"))
        #expect(text.contains("IN PROGRESS"))
        #expect(text.contains("job 47923657 (slurm-shard-0/): RUNNING"))
        #expect(text.contains("in progress 1"))
        #expect(!text.contains("imported 1"))
    }

    /// Once the scheduler reports the job ended, the same receipt imports in
    /// full on the next pass — with the capture at its final size.
    @Test func theSameReceiptImportsOnceItsJobHasEnded() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        fake.directories = [receipt]
        fake.inventories[receipt] = receiptInventory()
        fake.schedulerStates = ["47923657": .live("RUNNING")]

        let first = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(first.skippedInProgress.map(\.name) == [receipt])
        #expect(fake.transferred.isEmpty)

        fake.inventories[receipt] = receiptInventory(runningCaptureBytes: 14_336)
        fake.schedulerStates = ["47923657": .ended("COMPLETED")]
        let second = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(second.skippedInProgress.isEmpty)
        #expect(second.imported.map(\.name) == [receipt])
        #expect(fake.transferred == [receipt])
        #expect(second.violations.isEmpty)
        #expect(
            fake.localFiles[receipt]?.first { $0.relativePath == "slurm-shard-0/slurm-47923657.out" }?
                .size == 14_336)
    }

    /// A receipt whose every job carries the engine's end marker is settled
    /// by content: it imports, and the scheduler is never asked — even a
    /// scheduler that would have called the job live.
    @Test func aReceiptWithEndMarkersNeverAsksTheScheduler() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        fake.directories = [receipt]
        fake.inventories[receipt] =
            receiptInventory(runningCaptureBytes: 14_336)
            + remote(files: [("slurm-shard-0/slurm-47923657.exit", 2)])
        fake.schedulerStates = ["47923657": .live("RUNNING")]

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.schedulerAsked.isEmpty, "marked jobs are settled by content")
        #expect(report.imported.map(\.name) == [receipt])
        #expect(report.skippedInProgress.isEmpty)
        #expect(report.violations.isEmpty)
    }

    /// A scheduler that cannot be asked holds the receipt — a refusal to
    /// guess, with the failure as the reason — and touches nothing else: a
    /// stage directory in the same pass is judged by content as before.
    @Test func aSchedulerThatCannotBeAskedHoldsTheReceiptAndNothingElse() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [receipt, run]
        fake.inventories[receipt] = receiptInventory()
        fake.inventories[run] = remote(files: [("generations.jsonl", 4096), ("report.json", 64)])
        fake.schedulerFailure = ExperimentError(reason: "squeue exited 255: stale ControlMaster")

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred == [run], "the run is judged by content and imports")
        #expect(report.imported.map(\.name) == [run])
        #expect(report.skippedInProgress.map(\.name) == [receipt])
        #expect(report.violations.isEmpty, "a failed query is a hold, not a violation")
        #expect(report.failures.isEmpty)
        guard
            let outcome = report.directories.first(where: { $0.name == receipt })?.outcome,
            case .skippedJobsLive(let held, _) = outcome
        else {
            Issue.record("expected the receipt to be held")
            return
        }
        #expect(held.first?.isUnknown == true)
        #expect(held.first?.state.contains("squeue exited 255") == true)
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("Slurm job 47923657 (state unknown)"))
        #expect(text.contains("could not report"))
    }

    /// An id the scheduler answered nothing for — no row anywhere, although
    /// it did answer — is the seam's `.ended` positive absence; an id simply
    /// missing from the seam's answer is unknown and holds.
    @Test func anIdTheSchedulerDidNotAnswerForHolds() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        fake.directories = [receipt]
        fake.inventories[receipt] = receiptInventory()
        fake.schedulerStates = [:]

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(report.skippedInProgress.map(\.name) == [receipt])
        #expect(fake.transferred.isEmpty)

        fake.schedulerStates = ["47923657": .ended("not known to squeue or sacct")]
        let again = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(again.imported.map(\.name) == [receipt])
    }

    /// A frozen partial capture from a pre-gate import is held, not refused,
    /// while the job still runs — the reason says the partial is here — and
    /// only once the job has ended is the drift judged, with the repair
    /// named.
    @Test func aFrozenPartialReceiptIsHeldWhileTheJobRunsAndJudgedAfter() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        fake.directories = [receipt]
        fake.inventories[receipt] = receiptInventory(runningCaptureBytes: 14_336)
        fake.localFiles[receipt] = receiptInventory(runningCaptureBytes: 143)
        fake.schedulerStates = ["47923657": .live("RUNNING")]

        let held = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(held.violations.isEmpty, "drift is not judged while the job runs")
        #expect(held.failures.isEmpty)
        #expect(fake.transferred.isEmpty)
        guard case .skippedJobsLive(_, let localFiles)? = held.directories.first?.outcome else {
            Issue.record("expected a hold")
            return
        }
        #expect(localFiles == 10)
        let heldText = WorkspaceRunImport.summaryLines(held).joined(separator: "\n")
        #expect(heldText.contains("10 files from an earlier import already here"))

        fake.schedulerStates = ["47923657": .ended("COMPLETED")]
        let judged = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred.isEmpty, "a drifted directory must not be rsynced")
        guard case .refusedByteDrift(let message)? = judged.directories.first?.outcome else {
            Issue.record("expected a byte-drift refusal once the job ended")
            return
        }
        #expect(message.contains("slurm-shard-0/slurm-47923657.out: remote 14336 bytes, local 143 bytes"))
        #expect(message.contains("'\(receipt)-reimport'"))
        #expect(message.contains("--reimport-drifted"))
        #expect(judged.violations.count == 1)
    }

    // MARK: Drift repair: the reimport copy

    /// The five field receipts: local captures frozen at 143 bytes, the
    /// cluster's at 14 KB, every job long ended. Without the flag the
    /// refusal names the copy and the exact command; with it the cluster's
    /// copy comes home BESIDE the untouched local one, verified; and the
    /// next import reads the pair as resolved instead of as a violation.
    @Test func reimportDriftedBringsTheClusterCopyHomeBesideTheLocalOne() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        let copy = "\(receipt)-reimport"
        fake.directories = [receipt]
        fake.inventories[receipt] =
            receiptInventory(runningCaptureBytes: 14_336)
            + remote(files: [("slurm-shard-0/slurm-47923657.exit", 2)])
        fake.localFiles[receipt] = receiptInventory(runningCaptureBytes: 143)

        let refused = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(refused.failures.count == 1)
        #expect(refused.violations.first?.contains("'\(copy)'") == true)
        #expect(fake.transferred.isEmpty)

        let repaired = await WorkspaceRunImport.run(
            engine: fake.engine(),
            options: WorkspaceRunImport.Options(reimportDrifted: true))
        #expect(fake.transferred == ["\(receipt) -> \(copy)"])
        guard case .reimported(let named, let files, _)? = repaired.directories.first?.outcome
        else {
            Issue.record("expected a reimport, got \(String(describing: repaired.directories.first))")
            return
        }
        #expect(named == copy)
        #expect(files == 11)
        #expect(repaired.reimported.map(\.name) == [receipt])
        #expect(repaired.violations.isEmpty)
        #expect(repaired.failures.isEmpty)
        #expect(repaired.transferredAnything)
        #expect(
            fake.localFiles[receipt]?.first { $0.relativePath == "slurm-shard-0/slurm-47923657.out" }?
                .size == 143, "the local original is never rewritten")
        #expect(
            fake.localFiles[copy]?.first { $0.relativePath == "slurm-shard-0/slurm-47923657.out" }?
                .size == 14_336)
        #expect(fake.localFiles[copy]?.count == 11)
        let repairedText = WorkspaceRunImport.summaryLines(repaired).joined(separator: "\n")
        #expect(repairedText.contains("DRIFTED — the cluster's copy imported beside it as '\(copy)'"))
        #expect(repairedText.contains("reimported 1"))

        let settled = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred == ["\(receipt) -> \(copy)"], "nothing more transfers")
        guard case .driftResolved(let resolvedCopy, _)? = settled.directories.first?.outcome else {
            Issue.record("expected the drift to read as resolved")
            return
        }
        #expect(resolvedCopy == copy)
        #expect(settled.driftResolved.map(\.name) == [receipt])
        #expect(settled.violations.isEmpty)
        #expect(settled.failures.isEmpty)
        #expect(!settled.transferredAnything)
        let settledText = WorkspaceRunImport.summaryLines(settled).joined(separator: "\n")
        #expect(settledText.contains("drifted, resolved — '\(copy)' holds the cluster's copy"))
        #expect(settledText.contains("drift resolved 1"))
        #expect(!settledText.contains("VIOLATIONS"))
    }

    /// A dry run names the copy it would make and makes nothing.
    @Test func aDryRunReimportMakesNothing() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        fake.directories = [receipt]
        fake.inventories[receipt] =
            receiptInventory(runningCaptureBytes: 14_336)
            + remote(files: [("slurm-shard-0/slurm-47923657.exit", 2)])
        fake.localFiles[receipt] = receiptInventory(runningCaptureBytes: 143)

        let report = await WorkspaceRunImport.run(
            engine: fake.engine(),
            options: WorkspaceRunImport.Options(dryRun: true, reimportDrifted: true))
        #expect(fake.transferred.isEmpty)
        #expect(fake.localFiles["\(receipt)-reimport"] == nil)
        #expect(fake.catalogRebuilds == 0)
        guard case .reimported(let copy, _, _)? = report.directories.first?.outcome else {
            Issue.record("expected a would-reimport")
            return
        }
        #expect(copy == "\(receipt)-reimport")
        let text = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        #expect(text.contains("would be imported beside it"))
        #expect(text.contains("would reimport 1"))
    }

    /// A copy that itself differs from the cluster (the job wrote more after
    /// it) is named as drifted too, and the next ordinal is what a repair
    /// uses.
    @Test func aDriftedReimportCopyIsNamedAndTheNextOrdinalIsUsed() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        let first = "\(receipt)-reimport"
        let second = "\(receipt)-reimport2"
        fake.directories = [receipt]
        fake.inventories[receipt] =
            receiptInventory(runningCaptureBytes: 14_336)
            + remote(files: [("slurm-shard-0/slurm-47923657.exit", 2)])
        fake.localFiles[receipt] = receiptInventory(runningCaptureBytes: 143)
        fake.localFiles[first] = receiptInventory(runningCaptureBytes: 7_000)

        let refused = await WorkspaceRunImport.run(engine: fake.engine())
        guard case .refusedByteDrift(let message)? = refused.directories.first?.outcome else {
            Issue.record("expected a refusal")
            return
        }
        #expect(message.contains("\(first) is here from an earlier --reimport-drifted"))
        #expect(message.contains("'\(second)'"))

        let repaired = await WorkspaceRunImport.run(
            engine: fake.engine(),
            options: WorkspaceRunImport.Options(reimportDrifted: true))
        #expect(fake.transferred == ["\(receipt) -> \(second)"])
        guard case .reimported(let copy, _, _)? = repaired.directories.first?.outcome else {
            Issue.record("expected a reimport under the next ordinal")
            return
        }
        #expect(copy == second)
        #expect(fake.localFiles[first]?.count == 10, "the drifted copy is untouched")
    }

    /// A copy the cluster has since added files to (a continuation that ran
    /// after the reimport) is filled like any partial import, flag or no
    /// flag — its bytes agree; only files are missing.
    @Test func aReimportCopyWithGapsIsFilled() async {
        let fake = FakeImportRemote()
        let receipt = "\(stamp)-submit-bundle-alpha-run"
        let copy = "\(receipt)-reimport"
        let full =
            receiptInventory(runningCaptureBytes: 14_336)
            + remote(files: [("slurm-shard-0/slurm-47923657.exit", 2)])
        fake.directories = [receipt]
        fake.inventories[receipt] = full + remote(files: [("records/fedcba987654.json", 300)])
        fake.localFiles[receipt] = receiptInventory(runningCaptureBytes: 143)
        fake.localFiles[copy] = full

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.transferred == ["\(receipt) -> \(copy)"])
        guard case .reimported(let named, let files, let bytes)? = report.directories.first?.outcome
        else {
            Issue.record("expected the copy's gap to be filled")
            return
        }
        #expect(named == copy)
        #expect(files == 1)
        #expect(bytes == 300)
        #expect(fake.localFiles[copy]?.count == 12)
        #expect(report.violations.isEmpty)
    }

    /// A stage directory names no jobs, so the scheduler is never asked for
    /// a pass that holds only stages — the gate costs nothing where it does
    /// not apply.
    @Test func aPassWithoutReceiptsNeverAsksTheScheduler() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [("generations.jsonl", 4096), ("report.json", 64)])
        fake.schedulerFailure = ExperimentError(reason: "must not be asked")

        let report = await WorkspaceRunImport.run(engine: fake.engine())
        #expect(fake.schedulerAsked.isEmpty)
        #expect(report.imported.map(\.name) == [run])
    }

    /// Tightening 4: remote bytes that DIFFER from imported local bytes refuse,
    /// loudly, and nothing transfers.
    @Test func byteDriftRefusesAndTransfersNothing() async {
        let fake = FakeImportRemote()
        let run = "\(stamp)-exp-alpha-run"
        fake.directories = [run]
        fake.inventories[run] = remote(files: [("generations.jsonl", 4096), ("report.json", 64)])
        fake.localFiles[run] = remote(files: [("generations.jsonl", 2048), ("report.json", 64)])

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
        fake.inventories[run] = remote(files: [("config.json", 120), ("report.json", 64)])

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
        fake.inventories[old] = remote(files: [("config.json", 1), ("report.json", 1)])
        fake.inventories[new] = remote(files: [("config.json", 1), ("report.json", 1)])

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
        fake.inventories[run] = remote(files: [("experiment.json", 512), ("report.json", 64)])
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
        fake.inventories[run] = remote(files: [("experiment.json", 512), ("report.json", 64)])
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

    /// The receipt gate asks the scheduler directly, through the same SSH
    /// master: one `squeue` listing of the user's live jobs (never `-j`,
    /// which fails outright on a finished id) and one `sacct` for the ids —
    /// and never `steerlab-server jobs list`, which sweeps a controller's
    /// jobs. The site's declared command names are honoured.
    @Test func schedulerArgvsAskSqueueForTheUserAndSacctForTheIds() throws {
        var profile = ClusterSiteProfile.exampleCluster
        profile.transport = .ssh(
            host: "user@login.example.edu", proxyJump: nil, remotePort: 8080,
            vpnExpected: true)
        let squeue = try #require(WorkspaceRunImport.squeueArgv(site: profile))
        #expect(squeue.first == ClusterProvisioner.sshExecutablePath)
        #expect(squeue.contains("user@login.example.edu"))
        let listing = try #require(squeue.last)
        #expect(listing.hasPrefix("squeue -h -u $USER -o "))
        #expect(listing.contains("%i|%T"))
        #expect(!listing.contains(" -j "))
        #expect(!listing.contains("steerlab-server"))

        let sacct = try #require(WorkspaceRunImport.sacctArgv(site: profile, ids: ["1", "2"]))
        #expect(sacct.last == "sacct -n -X -P -j 1,2 -o JobID,State")
        #expect(WorkspaceRunImport.sacctArgv(site: profile, ids: []) == nil)

        guard case .slurm(var slurm) = profile.scheduler else {
            Issue.record("the example cluster declares Slurm")
            return
        }
        slurm.commands.query = "site-squeue"
        slurm.commands.accounting = "site-sacct"
        profile.scheduler = .slurm(slurm)
        #expect(WorkspaceRunImport.squeueArgv(site: profile)?.last?.hasPrefix("site-squeue ") == true)
        #expect(WorkspaceRunImport.sacctArgv(site: profile, ids: ["1"])?.last?.hasPrefix("site-sacct ") == true)

        var noScheduler = profile
        noScheduler.scheduler = .none
        #expect(WorkspaceRunImport.squeueArgv(site: noScheduler) == nil)
    }

    /// The live seam: a failed `squeue` THROWS (its listing is what proves a
    /// job live, so a listing that did not happen must not read as "nothing
    /// running"); a failed `sacct` leaves the ids it would have decided
    /// unknown; and both answering yields the policy's reading.
    @Test func theLiveSchedulerSeamRefusesToReadSilenceAsAbsence() async throws {
        var profile = ClusterSiteProfile.exampleCluster
        profile.transport = .ssh(
            host: "user@login.example.edu", proxyJump: nil, remotePort: 8080,
            vpnExpected: true)

        let healthy = ScriptedSchedulerShell(
            squeue: ClusterShellResult(exitCode: 0, lines: ["1|RUNNING"]),
            sacct: ClusterShellResult(exitCode: 0, lines: ["2|COMPLETED"]))
        let states = try await WorkspaceRunImport.schedulerStates(
            site: profile, ids: ["1", "2", "3"], shell: healthy)
        #expect(states["1"] == .live("RUNNING"))
        #expect(states["2"] == .ended("COMPLETED"))
        #expect(states["3"] == .ended("not known to squeue or sacct"))
        #expect(healthy.commands.count == 2)

        let deafAccounting = ScriptedSchedulerShell(
            squeue: ClusterShellResult(exitCode: 0, lines: []),
            sacct: ClusterShellResult(exitCode: 1, lines: ["sacct: error"]))
        let partial = try await WorkspaceRunImport.schedulerStates(
            site: profile, ids: ["2"], shell: deafAccounting)
        #expect(partial["2"] == .unknown("sacct did not answer"))

        let deafQueue = ScriptedSchedulerShell(
            squeue: ClusterShellResult(exitCode: 255, lines: ["ssh: connection closed"]),
            sacct: ClusterShellResult(exitCode: 0, lines: ["2|COMPLETED"]))
        await #expect(throws: ExperimentError.self) {
            try await WorkspaceRunImport.schedulerStates(
                site: profile, ids: ["2"], shell: deafQueue)
        }
        #expect(deafQueue.commands.count == 1, "sacct is not asked once squeue has failed")

        // Nothing to ask about: no command runs at all.
        let idle = ScriptedSchedulerShell(
            squeue: ClusterShellResult(exitCode: 0), sacct: ClusterShellResult(exitCode: 0))
        let none = try await WorkspaceRunImport.schedulerStates(
            site: profile, ids: ["not-an-id"], shell: idle)
        #expect(none.isEmpty)
        #expect(idle.commands.isEmpty)
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

/// A far side that answers `squeue` and `sacct` with canned results and
/// records every command it was handed.
private final class ScriptedSchedulerShell: ClusterShellRunner, @unchecked Sendable {
    // @unchecked Sendable: written only by the serialized test body and the
    // seam under test; never escapes the test.
    let squeue: ClusterShellResult
    let sacct: ClusterShellResult
    private(set) var commands: [String] = []

    init(squeue: ClusterShellResult, sacct: ClusterShellResult) {
        self.squeue = squeue
        self.sacct = sacct
    }

    func run(_ argv: [String]) async -> ClusterShellResult {
        let command = argv.last ?? ""
        commands.append(command)
        if command.hasPrefix("squeue") { return squeue }
        if command.hasPrefix("sacct") { return sacct }
        return ClusterShellResult(exitCode: 127, lines: ["unexpected: \(command)"])
    }
}

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

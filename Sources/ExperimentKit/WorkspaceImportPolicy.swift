import Foundation

// =============================================================================
// The workspace import policy, as CODE (open-issues §20, decided 2026-08-20).
//
// The Mac workspace is the durable source of truth; cluster scratch purges on
// the site's retention window. Until this file the policy was runner lore —
// executed by hand, per wave, by whoever happened to be driving. It is now one
// set of typed constants that BOTH the app's import hook and the CLI verb read,
// so "what comes home" cannot differ between the two surfaces or drift from the
// researcher-facing statement (a workspace's RUNS-GUIDE "Import policy").
//
// Three rules shape everything here:
//
//   * Classification is by SHAPE — directory-name grammar plus the artifacts a
//     directory actually contains — never by study vocabulary. A concept name,
//     a case family, or a wave label may not appear in this file; grouping keys
//     for the catalog come from each run's own `config.json`.
//
//   * An unrecognized directory shape is a REPORTED UNKNOWN and is imported
//     CONSERVATIVELY. Over-importing costs disk on a machine that has it;
//     silently skipping costs evidence on a filesystem that purges. When the
//     policy cannot say, it says so and brings the bytes home.
//
//   * Nothing here deletes anything. The NEVER-import rules produce a
//     purge-ELIGIBILITY report — a statement about what scratch may now drop —
//     and deletion remains a separate, deliberate act this layer never performs.
// =============================================================================

public enum WorkspaceImportPolicy {

    // MARK: - Directory-name grammar

    /// `<stamp>-<rest>`: every run directory both engines mint is stamped
    /// `yyyyMMddTHHmmssSSS`. A directory without a stamp is not a run
    /// directory (the mutable library subtrees, stray files, a researcher's
    /// own folder) and is classified `.notARunDirectory`.
    ///
    /// Hand-parsed rather than a `Regex` literal: a `Regex` is not `Sendable`,
    /// so a stored static one cannot exist in a strict-concurrency module, and
    /// rebuilding it per call to classify thousands of directory names is
    /// wasteful for a grammar this small.
    ///
    /// Returns `(stamp, rest)`, or nil when the name carries no run stamp.
    public static func splitRunStamp(_ name: String) -> (stamp: String, rest: String)? {
        guard let separator = name.firstIndex(of: "-") else { return nil }
        let stamp = String(name[name.startIndex..<separator])
        let rest = String(name[name.index(after: separator)...])
        guard !rest.isEmpty, isRunStamp(stamp) else { return nil }
        return (stamp, rest)
    }

    /// `yyyyMMdd` + `T` + at least one digit.
    public static func isRunStamp(_ text: String) -> Bool {
        guard let t = text.firstIndex(of: "T") else { return false }
        let day = text[text.startIndex..<t]
        let time = text[text.index(after: t)...]
        return day.count == 8 && day.allSatisfy(\.isNumber)
            && !time.isEmpty && time.allSatisfy(\.isNumber)
    }

    /// `-shard<I>of<N>` — the server's shard-partial suffix. The merged run's
    /// completeness stamp names partials by BASENAME, so the name is the join
    /// key for the merge-evidence gate; `shard.json` inside the directory is
    /// corroboration, not the identity.
    ///
    /// Returns `(body, index, count)` when the name ends in the suffix.
    public static func splitShardSuffix(
        _ text: String
    ) -> (body: String, index: Int, count: Int)? {
        guard let marker = text.range(of: "-shard", options: .backwards) else {
            return nil
        }
        let tail = text[marker.upperBound...]
        guard let of = tail.range(of: "of") else { return nil }
        let indexText = tail[tail.startIndex..<of.lowerBound]
        let countText = tail[of.upperBound...]
        guard
            !indexText.isEmpty, indexText.allSatisfy(\.isNumber),
            !countText.isEmpty, countText.allSatisfy(\.isNumber),
            let index = Int(indexText), let count = Int(countText)
        else { return nil }
        return (String(text[text.startIndex..<marker.lowerBound]), index, count)
    }

    /// The mutable LIBRARY subtrees inside the otherwise immutable `runs/`
    /// area. Engine vocabulary (CLAUDE.md), not study vocabulary: these are
    /// artifact libraries every workspace has, whatever it studies.
    public static let librarySubtrees: Set<String> = [
        "model-variants", "neutral-pcs", "jlens-lenses",
    ]

    /// Stage verbs a run directory's name may end in, longest-match first so
    /// `evaluate-judgment` is not shadowed by `evaluate`.
    public static let stageVerbs: [String] = [
        "pipeline", "evaluate-judgment", "evaluate", "analyze", "validate",
        "extract", "sweep", "confirm", "promote", "run",
    ]

    /// The file whose presence proves a directory is a shard PARTIAL.
    public static let shardStampFileName = "shard.json"

    /// The report a merged run writes; carries the merge completeness stamp.
    public static let reportFileName = "report.json"

    /// The final trained-adapter weight file. A submit directory containing
    /// one under `run/<name>/` is a finetune receipt whose weights the
    /// model-variant artifacts reference by workspace-relative path and the
    /// studies pin by hash — ALWAYS imported, by shape rather than by the
    /// study's name for the job.
    public static let adapterWeightFileName = "adapter_model.safetensors"

    // MARK: - What a directory IS

    /// The shape of a run directory. Every case is derived from the name
    /// grammar (plus, for `.shardPartial`, the shard stamp); none of them
    /// encodes what is being studied.
    public enum DirectoryKind: String, Sendable, Equatable, CaseIterable {
        /// A measured run, or one of the analysis stages over it.
        case run
        case analyze
        case evaluate
        case validate
        case extract
        case sweep
        case confirm
        case promote
        /// A pipeline ledger directory (the chain, not a stage).
        case pipeline
        /// A submission receipt: rendered sbatch, plan, manifest, and — for
        /// finetune jobs — the FINAL adapter weights.
        case submit
        /// Vector-artifact campaign directories (`optvec-*`, `sae-feature-*`,
        /// `derived-*`): unique or hash-pinned bytes, archival on the Mac.
        case vectorArtifact
        /// A lens-support directory.
        case lensSupport
        /// An interactive GPU-session receipt.
        case session
        /// One shard of a fan-out. NEVER imported — but only once a merged
        /// run's completeness stamp proves the family merged.
        case shardPartial
        /// A stamped run directory whose shape this policy does not
        /// recognize. Imported conservatively and REPORTED.
        case unknown
        /// Not a run directory at all (no timestamp stamp): a library
        /// subtree, or something the researcher put there.
        case notARunDirectory

        /// Whether the policy's ALWAYS-import list covers this shape.
        public var isAlwaysImported: Bool {
            switch self {
            case .run, .analyze, .evaluate, .validate, .extract, .sweep,
                .confirm, .promote, .pipeline, .submit, .vectorArtifact,
                .lensSupport, .session:
                true
            case .shardPartial, .unknown, .notARunDirectory:
                false
            }
        }

        /// Plain-language name for reports.
        public var label: String {
            switch self {
            case .run: "run"
            case .analyze: "analyze"
            case .evaluate: "evaluate"
            case .validate: "validate"
            case .extract: "extract"
            case .sweep: "sweep"
            case .confirm: "confirm"
            case .promote: "promote"
            case .pipeline: "pipeline"
            case .submit: "submit receipt"
            case .vectorArtifact: "vector artifact"
            case .lensSupport: "lens support"
            case .session: "session"
            case .shardPartial: "shard partial"
            case .unknown: "unknown shape"
            case .notARunDirectory: "not a run directory"
            }
        }
    }

    /// One directory's parsed identity.
    public struct Classification: Sendable, Equatable {
        public var name: String
        public var kind: DirectoryKind
        /// The name's timestamp stamp, when it has one.
        public var stamp: String?
        /// The name minus its stamp, its stage verb, and its shard suffix —
        /// the family key shard partials share with their merged run, and the
        /// catalog's fallback grouping key when a run has no `config.json`.
        public var stem: String
        public var shardIndex: Int?
        public var shardCount: Int?

        public init(
            name: String, kind: DirectoryKind, stamp: String? = nil,
            stem: String, shardIndex: Int? = nil, shardCount: Int? = nil
        ) {
            self.name = name
            self.kind = kind
            self.stamp = stamp
            self.stem = stem
            self.shardIndex = shardIndex
            self.shardCount = shardCount
        }

        public var isShardPartial: Bool { kind == .shardPartial }
    }

    /// Classify a run directory by NAME. `containsShardStamp` is the
    /// corroborating artifact read (a `shard.json` at the directory's root);
    /// pass false when the caller has not looked — the name suffix alone
    /// still identifies a partial, which is what the merged run's stamp
    /// joins on.
    public static func classify(
        directoryName name: String, containsShardStamp: Bool = false
    ) -> Classification {
        guard let split = splitRunStamp(name) else {
            return Classification(
                name: name,
                kind: .notARunDirectory,
                stem: name)
        }
        let stamp = split.stamp
        var rest = split.rest

        var shardIndex: Int?
        var shardCount: Int?
        if let shard = splitShardSuffix(rest) {
            shardIndex = shard.index
            shardCount = shard.count
            rest = shard.body
        }
        let isPartial = shardIndex != nil || containsShardStamp

        // Vector-artifact campaigns and lens support declare themselves by
        // prefix; they carry no stage verb.
        for prefix in ["optvec-", "sae-feature-", "derived-"] where rest.hasPrefix(prefix) {
            return Classification(
                name: name, kind: .vectorArtifact, stamp: stamp, stem: rest)
        }
        if rest.hasPrefix("jlens-support-") {
            return Classification(
                name: name, kind: .lensSupport, stamp: stamp, stem: rest)
        }
        if rest == "session" || rest.hasPrefix("session-") {
            return Classification(
                name: name, kind: .session, stamp: stamp, stem: rest)
        }

        // A re-execution prefix (`resume-`, `resume2-`) is bookkeeping about
        // the ATTEMPT, not about the shape: strip it so a resumed run groups
        // with the run it resumed. Superseded attempts are receipts and are
        // imported like any other run of their kind.
        var body = rest
        if body.hasPrefix("resume") {
            let after = body.dropFirst("resume".count)
            let digits = after.prefix { $0.isNumber }
            let remainder = after.dropFirst(digits.count)
            if remainder.hasPrefix("-") { body = String(remainder.dropFirst()) }
        }

        // `submit-…` is a receipt; `exp-…` is an executed stage.
        var origin: String?
        for prefix in ["submit-", "exp-"] where body.hasPrefix(prefix) {
            origin = String(prefix.dropLast())
            body = String(body.dropFirst(prefix.count))
            break
        }

        var verb: String?
        for candidate in stageVerbs where body.hasSuffix("-" + candidate) {
            verb = candidate
            body = String(body.dropLast(candidate.count + 1))
            break
        }

        let stem = body.isEmpty ? rest : body

        if isPartial {
            return Classification(
                name: name, kind: .shardPartial, stamp: stamp, stem: stem,
                shardIndex: shardIndex, shardCount: shardCount)
        }
        if origin == "submit" {
            return Classification(
                name: name, kind: .submit, stamp: stamp, stem: stem)
        }
        guard origin == "exp" else {
            // Stamped, but neither a declared origin nor a known prefix. The
            // conservative branch: reported, and imported anyway.
            return Classification(
                name: name, kind: .unknown, stamp: stamp, stem: stem)
        }
        let kind: DirectoryKind =
            switch verb {
            case "analyze": .analyze
            case "evaluate", "evaluate-judgment": .evaluate
            case "validate": .validate
            case "extract": .extract
            case "sweep": .sweep
            case "confirm": .confirm
            case "promote": .promote
            case "pipeline": .pipeline
            // A merged run is an ORDINARY run directory — the merge writes
            // `exp-<name>-run` exactly as a single-job run does. It is
            // distinguished from a single-job run only by its report's
            // completeness stamp, which is the merge-evidence gate's input,
            // not a naming question.
            case "run": .run
            default: .run
            }
        return Classification(
            name: name, kind: kind, stamp: stamp, stem: stem)
    }

    // MARK: - The per-directory decision

    public enum Decision: Sendable, Equatable {
        /// The policy's ALWAYS list covers this shape.
        case importAlways(kind: DirectoryKind)
        /// Unrecognized shape: imported anyway, and reported as an unknown.
        /// Over-importing is the deliberate failure direction.
        case importConservatively(reason: String)
        /// A shard partial. Never imported — the merged run carries the
        /// records — but its family's purge eligibility is decided by
        /// EVIDENCE, downstream, not by this decision.
        case skipShardPartial(shardIndex: Int?, shardCount: Int?)
        /// Not a run directory (a library subtree, or a stray file).
        case notApplicable(reason: String)

        public var transfers: Bool {
            switch self {
            case .importAlways, .importConservatively: true
            case .skipShardPartial, .notApplicable: false
            }
        }
    }

    public static func decision(for classification: Classification) -> Decision {
        switch classification.kind {
        case .shardPartial:
            .skipShardPartial(
                shardIndex: classification.shardIndex,
                shardCount: classification.shardCount)
        case .notARunDirectory:
            librarySubtrees.contains(classification.name)
                ? .notApplicable(
                    reason: "'\(classification.name)' is a mutable library "
                        + "subtree, not a run directory — it has its own "
                        + "lifecycle and is never swept by an import")
                : .notApplicable(
                    reason: "'\(classification.name)' carries no run stamp, "
                        + "so it is not a run directory")
        case .unknown:
            .importConservatively(
                reason: "'\(classification.name)' is a stamped directory of "
                    + "an unrecognized shape — imported rather than skipped, "
                    + "because a silent skip on a purging filesystem loses "
                    + "evidence, and reported so the policy can learn it")
        default:
            .importAlways(kind: classification.kind)
        }
    }

    // MARK: - What never travels, INSIDE a directory that does

    /// Path rules that exclude bytes from a directory the policy otherwise
    /// imports whole. Both are re-derivable or resume-only state whose only
    /// copy that matters is already coming home in another form.
    public enum ExclusionRule: String, Sendable, Equatable, CaseIterable {
        /// `*.tar.gz` evidence bundles inside submit directories:
        /// re-compressed duplicates of run directories this same pass
        /// imports in full.
        case evidenceBundleTarball
        /// `checkpoints/` trees under finetune runs: optimizer, scheduler,
        /// RNG, and per-step adapter snapshots — resume-training state only.
        /// The FINAL adapter weights are kept.
        case trainingCheckpointTree

        public var reason: String {
            switch self {
            case .evidenceBundleTarball:
                "evidence-bundle tarball — a re-compressed duplicate of a run "
                    + "directory imported in full"
            case .trainingCheckpointTree:
                "training checkpoint tree — optimizer/scheduler/RNG and "
                    + "per-step snapshots; resume-training state only, and the "
                    + "final adapter weights are imported"
            }
        }

        /// rsync filter arguments implementing the rule.
        public var rsyncFilters: [String] {
            switch self {
            case .evidenceBundleTarball: ["--exclude", "*.tar.gz"]
            case .trainingCheckpointTree: ["--exclude", "checkpoints/"]
            }
        }
    }

    /// Which exclusions apply to a directory of this kind. The tarball rule
    /// is scoped to submit receipts (the only place evidence bundles are
    /// written); the checkpoint rule applies everywhere, because no other run
    /// type writes a `checkpoints/` tree and a scoped-by-name rule would have
    /// to know what the finetune job was called.
    public static func exclusions(for kind: DirectoryKind) -> [ExclusionRule] {
        switch kind {
        case .submit: [.evidenceBundleTarball, .trainingCheckpointTree]
        default: [.trainingCheckpointTree]
        }
    }

    /// Whether a directory-relative path is excluded by the given rules. The
    /// verification pass uses this so an excluded remote file is never
    /// counted as a local gap.
    public static func isExcluded(
        relativePath path: String, rules: [ExclusionRule]
    ) -> Bool {
        let components = path.split(separator: "/").map(String.init)
        for rule in rules {
            switch rule {
            case .evidenceBundleTarball:
                if path.hasSuffix(".tar.gz") { return true }
            case .trainingCheckpointTree:
                if components.dropLast().contains("checkpoints") { return true }
            }
        }
        return false
    }

    // MARK: - The merge-evidence gate (tightening 1 — non-negotiable)

    /// The merge completeness stamp, as `merge_shard_runs` writes it: the
    /// `sharded` block of the MERGED run's `report.json`.
    ///
    /// ```json
    /// "sharded": { "shardCount": 4,
    ///              "shardRuns": ["<partial basename>", …],
    ///              "shardJobIDs": ["…"] }        // optional
    /// ```
    ///
    /// Only the merge writes it, and it is written only after every
    /// completeness proof has passed (every expected cell present exactly
    /// once, no cross-shard duplicates, every shard complete, one experiment
    /// hash, live manifest still at that hash). That is precisely why the
    /// gate reads THIS and not the existence of a directory whose name looks
    /// merged: a merged-looking directory with no stamp is a directory, not a
    /// proof.
    public struct MergeEvidence: Sendable, Equatable {
        /// The merged run directory's basename.
        public var mergedRun: String
        public var shardCount: Int
        /// The partial basenames the stamp names.
        public var shardRuns: [String]

        public init(mergedRun: String, shardCount: Int, shardRuns: [String]) {
            self.mergedRun = mergedRun
            self.shardCount = shardCount
            self.shardRuns = shardRuns
        }
    }

    /// A merged-looking run that carries NO stamp — the case the gate exists
    /// to refuse. Tracked separately so the refusal can name the candidate
    /// instead of reporting a bare orphan.
    public struct UnstampedMergeCandidate: Sendable, Equatable {
        public var runName: String
        public var stem: String

        public init(runName: String, stem: String) {
            self.runName = runName
            self.stem = stem
        }
    }

    /// One fan-out's partials, keyed by the family they belong to.
    public struct ShardFamily: Sendable, Equatable {
        public var stem: String
        public var declaredCount: Int?
        /// Partial basenames, sorted.
        public var partials: [String]

        public init(stem: String, declaredCount: Int?, partials: [String]) {
            self.stem = stem
            self.declaredCount = declaredCount
            self.partials = partials
        }
    }

    /// What the gate decided about one family.
    public enum ShardFamilyVerdict: Sendable, Equatable {
        /// A merged run is present in the workspace AND its report carries
        /// the completeness stamp naming every one of these partials.
        case purgeEligible(mergedRun: String)
        /// A merged-looking run exists but carries no completeness stamp.
        /// NOT eligible: the directory is not the proof.
        case mergedRunNotStamped(candidates: [String])
        /// No merged run anywhere. Surfaced LOUDLY and never eligible.
        case orphaned
        /// The stamp exists but does not name every partial (or disagrees on
        /// the count) — a different fan-out, or a partial from another
        /// attempt. Never eligible.
        case stampDoesNotCoverPartials(mergedRun: String, missing: [String])

        public var isPurgeEligible: Bool {
            if case .purgeEligible = self { return true }
            return false
        }

        /// Whether the verdict must be shouted rather than listed.
        public var isLoud: Bool { !isPurgeEligible }
    }

    /// Group shard partials into families. The family key is the stem the
    /// partial shares with the run it merges into; the declared count comes
    /// from the `-shard<I>of<N>` suffix, and disagreement inside a family
    /// (two different N) leaves the count nil, which no verdict can then call
    /// eligible.
    public static func shardFamilies(
        _ classifications: [Classification]
    ) -> [ShardFamily] {
        var grouped: [String: [Classification]] = [:]
        for classification in classifications where classification.isShardPartial {
            grouped[classification.stem, default: []].append(classification)
        }
        return grouped.keys.sorted().map { stem in
            let members = grouped[stem] ?? []
            let counts = Set(members.compactMap(\.shardCount))
            return ShardFamily(
                stem: stem,
                declaredCount: counts.count == 1 ? counts.first : nil,
                partials: members.map(\.name).sorted())
        }
    }

    /// The gate. `evidence` is read from the WORKSPACE's merged runs (after
    /// this pass's transfers, so a merged run that came home in the same
    /// import counts); `unstampedCandidates` are merged-looking local runs
    /// whose report carries no `sharded` block.
    ///
    /// A family is purge-eligible ONLY when some evidence names every one of
    /// its partials and agrees on the count. Everything else is loud.
    public static func verdict(
        family: ShardFamily,
        evidence: [MergeEvidence],
        unstampedCandidates: [UnstampedMergeCandidate]
    ) -> ShardFamilyVerdict {
        let partials = Set(family.partials)
        var covering: MergeEvidence?
        var partial: (evidence: MergeEvidence, missing: [String])?
        for candidate in evidence {
            let named = Set(candidate.shardRuns)
            let missing = partials.subtracting(named).sorted()
            let countAgrees =
                family.declaredCount == nil
                || family.declaredCount == candidate.shardCount
            if missing.isEmpty, countAgrees {
                covering = candidate
                break
            }
            if !named.isDisjoint(with: partials), partial == nil {
                partial = (
                    candidate,
                    missing.isEmpty
                        ? ["shard count \(candidate.shardCount) ≠ declared "
                            + "\(family.declaredCount.map(String.init) ?? "?")"]
                        : missing)
            }
        }
        if let covering { return .purgeEligible(mergedRun: covering.mergedRun) }
        if let partial {
            return .stampDoesNotCoverPartials(
                mergedRun: partial.evidence.mergedRun, missing: partial.missing)
        }
        let candidates = unstampedCandidates
            .filter { $0.stem == family.stem }
            .map(\.runName)
            .sorted()
        if !candidates.isEmpty {
            return .mergedRunNotStamped(candidates: candidates)
        }
        return .orphaned
    }

    /// The report line for one family's verdict — the loud text, spelled once.
    public static func message(
        family: ShardFamily, verdict: ShardFamilyVerdict
    ) -> String {
        let n = family.partials.count
        let shards = "\(n) shard partial\(n == 1 ? "" : "s")"
        switch verdict {
        case .purgeEligible(let mergedRun):
            return "\(family.stem): \(shards) may be purged on scratch — "
                + "merged run '\(mergedRun)' is in this workspace and its "
                + "report.json carries the merge completeness stamp naming "
                + "every partial"
        case .mergedRunNotStamped(let candidates):
            return "\(family.stem): NOT purge-eligible — "
                + "\(candidates.joined(separator: ", ")) look\(candidates.count == 1 ? "s" : "") "
                + "like the merged run but carr\(candidates.count == 1 ? "ies" : "y") no "
                + "merge completeness stamp (report.json has no `sharded` "
                + "block). A directory is not a proof; keep the \(shards) "
                + "until a merge is evidenced"
        case .stampDoesNotCoverPartials(let mergedRun, let missing):
            return "\(family.stem): NOT purge-eligible — merged run "
                + "'\(mergedRun)' is stamped, but its stamp does not cover "
                + "\(missing.joined(separator: ", ")). These partials belong "
                + "to a different fan-out or attempt; keep them"
        case .orphaned:
            return "\(family.stem): ORPHANED — \(shards) with no evidenced "
                + "merged run anywhere in this workspace. Never purge these; "
                + "merge the fan-out (or resume the incomplete shard) first"
        }
    }

    // MARK: - Content verification (tightening 2)

    /// One file, on either side: path relative to its run directory, and its
    /// size in bytes. Counts alone let a truncated file pass, so the stat is
    /// the unit of comparison on BOTH sides.
    public struct FileStat: Sendable, Equatable, Hashable {
        public var relativePath: String
        public var size: Int64

        public init(relativePath: String, size: Int64) {
            self.relativePath = relativePath
            self.size = size
        }
    }

    /// A hash that is already PINNED somewhere (an artifact sidecar, a
    /// battery or stimulus hash). Verification checks these wherever the pin
    /// exists; it does not invent hashing where nothing pinned one.
    public struct PinnedHash: Sendable, Equatable {
        public var relativePath: String
        public var sha256: String
        /// Where the pin came from, for the report ("vector sidecar", …).
        public var source: String

        public init(relativePath: String, sha256: String, source: String) {
            self.relativePath = relativePath
            self.sha256 = sha256
            self.source = source
        }
    }

    /// Everything verification can find.
    public enum Finding: Sendable, Equatable {
        /// Present remotely, absent locally: a GAP an idempotent re-import
        /// fills. Not a violation.
        case gap(relativePath: String, size: Int64)
        /// Same path, different size. An immutability VIOLATION: the local
        /// bytes are durable and are never overwritten.
        case sizeDrift(relativePath: String, remote: Int64, local: Int64)
        /// A pinned hash disagrees with the local bytes.
        case hashDrift(relativePath: String, pinned: String, actual: String, source: String)
        /// File counts disagree after gaps are accounted for — surfaced in
        /// its own right so a report never says "verified" over a mismatch.
        case countMismatch(remote: Int, local: Int)
        /// Present locally, absent remotely. Not a violation (scratch may
        /// have purged it, or a stage wrote locally) — reported.
        case localOnly(relativePath: String)

        /// Whether this finding refuses the directory (tightening 4).
        public var isViolation: Bool {
            switch self {
            case .sizeDrift, .hashDrift: true
            case .gap, .countMismatch, .localOnly: false
            }
        }
    }

    /// Compare a remote inventory against a local one. Excluded paths are
    /// dropped from BOTH sides first, so the policy's own NEVER-import rules
    /// can never be read as missing files — nor, on the local side, as extra
    /// ones. The symmetry is load-bearing: rsync applies the same rules, so a
    /// directory that already holds an excluded file (a tarball imported
    /// before the rule existed, a `checkpoints/` tree copied by hand) would
    /// otherwise fail its own count comparison for ever (2026-08-24 field
    /// report). One definition — `isExcluded` — decides both sides.
    ///
    /// `localHash` is consulted only for paths that already carry a pin, and
    /// only when the file exists locally and the rules keep it.
    public static func verify(
        remote: [FileStat],
        local: [FileStat],
        exclusions rules: [ExclusionRule],
        pinnedHashes: [PinnedHash] = [],
        localHash: (String) -> String? = { _ in nil }
    ) -> [Finding] {
        let remoteKept = remote.filter {
            !isExcluded(relativePath: $0.relativePath, rules: rules)
        }
        let localKept = local.filter {
            !isExcluded(relativePath: $0.relativePath, rules: rules)
        }
        let localBySize = Dictionary(
            localKept.map { ($0.relativePath, $0.size) },
            uniquingKeysWith: { first, _ in first })
        var findings: [Finding] = []
        for file in remoteKept.sorted(by: { $0.relativePath < $1.relativePath }) {
            guard let localSize = localBySize[file.relativePath] else {
                findings.append(.gap(relativePath: file.relativePath, size: file.size))
                continue
            }
            if localSize != file.size {
                findings.append(
                    .sizeDrift(
                        relativePath: file.relativePath, remote: file.size,
                        local: localSize))
            }
        }
        let remotePaths = Set(remoteKept.map(\.relativePath))
        for file in localKept.sorted(by: { $0.relativePath < $1.relativePath })
        where !remotePaths.contains(file.relativePath) {
            findings.append(.localOnly(relativePath: file.relativePath))
        }
        // Counts, stated independently of the per-file walk: a report that
        // only ever prints per-file rows can still leave a caller wondering
        // whether the totals agreed. Both counts are post-exclusion.
        let gaps = findings.filter { if case .gap = $0 { return true } else { return false } }
        if remoteKept.count != localKept.count + gaps.count {
            findings.append(
                .countMismatch(remote: remoteKept.count, local: localKept.count))
        }
        // `localBySize` is already post-exclusion, so a pin on an excluded
        // path is silently out of scope here — which is right: the bytes it
        // pins were never imported.
        for pin in pinnedHashes where localBySize[pin.relativePath] != nil {
            guard let actual = localHash(pin.relativePath) else { continue }
            if actual.caseInsensitiveCompare(pin.sha256) != .orderedSame {
                findings.append(
                    .hashDrift(
                        relativePath: pin.relativePath, pinned: pin.sha256,
                        actual: actual, source: pin.source))
            }
        }
        return findings
    }

    /// The refusal text for a directory whose bytes drifted (tightening 4).
    public static func immutabilityRefusal(
        directory: String, violations: [Finding]
    ) -> String {
        let rows = violations.map { finding -> String in
            switch finding {
            case .sizeDrift(let path, let remote, let local):
                "  \(path): remote \(remote) bytes, local \(local) bytes"
            case .hashDrift(let path, let pinned, let actual, let source):
                "  \(path): pinned \(String(pinned.prefix(12)))… (\(source)), "
                    + "local \(String(actual.prefix(12)))…"
            default:
                "  \(finding)"
            }
        }
        return "'\(directory)' is already in this workspace and the remote "
            + "bytes DIFFER from the local ones — an immutability violation, "
            + "not a gap to fill:\n" + rows.joined(separator: "\n")
            + "\nNothing was overwritten. runs/ is immutable: resolve this by "
            + "hand (keep the local directory and import the remote one under "
            + "a different name, or establish which is the real run) rather "
            + "than by re-running the import."
    }

    // MARK: - Authoring-locus divergence (open-issues §8 residual (a))

    // §8's forensics found the real defect behind the emptied-looking s4x
    // manifest: the study was authored/attached on the CLUSTER, this policy is
    // runs-only, and nothing ever surfaced that the Mac's live manifest holds
    // fewer arms than the run evidence sitting right next to it (71 of 210
    // study directories in that workspace carried the signature). Decided
    // 2026-08-20: the import STAYS runs-only — the Mac workspace is the source
    // of truth, and a shell written once at creation is byte-indistinguishable
    // from a draft the researcher deliberately emptied, so no mechanical
    // reconcile of `experiments/` can be safe — but the divergence is now
    // REPORTED loudly, per study, with the run snapshots named as evidence.
    // Adopting cluster authoring remains a deliberate human act; this section
    // only makes the gap impossible to miss.

    /// The arm counts one manifest holds — the same closed watched pair as the
    /// `armsCleared` lifecycle gate (`concepts` + `conditions`;
    /// `variantConditions` deliberately outside, for the same reason: the
    /// panel clears it legitimately, so watching it would cry wolf).
    public struct ManifestArms: Sendable, Equatable {
        public var studyName: String
        public var concepts: Int
        public var conditions: Int
        /// The manifest's lifecycle status, verbatim, when it declares one.
        public var status: String?

        public init(
            studyName: String, concepts: Int, conditions: Int,
            status: String? = nil
        ) {
            self.studyName = studyName
            self.concepts = concepts
            self.conditions = conditions
            self.status = status
        }
    }

    /// Parse the watched pair out of manifest JSON bytes — a run directory's
    /// `experiment.json` snapshot, or a live `experiments/<name>/` manifest.
    /// Nil when the bytes are not a manifest (no decodable `name`), which is
    /// the same answer as "no evidence" for every caller.
    public static func manifestArms(fromManifestJSON data: Data) -> ManifestArms? {
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any],
            let name = dictionary["name"] as? String,
            !name.isEmpty
        else { return nil }
        return ManifestArms(
            studyName: name,
            concepts: (dictionary["concepts"] as? [Any])?.count ?? 0,
            conditions: (dictionary["conditions"] as? [Any])?.count ?? 0,
            status: dictionary["status"] as? String)
    }

    /// One study whose live manifest holds fewer arms than its own imported
    /// run evidence does.
    public struct AuthoringDivergence: Sendable, Equatable {
        public var studyName: String
        /// The live workspace manifest — nil when
        /// `experiments/<name>/experiment.json` does not exist at all.
        public var live: ManifestArms?
        /// The strongest evidence across this study's run snapshots, per axis.
        public var evidenceConcepts: Int
        public var evidenceConditions: Int
        /// The run directories whose snapshots exceed the live copy, sorted
        /// (stamp-prefixed names sort chronologically).
        public var evidencedBy: [String]

        public init(
            studyName: String, live: ManifestArms?, evidenceConcepts: Int,
            evidenceConditions: Int, evidencedBy: [String]
        ) {
            self.studyName = studyName
            self.live = live
            self.evidenceConcepts = evidenceConcepts
            self.evidenceConditions = evidenceConditions
            self.evidencedBy = evidencedBy
        }
    }

    /// Compare every run snapshot in this pass against the live workspace
    /// manifests. A study diverges when the live copy holds FEWER arms on
    /// either axis than its own run evidence — counts, not bytes: a
    /// same-count-different-content drift is the epoch guard's job, and this
    /// report exists for the loss that guard cannot see (a manifest that was
    /// never armed on this machine at all).
    public static func authoringDivergences(
        snapshots: [(runName: String, arms: ManifestArms)],
        liveArms: (String) -> ManifestArms?
    ) -> [AuthoringDivergence] {
        var byStudy: [String: [(runName: String, arms: ManifestArms)]] = [:]
        for snapshot in snapshots {
            byStudy[snapshot.arms.studyName, default: []].append(snapshot)
        }
        var divergences: [AuthoringDivergence] = []
        for study in byStudy.keys.sorted() {
            let rows = byStudy[study] ?? []
            let evidenceConcepts = rows.map(\.arms.concepts).max() ?? 0
            let evidenceConditions = rows.map(\.arms.conditions).max() ?? 0
            let live = liveArms(study)
            let liveConcepts = live?.concepts ?? 0
            let liveConditions = live?.conditions ?? 0
            guard
                evidenceConcepts > liveConcepts
                    || evidenceConditions > liveConditions
            else { continue }
            divergences.append(
                AuthoringDivergence(
                    studyName: study,
                    live: live,
                    evidenceConcepts: evidenceConcepts,
                    evidenceConditions: evidenceConditions,
                    evidencedBy: rows
                        .filter {
                            $0.arms.concepts > liveConcepts
                                || $0.arms.conditions > liveConditions
                        }
                        .map(\.runName)
                        .sorted()))
        }
        return divergences
    }

    /// The report line for one divergence — loud, and it hands the researcher
    /// the deliberate repair rather than performing any part of it.
    public static func message(divergence: AuthoringDivergence) -> String {
        let liveText: String =
            if let live = divergence.live {
                "the live manifest holds \(live.concepts) concept"
                    + "\(live.concepts == 1 ? "" : "s") / \(live.conditions) "
                    + "condition\(live.conditions == 1 ? "" : "s")"
                    + (live.status.map { " (status: \($0))" } ?? "")
            } else {
                "no live manifest exists in this workspace "
                    + "(experiments/\(divergence.studyName)/experiment.json "
                    + "is absent)"
            }
        let n = divergence.evidencedBy.count
        let named = divergence.evidencedBy.prefix(3).joined(separator: ", ")
            + (n > 3 ? ", +\(n - 3) more" : "")
        let newest = divergence.evidencedBy.last ?? "<run>"
        return "\(divergence.studyName): AUTHORING DIVERGENCE — \(liveText), "
            + "but \(n) imported run snapshot\(n == 1 ? "" : "s") "
            + "carr\(n == 1 ? "ies" : "y") \(divergence.evidenceConcepts) "
            + "concept\(divergence.evidenceConcepts == 1 ? "" : "s") / "
            + "\(divergence.evidenceConditions) condition"
            + "\(divergence.evidenceConditions == 1 ? "" : "s") (\(named)). "
            + "The study was authored or attached on the cluster; this import "
            + "is runs-only and never writes experiments/. The run snapshot "
            + "stays authoritative for what executed. To adopt the cluster "
            + "authoring, copy runs/\(newest)/experiment.json over the live "
            + "DRAFT manifest yourself, then run `steerlab-cli experiment "
            + "verify \(divergence.studyName)` — a frozen manifest is never "
            + "overwritten (duplicate instead)."
    }

    // MARK: - `--since`

    /// Normalize a `--since` value to the run-stamp's own lexicographic
    /// space (`yyyyMMddTHHmmssSSS`), so the filter compares against the
    /// directory NAME — the run's start — rather than a filesystem mtime that
    /// a later touch would move.
    ///
    /// Accepts `yyyy-MM-dd`, `yyyyMMdd`, `yyyy-MM-ddTHH:mm:ss`, and a raw run
    /// stamp. Returns nil for anything else, which the caller turns into a
    /// typed usage refusal rather than a silently ignored flag.
    public static func normalizedSince(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return nil }
        let compact = trimmed
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: ":", with: "")
        guard compact.allSatisfy({ $0.isNumber || $0 == "T" }) else { return nil }
        let parts = compact.split(separator: "T", omittingEmptySubsequences: false)
        guard let day = parts.first, day.count == 8 else { return nil }
        if parts.count == 1 { return String(day) + "T000000000" }
        guard parts.count == 2 else { return nil }
        let time = String(parts[1])
        guard time.count <= 9, time.allSatisfy(\.isNumber) else { return nil }
        return String(day) + "T" + time.padding(
            toLength: 9, withPad: "0", startingAt: 0)
    }

    /// Whether a classified directory passes a `--since` filter. A directory
    /// with no parseable stamp always passes: conservative, like the unknown
    /// shape rule.
    public static func passesSince(
        _ classification: Classification, since: String?
    ) -> Bool {
        guard let since else { return true }
        guard let stamp = classification.stamp else { return true }
        return stamp.padding(toLength: max(stamp.count, since.count),
                             withPad: "0", startingAt: 0)
            >= since
    }
}

import Foundation

/// Evidence-status classification of a run directory (team finding P4):
/// a successfully executed pilot must not be mistakable for a frozen study.
/// The RULE is pure and unit-tested; `RunResults.Model` computes its inputs
/// from the run's own artifacts (manifest snapshot, experiment-hash stamp,
/// generation records) plus the workspace's live manifest and validate
/// evidence. Absence of an input degrades the read loudly (chips), never
/// silently upgrades it.
extension RunResults {

    public enum RunClass: String, Sendable, Equatable {
        /// Run stamped by a frozen manifest, and the stamped hash still
        /// matches the live manifest (epoch-verified).
        case frozenEvidenceGrade
        /// The run was frozen, but its stamped experiment hash no longer
        /// matches the live manifest. Never evidence-grade.
        case frozenEpochMismatch
        /// The run was frozen, but the live manifest/epoch was unavailable.
        /// Never silently promoted to evidence-grade.
        case frozenUnverified
        /// The manifest snapshot carries `freezeForced` — non-citable by
        /// stamp, checkable after the fact.
        case forcedFreeze
        /// Draft manifest, but scope-matched validate evidence exists.
        case validatedDraft
        /// No freeze, no matching validate evidence: a pilot.
        case draftPilot
        /// The stage that wrote this run DID NOT COMPLETE (retention
        /// 2026-07-24): a failure record retrieved from a failed job. The
        /// data in it is real and worth inspecting; the run is not a
        /// result. Dominates every other class — a partial run of a FROZEN
        /// study is still not a result, and this is the one classification
        /// a frozen stamp must not be able to outrank.
        case partialFailedRun
    }

    /// Tri-state for checks whose inputs may be unavailable (remote runs,
    /// vanished workspaces): unknown must render as unknown, not as a pass.
    public enum CheckState: Sendable, Equatable {
        case verified
        case failed
        case unknown
    }

    public struct Classification: Sendable, Equatable {
        public var runClass: RunClass
        /// Epoch check: stamped experiment hash vs the LIVE manifest's hash.
        public var epoch: CheckState
        /// P4's revision chip: the manifest never pinned a revision but the
        /// generation records resolved one at run time.
        public var unpinnedResolvedRevision: String?
        /// experiment.json bytes were present but the STRICT manifest decode
        /// failed (engine version skew: unknown enum values, shape drift).
        /// Classification then rests on a tolerant partial read of the
        /// critical keys and must render as a CAUTION — a frozen or
        /// freeze-forced server run whose snapshot a newer engine wrote must
        /// never lose its banner to a swallowed decode error (F8).
        public var snapshotUnreadable: Bool = false

        public var label: String {
            switch runClass {
            case .frozenEvidenceGrade: "Frozen (evidence-grade)"
            case .frozenEpochMismatch: "Frozen — epoch mismatch"
            case .frozenUnverified: "Frozen — epoch unverified"
            case .forcedFreeze: "Forced freeze — non-citable"
            case .validatedDraft: "Validated draft"
            case .draftPilot: "Draft/pilot"
            case .partialFailedRun: "Incomplete — failure record"
            }
        }

        public var detail: String {
            switch runClass {
            case .frozenEvidenceGrade:
                "run stamped by a frozen manifest; stamp matches the live manifest"
            case .frozenEpochMismatch:
                "frozen manifest snapshot, but the live manifest changed since "
                    + "this run (epoch mismatch) — do not cite without re-running"
            case .frozenUnverified:
                "frozen manifest snapshot; live manifest unavailable, so the "
                    + "epoch could not be verified — do not treat as evidence-grade"
            case .forcedFreeze:
                "freeze gates were force-skipped (stamped freezeForced) — "
                    + "non-citable by stamp"
            case .validatedDraft:
                "draft manifest with matching validate evidence for its exact "
                    + "validation scope"
            case .draftPilot:
                "no freeze and no matching validate evidence — treat as a pilot, "
                    + "not a result"
            case .partialFailedRun:
                "the stage that wrote this run did not complete (see FAILED.md "
                    + "and run-status.json) — the data it holds is real and "
                    + "retryable, but this is a failure record, not a result"
            }
        }

        /// The unresolved-revision warning line, when it applies.
        public var revisionWarning: String? {
            guard let revision = unpinnedResolvedRevision else { return nil }
            return "revision \(String(revision.prefix(12))) resolved at run time "
                + "but never pinned — pin before evidence use"
        }

        /// The unreadable-snapshot caution line, when it applies (F8).
        public var snapshotWarning: String? {
            guard snapshotUnreadable else { return nil }
            return "manifest snapshot present but not fully decodable "
                + "(engine version skew?) — treat as unverified"
        }

        /// Whether the header's epoch CHIPS should render: the frozen
        /// classes already bake the epoch state into their label, so a chip
        /// there could only repeat or contradict it — suppressed. Non-frozen
        /// classes carry no epoch in the label; the chip is the only signal.
        public var showsEpochChips: Bool {
            switch runClass {
            case .frozenEvidenceGrade, .frozenEpochMismatch, .frozenUnverified:
                false
            case .forcedFreeze, .validatedDraft, .draftPilot:
                true
            case .partialFailedRun:
                // Incompleteness is the headline; an epoch chip beside it
                // would invite reading the run as a result with a caveat.
                false
            }
        }

        /// Whether this run may be cited or counted as a result at all.
        /// The single question every consumer should ask instead of
        /// pattern-matching the enum — a new non-citable class must not
        /// require every call site to be found and updated.
        public var isCitableResult: Bool {
            switch runClass {
            case .frozenEvidenceGrade: true
            case .frozenEpochMismatch, .frozenUnverified, .forcedFreeze,
                .validatedDraft, .draftPilot, .partialFailedRun:
                false
            }
        }
    }

    /// Tolerant read of a manifest snapshot: the strict decode when it
    /// works, plus raw JSONSerialization reads of the classification-critical
    /// keys when it does not (F8). A snapshot written by a newer engine with
    /// an unknown enum value must degrade to a LOUD "unreadable" state with
    /// its status/freezeForced still honored where legible — never to a
    /// silently vanished classification banner.
    struct ManifestSnapshotReading: Sendable {
        var strict: ExperimentManifest?
        var statusRaw: String?
        var freezeForced: Bool?
        var name: String?
        var modelRevision: String?

        /// Bytes were present but the strict decode failed.
        var isUnreadable: Bool { strict == nil }
    }

    static func manifestSnapshotReading(from data: Data) -> ManifestSnapshotReading {
        if let manifest = try? JSONDecoder().decode(ExperimentManifest.self, from: data) {
            return ManifestSnapshotReading(
                strict: manifest,
                statusRaw: manifest.status.rawValue,
                freezeForced: manifest.freezeForced,
                name: manifest.name,
                modelRevision: manifest.modelRevision)
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any]
        else { return ManifestSnapshotReading() }
        return ManifestSnapshotReading(
            strict: nil,
            statusRaw: dictionary["status"] as? String,
            freezeForced: dictionary["freezeForced"] as? Bool,
            name: dictionary["name"] as? String,
            modelRevision: dictionary["modelRevision"] as? String)
    }

    /// The pure classification rule. `hasMatchingValidateEvidence` and
    /// `stampedHashMatchesLive` are nil when the check could not be run
    /// (no snapshot, no live manifest, remote browse).
    public static func classify(
        manifestStatus: String?,
        freezeForced: Bool?,
        hasMatchingValidateEvidence: Bool?,
        stampedHashMatchesLive: Bool?,
        manifestRevision: String?,
        recordRevisions: Set<String>
    ) -> Classification {
        let epoch: CheckState =
            switch stampedHashMatchesLive {
            case .some(true): .verified
            case .some(false): .failed
            case .none: .unknown
            }
        let runClass: RunClass
        if freezeForced == true {
            runClass = .forcedFreeze
        } else if manifestStatus == "frozen" || manifestStatus == "complete" {
            switch epoch {
            case .verified: runClass = .frozenEvidenceGrade
            case .failed: runClass = .frozenEpochMismatch
            case .unknown: runClass = .frozenUnverified
            }
        } else if hasMatchingValidateEvidence == true {
            runClass = .validatedDraft
        } else {
            runClass = .draftPilot
        }
        let unpinned: String? =
            if manifestRevision == nil, let resolved = recordRevisions.first,
                recordRevisions.count == 1
            {
                resolved
            } else if manifestRevision == nil, recordRevisions.count > 1 {
                // Mixed resolved revisions is its own smell; surface one and
                // let the count speak in the raw records.
                recordRevisions.sorted().first
            } else {
                nil
            }
        return Classification(
            runClass: runClass,
            epoch: epoch,
            unpinnedResolvedRevision: unpinned)
    }

    /// Disk wrapper: classify a LOCAL run directory from its own artifacts
    /// plus the workspace's live state. Every input is best-effort — a
    /// missing snapshot classifies as draft/pilot with epoch unknown; a
    /// PRESENT but strictly-undecodable snapshot classifies from the
    /// tolerant partial read and flags `snapshotUnreadable` (F8).
    public static func classification(
        runDirectory: URL, records: [Record]
    ) -> Classification {
        let snapshotURL = runDirectory.appending(component: "experiment.json")
        let reading = (try? Data(contentsOf: snapshotURL))
            .map(manifestSnapshotReading(from:))
        return classification(
            reading: reading, runDirectory: runDirectory, records: records)
    }

    /// The local classification rule over an already-performed snapshot
    /// read: epoch check against the live manifest (by the snapshot's name,
    /// legible even from a tolerant read) and validate-evidence lookup
    /// (needs the strict manifest — unknown when the snapshot is only
    /// partially readable, never silently a pass).
    static func classification(
        reading: ManifestSnapshotReading?, runDirectory: URL, records: [Record]
    ) -> Classification {
        // Incompleteness DOMINATES (retention 2026-07-24). Checked before
        // anything else because no other input can outrank it: a partial
        // run of a frozen, epoch-matched study is still a failure record,
        // and reading its frozen stamp first would label it evidence-grade.
        if RunStatusFile.isPartial(at: runDirectory) {
            return Classification(
                runClass: .partialFailedRun, epoch: .unknown,
                unpinnedResolvedRevision: nil)
        }
        var stampMatchesLive: Bool?
        if let name = reading?.name,
            let stamped = stampedExperimentHash(at: runDirectory),
            let live = try? ExperimentStore.load(name: name)
        {
            // Legacy fallback mirrors `RunEpoch.check`: runs stamped before
            // `createdAt` left the hash (2026-08-17) match via the legacy
            // canonicalization of the same manifest.
            stampMatchesLive =
                stamped == ExperimentStore.manifestHash(live)
                || stamped == ExperimentStore.legacyManifestHash(live)
        }

        var hasEvidence: Bool?
        if let manifest = reading?.strict {
            hasEvidence = ExperimentStore.validationEvidence(for: manifest) != nil
        }

        var result = classify(
            manifestStatus: reading?.statusRaw,
            freezeForced: reading?.freezeForced,
            hasMatchingValidateEvidence: hasEvidence,
            stampedHashMatchesLive: stampMatchesLive,
            manifestRevision: reading?.modelRevision,
            recordRevisions: Set(records.compactMap(\.modelRevision)))
        result.snapshotUnreadable = reading?.isUnreadable ?? false
        return result
    }

    /// The run's experiment-hash stamp: `experiment-hash.txt` first, then
    /// the canonical config.json `experimentHash` (same fallback order as
    /// the epoch guards on both engines).
    static func stampedExperimentHash(at runDirectory: URL) -> String? {
        if let text = try? String(
            contentsOf: runDirectory.appending(component: "experiment-hash.txt"),
            encoding: .utf8)
        {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        guard
            let data = try? Data(
                contentsOf: runDirectory.appending(component: RunMetadata.fileName)),
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any],
            let hash = dictionary["experimentHash"] as? String,
            !hash.isEmpty
        else { return nil }
        return hash
    }
}

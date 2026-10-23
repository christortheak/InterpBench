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
        /// The gates a forced freeze skipped, as the snapshot's
        /// `forcedGatesSkipped` stamp records them; nil when the snapshot
        /// does not say (never inferred).
        public var forcedGatesSkipped: [String]? = nil
        /// The conditions the snapshot's `capabilityBatteryNotApplied` stamp
        /// records the capability battery was not applied to, and why. Empty
        /// when the snapshot stamps none — an older run is never credited
        /// with an exemption its manifest does not carry.
        public var batteryNotApplied: [ExperimentManifest.BatteryNotApplied] = []

        /// Facts, not verdicts (2026-10-23 ruling): the badge says what the
        /// freeze state IS — frozen, frozen with force, a draft — and which
        /// checks applied, and leaves what the run supports to the reader.
        public var label: String {
            switch runClass {
            case .frozenEvidenceGrade:
                batteryNotApplied.isEmpty
                    ? "Frozen — every freeze check passed"
                    : "Frozen — every freeze check that applies passed"
            case .frozenEpochMismatch: "Frozen — epoch mismatch"
            case .frozenUnverified: "Frozen — epoch unverified"
            case .forcedFreeze: "Frozen with force"
            case .validatedDraft: "Validated draft"
            case .draftPilot: "Draft/pilot"
            case .partialFailedRun: "Incomplete — failure record"
            }
        }

        public var detail: String {
            switch runClass {
            case .frozenEvidenceGrade:
                "frozen without force; the settings stamp on this run matches "
                    + "the live manifest"
            case .frozenEpochMismatch:
                "frozen manifest snapshot, but the live manifest changed after "
                    + "this run (epoch mismatch): these results describe "
                    + "settings the study no longer has"
            case .frozenUnverified:
                "frozen manifest snapshot; the live manifest is unavailable, so "
                    + "whether this run matches the study's current settings "
                    + "could not be checked"
            case .forcedFreeze:
                "freeze checks skipped with --force: "
                    + (forcedGatesSkipped.map {
                        $0.isEmpty ? "none (every check would have passed)"
                            : $0.joined(separator: ", ")
                    } ?? "not recorded in the snapshot")
                    + " — the manifest records this permanently "
                    + "(freezeForced, forcedGatesSkipped)"
            case .validatedDraft:
                "draft manifest with matching validate evidence for its exact "
                    + "validation scope"
            case .draftPilot:
                "no freeze and no matching validate evidence: the study's "
                    + "settings were not fixed before this run"
            case .partialFailedRun:
                "the stage that wrote this run did not complete (see FAILED.md "
                    + "and run-status.json) — the data it holds is real and the "
                    + "stage can be retried, but the run is incomplete"
            }
        }

        /// One line per condition the snapshot stamps the capability battery
        /// was not applied to — the engines' own sentence — shown beside the
        /// freeze state. Empty for every other run.
        public var batteryNotAppliedLines: [String] {
            batteryNotApplied.map(FreezePolicy.batteryNotAppliedSentence)
        }

        /// The unresolved-revision warning line, when it applies.
        public var revisionWarning: String? {
            guard let revision = unpinnedResolvedRevision else { return nil }
            return "revision \(String(revision.prefix(12))) resolved at run time "
                + "but never pinned — the study does not record which revision "
                + "ran until it is pinned"
        }

        /// The unreadable-snapshot caution line, when it applies (F8).
        public var snapshotWarning: String? {
            guard snapshotUnreadable else { return nil }
            return "manifest snapshot present but not fully decodable "
                + "(engine version skew?) — the freeze state shown comes from "
                + "a partial read"
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
        var forcedGatesSkipped: [String]?
        var batteryNotApplied: [ExperimentManifest.BatteryNotApplied] = []

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
                modelRevision: manifest.modelRevision,
                forcedGatesSkipped: manifest.forcedGatesSkipped,
                batteryNotApplied: manifest.capabilityBatteryNotApplied ?? [])
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any]
        else { return ManifestSnapshotReading() }
        // The same tolerant read as `freezeForced`: the stamps a reader must
        // not lose to an undecodable snapshot.
        let notApplied = (dictionary["capabilityBatteryNotApplied"] as? [[String: Any]] ?? [])
            .compactMap { entry -> ExperimentManifest.BatteryNotApplied? in
                guard let condition = entry["condition"] as? String,
                    let reason = entry["reason"] as? String
                else { return nil }
                return .init(condition: condition, reason: reason)
            }
        return ManifestSnapshotReading(
            strict: nil,
            statusRaw: dictionary["status"] as? String,
            freezeForced: dictionary["freezeForced"] as? Bool,
            name: dictionary["name"] as? String,
            modelRevision: dictionary["modelRevision"] as? String,
            forcedGatesSkipped: dictionary["forcedGatesSkipped"] as? [String],
            batteryNotApplied: notApplied)
    }

    /// Adds the snapshot's freeze stamps — the gates a forced freeze skipped
    /// and the battery-not-applied entries — to a classification, for the
    /// header to state beside the freeze state. Reads only what the snapshot
    /// stamps.
    static func withSnapshotFacts(
        _ classification: Classification, reading: ManifestSnapshotReading?
    ) -> Classification {
        var result = classification
        result.forcedGatesSkipped = reading?.forcedGatesSkipped
        result.batteryNotApplied = reading?.batteryNotApplied ?? []
        return result
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
        return withSnapshotFacts(result, reading: reading)
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

import Foundation
import SteeringKit

/// Gate decisions over a manifest and supplied evidence values. This owner
/// never reads files, chooses a workspace, loads a model or performs a commit.
enum FreezePolicy {
    typealias Outcome = ExperimentStore.FreezeGateOutcome

    struct ValidationFacts {
        let hasMatchingEvidence: Bool
        var vacuousProblem: String? = nil
        var vacuousRepair: String? = nil
    }

    struct BatteryFacts {
        let hasMatchingEvidence: Bool
        var results: [CapabilityBatteryConditionResult] = []
        var expectedHash: String? = nil
    }

    static func freezeRefusal(_ failures: [ExperimentStore.FreezeGateOutcome]) -> Error? {
        guard let first = failures.first else { return nil }
        if let underlying = first.underlying { return underlying }
        let failed = Set(failures.map(\.gate))
        return ExperimentError(
            refusal: FreezeRefusal(
                gate: first.gate,
                gates: FreezeGate.allCases.filter(failed.contains),
                reason: first.refusal, repairAction: first.repairAction))
    }

    static func checkedOutcome(
        _ gate: FreezeGate, name: String, repairAction: String,
        result: Result<Void, Error>
    ) -> Outcome? {
        do {
            try result.get()
            return nil
        } catch let error as ExperimentError {
            return ExperimentStore.FreezeGateOutcome(
                gate: gate, refusal: error.reason,
                forced: ManifestDeclarationPolicy.plainGateReason(
                    error.reason, experimentName: name),
                repairAction: repairAction)
        } catch {
            return ExperimentStore.FreezeGateOutcome(
                gate: gate, refusal: "\(error)", forced: "\(error)",
                repairAction: repairAction, underlying: error)
        }
    }

    static func revision(_ manifest: ExperimentManifest, name: String) -> Outcome? {
        guard manifest.modelRevision == nil else { return nil }
        let forced =
            "model revision is not pinned and \(manifest.modelID) is not in "
            + "the local HF cache"
        let repair = "create with --revision, load the model once, or freeze --force"
        return ExperimentStore.FreezeGateOutcome(
            gate: .revision,
            refusal: "cannot freeze '\(name)': \(forced) — \(repair)",
            forced: forced, repairAction: repair)
    }

    static func symbolicRevision(_ manifest: ExperimentManifest, name: String) -> Outcome? {
        guard let symbolic = ManifestDeclarationPolicy.symbolicRevisionProblem(manifest) else {
            return nil
        }
        return ExperimentStore.FreezeGateOutcome(
            gate: .revision, refusal: "cannot freeze '\(name)': \(symbolic)",
            forced: symbolic,
            repairAction: "pin the immutable commit the symbolic revision "
                + "resolves to, or freeze --force")
    }

    static func dtype(_ manifest: ExperimentManifest, name: String) -> Outcome? {
        guard let badDtype = ManifestDeclarationPolicy.unloadableStudyDtypeProblem(manifest) else {
            return nil
        }
        return ExperimentStore.FreezeGateOutcome(
            gate: .measurementPins, refusal: "cannot freeze '\(name)': \(badDtype)",
            forced: badDtype,
            repairAction: "repoint the study dtype at a loadable value, "
                + "or freeze --force")
    }

    static func validation(
        _ manifest: ExperimentManifest, name: String, runSubstrate: String,
        facts: ValidationFacts
    ) -> Outcome? {
        guard ManifestDeclarationPolicy.modelOutputSurfacesOperative(manifest) else { return nil }
        let usesLegacyConceptVectors =
            !manifest.concepts.isEmpty || !manifest.conditions.isEmpty
        guard usesLegacyConceptVectors,
            !ManifestDeclarationPolicy.optvecExemptFromValidateGate(manifest)
        else { return nil }
        guard
            facts.hasMatchingEvidence
        else {
            let forced =
                "no validate run matches its exact pins (model+revision, "
                + "concepts, neutral corpus) on the run substrate " + runSubstrate
            // The engine named is the one whose evidence this gate
            // reads — `steerlab-cli` cannot satisfy a server-substrate
            // gate (gate-5 dry run #2, P2). Sentence structure is
            // unchanged; only the binary moves.
            let repair =
                "Run '\(ManifestDeclarationPolicy.validateCLI(forRunSubstrate: runSubstrate)) "
                + "experiment validate \(name)' first, or "
                + "freeze --force to record an unvalidated experiment"
            return ExperimentStore.FreezeGateOutcome(
                gate: .validateEvidence,
                refusal: "cannot freeze '\(name)': \(forced). \(repair)",
                forced: forced, repairAction: repair)
        }
        // Evidence EXISTS but probed nothing: same gate id, a remedy
        // naming the missing files (2026-08-17).
        guard
            let vacuous = facts.vacuousProblem
        else { return nil }
        // P5 (dry run #1): the gate's own repair FAILED as given.
        // Authoring the named validation.jsonl makes it appear after
        // an attach that pinned it absent, which is a verify()
        // violation — so the very next `validate` refuses. The
        // machine repair now carries the re-attach that re-pins it;
        // the composed PROSE is unchanged (it is the cross-engine
        // refusal string, asserted whole by VacuousValidationTests).
        let repair = facts.vacuousRepair ?? vacuous
        return ExperimentStore.FreezeGateOutcome(
            gate: .validateEvidence, refusal: "cannot freeze '\(name)': \(vacuous)",
            forced: vacuous, repairAction: repair)
    }

    // MARK: - The capability battery's one exemption

    /// Why the capability battery cannot be applied to one variant
    /// condition, or nil when it applies.
    ///
    /// Read from the condition's INLINE agent artifact — the bytes the
    /// manifest pins and the content hash covers — so the answer needs no
    /// file read and cannot differ between the freeze that stamps it and a
    /// later reader. A forward-referenced condition has no agent yet and
    /// keeps its own, older exemption. Server twin:
    /// `freeze_policy.battery_exemption_reason`.
    static func batteryExemptionReason(
        _ variant: ExperimentManifest.VariantCondition
    ) -> String? {
        guard variant.fromPromotion == nil,
            !(variant.artifact.interventionPolicies ?? []).isEmpty
        else { return nil }
        return ExperimentManifest.BatteryNotApplied.interventionPolicyReason
    }

    /// The `capabilityBatteryNotApplied` entries for a manifest, in manifest
    /// order; empty when the battery applies to every condition or the
    /// battery gate is not asked of this study kind at all. Server twin:
    /// `freeze_policy.battery_not_applied`.
    static func batteryNotApplied(
        _ manifest: ExperimentManifest
    ) -> [ExperimentManifest.BatteryNotApplied] {
        guard ManifestDeclarationPolicy.modelOutputSurfacesOperative(manifest) else {
            return []
        }
        return manifest.variantConditions.compactMap { variant in
            batteryExemptionReason(variant).map {
                .init(condition: variant.name, reason: $0)
            }
        }
    }

    /// The researcher-facing sentence for one entry. Both engines emit it
    /// verbatim (server twin: `freeze_policy.battery_not_applied_sentence`).
    static func batteryNotAppliedSentence(
        _ entry: ExperimentManifest.BatteryNotApplied
    ) -> String {
        if entry.reason == ExperimentManifest.BatteryNotApplied.interventionPolicyReason {
            return "The capability battery was not applied to \(entry.condition), "
                + "because its agent uses an intervention policy, which the "
                + "battery cannot run. This study has no capability control "
                + "for that agent."
        }
        return "The capability battery was not applied to \(entry.condition) "
            + "(recorded reason: \(entry.reason)). This study has no capability "
            + "control for that agent."
    }

    /// The researcher-facing sentence for an ERROR row: the validate run
    /// could not run the battery for this condition, and `error` is the
    /// engine's own account of why. Ends with a full stop exactly once.
    static func batteryErrorSentence(condition: String, error: String) -> String {
        let said = error.trimmingCharacters(in: .whitespacesAndNewlines)
        let stop = said.hasSuffix(".") ? "" : "."
        return "The capability battery could not run for \(condition): "
            + "\(said)\(stop)"
    }

    static func checkVariantBatteryEvidence(
        _ manifest: ExperimentManifest, facts: BatteryFacts
    ) throws {
        guard !manifest.variantConditions.isEmpty else { return }
        let name = manifest.name
        guard
            facts.hasMatchingEvidence
        else {
            throw ExperimentError(
                reason: "cannot freeze '\(name)': no validate run matches its exact pins "
                    + "(model+revision, variant capability battery). Variant studies "
                    + "validate the pinned battery per condition — run 'steerlab-cli "
                    + "experiment validate \(name)' first, or freeze --force")
        }
        let results = Dictionary(
            facts.results
                .map { ($0.condition, $0) },
            uniquingKeysWith: { first, _ in first })
        // Forward-referenced conditions (stage 4) are exempt: their agent
        // does not exist at validate time — their battery evidence is the
        // RUN's per-condition battery, produced after server-side
        // resolution.
        // A condition whose agent carries intervention policies is exempt
        // too (maintainer ruling 2026-10-04): the battery cannot run a
        // policy, so the evidence could never exist and only --force could
        // freeze the study. The exemption is this narrow on purpose —
        // baseline and every other condition are required exactly as before
        // — and freeze records it in the `capabilityBatteryNotApplied` stamp
        // instead of marking the study forced.
        let required =
            ["baseline"]
            + manifest.variantConditions
            .filter { $0.fromPromotion == nil && batteryExemptionReason($0) == nil }
            .map(\.name)
        // A not-applicable row is a record, never a score: it cannot stand in
        // for evidence a REQUIRED condition owes (evidence is matched by
        // scope, which does not cover conditions — a row written while a
        // condition's agent carried a policy must not satisfy the gate after
        // that agent is swapped for one the battery can run). Server twin:
        // the `accuracy is None` test in `check_battery_evidence`.
        //
        // An error row is a record too: the validate run could not run the
        // battery for that condition (its agent did not load), so there is no
        // reading. It is missing evidence for the gate, exactly as on the
        // server, and the refusal carries the row's own account of why.
        let missing = required.filter { results[$0]?.isScored != true }
        guard missing.isEmpty else {
            let failures = missing.compactMap { condition in
                results[condition]?.error.map {
                    batteryErrorSentence(condition: condition, error: $0)
                }
            }
            guard failures.isEmpty else {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': matching validate evidence has no "
                        + "capability-battery results for condition(s): "
                        + missing.joined(separator: ", ") + ". "
                        + failures.joined(separator: " ")
                        + " Repair what that message names, then re-run "
                        + "'steerlab-cli experiment validate \(name)' (each variant "
                        + "condition runs the pinned battery), or freeze --force")
            }
            throw ExperimentError(
                reason: "cannot freeze '\(name)': matching validate evidence has no "
                    + "capability-battery results for condition(s): "
                    + missing.joined(separator: ", ")
                    + " — re-run 'steerlab-cli experiment validate \(name)' (each "
                    + "variant condition runs the pinned battery), or freeze --force")
        }
        if let expected = facts.expectedHash {
            let drifted =
                required
                .filter { results[$0]?.batteryHash != expected }
                .sorted()
            guard drifted.isEmpty else {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': capability battery drifted since "
                        + "validation for condition(s): " + drifted.joined(separator: ", ")
                        + " — re-run 'steerlab-cli experiment validate \(name)', or "
                        + "freeze --force")
            }
        }
    }

    static func checkVariantValidity(
        _ manifest: ExperimentManifest, evidenceGradeVariants: Set<Int>
    ) throws {
        let name = manifest.name
        for (index, variant) in manifest.variantConditions.enumerated() {
            if variant.fromPromotion != nil {
                // Forward-referenced (stage 4): the artifact does not exist
                // at freeze time BY DESIGN — its pins land at run time on
                // the server and are recorded in the run directory.
                // verify() enforces the declaration shape.
                continue
            }
            if variant.artifactHash.isEmpty {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' has no "
                        + "pinned artifactHash")
            }
            for adapter in variant.artifact.adapters
            where (adapter.adapterHash ?? "").isEmpty {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' adapter "
                        + "'\(adapter.name)' has no adapterHash — re-save the variant "
                        + "with hashed adapter weights, or freeze --force")
            }
            let systemPrompt = (variant.artifact.systemPrompt ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !systemPrompt.isEmpty,
                (variant.artifact.systemPromptHash ?? "").isEmpty
            {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' has a "
                        + "system prompt but no systemPromptHash — re-save the variant, "
                        + "or freeze --force")
            }
            for injection in variant.artifact.injections
            where injection.vectorArtifactID.isEmpty {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' has an "
                        + "injection for '\(injection.concept)' without a "
                        + "vectorArtifactID pin")
            }
            // Trained-adapter arms owe the same story about their TRAINING
            // DATA (LoRA readiness §0 amendment 1). An evidence-grade adapter
            // whose dataset is not pinned into the manifest is unverifiable
            // the moment it freezes: the training files could change
            // afterwards with nothing to flag the drift. Exploratory adapters
            // are legal and produce an advisory instead, never a refusal.
            if !variant.artifact.adapters.isEmpty,
                evidenceGradeVariants.contains(index),
                (variant.trainingProvenance?.datasetManifestHash ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                throw ExperimentError(
                    reason: "cannot freeze '\(name)': variant '\(variant.name)' uses an "
                        + "evidence-grade adapter but carries no "
                        + "trainingProvenance.datasetManifestHash — its training data "
                        + "would stay outside the freeze pin surface. Re-attach the "
                        + "variant so freeze can pin the dataset manifest from the "
                        + "adapter's sidecar, or freeze --force")
            }
        }
    }
}

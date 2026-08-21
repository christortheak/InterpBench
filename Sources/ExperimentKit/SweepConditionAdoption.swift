import Foundation

/// Projected-condition adoption from a completed sweep run (2026-08-04 —
/// the systematic close of the server-manifest-mutation family).
///
/// THE INVENTORY this family covers. A server-executed verb may write into
/// its OWN copy of a draft manifest; the results bring the run home but not
/// the mutation, and each gap surfaces as a confusing refusal downstream:
///
/// 1. `_pin_model_revision` (every model-loading verb) — revision auto-pin.
///    Comes home via `EvidenceRevisionAdoption`: evidence import (manual +
///    auto) and sweep-run discovery all reconcile it.
/// 2. Sweep completion projects `<concept>-recommended` conditions
///    (`add_conditions`). Comes home HERE: discovery adopts the run's own
///    recommendations into a still-draft local manifest, so a subsequent
///    local `run` submission carries the arms the sweep selected.
/// 3. `complete_sweep_judgment` projects the same condition shape (deferred
///    judging) — same adoption applies (the recommendations.json is the
///    carrier either way).
/// 4. Variant studies may auto-pin the default capability battery
///    (`set_protocol`) — freeze-relevant only; the local freeze path
///    re-derives its own pin, so nothing to adopt.
///
/// Rules mirror `EvidenceRevisionAdoption`: adopting completes the
/// researcher's own declared intent; a CONFLICTING local condition is
/// flagged loudly and never overwritten; frozen manifests are immutable.
public enum SweepConditionAdoption {

    public enum Outcome: Equatable, Sendable {
        /// Conditions newly adopted (count) — plus how many were already
        /// present and identical.
        case adopted(experiment: String, added: Int, alreadyPresent: Int)
        /// Every projected condition is already in the local draft.
        case alreadyProjected(experiment: String)
        /// A same-named condition exists locally with a DIFFERENT cell —
        /// left untouched, named loudly.
        case conflict(experiment: String, condition: String)
        /// Nothing to adopt: the run has no successful recommendations.
        case noRecommendations
        /// No same-named local experiment.
        case noLocalExperiment
        /// The local manifest is not a draft — immutable.
        case notADraft(experiment: String)
        case saveFailed(experiment: String, message: String)
    }

    /// Adopt the run's successful recommendations as `<concept>-recommended`
    /// conditions on the same-named LOCAL draft.
    @discardableResult
    public static func adoptProjectedConditions(
        fromSweepRun runDirectory: URL
    ) -> Outcome {
        guard
            let snapshotData = try? Data(
                contentsOf: runDirectory.appending(component: "experiment.json")),
            let snapshot = try? JSONDecoder().decode(
                NameOnly.self, from: snapshotData)
        else { return .noLocalExperiment }
        guard
            let run = try? SweepRunCatalog.load(directory: runDirectory)
        else { return .noRecommendations }
        let selected = run.recommendations.compactMap {
            concept, recommendation
                -> (String, ExperimentManifest.SelectionProvenance)? in
            if case .selected(let provenance) = recommendation {
                return (concept, provenance)
            }
            return nil
        }
        guard !selected.isEmpty else { return .noRecommendations }
        guard var manifest = try? ExperimentStore.load(name: snapshot.name)
        else { return .noLocalExperiment }
        guard manifest.status == .draft else {
            return .notADraft(experiment: snapshot.name)
        }
        var added = 0
        var present = 0
        for (concept, provenance) in selected.sorted(by: { $0.0 < $1.0 }) {
            let name = "\(concept)-recommended"
            let condition = ExperimentManifest.Condition(
                name: name,
                slots: [
                    .init(
                        concept: concept,
                        layer: provenance.winningCell.layer,
                        alpha: provenance.winningCell.alpha)
                ],
                bandWidth: 1, alphaInNormUnits: true, selection: provenance)
            if let existing = manifest.conditions.first(where: { $0.name == name }) {
                let sameCell =
                    existing.slots.first?.layer == provenance.winningCell.layer
                    && existing.slots.first?.alpha == provenance.winningCell.alpha
                if sameCell {
                    present += 1
                    continue
                }
                // Loud, never overwritten: the local manifest is the source
                // of truth and a differing condition is a deliberate state.
                return .conflict(experiment: snapshot.name, condition: name)
            }
            manifest.conditions.append(condition)
            added += 1
        }
        guard added > 0 else { return .alreadyProjected(experiment: snapshot.name) }
        do {
            try ExperimentStore.save(manifest)
        } catch {
            return .saveFailed(
                experiment: snapshot.name, message: "\(error)")
        }
        return .adopted(
            experiment: snapshot.name, added: added, alreadyPresent: present)
    }

    /// Notice text for the panel — nil for the silent outcomes.
    public static func notice(
        for outcome: Outcome
    ) -> (message: String, isWarning: Bool)? {
        switch outcome {
        case .adopted(let experiment, let added, _):
            return (
                "adopted \(added) sweep-recommended condition\(added == 1 ? "" : "s") "
                    + "into draft '\(experiment)' (the sweep projected them "
                    + "on the server; the workspace is the source of truth)",
                false)
        case .conflict(let experiment, let condition):
            return (
                "draft '\(experiment)' already carries '\(condition)' with a "
                    + "DIFFERENT cell than the sweep recommends — left "
                    + "untouched; reconcile by hand",
                true)
        case .saveFailed(let experiment, let message):
            return ("could not save draft '\(experiment)': \(message)", true)
        case .alreadyProjected, .noRecommendations, .noLocalExperiment,
            .notADraft:
            return nil
        }
    }

    private struct NameOnly: Decodable { let name: String }
}

import Foundation

/// Model-revision adoption from imported validation evidence (external
/// review 2026-07-22, P1).
///
/// The gap this closes: a revision-less local draft submitted as a validate
/// bundle gets its revision RESOLVED on the server at model load
/// (`_pin_model_revision`), and the evidence bundle carries that resolved
/// revision home inside the run's manifest snapshot (`experiment.json`) —
/// but the Mac importer only landed the run. Local freeze readiness probes
/// the local HF cache for a revision, so a server-only model left the draft
/// permanently blocked on "model revision is not pinned" even though the
/// evidence names the exact commit the validation ran at.
///
/// Adopting the snapshot's revision into a still-unpinned local draft is
/// completing the researcher's own declared intent (they asked for THAT
/// validation to become THIS draft's freeze evidence), not policing. A
/// conflicting local pin is the opposite case: that evidence cannot satisfy
/// freeze for this draft, and the mismatch is flagged loudly instead of
/// silently overwritten.
public enum EvidenceRevisionAdoption {

    public enum Outcome: Equatable, Sendable {
        /// The local draft had no pinned revision; the evidence snapshot's
        /// revision was adopted into the draft and saved.
        case adopted(experiment: String, revision: String)
        /// The local pin already equals the evidence snapshot's — no-op.
        case alreadyPinned(experiment: String, revision: String)
        /// The local pin CONFLICTS with the evidence snapshot's revision:
        /// this evidence cannot satisfy freeze for this draft.
        case conflict(experiment: String, localRevision: String, evidenceRevision: String)
        /// The snapshot names a different base model than the local
        /// manifest — revision adoption would be meaningless.
        case modelMismatch(experiment: String, localModel: String, evidenceModel: String)
        /// The imported run carries no manifest snapshot, or the snapshot
        /// pins no revision — nothing to adopt.
        case noEvidenceRevision
        /// No local experiment matches the snapshot's name (e.g. evidence
        /// imported into a different workspace).
        case noLocalExperiment
        /// The local manifest is frozen/complete with no pin (legacy,
        /// predates the revision gate) — immutable, nothing adopted.
        case notADraft(experiment: String)
        /// The draft could not be saved (surfaced, never swallowed).
        case saveFailed(experiment: String, message: String)
    }

    /// The minimal slice of the run's manifest snapshot this needs.
    private struct Snapshot: Decodable {
        let name: String
        let modelID: String
        let modelRevision: String?
    }

    /// Reads `<runDirectory>/experiment.json` (the manifest snapshot every
    /// run writer on both engines stamps) and reconciles its model revision
    /// with the same-named LOCAL experiment. Pure decision + a draft save on
    /// the adoption arm; the caller surfaces `notice(for:)`.
    @discardableResult
    public static func adoptModelRevision(fromImportedRun runDirectory: URL) -> Outcome {
        guard
            let data = try? Data(
                contentsOf: runDirectory.appending(component: "experiment.json")),
            let snapshot = try? JSONDecoder().decode(Snapshot.self, from: data)
        else { return .noEvidenceRevision }
        guard
            let evidenceRevision = snapshot.modelRevision?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !evidenceRevision.isEmpty
        else { return .noEvidenceRevision }
        guard let local = try? ExperimentStore.load(name: snapshot.name) else {
            return .noLocalExperiment
        }
        guard local.modelID == snapshot.modelID else {
            return .modelMismatch(
                experiment: local.name,
                localModel: local.modelID,
                evidenceModel: snapshot.modelID)
        }
        if let localRevision = local.modelRevision {
            return localRevision == evidenceRevision
                ? .alreadyPinned(experiment: local.name, revision: localRevision)
                : .conflict(
                    experiment: local.name,
                    localRevision: localRevision,
                    evidenceRevision: evidenceRevision)
        }
        guard local.status == .draft else { return .notADraft(experiment: local.name) }
        do {
            try ExperimentStore.updateDraft(name: local.name) {
                $0.modelRevision = evidenceRevision
            }
        } catch {
            return .saveFailed(
                experiment: local.name, message: String(describing: error))
        }
        return .adopted(experiment: local.name, revision: evidenceRevision)
    }

    /// The user-visible note for an outcome; nil = stay silent (no-op
    /// cases: nothing to adopt, nothing local to adopt into, already
    /// matching, or an immutable legacy manifest).
    public static func notice(for outcome: Outcome) -> (message: String, isWarning: Bool)? {
        switch outcome {
        case .adopted(let experiment, let revision):
            return (
                "pinned model revision \(revision) into draft '\(experiment)' "
                    + "from the server's validation evidence — freeze can now "
                    + "see the revision the evidence was produced at",
                false)
        case .conflict(let experiment, let localRevision, let evidenceRevision):
            return (
                "imported validation evidence was produced at model revision "
                    + "\(evidenceRevision), but '\(experiment)' pins "
                    + "\(localRevision) — this evidence cannot satisfy freeze "
                    + "for that draft; re-run Validate Study at the pinned "
                    + "revision (or duplicate the study for the new one)",
                true)
        case .modelMismatch(let experiment, let localModel, let evidenceModel):
            return (
                "imported validation evidence was produced with model "
                    + "\(evidenceModel), but '\(experiment)' declares "
                    + "\(localModel) — this evidence cannot satisfy freeze "
                    + "for that draft",
                true)
        case .saveFailed(let experiment, let message):
            return (
                "could not pin the evidence revision into draft "
                    + "'\(experiment)': \(message)",
                true)
        case .alreadyPinned, .noEvidenceRevision, .noLocalExperiment, .notADraft:
            return nil
        }
    }
}

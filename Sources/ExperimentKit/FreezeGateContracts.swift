import Foundation

extension ExperimentStore {
    struct FreezeGateOutcome {
        let gate: FreezeGate
        let refusal: String
        let forced: String
        let repairAction: String
        var underlying: Error?
    }

    struct FreezeGateEntry {
        let gate: FreezeGate
        let evaluate: (ExperimentManifest) -> FreezeGateOutcome?
    }
}

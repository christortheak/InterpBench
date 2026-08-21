import Foundation
import Testing
@testable import ExperimentKit

/// The cross-engine freeze-gate vocabulary, held as a duplicated literal on
/// purpose — the `config.json` closed-key idiom
/// (`docs/WP0-AGENT-SURFACE-AUDIT.md` §3.1). Two independent literals with a
/// naming cross-reference is deliberately worse engineering than a shared
/// schema file and deliberately better parity enforcement: neither engine
/// can quietly follow the other.
@Suite struct FreezeGateVocabularyTests {

    /// THE cross-engine gate vocabulary. This literal is duplicated on
    /// purpose: adding, removing, or renaming a gate id must fail THIS test
    /// until the Python twin
    /// (`Server/tests/test_force_freeze.py::test_gate_id_vocabulary_is_closed`,
    /// asserting against `experiment_store.FORCED_GATE_IDS`) is updated in
    /// the same change. Order is part of the contract — it is the order
    /// `forcedGatesSkipped` is stamped in on both engines, and the order
    /// `FreezeRefusal.gates` reports failures in.
    @Test func matchesServerLiteral() {
        let contractIDs = [
            "revision", "validateEvidence", "batteryEvidence", "judgeValidity",
            "variantValidity", "gitClean", "measurementPins",
        ]
        #expect(FreezeGate.vocabulary == contractIDs)
        #expect(FreezeGate.allCases.map(\.rawValue) == contractIDs)
        // Round-trips: every wire id parses back to its case, so a stamp read
        // off an old manifest is never silently dropped.
        #expect(contractIDs.compactMap(FreezeGate.init(rawValue:)).count == contractIDs.count)
    }

    /// `FreezeRefusal` guarantees its reported gate is a member of the
    /// failure list even when a caller builds one by hand — the invariant
    /// the envelope's `error.gate`/`error.gates` pair relies on.
    @Test func aRefusalAlwaysContainsItsOwnGate() {
        let refusal = FreezeRefusal(
            gate: .gitClean, gates: [.revision], reason: "r", repairAction: "fix")
        #expect(refusal.gates.contains(.gitClean))
        #expect(refusal.gateIDs.allSatisfy(FreezeGate.vocabulary.contains))
    }
}

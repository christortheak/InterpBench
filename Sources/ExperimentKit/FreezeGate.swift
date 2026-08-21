import Foundation

/// The CLOSED cross-engine vocabulary of freeze gates.
///
/// Promoted here (WP0 step 2) from the literal array that lived inside
/// `ExperimentStore.freeze`'s `--force` branch, where it existed only to
/// order the `forcedGatesSkipped` stamp. The stamp was the ONLY place a gate
/// id ever survived: the refusal path computed no id at all, so an agent
/// could not tell "no validate evidence" from "dirty git tree" without
/// parsing prose (`docs/WP0-AGENT-SURFACE-AUDIT.md` §2.4, "the freeze
/// vocabulary is stamp-only"). The same seven ids now name a refusal.
///
/// The literal is duplicated on purpose across engines — the `config.json`
/// closed-key idiom (§3.1). The Python twin is
/// `experiment_store.FORCED_GATE_IDS`; adding, removing, or renaming an id
/// must fail `FreezeGateVocabularyTests.matchesServerLiteral` and
/// `Server/tests/test_force_freeze.py::test_gate_id_vocabulary_is_closed`
/// until both literals move in the same change.
///
/// `measurementPins` is emitted on BOTH engines despite the server's stale
/// doc comment: Swift stamps it for an unloadable study dtype, the server
/// for that and for an unqualified J-lens readout. Measurement-pin DRIFT
/// stays an unskippable `verify()` violation on both.
public enum FreezeGate: String, CaseIterable, Sendable, Codable {
    case revision
    case validateEvidence
    case batteryEvidence
    case judgeValidity
    case variantValidity
    case gitClean
    case measurementPins

    /// The vocabulary as wire strings, in the fixed cross-engine order —
    /// the order `forcedGatesSkipped` is stamped in and `FreezeRefusal.gates`
    /// is reported in. Python twin: `FORCED_GATE_IDS`.
    public static let vocabulary: [String] = allCases.map(\.rawValue)
}

/// A typed freeze-gate refusal: which gate declined, every gate that would
/// have declined, why, and the concrete repair.
///
/// Transported by `ExperimentError` (`ExperimentError.freezeRefusal`) rather
/// than thrown as its own error type, deliberately: `freeze` refusals are
/// caught as `ExperimentError` in ~40 test sites, the app's freeze panel, and
/// the web server, and WP0 step 2's contract is that freeze stays
/// byte-stable for humans — same prose, same exit code, same stamps. The
/// structured fields are strictly additive, which is also exactly what the
/// Python half does (`ExperimentStoreError` gains `gate`/`gates` while
/// `str(exc)` is unchanged).
///
/// `gate` is the gate whose message is in `reason` — i.e. the first failure
/// in freeze's HISTORICAL refusal order, so `gate` and `reason` can never
/// describe different gates. `gates` is every failure in the closed
/// vocabulary's order, matching the `forcedGatesSkipped` stamp. Those two
/// orders are NOT the same permutation (refusal order is
/// revision → measurementPins → validateEvidence → variantValidity →
/// batteryEvidence → judgeValidity → gitClean), so `gate` is not necessarily
/// `gates.first`; it is always a member of `gates`. Naming that divergence
/// beats papering over it (§2.4, "divergences the gate-id work must not
/// paper over").
public struct FreezeRefusal: Sendable, Equatable {
    /// The gate that declined — the one `reason` describes.
    public let gate: FreezeGate
    /// Every gate that failed, in `FreezeGate.vocabulary` order. Always
    /// contains `gate`.
    public let gates: [FreezeGate]
    /// The refusal prose, byte-identical to what `freeze` has always thrown.
    public let reason: String
    /// The concrete command or edit that repairs this gate — the remedy
    /// clause the gate's own refusals already print.
    public let repairAction: String

    public init(
        gate: FreezeGate, gates: [FreezeGate], reason: String, repairAction: String
    ) {
        self.gate = gate
        self.gates = gates.contains(gate) ? gates : ([gate] + gates)
        self.reason = reason
        self.repairAction = repairAction
    }

    /// Wire form of `gate` — what an envelope's `error.gate` carries.
    public var gateID: String { gate.rawValue }
    /// Wire form of `gates` — what an envelope's `error.gates` carries.
    public var gateIDs: [String] { gates.map(\.rawValue) }
}

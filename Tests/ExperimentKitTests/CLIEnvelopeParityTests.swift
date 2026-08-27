import Foundation
import Testing

@testable import ExperimentKit

/// The Swift half of the cross-engine envelope contract
/// (`docs/WP0-AGENT-SURFACE-AUDIT.md` §7 step 8, §3.1).
///
/// Every literal below is COPIED FROM THE SERVER and asserted against this
/// engine's constant. The server half does the mirror image — its literals are
/// copied from these Swift files and asserted against
/// `steerlab_server/cli_envelope.py` — in
/// `Server/tests/test_cli_envelope.py`.
///
/// Two independent literals with a naming cross-reference is deliberately
/// worse engineering than a shared schema file and deliberately better parity
/// enforcement: adding, removing, or renaming a header key, a state, an exit
/// code, a gate id, or an advisory code fails a test on BOTH engines until both
/// literals move in the same change. Neither engine can quietly follow the
/// other.
///
/// `FreezeGateVocabularyTests.matchesServerLiteral` already pins the seven
/// freeze gates in exactly this idiom; this suite extends the pattern to the
/// rest of the contract.
@Suite struct CLIEnvelopeParityTests {

    // MARK: - The envelope header

    /// Copied from `cli_envelope.CONTRACT_HEADER_KEYS`
    /// (`Server/steerlab_server/cli_envelope.py`). Server twin test:
    /// `test_cli_envelope.py::test_contract_header_keys_match_the_swift_literal`.
    @Test func headerKeysMatchServerLiteral() {
        let serverLiteral = [
            "changed", "engine", "message", "observedAt", "schemaVersion",
            "state", "verb",
        ]
        #expect(SteerLabCLIEnvelope.contractHeaderKeys == serverLiteral)
    }

    /// Copied from `cli_envelope.CONTRACT_OPTIONAL_KEYS`.
    @Test func optionalKeysMatchServerLiteral() {
        let serverLiteral = [
            "advisories", "error", "nextAction", "result", "workspace",
        ]
        #expect(SteerLabCLIEnvelope.contractOptionalKeys == serverLiteral)
    }

    /// Copied from `cli_envelope.SCHEMA_VERSION` and `cli_envelope.ENGINE`.
    /// The engine stamp must equal the SUBSTRATE constant the artifacts are
    /// scoped by, or a run's substrate and its CLI's substrate could disagree.
    @Test func schemaVersionAndEngineStampMatchServerLiterals() {
        #expect(SteerLabCLIEnvelope.schemaVersion == 1)
        #expect(SteerLabCLIEnvelope.serverEngine == "python-hf-transformers")
        #expect(SteerLabCLIEnvelope.serverEngine == WorkspaceScoping.serverSubstrate)
    }

    // MARK: - The state vocabulary

    /// Copied from `cli_envelope.STATE_EXIT_CODES`, in order.
    ///
    /// Order is part of the contract (it is this enum's declaration order on
    /// both engines), and so is every number: the whole point of the package
    /// is that a freeze refusal (65), a missing experiment (66), and an
    /// operational failure (70) stop being the same 1.
    @Test func stateVocabularyAndExitCodesMatchServerLiteral() {
        let serverLiteral: [(String, Int32)] = [
            ("ready", 0),
            ("planned", 0),
            ("running", 0),
            ("okWithAdvisories", 0),
            ("needsHumanAuthentication", 10),
            ("needsApproval", 11),
            ("pending", 12),
            ("degraded", 13),
            ("blocked", 64),
            ("refused", 65),
            ("notFound", 66),
            ("failed", 70),
        ]
        let local = SteerLabCLIState.allCases.map { ($0.rawValue, $0.exitCode) }
        #expect(local.map(\.0) == serverLiteral.map(\.0))
        #expect(local.map(\.1) == serverLiteral.map(\.1))
    }

    // MARK: - The two closed gate vocabularies

    /// Copied from `lifecycle_gates.LIFECYCLE_GATE_IDS`
    /// (`Server/steerlab_server/experiment/lifecycle_gates.py`). Server twin
    /// test: `test_lifecycle_gate_vocabulary_matches_the_swift_literal`.
    @Test func lifecycleGatesMatchServerLiteral() {
        let serverLiteral = [
            "statusImmutable", "pinDrift", "manifestEpoch", "promotionEpoch",
            "promotionEvidence", "artifactPin", "sweepInputDrift",
            "sweepSelectionRule", "sweepJudgeCapacity", "dataReadiness",
            "samplingPolicy", "thinkingModeConflict", "inertConditions",
            "responseFormat", "confirmationPool", "confirmationAgentShape",
            "parityThreshold", "missingPrerequisite", "armsCleared",
            "conceptInUse",
        ]
        #expect(LifecycleGate.vocabulary == serverLiteral)
        // Round-trips: every wire id parses back to its case, so a gate id read
        // off a server document is never silently dropped.
        #expect(
            serverLiteral.compactMap(LifecycleGate.init(rawValue:)).count
                == serverLiteral.count)
    }

    /// The two vocabularies must stay disjoint on both engines: an id in both
    /// would make `error.gate` ambiguous about which `switch` an agent should
    /// use, and the two have different skippability classes (`--force` skips
    /// freeze gates; nothing skips lifecycle gates).
    @Test func theTwoGateVocabulariesAreDisjointOnBothEngines() {
        #expect(!LifecycleGate.collidesWithFreezeVocabulary)
    }

    // MARK: - The advisory vocabulary

    /// Copied from `cli_envelope.ADVISORY_CODES`, in order.
    @Test func advisoryCodesMatchServerLiteral() {
        let serverLiteral = [
            "freezeGateSkipped", "vacuousValidation", "probeAtChanceFloor",
            "judgePanelTooSmall", "emptyAnalysis", "allEffectSizesZero",
            "sweepRecommendationsOnly", "sweepSelectionDefaulted",
            "choiceItemsWithoutInstrument", "revisionAdoption",
            "revisionAdoptionWarning", "siteQualifyWarning",
            "deprecatedImplicitSelection",
        ]
        #expect(CLIAdvisory.vocabulary == serverLiteral)
    }

    // MARK: - The cross-engine message strings

    /// The deferred twinned message from c86ce53, which step 8 gave an id on
    /// both engines. Copied from `exclusions.PIN_REQUIRED_MESSAGE` and
    /// `exclusions.PIN_REQUIRED_REPAIR`.
    ///
    /// The repair names `steerlab-cli` on BOTH engines because pinning is
    /// Mac-authority (§3.2) — a repair naming a verb the answering engine does
    /// not have would send an agent in a circle, which is what gate-5 dry run
    /// #1 measured.
    @Test func exclusionPinMessageAndRepairMatchServerLiterals() {
        #expect(
            ExclusionEngine.pinRequiredMessage
                == "exclusion rule failedAttentionCheck needs the task prompts "
                + "pinned (taskPromptsFile + taskPromptsHash) so analysis "
                + "grades the same items the run saw — pin the prompt set first")
        #expect(
            ExclusionEngine.pinRequiredRepair
                == "steerlab-cli experiment pin-prompts <name> <the prompt file "
                + "the run used> — analysis grades the items the run saw, so "
                + "the pin must name that exact file")
    }

    /// Copied from `sweep_selection.defaulted_selection_advisory`. Punch list
    /// #1, P3: the document's most emphatic rule — never select on
    /// markerDensity for a decision study — was silently violated with no
    /// advisory at all, on either engine.
    @Test func defaultedSelectionAdvisoryMatchesServerLiteral() {
        let serverLiteral =
            "no sweep.selection is declared, so the winning cell will be "
            + "chosen by markerDensity — a SURFACE-PROSE diagnostic — while "
            + "3 of 8 pinned item(s) carry options/target and could be scored "
            + "deterministically. Declare the criterion: steerlab-cli "
            + "experiment set-sweep-selection <name> --objective logprobShift "
            + "--choice-prompts <file>  (or --objective judgeScore with a "
            + "pinned rubric). Marker density is a manipulation check, not a "
            + "decision objective."
        #expect(
            SweepSelectionRule.defaultedSelectionAdvisory(
                spec: nil, choiceItemCount: 3, totalItemCount: 8)
                == serverLiteral)
        // Declared, or no choice-shaped items: no advisory on either engine.
        #expect(
            SweepSelectionRule.defaultedSelectionAdvisory(
                spec: nil, choiceItemCount: 0, totalItemCount: 8) == nil)
    }

    /// Copied from `experiment_store.concept_in_use_repair` /
    /// `.concept_not_pinned_repair`. `detach`'s two typed refusals are the
    /// same rules on both engines, so the repairs an agent follows verbatim
    /// must be the same bytes — and both name `steerlab-cli`, because
    /// authoring is Mac-authority (§3.2) whichever engine answered.
    /// Server twin test: `test_detach_repairs_match_the_swift_literals`.
    @Test func detachRepairsMatchServerLiterals() {
        #expect(
            ExperimentStore.conceptInUseRepair("demo")
                == "remove or re-declare those conditions first: steerlab-cli "
                + "experiment declare-condition demo <condition> … (re-declare "
                + "onto a concept that stays), then steerlab-cli experiment "
                + "detach demo <concept>…")
        #expect(
            ExperimentStore.conceptNotPinnedRepair("demo")
                == "steerlab-cli experiment list  (result.experiments[]"
                + ".concepts names what 'demo' pins), then steerlab-cli "
                + "experiment detach demo <one of those>")
    }

    // MARK: - The document itself

    /// A document this engine writes must satisfy the rules the server's
    /// reader assumes: closed key set, error present exactly when the state is
    /// not a success, one trailing newline, sorted keys.
    @Test func aLocalDocumentSatisfiesTheSharedRules() throws {
        let refusal = SteerLabCLIEnvelope.refusal(
            verb: "experiment verify", engine: SteerLabCLIEnvelope.localEngine,
            code: LifecycleGate.pinDrift.rawValue,
            gate: LifecycleGate.pinDrift.rawValue,
            reason: "1 pinned input(s) no longer match their hashes",
            repairAction: "steerlab-cli experiment verify demo",
            observedAt: Date(timeIntervalSince1970: 1_000))
        let text = try refusal.jsonText()
        #expect(text.hasSuffix("}\n"))
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(text.utf8))
                as? [String: Any])
        let allowed = Set(
            SteerLabCLIEnvelope.contractHeaderKeys
                + SteerLabCLIEnvelope.contractOptionalKeys)
        for key in object.keys { #expect(allowed.contains(key)) }
        for key in SteerLabCLIEnvelope.contractHeaderKeys {
            #expect(object[key] != nil)
        }
        #expect(refusal.exitCode == 65)
        // The same instant the server's goldens pin, so a reader comparing a
        // Swift golden with its server twin sees one timestamp.
        #expect(object["observedAt"] as? String == "1970-01-01T00:16:40Z")
    }
}

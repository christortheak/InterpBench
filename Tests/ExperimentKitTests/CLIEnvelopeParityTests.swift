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
            "conceptInUse", "sweepGridRule",
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
            "deprecatedImplicitSelection", "systemPromptNotApplied",
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

    /// Copied from `experiment_store.sweep_grid_repair` /
    /// `.absolute_layers_need_depth_repair` /
    /// `.absolute_layers_out_of_range_repair` /
    /// `.sweep_selection_owns_repair`. `set-sweep-grid`'s refusals are the same
    /// rules on both engines, so the repairs an agent follows verbatim must be
    /// the same bytes — and all of them name `steerlab-cli`, because a grid is
    /// authoring and authoring is Mac-authority (§3.2) whichever engine
    /// answered. Server twin test:
    /// `test_sweep_grid_repairs_match_the_swift_literals`.
    @Test func sweepGridRepairsMatchServerLiterals() {
        #expect(
            ExperimentStore.sweepGridRepair("demo")
                == "steerlab-cli experiment set-sweep-grid demo "
                + "--layer-fractions 0.5,0.7,0.85 --alphas 0.05,0.08,0.1,0.13  "
                + "(both axes ascend, each value once; alphas are "
                + "residual-norm units above 0)")
        #expect(
            ExperimentStore.absoluteLayersNeedDepthRepair("demo")
                == "steerlab-cli experiment extract demo  (any vector for the "
                + "pinned model states its depth) && steerlab-cli experiment "
                + "set-sweep-grid demo --layers <L>,…  ; or declare the grid "
                + "in depth fractions, which need no model: steerlab-cli "
                + "experiment set-sweep-grid demo --layer-fractions 0.5,0.7,0.85")
        #expect(
            ExperimentStore.absoluteLayersOutOfRangeRepair("demo", depth: 34)
                == "steerlab-cli experiment set-sweep-grid demo "
                + "--layers <0…33>,…  ; or declare depths instead, which "
                + "survive a change of model: steerlab-cli experiment "
                + "set-sweep-grid demo --layer-fractions 0.5,0.7,0.85")
        #expect(
            ExperimentStore.sweepSelectionOwnsRepair("demo", flag: "--objective")
                == "steerlab-cli experiment set-sweep-selection demo "
                + "--objective <value>  (the selection RULE is that verb's; "
                + "set-sweep-grid writes the layer × alpha grid the rule then "
                + "picks a winner from)")
    }

    /// The grid audit itself is a twin: the refusal PROSE is what an agent
    /// reads, so the two engines must not paraphrase each other. Server twin
    /// test: `test_sweep_grid_problem_matches_the_swift_literals`.
    @Test func sweepGridProblemsMatchServerLiterals() {
        func problem(
            _ fractions: [Double], _ alphas: [Double], maxTokens: Int = 80,
            dev: String = "d", battery: String = "b"
        ) -> String? {
            ExperimentStore.sweepGridProblem(
                layerFractions: fractions, alphas: alphas, devPromptsFile: dev,
                batteryFile: battery, maxTokens: maxTokens)
        }
        #expect(problem([0.5, 0.7], [0.05, 0.08]) == nil)
        #expect(
            problem([], [0.05])
                == "the layer axis is empty — a grid names at least one depth")
        #expect(
            problem([1.5], [0.05])
                == "layer fractions are depths in [0, 1] — got 1.5")
        #expect(
            problem([0.7, 0.5], [0.05])
                == "the layer axis does not ascend at 0.5 — declare depths in "
                + "increasing order, each one once (the sweep sorts and "
                + "deduplicates them, so an unordered declaration names a grid "
                + "it will not run)")
        #expect(
            problem([0.5], [])
                == "the alpha axis is empty — a grid names at least one dose")
        #expect(
            problem([0.5], [0])
                == "alphas are residual-norm units above 0 — got 0 (0 is the "
                + "baseline cell, which every sweep runs anyway)")
        #expect(
            problem([0.5], [0.1, 0.05])
                == "the alpha ladder does not ascend at 0.05 — declare doses "
                + "in increasing order, each one once (a ladder that doubles "
                + "back is not a dose-response)")
        #expect(
            problem([0.5], [0.05], maxTokens: 0)
                == "max tokens must be above 0 — got 0")
        #expect(
            problem([0.5], [0.05], dev: " ")
                == "the dev-prompts file is required — the sweep generates on it")
        #expect(
            problem([0.5], [0.05], battery: "")
                == "the capability-battery file is required — the sweep scores "
                + "every cell on it")
    }

    /// The midpoint conversion is the one piece of ARITHMETIC the two engines
    /// share on this path, and a fraction that resolves to a different block
    /// on one of them is a different study. Exercised over every layer of
    /// several plausible depths rather than a sample: the round trip is
    /// claimed to be exact, and "exact" is cheap to check.
    @Test func absoluteLayersSurviveTheDepthFractionRoundTrip() {
        for depth in [1, 2, 12, 18, 26, 28, 32, 34, 42, 48, 62, 64, 80, 126] {
            for layer in 0 ..< depth {
                let fraction = ExperimentStore.depthFraction(
                    forLayer: layer, depth: depth)
                #expect(
                    ExperimentManifest.SweepSpec(layerFractions: [fraction])
                        .resolvedLayers(layerCount: depth) == [layer],
                    "layer \(layer) of \(depth) did not survive the round trip")
            }
        }
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

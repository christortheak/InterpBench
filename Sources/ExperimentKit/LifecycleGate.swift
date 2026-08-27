import Foundation

// =============================================================================
// The SECOND closed refusal vocabulary (WP0-AGENT-SURFACE-AUDIT §2.4, §7 step
// 7) — everything on the agent path that declines a well-formed request and is
// NOT a freeze gate.
//
// Step 2 gave the seven freeze gates ids on the refusal path. Gate-5 dry run #1
// (§9) then found the systemic gap: `freezeGateFailed` was the ONLY
// machine-actionable refusal class. Four repairable refusals — attaching to a
// frozen manifest, validating after a validation.jsonl appeared post-attach,
// promoting with no sweep, running a frozen study with no pinned prompts —
// arrived as `failed/70/verbFailed` with a boilerplate repair ("read the reason
// and repair the named input"). An agent cannot act on that: it cannot tell a
// gate that declined from a crash, and the repair names no command.
//
// This vocabulary is deliberately SEPARATE from `FreezeGate`. Merging them
// would let an agent's `switch` over freeze gates silently absorb an epoch
// refusal, and the two have different skippability classes: `--force` skips
// freeze gates, and nothing skips these.
//
// Both vocabularies land in the envelope's `error`:
//
//   freeze:    { "code": "freezeGateFailed", "gate": "validateEvidence" }
//   lifecycle: { "code": "pinDrift",         "gate": "pinDrift"         }
//
// — i.e. a lifecycle refusal's `code` IS its gate id, because there is exactly
// one failure class per gate here, where freeze has one class (`the gate
// declined`) spanning seven gates.
//
// Server parity is step 8. The literal is duplicated per engine on purpose (the
// `config.json` closed-key idiom, §3.1): the Python twin will be
// `LIFECYCLE_GATE_IDS`, and `RefusalSiteRegistryTests` /
// `test_lifecycle_gates.py` must both move before an id may be added, removed,
// or renamed.
// =============================================================================

/// The closed vocabulary of NON-freeze refusals on the agent path.
///
/// Named per audit §2.4's proposed contract ("`LifecycleGate` — new, for
/// everything else on the agent path"), with one addition it did not name:
/// `missingPrerequisite`, for "the thing this verb needs was never declared"
/// (a frozen study with no pinned task prompts; an `evaluate` with no completed
/// run). That class is neither drift nor an epoch mismatch, and folding it into
/// either would tell an agent to repair a hash that does not exist.
public enum LifecycleGate: String, CaseIterable, Sendable, Codable {

    /// A frozen or complete manifest was asked to change. Immutability is the
    /// firewall: iterate by duplicating.
    case statusImmutable

    /// A pinned input no longer matches its pinned hash — or appeared after
    /// being pinned as absent. Every `verify()` violation, plus the task-prompt
    /// hash checks the run loop repeats at run time.
    case pinDrift

    /// A source run's stamped experiment hash ≠ the live manifest's
    /// (`analyze`, `evaluate`, `rescore-style`).
    case manifestEpoch

    /// The same guard on the promotion path, where the consequence differs: an
    /// agent minted from another epoch's sweep carries a birth certificate
    /// naming settings selected under a different study.
    case promotionEpoch

    /// Promotion has no sweep evidence to stand on — no recommendation, no
    /// readable sweep run, or a `--cell` override with no sweep at all
    /// ("hand-creation wearing a promotion badge").
    case promotionEvidence

    /// A pinned vector artifact's bytes are missing, unreadable, or hash
    /// differently than the pin claims.
    case artifactPin

    /// A sweep input (dev prompts, capability battery, choice prompts) drifted
    /// from the hash the manifest pinned it at.
    case sweepInputDrift

    /// The declared `sweep.selection` block cannot resolve: unknown metric,
    /// out-of-range constraint, missing or unreadable instrument file.
    case sweepSelectionRule

    /// The declared judge panel cannot be run on this engine — a second
    /// resident model where only one is possible, or a missing credential.
    case sweepJudgeCapacity

    /// `data check` found blocking data requirements.
    case dataReadiness

    /// The local greedy-only sampling policy: `temperature > 0`, or more than
    /// one seed, on the MLX substrate that pins no per-run seed.
    case samplingPolicy

    /// A logprob/ordinal arm declared with thinking mode on — the answer would
    /// be a marginal over reasoning paths.
    case thinkingModeConflict

    /// The declared study would run no measured condition: conditions inerted
    /// by the declared `studyType`, or a concept study with no injection arm.
    case inertConditions

    /// A declared outcome instrument cannot be honoured, and would therefore
    /// measure nothing: it is outside the closed instrument vocabulary, or it
    /// is option-consuming and cannot read the items it is pointed at (or a
    /// pinned applicability scope no longer selects them).
    case responseFormat

    /// A confirmation study's item pool overlaps the pool it must be disjoint
    /// from.
    case confirmationPool

    /// The agent a confirmation policy anchors on has the wrong shape —
    /// adapter-bearing, no injections, several injections, or ablating.
    case confirmationAgentShape

    /// `vectors compare` ran and the minimum cosine fell below the threshold.
    case parityThreshold

    /// The verb needs something that is not there: something the study never
    /// declared (pinned task prompts on a frozen study, a completed run to
    /// analyze, a concept to perturb), or a workspace file the caller NAMED
    /// that is not on disk (a pin verb's rubric, prompt set, taxonomy, or
    /// stimulus files). Both are "author the input, then re-run" — as against
    /// `pinDrift`, which repairs a hash that does exist.
    case missingPrerequisite

    /// A save would take a DRAFT manifest that holds concepts and/or
    /// conditions to BOTH-empty — the whole measured surface gone in one
    /// write. Open-issues §8: the shape is indistinguishable from a stale
    /// in-memory copy or a skeleton document landing on a populated one, and
    /// the loss is silent, because nothing downstream reads a manifest it did
    /// not just write. A deliberate clear says so (`mayClearArms` /
    /// `clearing_arms`).
    case armsCleared

    /// A concept pin cannot be removed because the manifest still DECLARES
    /// something that reads it by name: an injection condition's slot, a
    /// per-concept sweep-selection instrument, a variant condition's
    /// `fromPromotion` forward reference, or a confirmation perturbation
    /// policy. The same class `armsCleared` belongs to — a write that would
    /// silently take a declaration away from the measured surface — narrowed
    /// to the one direction `detach` can travel in. Detaching anyway would
    /// leave a dangling reference that only the next `verify` names, and the
    /// run in between would have measured a study nobody declared.
    case conceptInUse

    /// The declared sweep GRID cannot be run: an empty axis, a depth fraction
    /// outside [0, 1], an alpha at or below zero (0 is the implied baseline
    /// cell, not a rung), a ladder that does not ascend, or an absolute layer
    /// outside the pinned model's depth. `sweepSelectionRule`'s sibling, and
    /// deliberately not the same id: that gate says the RULE for picking a
    /// winner cannot resolve, this one says there are no honest cells to pick
    /// from, and the two repairs are different verbs.
    case sweepGridRule

    /// The vocabulary as wire strings, in the fixed cross-engine order. Python
    /// twin (step 8): `LIFECYCLE_GATE_IDS`.
    public static let vocabulary: [String] = allCases.map(\.rawValue)

    /// What the process exits with in HUMAN mode when this gate declines.
    ///
    /// Almost always 1 — deliberately. The audit schedules exactly one
    /// human-mode migration at this step (§7 row 7: "`data check` blockers
    /// (2 → 65)"), because `data check` is the only verb in the family whose
    /// human code was ever anything other than 1, and leaving it at 2 while
    /// the same refusal answers 65 in JSON mode is the migration debt §2.3
    /// says to pay once rather than alias. Flipping the OTHER refusals from 1
    /// to 65 would break `set -e` wrappers on a change the row does not
    /// discuss; in JSON mode every gate here is already 65.
    public var humanExitCode: Int32 {
        switch self {
        case .dataReadiness: SteerLabCLIState.refused.exitCode
        default: 1
        }
    }

    /// The two vocabularies must stay disjoint: an id in both would make
    /// `error.gate` ambiguous about which `switch` an agent should use.
    /// Asserted by `RefusalSiteRegistryTests`.
    public static var collidesWithFreezeVocabulary: Bool {
        !Set(vocabulary).isDisjoint(with: Set(FreezeGate.vocabulary))
    }
}

/// A typed lifecycle refusal: which gate declined, why, and the EXECUTABLE
/// repair.
///
/// Transported by `ExperimentError` (`ExperimentError.lifecycleRefusal`) rather
/// than thrown as its own type — the same decision `FreezeRefusal` made and for
/// the same reason: these refusals are caught as `ExperimentError` in hundreds
/// of places, and step 7's contract is that HUMAN output stays byte-stable.
/// `reason` is byte-identical to the prose the site has always thrown; the
/// structured fields are strictly additive.
///
/// **`repairAction` is a command, not advice.** Dry run #1 proved agents follow
/// a repair verbatim, which is also how it proved the `validateEvidence`
/// repair's own steps fail as given (§9, P5: it omitted the re-attach that
/// re-pins `validationHash` after the file is authored). A repair that is not
/// runnable is worse than none — it sends an agent in a circle.
public struct LifecycleRefusal: Sendable, Equatable {

    /// The gate that declined.
    public let gate: LifecycleGate
    /// The refusal prose, byte-identical to what the site has always thrown.
    public let reason: String
    /// The concrete command(s) that repair it. Several steps are joined with
    /// `" && "` when they must run in order, or `" ; then "` when a human edit
    /// sits between them.
    public let repairAction: String

    public init(gate: LifecycleGate, reason: String, repairAction: String) {
        self.gate = gate
        self.reason = reason
        self.repairAction = repairAction
    }

    /// Wire form — what an envelope's `error.code` and `error.gate` carry.
    public var gateID: String { gate.rawValue }
}

extension ExperimentError {

    /// Build a refusal that carries its gate id and its repair. The thrown
    /// `reason` is unchanged, so every existing catch site, printed line, and
    /// human exit code is untouched.
    public static func refusing(
        _ gate: LifecycleGate, _ reason: String, repair: String
    ) -> ExperimentError {
        ExperimentError(
            refusal: LifecycleRefusal(
                gate: gate, reason: reason, repairAction: repair))
    }
}

// MARK: - The registry

/// The inventory of every agent-path refusal site, as DATA.
///
/// Its purpose is exhaustiveness: `RefusalSiteRegistryTests
/// .everyAgentPathRefusalCarriesACode` walks it and asserts that every gate in
/// the closed vocabulary is claimed by a site, that every site names verbs the
/// CLI actually declares, and that every repair is a runnable command. Without
/// a registry the vocabulary would be a list of hopes: a new refusal could land
/// with no id and nothing would notice — which is exactly the state §2.4
/// measured ("the number of sites carrying a machine-readable id is zero on
/// both engines").
public enum RefusalSiteRegistry {

    /// How a gate reaches a caller on THIS engine.
    public enum Surfacing: String, Sendable {
        /// A throw site carries the gate id directly.
        case typedRefusal
        /// It surfaces inside `verify()`'s violation list, so the caller sees
        /// `pinDrift` with the specific rule named in `result.violations`.
        /// Recorded rather than hidden: audit §2.4's divergence list warns
        /// against a shared id implying a parity that does not exist, and the
        /// server refuses two of these separately.
        case verifyViolation
    }

    public struct Site: Sendable {
        public let gate: LifecycleGate
        /// The verbs that can hit it, as `ExperimentCLIParser` labels them.
        public let verbs: [String]
        /// Where it lives, for a reader who has to find it.
        public let origin: String
        /// A runnable repair template. `<name>` / `<concept>` are the caller's
        /// own values; the per-site refusal substitutes them.
        public let repairAction: String
        public let surfacing: Surfacing

        public init(
            gate: LifecycleGate, verbs: [String], origin: String,
            repairAction: String, surfacing: Surfacing = .typedRefusal
        ) {
            self.gate = gate
            self.verbs = verbs
            self.origin = origin
            self.repairAction = repairAction
            self.surfacing = surfacing
        }
    }

    public static let sites: [Site] = [
        .init(
            gate: .statusImmutable,
            verbs: [
                "experiment attach", "experiment detach",
                "experiment pin-prompts",
                "experiment pin-rubric", "experiment declare-condition",
                "experiment set-sweep-selection", "experiment set-sweep-grid",
                "experiment set-instruments",
                "experiment set-style-taxonomy", "experiment confirm",
                // `panel compile` writes a FILE before it writes the manifest,
                // so it checks the status itself and refuses before compiling
                // rather than letting `save` refuse afterwards and leave an
                // orphan casting in prompts/panels/compiled/ (open-issues §18).
                "panel compile",
            ],
            origin: "ExperimentStore.save / ExperimentStore.updateDraft; "
                + "ExperimentCLIRunner.compilePanel (pre-compile status check)",
            repairAction: "steerlab-cli experiment duplicate <name> <name>-v2 "
                + "&& steerlab-cli experiment <verb> <name>-v2 …"),
        .init(
            gate: .pinDrift,
            verbs: [
                "experiment verify", "experiment extract", "experiment validate",
                "experiment sweep", "experiment run", "experiment evaluate",
            ],
            origin: "ExperimentStore.verify → ExperimentTasks.loadVerified; "
                + "ExperimentTasks.loadTaskPrompts",
            repairAction: "steerlab-cli experiment verify <name> "
                + "(names every drifted pin) ; then restore the named files, or "
                + "steerlab-cli experiment duplicate <name> <name>-v2 and re-pin"),
        .init(
            gate: .manifestEpoch,
            verbs: [
                "experiment analyze", "experiment evaluate",
                "experiment rescore-style",
            ],
            origin: "RunEpoch.refusal → ExperimentTasks.analyze/evaluate/rescoreStyle",
            repairAction: "steerlab-cli experiment run <name> "
                + "(a run of the CURRENT manifest), or re-read the older run "
                + "with steerlab-cli experiment analyze <name> "
                + "--allow-unverified-epoch"),
        .init(
            gate: .promotionEpoch,
            verbs: ["experiment promote"],
            origin: "AgentPromotion.promote (RunEpoch.refusal)",
            repairAction: "steerlab-cli experiment sweep <name> "
                + "&& steerlab-cli experiment promote <name> <concept>"),
        .init(
            gate: .promotionEvidence,
            verbs: ["experiment promote"],
            origin: "AgentPromotion.promote — no recommendation / no sweep run",
            repairAction: "steerlab-cli experiment sweep <name> "
                + "&& steerlab-cli experiment promote <name> <concept>"),
        .init(
            gate: .artifactPin,
            verbs: ["experiment promote"],
            origin: "AgentPromotion.promote — vector-artifact byte pins",
            repairAction: "steerlab-cli experiment extract <name> "
                + "&& steerlab-cli experiment promote <name> <concept>"),
        .init(
            gate: .sweepInputDrift,
            verbs: ["experiment sweep"],
            origin: "ExperimentTasks.sweep — sweepInputPinViolation",
            repairAction: "restore the named sweep input to its pinned bytes ; "
                + "then steerlab-cli experiment sweep <name>  (a frozen pin is "
                + "never re-pinned: steerlab-cli experiment duplicate <name> "
                + "<name>-v2 to change it)"),
        .init(
            gate: .sweepSelectionRule,
            verbs: ["experiment sweep"],
            origin: "SweepSelectionRule.resolve / .resolveObjective",
            repairAction: "steerlab-cli experiment set-sweep-selection <name> "
                + "--objective markerDensity|judgeScore|logprobShift …"),
        .init(
            gate: .sweepJudgeCapacity,
            verbs: ["experiment sweep"],
            origin: "SweepSelectionRule.resolveObjective / sweepJudgePanel",
            repairAction: "steerlab-cli experiment pin-rubric <name> "
                + "<prompts/rubrics/file.md> --judges <name>:local "
                + "(a local judge with no model resolves to the study model — "
                + "the only judge a local sweep can run)"),
        .init(
            gate: .dataReadiness,
            verbs: ["data check"],
            origin: "StudyDataReadiness.summary → ExperimentCLIRunner data check",
            repairAction: "author the files named by result.blockers[].path, "
                + "then steerlab-cli data check <name>"),
        .init(
            gate: .samplingPolicy,
            verbs: ["experiment run"],
            origin: "ExperimentTasks.requireGreedyLocalDesign",
            repairAction: "set the study temperature to 0 and one seed for the "
                + "local engine, or submit the study to the Python server "
                + "(steerlab-cli remote submit-bundle …), which seeds per record"),
        .init(
            gate: .thinkingModeConflict,
            verbs: ["experiment verify", "experiment run"],
            origin: "ExperimentStore.verify — answer-token instrument with "
                + "qwenThinkingEnabled",
            repairAction: "steerlab-cli experiment set-instruments <name> "
                + "sampledText, or disable thinking mode on the study",
            surfacing: .verifyViolation),
        .init(
            gate: .inertConditions,
            verbs: ["experiment run"],
            origin: "ExperimentStore.inertConditionsProblem / "
                + "noMeasuredConditionsProblem",
            repairAction: "steerlab-cli experiment declare-condition <name> "
                + "<arm> --slots <concept>:<layer>:<alpha>"),
        .init(
            gate: .responseFormat,
            verbs: ["experiment run"],
            origin: "ExperimentTasks.checkResponseFormats "
                + "(ExperimentStore.unknownOutcomeInstrumentProblem; "
                + "ResponseFormat.refusal)",
            repairAction: "steerlab-cli experiment set-instruments <name> "
                + "sampledText, or re-author the items with "
                + "\"responseFormat\": \"label\" and re-pin with steerlab-cli "
                + "experiment pin-prompts <name> <file>"),
        .init(
            gate: .confirmationPool,
            verbs: ["experiment verify", "experiment confirm"],
            origin: "ExperimentStore.verify — confirm-phase pool disjointness",
            repairAction: "steerlab-cli experiment pin-prompts <name> "
                + "<a held-out item file disjoint from the screen's>",
            surfacing: .verifyViolation),
        .init(
            gate: .confirmationAgentShape,
            verbs: ["experiment confirm"],
            origin: "ConfirmationStudy.attach",
            repairAction: "steerlab-cli experiment promote <name> <concept> "
                + "(a single-injection, non-ablating agent is the only shape a "
                + "perturbation policy can anchor on)"),
        .init(
            gate: .parityThreshold,
            verbs: ["vectors compare"],
            origin: "ExperimentCLIRunner vectors compare",
            repairAction: "re-extract on the substrate that will run the study "
                + "(steerlab-cli experiment extract <name>), or pass "
                + "--threshold deliberately"),
        .init(
            gate: .missingPrerequisite,
            verbs: [
                "experiment run", "experiment evaluate", "experiment analyze",
                "experiment confirm",
                // The AUTHORING half (2026-08-18). Every verb whose argument is
                // a workspace FILE answered a path that is not on disk with an
                // untyped error — `pin-rubric` with a raw `NSCocoaErrorDomain`
                // dump, the other three with `failed`/70/`verbFailed`. A path
                // typo is the commonest authoring mistake there is, and it was
                // the one class of refusal an agent could not tell from a
                // crash.
                "experiment attach", "experiment pin-prompts",
                "experiment pin-rubric", "experiment set-style-taxonomy",
                "experiment extract", "experiment validate", "experiment sweep",
                // `detach` naming a concept the manifest does not pin. Same
                // class, one direction over: the thing the verb needs was
                // never declared, and the repair is to read what IS pinned.
                "experiment detach",
                // `set-sweep-grid --layers` on a model nothing has been
                // extracted for. ABSOLUTE layers are only meaning-bearing
                // against a depth, this workspace holds no artifact that
                // states one, and inventing a depth would silently name
                // different cells than the sweep will.
                "experiment set-sweep-grid",
                // The same class one family over: `--seat <id>=<path>` naming
                // an agent artifact that is not on disk, or a file that does
                // not decode as one (open-issues §18).
                "panel compile",
            ],
            origin: "ExperimentTasks.loadTaskPrompts (frozen, unpinned); "
                + "newestCompletedRun; ConfirmationStudy.attach; "
                + "JudgeRubricStore.resolveRubric (no rubric anywhere); "
                + "ExperimentStore.pinTaskPrompts / JudgeRubricStore.load / "
                + "ExperimentStore.pinReasoningStyleTaxonomy (a named input "
                + "file that is not on disk); ExperimentCLIRunner's "
                + "StimulusSetError.missingFile(s) classification",
            repairAction: "steerlab-cli experiment duplicate <name> <name>-v2 "
                + "&& steerlab-cli experiment pin-prompts <name>-v2 "
                + "prompts/…/file.jsonl && steerlab-cli experiment freeze <name>-v2"),
        .init(
            gate: .armsCleared,
            // Every verb that reaches `save` with a WHOLE document it did not
            // just read. The panel/WebServer authoring surface is the wide
            // one (open-issues §8's writer census), but it has no CLI label;
            // these are the declared verbs that can land the same shape.
            verbs: [
                "experiment attach", "experiment declare-condition",
                "panel compile", "experiment confirm",
            ],
            origin: "ExperimentStore.save (the arms guard) — reached by every "
                + "whole-document writer: ExperimentPanel's `selected` cache "
                + "(no file watcher; refreshed on selection), the WebServer "
                + "authoring routes that delegate to it, ConfirmationStudy"
                + ".attach, and the run paths that hold a manifest copy for "
                + "the length of a sweep",
            repairAction: "steerlab-cli experiment verify <name> ; then "
                + "steerlab-cli experiment attach <name> <concept>… && "
                + "steerlab-cli experiment declare-condition <name> …"),
        .init(
            gate: .conceptInUse,
            verbs: ["experiment detach"],
            origin: "ExperimentStore.detachConcepts — the concept-dependent "
                + "audit (conditions' slots, per-concept sweep-selection "
                + "instruments, variantConditions' fromPromotion, "
                + "perturbationPolicy)",
            repairAction: "steerlab-cli experiment declare-condition <name> "
                + "<condition> …  (re-declare onto a concept that stays), "
                + "then steerlab-cli experiment detach <name> <concept>…"),
        .init(
            gate: .sweepGridRule,
            verbs: ["experiment set-sweep-grid"],
            origin: "ExperimentStore.sweepGridProblem — the grid audit "
                + "(axis emptiness, fraction range, alpha sign, ladder "
                + "ascent, absolute layers against the pinned model's depth)",
            repairAction: "steerlab-cli experiment set-sweep-grid <name> "
                + "--layer-fractions 0.5,0.7,0.85 --alphas 0.05,0.08,0.1,0.13 "
                + "(both axes ascend; alphas are residual-norm units > 0, and "
                + "0 is the implied baseline cell)"),
    ]

    /// The site claiming a gate, or nil — nil is what the exhaustiveness test
    /// fails on.
    public static func site(for gate: LifecycleGate) -> Site? {
        sites.first { $0.gate == gate }
    }
}

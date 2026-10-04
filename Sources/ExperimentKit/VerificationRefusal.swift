import Foundation

// =============================================================================
// What a failed `ExperimentStore.verify` is CALLED when it reaches a caller.
//
// `verify` returns one list of sentences for three different situations, and
// every caller used to label all of them `pinDrift`, with a repair about
// restoring files to their pinned bytes:
//
//   * a pinned file changed, went missing, or appeared after being pinned as
//     absent — real drift, and the only case that repair fits;
//   * the study has nothing attached yet — the first thing a new author meets,
//     where nothing was ever pinned and there is nothing to restore;
//   * the study's own declaration is incomplete or contradicts itself.
//
// Freeze was worse off: it threw the same list untyped, so freezing an empty
// draft arrived as `failed` / 70 / `verbFailed` — the answer a crash gives.
//
// This type names the three. It reads the violation list and never re-derives
// a rule, so it cannot disagree with `verify` about WHETHER a study verifies —
// only about what to call the refusal and which repair to offer.
//
// Python twin: `Server/steerlab_server/experiment/verification_refusal.py`.
// The literals are duplicated there on purpose (the closed-vocabulary idiom
// `LifecycleGate` uses), and
// `VerificationRefusalTests.classificationAndRepairsMatchPython` holds the two
// to the same answers.
// =============================================================================

enum VerificationRefusal {

    /// `verify`'s sentence for a model-output study with no concept and no
    /// agent. Matched whole: it is the rule's entire text on both engines.
    static let emptyModelOutputViolation = "no concepts or variants attached"
    /// The same state for a multi-agent study: no panel scenario is pinned.
    static let emptyMultiAgentViolation = "multi-agent study needs a pinned scenario"

    /// Phrases a violation uses when it reports BYTES — a pinned file that
    /// changed, is gone, or turned up after being pinned as absent. A
    /// violation containing any of them is drift.
    ///
    /// Deliberately generous. A declaration problem mistaken for drift keeps
    /// the label it has always had; drift mistaken for a declaration problem
    /// would take `pinDrift` away from a caller that switches on it. So a
    /// phrase belongs here whenever a drift sentence on either engine uses it.
    static let driftMarkers = [
        "changed since",
        "changed after freeze",
        "missing",
        "appeared after",
        "drifted",
        "not found",
        "no longer",
        "no stories.jsonl",
        "has a markers.json",
        "not importable",
    ]

    /// Does this violation report changed, missing, or newly appeared bytes?
    static func isDrift(_ violation: String) -> Bool {
        driftMarkers.contains { violation.contains($0) }
    }

    /// The empty-study sentence in this list, or nil.
    static func emptyViolation(_ violations: [String]) -> String? {
        [emptyModelOutputViolation, emptyMultiAgentViolation].first {
            violations.contains($0)
        }
    }

    /// The lifecycle gate a non-empty violation list is refused under.
    ///
    /// Drift first: it is the existing code, callers switch on it, and a
    /// study with any drifted pin needs that repaired whatever else is wrong.
    /// Then the empty study. Everything else is the declaration.
    static func gate(_ violations: [String]) -> LifecycleGate {
        if violations.contains(where: isDrift) { return .pinDrift }
        if emptyViolation(violations) != nil { return .emptyStudy }
        return .studyDeclaration
    }

    /// The plain-words reason for an empty study — the same sentence from
    /// every verb on both clients — followed by anything else `verify`
    /// reported, so nothing it said is dropped.
    static func emptyReason(name: String, violations: [String]) -> String {
        let sentinel = emptyViolation(violations)
        let others = violations.filter { $0 != sentinel }
        var reason =
            "'\(name)' is empty: nothing is attached yet (no concept, "
            + "agent, or panel scenario), so there is nothing to measure"
        if !others.isEmpty {
            reason += "\nalso:\n  - " + others.joined(separator: "\n  - ")
        }
        return reason
    }

    /// What to attach, as commands on the client that is answering.
    static func emptyRepair(
        name: String, violations: [String],
        program: String = ExperimentCLIHelp.program
    ) -> String {
        let interview =
            "If the study is not planned yet, begin with the interview: "
            + "\(program) authoring study "
        let template =
            "Or start from a template (the design commands work on "
            + "templates): \(program) design list."
        if emptyViolation(violations) == emptyMultiAgentViolation {
            return "Attach what this study measures, then run the command again. "
                + "A multi-agent study needs a panel scenario: \(program) panel "
                + "list, then \(program) panel compile (its --help names the "
                + "reviewed-file flags). \(template) \(interview)multiAgent "
                + "--json."
        }
        return "Attach what this study measures, then run the command again. "
            + "A concept: \(program) experiment attach \(name) <concept>. "
            + "An agent: \(program) agent list, then \(program) experiment "
            + "attach-agent \(name) (its --help names the reviewed-file flags). "
            + "\(template) \(interview)conceptStudy --json."
    }

    /// The repair when no file changed and the declaration is the problem.
    static func declarationRepair(
        name: String, program: String = ExperimentCLIHelp.program
    ) -> String {
        "Each listed problem names one setting or input of this study "
            + "that is incomplete or inconsistent. Correct each one, then check "
            + "again with \(program) experiment verify \(name). A frozen study "
            + "is never edited: copy it first with \(program) experiment "
            + "duplicate \(name) \(name)-v2."
    }

    /// The repair for this list's gate. Real drift keeps the repair it has
    /// always had (`ExperimentTasks.pinDriftRepair`), including the one-command
    /// answer for a `validation.jsonl` that appeared after attach.
    static func repair(name: String, violations: [String]) -> String {
        switch gate(violations) {
        case .emptyStudy: emptyRepair(name: name, violations: violations)
        case .studyDeclaration: declarationRepair(name: name)
        default: ExperimentTasks.pinDriftRepair(name: name, violations: violations)
        }
    }

    /// The typed refusal for a failed verification.
    ///
    /// `reason` is the prose the site has always thrown and is kept for drift
    /// and declaration problems; an empty study gets `emptyReason` instead, so
    /// freezing, verifying, and validating one all say the same thing.
    static func error(
        name: String, violations: [String], reason: String
    ) -> ExperimentError {
        let gate = gate(violations)
        return .refusing(
            gate,
            gate == .emptyStudy
                ? emptyReason(name: name, violations: violations) : reason,
            repair: repair(name: name, violations: violations))
    }
}

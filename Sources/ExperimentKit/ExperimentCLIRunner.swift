import Foundation
import SteeringKit

// =============================================================================
// The non-cluster `steerlab-cli` verb dispatch (WP0-AGENT-SURFACE-AUDIT §7
// steps 4 and 5), moved here from `Sources/steerlab-cli/main.swift` so the
// agent-path verbs can be driven in-process by tests.
//
// Step 4 was a MOVE: every verb's parsing, refusal prose, side effects, and
// exit code were what they had been in `main.swift`. Step 5 keeps that
// promise for HUMAN output — byte-identical, except for the four deliberate
// fixes below — and adds the machine surface on top of it:
//
//   * `--json` produces exactly ONE `SteerLabCLIEnvelope` document on stdout,
//     with every diagnostic on stderr, for every verb this runner owns.
//   * In JSON mode the exit code comes from the envelope's state vocabulary,
//     so a gate refusal (65), a missing experiment (66), and an operational
//     failure (70) stop being the same `1`. Human mode keeps today's codes —
//     that migration is step 7.
//   * `result` carries what the verb actually learned, with FULL hashes.
//
// The deliberate human-mode changes, each of which was pinned as
// current-behaviour by a step-4 test that this step updates:
//
//   1. STRICT FLAGS (P0-4). An undeclared flag is a usage error, exit 64, in
//      both modes, before the verb does any work. `ExperimentCLIParser` owns
//      the table.
//   2. `attach` REFUSAL BEFORE SUCCESS (P1-6). The per-concept `pinned …`
//      lines are buffered and printed only once every concept resolved, so a
//      failed attach can no longer print a success line for work it did not
//      keep. (It never kept it — `save` is the last statement — so stdout was
//      simply lying.)
//   3. `experiment verify`'s `VIOLATION:` lines move to STDERR. A failure
//      reported on stdout is the audit's §2.1 note "prose, stdout even on
//      failure"; the `OK …` success line stays on stdout where it belongs.
//   4. `promote` and `confirm` print through the sink instead of their
//      default `print` log, so JSON mode can route them.
//
// Deliberately NOT changed: `data check`'s human report. The audit's own dry
// run named it "the model verb" and asked for the rest of the surface to look
// like it; splitting its report across two streams to satisfy a stream rule
// would degrade the one output the probe praised. Its stream fix is the JSON
// one — the whole classified list, with paths, moves into `result`, and in
// JSON mode the human report goes to stderr with everything else.
//
// What did NOT move, and why: `cluster` already lives in `ClusterCLIRunner`
// (this file mirrors it); `serve` blocks forever and returns no outcome;
// `--config` and `artifacts` are not on the agent path (audit §2.1).
//
// `panel` WAS in that list and joined the agent path on 2026-08-19
// (open-issues §18). The reason is not tidiness: `PanelComposition.compileAndPin`
// — bind the study's model and sampling settings to one seat casting, write
// `prompts/panels/compiled/<study>.json`, hand back the path + hash a manifest
// pins — was reachable only from the app, so the only headless way to cast a
// panel was to re-implement the compile transform outside the engine and
// hand-edit the pins into a draft. `panel compile` calls the real seam
// (`SeatCasting.compile`, the same call the study editor and the design
// instantiation table make), and `list`/`check` came along so the family has
// one help page and one output contract. Their human output is byte-identical
// to what `main.swift` printed; only the stream discipline and the envelope
// are new.
// =============================================================================

/// The translation of a mid-verb `exit(n)`: stop now, with this code, and
/// print nothing further. Never escapes the runner.
///
/// Step 5 gives it the envelope's vocabulary too. The four verbs that exit
/// from the middle of their own body — `data check`, `experiment verify`,
/// `vectors compare`, `remote import-chain` — are the runner's REFUSAL sites
/// as much as freeze is: they have said what is wrong, they have a repair,
/// and a machine caller needs the difference between them and a crash.
struct ExperimentCLIStop: Error {
    /// The human-mode exit code, unchanged from what the verb always used.
    let exitCode: Int32
    /// The JSON-mode state.
    var state: SteerLabCLIState = .refused
    /// Stable machine code for `error.code`.
    var code: String = "refused"
    /// Why, in plain language. Empty means "the verb already said it through
    /// the sink" — the envelope's message falls back to a summary.
    var reason: String = ""
    /// The concrete repair.
    var repairAction: String = ""
    /// What the verb learned before it stopped; still worth reporting.
    var payload: [String: JSONValue] = [:]
}

/// The named panel is neither in the library nor a readable file.
///
/// Its own type so the runner can answer `notFound` (66 in JSON mode) rather
/// than the untyped `failed`/70 a bare `ExperimentError` gets. The state
/// vocabulary already names a panel as a `notFound` subject, and the server's
/// `panel check` has exited 66 for this since it existed — Swift was the
/// engine that did not.
struct PanelNotFoundError: Error {
    let identifier: String

    var reason: String { "no panel '\(identifier)' — try `panel list`" }
}

/// Runs one `steerlab-cli` invocation for the families listed in
/// `namespaces`, returning what happened instead of calling `exit`.
public struct ExperimentCLIRunner: Sendable {

    /// The verb families this runner owns. `main.swift` asks before handing
    /// over, so the families that stay behind keep their own ladder rung and
    /// nothing is dispatched twice.
    public static let namespaces: Set<String> = [
        "init", "workspace", "data", "vectors", "remote", "experiment", "docs",
        "install", "panel",
    ]

    /// The top-level spelling of `install version`. `--version` is what a
    /// caller types (and what every other command-line tool answers to), so
    /// the binary rewrites it into the verb rather than growing a second
    /// implementation of the same report.
    public static let versionFlag = "--version"

    private let sink: ExperimentCLISink
    /// Injected clock, so a golden envelope fixture is stable. Same seam and
    /// same shape as `ClusterCLIRunner`'s `now`, which its tests pin to
    /// `Date(timeIntervalSince1970: 1_000)`.
    private let now: @Sendable () -> Date

    public init(
        sink: ExperimentCLISink = .standard,
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.sink = sink
        self.now = now
    }

    // MARK: Entry point

    /// Parse and run. Never throws — every refusal comes back as an outcome,
    /// because a caller must be able to decide what to do next from the
    /// returned value alone (the rule `ClusterCLIRunner.run` already follows).
    public func run(namespace: String, _ args: [String]) async -> ExperimentCLIOutcome {
        do {
            let invocation = try ExperimentCLIParser.parse(namespace: namespace, args)
            return await run(invocation)
        } catch let usage as ExperimentCLIUsageError {
            return usageOutcome(namespace: namespace, args: args, error: usage)
        } catch {
            // `parse` throws nothing else; answering keeps the contract total.
            return outcome(
                namespace: namespace, verb: namespace, state: .blocked,
                exitCode: 64,
                failure: .init(reason: "\(error)"))
        }
    }

    /// Run an already-parsed invocation. `main.swift` uses this form: it has
    /// to know `--json` before it can choose a sink.
    public func run(_ invocation: ExperimentCLIInvocation) async -> ExperimentCLIOutcome {
        let namespace = invocation.namespace
        let verb = invocation.label
        for notice in invocation.deprecations { sink.err(notice + "\n") }

        // `--help` runs NOTHING (WP0 step 11). It is answered here rather than
        // in the binary so the page is in-process testable, and it exits 0 in
        // both modes: asking what a verb accepts was never an error, and until
        // this step the strict parser answered it with 64 — the one refusal a
        // caller cannot repair by reading it.
        if invocation.help { return helpOutcome(invocation) }

        do {
            let result: ExperimentCLIResult
            switch namespace {
            case "init": result = try runInitCommand(invocation)
            case "workspace": result = try runWorkspaceCommand(invocation)
            case "data": result = try runDataCommand(invocation)
            case "vectors": result = try await runVectorsCommand(invocation)
            case "remote": result = try await runRemoteCommand(invocation)
            case "experiment": result = try await runExperimentCommand(invocation)
            case "docs": result = try runDocsCommand(invocation)
            case "install": result = try runInstallCommand(invocation)
            case "panel": result = try runPanelCommand(invocation)
            default:
                // Unreachable via `main.swift`, which checks `namespaces`
                // first. Answering rather than trapping keeps the runner's
                // never-throws contract total.
                return outcome(
                    namespace: namespace, verb: verb, state: .blocked, exitCode: 64,
                    failure: .init(reason: "unknown verb family '\(namespace)'"),
                    code: "unknownVerbFamily",
                    repairAction: "families: "
                        + Self.namespaces.sorted().joined(separator: " | "))
            }
            var envelope = SteerLabCLIEnvelope.success(
                verb: verb, engine: SteerLabCLIEnvelope.localEngine,
                message: result.message, changed: result.changed,
                advisories: result.advisories, observedAt: now(),
                workspace: result.workspaceOverride ?? workspacePath)
            // `success` chooses ready/okWithAdvisories; a verb that declared
            // its own success state (e.g. `pending`) keeps it.
            if result.state != .ready { envelope.state = result.state }
            envelope.nextAction = result.nextAction
            if !result.payload.isEmpty { envelope.result = result.payload }
            return ExperimentCLIOutcome(
                namespace: namespace, verb: verb, exitCode: 0, envelope: envelope)

        } catch let stop as ExperimentCLIStop {
            // A mid-verb `exit(n)`: the verb already said what it had to say
            // through the sink, and `main.swift` printed no trailing line.
            // That stays true — `failure` is nil, so the renderer writes
            // nothing — while the envelope reports the refusal in full.
            var envelope = SteerLabCLIEnvelope.refusal(
                verb: verb, engine: SteerLabCLIEnvelope.localEngine,
                code: stop.code,
                // A mid-verb stop whose code IS a lifecycle gate carries it in
                // `gate` too, so an agent reading `error.gate` sees the same
                // vocabulary here as on a thrown refusal — `verify`'s pin
                // drift must not be the one refusal that names no gate.
                gate: LifecycleGate.vocabulary.contains(stop.code)
                    ? stop.code : nil,
                reason: stop.reason.isEmpty
                    ? "\(verb) refused — see the diagnostics on stderr"
                    : stop.reason,
                repairAction: stop.repairAction, state: stop.state,
                observedAt: now(), workspace: workspacePath)
            if !stop.payload.isEmpty { envelope.result = stop.payload }
            return ExperimentCLIOutcome(
                namespace: namespace, verb: verb, exitCode: stop.exitCode,
                envelope: envelope)

        } catch let usage as ExperimentCLIUsageError {
            return usageOutcome(
                namespace: namespace, args: invocation.args, error: usage)

        } catch let error as ClusterCLIError where namespace == "remote" {
            // --site/--url misuse is a USAGE error (64); a site that simply is
            // not connected yet is retryable (13) and names its own repair.
            return outcome(
                namespace: namespace, verb: verb,
                state: error.exitCode == 13 ? .degraded : .blocked,
                exitCode: error.exitCode,
                failure: .init(
                    reason: error.reason, repairAction: error.repairAction),
                code: error.code, repairAction: error.repairAction)

        } catch let error as ExperimentError {
            // The freeze-gate refusal is the one error that already knows its
            // own gate ids (WP0 step 2); carry them out instead of dropping
            // them on the floor as `main.swift` did.
            var failure = ExperimentCLIOutcome.Failure(reason: "\(error)")
            if let refusal = error.freezeRefusal {
                failure.gate = refusal.gateID
                failure.gates = refusal.gateIDs
                // `gate` is passed EXPLICITLY, not derived from `gates.first`:
                // freeze evaluates in a historical refusal order and reports
                // `gates` in the closed vocabulary's order, which are
                // different permutations, so the derived form would have
                // named a gate the message does not describe (step 2's
                // flagged divergence, reconciled here).
                let envelope = SteerLabCLIEnvelope.refusal(
                    verb: verb, engine: SteerLabCLIEnvelope.localEngine,
                    code: "freezeGateFailed", gate: refusal.gateID,
                    gates: refusal.gateIDs, reason: refusal.reason,
                    repairAction: refusal.repairAction, state: .refused,
                    observedAt: now(), workspace: workspacePath)
                return ExperimentCLIOutcome(
                    namespace: namespace, verb: verb, exitCode: 1,
                    failure: failure, envelope: envelope)
            }
            // STEP 7: the second closed vocabulary. A gate that declined a
            // well-formed request against a healthy system is `refused` (65)
            // with its own id and a RUNNABLE repair — no longer
            // `failed/70/verbFailed` with "read the reason and repair the
            // named input", which is what dry run #1 measured on all four of
            // its repairable refusals (§9).
            //
            // Human mode keeps exit 1 here: the audit's §7 row 7 names ONE
            // human-mode migration (`data check`'s 2 → 65), and widening it to
            // every refusal would break `set -e` wrappers the row does not
            // discuss. The document is where the distinction lives.
            if let refusal = error.lifecycleRefusal {
                failure.gate = refusal.gateID
                let envelope = SteerLabCLIEnvelope.refusal(
                    verb: verb, engine: SteerLabCLIEnvelope.localEngine,
                    code: refusal.gateID, gate: refusal.gateID,
                    reason: refusal.reason,
                    // The store cannot know which verb asked, so the shared
                    // frozen-manifest repair carries a placeholder for it.
                    // Substituting HERE is what makes the repair runnable
                    // rather than nearly runnable — the distinction dry run #1
                    // was decided by.
                    repairAction: refusal.repairAction.replacingOccurrences(
                        of: "<the verb you just ran>",
                        with: invocation.verb ?? "<verb>"),
                    state: .refused, observedAt: now(), workspace: workspacePath)
                return ExperimentCLIOutcome(
                    namespace: namespace, verb: verb, exitCode: 1,
                    failure: failure, envelope: envelope)
            }
            // A value the verb's vocabulary does not contain: the same class
            // an undeclared flag lands in (`blocked`, 64 in JSON), with the
            // legal values as the repair. Gate-5 dry run #2 (P3) found these
            // arriving as `verbFailed`/70 — an operational failure, which
            // tells an agent to retry rather than to retype.
            if let malformed = error.malformedInvocation {
                return outcome(
                    namespace: namespace, verb: verb, state: .blocked,
                    exitCode: 1,
                    failure: .init(
                        reason: error.reason,
                        repairAction: malformed.repairAction),
                    code: "usage", repairAction: malformed.repairAction)
            }
            // Everything else in this family keeps human exit 1. In JSON mode
            // a MALFORMED INVOCATION is 64 and anything else is an
            // operational failure (70).
            let usage = Self.usageShape(of: error.reason)
            return outcome(
                namespace: namespace, verb: verb,
                state: usage == nil ? .failed : .blocked, exitCode: 1,
                failure: failure,
                code: usage ?? "verbFailed",
                repairAction: usage == nil
                    // The honest text (gate-5 dry run #2, P3): this is the
                    // catch-all for a throw that carries NO structure, so
                    // there is no repair to derive. Saying "read the reason
                    // and repair the named input" implied one had been
                    // derived and the input had been named; both are false
                    // here, and every throw that DOES know its repair now
                    // carries it (FreezeRefusal, LifecycleRefusal,
                    // MalformedInvocation) and never reaches this line.
                    ? "this was not a typed refusal — read the reason; if it "
                        + "names a file or a pin, repair that, otherwise the "
                        + "verb failed operationally and the reason is all "
                        + "there is"
                    // A missing or unrecognised sub-verb: the reason ALREADY
                    // is the roster, and repeating it as the repair told an
                    // agent nothing it had not just read. `--help` is the
                    // discoverable surface (step 11) and answers the next
                    // question — what the verb's arguments are — which the
                    // roster does not. A missing POSITIONAL keeps its
                    // `usage:` line: that is the specific answer.
                    : usage == "unknownVerb"
                        ? "\(ExperimentCLIHelp.program) \(namespace) --help  "
                            + "(the family's verbs), then "
                            + "\(ExperimentCLIHelp.program) \(namespace) "
                            + "<verb> --help  (one verb's arguments)"
                        : error.reason)

        } catch let error as StimulusSetError where Self.namesMissingFiles(error) {
            // The fourth verb in the same family (2026-08-18): `attach` against
            // a concept whose stimulus files are not on disk. The refusal is
            // already excellent PROSE — it names every absent file and the row
            // shape they must have — and it was untyped, so the envelope said
            // `failed`/70/`verbFailed`, the same answer a crash gives.
            //
            // Classified HERE rather than in `StimulusSet`: the gate
            // vocabulary is an ExperimentKit contract and SteeringKit is
            // concept-agnostic by hard requirement, so the id is attached at
            // the boundary that owns it. This clause covers every verb that
            // loads stimuli (attach, extract, validate, sweep), not just
            // attach. Human prose is `"\(error)"` exactly as before.
            let rerun =
                "steerlab-cli \(namespace) "
                + invocation.args.joined(separator: " ")
            return outcome(
                namespace: namespace, verb: verb, state: .refused,
                exitCode: LifecycleGate.missingPrerequisite.humanExitCode,
                failure: .init(
                    reason: "\(error)",
                    gate: LifecycleGate.missingPrerequisite.rawValue),
                code: LifecycleGate.missingPrerequisite.rawValue,
                repairAction: "author each named file as "
                    + #"{"text": "…"} JSONL rows under "#
                    + "prompts/concepts/<concept>/ (positive.jsonl + "
                    + "negative.jsonl), then \(rerun)")

        } catch let error as PanelNotFoundError {
            // The panel family's `notFound`. Prose byte-identical to what
            // `main.swift` printed before the family moved here.
            return outcome(
                namespace: namespace, verb: verb, state: .notFound, exitCode: 1,
                failure: .init(reason: error.reason), code: "notFound",
                repairAction: "steerlab-cli panel list  (the panels this "
                    + "workspace holds), then re-run with a name or a path from "
                    + "it")

        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            // The commonest agent mistake, and until now indistinguishable
            // from a real failure: a manifest, run, or artifact that is not
            // there. `notFound` is 66 in JSON mode; human mode stays 1.
            //
            // The human line keeps the raw `\(error)` dump it has always
            // printed; the ENVELOPE gets a clean sentence instead. That is
            // not politeness — Foundation's `CocoaError` description embeds
            // an `NSUnderlyingError` POINTER, so a document built from it
            // would differ on every invocation and no golden fixture (or
            // agent's equality check) could ever hold.
            let missing = error.userInfo[NSFilePathErrorKey] as? String
            // Gate-5 dry run #2 (P3): the commonest instance of this by far
            // is a MISTYPED EXPERIMENT NAME, and answering it with an absolute
            // `/…/experiments/typo/experiment.json` made an agent reason about
            // the filesystem when the fact is about the workspace's roster.
            // Name the experiment; the path stays in `result` nowhere and in
            // the human line, which is unchanged.
            let missingExperiment = missing.flatMap(Self.experimentName(inMissingPath:))
            return outcome(
                namespace: namespace, verb: verb, state: .notFound, exitCode: 1,
                failure: .init(reason: "\(error)"),
                code: "notFound",
                repairAction: missingExperiment.map { _ in
                    "steerlab-cli experiment list  (the experiments this "
                        + "workspace holds), then re-run with a name from it, "
                        + "or steerlab-cli experiment create <name> --model <id>"
                } ?? missing.map { "no file at \($0)" }
                    ?? "check the name against `steerlab-cli experiment list`",
                envelopeReason: missingExperiment.map {
                    "experiment '\($0)' not found in this workspace"
                } ?? missing.map { "no such file: \($0)" }
                    ?? "a file this verb needs does not exist")

        } catch {
            return outcome(
                namespace: namespace, verb: verb, state: .failed, exitCode: 1,
                failure: .init(reason: "\(error)"), code: "verbFailed",
                repairAction: "this was not a typed refusal — read the reason; "
                    + "if it names a file or a pin, repair that, otherwise the "
                    + "verb failed operationally and the reason is all there is")
        }
    }

    // MARK: Outcome construction

    /// Which data root answered — so an agent can tell a wrong-workspace
    /// answer from a wrong answer.
    private var workspacePath: String { ExperimentStore.workspaceRoot.path }

    /// Is this thrown prose a MALFORMED INVOCATION rather than a failure? The
    /// dispatch answers both through the same `ExperimentError`, so the shape
    /// of the message is all there is to read: `usage: …` for a missing
    /// positional, `verbs: …` / `remote verbs: …` for an unrecognised
    /// sub-verb (the audit appendix's "a usage error that should be 64").
    ///
    /// A stopgap, and deliberately a narrow one — step 11's declarative verb
    /// tables give usage its own type and this function goes away. It affects
    /// the JSON state only: the human line and its exit 1 are untouched.
    /// The experiment a missing-file path is about, when the path is a
    /// manifest — `…/experiments/<name>/experiment.json`.
    ///
    /// Deliberately shape-matched rather than threaded down from the verb:
    /// `ExperimentStore.load` is called from ~40 places and throws
    /// Foundation's own error, so the classification site is the one place
    /// that sees every instance. A path that is NOT a manifest keeps the
    /// file-shaped answer — inventing an experiment name for a missing rubric
    /// would be worse than the raw path.
    static func experimentName(inMissingPath path: String) -> String? {
        let components = URL(filePath: path).pathComponents
        guard components.last == "experiment.json", components.count >= 3 else {
            return nil
        }
        let name = components[components.count - 2]
        guard components[components.count - 3] == "experiments", !name.isEmpty else {
            return nil
        }
        return name
    }

    /// Is this stimulus-loading failure an ABSENT FILE — the repairable case —
    /// rather than a malformed row or an empty set?
    ///
    /// Only the absent-file cases are a `missingPrerequisite`: a malformed row
    /// already names `file:line` and its own expected shape, and folding it in
    /// would tell an agent to author a file that exists.
    static func namesMissingFiles(_ error: StimulusSetError) -> Bool {
        switch error {
        case .missingFile, .missingFiles: true
        case .malformedLine, .malformedTextRow, .empty: false
        }
    }

    static func usageShape(of reason: String) -> String? {
        if reason.hasPrefix("usage:") { return "usage" }
        if reason.hasPrefix("verbs:") || reason.hasPrefix("remote verbs:") {
            return "unknownVerb"
        }
        return nil
    }

    /// `envelopeReason` overrides the human `failure.reason` in the DOCUMENT
    /// only — used where the thrown error's own description is unstable or
    /// unreadable, and the human line must nonetheless stay byte-identical.
    private func outcome(
        namespace: String, verb: String, state: SteerLabCLIState, exitCode: Int32,
        failure: ExperimentCLIOutcome.Failure, code: String = "verbFailed",
        repairAction: String = "", envelopeReason: String? = nil
    ) -> ExperimentCLIOutcome {
        var envelope = SteerLabCLIEnvelope.refusal(
            verb: verb, engine: SteerLabCLIEnvelope.localEngine, code: code,
            gate: failure.gate, gates: failure.gates ?? [],
            reason: envelopeReason ?? failure.reason,
            repairAction: failure.repairAction ?? repairAction, state: state,
            observedAt: now(), workspace: workspacePath)
        if state.isSuccess { envelope.error = nil }
        return ExperimentCLIOutcome(
            namespace: namespace, verb: verb, exitCode: exitCode,
            failure: failure, envelope: envelope)
    }

    /// `--help`: the declared surface, exit 0, nothing run.
    ///
    /// The page goes through the SINK, so in `--json` mode it lands on stderr
    /// and stdout still carries exactly one document — whose `result` is the
    /// same page as data, because a machine caller should not have to parse
    /// the columns a human reads.
    private func helpOutcome(_ invocation: ExperimentCLIInvocation) -> ExperimentCLIOutcome {
        let namespace = invocation.namespace
        let spec = ExperimentCLIParser.spec(
            namespace: namespace, verb: invocation.verb)
        let text = spec.map(ExperimentCLIHelp.text(for:))
            ?? ExperimentCLIHelp.familyText(for: namespace)
        sink.out(text, terminator: "")
        let payload = spec.map(ExperimentCLIHelp.payload(for:))
            ?? ExperimentCLIHelp.familyPayload(for: namespace)
        let verb = spec?.label ?? namespace
        var envelope = SteerLabCLIEnvelope.success(
            verb: verb, engine: SteerLabCLIEnvelope.localEngine,
            message: spec?.purpose ?? "\(namespace) verbs and what each one does",
            changed: false, advisories: [], observedAt: now(),
            workspace: workspacePath)
        envelope.result = payload
        return ExperimentCLIOutcome(
            namespace: namespace, verb: verb, exitCode: 0, envelope: envelope)
    }

    /// An undeclared flag: 64 in BOTH modes, and nothing was run.
    private func usageOutcome(
        namespace: String, args: [String], error: ExperimentCLIUsageError
    ) -> ExperimentCLIOutcome {
        outcome(
            namespace: namespace, verb: error.verb, state: .blocked, exitCode: 64,
            failure: .init(reason: error.reason, repairAction: error.repairAction),
            code: ExperimentCLIUsageError.code, repairAction: error.repairAction)
    }

    // MARK: - init

    // The home layout (GENERAL-DISTRIBUTION-WORK-PLAN decision 8, work item
    // (a)). Creates `Workspaces/` and `Sites/` under the home and reports what
    // it found; idempotent, never destructive, and deliberately inert with
    // respect to everything else — no workspace, no git, no persisted
    // preference, so the workspace resolution chain is untouched.
    func runInitCommand(_ invocation: ExperimentCLIInvocation) throws
        -> ExperimentCLIResult
    {
        let usage = "usage: init [--home <path>]"
        var homePath: String?
        var index = 0
        while index < invocation.args.count {
            guard invocation.args[index] == "--home" else {
                // The family takes no positional; the strict parser has
                // already refused every undeclared FLAG.
                throw ExperimentError(reason: usage)
            }
            guard index + 1 < invocation.args.count else {
                throw ExperimentError(reason: usage)
            }
            homePath = invocation.args[index + 1]
            index += 2
        }

        let home =
            homePath.map { URL(filePath: $0).standardizedFileURL }
            ?? HomeLayout.defaultHome
        let planned = try HomeLayout.plan(home: home)
        let layout = try HomeLayout.apply(planned)

        let created = planned.missing.map(\.name)
        sink.out(
            "SteerLab home: \(layout.home.path)"
                + (planned.homeExisted ? "" : "  (created)"))
        // Pad, never truncate: a checkout folder can be named anything, and
        // `padding(toLength:)` would cut a long name down to the column width
        // — reporting a directory that is not the one on disk.
        let width = (HomeLayout.directoryNames.map(\.count).max() ?? 0) + 1
        func column(_ text: String) -> String {
            text.count >= width
                ? text
                : text + String(repeating: " ", count: width - text.count)
        }
        for entry in planned.entries {
            let status = entry.existed ? "existing" : "created "
            sink.out("  \(status)  \(column(entry.name + "/"))  \(entry.purpose)")
        }
        for checkout in layout.checkouts {
            sink.out(
                "  checkout  \(column(checkout.lastPathComponent + "/"))  "
                    + "code checkout (detected by content, not by name)")
        }
        if layout.checkouts.isEmpty, let external = layout.externalCheckout {
            sink.out(
                "  no code checkout in this home — the running binary's is "
                    + external.path)
        }
        sink.out("")
        sink.out(HomeLayout.sitesExplanation)
        sink.out(
            "next: steerlab-cli workspace init "
                + layout.home.appending(component: HomeLayout.workspacesDirectoryName)
                .appending(component: "<study>").path)

        var payload: [String: JSONValue] = [
            "home": .string(layout.home.path),
            "homeCreated": .bool(!planned.homeExisted),
            "created": .array(created.map { .string($0) }),
            "entries": .array(
                planned.entries.map { entry in
                    .object([
                        "name": .string(entry.name),
                        "path": .string(entry.url.path),
                        "created": .bool(!entry.existed),
                    ])
                }),
            "checkouts": .array(layout.checkouts.map { .string($0.path) }),
        ]
        if let external = layout.externalCheckout {
            payload["runningBinaryCheckout"] = .string(external.path)
        }

        let message =
            created.isEmpty
            ? "SteerLab home layout already complete at \(layout.home.path)"
            : "SteerLab home layout ready at \(layout.home.path) — created "
                + created.map { $0 + "/" }.joined(separator: ", ")
        return ExperimentCLIResult(
            message: message, changed: !created.isEmpty, payload: payload,
            nextAction: .init(
                verb: "workspace init "
                    + layout.home.appending(
                        component: HomeLayout.workspacesDirectoryName
                    ).appending(component: "<study>").path,
                detail: "init materializes directories only — it does not "
                    + "create a workspace and does not change how a workspace "
                    + "root is resolved"))
    }

    // MARK: - workspace

    // Workspace lifecycle: create/seed a data workspace folder.
    func runWorkspaceCommand(_ invocation: ExperimentCLIInvocation) throws
        -> ExperimentCLIResult
    {
        let args = invocation.args
        switch args.first {
        case "init":
            guard args.count >= 2 else {
                throw ExperimentError(reason: "usage: workspace init <path>")
            }
            let root = try WorkspaceStore.create(
                at: URL(filePath: args[1]).standardizedFileURL)
            sink.out("created workspace at \(root.path)")
            var payload: [String: JSONValue] = ["workspace": .string(root.path)]
            if let seed = try? CodeResources.workspaceSeed() {
                sink.out("  seeded from \(seed.path)")
                payload["seededFrom"] = .string(seed.path)
            }
            sink.out("  use it via: steerlab-cli --workspace \(root.path) <verb> …")
            sink.out("  or: export STEERLAB_WORKSPACE=\(root.path)")
            return ExperimentCLIResult(
                message: "created workspace at \(root.path)", changed: true,
                payload: payload,
                nextAction: .init(
                    verb: "experiment create <name> --model <id>",
                    detail: "run with --workspace \(root.path), or export "
                        + "STEERLAB_WORKSPACE=\(root.path)"),
                // The root this verb answered ABOUT is the one it just made,
                // not the one the invocation resolved to (which is still the
                // old/fallback root — this verb takes its target as a
                // positional and never switches the process over to it).
                workspaceOverride: root.path)
        default:
            throw ExperimentError(reason: "usage: workspace init <path>")
        }
    }

    // MARK: - data

    // Study-data readiness: what data the manifest still needs and where it goes.
    func runDataCommand(_ invocation: ExperimentCLIInvocation) throws
        -> ExperimentCLIResult
    {
        let args = invocation.args
        switch args.first {
        case "check":
            guard args.count >= 2 else {
                throw ExperimentError(reason: "usage: data check <experiment>")
            }
            let manifest = try ExperimentStore.load(name: args[1])
            let requirements = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: VectorCatalog.projectRoot)
            let summary = StudyDataReadiness.summary(requirements)
            func glyph(_ status: DataRequirement.Status) -> String {
                switch status {
                case .present: "✓"
                case .partial: "◐"
                case .invalid: "✗"
                case .missing: "✗"
                case .optional: "·"
                }
            }
            // Blockers first (files the run refuses, then absent files), then
            // partial, present, optional.
            let order: [DataRequirement.Status] = [
                .invalid, .missing, .partial, .present, .optional,
            ]
            for status in order {
                for requirement in requirements where requirement.status == status {
                    sink.out(
                        "\(glyph(status)) [\(status.rawValue)] \(requirement.title)  "
                            + "— \(requirement.path)")
                    sink.out("    \(requirement.detail)")
                }
            }
            sink.out(summary.line)
            // The classified item list, modelled on the human report the dry
            // run praised: every requirement, its status, and — the part an
            // agent cannot get any other way — the PATH it must author.
            let payload = Self.readinessPayload(
                experiment: manifest.name, requirements: requirements,
                order: order, summary: summary)
            if !summary.isReady {
                let plural = summary.blockers.count == 1 ? "" : "s"
                sink.err("data check failed: \(summary.blockers.count) blocker\(plural)\n")
                // THE human-mode exit migration (audit §2.3's stated migration
                // debt, scheduled by §7 row 7: "`data check` blockers (2 →
                // 65)"). It is the one code the row names, it lands once, and
                // it lands with the envelope's schemaVersion 1 rather than
                // being aliased: `data check` now answers 65 in BOTH modes, so
                // the family has exactly one refusal code again. Every other
                // refusal keeps human exit 1 — the row does not migrate them.
                throw ExperimentCLIStop(
                    exitCode: LifecycleGate.dataReadiness.humanExitCode,
                    state: .refused, code: LifecycleGate.dataReadiness.rawValue,
                    reason: "\(summary.blockers.count) blocking data requirement"
                        + "\(plural) for '\(manifest.name)'",
                    repairAction: "author the files named by "
                        + "result.blockers[].path, then steerlab-cli data check "
                        + "\(manifest.name)",
                    payload: payload)
            }
            return ExperimentCLIResult(
                message: summary.line, payload: payload)
        default:
            throw ExperimentError(reason: "usage: data check <experiment>")
        }
    }

    /// `data check`'s payload — the same classification the human report
    /// shows, in the same order, machine-readable.
    static func readinessPayload(
        experiment: String, requirements: [DataRequirement],
        order: [DataRequirement.Status], summary: ReadinessSummary
    ) -> [String: JSONValue] {
        func item(_ requirement: DataRequirement) -> JSONValue {
            var fields: [String: JSONValue] = [
                "id": .string(requirement.id),
                "title": .string(requirement.title),
                "kind": .string("\(requirement.kind)"),
                "status": .string(requirement.status.rawValue),
                "path": .string(requirement.path),
                "detail": .string(requirement.detail),
            ]
            if let template = requirement.templateID {
                fields["templateID"] = .string(template)
            }
            return .object(fields)
        }
        let ordered = order.flatMap { status in
            requirements.filter { $0.status == status }
        }
        return [
            "experiment": .string(experiment),
            "ready": .bool(summary.isReady),
            "summary": .string(summary.line),
            "counts": .object([
                "present": .number(Double(summary.presentCount)),
                "partial": .number(Double(summary.partialCount)),
                "invalid": .number(Double(summary.invalidCount)),
                "missing": .number(Double(summary.missingCount)),
                "optional": .number(Double(summary.optionalCount)),
            ]),
            "items": .array(ordered.map(item)),
            "blockers": .array(summary.blockers.map(item)),
        ]
    }

    // MARK: - install

    /// The shipped binary talking about itself (audit §6, step 12).
    ///
    /// `version` is what `steerlab --version` rewrites to. It is deliberately
    /// CHEAP: it reports whether the tree is stamped and names the check,
    /// rather than hashing a 147 MB executable on a verb whose whole job is to
    /// answer quickly. `verify` is the check, and it is the only hasher —
    /// `scripts/install-cli.sh --verify` runs this verb instead of
    /// re-implementing SHA-256 in shell.
    func runInstallCommand(_ invocation: ExperimentCLIInvocation) throws
        -> ExperimentCLIResult
    {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), args.count > index + 1 else {
                return nil
            }
            return args[index + 1]
        }
        let provenance = InstallProvenance.resolve()
        /// `--root` exists so the installer can stamp or check a STAGING tree
        /// before it becomes the live install — the atomicity the script
        /// depends on. Absent, both verbs act on the running executable's own
        /// directory, which is the ordinary case.
        let root = flag("--root").map { URL(filePath: $0).standardizedFileURL }
            ?? provenance.root

        switch args.first {
        case "version":
            sink.out(provenance.report, terminator: "")
            var result = ExperimentCLIResult(
                message: "\(provenance.engineVersion) — "
                    + (provenance.shape == .installed
                        ? "installed at \(provenance.root.path)"
                        : "unstamped build product at \(provenance.root.path)"),
                payload: provenance.payload)
            if provenance.shape == .installed {
                result.nextAction = .init(
                    verb: "install verify",
                    detail: "check the installed tree against its "
                        + "resource manifest")
            }
            return result

        case "stamp":
            let manifest = try InstallProvenance.stamp(
                root: root, sourceRevision: flag("--revision"))
            sink.out(
                "stamped \(manifest.files.count) file(s) into "
                    + "\(root.appending(component: InstallProvenance.manifestName).path)")
            return ExperimentCLIResult(
                message: "stamped \(manifest.files.count) installed file(s)",
                changed: true,
                payload: [
                    "installRoot": .string(root.path),
                    "fileCount": .number(Double(manifest.files.count)),
                    "appVersion": .string(manifest.appVersion),
                    "sourceRevision": manifest.sourceRevision.map {
                        JSONValue.string($0)
                    } ?? .null,
                ])

        case "verify":
            let target =
                root == provenance.root
                ? provenance
                : InstallProvenance.resolve(
                    executable: root.appending(
                        component: provenance.executable.lastPathComponent))
            let problems = try target.verifyInstalledTree()
            var payload: [String: JSONValue] = [
                "installRoot": .string(target.root.path),
                "fileCount": .number(
                    Double(target.manifest?.files.count ?? 0)),
                "metallibColocated": .bool(target.metallibColocated),
                "problems": .array(
                    problems.map { problem in
                        .object([
                            "kind": .string(problem.kind.rawValue),
                            "path": .string(problem.path),
                            "message": .string(problem.message),
                        ])
                    }),
            ]
            guard problems.isEmpty else {
                for problem in problems { sink.err("PROBLEM: \(problem.message)\n") }
                throw ExperimentCLIStop(
                    exitCode: 1, state: .refused, code: "installIntegrity",
                    reason: "\(problems.count) installed file(s) do not match "
                        + "\(target.root.path)/\(InstallProvenance.manifestName)",
                    repairAction: "reinstall: scripts/install-cli.sh",
                    payload: payload)
            }
            sink.out(
                "\(target.manifest?.files.count ?? 0) installed file(s) match "
                    + "\(InstallProvenance.manifestName)")
            if !target.metallibColocated {
                sink.out(
                    "  note: no colocated \(InstallProvenance.metallibName) — "
                        + "GPU verbs will need DYLD_FRAMEWORK_PATH")
            }
            // `--gpu` is the audit's §6.1 claim under test, and it needs no
            // model weights — so it is the one GPU check that runs on a
            // machine with an empty Hugging Face cache.
            if args.contains("--gpu") {
                let proof = InstallProvenance.probeMetalShaders()
                sink.out("  \(proof)")
                payload["metalProbe"] = .string(proof)
            }
            return ExperimentCLIResult(
                message: "the installed tree matches its resource manifest",
                payload: payload)

        default:
            throw ExperimentError(
                reason: "usage: install version | install stamp [--revision "
                    + "<commit>] [--root <dir>] | install verify [--root <dir>]")
        }
    }

    // MARK: - docs

    /// Regenerate — or check — this engine's marked regions in
    /// `docs/CLI-REFERENCE.md` (audit §5.3).
    ///
    /// Generation is deliberately NOT a build step: the document must be
    /// readable on GitHub with no toolchain, so the generated text is
    /// COMMITTED and a test compares. `--check` is the same comparison from a
    /// command line; it refuses (65) on drift, and the repair is one command.
    func runDocsCommand(_ invocation: ExperimentCLIInvocation) throws
        -> ExperimentCLIResult
    {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), args.count > index + 1 else {
                return nil
            }
            return args[index + 1]
        }
        guard args.first == "cli-reference" else {
            throw ExperimentError(
                reason: "usage: docs cli-reference [--check | --write] "
                    + "[--path <file>]")
        }
        let url: URL
        if let path = flag("--path") {
            url = URL(filePath: path).standardizedFileURL
        } else if let committed = CLIReferenceDocument.committedURL {
            url = committed
        } else {
            throw ExperimentError(
                reason: "no docs/CLI-REFERENCE.md next to this binary — pass "
                    + "--path <file>")
        }
        let document = try String(contentsOf: url, encoding: .utf8)
        let drifted = try CLIReferenceDocument.drift(in: document)
        let payload: [String: JSONValue] = [
            "path": .string(url.path),
            "regions": .array(
                CLIReferenceDocument.ownedRegionIDs.sorted().map { .string($0) }),
            "driftedRegions": .array(drifted.map { .string($0) }),
        ]

        if args.contains("--write") {
            let rewritten = try CLIReferenceDocument.rewrite(document)
            if rewritten == document {
                sink.out("docs/CLI-REFERENCE.md is already in sync")
                return ExperimentCLIResult(
                    message: "\(url.lastPathComponent) already in sync",
                    payload: payload)
            }
            try rewritten.write(to: url, atomically: true, encoding: .utf8)
            sink.out(
                "rewrote \(drifted.count) region(s) in \(url.path): "
                    + drifted.joined(separator: ", "))
            return ExperimentCLIResult(
                message: "rewrote \(drifted.count) generated region(s)",
                changed: true, payload: payload)
        }

        // `--check` is the default: a verb that rewrote a committed document
        // because a flag was forgotten would be the wrong default by a wide
        // margin.
        guard drifted.isEmpty else {
            for id in drifted { sink.err("DRIFT: region \(id) is out of date\n") }
            throw ExperimentCLIStop(
                exitCode: 1, state: .refused, code: "documentationDrift",
                reason: "\(drifted.count) generated region(s) in "
                    + "\(url.lastPathComponent) do not match the verb tables",
                repairAction: "steerlab-cli docs cli-reference --write, then "
                    + "commit the document",
                payload: payload)
        }
        sink.out(
            "docs/CLI-REFERENCE.md: \(CLIReferenceDocument.ownedRegionIDs.count) "
                + "generated region(s) match the verb tables")
        return ExperimentCLIResult(
            message: "every generated region matches the verb tables",
            payload: payload)
    }

    /// The envelope half of the `caseFamily` deprecation (2026-08-18): the
    /// same sentence `ExperimentTasks` logs, carried as the closed-vocabulary
    /// code `deprecatedImplicitSelection` so an agent can switch on it instead
    /// of grepping a log. Python twin:
    /// `cli._implicit_case_family_advisories`.
    ///
    /// Tolerant by design, like every other advisory helper here: a manifest
    /// that cannot be read is a problem the verb itself has already reported
    /// properly, and an advisory must never be the thing that fails a verb.
    /// Advisories never change the exit code.
    static func implicitCaseFamilyAdvisories(
        experimentNamed name: String
    ) -> [SteerLabCLIEnvelope.Advisory] {
        guard let manifest = try? ExperimentStore.load(name: name),
            manifest.usesImplicitCaseFamilyEndpoint
        else { return [] }
        return [
            .init(
                CLIAdvisory.deprecatedImplicitSelection,
                ExperimentManifest.implicitCaseFamilyAdvisory),
        ]
    }

    // MARK: - panel

    /// Accept either a path or a panel name, so `panel check`/`panel compile`
    /// work from a `panel list` line without copying the full path.
    ///
    /// Moved verbatim from `main.swift`'s `resolvePanel`, so the three verbs
    /// resolve their positional identically — which is the point: `compile`
    /// naming a different set of panels than `check` validates would be a
    /// silent trap.
    static func resolvePanel(_ identifier: String) throws -> MultiAgentScenarioRecord {
        let records = MultiAgentScenarioStore.scan()
        if let match = records.first(where: {
            $0.url.path == identifier || $0.scenario.name == identifier
                || $0.url.lastPathComponent == identifier
        }) {
            return match
        }
        let url = identifier.hasPrefix("/")
            ? URL(filePath: identifier)
            : ExperimentStore.workspaceRoot.appending(path: identifier)
        guard
            let data = try? Data(contentsOf: url),
            let scenario = try? JSONDecoder().decode(MultiAgentScenario.self, from: data)
        else {
            throw PanelNotFoundError(identifier: identifier)
        }
        return MultiAgentScenarioRecord(url: url, scenario: scenario)
    }

    /// One `--seat <id>=<agent-artifact-path>` pair, split at the FIRST `=`
    /// so a path containing one still parses.
    static func parseSeatPair(_ raw: String) throws -> (seat: String, path: String) {
        guard let separator = raw.firstIndex(of: "="), separator != raw.startIndex
        else {
            throw ExperimentError(
                reason: "usage: panel compile <path-or-name> --experiment <name> "
                    + "[--seat <seat>=<agent-artifact-path>]… "
                    + "[--model <id>] [--temperature <t>] [--max-tokens <n>] "
                    + "[--file-slug <slug>]")
        }
        let seat = String(raw[raw.startIndex ..< separator])
            .trimmingCharacters(in: .whitespaces)
        let path = String(raw[raw.index(after: separator)...])
            .trimmingCharacters(in: .whitespaces)
        guard !seat.isEmpty, !path.isEmpty else {
            throw ExperimentError(
                reason: "usage: panel compile <path-or-name> --experiment <name> "
                    + "[--seat <seat>=<agent-artifact-path>]… "
                    + "[--model <id>] [--temperature <t>] [--max-tokens <n>] "
                    + "[--file-slug <slug>]")
        }
        return (seat, path)
    }

    func runPanelCommand(_ invocation: ExperimentCLIInvocation) throws
        -> ExperimentCLIResult
    {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), args.count > index + 1 else {
                return nil
            }
            return args[index + 1]
        }
        /// Every occurrence of a repeatable value flag, in the order typed.
        func repeatedFlag(_ name: String) -> [String] {
            var values: [String] = []
            var index = 0
            while index < args.count {
                if args[index] == name, index + 1 < args.count {
                    values.append(args[index + 1])
                    index += 2
                } else {
                    index += 1
                }
            }
            return values
        }

        switch args.first {
        case "list":
            let records = MultiAgentScenarioStore.scan()
            if records.isEmpty {
                sink.out("no panels found in \(MultiAgentScenarioStore.directory.path)")
                return ExperimentCLIResult(
                    message: "no panels in \(MultiAgentScenarioStore.directory.path)",
                    payload: ["count": .number(0), "panels": .array([])])
            }
            for record in records {
                sink.out("\(record.scenario.name)\t\(record.url.path)")
                sink.out(
                    "  \(record.scenario.agents.count) agents, "
                        + "\(record.scenario.turns.count) turns")
            }
            return ExperimentCLIResult(
                message: "\(records.count) panel(s)",
                payload: [
                    "count": .number(Double(records.count)),
                    "panels": .array(
                        records.map { record in
                            .object([
                                "name": .string(record.scenario.name),
                                "path": .string(record.url.path),
                                "seatCount": .number(Double(record.scenario.agents.count)),
                                "turnCount": .number(Double(record.scenario.turns.count)),
                                // The property that decides whether a panel can
                                // be CAST at all, reported rather than left for
                                // the caller to infer from empty model fields.
                                "semantic": .bool(
                                    PanelComposition.isSemantic(record.scenario)),
                                "seats": .array(
                                    record.scenario.agents.map { .string($0.id) }),
                            ])
                        }),
                ])

        case "check":
            guard args.count >= 2 else {
                throw ExperimentError(reason: "usage: panel check <path-or-name>")
            }
            let record = try Self.resolvePanel(args[1])
            // Hard errors first — these make the panel unrunnable.
            try MultiAgentRunner.validate(record.scenario)
            let headline =
                "valid: \(record.scenario.agents.count) agents, "
                + "\(record.scenario.turns.count) turns — \(record.url.path)"
            sink.out(headline)
            // Then the silent-failure advisories (plan F1).
            let advisories = MultiAgentRunner.advisories(record.scenario)
            for advisory in advisories { sink.out("ADVISORY: \(advisory)") }
            if !advisories.isEmpty {
                sink.out(
                    "\(advisories.count) advisory(ies) — these do not block a run; "
                        + "they make prompts quietly wrong, so fix them before "
                        + "measuring.")
            }
            return ExperimentCLIResult(
                message: headline,
                payload: [
                    "name": .string(record.scenario.name),
                    "path": .string(record.url.path),
                    "seatCount": .number(Double(record.scenario.agents.count)),
                    "turnCount": .number(Double(record.scenario.turns.count)),
                    "semantic": .bool(PanelComposition.isSemantic(record.scenario)),
                    "seats": .array(record.scenario.agents.map { .string($0.id) }),
                    // Panel advisories are the scenario's OWN vocabulary (free
                    // prose from `MultiAgentRunner.advisories`), not the
                    // envelope's closed cross-engine advisory codes, so they
                    // ride in `result` where an open vocabulary is legal.
                    "panelAdvisories": .array(advisories.map { .string($0) }),
                ])

        case "compile":
            return try compilePanel(args: args, flag: flag, repeatedFlag: repeatedFlag)

        default:
            throw ExperimentError(
                reason: "verbs: list | check <path-or-name> | compile "
                    + "<path-or-name> --experiment <name>")
        }
    }

    /// `panel compile` — cast a semantic panel's seats and pin the compiled
    /// scenario into a draft study (open-issues §18).
    ///
    /// The whole verb is argument parsing plus ONE call: `SeatCasting.compile`,
    /// which is what the study editor's "save the seats" button and the design
    /// instantiation table both call. Nothing about the compile transform, the
    /// pin pair, or the compiled file's bytes is re-decided here — that is the
    /// defect being fixed, not a style preference.
    private func compilePanel(
        args: [String], flag: (String) -> String?,
        repeatedFlag: (String) -> [String]
    ) throws -> ExperimentCLIResult {
        let usage =
            "usage: panel compile <path-or-name> --experiment <name> "
            + "[--seat <seat>=<agent-artifact-path>]… "
            + "[--model <id>] [--temperature <t>] [--max-tokens <n>] "
            + "[--file-slug <slug>]"
        guard args.count >= 2, !args[1].hasPrefix("--"),
            let experimentName = flag("--experiment")
        else {
            throw ExperimentError(reason: usage)
        }
        let identifier = args[1]
        let record = try Self.resolvePanel(identifier)

        // A BOUND panel carries its own casting inside the file, other studies
        // may pin the same file, and re-casting it here would rewrite an input
        // under them — the same rule the study editor's `legacyBound` form
        // states. Malformed invocation rather than a gate refusal: nothing is
        // wrong with the study or the panel, the caller named the wrong kind of
        // file, and retyping fixes it.
        guard PanelComposition.isSemantic(record.scenario) else {
            throw ExperimentError.malformed(
                "panel '\(record.scenario.name)' is not semantic — it binds its "
                    + "own seats, so its casting lives in the file and cannot be "
                    + "re-cast from a study",
                repair: "steerlab-cli panel list  (semantic panels report "
                    + "\"semantic\": true), or migrate this one in the app's "
                    + "Panels editor, which splits it into a reusable scenario "
                    + "plus the casting it was carrying")
        }

        // The frozen refusal fires HERE, before anything is compiled: the
        // compiled scenario is a FILE, and letting `save` refuse afterwards
        // would leave an orphan in prompts/panels/compiled/ for a study that
        // never adopted it. The prose is byte-identical to
        // `ExperimentStore.updateDraft`'s; only the repair is verb-specific,
        // because the shared one spells `experiment <the verb you just ran>`
        // and this verb's arguments have a different shape.
        let existing = try ExperimentStore.load(name: experimentName)
        guard existing.status == .draft else {
            throw ExperimentError.refusing(
                .statusImmutable,
                "experiment '\(experimentName)' is \(existing.status.rawValue) — "
                    + "duplicate it to iterate",
                repair: "steerlab-cli experiment duplicate \(experimentName) "
                    + "\(experimentName)-v2 && steerlab-cli panel compile "
                    + "\(identifier) --experiment \(experimentName)-v2  "
                    + "(frozen studies are immutable; the duplicate is a draft "
                    + "again)")
        }

        let seatIDs = PanelComposition.seatIDs(record.scenario)
        var occupants: [String: SeatOccupant] = Dictionary(
            uniqueKeysWithValues: seatIDs.map { ($0, SeatOccupant.baseline) })
        for pair in repeatedFlag("--seat") {
            let (seat, path) = try Self.parseSeatPair(pair)
            guard seatIDs.contains(seat) else {
                throw ExperimentError.malformed(
                    "panel '\(record.scenario.name)' has no seat '\(seat)'",
                    repair: "its seats are: \(seatIDs.joined(separator: ", "))  "
                        + "(seats are keyed by the scenario's agent id, not its "
                        + "display name; unnamed seats stay baseline)")
            }
            occupants[seat] = try SeatCasting.occupant(artifactPath: path)
        }
        let assignment = SeatAssignment(seatIDs: seatIDs, occupants: occupants)

        // The three study parameters a compiled panel binds. They DEFAULT from
        // the manifest and, when given, are written to it before the compile —
        // never the other way round. A compiled scenario that disagreed with
        // the manifest it belongs to would be a second answer to "what model
        // does this study run".
        var declaredModel: String?
        if let model = flag("--model") {
            let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                throw ExperimentError.malformed(
                    "--model needs a model id",
                    repair: "steerlab-cli panel compile \(identifier) "
                        + "--experiment \(experimentName) --model <id>")
            }
            declaredModel = trimmed
        }
        var declaredTemperature: Double?
        if let raw = flag("--temperature") {
            guard let value = Double(raw), value >= 0 else {
                throw ExperimentError.malformed(
                    "--temperature must be a non-negative number, not '\(raw)'",
                    repair: "steerlab-cli panel compile \(identifier) "
                        + "--experiment \(experimentName) --temperature 0")
            }
            declaredTemperature = value
        }
        var declaredMaxTokens: Int?
        if let raw = flag("--max-tokens") {
            guard let value = Int(raw), value > 0 else {
                throw ExperimentError.malformed(
                    "--max-tokens must be a positive integer, not '\(raw)'",
                    repair: "steerlab-cli panel compile \(identifier) "
                        + "--experiment \(experimentName) --max-tokens 512")
            }
            declaredMaxTokens = value
        }
        let fileSlug = flag("--file-slug")?
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // `scan` enumerates the library with FileManager, which hands back the
        // filesystem's OWN spelling of a path — `/private/var/…` where the
        // workspace root is `/var/…`, and the same class of divergence
        // anywhere the workspace is reached through a symlink. Relativising
        // that against the root fails the prefix test, and the manifest would
        // then pin an ABSOLUTE provenance path, which does not survive moving
        // the workspace. `ArtifactIdentity.workspaceRelative` is the existing
        // resolve-then-relativise rule; it only ever turns an absolute answer
        // into a relative one.
        let semanticPath = ArtifactIdentity.workspaceRelative(
            FineTuneStore.relativePath(for: record.url))
        var compiled: (path: String, hash: String)?
        // A panel study whose declared kind is still `modelOutput` pins a
        // scenario the run loop never reads — the 2026-08-11 inert-conditions
        // shape, from the other side. Casting a panel INTO a study is the
        // declaration, so the verb makes it rather than refusing to a setter
        // the CLI does not have; the change is reported in `result` and on the
        // human line, never silent.
        let declaresMultiAgent = existing.studyKind != .multiAgent
        let manifest = try ExperimentStore.updateDraft(name: experimentName) {
            manifest in
            if let declaredModel { manifest.modelID = declaredModel }
            if let declaredTemperature { manifest.temperature = declaredTemperature }
            if let declaredMaxTokens { manifest.maxTokens = declaredMaxTokens }
            manifest.studyKind = .multiAgent
            manifest.studyType = StudyIntent.multiAgent.rawValue
            compiled = try SeatCasting.compile(
                assignment, semantic: record.scenario,
                semanticPath: semanticPath, into: &manifest,
                fileSlug: fileSlug?.isEmpty == false ? fileSlug : nil)
        }
        guard let compiled else {
            throw ExperimentError(reason: "panel compile wrote nothing")
        }

        let treated = assignment.ordered.filter { $0 != .baseline }.count
        let line =
            "cast \(seatIDs.count) seat(s) (\(treated) steered) into "
            + "'\(manifest.name)' and pinned \(compiled.path)"
        if declaresMultiAgent {
            sink.out(
                "declared '\(manifest.name)' a multi-agent study "
                    + "(studyKind=multiAgent) — a panel scenario is only read by "
                    + "the multi-agent run path")
        }
        sink.out(line)

        var seatPayload: [JSONValue] = []
        for seat in seatIDs {
            let occupant = assignment[seat] ?? .baseline
            switch occupant {
            case .baseline:
                seatPayload.append(
                    .object([
                        "seat": .string(seat), "occupant": .string("baseline"),
                        "agentName": .null, "artifactPath": .null,
                        "artifactHash": .null,
                    ]))
            case .agent(let name, let path, let hash):
                seatPayload.append(
                    .object([
                        "seat": .string(seat), "occupant": .string("agent"),
                        "agentName": .string(name),
                        "artifactPath": .string(path),
                        // FULL hash: the human line carries none at all, and a
                        // document that elided it would make the casting's
                        // provenance unobtainable from stdout.
                        "artifactHash": .string(hash),
                    ]))
            }
        }
        return ExperimentCLIResult(
            message: line, changed: true,
            payload: [
                "experiment": .string(manifest.name),
                "panel": .string(record.scenario.name),
                "compiledPath": .string(compiled.path),
                "scenarioHash": .string(compiled.hash),
                "semanticScenarioPath": manifest.multiAgentSemanticScenarioPath
                    .map { JSONValue.string($0) } ?? .null,
                "semanticScenarioHash": manifest.multiAgentSemanticScenarioHash
                    .map { JSONValue.string($0) } ?? .null,
                "modelID": .string(manifest.modelID),
                "temperature": .number(manifest.temperature),
                "maxTokens": .number(Double(manifest.maxTokens)),
                "studyKind": .string(manifest.studyKind.rawValue),
                "studyKindDeclared": .bool(declaresMultiAgent),
                "seatCount": .number(Double(seatIDs.count)),
                "treatedSeatCount": .number(Double(treated)),
                "seats": .array(seatPayload),
            ],
            nextAction: .init(verb: "experiment verify \(manifest.name)"))
    }

    // MARK: - vectors

    /// `vectors compare`'s THIRD outcome in human mode. Deliberately 2 and
    /// deliberately not `LifecycleGate…humanExitCode`: this is not a gate
    /// declining a well-formed request, it is "the request could not be
    /// answered", and 2 is what the Python twin has always exited there. The
    /// audit's rule that human codes do not churn cuts toward matching the
    /// engine that already publishes one.
    static let parityCouldNotCompareHumanExitCode: Int32 = 2

    /// The repair for a could-not-compare, per class. Both name BOTH operand
    /// paths and where extraction writes an artifact, because the commonest
    /// instance of this refusal is a caller who does not know an artifact is
    /// two files. Python twin: `cli._vectors`.
    static func parityCouldNotCompareRepair(
        kind: VectorParity.ParityError.Kind, pathA: String, pathB: String
    ) -> String {
        let artifactShape =
            "a vector artifact is <runDir>/<name>.safetensors PLUS its "
            + "<runDir>/<name>.json sidecar, both written by `steerlab-cli "
            + "experiment extract <name>`"
        switch kind {
        case .unreadableArtifact:
            return "check both operand paths and their sidecars — '\(pathA)' "
                + "and '\(pathB)': \(artifactShape)"
        case .incomparableArtifacts:
            return "both artifacts were read and are not comparable at all — "
                + "compare two artifacts extracted from the SAME model "
                + "('\(pathA)' vs '\(pathB)'); \(artifactShape)"
        }
    }

    func runVectorsCommand(_ invocation: ExperimentCLIInvocation) async throws
        -> ExperimentCLIResult
    {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), args.count > index + 1 else {
                return nil
            }
            return args[index + 1]
        }
        let usage = "usage: vectors backfill-norms <runDir/name> "
            + "[--corpus prompts/neutral/corpus.jsonl] [--model <id>] [--redenominate]\n"
            + "       vectors compare <a.safetensors> <b.safetensors> "
            + "[--out OUT] [--threshold T]"

        switch args.first {
        case "compare":
            // Cross-engine parity harness (WS7.3): per-layer cosine + norm ratio
            // between two vector artifacts, KEY-IDENTICAL JSON to the server's
            // `steerlab-server vectors compare`.
            //
            // THREE OUTCOMES, and they are three exit codes on purpose
            // (audit §2.4). A CI script's whole job here is to tell a real
            // parity failure from a broken invocation, and until 2026-08-18 it
            // could not: `pass` was 0, `compared and diverged` was 1/65, and
            // `could not compare at all` was whatever the throw happened to
            // land on — 1 in human mode and `failed`/70 in JSON, the same
            // answer a crash gives. The contract now:
            //
            //   pass                       0                        / 0
            //   compared, diverged         1  (parityThreshold)      / 65
            //   could not compare          2  (notFound)             / 66
            //
            // Human 2 for the third outcome matches the Python twin, which has
            // exited 2 there since the verb existed; this engine aligns to it
            // rather than the other way round, because a published human code
            // is the half CI wrappers already depend on.
            //
            // Hidden-size mismatch is COULD-NOT-COMPARE, not a failed
            // comparison: the artifacts exist and are simply from different
            // models. Layer-count mismatch stays a REPORT — that comparison
            // runs over the intersection and `layerCountMismatch` says so.
            guard args.count >= 3, !args[1].hasPrefix("--"), !args[2].hasPrefix("--") else {
                throw ExperimentError(reason: usage)
            }
            let threshold: Double
            if let raw = flag("--threshold") {
                guard let value = Double(raw) else {
                    throw ExperimentError(reason: "bad --threshold '\(raw)'")
                }
                threshold = value
            } else {
                threshold = VectorParity.defaultThreshold
            }
            let report: VectorParity.Report
            do {
                report = try VectorParity.compareArtifacts(
                    pathA: URL(filePath: args[1]),
                    pathB: URL(filePath: args[2]),
                    threshold: threshold)
            } catch let error as VectorParity.ParityError {
                // The human line is byte-identical to the Python twin's
                // (`vectors compare: <reason>` on stderr), so a researcher
                // reading two engines' logs reads one sentence.
                sink.err("vectors compare: \(error.reason)\n")
                throw ExperimentCLIStop(
                    exitCode: Self.parityCouldNotCompareHumanExitCode,
                    state: .notFound, code: "notFound",
                    reason: "vectors compare could not compare the artifacts: "
                        + error.reason,
                    repairAction: Self.parityCouldNotCompareRepair(
                        kind: error.kind, pathA: args[1], pathB: args[2]),
                    payload: [
                        "operandPaths": .array([
                            .string(args[1]), .string(args[2]),
                        ]),
                        "threshold": .number(threshold),
                    ])
            }
            let text = report.jsonText()
            // The legacy `--json <path>` spelling, kept working for one
            // release; `--out <path>` is the replacement and writes the
            // ENVELOPE, not the bare report.
            if let out = flag("--json") {
                try text.write(to: URL(filePath: out), atomically: true, encoding: .utf8)
                sink.err("wrote \(out)\n")
            }
            // The parity report is the verb's product: on stdout for a human,
            // and inside `result.report` — unchanged, key for key — for a
            // machine, so the cross-engine `diff` the format exists for still
            // works either way.
            if !invocation.json { sink.out(text, terminator: "") }
            let reportValue = Self.jsonValue(fromJSONText: text)
            var payload: [String: JSONValue] = [
                "threshold": .number(threshold),
                "passed": .bool(report.passed),
                "comparedLayerCount": .number(Double(report.comparedLayerCount)),
            ]
            if let minCosine = report.minCosine {
                payload["minCosine"] = .number(minCosine)
            }
            if let reportValue { payload["report"] = reportValue }
            if !report.passed {
                let minCosine = report.minCosine.map { "\($0)" } ?? "none"
                let line =
                    "vectors compare: FAIL — min cosine \(minCosine) < threshold "
                    + "\(threshold) over \(report.comparedLayerCount) compared layer(s)"
                sink.err(line + "\n")
                throw ExperimentCLIStop(
                    exitCode: LifecycleGate.parityThreshold.humanExitCode,
                    state: .refused,
                    code: LifecycleGate.parityThreshold.rawValue,
                    reason: line,
                    repairAction: "steerlab-cli experiment extract <name> on the "
                        + "substrate that will RUN the study (activations do not "
                        + "transfer across engines), or re-run this compare with "
                        + "an explicitly lowered --threshold",
                    payload: payload)
            }
            return ExperimentCLIResult(
                message: "parity OK — min cosine "
                    + (report.minCosine.map { "\($0)" } ?? "none")
                    + " ≥ threshold \(threshold) over "
                    + "\(report.comparedLayerCount) compared layer(s)",
                payload: payload)

        case "backfill-norms":
            // Residual-norm BACKFILL: measure per-layer typical residual norms for
            // an existing artifact (legacy / SAE import / reader-derived) on the
            // pinned neutral corpus and write a NEW artifact into a fresh
            // immutable run directory — the original is never modified.
            guard args.count >= 2, !args[1].hasPrefix("--") else {
                throw ExperimentError(reason: usage)
            }
            let reference = args[1]
            let base: URL =
                if reference.hasPrefix("/") {
                    URL(filePath: reference)
                } else if reference.hasPrefix("runs/") {
                    VectorCatalog.projectRoot.appending(path: reference)
                } else {
                    VectorCatalog.runsDirectory.appending(path: reference)
                }
            let sidecarURL = base.appendingPathExtension("json")
            guard FileManager.default.fileExists(atPath: sidecarURL.path) else {
                throw ExperimentError(
                    reason: "no vector sidecar at \(sidecarURL.path) — pass the "
                        + "artifact base path <runDir>/<name> (no extension)")
            }
            let sidecar = try JSONDecoder().decode(
                SteeringVectorSidecar.self, from: Data(contentsOf: sidecarURL))
            // --model is a convenience for loading; the hard sidecar==loaded guard
            // inside backfillNorms still applies (per-model measurement).
            let modelID = flag("--model") ?? sidecar.modelID
            let corpusURL: URL =
                if let corpus = flag("--corpus") {
                    corpus.hasPrefix("/")
                        ? URL(filePath: corpus)
                        : VectorCatalog.projectRoot.appending(path: corpus)
                } else {
                    VectorCatalog.projectRoot.appending(
                        components: "prompts", "neutral", "corpus.jsonl")
                }
            sink.out("loading \(modelID)…")
            let container = try await SteeredContainerLoader.load(modelID: modelID)
            let runDirectory = try VectorCatalog.makeUniqueRunDirectory(
                slug: "norms-\(sidecar.concept)")
            let artifact = try await NormBackfill.backfillNorms(
                container: container, modelID: modelID, artifact: base,
                corpusURL: corpusURL, runDirectory: runDirectory,
                redenominate: args.contains("--redenominate"))
            sink.out("backfilled norms → \(artifact.path).{safetensors,json}")
            sink.out("source artifact untouched: \(base.path)")
            return ExperimentCLIResult(
                message: "backfilled norms → \(artifact.path)", changed: true,
                payload: [
                    "artifact": .string(artifact.path),
                    "sourceArtifact": .string(base.path),
                    "runDirectory": .string(runDirectory.path),
                    "modelID": .string(modelID),
                    "corpus": .string(corpusURL.path),
                    "redenominated": .bool(args.contains("--redenominate")),
                ])

        default:
            throw ExperimentError(reason: usage)
        }
    }

    // MARK: - remote

    func runRemoteCommand(_ invocation: ExperimentCLIInvocation) async throws
        -> ExperimentCLIResult
    {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), args.count > index + 1 else {
                return nil
            }
            return args[index + 1]
        }
        guard let verb = args.first else {
            throw ExperimentError(
                reason: "usage: remote capabilities|package|upload|submit-bundle|jobs|logs|cancel|fetch|import|import-chain|variants|chat (--site <id> | --url <server>)")
        }
        // Site-aware resolution (CLUSTER-CLI-LIFECYCLE-PLAN §5.4): `--site` reads
        // the endpoint from the shared registry and the bearer token from the
        // Keychain, so neither ever appears in argv, shell history, or agent
        // context. `--url`/`--token` remain the compatibility path for unmanaged
        // servers. They are mutually exclusive — silently preferring one would make
        // a typo look like a working connection to the wrong server.
        let url: URL
        let token: String?
        var siteSummary: String?
        switch try ClusterRemoteSiteResolver.choose(site: flag("--site"), url: flag("--url")) {
        case .site(let reference):
            let resolved = try await ClusterRemoteSiteResolver.resolve(reference: reference)
            url = resolved.baseURL
            token = resolved.token
            // Presence and provenance only — never the value.
            sink.err("remote: \(resolved.redactedSummary)\n")
            siteSummary = resolved.redactedSummary
        case .url(let explicit):
            // 8080 is the port EVERY server surface serves (both `serve`
            // verbs, the app's default connection URL, the tunnel
            // instructions); the historical 8000 default meant the default
            // client and the default server did not meet.
            guard let parsed = URL(string: explicit ?? "http://127.0.0.1:8080") else {
                throw ExperimentError(reason: "bad --url '\(explicit ?? "")'")
            }
            url = parsed
            token = flag("--token") ?? ProcessInfo.processInfo.environment["STEERLAB_AUTH_TOKEN"]
        }
        let client = ClusterClient(profile: ClusterConnectionProfile(baseURL: url), token: token)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        /// The five verbs that wrote raw, unversioned JSON to stdout. Human
        /// mode still gets exactly those bytes; JSON mode gets the same
        /// object inside `result.response`, under a versioned envelope.
        func respond<T: Encodable>(_ value: T, message: String) throws
            -> ExperimentCLIResult
        {
            let text = String(
                data: try encoder.encode(value), encoding: .utf8) ?? "{}"
            if !invocation.json { sink.out(text) }
            var payload: [String: JSONValue] = ["endpoint": .string(url.absoluteString)]
            if let siteSummary { payload["site"] = .string(siteSummary) }
            if let decoded = Self.jsonValue(fromJSONText: text) {
                payload["response"] = decoded
            }
            return ExperimentCLIResult(message: message, payload: payload)
        }

        switch verb {
        case "capabilities":
            return try respond(
                try await client.capabilities(), message: "server capabilities read")
        case "upload":
            guard args.count >= 2 else { throw ExperimentError(reason: "usage: remote upload <bundle> [--url URL]") }
            return try respond(
                try await client.uploadBundle(URL(filePath: args[1])),
                message: "uploaded \(args[1])")
        case "submit-bundle":
            guard let path = flag("--bundle") ?? (args.count >= 2 ? args[1] : nil) else {
                throw ExperimentError(reason: "usage: remote submit-bundle <server-bundle-path> [--verb verify] [--executor local|slurm] [--dry-run]")
            }
            let submission = try await client.submitBundle(
                path: path,
                verb: flag("--verb") ?? "run",
                executor: flag("--executor") ?? "local",
                dryRun: args.contains("--dry-run"),
                resources: [
                    "gres": flag("--gres") ?? "",
                    "walltime": flag("--walltime") ?? "",
                ].filter { !$0.value.isEmpty })
            return try respond(submission, message: "submitted \(path)")
        case "jobs":
            let jobs = try await client.jobs()
            return try respond(jobs, message: "\(jobs.count) job(s)")
        case "logs":
            guard args.count >= 2 else { throw ExperimentError(reason: "usage: remote logs <job-id>") }
            try await client.streamJobLog(jobID: args[1]) { line in self.sink.out(line) }
            return ExperimentCLIResult(
                message: "streamed log for job \(args[1])",
                payload: ["jobID": .string(args[1])])
        case "cancel":
            guard args.count >= 2 else { throw ExperimentError(reason: "usage: remote cancel <job-id>") }
            try await client.cancelJob(args[1])
            sink.out("cancel requested")
            return ExperimentCLIResult(
                message: "cancel requested for job \(args[1])", changed: true,
                payload: ["jobID": .string(args[1])])
        case "fetch":
            guard let path = flag("--path") ?? (args.count >= 2 ? args[1] : nil) else {
                throw ExperimentError(reason: "usage: remote fetch <artifact-path> [--out dir]")
            }
            let out = URL(filePath: flag("--out") ?? ".")
            let local = try await client.downloadArtifact(path: path, to: out)
            sink.out(local.path)
            return ExperimentCLIResult(
                message: "fetched \(path)", changed: true,
                payload: [
                    "remotePath": .string(path), "localPath": .string(local.path),
                ])
        case "package":
            // Build a hash-pinned run bundle from a local experiment, headless.
            guard args.count >= 2 else {
                throw ExperimentError(reason: "usage: remote package <experiment-name>")
            }
            let manifest = try ExperimentStore.load(name: args[1])
            let bundle = try RunBundlePackager.packageExperiment(manifest)
            sink.out(bundle.path)
            return ExperimentCLIResult(
                message: "packaged '\(manifest.name)' → \(bundle.path)", changed: true,
                payload: [
                    "experiment": .string(manifest.name),
                    "bundle": .string(bundle.path),
                    "experimentHash": .string(ExperimentStore.manifestHash(manifest)),
                ])
        case "import":
            // Download a server evidence bundle and import it into local runs/ with
            // hash verification — the headless half of "run on remote, archive local".
            guard let path = flag("--path") ?? (args.count >= 2 ? args[1] : nil) else {
                throw ExperimentError(reason: "usage: remote import <server-evidence-path> [--out dir] [--sha256 hex]")
            }
            let out = URL(filePath: flag("--out") ?? ".steerlab-downloads")
            let local = try await client.downloadArtifact(path: path, to: out)
            let imported = try EvidenceBundleImporter.importEvidenceBundle(
                local, expectedSHA256: flag("--sha256"))
            // The adoption reconciliation every import path must run
            // (2026-08-06, a replication-run incident): this raw verb was the one
            // importer that skipped it, and a server-auto-pinned revision then
            // made the local analyze refuse on an epoch diff the researcher
            // never authored.
            let adoption = EvidenceRevisionAdoption.adoptModelRevision(
                fromImportedRun: imported)
            var advisories: [SteerLabCLIEnvelope.Advisory] = []
            if let notice = EvidenceRevisionAdoption.notice(for: adoption) {
                sink.err(
                    (notice.isWarning ? "warning: " : "") + notice.message + "\n")
                advisories.append(
                    .init(
                        code: (notice.isWarning
                            ? CLIAdvisory.revisionAdoptionWarning
                            : CLIAdvisory.revisionAdoption).rawValue,
                        detail: notice.message))
            }
            sink.out(imported.path)
            return ExperimentCLIResult(
                message: "imported \(path) → \(imported.path)", changed: true,
                payload: [
                    "remotePath": .string(path),
                    "importedRun": .string(imported.path),
                ],
                advisories: advisories)
        case "import-chain":
            // Whole-chain twin of `import` (2026-08-12): resolve the pipeline
            // run (ledger rule — completed disposition only when named by
            // experiment), then package + download + import the pipeline dir and
            // each stage dir skip-if-present, with revision adoption per
            // imported directory. All logic lives in EvidenceChainImport
            // (ExperimentKit) where tests reach it; this case parses and prints.
            guard args.count >= 2, !args[1].hasPrefix("--") else {
                throw ExperimentError(
                    reason: "usage: remote import-chain <pipeline-run-id-or-experiment-name> (--site <id> | --url <server> [--token <t>])")
            }
            let reference = args[1]
            var rows: [ClusterClient.PipelineRunSummary]
            do {
                rows = try await client.allPipelineRuns()
            } catch {
                // Older servers have no cross-experiment `GET /api/pipelines`;
                // the per-experiment route still resolves an experiment name
                // (its rows lack the `experiment` stamp — restore it so the
                // resolver's sibling filter sees them).
                rows = try await client.pipelineRuns(experiment: reference)
                for index in rows.indices where rows[index].experiment == nil {
                    rows[index].experiment = reference
                }
            }
            let chain = try EvidenceChainImport.resolveChain(
                argument: reference, rows: rows)
            let reports = await EvidenceChainImport.importChain(
                chain,
                engine: EvidenceChainImport.liveEngine(
                    client: client, workspaceRoot: VectorCatalog.projectRoot))
            let lines = EvidenceChainImport.summaryLines(reports)
            for line in lines { sink.out(line) }
            let payload: [String: JSONValue] = [
                "reference": .string(reference),
                "importedCount": .number(Double(reports.count)),
                "failedCount": .number(
                    Double(reports.filter(\.isFailure).count)),
                "summary": .array(lines.map { .string($0) }),
            ]
            let unreachable = reports.contains {
                if case .endpointUnreachable = $0.outcome { return true }
                return false
            }
            if reports.contains(where: \.isFailure) {
                throw ExperimentCLIStop(
                    exitCode: 1, state: .failed,
                    code: unreachable ? "chainImportEndpointUnreachable"
                        : "chainImportFailed",
                    reason: unreachable
                        ? "the server did not answer — the chain was refused "
                            + "before anything was written locally"
                        : "\(reports.filter(\.isFailure).count) of "
                            + "\(reports.count) chain stage(s) failed to import",
                    repairAction: unreachable
                        ? "`steerlab-cli cluster tunnel open`, then re-run "
                            + "`remote import-chain \(reference)`"
                        : "re-run `remote import-chain \(reference)` — "
                            + "already-imported stages are skipped",
                    payload: payload)
            }
            return ExperimentCLIResult(
                message: "imported \(reports.count) chain stage(s) for "
                    + "'\(reference)'", changed: true, payload: payload)
        case "variants":
            let variants = try await client.variants()
            return try respond(variants, message: "server variant library read")
        case "chat":
            guard let variant = flag("--variant"), let prompt = flag("--prompt") else {
                throw ExperimentError(reason: "usage: remote chat --variant <server-path> --prompt <text>")
            }
            try await client.streamVariantChat(
                variantPath: variant,
                variantHash: flag("--hash"),
                messages: [ChatWireMessage(role: "user", content: prompt)],
                maxTokens: Int(flag("--max-tokens") ?? "512") ?? 512,
                temperature: flag("--temperature").flatMap(Double.init),
                promptMode: flag("--prompt-mode") ?? "chatAssistant",
                systemPrompt: flag("--system"),
                stripInterventions: args.contains("--strip")
            ) { chunk in
                self.sink.outRaw(chunk)
            }
            sink.out("")
            return ExperimentCLIResult(
                message: "streamed one chat turn through \(variant)",
                payload: ["variant": .string(variant)])
        default:
            throw ExperimentError(
                reason: "remote verbs: capabilities | package | upload | submit-bundle | jobs | logs | cancel | fetch | import | import-chain | variants | chat")
        }
    }

    // MARK: - experiment

    func runExperimentCommand(_ invocation: ExperimentCLIInvocation) async throws
        -> ExperimentCLIResult
    {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), args.count > index + 1 else {
                return nil
            }
            return args[index + 1]
        }

        switch args.first {
        case "list":
            let manifests = ExperimentStore.list()
            if manifests.isEmpty { sink.out("no experiments") }
            for manifest in manifests {
                sink.out(
                    "\(manifest.name)  [\(manifest.status.rawValue)]  "
                        + "model=\(manifest.modelID)  "
                        + "concepts=\(manifest.concepts.map(\.name).joined(separator: "+"))  "
                        + "conditions=\(manifest.conditions.count)")
            }
            return ExperimentCLIResult(
                message: manifests.isEmpty
                    ? "no experiments" : "\(manifests.count) experiment(s)",
                payload: [
                    "count": .number(Double(manifests.count)),
                    "experiments": .array(
                        manifests.map { manifest in
                            .object([
                                "name": .string(manifest.name),
                                "status": .string(manifest.status.rawValue),
                                "modelID": .string(manifest.modelID),
                                "modelRevision": manifest.modelRevision.map {
                                    JSONValue.string($0)
                                } ?? .null,
                                "concepts": .array(
                                    manifest.concepts.map { .string($0.name) }),
                                "conditionCount": .number(
                                    Double(manifest.conditions.count)),
                                "freezeHash": manifest.freezeHash.map {
                                    JSONValue.string($0)
                                } ?? .null,
                            ])
                        }),
                ])

        case "create":
            guard args.count >= 2, let model = flag("--model") else {
                throw ExperimentError(
                    reason: "usage: experiment create <name> --model <id> "
                        + "[--revision <commit>] [--description …]")
            }
            let manifest = try ExperimentStore.create(
                name: args[1], description: flag("--description") ?? "",
                modelID: model, modelRevision: flag("--revision"))
            let line =
                "created draft '\(manifest.name)' (model \(manifest.modelID)"
                + (manifest.modelRevision.map { " @ \($0.prefix(12))…" } ?? "")
                + ")"
            sink.out(line)
            return ExperimentCLIResult(
                message: line, changed: true,
                payload: [
                    "name": .string(manifest.name),
                    "status": .string(manifest.status.rawValue),
                    "modelID": .string(manifest.modelID),
                    // FULL revision: the human line elides it, a document
                    // must not.
                    "modelRevision": manifest.modelRevision.map {
                        JSONValue.string($0)
                    } ?? .null,
                    "path": .string(ExperimentStore.directory
                        .appending(component: manifest.name).path),
                ],
                nextAction: .init(
                    verb: "experiment attach \(manifest.name) <concept>"))

        case "attach":
            guard args.count >= 3 else {
                throw ExperimentError(
                    reason: "usage: experiment attach <name> <concept>… "
                        + "[--method meanDifference|lat|emotionGrandMean|designatedReference] "
                        + "[--pool-from K] [--reference <stories-concept>] "
                        + "[--corpus a,b,c (emotionGrandMean: extra corpus members)] "
                        + "[--project-neutral K (legacy; verification-blocked)]")
            }
            var manifest = try ExperimentStore.load(name: args[1])
            let neutralPCs = flag("--project-neutral").flatMap(Int.init)
            if let neutralPCs, neutralPCs > 0 {
                sink.out(
                    "warning: --project-neutral uses legacy fixed-k PCs from pooled neutral "
                        + "texts; verified/frozen experiments reject it until the "
                        + "token-position neutral activation bank exists")
            }
            var options = ExtractionOptions()
            options.neutralPCCount = neutralPCs
            if let methodFlag = flag("--method") {
                // Recipe methods only: pinnedArtifact pins bytes (attach an
                // artifact instead) and optvec is server-trained — neither is an
                // extraction recipe this engine can run.
                guard let method = ExtractionMethod(rawValue: methodFlag),
                    method.isRecipeMethod
                else {
                    throw ExperimentError(
                        reason: "--method must be one of: "
                            + ExtractionMethod.allCases.filter(\.isRecipeMethod)
                                .map(\.rawValue).joined(separator: " | "))
                }
                options.method = method
            }
            if let poolFlag = flag("--pool-from") {
                guard let token = Int(poolFlag), token >= 0 else {
                    throw ExperimentError(reason: "--pool-from expects a token index ≥ 0")
                }
                options.readingPosition = .meanFromToken(token)
            }
            let flagValues = ["--project-neutral", "--method", "--pool-from", "--corpus",
                              "--reference"]
                .compactMap(flag)
            let concepts = args.dropFirst(2).filter { !$0.hasPrefix("--") }
                .filter { !flagValues.contains($0) }
            // REFUSAL BEFORE SUCCESS (audit §8, P1-6): the per-concept lines
            // are buffered and only flushed once every concept resolved. A
            // failed attach saved nothing — `save` is the last statement —
            // so printing a success line for it was stdout lying about a
            // mutation that never happened.
            var pinnedLines: [String] = []
            var pinned: [JSONValue] = []
            if options.method == .emotionGrandMean {
                let corpusMembers =
                    flag("--corpus")?
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty } ?? []
                try ExperimentStore.attachGrandMeanConcepts(
                    Array(concepts),
                    corpusConcepts: corpusMembers,
                    poolFromToken: flag("--pool-from").flatMap(Int.init),
                    into: &manifest)
                for concept in concepts {
                    let hash = manifest.grandMeanCorpus?.hashes[concept] ?? ""
                    pinnedLines.append(
                        "pinned \(concept) @ \(hash.prefix(12))… (emotionGrandMean, "
                            + "corpus of \(manifest.grandMeanCorpus?.concepts.count ?? 0) "
                            + "concept(s))")
                    pinned.append(
                        .object([
                            "concept": .string(concept),
                            "stimulusSetHash": .string(hash),
                            "method": .string(ExtractionMethod.emotionGrandMean.rawValue),
                        ]))
                }
            } else if options.method == .designatedReference {
                try ExperimentStore.save(manifest)
                for concept in concepts {
                    let updated = try ExperimentStore.attachConcept(
                        concept, method: .designatedReference,
                        poolFromToken: flag("--pool-from").flatMap(Int.init),
                        reference: flag("--reference"),
                        experimentName: manifest.name)
                    if let ref = updated.concepts.first(where: { $0.name == concept }) {
                        pinnedLines.append(
                            "pinned \(concept) @ \(ref.stimulusSetHash.prefix(12))… "
                                + "(designatedReference − '\(ref.designatedReference?.name ?? "?")', "
                                + "\(ref.options.readingPosition.label))")
                        pinned.append(
                            .object([
                                "concept": .string(concept),
                                "stimulusSetHash": .string(ref.stimulusSetHash),
                                "method": .string(
                                    ExtractionMethod.designatedReference.rawValue),
                                "reference": .string(
                                    ref.designatedReference?.name ?? ""),
                                "readingPosition": .string(
                                    ref.options.readingPosition.label),
                            ]))
                    }
                }
                manifest = try ExperimentStore.load(name: manifest.name)
            } else {
                for concept in concepts {
                    let directory = VectorCatalog.conceptsDirectory.appending(component: concept)
                    let stimuli = try StimulusSet(directory: directory)
                    let ref = ExperimentStore.makeConceptRef(
                        name: concept, stimulusSetHash: stimuli.hash, options: options)
                    manifest.concepts.removeAll { $0.name == concept }
                    manifest.concepts.append(ref)
                    pinnedLines.append(
                        "pinned \(concept) @ \(stimuli.hash.prefix(12))… "
                            + "(\(stimuli.positive.count)+\(stimuli.negative.count) stimuli, "
                            + "\(options.method.rawValue), \(options.readingPosition.label))"
                            + ((neutralPCs ?? 0) > 0
                                ? " [legacy draft-only: project top-\(neutralPCs!) neutral PCs]" : ""))
                    pinned.append(
                        .object([
                            "concept": .string(concept),
                            "stimulusSetHash": .string(stimuli.hash),
                            "positiveCount": .number(Double(stimuli.positive.count)),
                            "negativeCount": .number(Double(stimuli.negative.count)),
                            "method": .string(options.method.rawValue),
                            "readingPosition": .string(
                                options.readingPosition.label),
                        ]))
                }
            }
            // The corpus is a pinned input whenever it exists: it denominates
            // norm-unit alphas (and hosts the projection PCs when requested).
            var payload: [String: JSONValue] = [
                "experiment": .string(manifest.name),
                "pinned": .array(pinned),
            ]
            if let hash = ExperimentStore.pinNeutralCorpus(into: &manifest) {
                pinnedLines.append(
                    "pinned neutral corpus @ \(hash.prefix(12))… (norm denominator)")
                payload["neutralCorpusHash"] = .string(hash)
            } else if (neutralPCs ?? 0) > 0 {
                throw ExperimentError(
                    reason: "--project-neutral requires prompts/neutral/corpus.jsonl")
            }
            try ExperimentStore.save(manifest)
            for line in pinnedLines { sink.out(line) }
            return ExperimentCLIResult(
                message: pinnedLines.first ?? "nothing to pin", changed: true,
                payload: payload,
                nextAction: .init(verb: "experiment validate \(manifest.name)"))

        // MARK: The three headless authoring verbs (WP0 step 5½, P0-3)
        //
        // Before these, the CLI could pin CONCEPTS and nothing else: no task
        // prompts, no rubric, no arms. A headless agent could therefore
        // build a study the firewall accepts and that measures nothing — and
        // every refusal that named the missing piece named a remedy only the
        // Studies panel could perform ("enter draft rubric text"). Each verb
        // goes through the store setter the panel goes through, so a
        // CLI-authored and a GUI-authored manifest are indistinguishable
        // byte for byte, and each is draft-only through `updateDraft`'s one
        // immutability line.

        case "pin-prompts":
            // The measured task: the prompts every arm answers, pinned by
            // path + SHA-256 over the raw bytes. Parsed at pin time with the
            // RUN LOOP's own parser, so a file the run would refuse is
            // refused here instead of at generation time.
            guard args.count >= 3 else {
                throw ExperimentError(
                    reason: "usage: experiment pin-prompts <name> "
                        + "<prompts/…/file.jsonl>  (\"\" clears the pin)")
            }
            let pinned = try ExperimentStore.setTaskPromptsFile(
                args[2], experimentName: args[1])
            guard let file = pinned.taskPromptsFile,
                let hash = pinned.taskPromptsHash
            else {
                let line = "cleared the task-prompts pin on '\(pinned.name)'"
                sink.out(line)
                return ExperimentCLIResult(
                    message: line, changed: true,
                    payload: [
                        "experiment": .string(pinned.name),
                        "taskPromptsFile": .null,
                        "taskPromptsHash": .null,
                    ])
            }
            let itemCount = Self.taskPromptCount(file: file)
            let line =
                "pinned task prompts \(file) @ \(hash.prefix(12))…"
                + (itemCount.map { " (\($0) item(s))" } ?? "")
            sink.out(line)
            var promptPayload: [String: JSONValue] = [
                "experiment": .string(pinned.name),
                // FULL hash: the human line elides it, a document must not.
                "taskPromptsFile": .string(file),
                "taskPromptsHash": .string(hash),
            ]
            if let itemCount { promptPayload["itemCount"] = .number(Double(itemCount)) }
            // P13's root cause, said at the moment it becomes true: items that
            // carry options/target engage the deterministic answer-logprob
            // instrument ONLY when the manifest DECLARES it. The declaration
            // is provenance and is never inferred from the data — correctly —
            // but nothing used to say so, so a pinned choice task was silently
            // measured as sampled prose.
            var promptAdvisories: [SteerLabCLIEnvelope.Advisory] = []
            let counts = ExperimentTasks.choiceShapedItemCount(pinned)
            if let counts, counts.choice > 0,
                !ExecutionPlan.resolve(instruments: pinned.outcomeInstruments)
                    .scoresDirectly
            {
                promptPayload["choiceShapedItems"] = .number(Double(counts.choice))
                promptAdvisories.append(
                    .init(
                        CLIAdvisory.choiceItemsWithoutInstrument,
                        // The responseFormat clause was WRONG until gate-5 dry
                        // run #2 (P3): it said items "must also declare
                        // responseFormat: label", and an agent that believed
                        // it re-authored a working file. The dispatch
                        // (`ExperimentTasks`, `tasks.py`) reads any item with
                        // a non-empty `options`; an ABSENT responseFormat is
                        // deliberately permissive (legacy files predate the
                        // field). `label` is only load-bearing in the two
                        // negative directions stated here.
                        "\(counts.choice) of \(counts.total) pinned item(s) "
                            + "carry options, but this study declares no "
                            + "direct-scoring instrument, so the run will sample "
                            + "PROSE and parse it rather than reading the "
                            + "answer-token logprobs. Pinned fields are "
                            + "preserved; they never enable measurement. "
                            + "Declare it: steerlab-cli experiment "
                            + "set-instruments \(pinned.name) answerTokenLogprob "
                            + "(an item needs `options`; `responseFormat` is "
                            + "OPTIONAL and absence is fine — but an item that "
                            + "declares \"json\" or \"freeText\" is refused at "
                            + "run start, and if you declare an "
                            + "outcomeInstrumentScope only rows whose declared "
                            + "format it lists are measured)"))
            }
            return ExperimentCLIResult(
                message: line, changed: true, payload: promptPayload,
                advisories: promptAdvisories,
                nextAction: .init(
                    verb: "experiment declare-condition \(pinned.name) "
                        + "<condition> --slots <concept>:<layer>:<alpha>"))

        case "pin-rubric":
            // The judging instrument: a versioned rubric FILE under
            // prompts/rubrics/ (inline text is draft-only and cannot freeze),
            // plus the judge panel, plus the explicit `evaluation`
            // declaration the pin pair implies.
            guard args.count >= 3 else {
                throw ExperimentError(
                    reason: "usage: experiment pin-rubric <name> "
                        + "<prompts/rubrics/file.md> "
                        + "[--judges <name>:<kind>[:<model>[:<provider>]][,…]]  "
                        + "(kinds: \(ExperimentStore.knownJudgeKinds.joined(separator: " | ")); "
                        + "\"\" clears the pin)")
            }
            let judges = try flag("--judges").map(Self.parseJudges)
            let rubric = try ExperimentStore.setJudgeRubric(
                file: args[2], judges: judges, experimentName: args[1])
            var rubricLines: [String] = []
            if let file = rubric.judgeRubricFile, let hash = rubric.judgeRubricHash {
                rubricLines.append(
                    "pinned judge rubric \(file) @ \(hash.prefix(12))…")
            } else {
                rubricLines.append("cleared the judge-rubric pin on '\(rubric.name)'")
            }
            let panel = rubric.judges ?? []
            if !panel.isEmpty {
                rubricLines.append(
                    "judges: "
                        + panel.map { "\($0.name) (\($0.kind))" }
                            .joined(separator: ", "))
            }
            // Freeze wants ≥2 DISTINCT judges for agreement statistics; say
            // so now rather than at the gate, as an advisory (never an exit
            // code — a one-judge draft is a legal draft).
            var rubricAdvisories: [SteerLabCLIEnvelope.Advisory] = []
            if !panel.isEmpty, panel.count < 2 {
                rubricAdvisories.append(
                    .init(
                        code: CLIAdvisory.judgePanelTooSmall.rawValue,
                        detail: "\(panel.count) judge pinned; freeze's "
                            + "judgeValidity gate requires at least 2 so the "
                            + "report can carry agreement statistics"))
            }
            for line in rubricLines { sink.out(line) }
            return ExperimentCLIResult(
                message: rubricLines[0], changed: true,
                payload: [
                    "experiment": .string(rubric.name),
                    "judgeRubricFile": rubric.judgeRubricFile.map {
                        JSONValue.string($0)
                    } ?? .null,
                    "judgeRubricHash": rubric.judgeRubricHash.map {
                        JSONValue.string($0)
                    } ?? .null,
                    "judges": .array(
                        panel.map { judge in
                            var fields: [String: JSONValue] = [
                                "name": .string(judge.name),
                                "kind": .string(judge.kind),
                            ]
                            if let model = judge.model { fields["model"] = .string(model) }
                            if let provider = judge.provider {
                                fields["provider"] = .string(provider)
                            }
                            return .object(fields)
                        }),
                    "evaluationKind": rubric.evaluation.map {
                        JSONValue.string($0.kind.rawValue)
                    } ?? .null,
                ],
                advisories: rubricAdvisories)

        case "declare-condition":
            // THE arm. A concept study with no injection condition, no agent
            // and no SAE latent runs the implicit baseline alone and
            // measures nothing (`noMeasuredConditionsProblem`) — and until
            // this verb existed the only headless way to mint one was
            // `promote`, which requires a sweep, which requires a model.
            //
            // The slot spelling is the manifest's own: concept, layer,
            // alpha (α when steering, λ when ablating), mode. A multi-slot
            // condition IS the linear mix h + Σ αᵢ·vᵢ, hashed as one
            // condition like any other.
            guard args.count >= 3 else {
                throw ExperimentError(
                    reason: "usage: experiment declare-condition <name> <condition> "
                        + "--slots <concept>:<layer>:<alpha>[:add|ablate][,…] "
                        + "--alpha-units norm|raw [--band-width K] "
                        + "[--control randomMatchedNorm|randomDirectionAblation]\n"
                        + "       experiment declare-condition <name> <condition> "
                        + "--baseline --alpha-units norm|raw  "
                        + "(an explicit no-intervention arm)")
            }
            let conditionName = args[2].trimmingCharacters(in: .whitespacesAndNewlines)
            let isBaseline = args.contains("--baseline")
            let slotSpec = flag("--slots")
            if isBaseline, slotSpec != nil {
                throw ExperimentError(
                    reason: "--baseline and --slots are exclusive — a baseline "
                        + "arm is the condition with NO slots")
            }
            guard isBaseline || slotSpec != nil else {
                throw ExperimentError(
                    reason: "declare-condition needs --slots "
                        + "<concept>:<layer>:<alpha>[:add|ablate][,…], or "
                        + "--baseline for an explicit no-intervention arm")
            }
            let slots = try slotSpec.map(Self.parseSlots) ?? []
            var bandWidth = 1
            if let raw = flag("--band-width") {
                guard let value = Int(raw), value >= 1 else {
                    throw ExperimentError(
                        reason: "--band-width expects an integer ≥ 1 — got '\(raw)'")
                }
                bandWidth = value
            }
            // REQUIRED, as of Phase 1a of the portability program (Phase-0 gap
            // G6, docs/PORTABILITY-CONTRACTS.md). It used to default to `norm`
            // here and to `raw` on the server's `_condition_entry`, so the
            // same undeclared arm authored a different study depending on
            // which engine served it — and α units are dose semantics, not a
            // display setting. Neither engine guesses now: both refuse a NEW
            // declaration that does not say, and both name the same two
            // spellings of the repair. Baselines are not exempt: a slot-less
            // arm carries no α, but the key it stamps is still part of the
            // document the other engine reads, and two engines stamping
            // different values for the same call is the gap itself.
            let alphaInNormUnits: Bool
            switch flag("--alpha-units") {
            case "norm": alphaInNormUnits = true
            case "raw": alphaInNormUnits = false
            case nil:
                throw ExperimentError.malformed(
                    "declare-condition needs --alpha-units norm|raw — α units "
                        + "are dose semantics, so this engine no longer "
                        + "supplies a default the server engine does not share",
                    repair: ExperimentManifest.alphaUnitsRepairAction)
            case let raw?:
                throw ExperimentError.malformed(
                    "--alpha-units must be norm | raw — got '\(raw)'. "
                        + "norm denominates α by the residual-stream norm at "
                        + "that layer on the pinned neutral corpus, which is "
                        + "what makes α comparable across concepts",
                    repair: "re-run with --alpha-units norm (the project "
                        + "convention) or --alpha-units raw")
            }
            var controlType: String?
            if let raw = flag("--control") {
                guard Self.knownControlTypes.contains(raw) else {
                    throw ExperimentError(
                        reason: "unknown --control '\(raw)' — known: "
                            + Self.knownControlTypes.joined(separator: " | "))
                }
                guard !slots.isEmpty else {
                    throw ExperimentError(
                        reason: "--control needs slots — a control cell "
                            + "substitutes a random direction for the concept's "
                            + "in the SAME slots")
                }
                controlType = raw
            }
            let declared = try ExperimentStore.upsertCondition(
                .init(
                    name: conditionName, slots: slots, bandWidth: bandWidth,
                    alphaInNormUnits: alphaInNormUnits, controlType: controlType),
                experimentName: args[1])
            let described = slots
                .map { "\($0.concept)@L\($0.layer) \($0.effectiveMode.rawValue) \($0.alpha)" }
                .joined(separator: " + ")
            let conditionLine =
                "declared '\(conditionName)' — "
                + (slots.isEmpty ? "no-intervention baseline" : described)
                + (controlType.map { " [\($0)]" } ?? "")
                + " (\(declared.conditions.count) condition(s))"
            sink.out(conditionLine)
            return ExperimentCLIResult(
                message: conditionLine, changed: true,
                payload: [
                    "experiment": .string(declared.name),
                    "condition": .object([
                        "name": .string(conditionName),
                        "bandWidth": .number(Double(bandWidth)),
                        "alphaInNormUnits": .bool(alphaInNormUnits),
                        "controlType": controlType.map { JSONValue.string($0) } ?? .null,
                        "slots": .array(
                            slots.map { slot in
                                .object([
                                    "concept": .string(slot.concept),
                                    "layer": .number(Double(slot.layer)),
                                    "alpha": .number(slot.alpha),
                                    "mode": .string(slot.effectiveMode.rawValue),
                                ])
                            }),
                    ]),
                    "conditionCount": .number(Double(declared.conditions.count)),
                    "conditions": .array(
                        declared.conditions.map { .string($0.name) }),
                ],
                nextAction: .init(verb: "experiment freeze \(declared.name)"))

        case "verify":
            guard args.count >= 2 else {
                throw ExperimentError(reason: "usage: experiment verify <name>")
            }
            let manifest = try ExperimentStore.load(name: args[1])
            let violations = ExperimentStore.verify(manifest)
            if violations.isEmpty {
                sink.out("OK [\(manifest.status.rawValue)] — all pinned inputs verified")
                return ExperimentCLIResult(
                    message: "OK [\(manifest.status.rawValue)] — all pinned "
                        + "inputs verified",
                    payload: [
                        "experiment": .string(manifest.name),
                        "status": .string(manifest.status.rawValue),
                        "verified": .bool(true),
                        "violations": .array([]),
                        "experimentHash": .string(
                            ExperimentStore.manifestHash(manifest)),
                    ])
            }
            // STREAM FIX: violations are a FAILURE and belong on stderr. They
            // went to stdout, which the audit records as "prose, stdout even
            // on failure" (§2.1) — a caller redirecting stdout to a file got
            // the bad news in the file and nothing on the terminal.
            for violation in violations { sink.err("VIOLATION: \(violation)\n") }
            throw ExperimentCLIStop(
                exitCode: LifecycleGate.pinDrift.humanExitCode, state: .refused,
                code: LifecycleGate.pinDrift.rawValue,
                reason: "\(violations.count) pinned input(s) of "
                    + "'\(manifest.name)' no longer match their hashes",
                // The same repair `loadVerified` hands every other verb — an
                // appeared-after-attach violation is repaired by ONE attach,
                // and nothing on the surface used to say so (§9, P5).
                repairAction: ExperimentTasks.pinDriftRepair(
                    name: manifest.name, violations: violations),
                payload: [
                    "experiment": .string(manifest.name),
                    "status": .string(manifest.status.rawValue),
                    "verified": .bool(false),
                    "violations": .array(violations.map { .string($0) }),
                ])

        case "freeze":
            guard args.count >= 2 else {
                throw ExperimentError(
                    reason: "usage: experiment freeze <name> [--force] "
                        + "[--run-substrate local|server]")
            }
            // Mac-authority mode (2026-07-21): --run-substrate server matches
            // the validate/battery-evidence gate against evidence produced on
            // (or imported from) the server engine — the substrate the study's
            // measured runs will execute on — instead of this engine.
            var runSubstrate = ExperimentStore.evidenceSubstrate
            if let flagIndex = args.firstIndex(of: "--run-substrate") {
                guard args.count > flagIndex + 1 else {
                    throw ExperimentError(
                        reason: "--run-substrate needs a value: local | server")
                }
                switch args[flagIndex + 1] {
                case "local", ExperimentStore.evidenceSubstrate:
                    runSubstrate = ExperimentStore.evidenceSubstrate
                case "server", WorkspaceScoping.serverSubstrate:
                    runSubstrate = WorkspaceScoping.serverSubstrate
                default:
                    throw ExperimentError(
                        reason: "unknown --run-substrate '\(args[flagIndex + 1])' — "
                            + "local | server")
                }
            }
            let manifest = try ExperimentStore.freeze(
                name: args[1], force: args.contains("--force"),
                runSubstrate: runSubstrate)
            var frozenLine =
                "frozen '\(manifest.name)' @ \(manifest.freezeHash?.prefix(12) ?? "?")… "
                + "(model rev \(manifest.modelRevision?.prefix(12) ?? "UNPINNED"), "
                + "git \(manifest.gitCommit?.prefix(8) ?? "unknown"))"
            if runSubstrate != ExperimentStore.evidenceSubstrate {
                frozenLine += " — evidence matched for run substrate \(runSubstrate)"
            }
            sink.out(frozenLine)
            // FULL hashes: the human line elides all three, which is exactly
            // the P1 the dry run reported — "freezeHash key discoverable only
            // by reading experiment.json".
            var payload: [String: JSONValue] = [
                "experiment": .string(manifest.name),
                "status": .string(manifest.status.rawValue),
                "freezeHash": manifest.freezeHash.map { JSONValue.string($0) } ?? .null,
                "modelID": .string(manifest.modelID),
                "modelRevision": manifest.modelRevision.map {
                    JSONValue.string($0)
                } ?? .null,
                "gitCommit": manifest.gitCommit.map { JSONValue.string($0) } ?? .null,
                "frozenAt": manifest.frozenAt.map { JSONValue.string($0) } ?? .null,
                "frozenBy": manifest.frozenBy.map { JSONValue.string($0) } ?? .null,
                "runSubstrate": .string(runSubstrate),
                "forced": .bool(manifest.freezeForced == true),
            ]
            // A forced freeze is non-citable, and the envelope must say so as
            // loudly as the stamp does — advisories, never a changed exit
            // code (a `set -e` wrapper must not break on a deliberate force).
            var advisories: [SteerLabCLIEnvelope.Advisory] = []
            if let skipped = manifest.forcedGatesSkipped, !skipped.isEmpty {
                payload["forcedGatesSkipped"] = .array(skipped.map { .string($0) })
                for gate in skipped {
                    advisories.append(
                        .init(
                            code: CLIAdvisory.freezeGateSkipped.rawValue,
                            detail: "gate '\(gate)' would have failed and was "
                                + "skipped by --force; this freeze is stamped "
                                + "freezeForced and is not citable"))
                }
            }
            return ExperimentCLIResult(
                message: frozenLine, changed: true, payload: payload,
                advisories: advisories,
                nextAction: .init(verb: "experiment run \(manifest.name)"))

        case "duplicate":
            guard args.count >= 3 else {
                throw ExperimentError(reason: "usage: experiment duplicate <name> <new-name>")
            }
            let copy = try ExperimentStore.duplicate(name: args[1], as: args[2])
            sink.out("created draft '\(copy.name)' from '\(args[1])'")
            return ExperimentCLIResult(
                message: "created draft '\(copy.name)' from '\(args[1])'",
                changed: true,
                payload: [
                    "name": .string(copy.name),
                    "source": .string(args[1]),
                    "status": .string(copy.status.rawValue),
                ])

        case "extract":
            guard args.count >= 2 else {
                throw ExperimentError(reason: "usage: experiment extract <name>")
            }
            try await ExperimentTasks.extract(experimentName: args[1])
            return ExperimentCLIResult(
                message: "extracted vectors for '\(args[1])'", changed: true,
                payload: ["experiment": .string(args[1])],
                nextAction: .init(verb: "experiment validate \(args[1])"))

        case "validate":
            guard args.count >= 2 else {
                throw ExperimentError(reason: "usage: experiment validate <name>")
            }
            let runDirectory = try await ExperimentTasks.validate(
                experimentName: args[1])
            // The VACUITY LEDGER (2026-08-17 firewall repair) is the single
            // most consequential thing this verb can report: a validate run
            // that scored no held-out probe for a pinned concept looks
            // identical on the surface and cannot satisfy freeze's
            // validateEvidence gate. It is stamped in the run's report; the
            // envelope names it rather than making an agent go read the file.
            let vacuous = Self.vacuousConcepts(inRunAt: runDirectory)
            var payload: [String: JSONValue] = [
                "experiment": .string(args[1]),
                "runDirectory": .string(runDirectory.path),
                "vacuous": .bool(!(vacuous ?? []).isEmpty),
            ]
            if let vacuous {
                payload["vacuousConcepts"] = .array(vacuous.map { .string($0) })
            }
            var advisories: [SteerLabCLIEnvelope.Advisory] = []
            for concept in vacuous ?? [] {
                advisories.append(
                    .init(
                        CLIAdvisory.vacuousValidation,
                        "no held-out probe was scored for '\(concept)' — "
                            + "this evidence will NOT satisfy freeze's "
                            + "validateEvidence gate"))
            }
            // THE QUALITY NUMBERS (punch list #1, P4). A chance-level probe —
            // accuracy 0.50 — froze and ran with no machine signal at all,
            // because the envelope reported only that validate had happened.
            // The per-concept scores now ride the document, and a probe at or
            // below the floor says so in the closed advisory vocabulary.
            let scores = Self.validationScores(inRunAt: runDirectory)
            if !scores.isEmpty {
                payload["validation"] = .array(scores.map(\.payload))
            }
            for score in scores where score.isAtOrBelowChance {
                advisories.append(
                    .init(CLIAdvisory.probeAtChanceFloor, score.advisoryDetail))
            }
            return ExperimentCLIResult(
                message: (vacuous ?? []).isEmpty
                    ? "validated '\(args[1])' → \(runDirectory.lastPathComponent)"
                        + (scores.isEmpty
                            ? ""
                            : " — " + scores.map(\.summary).joined(separator: ", "))
                    : "VACUOUS validation for '\(args[1])' — no held-out probe "
                        + "for \((vacuous ?? []).joined(separator: ", "))",
                changed: true, payload: payload, advisories: advisories,
                nextAction: .init(
                    verb: (vacuous ?? []).isEmpty
                        ? "experiment freeze \(args[1])"
                        : "author the named validation.jsonl sets, then "
                            + "experiment validate \(args[1])"))

        case "sweep":
            guard args.count >= 2 else {
                throw ExperimentError(reason: "usage: experiment sweep <name>")
            }
            // The defaulted-criterion advisory is computed BEFORE the sweep
            // runs, from the manifest alone, so it survives a sweep that then
            // fails — and so an agent reading `--json` sees it whether or not
            // the grid completed (punch list #1, P3).
            var sweepAdvisories: [SteerLabCLIEnvelope.Advisory] = []
            if let manifest = try? ExperimentStore.load(name: args[1]),
                let counts = ExperimentTasks.choiceShapedItemCount(manifest),
                let advisory = SweepSelectionRule.defaultedSelectionAdvisory(
                    spec: manifest.sweep?.selection,
                    choiceItemCount: counts.choice, totalItemCount: counts.total)
            {
                sweepAdvisories.append(
                    .init(CLIAdvisory.sweepSelectionDefaulted, advisory))
            }
            let sweepOutcome = try await ExperimentTasks.sweep(
                experimentName: args[1])
            // A sweep on a non-draft manifest writes NO
            // `<concept>-recommended` condition — it reports into the run
            // directory only. That caveat was a stderr line; an agent that
            // missed it promoted from a manifest carrying no recommendation
            // and could not say why (punch list #1, P2).
            if sweepOutcome.recommendationsOnly {
                sweepAdvisories.append(
                    .init(
                        CLIAdvisory.sweepRecommendationsOnly,
                        "'\(sweepOutcome.experiment)' is "
                            + "\(sweepOutcome.manifestStatus), so the sweep wrote "
                            + "no <concept>-recommended condition — the "
                            + "recommendations live in "
                            + "\(sweepOutcome.runDirectory.lastPathComponent)/"
                            + "recommendations.json, and promote reads them from "
                            + "there"))
            }
            let selectedCount = sweepOutcome.recommendations.filter(\.selected).count
            return ExperimentCLIResult(
                message: "swept '\(args[1])' → \(selectedCount) of "
                    + "\(sweepOutcome.recommendations.count) concept(s) "
                    + "selected a cell on \(sweepOutcome.criterion) in "
                    + sweepOutcome.runDirectory.lastPathComponent,
                changed: true,
                payload: Self.sweepPayload(sweepOutcome),
                advisories: sweepAdvisories,
                nextAction: .init(
                    verb: sweepOutcome.recommendations.first(where: \.selected)
                        .map { "experiment promote \(args[1]) \($0.concept)" }
                        ?? "experiment set-sweep-selection \(args[1]) --objective "
                            + "judgeScore|logprobShift  (no cell was selected)"))

        case "run":
            guard args.count >= 2 else {
                throw ExperimentError(
                    reason: "usage: experiment run <name> [--prompts prompts/path.jsonl]")
            }
            let runDirectory = try await ExperimentTasks.run(
                experimentName: args[1], promptsFile: flag("--prompts"))
            return ExperimentCLIResult(
                message: "ran '\(args[1])' → \(runDirectory.lastPathComponent)",
                changed: true,
                payload: [
                    "experiment": .string(args[1]),
                    "runDirectory": .string(runDirectory.path),
                ],
                advisories: Self.implicitCaseFamilyAdvisories(
                    experimentNamed: args[1]),
                nextAction: .init(verb: "experiment analyze \(args[1])"))

        case "analyze":
            // Headless statistics: paired-to-baseline effect sizes (bootstrap CI
            // + Wilcoxon) over the newest completed run, under the epoch guard
            // (the run's experiment-hash stamp must equal the live manifest's
            // content hash). Pure CPU — no model load.
            guard args.count >= 2 else {
                throw ExperimentError(
                    reason: "usage: experiment analyze <name> [--allow-unverified-epoch]")
            }
            let runDirectory = try ExperimentTasks.analyze(
                experimentName: args[1],
                allowUnverifiedEpoch: args.contains("--allow-unverified-epoch"))
            var payload = Self.analysisPayload(inRunAt: runDirectory)
            payload["experiment"] = .string(args[1])
            payload["runDirectory"] = .string(runDirectory.path)
            let entries = Self.effectSizeCount(payload)
            // An analysis with zero effect-size entries "succeeded" and
            // measured nothing — the dry run's P0-2. The refusal half landed
            // as a firewall repair; the envelope still says so out loud.
            //
            // WHY it measured nothing is read off the source run rather than
            // assumed (WP0 dry run #2): the fixed sentence claimed "no
            // non-baseline condition" on a run that had two of them and 24
            // records, which is the one thing the reader must not be told.
            var advisories: [SteerLabCLIEnvelope.Advisory] = []
            if entries == 0 {
                let detail: String =
                    if case .string(let sourceRun)? = payload["sourceRun"] {
                        ExperimentTasks.emptyAnalysisDetail(
                            sourceRunNamed: sourceRun)
                    } else {
                        ExperimentTasks.emptyAnalysisNoContrast
                    }
                advisories.append(.init(CLIAdvisory.emptyAnalysis, detail))
            }
            // Zero ENTRIES had an advisory; entries that are all exactly zero
            // did not (punch list #1, P14). They are a different fact: the
            // pairing worked and every intervention moved nothing, which is
            // either a real null or — far likelier at this stage — an arm that
            // was declared and never actually injected.
            if entries > 0, Self.allEffectSizesAreZero(inRunAt: runDirectory) {
                advisories.append(
                    .init(
                        CLIAdvisory.allEffectSizesZero,
                        "all \(entries) effect size(s) are exactly 0.0 — the "
                            + "pairing worked and no condition moved any metric. "
                            + "Check that the arms actually injected: compare "
                            + "generations.jsonl across conditions in "
                            + "\(runDirectory.lastPathComponent), and confirm the "
                            + "declared studyType does not inert them "
                            + "(steerlab-cli experiment verify \(args[1]))"))
            }
            return ExperimentCLIResult(
                message: "analyzed '\(args[1])' → \(entries) effect-size "
                    + "entr\(entries == 1 ? "y" : "ies") in "
                    + runDirectory.lastPathComponent,
                changed: true, payload: payload, advisories: advisories)

        // MARK: The two step-7 authoring verbs (punch list #1, P3 + P13)

        case "set-sweep-selection":
            // THE criterion. Absent, the sweep resolves to markerDensity —
            // surface prose — which the methods note forbids as a promotion
            // objective for decision studies. Until this verb existed that
            // rule could not be followed headlessly at all: `sweep.selection`
            // was reachable only from the Optimizations panel.
            guard args.count >= 2 else {
                throw ExperimentError(
                    reason: "usage: experiment set-sweep-selection <name> "
                        + "--objective \(SweepSelectionRule.knownMetrics.joined(separator: "|")) "
                        + "[--choice-prompts prompts/…/choices.jsonl] "
                        + "[--capability-tolerance 0.15] [--coherence-floor 0.45] "
                        + "[--control-margin M] [--control-apply-to winner|topK] "
                        + "[--control-top-k K]\n"
                        + "       experiment set-sweep-selection <name> "
                        + "--objective \"\"  (clears the declaration — the sweep "
                        + "then resolves to the historical markerDensity rule)")
            }
            guard let objective = flag("--objective") else {
                throw ExperimentError(
                    reason: "set-sweep-selection needs --objective "
                        + SweepSelectionRule.knownMetrics.joined(separator: "|")
                        + " — the criterion is DATA, declared before the sweep "
                        + "runs, which is what makes the selection "
                        + "preregistered rather than chosen after seeing the grid")
            }
            if objective.isEmpty {
                let cleared = try ExperimentStore.setSweepSelection(
                    nil, experimentName: args[1])
                let line = "cleared the sweep selection on '\(cleared.name)' — "
                    + "the sweep resolves to markerDensity, tolerance 0.15, "
                    + "coherence floor 0.45"
                sink.out(line)
                return ExperimentCLIResult(
                    message: line, changed: true,
                    payload: [
                        "experiment": .string(cleared.name),
                        "selection": .null,
                    ])
            }
            let selection = try Self.parseSweepSelection(
                objective: objective,
                choicePrompts: flag("--choice-prompts"),
                capabilityTolerance: flag("--capability-tolerance"),
                coherenceFloor: flag("--coherence-floor"),
                controlMargin: flag("--control-margin"),
                controlApplyTo: flag("--control-apply-to"),
                controlTopK: flag("--control-top-k"))
            let declared = try ExperimentStore.setSweepSelection(
                selection, experimentName: args[1])
            let resolved = try SweepSelectionRule.resolve(declared.sweep?.selection)
            let selectionLine =
                "declared sweep selection on '\(declared.name)': objective "
                + "\(resolved.metric), capability tolerance "
                + "\(resolved.capabilityTolerance), coherence floor "
                + "\(resolved.coherenceFloor)"
                + (resolved.matchedNormRandomMargin.map {
                    ", matched-norm random margin \($0) (\(resolved.controlApplyTo))"
                } ?? "")
            sink.out(selectionLine)
            var selectionPayload: [String: JSONValue] = [
                "experiment": .string(declared.name),
                "objective": .string(resolved.metric),
                "capabilityTolerance": .number(resolved.capabilityTolerance),
                "coherenceFloor": .number(resolved.coherenceFloor),
                "implementedOnThisEngine": .bool(
                    SweepSelectionRule.implementedMetrics.contains(resolved.metric)),
            ]
            if let margin = resolved.matchedNormRandomMargin {
                selectionPayload["matchedNormRandomMargin"] = .number(margin)
                selectionPayload["controlApplyTo"] = .string(resolved.controlApplyTo)
                if let topK = resolved.controlTopK {
                    selectionPayload["controlTopK"] = .number(Double(topK))
                }
            }
            if let file = declared.sweep?.selection?.objective?.choicePromptsFile {
                selectionPayload["choicePromptsFile"] = .string(file)
                // FULL hash: the pin is what freeze checks and what the sweep
                // refuses on when it drifts.
                if let hash = declared.sweep?.selection?.objective?.choicePromptsHash {
                    selectionPayload["choicePromptsHash"] = .string(hash)
                }
            }
            var selectionAdvisories: [SteerLabCLIEnvelope.Advisory] = []
            if case .declaredAhead(let metric) = SweepSpecForm.validateSelection(
                declared.sweep?.selection)
            {
                selectionAdvisories.append(
                    .init(
                        CLIAdvisory.sweepSelectionDefaulted,
                        "objective '\(metric)' is declared but not implemented "
                            + "on this engine — the sweep will REFUSE at start "
                            + "(declaring ahead is legal manifest data)"))
            }
            return ExperimentCLIResult(
                message: selectionLine, changed: true, payload: selectionPayload,
                advisories: selectionAdvisories,
                nextAction: .init(verb: "experiment sweep \(declared.name)"))

        case "set-instruments":
            // WHAT the study measures. `outcomeInstruments` is provenance —
            // written explicitly, never inferred from the prompt data — which
            // is correct and was, until now, unreachable headlessly: pinning
            // items that carry `options` + `target` did NOT engage the
            // answer-logprob instrument, and no verb could say so (dry run #1,
            // P13). Fields preserved ≠ measurement enabled.
            guard args.count >= 3 else {
                throw ExperimentError(
                    reason: "usage: experiment set-instruments <name> "
                        + "<instrument>[,…]  (known: "
                        + ExperimentStore.knownOutcomeInstruments.joined(separator: " | ")
                        + ") [--ordinal-aggregation "
                        + ExperimentStore.knownOrdinalAggregations.joined(separator: "|")
                        + "]  (\"\" clears the declaration — the engine then "
                        + "defaults to sampled text)")
            }
            let instruments = args[2]
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            var updated = try ExperimentStore.setOutcomeInstruments(
                instruments.isEmpty ? nil : instruments, experimentName: args[1])
            if let aggregation = flag("--ordinal-aggregation") {
                updated = try ExperimentStore.setOrdinalAggregation(
                    aggregation.isEmpty ? nil : aggregation,
                    experimentName: args[1])
            }
            let plan = ExecutionPlan.resolve(instruments: updated.outcomeInstruments)
            // The one true sentence about responseFormat, said on the verb
            // that turns the instrument on (gate-5 dry run #2, P3: three
            // surfaces disagreed about whether `label` is required). It is
            // not: the dispatch reads `options`, absence of responseFormat is
            // permissive, and the field only ever SUBTRACTS.
            let declaredList: String =
                updated.outcomeInstruments?.joined(separator: ", ")
                ?? "(cleared — engine default: sampledText)"
            let readabilityClause: String =
                plan.scoresDirectly
                ? " — it reads every pinned item that carries `options`; "
                    + "`responseFormat` is optional (absence is fine), a "
                    + "declared \"json\"/\"freeText\" is refused at run start, "
                    + "and a declared outcomeInstrumentScope measures only the "
                    + "formats it lists"
                : ""
            let instrumentLine =
                "declared outcome instruments on '\(updated.name)': "
                + declaredList + " — this study \(plan.summary)"
                + readabilityClause
            sink.out(instrumentLine)
            var instrumentAdvisories: [SteerLabCLIEnvelope.Advisory] = []
            // A declared value nothing will read is a design mistake worth
            // saying out loud at the moment it is declared, not at run time.
            if let inert = ExecutionPlan.inertSamplingAdvisory(
                instruments: updated.outcomeInstruments,
                temperature: updated.temperature,
                samplesPerItem: updated.samplesPerItem)
            {
                instrumentAdvisories.append(
                    .init(CLIAdvisory.choiceItemsWithoutInstrument, inert))
            }
            return ExperimentCLIResult(
                message: instrumentLine, changed: true,
                payload: [
                    "experiment": .string(updated.name),
                    "outcomeInstruments": .array(
                        (updated.outcomeInstruments ?? []).map { .string($0) }),
                    "ordinalAggregation": updated.ordinalAggregation.map {
                        JSONValue.string($0)
                    } ?? .null,
                    "generatesSampledText": .bool(plan.generatesSampledText),
                    "scoresDirectly": .bool(plan.scoresDirectly),
                    "samplingIsOperative": .bool(plan.samplingIsOperative),
                ],
                advisories: instrumentAdvisories)

        case "set-style-taxonomy":
            // Pin a reasoning-style taxonomy (prompts/taxonomies/<name>.json)
            // into a draft manifest: validates the file loads on this engine,
            // then stamps reasoningStyleTaxonomyPath + reasoningStyleTaxonomyHash
            // (SHA-256 of the file bytes). Drift after pinning is a verify
            // violation; no pin = no reasoning-style scoring.
            guard args.count >= 3 else {
                throw ExperimentError(
                    reason: "usage: experiment set-style-taxonomy <name> "
                        + "<prompts/taxonomies/file.json>")
            }
            let pinned = try ExperimentStore.pinReasoningStyleTaxonomy(
                experimentName: args[1], path: args[2])
            sink.out(
                "pinned reasoning-style taxonomy \(args[2]) @ "
                    + "\(pinned.reasoningStyleTaxonomyHash?.prefix(12) ?? "?")…")
            return ExperimentCLIResult(
                message: "pinned reasoning-style taxonomy \(args[2])",
                changed: true,
                payload: [
                    "experiment": .string(pinned.name),
                    "taxonomyPath": .string(args[2]),
                    "taxonomyHash": pinned.reasoningStyleTaxonomyHash.map {
                        JSONValue.string($0)
                    } ?? .null,
                ])

        case "rescore-style":
            // Post-hoc reasoning-style scoring: recompute rs_<featureID> values
            // for an existing completed run through the pinned taxonomy — NEW
            // files (reasoning-style.csv + reasoning-style.json) in a fresh run
            // directory; the source run is never mutated. Epoch-guarded like
            // analyze. Pure CPU — no model load.
            guard args.count >= 2 else {
                throw ExperimentError(
                    reason: "usage: experiment rescore-style <name> [--run <run-dir>] "
                        + "[--allow-unverified-epoch]")
            }
            let runDirectory = try ExperimentTasks.rescoreStyle(
                experimentName: args[1],
                runDirectoryName: flag("--run"),
                allowUnverifiedEpoch: args.contains("--allow-unverified-epoch"))
            return ExperimentCLIResult(
                message: "rescored reasoning style for '\(args[1])' → "
                    + runDirectory.lastPathComponent,
                changed: true,
                payload: [
                    "experiment": .string(args[1]),
                    "runDirectory": .string(runDirectory.path),
                ])

        case "evaluate":
            // Paired-judge evaluation of a completed run (same epoch guard as
            // analyze). Default source: the newest completed run; --run selects
            // a specific run directory (name under runs/, or a path).
            guard args.count >= 2 else {
                throw ExperimentError(
                    reason: "usage: experiment evaluate <name> [--run <run-dir>] "
                        + "[--allow-unverified-epoch]")
            }
            let sourceRun: URL
            if let runFlag = flag("--run") {
                sourceRun =
                    runFlag.hasPrefix("/")
                    ? URL(filePath: runFlag)
                    : ExperimentStore.runsDirectory.appending(path: runFlag)
            } else if let newest = ExperimentTasks.newestCompletedRun(
                experimentName: args[1])
            {
                sourceRun = newest
            } else {
                throw ExperimentError.refusing(
                    .missingPrerequisite,
                    "no completed study run found for '\(args[1])' — run "
                        + "'steerlab-cli experiment run \(args[1])' first, or pass --run",
                    repair: "steerlab-cli experiment run \(args[1]) && "
                        + "steerlab-cli experiment evaluate \(args[1])")
            }
            let evaluation = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: args[1],
                sourceRunDirectory: sourceRun,
                allowUnverifiedEpoch: args.contains("--allow-unverified-epoch"))
            return ExperimentCLIResult(
                message: "evaluated '\(args[1])' → "
                    + evaluation.lastPathComponent,
                changed: true,
                payload: [
                    "experiment": .string(args[1]),
                    "sourceRun": .string(sourceRun.path),
                    "evaluationDirectory": .string(evaluation.path),
                ])

        case "promote":
            // Headless Promote: mint an agent (variant artifact) from the sweep-
            // selected cell. --cell L:ALPHA is the loud manual override. Pure
            // CPU — no model load, promotable from any manifest status.
            guard args.count >= 3 else {
                throw ExperimentError(
                    reason: "usage: experiment promote <name> <concept> "
                        + "[--agent-name <name>] [--cell <layer>:<alpha> --reason <why>]")
            }
            var cell: (layer: Int, alpha: Double)?
            if let raw = flag("--cell") {
                let parts = raw.split(separator: ":", maxSplits: 1)
                guard parts.count == 2, let layer = Int(parts[0]),
                      let alpha = Double(parts[1])
                else {
                    throw ExperimentError(
                        reason: "--cell must be <layer>:<alpha>, e.g. 17:0.4")
                }
                cell = (layer: layer, alpha: alpha)
            }
            let record = try AgentPromotion.promote(
                experimentName: args[1], concept: args[2],
                agentName: flag("--agent-name"), cell: cell,
                overrideReason: flag("--reason"),
                log: { self.sink.out($0) })
            var payload: [String: JSONValue] = [
                "experiment": .string(args[1]),
                "concept": .string(args[2]),
                "agentName": .string(record.artifact.name),
                "artifact": .string(record.url.path),
            ]
            if cell != nil {
                payload["promotedBy"] = .string("manualOverride")
                if let reason = flag("--reason") {
                    payload["overrideReason"] = .string(reason)
                }
            } else {
                payload["promotedBy"] = .string("criterion")
            }
            return ExperimentCLIResult(
                message: "promoted '\(args[2])' → agent '\(record.artifact.name)'",
                changed: true, payload: payload,
                nextAction: .init(
                    verb: "experiment confirm \(args[1]) --agent \(record.artifact.name)"))

        case "confirm":
            // Confirmation stage: declare a perturbation policy around a promoted
            // agent's anchor cell — expands mechanically into ordinary hashed
            // conditions on the DRAFT manifest (ConfirmationStudy.attach).
            guard args.count >= 2, let agent = flag("--agent") else {
                throw ExperimentError(
                    reason: "usage: experiment confirm <name> --agent "
                        + "<variant-name-or-path> [--deltas 0.2,0.5] [--no-control] "
                        + "(default deltas: 0.2)")
            }
            var deltas: [Double] = [0.2]
            if let raw = flag("--deltas") {
                let parsed = raw.split(separator: ",").compactMap {
                    Double($0.trimmingCharacters(in: .whitespaces))
                }
                guard !parsed.isEmpty, parsed.count == raw.split(separator: ",").count
                else {
                    throw ExperimentError(
                        reason: "--deltas must be comma-separated numbers, e.g. 0.2,0.5")
                }
                deltas = parsed
            }
            let manifest = try ConfirmationStudy.attach(
                experimentName: args[1], agent: agent, deltas: deltas,
                includeControl: !args.contains("--no-control"),
                log: { self.sink.out($0) })
            return ExperimentCLIResult(
                message: "attached a confirmation policy around '\(agent)' — "
                    + "\(manifest.conditions.count) condition(s)",
                changed: true,
                payload: [
                    "experiment": .string(manifest.name),
                    "agent": .string(agent),
                    "deltas": .array(deltas.map { .number($0) }),
                    "includeControl": .bool(!args.contains("--no-control")),
                    "conditionCount": .number(Double(manifest.conditions.count)),
                    "conditions": .array(
                        manifest.conditions.map { .string($0.name) }),
                ],
                nextAction: .init(verb: "experiment freeze \(manifest.name)"))

        default:
            throw ExperimentError(
                reason: "verbs: list | create | attach | pin-prompts | pin-rubric "
                    + "| declare-condition | set-sweep-selection | set-instruments "
                    + "| set-style-taxonomy | verify "
                    + "| freeze | duplicate | extract | validate | sweep | run "
                    + "| analyze | rescore-style | evaluate | promote | confirm")
        }
    }

    // MARK: - Authoring-verb argument parsing (WP0 step 5½)

    /// The condition control vocabulary, as the manifest spells it. Not a
    /// free-text field: an unknown `controlType` silently produces a cell
    /// that duplicates the treatment (the substitution never fires).
    static let knownControlTypes = ["randomMatchedNorm", "randomDirectionAblation"]

    /// `<concept>:<layer>:<alpha>[:add|ablate]`, comma-separated — the
    /// manifest's own slot vocabulary, in the comma-list flag idiom the
    /// family already uses (`--deltas`, `--corpus`).
    ///
    /// A multi-slot spec IS the linear mix `h + Σ αᵢ·vᵢ`; it hashes into the
    /// manifest as one condition like any other, which is what makes a mix a
    /// first-class experimental cell rather than an ad-hoc run-time option.
    static func parseSlots(
        _ spec: String
    ) throws -> [ExperimentManifest.Condition.Slot] {
        let shape =
            "each slot is <concept>:<layer>:<alpha>[:add|ablate], "
            + "comma-separated — e.g. fear:17:0.4,sympathy:17:-0.2"
        var slots: [ExperimentManifest.Condition.Slot] = []
        for raw in spec.split(separator: ",") {
            let field = raw.trimmingCharacters(in: .whitespaces)
            guard !field.isEmpty else { continue }
            let parts = field.split(separator: ":", omittingEmptySubsequences: false)
                .map(String.init)
            guard parts.count == 3 || parts.count == 4 else {
                throw ExperimentError(
                    reason: "bad slot '\(field)' — \(shape)")
            }
            let concept = parts[0].trimmingCharacters(in: .whitespaces)
            guard !concept.isEmpty else {
                throw ExperimentError(reason: "bad slot '\(field)' — \(shape)")
            }
            guard let layer = Int(parts[1]), layer >= 0 else {
                throw ExperimentError(
                    reason: "bad layer '\(parts[1])' in slot '\(field)' — a "
                        + "block index ≥ 0")
            }
            guard let alpha = Double(parts[2]) else {
                throw ExperimentError(
                    reason: "bad alpha '\(parts[2])' in slot '\(field)' — a "
                        + "number, in residual-norm units unless "
                        + "--alpha-units raw")
            }
            var mode: InterventionPlan.Mode?
            if parts.count == 4 {
                let raw = parts[3].trimmingCharacters(in: .whitespaces)
                guard let parsed = InterventionPlan.Mode(rawValue: raw) else {
                    throw ExperimentError(
                        reason: "unknown mode '\(raw)' in slot '\(field)' — "
                            + InterventionPlan.Mode.allCases.map(\.rawValue)
                                .joined(separator: " | "))
                }
                // An explicit `add` is never WRITTEN (manifest bytes are the
                // content hash — a key appearing on every existing condition
                // would re-identify every frozen study in the workspace), so
                // it is parsed and then dropped back to nil.
                mode = parsed == .add ? nil : parsed
            }
            slots.append(
                .init(concept: concept, layer: layer, alpha: alpha, mode: mode))
        }
        guard !slots.isEmpty else {
            throw ExperimentError(reason: "--slots is empty — \(shape)")
        }
        return slots
    }

    /// The sweep-selection flags → the manifest's own `SweepSelection` shape
    /// (WP0 step 7, P3).
    ///
    /// Every value is range-checked by `ExperimentStore.setSweepSelection`
    /// through the panel's own validators, so this function's only job is
    /// SHAPE: an unparseable number must not become a silently-dropped
    /// constraint. `--control-apply-to topK` without `--control-top-k` is
    /// refused here rather than resolved to a default, because how many cells
    /// a control covers is a preregistration decision.
    static func parseSweepSelection(
        objective: String,
        choicePrompts: String?,
        capabilityTolerance: String?,
        coherenceFloor: String?,
        controlMargin: String?,
        controlApplyTo: String?,
        controlTopK: String?
    ) throws -> ExperimentManifest.SweepSelection {
        func number(_ raw: String?, _ flag: String) throws -> Double? {
            guard let raw else { return nil }
            guard let value = Double(raw), value.isFinite else {
                throw ExperimentError(
                    reason: "\(flag) expects a number — got '\(raw)'")
            }
            return value
        }
        guard SweepSelectionRule.knownMetrics.contains(objective) else {
            throw ExperimentError(
                reason: "unknown --objective '\(objective)' — known metrics: "
                    + SweepSelectionRule.knownMetrics.joined(separator: ", ")
                    + ". markerDensity is a manipulation check; decision "
                    + "studies select on judgeScore or logprobShift.")
        }
        if choicePrompts != nil, objective != "logprobShift" {
            throw ExperimentError(
                reason: "--choice-prompts belongs to the logprobShift "
                    + "objective — '\(objective)' reads no choice file")
        }
        var block = ExperimentManifest.SweepSelection.Objective(metric: objective)
        block.choicePromptsFile = choicePrompts
        var selection = ExperimentManifest.SweepSelection(objective: block)
        let tolerance = try number(capabilityTolerance, "--capability-tolerance")
        let floor = try number(coherenceFloor, "--coherence-floor")
        if tolerance != nil || floor != nil {
            selection.constraints = .init(
                capabilityTolerance: tolerance, coherenceFloor: floor)
        }
        if let margin = try number(controlMargin, "--control-margin") {
            var topK: Int?
            if let raw = controlTopK {
                guard let value = Int(raw), value >= 1 else {
                    throw ExperimentError(
                        reason: "--control-top-k expects an integer ≥ 1 — got "
                            + "'\(raw)'")
                }
                topK = value
            }
            let applyTo = controlApplyTo ?? "winner"
            guard ["winner", "topK"].contains(applyTo) else {
                throw ExperimentError(
                    reason: "--control-apply-to must be winner | topK — got "
                        + "'\(applyTo)'")
            }
            if applyTo == "topK", topK == nil {
                throw ExperimentError(
                    reason: "--control-apply-to topK needs --control-top-k K — "
                        + "how many cells the control covers is a "
                        + "preregistration decision, not a default")
            }
            selection.controls = .init(
                matchedNormRandomMargin: margin,
                applyTo: applyTo == "winner" ? nil : applyTo,
                topK: applyTo == "winner" ? nil : topK)
        } else if controlApplyTo != nil || controlTopK != nil {
            throw ExperimentError(
                reason: "--control-apply-to / --control-top-k describe a "
                    + "matched-norm random control that is not declared — add "
                    + "--control-margin M")
        }
        return selection
    }

    /// `<name>:<kind>[:<model>[:<provider>]]`, comma-separated. The fourth
    /// field is the OpenRouter serving PROVIDER — the pin without which the
    /// same slug can be answered by different backends, which is why verify
    /// treats a missing one as a violation. A local judge's blank model
    /// resolves to the STUDY model at its pinned revision; a claude judge's
    /// to the default judge model.
    static func parseJudges(
        _ spec: String
    ) throws -> [ExperimentManifest.JudgeRef] {
        let shape =
            "each judge is <name>:<kind>[:<model>[:<provider>]], "
            + "comma-separated — kinds: "
            + ExperimentStore.knownJudgeKinds.joined(separator: " | ")
        var judges: [ExperimentManifest.JudgeRef] = []
        for raw in spec.split(separator: ",") {
            let field = raw.trimmingCharacters(in: .whitespaces)
            guard !field.isEmpty else { continue }
            let parts = field.split(separator: ":", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            guard parts.count >= 2, parts.count <= 4, !parts[0].isEmpty else {
                throw ExperimentError(reason: "bad judge '\(field)' — \(shape)")
            }
            guard ExperimentStore.knownJudgeKinds.contains(parts[1]) else {
                throw ExperimentError(
                    reason: "unknown judge kind '\(parts[1])' in '\(field)' — "
                        + ExperimentStore.knownJudgeKinds.joined(separator: " | "))
            }
            let model = parts.count >= 3 && !parts[2].isEmpty ? parts[2] : nil
            let provider = parts.count == 4 && !parts[3].isEmpty ? parts[3] : nil
            if parts[1] != "openrouter", provider != nil {
                throw ExperimentError(
                    reason: "judge '\(parts[0])' is \(parts[1]), which pins no "
                        + "serving provider — the fourth field is OpenRouter's")
            }
            judges.append(
                .init(
                    name: parts[0], kind: parts[1], model: model,
                    provider: provider))
        }
        guard !judges.isEmpty else {
            throw ExperimentError(reason: "--judges is empty — \(shape)")
        }
        return judges
    }

    /// How many items the just-pinned prompt file holds — reported so the
    /// agent learns the size of the measured task from the pin, not from a
    /// later run. nil when the file cannot be re-read (the pin itself
    /// already parsed it, so this is belt-and-braces).
    static func taskPromptCount(file: String) -> Int? {
        guard let data = try? Data(contentsOf: ExperimentStore.resolveProjectPath(file)),
            let prompts = try? ExperimentTasks.parseTaskPrompts(data)
        else { return nil }
        return prompts.count
    }

    // MARK: - Payload helpers

    /// Decode a JSON document a verb already produced as text, so its exact
    /// keys ride into `result` rather than being re-modelled (and re-drifted)
    /// here. nil when the text is not decodable — the payload simply omits it.
    static func jsonValue(fromJSONText text: String) -> JSONValue? {
        try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))
    }

    /// The sweep's decision, machine-readable (punch list #1, P2).
    static func sweepPayload(
        _ outcome: ExperimentTasks.SweepOutcome
    ) -> [String: JSONValue] {
        [
            "experiment": .string(outcome.experiment),
            "runDirectory": .string(outcome.runDirectory.path),
            "manifestStatus": .string(outcome.manifestStatus),
            "recommendationsOnly": .bool(outcome.recommendationsOnly),
            "cancelled": .bool(outcome.cancelled),
            "criterion": .string(outcome.criterion),
            // FULL hash: the dev split is what the selection is provenance
            // FOR, and the human line never showed it.
            "devPromptsHash": .string(outcome.devPromptsHash),
            "recommendations": .array(
                outcome.recommendations.map { entry in
                    var fields: [String: JSONValue] = [
                        "concept": .string(entry.concept),
                        "selected": .bool(entry.selected),
                    ]
                    if let layer = entry.layer, let alpha = entry.alpha {
                        fields["winningCell"] = .object([
                            "layer": .number(Double(layer)),
                            "alpha": .number(alpha),
                        ])
                    }
                    if let criterion = entry.criterion {
                        fields["criterion"] = .string(criterion)
                    }
                    if !entry.metrics.isEmpty {
                        fields["metrics"] = .object(
                            entry.metrics.mapValues { JSONValue.number($0) })
                    }
                    if let control = entry.control {
                        fields["control"] = .string(control)
                    }
                    if let failure = entry.failure {
                        fields["failure"] = .string(failure)
                    }
                    return .object(fields)
                }),
        ]
    }

    /// One concept's held-out probe score, as the validate run recorded it
    /// (punch list #1, P4).
    struct ValidationScore {
        var concept: String
        var layer: Int?
        var accuracy: Double?
        var balancedAccuracy: Double?
        var auc: Double?
        var scenarios: Int?
        var oneSided: Bool

        /// The chance floor for a two-class held-out probe. Balanced accuracy
        /// is preferred where it exists — a probe on an unbalanced set can
        /// read 0.7 while separating nothing — and a threshold that put every
        /// item on one side is at the floor by construction, whatever the
        /// accuracy says.
        var isAtOrBelowChance: Bool {
            if oneSided { return true }
            let value = balancedAccuracy ?? accuracy
            guard let value else { return false }
            return value <= 0.5 + 1e-9
        }

        var summary: String {
            let percent = accuracy.map { "\(Int(($0 * 100).rounded()))%" } ?? "—"
            return "\(concept) \(percent)"
                + (auc.map { String(format: " (AUC %.2f)", $0) } ?? "")
        }

        var advisoryDetail: String {
            let value = balancedAccuracy ?? accuracy
            let read = value.map { String(format: "%.2f", $0) } ?? "unscored"
            return "'\(concept)' scores \(read) on its held-out probe"
                + (oneSided
                    ? " and the transfer threshold put EVERY item on one side, "
                        + "so the number is measuring the threshold, not the vector"
                    : " — at or below the 0.50 chance floor for a two-class probe")
                + ". This evidence still SATISFIES freeze's validateEvidence "
                + "gate (the gate asks whether a probe was scored, not how "
                + "well), so a non-discriminating direction can be frozen and "
                + "run. Author more or better never-named scenarios, or sweep "
                + "the reading layer, before treating this concept as measured."
        }

        var payload: JSONValue {
            var fields: [String: JSONValue] = ["concept": .string(concept)]
            if let layer { fields["layer"] = .number(Double(layer)) }
            if let accuracy { fields["accuracy"] = .number(accuracy) }
            if let balancedAccuracy {
                fields["balancedAccuracy"] = .number(balancedAccuracy)
            }
            if let auc { fields["auc"] = .number(auc) }
            if let scenarios { fields["scenarios"] = .number(Double(scenarios)) }
            fields["oneSidedPredictions"] = .bool(oneSided)
            fields["atOrBelowChance"] = .bool(isAtOrBelowChance)
            return .object(fields)
        }
    }

    /// Read the per-concept probe scores out of a validate run's own report.
    ///
    /// The report's canonical shape is a `depths` list (one entry per declared
    /// reading depth) with a flat mirror only when exactly one depth resolves
    /// — so the DEEPEST-scoring reading is not assumed: the first depth is the
    /// one reported, matching the human line the run already printed. A
    /// concept whose entry is a string ("no validation.jsonl — convergent gate
    /// NOT run") is not a score and is skipped; the vacuity ledger already
    /// names it.
    static func validationScores(inRunAt runDirectory: URL) -> [ValidationScore] {
        for name in ["validation-report.json", "report.json"] {
            guard
                let data = try? Data(
                    contentsOf: runDirectory.appending(component: name)),
                let object = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
                let validation = object["validation"] as? [String: Any]
            else { continue }
            var scores: [ValidationScore] = []
            for concept in validation.keys.sorted() {
                guard let entry = validation[concept] as? [String: Any] else {
                    continue
                }
                let depth =
                    (entry["depths"] as? [[String: Any]])?.first ?? entry
                let diagnostics = depth["diagnostics"] as? [String: Any] ?? [:]
                scores.append(
                    ValidationScore(
                        concept: concept,
                        layer: depth["layer"] as? Int,
                        accuracy: (depth["accuracy"] as? NSNumber)?.doubleValue,
                        balancedAccuracy: (diagnostics["balancedAccuracy"]
                            as? NSNumber)?.doubleValue,
                        auc: (diagnostics["auc"] as? NSNumber)?.doubleValue,
                        scenarios: entry["scenarios"] as? Int,
                        oneSided: (diagnostics["oneSidedPredictions"] as? Bool)
                            ?? false))
            }
            if !scores.isEmpty { return scores }
        }
        return []
    }

    /// True when the analysis produced entries and EVERY paired mean
    /// difference is exactly zero (punch list #1, P14).
    static func allEffectSizesAreZero(inRunAt runDirectory: URL) -> Bool {
        guard
            let data = try? Data(
                contentsOf: runDirectory.appending(component: "analysis.json")),
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let entries = object["effectSizes"] as? [[String: Any]],
            !entries.isEmpty
        else { return false }
        return entries.allSatisfy { entry in
            guard let value = (entry["meanDiff"] as? NSNumber)?.doubleValue
            else { return false }
            return value == 0
        }
    }

    /// The vacuity ledger a validate run stamped into its own report. nil when
    /// the run predates the stamp (legacy evidence, which keeps satisfying the
    /// gate exactly as it did).
    static func vacuousConcepts(inRunAt runDirectory: URL) -> [String]? {
        for name in ["validation-report.json", "report.json"] {
            guard let data = try? Data(
                contentsOf: runDirectory.appending(component: name)),
                let object = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any]
            else { continue }
            if let vacuous = object["vacuousConcepts"] as? [String] {
                return vacuous.sorted()
            }
        }
        return nil
    }

    /// A compact summary of an analyze run's `analysis.json` — the effect-size
    /// LEDGER, not the full table, which lives in `effect-sizes.csv` beside it.
    /// The name of THIS engine's `effect-sizes.csv` column set, reported as
    /// `result.effectSizesSchema`. The server's is `endpoint-deltaMean`.
    ///
    /// The two engines' effect-size columns differ by idiom (audit §3.2) and
    /// always have: Swift writes `condition,metric,n,meanDiff,…`, the server
    /// writes `condition,endpoint,n,deltaMean,…`. The columns are NOT being
    /// unified — existing run directories are immutable and both engines'
    /// readers depend on their own — so the envelope NAMES the dialect
    /// instead, which is what a cross-engine reader actually needs. Any
    /// future unification is a schema-versioned change announced by this
    /// field, never a silent rewrite. Server twin:
    /// `cli_payloads.EFFECT_SIZES_SCHEMA`.
    static let effectSizesSchema = "metric-meanDiff"

    static func analysisPayload(inRunAt runDirectory: URL) -> [String: JSONValue] {
        guard
            let data = try? Data(
                contentsOf: runDirectory.appending(component: "analysis.json")),
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else { return [:] }
        let entries = object["effectSizes"] as? [[String: Any]] ?? []
        var conditions = Set<String>()
        var metrics = Set<String>()
        var significant = 0
        for entry in entries {
            if let condition = entry["condition"] as? String {
                conditions.insert(condition)
            }
            if let metric = entry["metric"] as? String { metrics.insert(metric) }
            if let adjusted = entry["adjustedP"] as? Double, adjusted < 0.05 {
                significant += 1
            }
        }
        var payload: [String: JSONValue] = [
            "effectSizeCount": .number(Double(entries.count)),
            "conditions": .array(conditions.sorted().map { .string($0) }),
            "metrics": .array(metrics.sorted().map { .string($0) }),
            "significantAtAdjusted05": .number(Double(significant)),
            "effectSizesCSV": .string(
                runDirectory.appending(component: "effect-sizes.csv").path),
            "effectSizesSchema": .string(Self.effectSizesSchema),
        ]
        if let sourceRun = object["sourceRun"] as? String {
            payload["sourceRun"] = .string(sourceRun)
        }
        if let hash = object["experimentHash"] as? String {
            payload["experimentHash"] = .string(hash)
        }
        if let unverified = object["epochUnverified"] as? Bool {
            payload["epochUnverified"] = .bool(unverified)
        }
        return payload
    }

    static func effectSizeCount(_ payload: [String: JSONValue]) -> Int {
        guard case .number(let count)? = payload["effectSizeCount"] else { return 0 }
        return Int(count)
    }
}

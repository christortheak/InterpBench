import Foundation

// =============================================================================
// The output seam and the structured outcome for the non-cluster verb dispatch
// (WP0-AGENT-SURFACE-AUDIT §7 step 4).
//
// Step 4 is a MOVE, not a rewrite: `ExperimentCLIRunner` carries
// `main.swift`'s dispatch verbatim into `ExperimentKit` so the verbs become
// driveable in-process by tests. Two things have to exist for that to be
// possible without changing a byte of what the binary prints:
//
//   1. A SINK. The dispatch printed as it worked (`pinned french @ …`,
//      progress during extract, the blocker count during `data check`). Those
//      writes are now injected, so a test can collect them instead of racing
//      the process's real stdout.
//   2. A STRUCTURED OUTCOME. The dispatch called `exit(n)` from the middle of
//      four verbs and let `main.swift`'s catch print the trailing error line.
//      Both now come back as data.
//
// Step 5 wires the envelope: `ExperimentCLIOutcome` now carries BOTH exit
// codes — `exitCode`, today's 0/1/2/64 human vocabulary, and
// `envelope.exitCode`, the `SteerLabCLIState` vocabulary — and the binary
// picks by mode. That split is the audit's compatibility posture (§7 rows 5
// and 7): `--json` is a NEW surface and can be born speaking the new codes,
// while flipping the codes a human's `set -e` wrapper already depends on is
// step 7's uniform-refusal change. The one exception is a MALFORMED
// invocation, which was never a refusal: an undeclared flag exits 64 in both
// modes (P0-4).
// =============================================================================

/// Where the moved dispatch writes.
///
/// **The two stdout methods are not interchangeable, and that is the point.**
/// `out` is the `print` twin — Swift's buffered stdio, which is what every
/// moved `print(…)` used. `outRaw` is the `FileHandle.standardOutput.write`
/// twin — unbuffered, which is what `remote chat`'s token streaming used, and
/// a streaming verb that only flushed at buffer boundaries would be a real
/// behaviour change. Keeping both means each moved site maps to the mechanism
/// it already had.
///
/// `err` takes the terminator inside the string, because every moved stderr
/// site already spelled its own `\n`.
public struct ExperimentCLISink: Sendable {

    private let writeOut: @Sendable (String) -> Void
    private let writeOutRaw: @Sendable (String) -> Void
    private let writeErr: @Sendable (String) -> Void

    public init(
        out: @escaping @Sendable (String) -> Void,
        outRaw: @escaping @Sendable (String) -> Void,
        err: @escaping @Sendable (String) -> Void
    ) {
        self.writeOut = out
        self.writeOutRaw = outRaw
        self.writeErr = err
    }

    /// The `print(_:terminator:)` twin. Buffered stdio.
    public func out(_ text: String, terminator: String = "\n") {
        writeOut(text + terminator)
    }

    /// The `FileHandle.standardOutput.write` twin. Unbuffered; used only where
    /// the moved code already wrote the file handle directly.
    public func outRaw(_ text: String) { writeOutRaw(text) }

    /// The `FileHandle.standardError.write` twin. The caller supplies `\n`,
    /// exactly as the moved sites do.
    public func err(_ text: String) { writeErr(text) }

    /// The process sink: the same two mechanisms `main.swift` used, so the
    /// bytes and the buffering both survive the move.
    public static let standard = ExperimentCLISink(
        out: { text in print(text, terminator: "") },
        outRaw: { text in FileHandle.standardOutput.write(Data(text.utf8)) },
        err: { text in FileHandle.standardError.write(Data(text.utf8)) })

    /// A sink that discards everything — for callers that only want the
    /// outcome (the app, and tests that assert on exit codes alone).
    public static let discarding = ExperimentCLISink(
        out: { _ in }, outRaw: { _ in }, err: { _ in })

    /// The `--json` sink: EVERYTHING a verb says goes to stderr, so stdout
    /// carries exactly one JSON document (audit §2.2, the rule the cluster
    /// family already follows at `main.swift`'s streaming sink).
    ///
    /// This covers the dispatch's own writes. Library code deeper in the call
    /// graph still calls `print` directly — `ExperimentTasks` alone has 69
    /// such sites, on the model-loading verbs — so the binary ALSO redirects
    /// file descriptor 1 to stderr for the duration of a JSON-mode
    /// invocation and writes the envelope to the saved descriptor. Two
    /// mechanisms for one rule, because the rule is the contract: an agent
    /// that gets a progress line mixed into its document has no way to
    /// recover.
    public static let diagnostics = ExperimentCLISink(
        out: { text in FileHandle.standardError.write(Data(text.utf8)) },
        outRaw: { text in FileHandle.standardError.write(Data(text.utf8)) },
        err: { text in FileHandle.standardError.write(Data(text.utf8)) })
}

/// What a verb produced, beyond what it printed — the material of the
/// envelope's `result`, `message`, `changed`, and advisories.
///
/// Every agent-path verb returns one. A verb with nothing structured to say
/// still returns its one-sentence `message`, because the envelope's header is
/// a closed key set and `message` is in it.
public struct ExperimentCLIResult: Sendable {

    /// One sentence a human can act on — the envelope's `message`.
    public var message: String
    /// Did the invocation mutate durable state? Read-only verbs answer false
    /// even when they report a problem.
    public var changed: Bool
    /// The success state. `.ready` unless advisories promote it.
    public var state: SteerLabCLIState
    /// The per-verb payload. **Hashes here are FULL** — the elided
    /// `prefix(12)…` display stays in the human line, where it is a courtesy;
    /// in a machine document it is a provenance hole (audit §8, P1: "hashes
    /// ELIDED so provenance is unobtainable from stdout").
    public var payload: [String: JSONValue]
    /// Non-blocking observations. Never change the exit code.
    public var advisories: [SteerLabCLIEnvelope.Advisory]
    /// The exact next command, when the verb knows it.
    public var nextAction: SteerLabCLIEnvelope.NextAction?
    /// The workspace this verb ANSWERED ABOUT, when that is not the root the
    /// invocation resolved to.
    ///
    /// Exactly one verb needs it: `workspace init <path>` creates a NEW root
    /// and its whole answer is about that root, while the resolved root is
    /// still whatever `STEERLAB_WORKSPACE` or the fallback named — so the
    /// envelope's `workspace` reported a directory the answer had nothing to
    /// do with (gate-5 dry run #2, P3). The field's documented meaning is
    /// "the data root that answered"; this keeps that meaning true rather
    /// than redefining it. Every other verb leaves it nil and reports the
    /// resolved root, unchanged.
    public var workspaceOverride: String?

    public init(
        message: String = "", changed: Bool = false,
        state: SteerLabCLIState = .ready,
        payload: [String: JSONValue] = [:],
        advisories: [SteerLabCLIEnvelope.Advisory] = [],
        nextAction: SteerLabCLIEnvelope.NextAction? = nil,
        workspaceOverride: String? = nil
    ) {
        self.message = message
        self.changed = changed
        self.state = state
        self.payload = payload
        self.advisories = advisories
        self.nextAction = nextAction
        self.workspaceOverride = workspaceOverride
    }
}

/// Collects what a sink was handed, so an in-process test can assert on the
/// exact stdout/stderr bytes a verb produces.
///
/// `@unchecked Sendable` justified: the only mutable state is guarded by
/// `lock` on every path, and the type exposes no way to reach it unguarded.
/// A value-typed alternative cannot work — the sink's closures are
/// `@Sendable` and must accumulate into shared storage.
public final class ExperimentCLIRecorder: @unchecked Sendable {

    private let lock = NSLock()
    private var outText = ""
    private var errText = ""

    public init() {}

    /// Everything written to stdout, in order, terminators included.
    public var standardOutput: String { lock.withLock { outText } }
    /// Everything written to stderr, in order, terminators included.
    public var standardError: String { lock.withLock { errText } }

    /// stdout split into lines, trailing empty element dropped — the usual
    /// shape for an assertion.
    public var outputLines: [String] {
        var lines = standardOutput.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        return lines
    }

    public var sink: ExperimentCLISink {
        ExperimentCLISink(
            out: { [self] text in lock.withLock { outText += text } },
            outRaw: { [self] text in lock.withLock { outText += text } },
            err: { [self] text in lock.withLock { errText += text } })
    }
}

/// What one non-cluster `steerlab-cli` invocation produced — the
/// "(envelope, exit code)" pair of audit step 4, completed at step 5.
///
/// **Two exit codes, deliberately.** `exitCode` is the HUMAN-mode code and is
/// still authoritative there: 0, 1, 2, 64, exactly what the verbs have always
/// returned, so no existing wrapper breaks. `envelope.exitCode` is derived
/// from `envelope.state` in the `SteerLabCLIState` vocabulary and is what
/// `--json` mode exits with, so an agent can finally tell a freeze-gate
/// refusal (65) from a missing experiment (66) from an operational failure
/// (70) — the whole reason the package exists (audit §2.3). The two agree
/// wherever the vocabularies agree: 0 is 0, and an undeclared flag is 64 in
/// both.
public struct ExperimentCLIOutcome: Sendable, Equatable {

    /// A typed failure, carrying what the refusal already knew and used to
    /// throw away. `gate`/`gates` are populated from
    /// `ExperimentError.freezeRefusal` (WP0 step 2) — the gate id now reaches
    /// a machine caller instead of dying inside prose.
    public struct Failure: Sendable, Equatable {
        /// Byte-identical to what `main.swift` interpolated as `\(error)`.
        public var reason: String
        /// The second stderr line, when the refusal had one (`remote`'s
        /// typed `ClusterCLIError`). `nil` for every prose-only refusal.
        public var repairAction: String?
        /// The freeze gate that declined, wire form. `nil` unless this is a
        /// freeze-gate refusal.
        public var gate: String?
        /// Every freeze gate that failed, in `FreezeGate.vocabulary` order.
        public var gates: [String]?

        public init(
            reason: String, repairAction: String? = nil,
            gate: String? = nil, gates: [String]? = nil
        ) {
            self.reason = reason
            self.repairAction = repairAction
            self.gate = gate
            self.gates = gates
        }
    }

    /// The verb family, as it appears in the stderr prefix: `experiment`,
    /// `data`, `vectors`, `remote`, `workspace`.
    public var namespace: String
    /// The verb as typed, e.g. `experiment freeze` — what the envelope
    /// header carries.
    public var verb: String
    /// What the process exits with in HUMAN mode. Today's vocabulary,
    /// unchanged.
    public var exitCode: Int32
    /// Present iff the invocation refused or failed.
    public var failure: Failure?
    /// The one document `--json` mode writes to stdout. Its `state` is
    /// authoritative and its `exitCode` is what JSON mode exits with.
    public var envelope: SteerLabCLIEnvelope

    public init(
        namespace: String, verb: String, exitCode: Int32,
        failure: Failure? = nil, envelope: SteerLabCLIEnvelope? = nil
    ) {
        self.namespace = namespace
        self.verb = verb
        self.exitCode = exitCode
        self.failure = failure
        self.envelope =
            envelope
            ?? SteerLabCLIEnvelope(
                verb: verb, engine: SteerLabCLIEnvelope.localEngine,
                state: exitCode == 0 ? .ready : (exitCode == 64 ? .blocked : .failed),
                message: failure?.reason ?? "")
    }

    public var isSuccess: Bool { exitCode == 0 }

    /// The exit code for the mode the caller is in. One accessor, so the
    /// binary never has to remember which vocabulary it is in.
    public func exitCode(json: Bool) -> Int32 {
        json ? envelope.exitCode : exitCode
    }
}

/// Renders an outcome to the bytes `main.swift` has always written after a
/// verb returned — the mirror of `ClusterCLIRenderer.humanText`.
///
/// One place, so the "prose today, envelope at step 5" switch is a one-line
/// change at the call site rather than a hunt through the dispatch.
public enum ExperimentCLIRenderer {

    /// The trailing stderr text, or `nil` when the verb said everything it had
    /// to say through the sink. Terminated; write it verbatim.
    public static func standardErrorText(_ outcome: ExperimentCLIOutcome) -> String? {
        guard let failure = outcome.failure else { return nil }
        var text = "steerlab-cli \(outcome.namespace): \(failure.reason)\n"
        if let repair = failure.repairAction { text += "  \(repair)\n" }
        return text
    }
}

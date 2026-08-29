import Foundation
import SteeringKit

// =============================================================================
// The shared agent-path machine protocol (WP0-AGENT-SURFACE-AUDIT §2.2–§2.3) —
// Step 1: the types, wired to nothing.
//
// ONE envelope for every agent-path verb on both engines, not one per verb
// family. Gate 5 is defined as an agent with no prior context: one document
// shape and one state vocabulary is the whole product, and per-family envelopes
// would make the agent learn N contracts before it could act on the first
// refusal.
//
// The required header is byte-identical to the six fields `ClusterCLIEnvelope`
// already emits (`schemaVersion`, `verb`, `state`, `changed`, `observedAt`,
// `message`), extended with `engine`. Everything else is optional and omitted
// when absent — a caller testing for a key gets a straight answer rather than a
// null. `cluster` itself is GRANDFATHERED (audit §2.2): its 25 optional
// cluster-specific fields stay at top level on `ClusterCLIEnvelope`, and this
// type does not replace it.
//
// The security rule is carried forward STRUCTURALLY, not editorially: there is
// no property on this type, or on any type it nests, that can hold a
// credential. Presence Booleans and provenance labels only. If a verb one day
// wants to report a judge API key, it reports `keyAvailable` and `keySource`.
// A `result` payload is per-verb and open, and the same rule binds it: never
// put a secret in it. `CLIEnvelopeTests.noPropertyCanHoldACredential` checks
// the shape recursively rather than by grepping prose.
//
// Wired to nothing at Step 1: no verb, no `main.swift`, no cluster path
// constructs one yet. The only cluster-side change is that
// `ClusterLifecycleState.exitCode` now delegates to `SteerLabCLIState`, which
// is behaviour-identical case by case and twin-tested.
// =============================================================================

// MARK: - State vocabulary

/// The shared top-level state vocabulary for every agent-path verb on both
/// engines (audit §2.3). It extends `ClusterLifecycleState` — whose nine cases
/// it reproduces name for name and exit code for exit code — by exactly three:
/// `okWithAdvisories`, `refused`, and `notFound`.
///
/// **The JSON `state` is authoritative; the exit code is a convenience for
/// shell callers.** That doctrine is inherited verbatim from the cluster
/// vocabulary, and it is why both live in one place: the envelope and the
/// process can then never disagree.
///
/// Deliberately a CLOSED vocabulary. Decoding an unknown state fails rather
/// than degrading to a guess, because an agent that mis-reads a refusal as a
/// success does more damage than one that stops.
public enum SteerLabCLIState: String, Codable, Sendable, CaseIterable {

    // Success — exit 0.

    /// The requested target is reached.
    case ready
    /// Work remains and nothing is blocking it.
    case planned
    /// The work is in progress.
    case running
    /// Succeeded, and `advisories` is non-empty. **Exit 0, always.** An agent
    /// that treated an advisory as a failure would refuse to walk a legitimate
    /// lifecycle, and a `set -e` wrapper would break on every forced-freeze
    /// warning. The distinction lives in `state` and `advisories`, never in
    /// the exit code.
    case okWithAdvisories

    // Waiting on something outside this process.

    /// A human must authenticate in their own Terminal.
    case needsHumanAuthentication
    /// A mutation needs its explicit `--allow-…` flag.
    case needsApproval
    /// Valid asynchronous work is in flight; repeating the command is correct.
    case pending
    /// Retryable: a layer could not be read.
    case degraded

    // Refusals and failures.

    /// Malformed invocation or unusable configuration (`EX_USAGE`). Narrowed
    /// from cluster's historical "invalid usage *or* invalid site
    /// configuration": the usage half stays here, and the site-shaped half
    /// stays with it.
    case blocked
    /// A gate declined a well-formed request against a healthy system
    /// (`EX_DATAERR`). The reason the package exists: a freeze-gate refusal, a
    /// missing file, and a network failure must not all be exit 1.
    case refused
    /// The named experiment, run, or panel does not exist (`EX_NOINPUT`).
    case notFound
    /// Non-retryable operational failure.
    case failed

    /// Recommended process exit code (audit §2.3). Lives here so both engines
    /// and every client agree by construction.
    ///
    /// Note the codes NOT in this table: 2 (`data check` blockers on both
    /// engines today, and the server's general "refused/bad input") migrates to
    /// 65 in one gated commit, and 85 (checkpoint-requested, server-only) is
    /// deliberately outside the state vocabulary.
    public var exitCode: Int32 {
        switch self {
        case .ready, .planned, .running, .okWithAdvisories: 0
        case .needsHumanAuthentication: 10
        case .needsApproval: 11
        case .pending: 12
        case .degraded: 13
        case .blocked: 64
        case .refused: 65
        case .notFound: 66
        case .failed: 70
        }
    }

    /// Whether this state means the verb did what was asked. Advisories do not
    /// change the answer.
    public var isSuccess: Bool { exitCode == 0 }
}

// MARK: - Advisory vocabulary

/// The closed vocabulary of advisory codes an agent-path verb may attach
/// (WP0 step 7).
///
/// Advisories NEVER change the exit code — that rule lives on
/// `SteerLabCLIState.okWithAdvisories` and is what keeps a `set -e` wrapper
/// from breaking on a legitimate lifecycle. What this type adds is that the
/// codes are a vocabulary rather than string literals scattered through the
/// dispatch: an agent can switch on them, and step 8's server twin
/// (`ADVISORY_CODES`) has one list to match instead of a grep.
///
/// Every code here answers a question an agent could not otherwise ask without
/// reading run artifacts off disk.
public enum CLIAdvisory: String, CaseIterable, Sendable, Codable {

    /// A freeze gate would have failed and `--force` skipped it. The freeze is
    /// stamped `freezeForced` and is not citable.
    case freezeGateSkipped
    /// A validate run scored no held-out probe for a pinned concept.
    case vacuousValidation
    /// A probe scored at or below the chance floor, or put every held-out item
    /// on one side of the threshold. The vector is not discriminating; the
    /// evidence will nonetheless satisfy the validateEvidence gate, which is
    /// exactly why the number has to reach the document (dry run #1, P4).
    case probeAtChanceFloor
    /// One judge is pinned where freeze's `judgeValidity` gate wants two.
    case judgePanelTooSmall
    /// `analyze` produced zero effect-size entries — the source run has no
    /// non-baseline condition to pair against.
    case emptyAnalysis
    /// `analyze` produced entries whose paired mean differences are ALL exactly
    /// zero. Not the same as no entries: the pairing worked and the
    /// intervention moved nothing, which is either a real null or an inert
    /// condition wearing a measured one's name (dry run #1, P14).
    case allEffectSizesZero
    /// A sweep ran against a non-draft manifest, so its recommendations were
    /// reported into the run directory only and no `<concept>-recommended`
    /// condition was written.
    case sweepRecommendationsOnly
    /// No `sweep.selection` was declared, so the sweep selected on
    /// `markerDensity` — a surface-prose diagnostic — while the task set is
    /// choice-shaped. The document's most emphatic rule, silently violated
    /// (dry run #1, P3).
    case sweepSelectionDefaulted
    /// Pinned items carry `options`/`target` but no direct-scoring instrument
    /// is declared, so the deterministic answer-logprob instrument will not
    /// engage and the arm will be scored on sampled prose (dry run #1, P13).
    case choiceItemsWithoutInstrument
    /// An imported run's model revision was adopted into the local manifest.
    case revisionAdoption
    /// The same, where the adoption is contested and needs a human look.
    case revisionAdoptionWarning
    /// A `site qualify` check warned rather than failed — the node runs, but
    /// something about it is unpinned, drifted, or only partly verified.
    /// Emitted by the Python engine only (`steerlab-server site qualify`,
    /// WP6/gate 7): qualification runs ON the node being qualified, and this
    /// engine never runs on a cluster node. The code lives in BOTH
    /// vocabularies regardless — the advisory vocabulary is closed and
    /// identical across engines by contract, so that an agent's exhaustive
    /// `switch` compiles against either one.
    case siteQualifyWarning
    /// A measurement instrument was selected by a DEPRECATED implicit rule
    /// rather than a declaration, and the verb went ahead with it.
    ///
    /// Today's one instance: `caseFamily: "sentencing"` selecting the built-in
    /// duration endpoint (`parsedMonths`). `caseFamily` is a provenance LABEL
    /// — the manifest's `numericParser` + the workspace parser registry are
    /// the declared mechanism, and the registry's shipped `sentencing-months`
    /// entry reproduces the built-in parser exactly. Old manifests keep
    /// working; they just say so now.
    ///
    /// Named for the MECHANISM (an implicit selection, deprecated) and not for
    /// the field, deliberately. The vocabulary is closed and cross-engine, so
    /// every addition costs a synchronized move of two literals plus both
    /// twin tests; a `deprecatedCaseFamily` code would buy one deprecation and
    /// force a second code the next time an implicit selection is retired,
    /// while an agent's `switch` cares about exactly one thing here — the run
    /// was configured by inference rather than by declaration.
    case deprecatedImplicitSelection
    /// A declared system prompt will NOT reach every item of the run,
    /// because a more specific declaration replaces it for some of them.
    ///
    /// Today's one instance: pinned task items whose scripted `transcript`
    /// opens with the item's own `system` turn. That turn replaces the
    /// study's frame for that item — the transcript is the more specific
    /// declaration, one rule on both engines — so a persona typed into the
    /// study frame silently does not arm those items. Everywhere else the
    /// frame DOES reach the model, by whichever route the family allows: a
    /// genuine system turn where the chat template has a system role, the
    /// first user turn where it does not.
    ///
    /// Named for the MECHANISM (a declaration that does not apply) rather
    /// than for the transcript case, on the same reasoning as
    /// `deprecatedImplicitSelection`: the vocabulary is closed and
    /// cross-engine, and what an agent's `switch` cares about is that
    /// something it wrote will not arm every arm of the run.
    case systemPromptNotApplied

    /// A capability reading was taken under ONE operating regime where the
    /// battery charter asks for two. Today's one instance: a standalone
    /// `battery run` against a format-2 battery, which declares only short
    /// greedy answers — so the report carries accuracy and no generation
    /// health, and cannot express length inflation, variance collapse, or
    /// incoherence.
    ///
    /// Named for the MECHANISM (a reading with one regime), not for the
    /// format: the vocabulary is closed and cross-engine, and what an agent's
    /// `switch` cares about is that the floor it is about to cite is
    /// narrower than the charter's floor. It is deliberately NOT a lint
    /// finding — a format-2 battery is a complete PINNED per-condition
    /// control and is the only format a study may pin; it falls short only
    /// relative to a FLOOR reading, and the verb taking one is the only
    /// thing that knows a floor reading was asked for.
    case singleRegimeCapabilityReading

    public static let vocabulary: [String] = allCases.map(\.rawValue)
}

extension SteerLabCLIEnvelope.Advisory {
    /// Build an advisory from the closed vocabulary, so a typo cannot mint a
    /// code no agent knows.
    public init(_ code: CLIAdvisory, _ detail: String) {
        self.init(code: code.rawValue, detail: detail)
    }
}

// MARK: - Envelope

/// The one versioned document an agent-path verb writes to stdout in `--json`
/// mode (audit §2.2).
///
/// Rules carried verbatim from `ClusterCLIEnvelope`, because they are the
/// reason it works:
///
/// - **Exactly one JSON document on stdout**; human diagnostics go to stderr;
///   no ANSI sequences anywhere.
/// - **Sorted keys, ISO-8601 dates, trailing newline** — so a golden fixture is
///   stable and no caller ever parses a float timestamp.
/// - **Commands are argv arrays, never shell strings** (a verb that reports one
///   puts it in `result` as an array).
/// - **No property can hold a credential**, by construction.
/// - **`--json` is honoured even when parsing itself fails**: an agent that
///   asked for machine output must not get prose on the one path it cannot
///   parse.
///
/// The header is a CLOSED key set, pinned by `contractHeaderKeys` and asserted
/// by `CLIEnvelopeTests.theWireNamesAreTheContractLiteral`; the Python twin
/// literal lands at audit Step 8 (`test_cli_envelope.py`), following the
/// duplicated-literal parity pattern already used for `config.json`
/// (`RunMetadata.contractKeys` ↔ `test_run_config.py`). `result` is OPEN and
/// per-verb, with its own committed golden fixture per verb — it is the one
/// place a verb may say something this type does not know about.
public struct SteerLabCLIEnvelope: Codable, Sendable, Equatable {

    /// The ENVELOPE's version, never the payload's.
    public static let schemaVersion = 1

    /// This engine's stamp, aliased to the extraction engine's own constant so
    /// the two can never drift apart.
    public static let localEngine = RepEReader.substrate
    /// The Python server's stamp, aliased to the same constant the artifact
    /// scoping rules use.
    public static let serverEngine = WorkspaceScoping.serverSubstrate

    // MARK: Nested payloads

    /// A non-blocking observation: something the caller should know that did
    /// not stop the verb. **Never affects the exit code** — see
    /// `SteerLabCLIState.okWithAdvisories`. A `--strict-advisories` opt-in may
    /// later promote advisories to a refusal for CI; that is a caller policy,
    /// not a property of this type.
    public struct Advisory: Codable, Sendable, Equatable {
        /// Stable machine id, e.g. `unpromotedSourceAgent`. Never prose.
        public var code: String
        /// Plain-language detail for a human reading the same document.
        public var detail: String

        public init(code: String, detail: String) {
            self.code = code
            self.detail = detail
        }

        enum CodingKeys: String, CodingKey {
            case code, detail
        }
    }

    /// The exact next command's shape. Field-for-field identical on the wire to
    /// `ClusterCLIEnvelope.NextAction` — deliberately a separate Swift type so
    /// the shared envelope does not depend on the grandfathered cluster one,
    /// and the wire names are pinned by `contractOptionalKeys` on both.
    public struct NextAction: Codable, Sendable, Equatable {
        /// The verb to run next, e.g. `experiment validate demo`.
        public var verb: String
        /// True when no agent can do this — a person must act at a terminal.
        public var requiresHuman: Bool
        /// The flag(s) that would authorize the mutation, when it is gated.
        public var missingPermissionFlags: [String]
        /// Optional elaboration; absent rather than empty.
        public var detail: String?

        public init(
            verb: String, requiresHuman: Bool = false,
            missingPermissionFlags: [String] = [], detail: String? = nil
        ) {
            self.verb = verb
            self.requiresHuman = requiresHuman
            self.missingPermissionFlags = missingPermissionFlags
            self.detail = detail
        }

        enum CodingKeys: String, CodingKey {
            case verb, requiresHuman, missingPermissionFlags, detail
        }
    }

    /// A typed refusal: stable code, the GATE that declined, every gate that
    /// failed, a plain-language reason, and a concrete repair.
    ///
    /// `gate`/`gates` are the fields the audit identifies as missing today
    /// (§2.4): both engines compute a gate id and then drop it on the refusal
    /// path, and Swift's freeze concatenates multiple failures into one
    /// bullet-list string, so even the NUMBER of failed gates is unrecoverable
    /// without parsing prose. `gates` therefore carries **all** failures in the
    /// closed vocabulary's order.
    ///
    /// **`gate` is the gate whose prose is in `reason`, and it is NOT
    /// necessarily `gates.first`** (WP0 step 5 reconciliation). `freeze`
    /// evaluates its gates in a HISTORICAL refusal order (revision →
    /// measurementPins → validateEvidence → variantValidity → batteryEvidence
    /// → judgeValidity → gitClean) but stamps and reports `gates` in the
    /// closed `FreezeGate.vocabulary` order — two different permutations, as
    /// `FreezeRefusal`'s own doc comment says. Step 1's factory derived
    /// `gate = gates.first`, which would have made the envelope name a
    /// different gate than the message it carries; the factory now takes the
    /// gate explicitly and the wiring passes `FreezeRefusal.gate` through, so
    /// `gate` and `reason` can never describe different gates. The invariant
    /// that survives is the weaker, true one: `gate` is always a MEMBER of
    /// `gates`.
    ///
    /// `gates.first` remains the fallback when a refusal names no gate of its
    /// own — a refusal site with a single gate is the same either way.
    public struct Failure: Codable, Sendable, Equatable {
        /// Stable failure code, e.g. `freezeGateFailed`. Never prose.
        public var code: String
        /// The gate that declined, from a closed vocabulary
        /// (`FreezeGate`: `revision`, `validateEvidence`, `batteryEvidence`,
        /// `judgeValidity`, `variantValidity`, `gitClean`, `measurementPins`;
        /// or the wider `LifecycleGate` set). Absent when the refusal is not
        /// gate-shaped. The gate `reason` describes — a MEMBER of `gates`,
        /// not necessarily its first element (see above).
        public var gate: String?
        /// Every gate that failed, in `FreezeGate.vocabulary` order — the
        /// same order and the same ids the `forcedGatesSkipped` stamp uses.
        /// When present it contains `gate`.
        public var gates: [String]?
        /// Why, in plain language, for the human reading the same document.
        public var reason: String
        /// The concrete command or edit that repairs it.
        public var repairAction: String

        public init(
            code: String, gate: String? = nil, gates: [String]? = nil,
            reason: String, repairAction: String
        ) {
            self.code = code
            self.gate = gate
            self.gates = gates
            self.reason = reason
            self.repairAction = repairAction
        }

        enum CodingKeys: String, CodingKey {
            case code, gate, gates, reason, repairAction
        }
    }

    // MARK: Required header (closed key set)

    /// The envelope's schema version — 1 today, bumped only in a change that
    /// updates both engines' key literals together.
    public let schemaVersion: Int
    /// The verb exactly as typed, e.g. `experiment freeze`.
    public var verb: String
    /// Which engine answered: `SteerLabCLIEnvelope.localEngine` or
    /// `.serverEngine`. A provenance label, never a capability claim.
    public var engine: String
    /// Authoritative outcome. The exit code is derived from it.
    public var state: SteerLabCLIState
    /// Did this invocation mutate durable state? Read-only verbs answer false
    /// even when they report a problem.
    public var changed: Bool
    /// When the observation was made, ISO-8601 on the wire.
    public var observedAt: Date
    /// One sentence a human can act on. Never the sole carrier of machine
    /// meaning — that is `state`, `error.code`, and `error.gate`.
    public var message: String

    // MARK: Optional fields

    /// Which data root answered — the workspace path, so an agent can tell a
    /// wrong-workspace answer from a wrong answer. A path, not a credential.
    public var workspace: String?
    /// Non-blocking observations. Omitted when empty, never emitted as `[]`,
    /// so a caller's key test is a straight answer.
    public var advisories: [Advisory]?
    /// The exact next command, machine-readable — no prose parsing required.
    public var nextAction: NextAction?
    /// Present iff `state` is `refused`, `blocked`, `notFound`, or `failed`.
    public var error: Failure?
    /// Per-verb payload, open by design. Absent when there is nothing to
    /// report. The credential rule binds here too.
    public var result: [String: JSONValue]?

    // MARK: Wire-name contract

    /// The closed header: every one of these keys is present in every
    /// document, on both engines. Byte-identical to `ClusterCLIEnvelope`'s six
    /// required fields plus `engine`.
    ///
    /// Twin literal: `test_cli_envelope.py` on the server (audit Step 8), in
    /// the duplicated-literal pattern of `RunMetadata.contractKeys` ↔
    /// `test_run_config.py`. Two independent literals with a naming
    /// cross-reference is deliberately worse engineering than a shared schema
    /// file and deliberately better parity enforcement: neither engine can
    /// quietly follow the other.
    public static let contractHeaderKeys: [String] = [
        "changed", "engine", "message", "observedAt", "schemaVersion", "state", "verb",
    ]

    /// Every key that MAY appear beyond the header, and nothing else may.
    /// Sorted, matching the encoder's `.sortedKeys` output order.
    public static let contractOptionalKeys: [String] = [
        "advisories", "error", "nextAction", "result", "workspace",
    ]

    enum CodingKeys: String, CodingKey {
        case schemaVersion, verb, engine, state, changed, observedAt, message
        case workspace, advisories, nextAction, error, result
    }

    // MARK: Construction

    public init(
        verb: String,
        engine: String,
        state: SteerLabCLIState,
        message: String,
        changed: Bool = false,
        observedAt: Date = Date(),
        workspace: String? = nil
    ) {
        self.schemaVersion = Self.schemaVersion
        self.verb = verb
        self.engine = engine
        self.state = state
        self.message = message
        self.changed = changed
        self.observedAt = observedAt
        self.workspace = workspace
    }

    /// A success, with the state chosen by whether anything was advised — so a
    /// caller cannot accidentally report `ready` while carrying advisories.
    public static func success(
        verb: String,
        engine: String,
        message: String,
        changed: Bool = false,
        advisories: [Advisory] = [],
        observedAt: Date = Date(),
        workspace: String? = nil
    ) -> SteerLabCLIEnvelope {
        var envelope = SteerLabCLIEnvelope(
            verb: verb, engine: engine,
            state: advisories.isEmpty ? .ready : .okWithAdvisories,
            message: message, changed: changed, observedAt: observedAt,
            workspace: workspace)
        envelope.add(advisories)
        return envelope
    }

    /// A gate refusal against a healthy system: exit 65 by default, with the
    /// gate named rather than described.
    ///
    /// `gate` is EXPLICIT (WP0 step 5). The caller passes the gate whose
    /// message it is carrying in `reason` — `FreezeRefusal.gateID` on the
    /// freeze path — because freeze's refusal order and the stamp vocabulary's
    /// order are different permutations, so deriving the gate from
    /// `gates.first` would have named a gate the message does not describe.
    /// Omitting it falls back to `gates.first`, which is correct wherever a
    /// refusal site has one gate or reports them in vocabulary order.
    public static func refusal(
        verb: String,
        engine: String,
        code: String,
        gate: String? = nil,
        gates: [String] = [],
        reason: String,
        repairAction: String,
        state: SteerLabCLIState = .refused,
        observedAt: Date = Date(),
        workspace: String? = nil
    ) -> SteerLabCLIEnvelope {
        var envelope = SteerLabCLIEnvelope(
            verb: verb, engine: engine, state: state, message: reason,
            observedAt: observedAt, workspace: workspace)
        // The membership invariant is enforced here rather than trusted: a
        // named gate that is somehow absent from the list is added, so a
        // caller reading only `gates` can never miss the gate the message is
        // about.
        var allGates = gates
        if let gate, !allGates.contains(gate) { allGates.insert(gate, at: 0) }
        envelope.error = Failure(
            code: code, gate: gate ?? allGates.first,
            gates: allGates.isEmpty ? nil : allGates,
            reason: reason, repairAction: repairAction)
        return envelope
    }

    /// An operational failure — something broke, no gate declined anything.
    public static func failure(
        verb: String,
        engine: String,
        code: String,
        reason: String,
        repairAction: String,
        state: SteerLabCLIState = .failed,
        observedAt: Date = Date(),
        workspace: String? = nil
    ) -> SteerLabCLIEnvelope {
        var envelope = SteerLabCLIEnvelope(
            verb: verb, engine: engine, state: state, message: reason,
            observedAt: observedAt, workspace: workspace)
        envelope.error = Failure(
            code: code, reason: reason, repairAction: repairAction)
        return envelope
    }

    // MARK: Mutation

    /// Append advisories, keeping the "omitted when empty" wire rule. Does NOT
    /// touch `state`: promoting `ready` to `okWithAdvisories` is the caller's
    /// decision at the point it knows the verb succeeded.
    public mutating func add(_ newAdvisories: [Advisory]) {
        guard !newAdvisories.isEmpty else { return }
        advisories = (advisories ?? []) + newAdvisories
    }

    public mutating func add(advisory: Advisory) { add([advisory]) }

    // MARK: Derived

    /// The exit code a shell caller sees. Derived from `state`, so the document
    /// and the process agree by construction — there is no second table.
    public var exitCode: Int32 { state.exitCode }

    // MARK: Encoding

    /// The one JSON document, newline-terminated.
    ///
    /// Same mechanism as `ClusterCLIEnvelope.jsonText()`: sorted keys so a
    /// golden comparison is stable, ISO-8601 dates so a caller never parses a
    /// float, unescaped slashes so a path stays readable, and exactly one
    /// trailing newline so `read`-style callers terminate.
    /// `CLIEnvelopeTests.serializationMatchesTheClusterEnvelopeMechanism`
    /// pins the two together.
    public func jsonText() throws -> String {
        let data = try Self.makeEncoder().encode(self)
        return (String(data: data, encoding: .utf8) ?? "{}") + "\n"
    }

    /// Decode a document produced by either engine. Strict on purpose: an
    /// unknown `state` throws rather than being silently treated as success.
    public static func decode(fromJSON text: String) throws -> SteerLabCLIEnvelope {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(SteerLabCLIEnvelope.self, from: Data(text.utf8))
    }

    static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

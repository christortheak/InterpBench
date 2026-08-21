import Foundation
import Observation

// GPU-session client state (Wave 2 of docs/GPU-SESSION-PLAN.md).
//
// The controller server stays the single front door: the app keeps ONE URL
// and ONE token, and when a session is up the controller transparently
// proxies /api/load, generation, and probe routes to the GPU worker — so the
// existing remote paths just start working. Everything here is therefore
// *presentation and lifecycle* state: what the session record says, when to
// poll it, and how it reads in the Playground control and the connection
// dot. The one hard UX rule (plan §2.7, acceptance criterion 3): a session
// ending — for ANY reason — must read as "GPU session ended — controller
// still connected", never as a lost server connection.

// MARK: - Display state

/// The plan's §2.7 UX states, mapped from the wire record. `other` carries
/// an unrecognized non-terminal state verbatim (a newer controller must
/// display honestly, not crash or lie).
public enum GPUSessionDisplayState: Equatable, Sendable {
    case off
    case queued
    case starting
    case ready
    case busy
    case idle(remainingSeconds: Int?)
    case ending
    case ended
    case failed
    /// The scheduler conversation broke and the server refuses to guess
    /// whether the allocation is gone: NONTERMINAL, blocks new sessions,
    /// cleared only by a positive scheduler answer, worker recovery, or the
    /// operator's release-after-manual-verification.
    case unknownState
    case other(String)

    public var label: String {
        switch self {
        case .off: "Off"
        case .queued: "Queued"
        case .starting: "Starting"
        case .ready: "Ready"
        case .busy: "Busy"
        case .idle(let seconds): Self.idleLabel(remainingSeconds: seconds)
        case .ending: "Ending"
        case .ended: "Ended"
        case .failed: "Failed"
        case .unknownState: "Unknown — verify on cluster"
        case .other(let raw): raw
        }
    }

    /// "Idle 18m" from the worker's countdown; bare "Idle" when the
    /// controller reports no countdown. Rounds UP so the label never claims
    /// less time than actually remains, floor 1m while any time is left.
    public static func idleLabel(remainingSeconds: Int?) -> String {
        guard let remainingSeconds, remainingSeconds > 0 else { return "Idle" }
        let minutes = max(1, Int((Double(remainingSeconds) / 60).rounded(.up)))
        return "Idle \(minutes)m"
    }

    /// Terminal notice states (session over; a new one can start).
    public var isTerminal: Bool {
        switch self {
        case .ended, .failed: true
        default: false
        }
    }
}

// MARK: - Transport seam

/// The four controller calls the session controller makes, as injectable
/// closures so the lifecycle logic is testable without a URLSession. The
/// live wiring wraps `ClusterClient`.
public struct GPUSessionTransport: Sendable {
    public var status: @Sendable () async throws -> GPUSessionRecord?
    public var start: @Sendable (GPUSessionStartRequest) async throws -> GPUSessionRecord
    public var stop: @Sendable () async throws -> GPUSessionRecord?
    public var keepalive: @Sendable () async throws -> Void
    /// Operator clear for an `unknown` session (DELETE ?force=1): releases
    /// the slot after MANUAL verification that the allocation is gone. Nil
    /// on transports that predate the verb (falls back to plain stop).
    public var forceClear: (@Sendable () async throws -> GPUSessionRecord?)?

    public init(
        status: @escaping @Sendable () async throws -> GPUSessionRecord?,
        start: @escaping @Sendable (GPUSessionStartRequest) async throws -> GPUSessionRecord,
        stop: @escaping @Sendable () async throws -> GPUSessionRecord?,
        keepalive: @escaping @Sendable () async throws -> Void,
        forceClear: (@Sendable () async throws -> GPUSessionRecord?)? = nil
    ) {
        self.status = status
        self.start = start
        self.stop = stop
        self.keepalive = keepalive
        self.forceClear = forceClear
    }

    public init(client: ClusterClient) {
        self.init(
            status: { try await client.gpuSessionStatus() },
            start: { try await client.startGPUSession($0) },
            stop: { try await client.stopGPUSession() },
            keepalive: { try await client.gpuSessionKeepalive() },
            forceClear: { try await client.stopGPUSession(force: true) })
    }
}

// MARK: - Controller

/// Client-side GPU-session lifecycle: holds the latest record, polls
/// `GET /api/session` (~10 s) while a session exists or a start is in
/// flight, and maps the record to the plan's display states. Owned by
/// `ClusterConnectionStore`; the Playground control and the connection dot
/// render this and nothing else.
///
/// Two invariants live here, not in views:
/// - **Never poll without the capability.** Older servers 404 on the session
///   routes; the poll (like every affordance) is gated on
///   `chat.gpuSession`.
/// - **Session end is not a disconnection.** When the session ends for any
///   reason — record transitions to ended/failed, or the session goes null
///   after having existed — `sessionNotice` becomes the controller-still-
///   connected message and NOTHING here (or in the views reading it) touches
///   the server connection presentation. The app falls back to
///   controller-only behavior; the dot stays green.
@Observable @MainActor
public final class GPUSessionController {

    /// Live wiring, installed by `ClusterConnectionStore`: the transport for
    /// the active server (nil when the URL doesn't parse) and the capability
    /// verdict from the last-fetched capabilities.
    @ObservationIgnored public var transportProvider: (@MainActor () -> GPUSessionTransport?)?
    @ObservationIgnored public var capabilityProvider: (@MainActor () -> Bool)?

    /// Poll cadence while a session exists / a start is in flight.
    /// Injectable for tests.
    @ObservationIgnored public var pollInterval: Duration = .seconds(10)

    /// The latest session record from the controller. Kept through terminal
    /// states (ended/failed) so the notice can name what happened; cleared by
    /// a workspace/connection change or a fresh start.
    public private(set) var record: GPUSessionRecord?

    /// A `start(request:)` call is awaiting the controller's answer.
    public private(set) var isStartInFlight = false

    /// The terminal outcome once a session has ended (`.ended` / `.failed`),
    /// sticky until a new start or a connection change — a null status after
    /// a session existed must keep reading "ended", not snap back to "Off".
    public private(set) var terminalOutcome: GPUSessionDisplayState?

    /// The one plan-mandated message when a session ends for any reason.
    /// Shown next to the session control and NEVER via the connection
    /// status line — the controller connection is intact and must look it.
    public private(set) var sessionNotice: String?

    /// Last start/stop/keepalive problem, for the session control's caption.
    /// Deliberately never written to `ClusterConnectionStore.status`: the
    /// connection presentation keys off that line and a session hiccup must
    /// not paint the server red.
    public private(set) var lastActionError: String?

    /// A session record has been observed since the last reset — the
    /// null-after-existed half of ended-detection.
    private var hadSession = false

    /// Fires once each time the session BECOMES reachable (ready/busy/idle
    /// after none/queued/starting): from that moment the controller proxies
    /// `/api/state` to the worker, so anything fetched earlier (typically "no
    /// model loaded", which locks the chat composer) is stale and needs one
    /// refresh. Set by `ClusterConnectionStore`.
    @ObservationIgnored public var onBecameReady: (@MainActor () -> Void)?
    private var wasReachable = false

    private var pollTask: Task<Void, Never>?

    public init() {}

    // MARK: Derived state

    public var capabilityAvailable: Bool { capabilityProvider?() ?? false }

    /// A live (non-terminal) session record exists.
    public var isActive: Bool {
        guard let record else { return false }
        return !record.isTerminal && terminalOutcome == nil
    }

    /// The model-job warning predicate against the live controller state
    /// (cluster-testing item 2): a model-running submission NOW would find
    /// no GPU worker and none is on the way. Rule in
    /// `GPUSessionPreflight.shouldWarn` (pure, unit-tested).
    public var shouldWarnBeforeModelRunningJob: Bool {
        GPUSessionPreflight.shouldWarn(
            capabilityAvailable: capabilityAvailable,
            isSessionActive: isActive,
            isStartInFlight: isStartInFlight)
    }

    /// Poll only while there is something to watch — and never without the
    /// capability (older servers 404; a capability-less poll loop is exactly
    /// the ambient traffic the plan forbids).
    public var shouldPoll: Bool {
        capabilityAvailable && (isStartInFlight || isActive)
    }

    public var isPolling: Bool { pollTask != nil }

    /// The §2.7 state line: Off / Queued / Starting / Ready / Busy /
    /// "Idle 18m" / Ending — plus Ended/Failed as terminal notices.
    public var displayState: GPUSessionDisplayState {
        if let terminalOutcome { return terminalOutcome }
        guard let record else { return isStartInFlight ? .queued : .off }
        if record.busy == true { return .busy }
        switch record.state {
        case "queued": return .queued
        case "starting": return .starting
        case "ready": return .ready
        case "busy": return .busy
        case "idle": return .idle(remainingSeconds: record.idleRemainingSeconds)
        case "ending": return .ending
        case "ended": return .ended
        case "failed": return .failed
        case "unknown": return .unknownState
        default: return .other(record.state)
        }
    }

    /// "1h 42m walltime left" for the live session; nil otherwise.
    public var remainingWalltimeDescription: String? {
        guard isActive else { return nil }
        return record?.remainingWalltimeDescription()
    }

    /// The plan's exact controller-still-connected wording. One string for
    /// every cause (idle expiry, walltime, scancel, failure): the point is
    /// what the app is still connected to, not why the worker went away —
    /// `displayState` carries Ended vs Failed.
    public static let sessionEndedMessage =
        "GPU session ended — controller still connected"

    // MARK: Lifecycle actions

    /// Start a session. A 409 (one already live) is not a failure: the
    /// existing record rides in the refusal body and is adopted, so the UI
    /// lands on the running session either way (plan §2.5: start is
    /// idempotent-or-409, never two workers).
    public func start(request: GPUSessionStartRequest) async {
        guard capabilityAvailable, let transport = transportProvider?() else {
            lastActionError = "this server does not offer GPU sessions"
            return
        }
        isStartInFlight = true
        sessionNotice = nil
        terminalOutcome = nil
        lastActionError = nil
        updatePolling()
        defer {
            isStartInFlight = false
            updatePolling()
        }
        do {
            ingest(try await transport.start(request))
        } catch let error as ClusterClient.ClientError {
            if case .badResponse(409, let body) = error,
                let existing = GPUSessionRecord.record(fromResponseBody: body)
            {
                ingest(existing)
                lastActionError = "a GPU session is already running — showing it"
            } else {
                lastActionError = "could not start GPU session: "
                    + ClusterClient.unwrappingDetail(error).description
            }
        } catch {
            lastActionError = "could not start GPU session: \(error.localizedDescription)"
        }
    }

    /// End the session. The controller answers with the record as it last
    /// saw it (typically "ending"); polling continues until it lands
    /// terminal or null.
    public func stop() async {
        guard let transport = transportProvider?() else { return }
        lastActionError = nil
        do {
            ingest(try await transport.stop())
        } catch {
            lastActionError = "could not stop GPU session: \(error.localizedDescription)"
        }
    }

    /// Operator clear for a session stuck in `unknown` — the state where the
    /// scheduler conversation broke and the server refuses to guess whether
    /// the allocation is gone. Call ONLY after checking by hand (the
    /// stateDetail names the `sacct -j` command); it releases the slot.
    public func releaseVerifiedGone() async {
        guard let transport = transportProvider?() else { return }
        lastActionError = nil
        do {
            let clear = transport.forceClear ?? transport.stop
            ingest(try await clear())
        } catch {
            lastActionError = "could not release GPU session: \(error.localizedDescription)"
        }
    }

    /// Explicit "Keep session" ping — the one deliberate idle-timer reset
    /// (plan §2.3). Refreshes status afterwards so the countdown snaps back.
    public func keepSession() async {
        guard let transport = transportProvider?() else { return }
        lastActionError = nil
        do {
            try await transport.keepalive()
            await syncOnce()
        } catch {
            lastActionError = "keepalive failed: \(error.localizedDescription)"
        }
    }

    /// One status fetch. A transport failure is a transient poll problem —
    /// it must NOT read as "session ended" (only a successful answer with no
    /// session counts); the last record stays.
    public func syncOnce() async {
        guard capabilityAvailable, let transport = transportProvider?() else { return }
        do {
            ingest(try await transport.status())
        } catch {
            // keep the last record; the next poll (or reconnect) decides
        }
    }

    /// Called after a successful connect: when the capability is present,
    /// look for a session already running (a controller restart is a blip —
    /// the record survives it) and begin polling if one exists. No-op —
    /// and no request at all — without the capability.
    public func refreshAfterConnect() {
        guard capabilityAvailable else { return }
        Task { await syncOnce() }
    }

    /// Workspace/connection change: this state belonged to the previous
    /// server. Deliberately NOT ended-detection — no notice fires.
    public func resetForConnectionChange() {
        pollTask?.cancel()
        pollTask = nil
        record = nil
        hadSession = false
        wasReachable = false
        isStartInFlight = false
        terminalOutcome = nil
        sessionNotice = nil
        lastActionError = nil
    }

    /// Dismiss the terminal notice (the user acknowledged it); the control
    /// returns to Off and a fresh start is offered.
    public func clearSessionNotice() {
        sessionNotice = nil
        terminalOutcome = nil
        if record?.isTerminal == true {
            record = nil
            hadSession = false
        }
    }

    // MARK: Record ingestion (the ended-detection seam — internal for tests)

    /// Adopt a status answer. Ended-detection lives here: a record in a
    /// terminal state, or a null session after one existed, produces the
    /// controller-still-connected notice — and nothing else changes about
    /// the connection presentation.
    func ingest(_ new: GPUSessionRecord?) {
        if let new {
            record = new
            hadSession = true
            if new.isTerminal {
                terminalOutcome = new.state == "failed" ? .failed : .ended
                sessionNotice = Self.sessionEndedMessage
            } else {
                terminalOutcome = nil
            }
        } else {
            if hadSession, terminalOutcome == nil {
                terminalOutcome = .ended
                sessionNotice = Self.sessionEndedMessage
            }
            record = nil
        }
        let reachable =
            record?.isTerminal == false
            && ["ready", "busy", "idle"].contains(record?.state ?? "")
        if reachable, !wasReachable {
            onBecameReady?()
        }
        wasReachable = reachable
        updatePolling()
    }

    // MARK: Polling

    private func updatePolling() {
        if shouldPoll {
            guard pollTask == nil else { return }
            pollTask = Task { [weak self] in
                while !Task.isCancelled {
                    guard let self else { return }
                    try? await Task.sleep(for: self.pollInterval)
                    if Task.isCancelled { return }
                    await self.syncOnce()
                    if !self.shouldPoll {
                        self.pollTask = nil
                        return
                    }
                }
            }
        } else {
            pollTask?.cancel()
            pollTask = nil
        }
    }
}

// MARK: - Model-first sizing (plan §2.7)

/// Pure suggestion of `{gres, walltime, idleMinutes}` from a model id and
/// the site profile's GPU inventory (`gpuTypes`/`gpuVRAMGB`).
///
/// This is deliberately a coarser instrument than the server's WS4
/// `memoryFit` preflight (which reads the cached weights' actual bytes and
/// the config's KV geometry server-side — `submissions.py`): at session-start
/// time the Mac has neither, so the estimate comes from the parameter count
/// parsed out of the model id — bf16 (2 bytes/param) plus 30 % KV/activation
/// headroom. The selection rule matches the preflight's fix-suggestion:
/// smallest declared GPU type that fits. An unparseable id defaults to the
/// largest declared GPU — the safe over-ask for an interactive session the
/// user is watching.
public enum GPUSessionSizing {

    public struct Suggestion: Equatable, Sendable {
        /// Chosen GPU type name (nil when the site declares none with VRAM).
        public var gpuType: String?
        /// Concrete `--gres` string ("gpu:A100:1"); nil when no type chosen.
        public var gres: String?
        public var walltime: String
        public var idleMinutes: Int
        /// Estimated need in GB; nil when the id yielded no parameter count.
        public var estimatedVRAMGB: Double?
        /// One human-readable line saying how the pick was made.
        public var rationale: String
    }

    public static let defaultWalltime = "02:00:00"
    public static let defaultIdleMinutes = 30
    /// bf16 weights.
    static let bytesPerParameter = 2.0
    /// KV cache + activations headroom over bare weights.
    static let overheadFactor = 1.3

    /// Parameter count in billions parsed from a model id: the LARGEST
    /// `<number>b`/`<number>B` token at a word boundary ("gemma-3-27b" → 27,
    /// "Qwen3-14B-MLX-8bit" → 14 — "8bit"/"4bit" never match because a word
    /// character follows the `b`; the largest match wins so family version
    /// digits like the "3" in "gemma-3" lose to the real size). Nil when no
    /// token matches.
    public static func parameterBillions(fromModelID modelID: String) -> Double? {
        let regex = /(\d+(?:\.\d+)?)[bB]\b/
        let values = modelID.matches(of: regex).compactMap { Double($0.output.1) }
        return values.max()
    }

    /// Estimated VRAM need in GB for a parameter count (bf16 + headroom).
    public static func estimatedVRAMGB(parameterBillions: Double) -> Double {
        parameterBillions * bytesPerParameter * overheadFactor
    }

    /// Suggest a session shape for `modelID` on a site. `slurm` may be nil
    /// (a direct workstation server advertising the capability): the
    /// defaults still apply, with no gres suggestion.
    public static func suggest(
        modelID: String, slurm: ClusterSiteProfile.SlurmSiteData?
    ) -> Suggestion {
        let need = parameterBillions(fromModelID: modelID).map(estimatedVRAMGB(parameterBillions:))
        // Declared inventory: the site's gpuTypes with a VRAM entry, in the
        // declared order; a site that only filled the VRAM table still works.
        var inventory: [(type: String, vramGB: Int)] = []
        if let slurm {
            let declared = slurm.gpuTypes.isEmpty
                ? slurm.gpuVRAMGB.keys.sorted() : slurm.gpuTypes
            inventory = declared.compactMap { type in
                slurm.gpuVRAMGB[type].map { (type, $0) }
            }
        }
        guard !inventory.isEmpty else {
            return Suggestion(
                gpuType: nil, gres: slurm?.defaultGres, walltime: defaultWalltime,
                idleMinutes: defaultIdleMinutes, estimatedVRAMGB: need,
                rationale: "site declares no GPU VRAM data — using the site default")
        }
        // Deterministic ordering: by VRAM, ties alphabetical.
        let sorted = inventory.sorted {
            ($0.vramGB, $0.type) < ($1.vramGB, $1.type)
        }
        let chosen: (type: String, vramGB: Int)
        let rationale: String
        if let need {
            if let fitting = sorted.first(where: { Double($0.vramGB) >= need }) {
                chosen = fitting
                rationale = String(
                    format: "≈%.0f GB needed (bf16 + 30%% headroom) — smallest fitting GPU",
                    need.rounded(.up))
            } else {
                // Nothing declared fits: over-ask the largest and say so
                // rather than silently suggesting an OOM.
                chosen = sorted[sorted.count - 1]
                rationale = String(
                    format: "≈%.0f GB needed exceeds every declared GPU — "
                        + "largest available (may not fit)",
                    need.rounded(.up))
            }
        } else {
            chosen = sorted[sorted.count - 1]
            rationale = "could not parse a parameter count from the model id — "
                + "defaulting to the largest declared GPU"
        }
        return Suggestion(
            gpuType: chosen.type, gres: "gpu:\(chosen.type):1",
            walltime: defaultWalltime, idleMinutes: defaultIdleMinutes,
            estimatedVRAMGB: need, rationale: rationale)
    }
}

// MARK: - Model-job preflight (warn before GPU-less submissions)

/// Cluster-testing item 2 (2026-07-21): the researcher submitted a training
/// job with no GPU session and it sat waiting with no explanation. Before a
/// MODEL-RUNNING job goes to a cluster whose profile offers GPU sessions,
/// the app warns — with an override, never a refusal (honest-researcher
/// rule: letting Slurm queue the job for later is a legitimate choice).
/// The rule and the dialog copy live here so every submission surface and
/// the tests share one predicate.
public enum GPUSessionPreflight {

    /// True exactly when a model-running job submitted NOW would find no
    /// GPU worker: the site offers GPU sessions, none is live, and none is
    /// already starting. Local workspaces and capability-less servers never
    /// warn (there is nothing to start).
    public static func shouldWarn(
        capabilityAvailable: Bool,
        isSessionActive: Bool,
        isStartInFlight: Bool
    ) -> Bool {
        capabilityAvailable && !isSessionActive && !isStartInFlight
    }

    /// Whether an experiment verb executes the pinned model (and therefore
    /// needs a GPU node). `analyze` is statistics over an already-completed
    /// run — never warned. Unknown verbs warn: the safe default for any
    /// future model verb is a dismissible dialog, not silence.
    public static func isModelRunningVerb(_ verb: String) -> Bool {
        verb.trimmingCharacters(in: .whitespaces).lowercased() != "analyze"
    }

    /// The dialog body, in plain language. The actions are the caller's
    /// (Start GPU session then submit / Submit anyway / Cancel).
    public static func warningMessage(substrate: String) -> String {
        "No GPU session is running on \(substrate). This job needs a GPU "
            + "node — it will wait or fail until one is available. Start a "
            + "GPU session first, or submit anyway?"
    }
}

// MARK: - Submission-resource preflight (2026-07-21 incident, part 1)

/// The OTHER half of the model-job warning, born from a real cluster
/// failure (2026-07-21): a study bundle submitted with executor "local" on
/// a Slurm site executed INSIDE the controller job's allocation (batch
/// partition, 1 CPU, 16 GB, no GPU) and died mid-load — the GPU-SESSION
/// warning said nothing, because a session is not what that submission
/// needed. What it needed was its OWN GPU allocation: executor "slurm" and
/// a non-empty gres.
///
/// One dialog, the most relevant message: a missing allocation is MORE
/// specific than a missing session, so it wins; when both apply the one
/// dialog says both. Pure predicate + copy here (unit-tested); the app's
/// `ModelJobGPUGate` is presentation glue. Warning with an override, never
/// a refusal — CPU smoke tests with tiny models are legitimate.
public enum ModelJobSubmissionPreflight {

    /// The Remote-options half of a BUNDLE submission (the paths that carry
    /// executor + resources). Direct server verbs (validate/extract/LoRA)
    /// pass nil and keep the session-only warning.
    public struct BundleOptions: Sendable, Equatable {
        public var executor: String
        public var gres: String
        public var verb: String
        public var dryRun: Bool

        public init(executor: String, gres: String, verb: String, dryRun: Bool) {
            self.executor = executor
            self.gres = gres
            self.verb = verb
            self.dryRun = dryRun
        }
    }

    public enum Concern: Sendable, Equatable {
        /// The submission itself requests no GPU (non-slurm executor and/or
        /// empty gres) on a Slurm site — the 2026-07-21 failure mode.
        case noGPUAllocation(alsoNoSession: Bool)
        /// Only the interactive GPU session is missing (the historical
        /// warning). Fires for non-bundle submissions (`options == nil`)
        /// and for bundles on non-Slurm sites; a correctly-allocated Slurm
        /// bundle (executor slurm, non-empty gres) never warns — it owns
        /// its GPU allocation and does not need a session.
        case noGPUSession
    }

    /// True when the submission's own options would leave it GPU-less on a
    /// Slurm site: anything but the slurm executor runs inside the
    /// controller's allocation, and a slurm job with empty gres gets no GPU.
    public static func missingGPUAllocation(executor: String, gres: String) -> Bool {
        executor.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != "slurm"
            || gres.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// The one rule behind the one dialog. Nil = submit silently.
    ///
    /// - `siteUsesSlurmScheduler`: the active site profile declares a Slurm
    ///   scheduler — on anything else (workstation, local dev server) the
    ///   allocation half never fires; executor "local" is simply correct
    ///   there.
    /// - `options`: nil for non-bundle submissions (session-only rule).
    /// - `sessionWarning`: the existing `GPUSessionPreflight.shouldWarn`
    ///   verdict against live controller state. On a Slurm site it only
    ///   matters when the ALLOCATION is missing too: a batch bundle with
    ///   executor slurm + non-empty gres owns its GPU allocation, so the
    ///   interactive session's absence is irrelevant to it (external review
    ///   2026-07-22 — a correct submission must not warn).
    public static func concern(
        siteUsesSlurmScheduler: Bool,
        options: BundleOptions?,
        sessionWarning: Bool
    ) -> Concern? {
        if let options {
            // Dry runs prepare without executing, and non-model verbs
            // (analyze) never load the model — neither warns.
            guard !options.dryRun,
                GPUSessionPreflight.isModelRunningVerb(options.verb)
            else { return nil }
            if siteUsesSlurmScheduler {
                if missingGPUAllocation(
                    executor: options.executor, gres: options.gres)
                {
                    return .noGPUAllocation(alsoNoSession: sessionWarning)
                }
                // Executor slurm + non-empty gres on a Slurm site: the job
                // gets its OWN GPU allocation from the scheduler — submit
                // silently, session or no session.
                return nil
            }
        }
        return sessionWarning ? .noGPUSession : nil
    }

    /// Dialog title — names the actual problem.
    public static func title(for concern: Concern, jobLabel: String) -> String {
        switch concern {
        case .noGPUAllocation:
            return "No GPU allocation — submit \(jobLabel)?"
        case .noGPUSession:
            return "No GPU session — submit \(jobLabel)?"
        }
    }

    /// Dialog body in plain language: what would happen, why, and the ways
    /// forward. For `.noGPUSession` the historical
    /// `GPUSessionPreflight.warningMessage` copy is reused unchanged.
    public static func message(
        for concern: Concern, substrate: String, options: BundleOptions?
    ) -> String {
        switch concern {
        case .noGPUSession:
            return GPUSessionPreflight.warningMessage(substrate: substrate)
        case .noGPUAllocation(let alsoNoSession):
            let executor = options?.executor
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let gres = options?.gres
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            var reasons: [String] = []
            if executor.lowercased() != "slurm" {
                reasons.append(
                    "the executor is '\(executor.isEmpty ? "unset" : executor)', "
                        + "so the job would execute inside the controller's own "
                        + "small CPU allocation instead of its own Slurm job")
            }
            if gres.isEmpty {
                reasons.append(
                    "the GPU gres field is empty, so Slurm would grant no GPU")
            }
            var text = "This job would run without a GPU allocation on "
                + "\(substrate): "
                + reasons.joined(separator: ", and ")
                + ". On the controller's small CPU allocation, a "
                + "multi-billion-parameter model will fail. Fix the options to "
                + "request a GPU, or submit anyway (CPU smoke tests with a "
                + "small model are legitimate)."
            if alsoNoSession {
                text += " There is also no GPU session running — that affects "
                    + "interactive chat, not this batch submission's resources."
            }
            return text
        }
    }

    /// The "Fix options" action's pure half: what the Remote options fields
    /// become. Executor snaps to "slurm"; an empty gres prefills with the
    /// site's first GPU vocabulary entry (a non-empty gres is the
    /// researcher's choice and stays).
    public static func fixedOptions(
        executor: String, gres: String, siteGPUTypes: [String]
    ) -> (executor: String, gres: String) {
        let fixedExecutor =
            executor.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                == "slurm" ? executor : "slurm"
        let fixedGres =
            gres.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? (siteGPUTypes.first ?? gres) : gres
        return (fixedExecutor, fixedGres)
    }

    /// The dialog's fix-affordance label, shared with tests.
    public static let fixOptionsButtonLabel = "Fix options (request a GPU)"
}

// MARK: - Stop-with-jobs-running check (toolbar Stop)

/// Cluster-testing item 1's stop half: ending the session out from under
/// unfinished server jobs deserves one honest confirmation. Pure rules so
/// the toolbar control stays glue.
public enum GPUSessionStopCheck {

    /// Jobs the server has not marked finished (any non-terminal status).
    public static func unfinishedJobCount(_ jobs: [RemoteJobRecord]) -> Int {
        jobs.count { $0.finishedAt == nil }
    }

    /// Nil → stop without asking; non-nil → show this confirmation first.
    /// A nil count means the job list could not be fetched — the honest
    /// default is to confirm, saying we could not check, never to pretend
    /// we did.
    public static func confirmationMessage(unfinishedJobCount: Int?) -> String? {
        guard let count = unfinishedJobCount else {
            return "Could not check the server's job list — jobs may still "
                + "be running. Ending the GPU session leaves any unfinished "
                + "job waiting or failing until a new session starts."
        }
        guard count > 0 else { return nil }
        let jobs = count == 1 ? "1 server job has" : "\(count) server jobs have"
        return "\(jobs) not finished. Ending the GPU session leaves "
            + (count == 1 ? "it" : "them")
            + " waiting or failing until a new session starts."
    }
}

// MARK: - "No GPU session" refusal hint

/// Recognizes the controller's 409 "no GPU session — start one" refusal on
/// proxied interactive routes and appends the actionable pointer at the
/// app's own control. Tolerant string match on "GPU session" in the 409
/// detail (the wire contract fixes the status code, not the exact prose).
public enum GPUSessionRefusal {

    /// Non-nil exactly when this is the no-session 409: the server's own
    /// detail plus the pointer at the Playground control. Nil for every
    /// other error — those must surface as what they are.
    public static func hint(code: Int, body: String) -> String? {
        guard code == 409 else { return nil }
        let detail = Self.detail(fromErrorBody: body) ?? body
        guard detail.lowercased().contains("gpu session") else { return nil }
        return detail + " — start one with the GPU Session control in the Playground"
    }

    /// FastAPI `{"detail": "…"}` unwrap (string details only), nil otherwise.
    static func detail(fromErrorBody body: String) -> String? {
        struct Detail: Decodable { var detail: String }
        guard let data = body.data(using: .utf8) else { return nil }
        return (try? JSONDecoder().decode(Detail.self, from: data))?.detail
    }
}
